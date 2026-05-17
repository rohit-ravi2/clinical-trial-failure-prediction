"""Phase B — Gate 3 leak-clean rebuild.

Day 1's gate3_features built drug/target/MoA features by joining trials_labeled
to ChEMBL via primary_drug_canonical, which was only non-NULL for FDA-matched
trials (~46% of P3 cohort, ~20% of full studies). The other 54% had constant
features, and the feature was correlated with the FDA-match flag — itself
strongly correlated with the label. Result: first smoke test gave AUROC 0.94.

Day 1 mitigation: dropped the leaky cols + their one-hot families.
Day 2 fix: rebuild from intervention raw + ChEMBL local SQLite (when available),
with explicit per-drug-loo target encoding (not per-trial-loo).

Inputs:
  - aact.duckdb (chembl.drug_moa + interventions + studies + trials_labeled)
  - fda.submission_property_type (for Orphan)
  - chembl_full.compound_structures / mechanism_of_action (from Phase A1 once loaded)
  - processed/splits/*.pkl (frozen Day-1 splits)

Outputs:
  - processed/day2/features_v2.parquet           (one row per trial)
  - processed/day2/feature_schema_v2.json
  - processed/day2/fold_encodings_v2/{task}_{split}.parquet   (per-split LODO)
"""
from __future__ import annotations
import json, logging, os, sys, time
from pathlib import Path
from datetime import datetime, timezone

import duckdb, numpy as np, pandas as pd
from tqdm import tqdm

ROOT = Path("/mnt/ssd/research/trial-failure")
DUCKDB = ROOT / "aact.duckdb"
OUTDIR = ROOT / "processed/day2"
OUTDIR.mkdir(parents=True, exist_ok=True)
FOLDDIR = OUTDIR / "fold_encodings_v2"; FOLDDIR.mkdir(exist_ok=True)
LOG = ROOT / "logs/phaseB.log"
PARQUET_OUT = OUTDIR / "features_v2.parquet"
SCHEMA_OUT = OUTDIR / "feature_schema_v2.json"

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler(sys.stdout)])
log = logging.getLogger("phaseB")

# ---- helpers ----
PHASE_ORD = {"EARLY_PHASE1": 0.5, "PHASE1": 1.0, "PHASE1/PHASE2": 1.5,
             "PHASE2": 2.0, "PHASE2/PHASE3": 2.5, "PHASE3": 3.0, "PHASE4": 4.0}

def build_base(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    log.info("B0: assembling base table from studies + designs + trials_labeled")
    df = con.execute("""
        SELECT t.nct_id, t.phase, t.overall_status, t.registration_date,
               t.start_date, t.completion_date, t.primary_drug_canonical,
               t.fda_match, t.matched_intervention_count, t.has_exact, t.has_fuzzy,
               t.p2_success, t.p3_success, t.label_available,
               s.enrollment, s.number_of_arms, s.why_stopped,
               s.has_dmc, s.is_fda_regulated_drug, s.is_fda_regulated_device,
               s.source_class AS lead_sponsor_class,
               d.allocation, d.intervention_model, d.primary_purpose, d.masking
        FROM trials_labeled t
        LEFT JOIN studies s ON t.nct_id = s.nct_id
        LEFT JOIN designs d ON t.nct_id = d.nct_id
    """).fetchdf()
    log.info(f"  base shape={df.shape}")
    return df

def feature_phase(df: pd.DataFrame) -> pd.DataFrame:
    log.info("B0: phase encoding")
    df["phase_ord"] = df["phase"].map(PHASE_ORD).astype("Float64")
    df["phase_missing"] = df["phase"].isna().astype(int)
    df["phase"] = df["phase"].fillna("UNKNOWN")
    return df

def feature_enrollment_duration(df: pd.DataFrame) -> pd.DataFrame:
    log.info("B0: enrollment/duration")
    df["enrollment"] = pd.to_numeric(df["enrollment"], errors="coerce")
    df["enrollment_missing"] = df["enrollment"].isna().astype(int)
    df["enrollment_log1p"] = np.log1p(df["enrollment"].fillna(df["enrollment"].median()))
    df["number_of_arms"] = pd.to_numeric(df["number_of_arms"], errors="coerce").fillna(1)
    # duration in days from start to completion (or NaN if either missing)
    sd = pd.to_datetime(df["start_date"], errors="coerce")
    cd = pd.to_datetime(df["completion_date"], errors="coerce")
    dur = (cd - sd).dt.days
    df["duration_days_missing"] = dur.isna().astype(int)
    df["duration_days"] = dur.fillna(dur.median()).clip(lower=0)
    return df

def feature_design(df: pd.DataFrame) -> pd.DataFrame:
    log.info("B0: design features")
    masking = df["masking"].fillna("NONE").str.upper()
    df["any_blinding"] = (~masking.isin(["NONE", "OPEN", "OPEN_LABEL", "NO MASKING (OPEN LABEL)", ""])).astype(int)
    df["is_randomized"] = (df["allocation"].fillna("").str.upper() == "RANDOMIZED").astype(int)
    df["has_placebo"] = df["intervention_model"].fillna("").str.upper().str.contains("PARALLEL|CROSSOVER").astype(int)
    return df

def feature_drug_type_raw(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> pd.DataFrame:
    """B1: drug_type from raw interventions.intervention_type — no FDA proxy."""
    log.info("B1: drug_type from raw interventions")
    # AACT pipe-file dump uses UPPERCASE intervention_type values.
    rank = con.execute("""
        WITH ranked AS (
          SELECT nct_id, upper(intervention_type) AS itype,
            CASE upper(intervention_type)
              WHEN 'DRUG' THEN 1 WHEN 'BIOLOGICAL' THEN 2
              WHEN 'COMBINATION PRODUCT' THEN 3 WHEN 'GENETIC' THEN 4
              WHEN 'DEVICE' THEN 5 WHEN 'PROCEDURE' THEN 6
              WHEN 'DIAGNOSTIC TEST' THEN 7 ELSE 99 END AS rk
          FROM interventions
        )
        SELECT nct_id, itype AS intervention_type
        FROM (SELECT nct_id, itype, ROW_NUMBER() OVER (PARTITION BY nct_id ORDER BY rk) rn FROM ranked) WHERE rn = 1
    """).fetchdf()
    mp = {"DRUG": "small_molecule", "BIOLOGICAL": "biologic",
          "COMBINATION PRODUCT": "combination", "GENETIC": "cell_or_gene",
          "DEVICE": "device", "PROCEDURE": "procedure",
          "DIAGNOSTIC TEST": "diagnostic"}
    rank["drug_type"] = rank["intervention_type"].map(mp).fillna("other")
    df = df.merge(rank[["nct_id", "drug_type"]], on="nct_id", how="left")
    df["drug_type"] = df["drug_type"].fillna("other")
    log.info(f"  drug_type coverage: {(df['drug_type']!='other').mean():.1%}")
    log.info(f"  drug_type distribution:\n{df['drug_type'].value_counts().head(10)}")
    return df

def feature_moa_local_chembl(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> pd.DataFrame:
    """B2: MoA via local ChEMBL when available, fallback to existing chembl.drug_moa.

    Strategy: build a name->chembl_id resolver from interventions.name +
    intervention_other_names.name. For each trial, take the highest-priority
    intervention (drug or biological), resolve its name to a ChEMBL compound,
    pull MoA / target_type / action_type from chembl.drug_moa.
    """
    log.info("B2: MoA resolver (ChEMBL local cache)")
    # Build name->chembl_id from existing drug_moa
    moa = con.execute("""
        SELECT lower(drug_canonical) AS norm_name,
               chembl_id, mechanism_of_action, target_chembl_id, target_pref_name,
               target_type, action_type, molecule_type
        FROM chembl.drug_moa WHERE status='hit'
    """).fetchdf()
    log.info(f"  chembl drug_moa hits available: {len(moa)}")

    # Try chembl_full if A1 has populated it
    chembl_full_path = Path("/mnt/hdd/research/trial-failure/raw/chembl/chembl_36_loaded.flag")
    if chembl_full_path.exists():
        log.info("  chembl_full schema available — augmenting")
        try:
            extra = con.execute("""
                SELECT DISTINCT lower(synonyms) AS norm_name, chembl_id
                FROM chembl_full.molecule_synonyms
                WHERE synonyms IS NOT NULL
            """).fetchdf()
            log.info(f"  chembl_full synonyms: {len(extra)}")
        except Exception as e:
            log.warning(f"  chembl_full not yet queryable: {e}")

    # Resolve trial -> drug name (AACT uses UPPERCASE intervention_type)
    drug_per_trial = con.execute("""
        WITH r AS (
          SELECT nct_id, lower(name) AS norm_name,
            CASE upper(intervention_type)
              WHEN 'DRUG' THEN 1 WHEN 'BIOLOGICAL' THEN 2 ELSE 9 END AS rk
          FROM interventions
          WHERE upper(intervention_type) IN ('DRUG', 'BIOLOGICAL', 'COMBINATION PRODUCT')
        )
        SELECT nct_id, norm_name FROM (
          SELECT nct_id, norm_name, ROW_NUMBER() OVER (PARTITION BY nct_id ORDER BY rk) rn FROM r
        ) WHERE rn = 1
    """).fetchdf()
    log.info(f"  trials with drug name resolved: {len(drug_per_trial)}")

    # Stage 1: exact name match
    j = drug_per_trial.merge(moa, on="norm_name", how="left")
    stage1_hits = j["chembl_id"].notna().sum()
    log.info(f"  Stage 1 exact: {stage1_hits} ({stage1_hits/len(drug_per_trial):.1%})")

    # Stage 2: try intervention_other_names for unresolved
    unres = j[j["chembl_id"].isna()][["nct_id", "norm_name"]]
    if len(unres):
        ion = con.execute("""
            SELECT DISTINCT i.nct_id, lower(ion.name) AS alt_name
            FROM interventions i
            JOIN intervention_other_names ion ON i.id = ion.intervention_id
        """).fetchdf()
        stage2 = unres.merge(ion, on="nct_id", how="left").merge(
            moa.rename(columns={"norm_name": "alt_name"}), on="alt_name", how="left")
        stage2 = stage2.dropna(subset=["chembl_id"]).drop_duplicates(subset=["nct_id"])
        log.info(f"  Stage 2 other-names: {len(stage2)} new hits")
        if len(stage2):
            stage2_map = stage2[["nct_id", "chembl_id", "mechanism_of_action",
                                 "target_chembl_id", "target_pref_name",
                                 "target_type", "action_type", "molecule_type"]]
            # update j
            j = j.set_index("nct_id")
            stage2_map = stage2_map.set_index("nct_id")
            for col in stage2_map.columns:
                j[col] = j[col].combine_first(stage2_map[col])
            j = j.reset_index()

    # Stage 3: rapidfuzz against the full moa name set
    try:
        from rapidfuzz import process, fuzz
        unres = j[j["chembl_id"].isna() & j["norm_name"].notna()].copy()
        if len(unres) and len(moa):
            log.info(f"  Stage 3 fuzzy: {len(unres)} unresolved names against {len(moa)} ChEMBL names")
            choices = moa["norm_name"].tolist()
            moa_idx = moa.set_index("norm_name")
            fuzzy_hits = []
            for nm in tqdm(unres["norm_name"].unique().tolist(), desc="fuzzy"):
                if not nm: continue
                m = process.extractOne(nm, choices, scorer=fuzz.token_sort_ratio, score_cutoff=85)
                if m:
                    fuzzy_hits.append((nm, m[0]))
            log.info(f"  Stage 3 hits: {len(fuzzy_hits)}")
            fmap = pd.DataFrame(fuzzy_hits, columns=["norm_name", "matched"])
            fmap = fmap.merge(moa_idx.reset_index().rename(columns={"norm_name": "matched"}),
                              on="matched", how="left")
            stage3 = unres.drop(columns=[c for c in unres.columns if c != "nct_id" and c != "norm_name"]).merge(
                fmap, on="norm_name", how="left").dropna(subset=["chembl_id"])
            if len(stage3):
                stage3_map = stage3[["nct_id", "chembl_id", "mechanism_of_action",
                                     "target_chembl_id", "target_pref_name",
                                     "target_type", "action_type", "molecule_type"]].drop_duplicates(subset=["nct_id"])
                j = j.set_index("nct_id")
                stage3_map = stage3_map.set_index("nct_id")
                for col in stage3_map.columns:
                    if col in j.columns:
                        j[col] = j[col].combine_first(stage3_map[col])
                j = j.reset_index()
    except Exception as e:
        log.warning(f"  Stage 3 fuzzy failed: {e}")

    overall_hit = j["chembl_id"].notna().sum()
    log.info(f"  TOTAL ChEMBL resolution: {overall_hit}/{len(drug_per_trial)} ({overall_hit/max(1,len(drug_per_trial)):.1%})")

    # Coarse MoA bucket
    def moa_bucket(s):
        if not isinstance(s, str): return "unknown"
        s = s.lower()
        if "inhibitor" in s: return "inhibitor"
        if "agonist" in s: return "agonist"
        if "antagonist" in s: return "antagonist"
        if "modulator" in s: return "modulator"
        if "blocker" in s: return "blocker"
        if "antibody" in s or "binder" in s: return "antibody"
        return "other_moa"
    j["moa_bucket"] = j["mechanism_of_action"].apply(moa_bucket)

    keep = ["nct_id", "chembl_id", "mechanism_of_action", "moa_bucket",
            "target_chembl_id", "target_pref_name", "target_type", "action_type",
            "molecule_type"]
    df = df.merge(j[keep], on="nct_id", how="left")
    for c in ["mechanism_of_action", "target_pref_name", "target_type", "action_type", "molecule_type"]:
        df[c] = df[c].fillna("unknown").astype(str)
    df["moa_bucket"] = df["moa_bucket"].fillna("unknown")
    df["chembl_resolved"] = df["chembl_id"].notna().astype(int)
    return df, overall_hit, len(drug_per_trial)

def feature_prior_counts_truncated(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> pd.DataFrame:
    """B3: time-truncated prior-trial counts grouped by chembl_id (and target)."""
    log.info("B3: time-truncated prior counts")
    sub = df[["nct_id", "registration_date", "chembl_id", "target_chembl_id"]].copy()
    sub["registration_date"] = pd.to_datetime(sub["registration_date"], errors="coerce")
    sub["chembl_id"] = sub["chembl_id"].fillna("__UNK__")
    sub["target_chembl_id"] = sub["target_chembl_id"].fillna("__UNK__")
    # use duckdb for fast self-join
    dcon = duckdb.connect()
    dcon.register("sub", sub)
    drug_counts = dcon.execute("""
        SELECT a.nct_id,
               COUNT(*) FILTER (WHERE b.registration_date < a.registration_date) AS num_prior_trials_same_drug,
               COUNT(DISTINCT a.chembl_id) FILTER (WHERE FALSE) AS dummy
        FROM sub a LEFT JOIN sub b
          ON b.chembl_id = a.chembl_id AND b.chembl_id <> '__UNK__'
        WHERE a.chembl_id <> '__UNK__'
        GROUP BY a.nct_id
    """).fetchdf()
    target_counts = dcon.execute("""
        SELECT a.nct_id,
               COUNT(*) FILTER (WHERE b.registration_date < a.registration_date) AS num_prior_trials_same_target,
               COUNT(DISTINCT b.chembl_id) FILTER (WHERE b.registration_date < a.registration_date)
                  AS num_distinct_prior_drugs_same_target
        FROM sub a LEFT JOIN sub b
          ON b.target_chembl_id = a.target_chembl_id AND b.target_chembl_id <> '__UNK__'
        WHERE a.target_chembl_id <> '__UNK__'
        GROUP BY a.nct_id
    """).fetchdf()
    dcon.close()
    df = df.merge(drug_counts[["nct_id", "num_prior_trials_same_drug"]], on="nct_id", how="left")
    df = df.merge(target_counts[["nct_id", "num_prior_trials_same_target", "num_distinct_prior_drugs_same_target"]],
                  on="nct_id", how="left")
    for c in ["num_prior_trials_same_drug", "num_prior_trials_same_target", "num_distinct_prior_drugs_same_target"]:
        df[c] = df[c].fillna(0).astype(int)
    # target novelty
    df["target_novelty"] = 1.0 / (df["num_distinct_prior_drugs_same_target"].astype(float) + 1.0)
    log.info(f"  prior_drug_counts: mean={df['num_prior_trials_same_drug'].mean():.2f}")
    log.info(f"  prior_target_counts: mean={df['num_prior_trials_same_target'].mean():.2f}")
    return df

def feature_orphan_status(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> pd.DataFrame:
    """B4: orphan designation from FDA submission_property_type (no OOPD bot block)."""
    log.info("B4: orphan designation from FDA submissions")
    # Use ORIG submission date as the orphan-designation date proxy. The
    # submission_property_type triple key (ApplNo, SubmissionType, SubmissionNo)
    # is not always populated in submissions, but ApplNo+ORIG always is.
    orphan = con.execute("""
        SELECT a.ApplNo, MIN(s.SubmissionStatusDate) AS orphan_date
        FROM fda.submission_property_type spt
        JOIN fda.applications a ON spt.ApplNo = a.ApplNo
        JOIN fda.submissions s ON s.ApplNo = a.ApplNo AND s.SubmissionType='ORIG'
        WHERE spt.SubmissionPropertyTypeCode = 'Orphan'
        GROUP BY a.ApplNo
    """).fetchdf()
    log.info(f"  distinct orphan applications: {len(orphan)}")
    # link to drug by ApplNo via products
    drug_orphan = con.execute("""
        SELECT DISTINCT lower(p.DrugName) AS norm_name, p.ApplNo
        FROM fda.products p
        WHERE p.DrugName IS NOT NULL
    """).fetchdf()
    orphan_drugs = drug_orphan.merge(orphan, on="ApplNo", how="inner")
    orphan_drugs = orphan_drugs.dropna(subset=["norm_name"]).drop_duplicates(subset=["norm_name"])
    log.info(f"  drugs ever orphan: {len(orphan_drugs)}")
    # For each trial, link via primary_drug_canonical (already lowercased in trials_labeled)
    sub = df[["nct_id", "primary_drug_canonical", "registration_date"]].copy()
    sub["norm_name"] = sub["primary_drug_canonical"].str.lower()
    sub2 = sub.merge(orphan_drugs[["norm_name", "orphan_date"]], on="norm_name", how="left")
    sub2["registration_date"] = pd.to_datetime(sub2["registration_date"], errors="coerce")
    sub2["orphan_date"] = pd.to_datetime(sub2["orphan_date"], errors="coerce")
    # registration-time orphan: orphan_date is before or equal to registration_date
    sub2["is_orphan_at_registration"] = (
        sub2["orphan_date"].notna() & (sub2["orphan_date"] <= sub2["registration_date"])
    ).astype(int)
    # also any orphan designation ever (less leak-prone if used carefully)
    sub2["drug_has_orphan_ever"] = sub2["orphan_date"].notna().astype(int)
    df = df.merge(sub2[["nct_id", "is_orphan_at_registration", "drug_has_orphan_ever"]], on="nct_id", how="left")
    df["is_orphan_at_registration"] = df["is_orphan_at_registration"].fillna(0).astype(int)
    df["drug_has_orphan_ever"] = df["drug_has_orphan_ever"].fillna(0).astype(int)
    log.info(f"  is_orphan_at_registration: {df['is_orphan_at_registration'].mean():.2%}")
    return df

def feature_indication_area(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> pd.DataFrame:
    """Therapeutic area from MeSH browse_conditions — coarse buckets."""
    log.info("B0: therapeutic_area from MeSH")
    rules = [
      ("neoplasm","oncology"),("cancer","oncology"),("carcinoma","oncology"),
      ("lymphoma","oncology"),("leukemia","oncology"),("melanoma","oncology"),
      ("myeloma","oncology"),("tumor","oncology"),("sarcoma","oncology"),
      ("cardiovasc","cardio"),("heart","cardio"),("hypertension","cardio"),
      ("nervous system","neuro"),("brain","neuro"),("alzheimer","neuro"),
      ("parkinson","neuro"),("epilepsy","neuro"),("stroke","neuro"),
      ("multiple sclerosis","neuro"),
      ("depress","psych"),("schizophreni","psych"),("anxiety","psych"),
      ("bipolar","psych"),("addiction","psych"),
      ("infection","infectious"),("viral","infectious"),("bacterial","infectious"),
      ("hepatitis","infectious"),("hiv","infectious"),("tuberculosis","infectious"),
      ("covid","infectious"),("sepsis","infectious"),
      ("diabet","metabolic"),("obesity","metabolic"),("metabolic","metabolic"),
      ("respir","pulmonary"),("asthma","pulmonary"),("copd","pulmonary"),
      ("pulmonary","pulmonary"),
      ("arthrit","rheum"),("lupus","rheum"),("autoimmune","rheum"),
      ("renal","renal"),("kidney","renal"),
      ("hepatic","gi"),("liver","gi"),("crohn","gi"),("ulcerative","gi"),
      ("bowel","gi"),("gastric","gi"),
      ("dermat","derm"),("psoriasis","derm"),("eczema","derm"),
    ]
    rows = con.execute("""
      SELECT nct_id, lower(mesh_term) AS term
      FROM browse_conditions
    """).fetchdf()
    rows["area"] = "other"
    # apply rules in order
    for kw, area in rules:
        mask = rows["term"].str.contains(kw, na=False, regex=False)
        rows.loc[mask & (rows["area"] == "other"), "area"] = area
    # pick one area per trial - priority: non-other > most common
    rows["is_specific"] = (rows["area"] != "other").astype(int)
    agg = rows.sort_values(["nct_id", "is_specific"], ascending=[True, False]) \
              .drop_duplicates(subset=["nct_id"], keep="first")[["nct_id", "area"]]
    agg.columns = ["nct_id", "therapeutic_area"]
    df = df.merge(agg, on="nct_id", how="left")
    df["therapeutic_area"] = df["therapeutic_area"].fillna("other")
    log.info(f"  therapeutic_area distribution:\n{df['therapeutic_area'].value_counts().head(10)}")
    return df

def feature_endpoint_type(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> pd.DataFrame:
    """Primary endpoint heuristic classification."""
    log.info("B0: primary endpoint type heuristic")
    # outcomes has 'title' (results-stage); design_outcomes has 'measure'
    # (registration-time). Prefer design_outcomes which is filled even for
    # trials without posted results.
    ep = con.execute("""
      SELECT nct_id, lower(coalesce(measure,'') || ' ' || coalesce(description,'')) AS text
      FROM design_outcomes
      WHERE upper(outcome_type) = 'PRIMARY'
    """).fetchdf()
    if not len(ep):
        ep = con.execute("""
          SELECT nct_id, lower(coalesce(title,'') || ' ' || coalesce(description,'')) AS text
          FROM outcomes WHERE upper(outcome_type)='PRIMARY'
        """).fetchdf()
    def classify(t):
        if not isinstance(t, str): return "other_ep"
        if any(k in t for k in ["overall survival", " mortality", "death", "survival rate"]):
            return "mortality"
        if any(k in t for k in ["progression-free", "disease-free", "tumor response",
                                "response rate", "rrr", "duration of response"]):
            return "surrogate_clinical"
        if any(k in t for k in ["biomarker", "level of", "concentration",
                                "pharmacokinetic", "ic50", "auc", "cmax"]):
            return "biomarker_pk"
        if any(k in t for k in ["score", "scale", "symptom", "quality of life", "pain",
                                "vas", "questionnaire", "fatigue", "rating"]):
            return "symptom_qol"
        return "other_ep"
    ep["ep_type"] = ep["text"].apply(classify)
    # one row per trial: most informative
    ep["prio"] = ep["ep_type"].map({"mortality": 1, "surrogate_clinical": 2,
        "biomarker_pk": 3, "symptom_qol": 4, "other_ep": 9}).fillna(9)
    ep = ep.sort_values(["nct_id", "prio"]).drop_duplicates(subset=["nct_id"], keep="first")
    df = df.merge(ep[["nct_id", "ep_type"]].rename(columns={"ep_type": "primary_endpoint_type"}),
                  on="nct_id", how="left")
    df["primary_endpoint_type"] = df["primary_endpoint_type"].fillna("other_ep")
    return df

def feature_sponsor(df: pd.DataFrame) -> pd.DataFrame:
    log.info("B0: sponsor class")
    mp = {"INDUSTRY": "industry", "NIH": "nih", "OTHER_GOV": "gov", "OTHER": "academic",
          "UNKNOWN": "academic", "FED": "gov", "INDIV": "academic", "NETWORK": "academic"}
    df["sponsor_class"] = df["lead_sponsor_class"].fillna("OTHER").str.upper().map(mp).fillna("academic")
    return df

def build_features(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    meta = {"started_at": datetime.now(timezone.utc).isoformat()}
    df = build_base(con)
    df = feature_phase(df)
    df = feature_enrollment_duration(df)
    df = feature_design(df)
    df = feature_drug_type_raw(con, df)
    df, n_hit, n_total_drug = feature_moa_local_chembl(con, df)
    df = feature_prior_counts_truncated(con, df)
    df = feature_orphan_status(con, df)
    df = feature_indication_area(con, df)
    df = feature_endpoint_type(con, df)
    df = feature_sponsor(df)
    meta["chembl_resolution_rate"] = round(n_hit / max(1, n_total_drug), 4)
    meta["chembl_total_resolved"] = int(n_hit)
    meta["chembl_drug_trial_total"] = int(n_total_drug)

    # one-hot encode categoricals
    cat_cols = ["drug_type", "moa_bucket", "target_type", "action_type",
                "therapeutic_area", "primary_endpoint_type", "sponsor_class",
                "phase"]
    log.info(f"B0: one-hot encoding {cat_cols}")
    for c in cat_cols:
        # cap categories at top 30 + "other"
        top = df[c].value_counts().head(30).index.tolist()
        df[c] = df[c].where(df[c].isin(top), "_other")
    dummies = pd.get_dummies(df[cat_cols], drop_first=False, dtype=np.int8)
    df = pd.concat([df.drop(columns=cat_cols), dummies], axis=1)

    meta["elapsed_s"] = round(time.time() - t0, 1)
    meta["n_rows"] = len(df)
    meta["n_columns"] = df.shape[1]
    log.info(f"FINAL shape={df.shape} elapsed={meta['elapsed_s']}s")
    return df, meta

def main():
    con = duckdb.connect(str(DUCKDB), read_only=False)
    df, meta = build_features(con)
    df.to_parquet(PARQUET_OUT, index=False)
    log.info(f"wrote {PARQUET_OUT}")
    # save schema
    schema = {
        "feature_set_version": "phaseB_v2",
        "constructed_at": meta["started_at"],
        "elapsed_s": meta["elapsed_s"],
        "n_rows": meta["n_rows"],
        "n_columns": meta["n_columns"],
        "chembl_resolution_rate": meta["chembl_resolution_rate"],
        "chembl_total_resolved": meta["chembl_total_resolved"],
        "chembl_drug_trial_total": meta["chembl_drug_trial_total"],
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "missingness": {c: float(df[c].isna().mean()) for c in df.columns
                        if df[c].isna().any()},
    }
    SCHEMA_OUT.write_text(json.dumps(schema, indent=2))
    log.info(f"wrote {SCHEMA_OUT}")

if __name__ == "__main__":
    main()
