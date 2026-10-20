"""
Gate 3: Feature engineering (Lo et al. 2019 tabular baseline features).

Builds a feature matrix at the trial level from AACT + FDA + ChEMBL.

Outputs:
  work/processed/features_tabular.parquet
  work/processed/features_tabular_schema.json

Resumable: ChEMBL MoA pulls are cached in `chembl.drug_moa` table inside the
existing DuckDB (`aact.duckdb`). Only drugs not yet cached are queried.

Run:
  source venv/bin/activate
  python work/src/gate3_features.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm

# Silence chembl-webresource-client's verbose HTTP logging.
for _name in (
    "chembl_webresource_client",
    "urllib3",
    "requests_cache",
    "requests",
):
    logging.getLogger(_name).setLevel(logging.WARNING)

# ---------- paths and constants ----------
PROJECT_ROOT = Path("work")
DUCKDB_PATH = PROJECT_ROOT / "aact.duckdb"
RAW_AACT_DIR = Path("work/raw/aact/extracted")
PROCESSED_DIR = PROJECT_ROOT / "processed"
LOGS_DIR = PROJECT_ROOT / "logs"
GATES_LOG = LOGS_DIR / "gates.log"
FEATURES_PARQUET = PROCESSED_DIR / "features_tabular.parquet"
SCHEMA_JSON = PROCESSED_DIR / "features_tabular_schema.json"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "gate3.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("gate3")


def notify(msg: str, urgent: bool = False) -> None:
    """Fire a push to the user's phone via ~/bin/notify. Best-effort."""
    msg = msg[:150]
    try:
        cmd = [os.environ.get("NOTIFY_BIN", "notify"), msg, "trial-failure G3"]
        if urgent:
            cmd.append("urgent")
        subprocess.run(cmd, check=False, timeout=5)
    except Exception:
        pass


# ---------- helpers ----------

PHASE_ORDINAL = {
    "EARLY_PHASE1": 0.5,
    "PHASE1": 1.0,
    "PHASE1/PHASE2": 1.5,
    "PHASE2": 2.0,
    "PHASE2/PHASE3": 2.5,
    "PHASE3": 3.0,
    "PHASE4": 4.0,
}

# MeSH ancestor → high-level therapeutic area.
# Take the first match in this priority order per trial. Curated; intentionally
# coarse — we only need coarse buckets for indication base rates.
MESH_TO_AREA: list[tuple[str, str]] = [
    # specific keywords first
    ("neoplasms", "oncology"),
    ("tumor", "oncology"),
    ("cancer", "oncology"),
    ("carcinoma", "oncology"),
    ("lymphoma", "oncology"),
    ("leukemia", "oncology"),
    ("myeloma", "oncology"),
    ("melanoma", "oncology"),
    ("cardiovascular", "cardiology"),
    ("heart disease", "cardiology"),
    ("vascular diseases", "cardiology"),
    ("hypertension", "cardiology"),
    ("hypotension", "cardiology"),
    ("nervous system", "neurology"),
    ("brain diseases", "neurology"),
    ("neurologic", "neurology"),
    ("neurodegenerative", "neurology"),
    ("alzheimer", "neurology"),
    ("parkinson", "neurology"),
    ("epilepsy", "neurology"),
    ("stroke", "neurology"),
    ("mental disorders", "psychiatry"),
    ("depressive", "psychiatry"),
    ("anxiety", "psychiatry"),
    ("schizophrenia", "psychiatry"),
    ("substance-related disorders", "psychiatry"),
    ("infections", "infectious_disease"),
    ("infection", "infectious_disease"),
    ("virus diseases", "infectious_disease"),
    ("bacterial infections", "infectious_disease"),
    ("hiv", "infectious_disease"),
    ("immune system", "immunology"),
    ("autoimmune", "immunology"),
    ("rheumatic", "immunology"),
    ("arthritis", "immunology"),
    ("endocrine", "endocrinology"),
    ("diabetes", "endocrinology"),
    ("thyroid", "endocrinology"),
    ("metabolic", "endocrinology"),
    ("nutritional", "endocrinology"),
    ("respiratory tract diseases", "respiratory"),
    ("lung diseases", "respiratory"),
    ("asthma", "respiratory"),
    ("copd", "respiratory"),
    ("pulmonary", "respiratory"),
    ("skin and connective tissue", "dermatology"),
    ("skin diseases", "dermatology"),
    ("dermatitis", "dermatology"),
    ("psoriasis", "dermatology"),
    ("digestive system", "gastroenterology"),
    ("gastrointestinal", "gastroenterology"),
    ("liver diseases", "gastroenterology"),
    ("hepatic", "gastroenterology"),
    ("urogenital", "nephrology_urology"),
    ("kidney diseases", "nephrology_urology"),
    ("renal", "nephrology_urology"),
    ("hemic and lymphatic", "hematology"),
    ("blood", "hematology"),
    ("anemia", "hematology"),
    ("pregnancy complications", "obgyn"),
    ("female urogenital", "obgyn"),
    ("eye diseases", "ophthalmology"),
    ("otorhinolaryngologic", "ent"),
    ("musculoskeletal", "musculoskeletal"),
    ("bone diseases", "musculoskeletal"),
    ("congenital", "rare_disease"),
    ("hereditary", "rare_disease"),
    ("genetic diseases", "rare_disease"),
]

# Endpoint regexes
ENDPOINT_REGEXES = [
    ("mortality", re.compile(r"\b(mortalit|death|overall survival|os\b|survival)\b", re.I)),
    ("functional", re.compile(r"\b(ecog|karnofsky|fev1|6-minute walk|six minute walk|exercise|step\b)\b", re.I)),
    ("symptom_pro", re.compile(r"\b(symptom|quality of life|qol|fatigue|pain|patient[- ]reported|prom\b|pro\b)\b", re.I)),
    ("surrogate", re.compile(r"\b(biomarker|laboratory|imaging|response rate|tumor response|hba1c|cd4|viral load|change from baseline|pharmacokinetic|pharmacodynamic|auc\b|cmax|tmax)\b", re.I)),
]

DRUG_TYPE_HEURISTICS = [
    # Order matters: more specific first.
    (re.compile(r"\b(placebo)\b", re.I), "placebo"),
    (re.compile(r"\b(vaccine|vaccin|toxoid|antigen)\b", re.I), "vaccine"),
    (re.compile(r"(mab\b|umab\b|ximab\b|zumab\b|tuzumab\b)", re.I), "biologic"),
    (re.compile(r"(cept\b|fusion protein)", re.I), "biologic"),
    (re.compile(r"\b(antibody|immunoglobulin|recombinant)\b", re.I), "biologic"),
    (re.compile(r"\b(car[- ]?t|cell therapy|stem cell|gene therapy|crispr|aav\b|lentivir)\b", re.I), "cell_gene_therapy"),
    (re.compile(r"(tide\b|peptide)", re.I), "peptide"),
    (re.compile(r"(sirna|antisense|oligonucleotide|mrna|sirna)", re.I), "oligonucleotide"),
]


def classify_endpoint_type(text: str | None) -> str:
    if text is None or pd.isna(text) or not str(text).strip():
        return "unknown"
    for label, rgx in ENDPOINT_REGEXES:
        if rgx.search(text):
            return label
    return "other"


def classify_drug_type(intervention_type, name) -> str:
    """Map to small_molecule / biologic / cell_gene_therapy / vaccine / peptide /
    oligonucleotide / placebo / other."""
    if intervention_type is None or (isinstance(intervention_type, float) and np.isnan(intervention_type)):
        intervention_type = ""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        name = ""
    it = str(intervention_type).upper()
    nm = str(name).lower()
    # name-based heuristics first (more discriminating than AACT bucket)
    for rgx, label in DRUG_TYPE_HEURISTICS:
        if rgx.search(nm):
            return label
    if it == "BIOLOGICAL":
        return "biologic"
    if it == "GENETIC":
        return "cell_gene_therapy"
    if it == "DRUG":
        return "small_molecule"
    if it in ("DIETARY_SUPPLEMENT", "RADIATION", "COMBINATION_PRODUCT",
              "DEVICE", "OTHER", "BEHAVIORAL", "PROCEDURE", "DIAGNOSTIC_TEST"):
        return "other"
    return "other"


def mesh_to_area_ordered(mesh_terms: list[str]) -> str:
    """Pick first matching therapeutic area by walking the priority list."""
    if not mesh_terms:
        return "other"
    blob = " | ".join(mesh_terms).lower()
    for keyword, area in MESH_TO_AREA:
        if keyword in blob:
            return area
    return "other"


# ---------- ChEMBL MoA pull (resumable) ----------

def ensure_chembl_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("CREATE SCHEMA IF NOT EXISTS chembl;")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS chembl.drug_moa (
            drug_canonical VARCHAR PRIMARY KEY,
            chembl_id VARCHAR,
            pref_name VARCHAR,
            molecule_type VARCHAR,
            mechanism_of_action VARCHAR,
            target_chembl_id VARCHAR,
            target_pref_name VARCHAR,
            target_type VARCHAR,
            action_type VARCHAR,
            queried_at TIMESTAMP,
            status VARCHAR  -- 'hit', 'no_molecule', 'no_mechanism', 'error'
        );
        """
    )


def fetch_chembl_for_drug(drug_name: str, molecule_client, mechanism_client, target_client) -> dict:
    """Best-effort: return MoA + target type for one drug. Always returns a row."""
    out = {
        "drug_canonical": drug_name,
        "chembl_id": None,
        "pref_name": None,
        "molecule_type": None,
        "mechanism_of_action": None,
        "target_chembl_id": None,
        "target_pref_name": None,
        "target_type": None,
        "action_type": None,
        "queried_at": datetime.now(timezone.utc),
        "status": "error",
    }
    try:
        mols = list(
            molecule_client.filter(pref_name__iexact=drug_name).only(
                ["molecule_chembl_id", "pref_name", "molecule_type"]
            )[:1]
        )
        # Fallback to synonym search if exact pref_name fails
        if not mols:
            mols = list(
                molecule_client.filter(molecule_synonyms__molecule_synonym__iexact=drug_name).only(
                    ["molecule_chembl_id", "pref_name", "molecule_type"]
                )[:1]
            )
        if not mols:
            out["status"] = "no_molecule"
            return out
        mol = mols[0]
        out["chembl_id"] = mol.get("molecule_chembl_id")
        out["pref_name"] = mol.get("pref_name")
        out["molecule_type"] = mol.get("molecule_type")
        # mechanism lookup
        mechs = list(
            mechanism_client.filter(molecule_chembl_id=out["chembl_id"]).only(
                ["mechanism_of_action", "target_chembl_id", "action_type"]
            )[:1]
        )
        if not mechs:
            out["status"] = "no_mechanism"
            return out
        mech = mechs[0]
        out["mechanism_of_action"] = (mech.get("mechanism_of_action") or "").strip().lower() or None
        out["target_chembl_id"] = mech.get("target_chembl_id")
        out["action_type"] = mech.get("action_type")
        if out["target_chembl_id"]:
            tgts = list(
                target_client.filter(target_chembl_id=out["target_chembl_id"]).only(
                    ["target_chembl_id", "target_type", "pref_name"]
                )[:1]
            )
            if tgts:
                out["target_pref_name"] = tgts[0].get("pref_name")
                out["target_type"] = tgts[0].get("target_type")
        out["status"] = "hit"
    except Exception as e:
        log.warning(f"ChEMBL query failed for {drug_name!r}: {e}")
        out["status"] = "error"
    return out


def pull_chembl_moa(
    con: duckdb.DuckDBPyConnection,
    drugs: list[str],
    batch_notify: int = 500,
    max_workers: int = 12,
) -> None:
    """Resumable, parallel ChEMBL pull.

    Skips drugs already in chembl.drug_moa. Uses a thread pool — the chembl
    client is HTTP-bound so threads (not processes) are the right tool.
    """
    ensure_chembl_schema(con)
    cached = set(
        r[0] for r in con.execute("SELECT drug_canonical FROM chembl.drug_moa").fetchall()
    )
    todo = [d for d in drugs if d and d not in cached]
    log.info(f"ChEMBL cache: {len(cached)} cached; {len(todo)} drugs to query")
    if not todo:
        return

    from chembl_webresource_client.new_client import new_client
    molecule_client = new_client.molecule
    mechanism_client = new_client.mechanism
    target_client = new_client.target

    notify(
        f"G3 ChEMBL pull: {len(todo)} drugs, {max_workers} workers, "
        f"ETA ~{len(todo)*2.5/max_workers/60:.0f}min"
    )

    hits = 0
    errors = 0
    no_mol = 0
    no_mech = 0
    batch_rows: list[dict] = []
    last_notify = 0
    t0 = time.time()
    done_count = 0

    def _work(d: str) -> dict:
        return fetch_chembl_for_drug(d, molecule_client, mechanism_client, target_client)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_work, d): d for d in todo}
        pbar = tqdm(total=len(todo), desc="ChEMBL MoA")
        for fut in as_completed(futs):
            row = fut.result()
            if row["status"] == "hit":
                hits += 1
            elif row["status"] == "error":
                errors += 1
            elif row["status"] == "no_molecule":
                no_mol += 1
            elif row["status"] == "no_mechanism":
                no_mech += 1
            batch_rows.append(row)
            done_count += 1
            pbar.update(1)

            if len(batch_rows) >= 100:
                df = pd.DataFrame(batch_rows)
                con.register("_moa_batch", df)
                con.execute(
                    """
                    INSERT INTO chembl.drug_moa
                    SELECT drug_canonical, chembl_id, pref_name, molecule_type,
                           mechanism_of_action, target_chembl_id, target_pref_name,
                           target_type, action_type, queried_at, status
                    FROM _moa_batch
                    WHERE drug_canonical NOT IN (SELECT drug_canonical FROM chembl.drug_moa)
                    """
                )
                con.unregister("_moa_batch")
                batch_rows = []

            if done_count - last_notify >= batch_notify:
                elapsed = time.time() - t0
                rate = done_count / max(elapsed, 1e-6)
                eta_min = (len(todo) - done_count) / max(rate, 1e-6) / 60
                notify(
                    f"G3 ChEMBL {done_count}/{len(todo)} hits={hits} "
                    f"nomol={no_mol} err={errors} ETA {eta_min:.0f}m"
                )
                last_notify = done_count
        pbar.close()

    # final flush
    if batch_rows:
        df = pd.DataFrame(batch_rows)
        con.register("_moa_batch", df)
        con.execute(
            """
            INSERT INTO chembl.drug_moa
            SELECT drug_canonical, chembl_id, pref_name, molecule_type,
                   mechanism_of_action, target_chembl_id, target_pref_name,
                   target_type, action_type, queried_at, status
            FROM _moa_batch
            WHERE drug_canonical NOT IN (SELECT drug_canonical FROM chembl.drug_moa)
            """
        )
        con.unregister("_moa_batch")

    notify(
        f"G3 ChEMBL done hits={hits} nomol={no_mol} nomech={no_mech} err={errors}"
    )
    log.info(
        f"ChEMBL pull complete: hits={hits} no_molecule={no_mol} "
        f"no_mechanism={no_mech} errors={errors}"
    )


# ---------- table loaders for the extra AACT files we need ----------

def ensure_designs_loaded(con: duckdb.DuckDBPyConnection) -> None:
    """Load designs.txt (allocation, masking) if not already present."""
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).fetchall()
    have = {r[0] for r in rows}
    if "designs" not in have:
        log.info("Loading designs.txt into DuckDB...")
        con.execute(
            f"""
            CREATE TABLE designs AS
            SELECT * FROM read_csv_auto(
                '{RAW_AACT_DIR / "designs.txt"}',
                delim='|', header=true, quote='"', escape='"',
                ignore_errors=true, sample_size=-1
            )
            """
        )
        n = con.execute("SELECT COUNT(*) FROM designs").fetchone()[0]
        log.info(f"designs loaded: {n} rows")
    if "design_outcomes" not in have:
        log.info("Loading design_outcomes.txt into DuckDB...")
        con.execute(
            f"""
            CREATE TABLE design_outcomes AS
            SELECT * FROM read_csv_auto(
                '{RAW_AACT_DIR / "design_outcomes.txt"}',
                delim='|', header=true, quote='"', escape='"',
                ignore_errors=true, sample_size=-1
            )
            """
        )
        n = con.execute("SELECT COUNT(*) FROM design_outcomes").fetchone()[0]
        log.info(f"design_outcomes loaded: {n} rows")


# ---------- feature assembly ----------

def build_feature_matrix(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Pulls everything into pandas once we've got the heavy tables in DuckDB.

    Returns a DataFrame indexed by nct_id, with ALL features (numeric + raw
    categoricals + one-hot).
    """
    log.info("Building base features from SQL...")
    base = con.execute(
        """
        WITH s AS (
            SELECT
                s.nct_id,
                s.phase,
                s.overall_status,
                s.start_date,
                s.completion_date,
                s.primary_completion_date,
                s.last_update_submitted_date,
                s.enrollment,
                s.number_of_arms,
                s.is_fda_regulated_drug,
                cv.has_us_facility,
                cv.actual_duration AS cv_actual_duration
            FROM studies s
            LEFT JOIN calculated_values cv USING (nct_id)
        ),
        des AS (
            SELECT nct_id,
                   UPPER(NULLIF(allocation,'')) AS allocation,
                   UPPER(NULLIF(masking,'')) AS masking,
                   UPPER(NULLIF(intervention_model,'')) AS intervention_model,
                   UPPER(NULLIF(primary_purpose,'')) AS primary_purpose
            FROM designs
        ),
        lead_sp AS (
            SELECT nct_id, MAX(UPPER(agency_class)) AS agency_class
            FROM sponsors
            WHERE LOWER(lead_or_collaborator)='lead'
            GROUP BY nct_id
        ),
        intv_agg AS (
            SELECT
                nct_id,
                COUNT(*) AS n_interventions,
                SUM(CASE WHEN UPPER(intervention_type)='DRUG' THEN 1 ELSE 0 END) AS n_drug_intv,
                SUM(CASE WHEN UPPER(intervention_type)='BIOLOGICAL' THEN 1 ELSE 0 END) AS n_biological_intv,
                MAX(CASE WHEN LOWER(name) LIKE '%placebo%'
                          OR UPPER(intervention_type) IN ('PLACEBO','PLACEBO CONTROL')
                         THEN 1 ELSE 0 END) AS has_placebo
            FROM interventions
            GROUP BY nct_id
        ),
        first_primary AS (
            SELECT nct_id, MIN(id) AS first_id
            FROM design_outcomes
            WHERE LOWER(outcome_type)='primary'
            GROUP BY nct_id
        ),
        first_primary_text AS (
            SELECT fp.nct_id, dot.measure AS first_primary_measure
            FROM first_primary fp
            JOIN design_outcomes dot ON dot.nct_id=fp.nct_id AND dot.id=fp.first_id
        ),
        all_primary_text AS (
            SELECT nct_id, STRING_AGG(measure, ' || ') AS all_primary_measures
            FROM design_outcomes
            WHERE LOWER(outcome_type)='primary'
            GROUP BY nct_id
        )
        SELECT t.nct_id,
               t.phase,
               t.registration_date,
               t.start_date AS t_start_date,
               t.completion_date AS t_completion_date,
               t.label_available,
               t.p2_success,
               t.p3_success,
               t.primary_drug_canonical,
               t.matched_canonicals,
               t.termination_reason_raw,
               s.overall_status,
               s.start_date,
               s.completion_date,
               s.primary_completion_date,
               s.last_update_submitted_date,
               s.enrollment,
               s.number_of_arms,
               s.is_fda_regulated_drug,
               s.has_us_facility,
               s.cv_actual_duration,
               des.allocation,
               des.masking,
               des.intervention_model,
               des.primary_purpose,
               lead_sp.agency_class AS sponsor_agency_class,
               intv_agg.n_interventions,
               intv_agg.n_drug_intv,
               intv_agg.n_biological_intv,
               intv_agg.has_placebo,
               fpt.first_primary_measure,
               apt.all_primary_measures
        FROM trials_labeled t
        LEFT JOIN s ON s.nct_id=t.nct_id
        LEFT JOIN des ON des.nct_id=t.nct_id
        LEFT JOIN lead_sp ON lead_sp.nct_id=t.nct_id
        LEFT JOIN intv_agg ON intv_agg.nct_id=t.nct_id
        LEFT JOIN first_primary_text fpt ON fpt.nct_id=t.nct_id
        LEFT JOIN all_primary_text apt ON apt.nct_id=t.nct_id
        """
    ).fetchdf()
    log.info(f"Base table: {base.shape}")

    # --- MeSH therapeutic area ---
    log.info("Pulling MeSH ancestors for therapeutic area...")
    mesh = con.execute(
        """
        SELECT nct_id,
               STRING_AGG(downcase_mesh_term, ' | ') AS mesh_blob
        FROM browse_conditions
        WHERE mesh_type IN ('mesh-ancestor','mesh-list')
        GROUP BY nct_id
        """
    ).fetchdf()
    log.info(f"MeSH blob table: {len(mesh)}")

    base = base.merge(mesh, on="nct_id", how="left")

    def _area_from_blob(blob):
        if blob is None or (isinstance(blob, float) and np.isnan(blob)):
            return "other"
        return mesh_to_area_ordered([blob])

    base["therapeutic_area"] = base["mesh_blob"].map(_area_from_blob)

    # --- primary drug intervention type lookup for drug_type ---
    log.info("Joining primary-drug intervention_type for drug_type classification...")
    # Use intervention_fda_matches to get the canonical → intervention_type
    # Simpler: do trials_labeled.primary_drug_canonical + check interventions
    primary_intv = con.execute(
        """
        WITH agg AS (
            SELECT m.canonical AS canonical, i.intervention_type, i.name,
                   COUNT(*) AS n
            FROM intervention_fda_matches m
            JOIN interventions i ON i.id=m.intervention_id
            WHERE m.canonical IS NOT NULL
            GROUP BY 1,2,3
        ),
        ranked AS (
            SELECT canonical, intervention_type, name,
                   ROW_NUMBER() OVER (PARTITION BY canonical ORDER BY n DESC) AS rk
            FROM agg
        )
        SELECT canonical AS primary_drug_canonical, intervention_type AS primary_intv_type, name AS primary_intv_name
        FROM ranked WHERE rk=1
        """
    ).fetchdf()
    base = base.merge(primary_intv, on="primary_drug_canonical", how="left")
    base["drug_type"] = [
        classify_drug_type(it, nm)
        for it, nm in zip(base["primary_intv_type"], base["primary_intv_name"])
    ]

    # --- ChEMBL MoA join ---
    log.info("Joining ChEMBL MoA cache...")
    moa = con.execute(
        """
        SELECT drug_canonical AS primary_drug_canonical,
               mechanism_of_action,
               target_type,
               action_type,
               status AS chembl_status
        FROM chembl.drug_moa
        """
    ).fetchdf()
    base = base.merge(moa, on="primary_drug_canonical", how="left")
    base["mechanism_of_action"] = base["mechanism_of_action"].fillna("unknown")
    base["target_type"] = base["target_type"].fillna("unknown")
    base["action_type"] = base["action_type"].fillna("unknown")

    # --- prior trials same drug / same target ---
    log.info("Computing prior trial counts (same drug, same target)...")
    # same drug: count trials with same primary_drug_canonical strictly earlier
    # we use registration_date as the anchor (already in trials_labeled)
    # same target: same mechanism_of_action (proxy)
    base["registration_date"] = pd.to_datetime(base["registration_date"])
    # For efficiency: sort by date, then groupby cumcount
    base_sorted = base.sort_values("registration_date")
    base["num_prior_trials_same_drug"] = (
        base_sorted.groupby("primary_drug_canonical").cumcount().reindex(base.index)
    )
    base["num_prior_trials_same_drug"] = base["num_prior_trials_same_drug"].fillna(0).astype(int)
    base["num_prior_trials_same_target"] = (
        base_sorted.groupby("mechanism_of_action").cumcount().reindex(base.index)
    )
    base["num_prior_trials_same_target"] = (
        base["num_prior_trials_same_target"].fillna(0).astype(int)
    )
    # For "unknown" target, this count is meaningless; mask
    unk_mask = base["mechanism_of_action"] == "unknown"
    base.loc[unk_mask, "num_prior_trials_same_target"] = 0
    base["target_novelty"] = 1.0 / (1.0 + base["num_prior_trials_same_target"])

    # --- orphan / rare disease ---
    log.info("Computing orphan-drug designation flag...")
    orphan = con.execute(
        """
        SELECT DISTINCT a.ApplNo
        FROM fda.submission_property_type spt
        JOIN fda.applications a ON a.ApplNo=spt.ApplNo
        WHERE spt.SubmissionPropertyTypeCode='Orphan'
        """
    ).fetchdf()
    orphan_apps = set(orphan["ApplNo"].astype(str))

    def _has_orphan_app(applno_list):
        if applno_list is None or (isinstance(applno_list, float) and np.isnan(applno_list)):
            return 0
        toks = [t.strip() for t in str(applno_list).split(",") if t.strip()]
        return int(any(t in orphan_apps for t in toks))

    # We have approved_applno_list in trials_labeled
    appl_lookup = con.execute(
        "SELECT nct_id, approved_applno_list FROM trials_labeled"
    ).fetchdf()
    base = base.merge(appl_lookup, on="nct_id", how="left")
    base["has_orphan_app"] = base["approved_applno_list"].map(_has_orphan_app)

    # Fallback heuristic
    rare_areas = {"rare_disease"}
    base["is_rare_disease"] = (
        (base["has_orphan_app"] == 1)
        | (
            (base["enrollment"].fillna(99999) < 100)
            & (base["therapeutic_area"].isin(rare_areas))
        )
    ).astype(int)

    # --- trial design derivations ---
    log.info("Deriving trial design features...")
    base["phase_ord"] = base["phase"].map(PHASE_ORDINAL)
    # Duration
    comp = base["completion_date"]
    comp = comp.fillna(base["primary_completion_date"]).fillna(base["last_update_submitted_date"])
    duration = (pd.to_datetime(comp) - pd.to_datetime(base["start_date"])).dt.days
    base["duration_days"] = duration.where(duration >= 0)
    # Enrollment
    base["enrollment_log1p"] = np.log1p(base["enrollment"].clip(lower=0))
    # has_placebo
    base["has_placebo"] = base["has_placebo"].fillna(0).astype(int)
    # is_randomized
    base["is_randomized"] = (base["allocation"] == "RANDOMIZED").astype(int)
    # masking ordinal
    masking_ord = {
        "NONE": 0, "OPEN": 0,
        "SINGLE": 1,
        "DOUBLE": 2,
        "TRIPLE": 3,
        "QUADRUPLE": 4,
    }
    base["masking_ord"] = base["masking"].map(masking_ord)
    base["any_blinding"] = (base["masking_ord"].fillna(0) > 0).astype(int)
    # primary endpoint type
    base["primary_endpoint_type"] = base["first_primary_measure"].map(classify_endpoint_type)
    base["primary_endpoint_types_all"] = base["all_primary_measures"].map(
        lambda blob: ",".join(sorted({classify_endpoint_type(p) for p in (blob.split("||") if isinstance(blob, str) else [])}))
        if isinstance(blob, str) else ""
    )

    # --- sponsor_type bucket ---
    def _sponsor_type(agency):
        if agency is None or (isinstance(agency, float) and np.isnan(agency)):
            return "unknown"
        a = str(agency).upper()
        if a == "INDUSTRY":
            return "industry"
        if a == "NIH":
            return "nih"
        if a in ("FED", "OTHER_GOV", "NETWORK"):
            return "government"
        if a in ("OTHER", "AMBIG", "INDIV", "UNKNOWN"):
            return "academic_or_other"
        return "unknown"

    base["sponsor_type"] = base["sponsor_agency_class"].map(_sponsor_type)

    # --- mechanism_of_action bucketing (>=20 trials else 'other_low_volume') ---
    log.info("Bucketing MoA classes...")
    moa_counts = base["mechanism_of_action"].value_counts()
    common_moa = set(moa_counts[moa_counts >= 20].index)
    base["moa_bucket"] = base["mechanism_of_action"].where(
        base["mechanism_of_action"].isin(common_moa), other="other_low_volume"
    )

    # --- naive base rates (whole-cohort, label_available=True) ---
    log.info("Computing naive base rates...")
    train_mask = base["label_available"] == True  # noqa: E712
    p3 = base.loc[train_mask, "p3_success"].astype(float)
    # indication
    ind_rates = base.loc[train_mask].groupby("therapeutic_area")["p3_success"].mean()
    base["indication_success_rate_naive"] = base["therapeutic_area"].map(ind_rates).astype(float)
    # drug class
    cls_rates = base.loc[train_mask].groupby("moa_bucket")["p3_success"].mean()
    base["drug_class_success_rate_naive"] = base["moa_bucket"].map(cls_rates).astype(float)
    # per-fold placeholders (Gate 4 will overwrite)
    base["indication_success_rate_fold"] = np.nan
    base["drug_class_success_rate_fold"] = np.nan

    return base


def select_and_encode(base: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Pick the columns we want; add one-hot encodings; add missingness indicators
    for any feature with >10% missing.

    Returns (final_df indexed by nct_id, schema_dict).
    """
    # Keep both raw categoricals and encoded versions.
    keep_cols = [
        # label-related (kept on the side; not features)
        "nct_id", "phase", "registration_date", "label_available", "p2_success", "p3_success",
        "termination_reason_raw", "primary_drug_canonical",
        # design
        "phase_ord", "enrollment", "enrollment_log1p", "duration_days",
        "number_of_arms", "has_placebo", "is_randomized", "masking_ord", "any_blinding",
        "primary_endpoint_type", "primary_endpoint_types_all",
        "sponsor_type", "sponsor_agency_class",
        # drug
        "drug_type", "mechanism_of_action", "moa_bucket", "target_type", "action_type",
        "num_prior_trials_same_drug", "num_prior_trials_same_target", "target_novelty",
        # indication
        "therapeutic_area", "is_rare_disease", "has_orphan_app",
        # historical
        "indication_success_rate_naive", "drug_class_success_rate_naive",
        "indication_success_rate_fold", "drug_class_success_rate_fold",
    ]
    df = base[keep_cols].copy()

    # one-hot encode categorical features
    onehot_cols = ["phase", "primary_endpoint_type", "sponsor_type",
                   "drug_type", "therapeutic_area", "target_type", "action_type",
                   "moa_bucket"]
    encoded_parts = []
    for c in onehot_cols:
        dummies = pd.get_dummies(df[c].astype("string").fillna("unknown"),
                                 prefix=f"oh_{c}", dtype="int8")
        encoded_parts.append(dummies)
    df_encoded = pd.concat([df] + encoded_parts, axis=1)

    # missingness indicators
    miss_summary = {}
    for c in df.columns:
        if c == "nct_id":
            continue
        miss = df[c].isna().mean()
        miss_summary[c] = float(miss)
    high_miss = [c for c, m in miss_summary.items() if m > 0.10
                 and c not in ("nct_id", "termination_reason_raw",
                               "p2_success", "p3_success", "registration_date")]
    for c in high_miss:
        df_encoded[f"{c}_is_missing"] = df_encoded[c].isna().astype("int8")

    # Schema
    schema = {
        "feature_set_version": "gate3_v1",
        "constructed_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(df_encoded)),
        "n_columns": int(df_encoded.shape[1]),
        "high_missingness_features": high_miss,
        "missingness_rates": miss_summary,
        "fold_columns_placeholder": [
            "indication_success_rate_fold",
            "drug_class_success_rate_fold",
        ],
        "note": (
            "indication_success_rate_naive and drug_class_success_rate_naive "
            "are computed over the whole cohort with label_available=True. "
            "Do NOT use them in Gate 5 supervised models — use the *_fold "
            "columns, which Gate 4 will overwrite per validation split."
        ),
        "columns": [],
    }
    for c in df_encoded.columns:
        schema["columns"].append({
            "name": c,
            "dtype": str(df_encoded[c].dtype),
            "missing_rate": float(df_encoded[c].isna().mean()) if c != "nct_id" else 0.0,
        })

    return df_encoded.set_index("nct_id"), schema


# ---------- main ----------

def main() -> int:
    t0 = time.time()
    notify("G3 start: feature engineering")
    log.info(f"Gate 3 start. DuckDB={DUCKDB_PATH}")

    con = duckdb.connect(str(DUCKDB_PATH))

    # 1. Load any missing AACT tables (designs, design_outcomes).
    ensure_designs_loaded(con)

    # 2. ChEMBL pull (resumable).
    drugs = [r[0] for r in con.execute(
        "SELECT DISTINCT primary_drug_canonical FROM trials_labeled "
        "WHERE primary_drug_canonical IS NOT NULL"
    ).fetchall()]
    log.info(f"Unique primary drugs: {len(drugs)}")
    pull_chembl_moa(con, drugs, batch_notify=500)

    # 3. Feature assembly.
    notify("G3 ChEMBL phase complete; assembling features")
    base = build_feature_matrix(con)

    # 4. Select/encode/save.
    df_final, schema = select_and_encode(base)
    log.info(f"Final feature matrix: {df_final.shape}")

    df_final.to_parquet(FEATURES_PARQUET, index=True)
    with open(SCHEMA_JSON, "w") as f:
        json.dump(schema, f, indent=2, default=str)
    log.info(f"Wrote {FEATURES_PARQUET} and {SCHEMA_JSON}")

    # 5. Diagnostics for the gate log.
    n_samples, n_features = df_final.shape
    moa_cov = con.execute(
        "SELECT COUNT(*) total, SUM(CASE WHEN status='hit' THEN 1 ELSE 0 END) hits "
        "FROM chembl.drug_moa"
    ).fetchone()
    high_miss = schema["high_missingness_features"]
    elapsed = int(time.time() - t0)

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gate_line = (
        f"[{stamp}] GATE_3: PASS shape={n_samples}x{n_features} "
        f"chembl_total={moa_cov[0]} chembl_hits={moa_cov[1]} "
        f"high_miss_features={len(high_miss)} elapsed_s={elapsed}\n"
    )
    with open(GATES_LOG, "a") as f:
        f.write(gate_line)
    log.info(gate_line.strip())

    # Diagnostics dump.
    diag_path = LOGS_DIR / "gate3_diagnostics.json"
    diag = {
        "n_samples": n_samples,
        "n_features": n_features,
        "chembl_total": int(moa_cov[0]),
        "chembl_hits": int(moa_cov[1]),
        "high_missingness_features": [
            {"name": c, "missing_rate": schema["missingness_rates"][c]}
            for c in high_miss
        ],
        "therapeutic_area_top10": (
            base["therapeutic_area"].value_counts().head(10).to_dict()
        ),
        "moa_bucket_top10": (
            base["moa_bucket"].value_counts().head(10).to_dict()
        ),
        "phase_ord_distribution": (
            base["phase_ord"].value_counts(dropna=False).sort_index().to_dict()
        ),
        "is_randomized_rate": float(base["is_randomized"].mean()),
        "has_placebo_rate": float(base["has_placebo"].mean()),
        "p3_base_rate_among_labeled": float(
            base.loc[base["label_available"] == True, "p3_success"].mean()  # noqa: E712
        ),
        "p2_base_rate_among_labeled": float(
            base.loc[base["label_available"] == True, "p2_success"].mean()  # noqa: E712
        ),
        "elapsed_s": elapsed,
    }
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2, default=str)

    notify(f"G3 PASS shape={n_samples}x{n_features} chembl_hits={moa_cov[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
