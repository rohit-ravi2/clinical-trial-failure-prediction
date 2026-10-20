"""Gate 2: Drugs@FDA download + AACT join + label construction.

Resumable:
  - skips FDA zip download if file exists with matching Content-Length
  - skips extraction if extracted/ already populated
  - skips FDA table loads if fda.applications exists (use --force-fda)
  - skips trials_labeled rebuild if table exists (use --force-labels)

Hard-stop criteria:
  - Trial-level FDA match rate < 25% across Phase 2/3 trials.
  - Phase 3 base rate outside [8%, 25%].

Outputs:
  - work/raw/fda/drugsfda.zip
  - work/raw/fda/extracted/*.txt
  - DuckDB schema `fda` with applications/products/submissions/etc.
  - DuckDB table `trials_labeled`
  - work/logs/gate2_fda.log
  - work/logs/gates.log (append)
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import requests
from tqdm import tqdm

try:
    from rapidfuzz import fuzz, process as rf_process
except ImportError as e:
    raise SystemExit("rapidfuzz required: pip install rapidfuzz") from e

# ---- paths -----------------------------------------------------------------
RAW_DIR = Path("work/raw/fda")
EXTRACT_DIR = RAW_DIR / "extracted"
ZIP_PATH = RAW_DIR / "drugsfda.zip"
DUCKDB_PATH = Path("work/aact.duckdb")
LOGS_DIR = Path("work/logs")
GATES_LOG = LOGS_DIR / "gates.log"
GATE2_LOG = LOGS_DIR / "gate2_fda.log"

FDA_URL = "https://www.fda.gov/media/89850/download"
FDA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/zip,application/octet-stream,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

# Files we need from FDA zip
FDA_TABLE_FILES = {
    "applications": "Applications.txt",
    "products": "Products.txt",
    "submissions": "Submissions.txt",
    "marketing_status": "MarketingStatus.txt",
    "marketing_status_lookup": "MarketingStatus_Lookup.txt",
    "application_docs": "ApplicationDocs.txt",
    "submission_property_type": "SubmissionPropertyType.txt",
    "te": "TE.txt",
}

# ---- logging ---------------------------------------------------------------
RAW_DIR.mkdir(parents=True, exist_ok=True)
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(GATE2_LOG, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("gate2")


def append_gate_line(line: str) -> None:
    with GATES_LOG.open("a") as f:
        f.write(line.rstrip() + "\n")


def notify(msg: str, urgent: bool = False) -> None:
    try:
        args = [os.path.expanduser("~/bin/notify"), msg, "trial-failure"]
        if urgent:
            args.append("urgent")
        subprocess.run(args, timeout=10, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


# =============================================================================
# 1. FDA download + extract
# =============================================================================

def download_fda() -> tuple[int, str]:
    """Returns (size_bytes, snapshot_filename)."""
    head = requests.head(FDA_URL, timeout=30, allow_redirects=True, headers=FDA_HEADERS)
    head.raise_for_status()
    expected = int(head.headers.get("Content-Length", "0"))
    cd = head.headers.get("content-disposition", "")
    m = re.search(r"filename=([^;]+)", cd)
    snapshot_name = m.group(1).strip() if m else "drugsfda.zip"

    if ZIP_PATH.exists():
        have = ZIP_PATH.stat().st_size
        if expected and have == expected:
            log.info("FDA zip exists matching size %d; skip download", have)
            return have, snapshot_name
        log.warning("FDA zip size mismatch (have=%d expected=%d); re-downloading",
                    have, expected)
        ZIP_PATH.unlink()

    log.info("downloading FDA zip (%.2f MB) from %s", expected / 1e6, FDA_URL)
    with requests.get(FDA_URL, stream=True, timeout=120, headers=FDA_HEADERS) as r:
        r.raise_for_status()
        with ZIP_PATH.open("wb") as f, tqdm(
            total=expected, unit="B", unit_scale=True, unit_divisor=1024,
            desc="drugsfda.zip"
        ) as bar:
            for piece in r.iter_content(chunk_size=1 << 20):
                if piece:
                    f.write(piece)
                    bar.update(len(piece))
    got = ZIP_PATH.stat().st_size
    if expected and got != expected:
        raise RuntimeError(f"FDA download size mismatch: got {got}, expected {expected}")
    log.info("FDA download complete: %d bytes (%s)", got, snapshot_name)
    return got, snapshot_name


def extract_fda() -> list[Path]:
    existing = sorted(EXTRACT_DIR.glob("*.txt"))
    if len(existing) >= 5:
        log.info("FDA extracted dir already has %d .txt files; skip", len(existing))
        return existing
    log.info("extracting %s -> %s", ZIP_PATH, EXTRACT_DIR)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(EXTRACT_DIR)
    contents = list(EXTRACT_DIR.iterdir())
    if len(contents) == 1 and contents[0].is_dir():
        sub = contents[0]
        for p in sub.iterdir():
            p.rename(EXTRACT_DIR / p.name)
        sub.rmdir()
    files = sorted(EXTRACT_DIR.glob("*.txt"))
    log.info("FDA extraction produced %d .txt files", len(files))
    return files


# =============================================================================
# 2. Load FDA into DuckDB under `fda` schema
# =============================================================================

def load_fda_tables(con: duckdb.DuckDBPyConnection, force: bool = False) -> dict[str, int]:
    have_schemas = {r[0] for r in con.execute(
        "SELECT schema_name FROM information_schema.schemata"
    ).fetchall()}
    if "fda" not in have_schemas:
        con.execute("CREATE SCHEMA fda")

    have_tables = {r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='fda'"
    ).fetchall()}

    counts: dict[str, int] = {}
    for tname, fname in FDA_TABLE_FILES.items():
        fpath = EXTRACT_DIR / fname
        if not fpath.exists():
            log.warning("FDA file missing: %s", fname)
            continue
        if (not force) and (tname in have_tables):
            n = con.execute(f"SELECT COUNT(*) FROM fda.{tname}").fetchone()[0]
            if n > 0:
                log.info("  fda.%s already loaded (%d rows); skip", tname, n)
                counts[tname] = n
                continue
        log.info("loading fda.%s from %s (%.2f MB)", tname, fname,
                 fpath.stat().st_size / 1e6)
        con.execute(f"DROP TABLE IF EXISTS fda.{tname}")
        sql = f"""
            CREATE TABLE fda.{tname} AS
            SELECT * FROM read_csv_auto(
                '{fpath.as_posix()}',
                delim='\t',
                header=True,
                quote='',
                sample_size=-1,
                ignore_errors=True,
                null_padding=True,
                strict_mode=False
            )
        """
        con.execute(sql)
        n = con.execute(f"SELECT COUNT(*) FROM fda.{tname}").fetchone()[0]
        log.info("  -> fda.%s: %d rows", tname, n)
        counts[tname] = n
    return counts


# =============================================================================
# 3. Canonicalization of drug names
# =============================================================================

# Salt forms, hydrates, free forms, etc. — strip from end of drug name
SALT_PATTERNS = [
    r"\bhydrochloride\b", r"\bhcl\b", r"\bsodium\b", r"\bpotassium\b",
    r"\bcalcium\b", r"\bmagnesium\b", r"\bsulfate\b", r"\bsulphate\b",
    r"\bphosphate\b", r"\bcitrate\b", r"\bmesylate\b", r"\bmesilate\b",
    r"\bdimesylate\b", r"\btartrate\b", r"\bbitartrate\b", r"\bmaleate\b",
    r"\bfumarate\b", r"\bsuccinate\b", r"\bacetate\b", r"\boxalate\b",
    r"\btosylate\b", r"\bbesylate\b", r"\bbesilate\b", r"\bglucuronate\b",
    r"\bgluconate\b", r"\blactate\b", r"\bpamoate\b", r"\bdihydrochloride\b",
    r"\btrihydrate\b", r"\bdihydrate\b", r"\bmonohydrate\b", r"\bhydrate\b",
    r"\banhydrous\b", r"\bbromide\b", r"\bchloride\b", r"\biodide\b",
    r"\bfree\s+base\b", r"\bfree\s+acid\b", r"\bbenzoate\b", r"\bcarbonate\b",
    r"\bbicarbonate\b", r"\bglycolate\b", r"\boleate\b", r"\bpalmitate\b",
    r"\bstearate\b", r"\bnitrate\b", r"\bdisodium\b", r"\bmonosodium\b",
    r"\bethylsuccinate\b", r"\bxinafoate\b", r"\bpropionate\b",
    r"\bdipropionate\b", r"\bvalerate\b", r"\benanthate\b",
    r"\bdecanoate\b", r"\bundecanoate\b", r"\bcypionate\b",
    r"\bmonohydrochloride\b", r"\bhemihydrate\b", r"\bsesquihydrate\b",
]
SALT_RE = re.compile("|".join(SALT_PATTERNS), re.IGNORECASE)

# Strip route/dose suffixes that sometimes leak into intervention names
DOSE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|ug|g|ml|iu|units?|%)\b"
    r"|\bextended[\s-]?release\b|\bimmediate[\s-]?release\b|\bdelayed[\s-]?release\b"
    r"|\bcontrolled[\s-]?release\b|\boral\b|\binjection\b|\binjectable\b|\btablets?\b"
    r"|\bcapsules?\b|\bsolution\b|\bsuspension\b|\bcream\b|\bointment\b|\bgel\b"
    r"|\bsyrup\b|\bspray\b|\bpatch\b|\bpowder\b|\bsoft[\s-]?gel\b|\bhard[\s-]?gel\b"
    r"|\bxr\b|\bcr\b|\bir\b|\bla\b|\bsr\b|\bod\b|\bbid\b|\btid\b|\bqid\b"
    r"|\bfor\s+injection\b|\bfor\s+oral\s+use\b",
    re.IGNORECASE,
)

# Generic noise tokens
NOISE_RE = re.compile(
    r"\bplacebo\b|\bsaline\b|\bvehicle\b|\bmatching\s+placebo\b"
    r"|\bbest\s+supportive\s+care\b|\bno\s+intervention\b|\bstandard\s+of\s+care\b"
    r"|\busual\s+care\b|\bcontrol\b",
    re.IGNORECASE,
)

PAREN_RE = re.compile(r"\([^)]*\)")
BRACKET_RE = re.compile(r"\[[^\]]*\]")
PUNCT_RE = re.compile(r"[^\w\s\-]")
WS_RE = re.compile(r"\s+")

# Common prefixes used in AACT intervention names
PREFIX_RE = re.compile(
    r"^\s*(?:drug[:\-\s]+|biological[:\-\s]+|biologic[:\-\s]+|other[:\-\s]+)",
    re.IGNORECASE,
)


def canonicalize(name: str) -> str:
    if name is None:
        return ""
    s = str(name).strip().lower()
    if not s:
        return ""
    s = PREFIX_RE.sub("", s)
    s = PAREN_RE.sub(" ", s)
    s = BRACKET_RE.sub(" ", s)
    s = DOSE_RE.sub(" ", s)
    s = SALT_RE.sub(" ", s)
    s = PUNCT_RE.sub(" ", s)
    s = WS_RE.sub(" ", s).strip()
    return s


def is_noise(name) -> bool:
    if name is None:
        return True
    try:
        if pd.isna(name):
            return True
    except (TypeError, ValueError):
        pass
    s = str(name)
    if not s.strip():
        return True
    return bool(NOISE_RE.search(s))


# =============================================================================
# 4. Build the matching tables
# =============================================================================

def build_intervention_surface(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Return DataFrame of (nct_id, intervention_id, raw_name, canonical_name, source).
    Restricted to drug/biological intervention types. Combined with intervention_other_names.
    """
    log.info("building AACT intervention surface area (drug/biological only)...")

    primary = con.execute("""
        SELECT
            i.nct_id,
            i.id AS intervention_id,
            i.name AS raw_name,
            'primary' AS source
        FROM interventions i
        WHERE LOWER(i.intervention_type) IN ('drug', 'biological')
    """).fetchdf()
    log.info("  primary intervention names: %d", len(primary))

    other = con.execute("""
        SELECT
            i.nct_id,
            i.id AS intervention_id,
            ion.name AS raw_name,
            'other_name' AS source
        FROM interventions i
        JOIN intervention_other_names ion ON ion.intervention_id = i.id
        WHERE LOWER(i.intervention_type) IN ('drug', 'biological')
    """).fetchdf()
    log.info("  other intervention names: %d", len(other))

    df = pd.concat([primary, other], ignore_index=True)
    log.info("  total surface rows: %d", len(df))

    tqdm.pandas(desc="canonicalize-aact")
    df["canonical"] = df["raw_name"].progress_apply(canonicalize)
    df["noise"] = df["raw_name"].progress_apply(is_noise)
    df = df[~df["noise"] & (df["canonical"].str.len() > 1)].reset_index(drop=True)
    log.info("  after noise filter: %d", len(df))
    return df


def build_fda_surface(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Return DataFrame of (canonical, raw_name, applno, appltype).
    One row per (canonical, applno) — multiple products under one application
    collapse on canonical name (e.g. dose strengths).
    """
    log.info("building FDA drug name surface area (NDA + BLA only, no ANDA generics)...")
    df = con.execute("""
        SELECT DISTINCT
            p.ApplNo AS applno,
            a.ApplType AS appltype,
            a.SponsorName AS sponsor,
            p.DrugName AS drug_name,
            p.ActiveIngredient AS active_ingredient
        FROM fda.products p
        JOIN fda.applications a ON a.ApplNo = p.ApplNo
        WHERE (p.DrugName IS NOT NULL OR p.ActiveIngredient IS NOT NULL)
          AND a.ApplType IN ('NDA', 'BLA')
    """).fetchdf()
    log.info("  raw FDA product rows: %d", len(df))

    # Build long-form: one row per (canonical_name, applno) from drug_name OR active_ingredient
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="canonicalize-fda"):
        for col in ("drug_name", "active_ingredient"):
            v = r.get(col)
            if v is None or pd.isna(v):
                continue
            # active_ingredient may contain multiple separated by ";" or "/"
            parts = re.split(r"[;/]|\sand\s|\swith\s|\splus\s", str(v))
            for part in parts:
                c = canonicalize(part)
                if len(c) > 1:
                    rows.append({
                        "applno": r["applno"],
                        "appltype": r["appltype"],
                        "sponsor": r["sponsor"],
                        "raw_name": part.strip(),
                        "canonical": c,
                        "source_col": col,
                    })
    long = pd.DataFrame(rows).drop_duplicates(subset=["applno", "canonical"])
    log.info("  FDA canonical (applno, name) rows: %d distinct canonical: %d",
             len(long), long["canonical"].nunique())
    return long


# =============================================================================
# 5. Matching
# =============================================================================

def match_drugs(aact_df: pd.DataFrame, fda_df: pd.DataFrame
                ) -> tuple[pd.DataFrame, dict]:
    """Two-pass match. Returns (matches_df, stats).
    matches_df columns: nct_id, intervention_id, source, raw_name, canonical,
    match_type, matched_canonical, applno_list (semicolon-joined), score.
    """
    stats: dict = {}

    # FDA canonical → set of applnos
    fda_groups = fda_df.groupby("canonical")["applno"].apply(
        lambda s: ";".join(sorted({str(x) for x in s}))
    ).to_dict()
    fda_canon_set = set(fda_groups.keys())
    log.info("  FDA canonical name set size: %d", len(fda_canon_set))
    stats["fda_canonical_count"] = len(fda_canon_set)

    # PASS 1: exact match on canonical name
    log.info("PASS 1: exact canonical match...")
    aact_df = aact_df.copy()
    aact_df["match_type"] = "none"
    aact_df["matched_canonical"] = None
    aact_df["applno_list"] = None
    aact_df["score"] = 0.0

    exact_mask = aact_df["canonical"].isin(fda_canon_set)
    aact_df.loc[exact_mask, "match_type"] = "exact"
    aact_df.loc[exact_mask, "matched_canonical"] = aact_df.loc[exact_mask, "canonical"]
    aact_df.loc[exact_mask, "applno_list"] = aact_df.loc[exact_mask, "canonical"].map(fda_groups)
    aact_df.loc[exact_mask, "score"] = 100.0
    stats["exact_rows"] = int(exact_mask.sum())
    log.info("  exact matched rows: %d / %d (%.1f%%)",
             stats["exact_rows"], len(aact_df), 100 * stats["exact_rows"] / max(len(aact_df), 1))

    # PASS 2: fuzzy on unique residual canonicals
    log.info("PASS 2: fuzzy match (rapidfuzz token_set_ratio, threshold>=92)...")
    residual = aact_df.loc[~exact_mask, "canonical"].unique()
    residual = [c for c in residual if c and len(c) >= 3]
    log.info("  unique residual canonicals: %d", len(residual))

    fda_choices = list(fda_canon_set)
    fuzzy_map: dict[str, tuple[str | None, float]] = {}
    THRESHOLD = 92

    # rapidfuzz's process.extractOne is fast in C; iterate residuals
    for q in tqdm(residual, desc="fuzzy"):
        # Use WRatio for robust similarity
        result = rf_process.extractOne(
            q, fda_choices,
            scorer=fuzz.WRatio,
            score_cutoff=THRESHOLD,
        )
        if result is not None:
            choice, score, _ = result
            fuzzy_map[q] = (choice, float(score))
        else:
            fuzzy_map[q] = (None, 0.0)

    log.info("  fuzzy matched unique canonicals: %d / %d",
             sum(1 for v in fuzzy_map.values() if v[0] is not None), len(residual))

    def assign_fuzzy(row):
        if row["match_type"] != "none":
            return row
        v = fuzzy_map.get(row["canonical"])
        if v is None or v[0] is None:
            return row
        choice, score = v
        row = row.copy()
        row["match_type"] = "fuzzy"
        row["matched_canonical"] = choice
        row["applno_list"] = fda_groups.get(choice)
        row["score"] = score
        return row

    # Vectorized: build choice and score columns via map, then fill
    aact_df["__fuzzy_choice"] = aact_df["canonical"].map(lambda c: fuzzy_map.get(c, (None, 0.0))[0])
    aact_df["__fuzzy_score"] = aact_df["canonical"].map(lambda c: fuzzy_map.get(c, (None, 0.0))[1])
    fuzzy_mask = (aact_df["match_type"] == "none") & aact_df["__fuzzy_choice"].notna()
    aact_df.loc[fuzzy_mask, "match_type"] = "fuzzy"
    aact_df.loc[fuzzy_mask, "matched_canonical"] = aact_df.loc[fuzzy_mask, "__fuzzy_choice"]
    aact_df.loc[fuzzy_mask, "applno_list"] = aact_df.loc[fuzzy_mask, "__fuzzy_choice"].map(fda_groups)
    aact_df.loc[fuzzy_mask, "score"] = aact_df.loc[fuzzy_mask, "__fuzzy_score"]
    aact_df.drop(columns=["__fuzzy_choice", "__fuzzy_score"], inplace=True)

    stats["fuzzy_rows"] = int(fuzzy_mask.sum())
    stats["unmatched_rows"] = int((aact_df["match_type"] == "none").sum())
    log.info("  fuzzy matched rows: %d", stats["fuzzy_rows"])
    log.info("  unmatched rows: %d", stats["unmatched_rows"])
    return aact_df, stats


# =============================================================================
# 6. Label construction
# =============================================================================

def build_trials_labeled(con: duckdb.DuckDBPyConnection, matches: pd.DataFrame
                         ) -> tuple[int, dict]:
    """Build trials_labeled table joining trial-level data with FDA matches.

    Definitions:
      - phase2_success: phase contains 'PHASE2' AND overall_status='COMPLETED'
        AND (drug appears in later Phase3 trial OR has any FDA approval).
      - phase3_success: phase contains 'PHASE3' AND overall_status='COMPLETED'
        AND drug has any approved NDA/BLA in FDA data.
      - label_available = False for:
            - registration_date >= 2021-01-01 AND outcome ambiguous
              (i.e., still ACTIVE / RECRUITING / not yet terminal)
            - overall_status='UNKNOWN' AND no FDA evidence
            - missing registration date
    """
    log.info("registering matches as DuckDB table tmp_matches (%d rows)", len(matches))
    # Keep only matched rows for the join
    matched_only = matches[matches["match_type"] != "none"][
        ["nct_id", "intervention_id", "raw_name", "canonical", "matched_canonical",
         "match_type", "applno_list", "score"]
    ].copy()
    con.register("tmp_matches", matched_only)

    # Per-trial summary: which interventions matched, list of FDA applnos
    log.info("aggregating per-trial match info...")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE trial_match_agg AS
        SELECT
            nct_id,
            COUNT(DISTINCT intervention_id) AS matched_intervention_count,
            STRING_AGG(DISTINCT canonical, ';') AS matched_canonicals,
            STRING_AGG(DISTINCT applno_list, ';') AS all_applnos_raw,
            MAX(CASE WHEN match_type='exact' THEN 1 ELSE 0 END) AS has_exact,
            MAX(CASE WHEN match_type='fuzzy' THEN 1 ELSE 0 END) AS has_fuzzy
        FROM tmp_matches
        GROUP BY nct_id
    """)

    # Drug-level cross-trial lookup: canonical drug → set of nct_ids with phase, status, registration_date
    log.info("building drug-level cross-trial table...")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE drug_trial_long AS
        SELECT
            tm.canonical AS drug_canonical,
            s.nct_id,
            s.phase,
            s.overall_status,
            s.study_first_submitted_date AS registration_date,
            s.completion_date,
            COALESCE(tma.matched_canonicals, '') AS trial_matched_canonicals,
            tma.all_applnos_raw
        FROM tmp_matches tm
        JOIN studies s ON s.nct_id = tm.nct_id
        LEFT JOIN trial_match_agg tma ON tma.nct_id = s.nct_id
    """)

    # NDA/BLA approval status per drug: does this canonical appear in fda.submissions
    # with SubmissionStatus='AP' (approved)? Use applno set from matches.
    # An applno is "approved" if any submission row has SubmissionStatus = 'AP'.
    log.info("building approved-applno set with first-approval dates...")
    # Approval = at least one ORIG-type submission with SubmissionStatus='AP'
    # AND the parent application is NDA or BLA (exclude ANDA generics).
    # We also pull the earliest approval date for that applno so we can
    # time-restrict trial-level success labels (a Phase 3 trial whose drug
    # was approved decades earlier or much later is not a fresh approval
    # signal — it's repurposing or unrelated development).
    con.execute("""
        CREATE OR REPLACE TEMP TABLE approved_applnos AS
        SELECT
            CAST(s.ApplNo AS VARCHAR) AS applno,
            MIN(s.SubmissionStatusDate) AS first_approval_date
        FROM fda.submissions s
        JOIN fda.applications a ON CAST(a.ApplNo AS VARCHAR) = CAST(s.ApplNo AS VARCHAR)
        WHERE UPPER(s.SubmissionStatus) = 'AP'
          AND UPPER(s.SubmissionType) = 'ORIG'
          AND a.ApplType IN ('NDA', 'BLA')
        GROUP BY 1
    """)
    n_approved = con.execute("SELECT COUNT(*) FROM approved_applnos").fetchone()[0]
    log.info("  approved applnos: %d", n_approved)

    # For each drug_canonical, does it map to any approved NDA/BLA?
    # FDA application types: 'NDA' (small molecule), 'BLA' (biologic),
    # 'ANDA' (generic — exclude from "new drug approval"),
    # 'NDA Authorized Generic' etc. Use AppType from applications.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE drug_approval_status AS
        SELECT
            tm.canonical AS drug_canonical,
            MAX(CASE WHEN a.ApplType IN ('NDA', 'BLA') AND aa.applno IS NOT NULL THEN 1 ELSE 0 END) AS has_nda_bla_approval,
            MIN(aa.first_approval_date) AS first_approval_date,
            STRING_AGG(DISTINCT CASE WHEN aa.applno IS NOT NULL THEN CAST(a.ApplNo AS VARCHAR) END, ';') AS approved_applno_list
        FROM tmp_matches tm
        LEFT JOIN UNNEST(STRING_SPLIT(tm.applno_list, ';')) AS u(applno) ON TRUE
        LEFT JOIN fda.applications a ON CAST(a.ApplNo AS VARCHAR) = u.applno
        LEFT JOIN approved_applnos aa ON aa.applno = CAST(a.ApplNo AS VARCHAR)
        GROUP BY tm.canonical
    """)
    n_drugs_with_approval = con.execute(
        "SELECT COUNT(*) FROM drug_approval_status WHERE has_nda_bla_approval=1"
    ).fetchone()[0]
    log.info("  drug canonicals with any approved NDA/BLA: %d", n_drugs_with_approval)

    # Drug-level advancement: did this canonical drug ever appear in a Phase 3 trial?
    log.info("building phase-advancement table...")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE drug_max_phase AS
        SELECT
            drug_canonical,
            MAX(CASE WHEN phase LIKE '%PHASE3%' OR phase LIKE '%PHASE4%' THEN 1 ELSE 0 END) AS has_any_phase3_trial,
            MAX(CASE WHEN phase LIKE '%PHASE2%' OR phase LIKE '%PHASE3%' OR phase LIKE '%PHASE4%' THEN 1 ELSE 0 END) AS has_any_phase2_trial,
            MIN(CASE WHEN phase LIKE '%PHASE3%' THEN registration_date END) AS first_phase3_date
        FROM drug_trial_long
        GROUP BY drug_canonical
    """)

    # Per-trial primary drug canonical: pick the one with highest match score for each nct_id
    log.info("building per-trial primary drug...")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE trial_primary_drug AS
        SELECT
            tm.nct_id,
            tm.canonical AS primary_drug_canonical,
            tm.matched_canonical AS primary_matched_canonical
        FROM tmp_matches tm
        JOIN (
            SELECT nct_id, MAX(score) AS max_score
            FROM tmp_matches
            GROUP BY nct_id
        ) j ON j.nct_id = tm.nct_id AND j.max_score = tm.score
        QUALIFY ROW_NUMBER() OVER (PARTITION BY tm.nct_id ORDER BY tm.match_type, tm.canonical) = 1
    """)

    # Now assemble trials_labeled
    log.info("assembling trials_labeled...")
    con.execute("DROP TABLE IF EXISTS trials_labeled")
    con.execute("""
        CREATE TABLE trials_labeled AS
        WITH base AS (
            SELECT
                s.nct_id,
                s.phase,
                s.overall_status,
                s.study_first_submitted_date AS registration_date,
                s.completion_date,
                s.why_stopped AS termination_reason_raw,
                s.start_date,
                tma.matched_intervention_count,
                tma.matched_canonicals,
                tpd.primary_drug_canonical,
                tpd.primary_matched_canonical,
                COALESCE(tma.has_exact, 0) AS has_exact,
                COALESCE(tma.has_fuzzy, 0) AS has_fuzzy,
                (tma.matched_intervention_count IS NOT NULL
                 AND tma.matched_intervention_count > 0) AS fda_match,
                das.has_nda_bla_approval,
                das.first_approval_date,
                das.approved_applno_list,
                dmp.has_any_phase3_trial,
                dmp.first_phase3_date
            FROM studies s
            LEFT JOIN trial_match_agg tma ON tma.nct_id = s.nct_id
            LEFT JOIN trial_primary_drug tpd ON tpd.nct_id = s.nct_id
            LEFT JOIN drug_approval_status das
                ON das.drug_canonical = tpd.primary_drug_canonical
            LEFT JOIN drug_max_phase dmp
                ON dmp.drug_canonical = tpd.primary_drug_canonical
        )
        SELECT
            nct_id,
            phase,
            overall_status,
            registration_date,
            completion_date,
            start_date,
            termination_reason_raw,
            matched_intervention_count,
            matched_canonicals,
            primary_drug_canonical,
            primary_matched_canonical,
            has_exact,
            has_fuzzy,
            fda_match,
            COALESCE(has_nda_bla_approval, 0) AS has_nda_bla_approval,
            first_approval_date,
            approved_applno_list,
            COALESCE(has_any_phase3_trial, 0) AS drug_has_any_phase3_trial,
            first_phase3_date AS drug_first_phase3_date,
            -- Time-window flag: approval is "fresh" relative to this trial if
            -- the drug's first NDA/BLA approval comes within [registration -
            -- 3y, completion + 10y]. The wide forward window catches Phase 3
            -- trials whose data supported a later approval (typical 1-3 year
            -- review). The narrow backward window catches drugs whose Phase 3
            -- ran shortly before their pivotal approval. Both windows exclude
            -- "drug was approved decades ago, this is repurposing" cases.
            CASE
                WHEN has_nda_bla_approval = 1 AND first_approval_date IS NOT NULL THEN
                    CASE
                        WHEN registration_date IS NULL THEN 0
                        WHEN first_approval_date BETWEEN
                                registration_date - INTERVAL '3 years'
                                AND COALESCE(completion_date, registration_date + INTERVAL '15 years')
                                     + INTERVAL '10 years'
                            THEN 1
                        ELSE 0
                    END
                ELSE 0
            END AS approval_in_window,
            -- Phase 2 label
            CASE
                WHEN phase NOT LIKE '%PHASE2%' THEN NULL
                WHEN overall_status IN ('TERMINATED','WITHDRAWN','SUSPENDED') THEN FALSE
                WHEN overall_status = 'COMPLETED' AND (
                    COALESCE(has_any_phase3_trial, 0) = 1
                    OR COALESCE(has_nda_bla_approval, 0) = 1
                ) THEN TRUE
                WHEN overall_status = 'COMPLETED' THEN FALSE
                ELSE NULL
            END AS p2_success,
            -- Phase 3 label: trial completed AND drug's first NDA/BLA approval
            -- is "fresh" relative to trial (approval_in_window=1). This
            -- excludes repurposing trials of drugs approved decades earlier.
            CASE
                WHEN phase NOT LIKE '%PHASE3%' THEN NULL
                WHEN overall_status IN ('TERMINATED','WITHDRAWN','SUSPENDED') THEN FALSE
                WHEN overall_status = 'COMPLETED' AND COALESCE(has_nda_bla_approval, 0) = 1
                     AND first_approval_date IS NOT NULL
                     AND registration_date IS NOT NULL
                     AND first_approval_date BETWEEN
                            registration_date - INTERVAL '3 years'
                            AND COALESCE(completion_date, registration_date + INTERVAL '15 years')
                                 + INTERVAL '10 years'
                     THEN TRUE
                WHEN overall_status = 'COMPLETED' THEN FALSE
                ELSE NULL
            END AS p3_success,
            -- Survivorship-bias-aware label_available flag
            CASE
                WHEN registration_date IS NULL THEN FALSE
                WHEN overall_status IN ('TERMINATED','WITHDRAWN','SUSPENDED','COMPLETED')
                     AND (fda_match = TRUE OR overall_status IN ('TERMINATED','WITHDRAWN','SUSPENDED'))
                     THEN TRUE
                WHEN overall_status IN ('TERMINATED','WITHDRAWN','SUSPENDED','COMPLETED')
                     AND registration_date < DATE '2021-01-01'
                     THEN TRUE
                WHEN overall_status = 'COMPLETED'
                     AND registration_date < DATE '2021-01-01'
                     THEN TRUE
                WHEN overall_status = 'UNKNOWN'
                     AND COALESCE(has_nda_bla_approval, 0) = 1
                     THEN TRUE
                ELSE FALSE
            END AS label_available,
            CASE
                WHEN registration_date IS NULL THEN 'no_reg_date'
                WHEN overall_status = 'UNKNOWN' AND COALESCE(has_nda_bla_approval, 0) = 0 THEN 'unknown_status_no_fda'
                WHEN overall_status NOT IN ('TERMINATED','WITHDRAWN','SUSPENDED','COMPLETED','UNKNOWN')
                     AND registration_date >= DATE '2021-01-01' THEN 'post_2020_active'
                WHEN overall_status NOT IN ('TERMINATED','WITHDRAWN','SUSPENDED','COMPLETED','UNKNOWN')
                     THEN 'active_pre_2021'
                ELSE 'labeled'
            END AS label_available_reason
        FROM base
    """)

    n_total = con.execute("SELECT COUNT(*) FROM trials_labeled").fetchone()[0]
    log.info("trials_labeled row count: %d", n_total)

    # Base rate stats
    stats: dict = {"n_trials_labeled": n_total}

    # Phase 2/3 trial-level match rate (only phase 2/3 trials)
    p23 = con.execute("""
        SELECT
            COUNT(*) AS n_total,
            SUM(CASE WHEN fda_match THEN 1 ELSE 0 END) AS n_matched
        FROM trials_labeled
        WHERE phase LIKE '%PHASE2%' OR phase LIKE '%PHASE3%'
    """).fetchone()
    stats["p23_total"] = p23[0]
    stats["p23_matched"] = p23[1] or 0
    stats["p23_match_rate"] = (stats["p23_matched"] / stats["p23_total"]) if stats["p23_total"] else 0.0
    log.info("Phase 2/3 trial-level match rate: %d/%d = %.2f%%",
             stats["p23_matched"], stats["p23_total"], 100 * stats["p23_match_rate"])

    # Phase 2 base rate
    p2 = con.execute("""
        SELECT
            COUNT(*) FILTER (WHERE label_available AND p2_success IS NOT NULL) AS n_labeled,
            SUM(CASE WHEN label_available AND p2_success THEN 1 ELSE 0 END) AS n_success
        FROM trials_labeled
        WHERE phase LIKE '%PHASE2%'
    """).fetchone()
    stats["p2_labeled"] = p2[0]
    stats["p2_success"] = p2[1] or 0
    stats["p2_base_rate"] = (stats["p2_success"] / stats["p2_labeled"]) if stats["p2_labeled"] else 0.0
    log.info("Phase 2 base rate (label_available=True): %d/%d = %.2f%%",
             stats["p2_success"], stats["p2_labeled"], 100 * stats["p2_base_rate"])

    # Phase 3 base rate
    p3 = con.execute("""
        SELECT
            COUNT(*) FILTER (WHERE label_available AND p3_success IS NOT NULL) AS n_labeled,
            SUM(CASE WHEN label_available AND p3_success THEN 1 ELSE 0 END) AS n_success
        FROM trials_labeled
        WHERE phase LIKE '%PHASE3%'
    """).fetchone()
    stats["p3_labeled"] = p3[0]
    stats["p3_success"] = p3[1] or 0
    stats["p3_base_rate"] = (stats["p3_success"] / stats["p3_labeled"]) if stats["p3_labeled"] else 0.0
    log.info("Phase 3 base rate (label_available=True): %d/%d = %.2f%%",
             stats["p3_success"], stats["p3_labeled"], 100 * stats["p3_base_rate"])

    # label_available breakdown
    la_dist = con.execute("""
        SELECT label_available, label_available_reason, COUNT(*) AS n
        FROM trials_labeled
        GROUP BY 1, 2
        ORDER BY n DESC
    """).fetchdf()
    log.info("label_available distribution:\n%s", la_dist.to_string(index=False))
    stats["label_available_dist"] = la_dist.to_dict(orient="records")

    return n_total, stats


# =============================================================================
# 7. Example name dumps for human review
# =============================================================================

def dump_examples(matches: pd.DataFrame, n: int = 10) -> dict:
    """Pull 10 example matched + 10 example unmatched intervention names from
    Phase 2/3 trials specifically (the gate criterion population)."""
    matched = matches[matches["match_type"].isin(["exact", "fuzzy"])]
    unmatched = matches[matches["match_type"] == "none"]
    ex_matched = matched.sample(min(n, len(matched)), random_state=42)[
        ["raw_name", "canonical", "matched_canonical", "match_type", "score"]
    ].to_dict(orient="records")
    ex_unmatched = unmatched.sample(min(n, len(unmatched)), random_state=42)[
        ["raw_name", "canonical"]
    ].to_dict(orient="records")
    return {"matched": ex_matched, "unmatched": ex_unmatched}


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-fda", action="store_true",
                        help="Re-load FDA tables even if present")
    parser.add_argument("--force-labels", action="store_true",
                        help="Rebuild trials_labeled even if present")
    args = parser.parse_args()

    t_start = time.time()
    log.info("=" * 70)
    log.info("GATE 2 start")
    notify("Gate2 starting: downloading FDA Drugs@FDA snapshot")

    # 1. FDA download + extract
    fda_size, snapshot_name = download_fda()
    extract_fda()
    notify("Gate2: FDA download+extract done")

    # 2. Load into DuckDB
    con = duckdb.connect(DUCKDB_PATH.as_posix())
    con.execute("PRAGMA memory_limit='24GB'")
    con.execute("PRAGMA threads=8")
    fda_counts = load_fda_tables(con, force=args.force_fda)
    log.info("FDA table counts: %s", fda_counts)

    # Check if trials_labeled already exists and we can skip
    have_labels = con.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema='main' AND table_name='trials_labeled'"
    ).fetchone()

    # 3. Build matching surfaces
    aact_surface = build_intervention_surface(con)
    fda_surface = build_fda_surface(con)

    # 4. Match
    matches, match_stats = match_drugs(aact_surface, fda_surface)

    # Persist matches table for later (Gate 3 will use it)
    log.info("persisting matches table (full surface, n=%d)...", len(matches))
    con.register("tmp_matches_full", matches)
    con.execute("DROP TABLE IF EXISTS intervention_fda_matches")
    con.execute("CREATE TABLE intervention_fda_matches AS SELECT * FROM tmp_matches_full")

    notify("Gate2: matching done; building labels")

    # 5. Labels
    n_trials_labeled, label_stats = build_trials_labeled(con, matches)

    # 6. Example dumps
    examples = dump_examples(matches, n=10)
    log.info("Example matched (sample of 10):")
    for ex in examples["matched"]:
        log.info("  MATCH %s [%s, %.0f]: %r -> %r",
                 ex["match_type"], ex.get("canonical"), ex["score"],
                 ex["raw_name"], ex["matched_canonical"])
    log.info("Example unmatched (sample of 10):")
    for ex in examples["unmatched"]:
        log.info("  NOMATCH raw=%r canonical=%r", ex["raw_name"], ex["canonical"])

    # 7. Gate decision
    p23_rate = label_stats["p23_match_rate"]
    p3_rate = label_stats["p3_base_rate"]
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    hard_stops = []
    if p23_rate < 0.25:
        hard_stops.append(f"p23_match_rate={p23_rate:.3f}<0.25")
    if p3_rate < 0.08 or p3_rate > 0.25:
        hard_stops.append(f"p3_base_rate={p3_rate:.3f}_outside[0.08,0.25]")

    summary = (
        f"snapshot={snapshot_name} fda_apps={fda_counts.get('applications', -1)} "
        f"fda_products={fda_counts.get('products', -1)} "
        f"p23_total={label_stats['p23_total']} p23_matched={label_stats['p23_matched']} "
        f"p23_match_rate={p23_rate*100:.2f}% "
        f"p2_base_rate={label_stats['p2_base_rate']*100:.2f}%(n={label_stats['p2_labeled']}) "
        f"p3_base_rate={p3_rate*100:.2f}%(n={label_stats['p3_labeled']}) "
        f"exact={match_stats['exact_rows']} fuzzy={match_stats['fuzzy_rows']} "
        f"unmatched={match_stats['unmatched_rows']} "
        f"elapsed_s={int(time.time()-t_start)}"
    )

    if hard_stops:
        line = f"[{ts}] GATE_2: HARD_STOP ({';'.join(hard_stops)}) {summary}"
        append_gate_line(line)
        log.error(line)
        notify(f"Gate2 HARD-STOP: {';'.join(hard_stops)}", urgent=True)
        return 2

    line = f"[{ts}] GATE_2: PASS {summary}"
    append_gate_line(line)
    log.info(line)
    notify(f"Gate2 PASS: p23match={p23_rate*100:.1f}% p3base={p3_rate*100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
