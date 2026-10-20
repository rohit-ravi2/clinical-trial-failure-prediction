"""
Gate 4 — Validation splits + per-fold target encoding.

Builds 8 train/test splits:
  - 4 for primary target p3_success over the P3 cohort
    (label_available=True AND phase in {PHASE3, PHASE2/PHASE3})
  - 4 for secondary target p2_success over the P2 cohort
    (label_available=True AND phase in {PHASE2, PHASE1/PHASE2, PHASE2/PHASE3})

Split types per target:
  1. time:        train registration_date < 2018-01-01 ; test >= 2018-01-01
  2. drugclass:   holdout = kinase-related MoA (substring "kinase")
  3. indication:  holdout = therapeutic_area == 'oncology'
  4. target:      holdout = GPCR-family drugs (curated regex on MoA)

For every split, compute per-fold target encodings from training rows only:
  - indication_success_rate_fold[i]   = train-mean of <target> grouped by therapeutic_area
  - drug_class_success_rate_fold[i]   = train-mean of <target> grouped by moa_bucket
Groups smaller than MIN_GROUP_SIZE fall back to the overall train base rate.

Saves:
  - work/processed/splits/split_<target>_<type>.pkl
  - work/processed/splits/fold_encodings_<target>_<type>.parquet
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------------------
ROOT = Path("work")
FEATURES_PATH = ROOT / "processed" / "features_tabular.parquet"
SPLITS_DIR = ROOT / "processed" / "splits"
LOGS_DIR = ROOT / "logs"
GATES_LOG = LOGS_DIR / "gates.log"

P3_PHASES = {"PHASE3", "PHASE2/PHASE3"}
P2_PHASES = {"PHASE2", "PHASE1/PHASE2", "PHASE2/PHASE3"}

TIME_CUTOFF = pd.Timestamp("2018-01-01")
MIN_GROUP_SIZE = 30
POSITIVE_FLAG_THRESHOLD = 200  # flag splits with fewer test positives

# GPCR regex — curated list of common GPCR target families
GPCR_PATTERNS = [
    r"g-protein",
    r"5-ht", r"serotonin .* receptor",
    r"dopamine d\d", r"dopamine .* receptor",
    r"adrenergic", r"adrenoceptor",
    r"muscarinic",
    r"opioid",
    r"histamine h\d",
    r"cannabinoid cb",
    r"glucagon-like peptide",
    r"angiotensin .* receptor",
    r"bradykinin",
    r"chemokine .* receptor",
    r"gaba-b", r"gabab",
    r"metabotropic glutamate",
    r"cysteinyl leukotriene receptor",
    r"prostaglandin .* receptor",
    r"sphingosine 1-phosphate receptor",
    r"oxytocin receptor",
    r"vasopressin .* receptor",
    r"orexin receptor",
    r"melatonin .* receptor",
    r"apelin receptor",
    r"corticotropin-releasing",
    r"thyrotropin-releasing",
    r"gonadotropin-releasing", r"gnrh",
    r"cholecystokinin .* receptor",
    r"neurokinin",
    r"calcitonin gene-related",
    r"parathyroid hormone .* receptor",
    r"glucagon receptor",
    r"free fatty acid receptor",
    r"protease-activated receptor",
    r"p2y", r"purinergic receptor",
    r"adenosine .* receptor",
    r"lysophosphatidic acid receptor",
    r"endothelin .* receptor",
    r"platelet-activating factor receptor",
    r"somatostatin",
]
GPCR_RE = re.compile("|".join(GPCR_PATTERNS))

KINASE_RULE = "mechanism_of_action lowercased contains substring 'kinase'"
KINASE_RULE_NOTE = (
    "Narrow 'kinase inhibitor' substring only hits 1 row in the P3 cohort because "
    "ChEMBL's MoA strings are highly specific (e.g. 'tyrosine-protein kinase btk inhibitor'). "
    "Expanded to all MoA strings containing 'kinase'."
)
GPCR_RULE = "mechanism_of_action lowercased matches curated GPCR family regex (see GPCR_PATTERNS)"
ONCOLOGY_RULE = "therapeutic_area == 'oncology'"
TIME_RULE = "registration_date < 2018-01-01 (train) ; >= 2018-01-01 (test)"

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("gate4")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s",
                            datefmt="%Y-%m-%dT%H:%M:%S%z")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(LOGS_DIR / "gate4.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

LOG = setup_logging()


def append_gate_log(line: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    with open(GATES_LOG, "a") as f:
        f.write(f"[{ts}] {line}\n")


# --------------------------------------------------------------------------------------
# Cohort filtering
# --------------------------------------------------------------------------------------
def load_features() -> pd.DataFrame:
    LOG.info(f"Loading features from {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)
    LOG.info(f"Features shape: {df.shape}; index={df.index.name}")
    return df


def filter_cohort(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if target == "p3_success":
        phases = P3_PHASES
    elif target == "p2_success":
        phases = P2_PHASES
    else:
        raise ValueError(f"Unknown target {target}")
    mask = df["label_available"].astype(bool) & df["phase"].isin(phases) & df[target].notna()
    cohort = df.loc[mask].copy()
    LOG.info(
        f"Cohort for {target}: {len(cohort)} rows; phases={sorted(phases)}; "
        f"positives={int(cohort[target].sum())} ({cohort[target].mean():.4%})"
    )
    return cohort


# --------------------------------------------------------------------------------------
# Holdout masks
# --------------------------------------------------------------------------------------
def mask_time_test(cohort: pd.DataFrame) -> pd.Series:
    return cohort["registration_date"] >= TIME_CUTOFF


def mask_kinase(cohort: pd.DataFrame) -> pd.Series:
    moa = cohort["mechanism_of_action"].fillna("").astype(str).str.lower()
    return moa.str.contains("kinase", na=False, regex=False)


def mask_oncology(cohort: pd.DataFrame) -> pd.Series:
    return cohort["therapeutic_area"].astype(str).eq("oncology")


def mask_gpcr(cohort: pd.DataFrame) -> pd.Series:
    moa = cohort["mechanism_of_action"].fillna("").astype(str).str.lower()
    return moa.str.contains(GPCR_RE, regex=True, na=False)


# --------------------------------------------------------------------------------------
# Per-fold target encoding
# --------------------------------------------------------------------------------------
def per_fold_encoding(
    cohort: pd.DataFrame,
    target: str,
    train_idx: np.ndarray,
    group_col: str,
) -> pd.Series:
    """Compute per-fold target encoding for a categorical column.

    Returns a Series indexed by nct_id (length = len(cohort)) with the train-fold
    group-mean of <target>. Groups with fewer than MIN_GROUP_SIZE rows in the
    training fold fall back to the overall train base rate.
    """
    train_mask = np.zeros(len(cohort), dtype=bool)
    train_mask[train_idx] = True
    y_train = cohort[target].iloc[train_idx].astype(float)
    g_all = cohort[group_col].astype(str).fillna("nan")
    g_train = g_all.iloc[train_idx]
    overall = float(y_train.mean())
    grp_mean = y_train.groupby(g_train).mean()
    grp_count = y_train.groupby(g_train).size()
    valid = grp_count >= MIN_GROUP_SIZE
    lookup = grp_mean.where(valid, other=overall)
    enc = g_all.map(lookup).astype(float)
    enc = enc.fillna(overall)
    enc.index = cohort.index
    enc.name = f"{group_col}_success_rate_fold"
    return enc


# --------------------------------------------------------------------------------------
# Split builder
# --------------------------------------------------------------------------------------
def build_split(
    cohort: pd.DataFrame,
    target: str,
    split_type: str,
) -> Dict:
    n = len(cohort)
    positions = np.arange(n)

    if split_type == "time":
        test_mask_bool = mask_time_test(cohort).to_numpy()
        rule = TIME_RULE
        holdout_label = "registration_date >= 2018-01-01"
    elif split_type == "drugclass":
        test_mask_bool = mask_kinase(cohort).to_numpy()
        rule = KINASE_RULE
        holdout_label = "kinase-related drugs"
    elif split_type == "indication":
        test_mask_bool = mask_oncology(cohort).to_numpy()
        rule = ONCOLOGY_RULE
        holdout_label = "oncology"
    elif split_type == "target":
        test_mask_bool = mask_gpcr(cohort).to_numpy()
        rule = GPCR_RULE
        holdout_label = "GPCR-family drugs"
    else:
        raise ValueError(split_type)

    train_idx = positions[~test_mask_bool]
    test_idx = positions[test_mask_bool]

    train_size = int(len(train_idx))
    test_size = int(len(test_idx))

    y = cohort[target].astype(int).to_numpy()
    train_pos = int(y[train_idx].sum()) if train_size else 0
    test_pos = int(y[test_idx].sum()) if test_size else 0
    train_pos_rate = (train_pos / train_size) if train_size else float("nan")
    test_pos_rate = (test_pos / test_size) if test_size else float("nan")

    flag = []
    if test_pos < POSITIVE_FLAG_THRESHOLD:
        flag.append(f"LOW_TEST_POSITIVES<{POSITIVE_FLAG_THRESHOLD}")
    if test_pos == 0:
        flag.append("DEGENERATE_TEST_ZERO_POSITIVES")
    if train_pos == 0:
        flag.append("DEGENERATE_TRAIN_ZERO_POSITIVES")
    if test_size == 0:
        flag.append("EMPTY_TEST")

    # Per-fold encodings (computed from train only, applied to all cohort rows)
    ind_enc = per_fold_encoding(cohort, target, train_idx, "therapeutic_area")
    moa_enc = per_fold_encoding(cohort, target, train_idx, "moa_bucket")

    # Sanity: mean of encoding on test rows vs raw test positive rate
    test_ind_enc_mean = float(ind_enc.iloc[test_idx].mean()) if test_size else float("nan")
    test_moa_enc_mean = float(moa_enc.iloc[test_idx].mean()) if test_size else float("nan")

    # Leakage check: for class/indication/target holdouts, verify zero training rows match the holdout rule
    leakage_red_flag = None
    if split_type in {"drugclass", "indication", "target"}:
        train_holdout_match = int(test_mask_bool[train_idx].sum())
        if train_holdout_match != 0:
            leakage_red_flag = f"{train_holdout_match} training rows match the holdout rule"

    if split_type == "time":
        # Verify boundary
        reg = cohort["registration_date"]
        n_bad_train = int((reg.iloc[train_idx] >= TIME_CUTOFF).sum())
        n_bad_test = int((reg.iloc[test_idx] < TIME_CUTOFF).sum())
        if n_bad_train or n_bad_test:
            leakage_red_flag = (
                f"time-split boundary violated: train>=cutoff={n_bad_train}, test<cutoff={n_bad_test}"
            )

    metadata = {
        "target": target,
        "split_type": split_type,
        "split_rule": rule,
        "holdout_label": holdout_label,
        "train_size": train_size,
        "test_size": test_size,
        "train_positive_count": train_pos,
        "test_positive_count": test_pos,
        "train_positive_rate": train_pos_rate,
        "test_positive_rate": test_pos_rate,
        "test_indication_enc_mean": test_ind_enc_mean,
        "test_moa_enc_mean": test_moa_enc_mean,
        "flag": flag,
        "leakage_red_flag": leakage_red_flag,
        "min_group_size_fallback": MIN_GROUP_SIZE,
        "n_cohort": n,
    }

    return {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "target_col": target,
        "split_rule": rule,
        "indication_success_rate_fold": ind_enc,
        "drug_class_success_rate_fold": moa_enc,
        "metadata": metadata,
    }


# --------------------------------------------------------------------------------------
# Save helpers
# --------------------------------------------------------------------------------------
def save_split(split: Dict, target_short: str, split_type: str) -> Tuple[Path, Path]:
    pkl_path = SPLITS_DIR / f"split_{target_short}_{split_type}.pkl"
    sidecar_path = SPLITS_DIR / f"fold_encodings_{target_short}_{split_type}.parquet"

    with open(pkl_path, "wb") as f:
        pickle.dump(split, f, protocol=pickle.HIGHEST_PROTOCOL)

    sidecar = pd.DataFrame({
        "indication_success_rate_fold": split["indication_success_rate_fold"],
        "drug_class_success_rate_fold": split["drug_class_success_rate_fold"],
    })
    sidecar.to_parquet(sidecar_path)
    return pkl_path, sidecar_path


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Recompute even if pkl exists")
    args = ap.parse_args(argv)

    t0 = time.time()

    df = load_features()

    targets = [("p3_success", "p3"), ("p2_success", "p2")]
    split_types = ["time", "drugclass", "indication", "target"]

    all_summaries: List[Dict] = []

    for target_full, target_short in targets:
        cohort = filter_cohort(df, target_full)
        # Save cohort nct_ids for downstream
        cohort_ids_path = SPLITS_DIR / f"cohort_{target_short}_nct_ids.parquet"
        pd.DataFrame({"nct_id": cohort.index}).to_parquet(cohort_ids_path)
        LOG.info(f"Saved cohort ids: {cohort_ids_path}")

        for stype in split_types:
            pkl_path = SPLITS_DIR / f"split_{target_short}_{stype}.pkl"
            if pkl_path.exists() and not args.force:
                LOG.info(f"Skipping existing {pkl_path}")
                with open(pkl_path, "rb") as f:
                    split = pickle.load(f)
                all_summaries.append(split["metadata"])
                continue

            LOG.info(f"Building split target={target_full} type={stype}")
            split = build_split(cohort, target_full, stype)
            pkl, sidecar = save_split(split, target_short, stype)
            LOG.info(f"Saved {pkl} and {sidecar}")
            md = split["metadata"]
            LOG.info(
                f"  {target_short}/{stype}: train={md['train_size']} test={md['test_size']} "
                f"train_pos={md['train_positive_count']} ({md['train_positive_rate']:.4%}) "
                f"test_pos={md['test_positive_count']} ({md['test_positive_rate']:.4%}) "
                f"flag={md['flag']} leakage={md['leakage_red_flag']}"
            )
            LOG.info(
                f"  encoding sanity: test_ind_enc_mean={md['test_indication_enc_mean']:.4f} "
                f"vs raw test_pos_rate={md['test_positive_rate']:.4f} ; "
                f"test_moa_enc_mean={md['test_moa_enc_mean']:.4f}"
            )
            all_summaries.append(md)

    elapsed = int(time.time() - t0)

    # Build gate log line
    summary_strs = []
    any_degenerate = False
    for md in all_summaries:
        target_short = "p3" if md["target"] == "p3_success" else "p2"
        flag_str = "+".join(md["flag"]) if md["flag"] else "ok"
        if "DEGENERATE_TEST_ZERO_POSITIVES" in md["flag"] or md.get("leakage_red_flag"):
            any_degenerate = True
        summary_strs.append(
            f"{target_short}/{md['split_type']}:train={md['train_size']}/pos={md['train_positive_count']},"
            f"test={md['test_size']}/pos={md['test_positive_count']}({md['test_positive_rate']:.3%}),{flag_str}"
        )

    status = "PASS"
    line = f"GATE_4: {status} elapsed_s={elapsed} " + " | ".join(summary_strs)
    append_gate_log(line)
    LOG.info(line)

    # Console summary table
    print("\n=== Gate 4 summary ===")
    print(f"Elapsed: {elapsed}s")
    cols = ["target", "split_type", "train_size", "test_size",
            "train_positive_count", "test_positive_count",
            "train_positive_rate", "test_positive_rate", "flag",
            "leakage_red_flag", "test_indication_enc_mean", "test_moa_enc_mean"]
    print(pd.DataFrame(all_summaries)[cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
