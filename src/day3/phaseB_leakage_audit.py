"""
Day-3 Phase B: Tighter leakage audit on features_v2 x p3_time (the worst offender).

Sections:
  B1 - Univariate test-set AUROC per feature (top-15 absolute deviation from 0.5)
  B2 - Univariate train-set AUROC + delta (test - train)
  B3 - ChEMBL selection bias (null-indicator predictiveness)
  B4 - MoA-bucket overlap audit (kinase inhibitor leakage into train buckets)
  B5 - FDA OOPD edge case (orphan designation same week as trial reg)
  B6 - Leave-one-DRUG-out target encoding sanity check on indication & drug-class fold encodings
  B7 - Synthesis: drop confirmed leaks, retrain on all 8 splits, compare audited AUROC

All outputs to /mnt/ssd/research/trial-failure/results/day3/.
"""
from __future__ import annotations
import json
import logging
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/mnt/ssd/research/trial-failure/logs/day3_phaseB.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("phaseB")

ROOT = Path("/mnt/ssd/research/trial-failure")
FEAT_PATH = ROOT / "processed/day2/features_v2.parquet"
SPLIT_DIR = ROOT / "processed/splits"
MODEL_DIR = ROOT / "models/day2"
OUT_DIR = ROOT / "results/day3"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_univariate_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Univariate AUROC: rank by feature value, score against label.

    Tries both directions and takes max(auc, 1-auc) so we measure |signal|
    regardless of sign. NaNs are treated as a separate value (replaced with
    a sentinel below the min).
    """
    if len(np.unique(y)) < 2:
        return float("nan")
    if pd.api.types.is_bool_dtype(x):
        xn = x.astype(float)
    elif pd.api.types.is_numeric_dtype(x):
        xn = pd.to_numeric(x, errors="coerce").astype(float)
    else:
        # categorical / string: hash-encode is meaningless; skip
        return float("nan")
    xn = np.asarray(xn, dtype=float)
    # NaN handling: replace with sentinel below min
    mask = np.isnan(xn)
    if mask.all():
        return float("nan")
    if mask.any():
        finite_min = np.nanmin(xn)
        xn = np.where(mask, finite_min - 1.0, xn)
    # if constant after NaN fill
    if np.unique(xn).size < 2:
        return float("nan")
    try:
        auc = roc_auc_score(y, xn)
    except Exception:
        return float("nan")
    return max(auc, 1.0 - auc)


def load_cohort_and_split(target: str, split_type: str):
    """Load the per-trial labeled cohort and the named split.

    CRITICAL: the split's integer indices are anchored to *cohort order*
    (the order of NCT ids in cohort_<task>_nct_ids.parquet), NOT to the
    original features_v2.parquet row order. Day-2 training used
    `df.set_index('nct_id').loc[cohort_ncts]` which reorders rows to
    match cohort order. We must do the same here, otherwise the positional
    indices select the wrong rows.

    Returns:
        feats: DataFrame in cohort order (one row per trial in cohort).
        train_idx, test_idx: numpy int arrays into that order.
        sp: full split dict (with fold encodings).
    """
    full = pd.read_parquet(FEAT_PATH)
    cohort_ids = pd.read_parquet(SPLIT_DIR / f"cohort_{target.split('_')[0]}_nct_ids.parquet")
    if "nct_id" in cohort_ids.columns:
        keep = cohort_ids["nct_id"].tolist()
    else:
        keep = cohort_ids.iloc[:, 0].tolist()
    feats = full.set_index("nct_id").loc[keep].reset_index()
    sp = pickle.load(open(SPLIT_DIR / f"split_{target.split('_')[0]}_{split_type}.pkl", "rb"))
    return feats, np.asarray(sp["train_idx"]), np.asarray(sp["test_idx"]), sp


def attach_fold_encodings(feats: pd.DataFrame, sp: dict) -> pd.DataFrame:
    out = feats.copy()
    ind_enc = sp["indication_success_rate_fold"]
    moa_enc = sp["drug_class_success_rate_fold"]
    out["indication_success_rate_fold"] = out["nct_id"].map(ind_enc).astype(float)
    out["drug_class_success_rate_fold"] = out["nct_id"].map(moa_enc).astype(float)
    return out


def b1_b2_univariate(feats: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray,
                     label_col: str, model_features: list[str]) -> pd.DataFrame:
    """B1: univariate test AUROC; B2: train AUROC + delta."""
    y_train = feats.loc[train_idx, label_col].astype(float).to_numpy()
    y_test = feats.loc[test_idx, label_col].astype(float).to_numpy()
    rows = []
    for f in model_features:
        if f not in feats.columns:
            continue
        x_train = feats.loc[train_idx, f]
        x_test = feats.loc[test_idx, f]
        if not (pd.api.types.is_numeric_dtype(x_train) or pd.api.types.is_bool_dtype(x_train)):
            continue
        auc_tr = safe_univariate_auc(x_train.to_numpy(), y_train)
        auc_te = safe_univariate_auc(x_test.to_numpy(), y_test)
        rows.append({
            "feature": f,
            "train_auc": auc_tr,
            "test_auc": auc_te,
            "delta_test_minus_train": (auc_te - auc_tr) if not (np.isnan(auc_tr) or np.isnan(auc_te)) else float("nan"),
            "test_dev_from_half": (auc_te - 0.5) if not np.isnan(auc_te) else float("nan"),
            "train_null_rate": float(x_train.isna().mean()),
            "test_null_rate": float(x_test.isna().mean()),
        })
    out = pd.DataFrame(rows).sort_values("test_auc", ascending=False, key=lambda s: s.abs())
    out["abs_test_dev"] = out["test_dev_from_half"].abs()
    out = out.sort_values("abs_test_dev", ascending=False).reset_index(drop=True)
    return out


def b3_chembl_null_indicator(feats: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray,
                             label_col: str) -> pd.DataFrame:
    """For each ChEMBL-derived column, compute null-indicator's univariate AUROC.

    chembl_resolved was dropped already; but the null-indicator of any ChEMBL
    column should be the SAME information.
    """
    chembl_cols = [c for c in feats.columns if c.startswith("moa_bucket_")
                   or c.startswith("target_type_") or c.startswith("action_type_")]
    # Test the *master* null indicator: chembl_id, target_chembl_id, mechanism_of_action
    candidates = ["chembl_id", "target_chembl_id", "primary_drug_canonical",
                  "matched_intervention_count", "chembl_resolved"]
    rows = []
    y_test = feats.loc[test_idx, label_col].astype(float).to_numpy()
    y_train = feats.loc[train_idx, label_col].astype(float).to_numpy()
    for c in candidates:
        if c not in feats.columns:
            rows.append({"col": c, "present": False, "test_null_indicator_auc": None,
                         "train_null_indicator_auc": None})
            continue
        null_train = feats.loc[train_idx, c].isna().astype(float).to_numpy()
        null_test = feats.loc[test_idx, c].isna().astype(float).to_numpy()
        auc_tr = safe_univariate_auc(null_train, y_train) if null_train.std() > 0 else float("nan")
        auc_te = safe_univariate_auc(null_test, y_test) if null_test.std() > 0 else float("nan")
        rows.append({
            "col": c,
            "present": True,
            "train_null_rate": float(null_train.mean()),
            "test_null_rate": float(null_test.mean()),
            "train_null_indicator_auc": auc_tr,
            "test_null_indicator_auc": auc_te,
        })
    return pd.DataFrame(rows)


def b4_moa_kinase_overlap(feats: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> dict:
    """For the drug-class split (kinase inhibitor holdout), check that no
    training MoA bucket has >50% kinase trials.

    Heuristic for 'kinase inhibitor': mechanism_of_action contains 'kinase'
    AND 'inhibitor' (substring match).
    """
    train = feats.loc[train_idx].copy()
    test = feats.loc[test_idx].copy()
    moa_train = train["mechanism_of_action"].fillna("").str.lower()
    moa_test = test["mechanism_of_action"].fillna("").str.lower()
    train["is_kinase"] = moa_train.str.contains("kinase") & moa_train.str.contains("inhibitor")
    test["is_kinase"] = moa_test.str.contains("kinase") & moa_test.str.contains("inhibitor")
    moa_bucket_cols = [c for c in feats.columns if c.startswith("moa_bucket_")]
    per_bucket = []
    for c in moa_bucket_cols:
        n_train = int(train[c].sum())
        n_kinase_train = int((train["is_kinase"] & (train[c] == 1)).sum())
        frac = n_kinase_train / max(1, n_train)
        per_bucket.append({"bucket": c, "n_train": n_train, "n_kinase_train": n_kinase_train,
                           "frac_kinase": frac, "flag_gt_50pct": frac > 0.5})
    return {
        "train_kinase_count": int(train["is_kinase"].sum()),
        "test_kinase_count": int(test["is_kinase"].sum()),
        "test_kinase_frac": float(test["is_kinase"].mean()),
        "train_kinase_frac": float(train["is_kinase"].mean()),
        "per_bucket": per_bucket,
    }


def b5_oopd_edge_case(feats: pd.DataFrame) -> dict:
    """Check `is_orphan_at_registration` for trials where the orphan
    designation came within +/- 7 days of trial registration.

    The full OOPD designation date isn't on the feature matrix, so we
    can only inspect what the value distribution looks like. Document
    counts and the convention used (which we infer from the column).
    """
    val_counts = feats["is_orphan_at_registration"].value_counts(dropna=False).to_dict()
    # No designation_date column on feats; we report convention is "registration_date >= designation_date"
    # (based on Day-2 docs). Without that column we can't resolve edge cases, so document the limitation.
    return {
        "is_orphan_at_registration_value_counts": {str(k): int(v) for k, v in val_counts.items()},
        "drug_has_orphan_ever_value_counts": {
            str(k): int(v) for k, v in feats["drug_has_orphan_ever"].value_counts(dropna=False).to_dict().items()
        },
        "convention": ("Day-2 sets is_orphan_at_registration=1 iff designation_date<=registration_date. "
                       "Edge-case trials with designation_date within +/-7 days of reg cannot be re-audited "
                       "from features_v2.parquet alone because designation_date is not retained on the feature "
                       "matrix. Documented limitation."),
        "drug_has_orphan_ever_in_model_features": False,  # confirmed in model inspection
        "is_orphan_at_registration_in_model_features": True,
    }


def b6_leave_one_drug_out_check(feats: pd.DataFrame, sp: dict, label_col: str,
                                target: str, n_sample: int = 10, rng_seed: int = 7) -> dict:
    """Pick 10 random test trials. For each, recompute indication and
    drug-class success rates from train rows EXCLUDING all trials of the
    test trial's compound (primary_drug_canonical). Compare to stored fold
    encoding value.
    """
    rng = np.random.default_rng(rng_seed)
    train_idx = np.asarray(sp["train_idx"])
    test_idx = np.asarray(sp["test_idx"])
    if len(test_idx) == 0:
        return {"status": "empty_test"}
    chosen = rng.choice(test_idx, size=min(n_sample, len(test_idx)), replace=False)
    train = feats.loc[train_idx]
    rows = []
    for i in chosen:
        row = feats.loc[i]
        drug = row.get("primary_drug_canonical")
        # therapeutic area: pick the one-hot column = 1
        ta_cols = [c for c in feats.columns if c.startswith("therapeutic_area_")]
        ta = None
        for c in ta_cols:
            if row.get(c) == 1:
                ta = c
                break
        moa_cols = [c for c in feats.columns if c.startswith("moa_bucket_")]
        moa = None
        for c in moa_cols:
            if row.get(c) == 1:
                moa = c
                break
        # recompute excluding this drug
        if pd.notna(drug) and drug:
            train_excl = train[train["primary_drug_canonical"] != drug]
        else:
            train_excl = train
        if ta is not None:
            same_ta = train_excl[train_excl[ta] == 1]
            recomputed_ta = float(same_ta[label_col].mean()) if len(same_ta) >= 30 else None
        else:
            recomputed_ta = None
        if moa is not None:
            same_moa = train_excl[train_excl[moa] == 1]
            recomputed_moa = float(same_moa[label_col].mean()) if len(same_moa) >= 30 else None
        else:
            recomputed_moa = None
        stored_ta = sp["indication_success_rate_fold"].get(row["nct_id"])
        stored_moa = sp["drug_class_success_rate_fold"].get(row["nct_id"])
        rows.append({
            "nct_id": row["nct_id"],
            "drug": str(drug),
            "ta_col": ta,
            "moa_col": moa,
            "stored_ta": float(stored_ta) if stored_ta is not None and pd.notna(stored_ta) else None,
            "recomputed_ta": recomputed_ta,
            "ta_diff_abs": (abs(stored_ta - recomputed_ta)
                            if recomputed_ta is not None and pd.notna(stored_ta) else None),
            "stored_moa": float(stored_moa) if stored_moa is not None and pd.notna(stored_moa) else None,
            "recomputed_moa": recomputed_moa,
            "moa_diff_abs": (abs(stored_moa - recomputed_moa)
                             if recomputed_moa is not None and pd.notna(stored_moa) else None),
        })
    # max diff across sample
    max_ta_diff = max((r["ta_diff_abs"] for r in rows if r["ta_diff_abs"] is not None), default=0.0)
    max_moa_diff = max((r["moa_diff_abs"] for r in rows if r["moa_diff_abs"] is not None), default=0.0)
    return {
        "n_sampled": len(chosen),
        "max_ta_diff_abs": max_ta_diff,
        "max_moa_diff_abs": max_moa_diff,
        "encoder_leak_flag_gt_0p01": (max_ta_diff > 0.01 or max_moa_diff > 0.01),
        "samples": rows,
    }


def compute_metrics(y_true, y_proba) -> dict:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_proba, dtype=float)
    if len(np.unique(y)) < 2:
        return {"auroc": float("nan"), "brier": float(((p - y) ** 2).mean()),
                "prauc": float("nan"), "ece": float("nan")}
    auroc = float(roc_auc_score(y, p))
    brier = float(brier_score_loss(y, p))
    prauc = float(average_precision_score(y, p))
    bins = np.linspace(0, 1, 11)
    inds = np.digitize(p, bins[1:-1])
    ece = 0.0
    n = len(y)
    for b in range(10):
        m = inds == b
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(p[m].mean() - y[m].mean())
    return {"auroc": auroc, "brier": brier, "prauc": prauc, "ece": float(ece)}


def retrain_split(target: str, split_type: str, dropped: list[str]) -> dict:
    """Retrain LightGBM on a single split after dropping `dropped` features.

    Returns dict with original and audited AUROC.
    """
    feats, train_idx, test_idx, sp = load_cohort_and_split(target, split_type)
    feats = attach_fold_encodings(feats, sp)
    label_col = target

    # load original model to get the original feature list
    orig_model_pkl = pickle.load(open(MODEL_DIR / f"lgbm_v2_{target.split('_')[0]}_{split_type}.pkl", "rb"))
    orig_features = orig_model_pkl["features"]
    audited_features = [f for f in orig_features if f not in dropped]

    # Sanity: ensure every audited feature exists in feats
    audited_features = [f for f in audited_features if f in feats.columns]

    X_tr = feats.loc[train_idx, audited_features].copy()
    X_te = feats.loc[test_idx, audited_features].copy()
    y_tr = feats.loc[train_idx, label_col].astype(float).to_numpy()
    y_te = feats.loc[test_idx, label_col].astype(float).to_numpy()

    # cast bool/Int64 to numeric
    for c in X_tr.columns:
        if X_tr[c].dtype.name in ("boolean", "Int64", "Int32"):
            X_tr[c] = X_tr[c].astype(float)
            X_te[c] = X_te[c].astype(float)
        elif X_tr[c].dtype == bool:
            X_tr[c] = X_tr[c].astype(float)
            X_te[c] = X_te[c].astype(float)

    # Match Day-2 LightGBM defaults (light params)
    params = dict(
        objective="binary",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
        verbose=-1,
    )
    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_tr, y_tr)
    p_te = clf.predict_proba(X_te)[:, 1]

    # isotonic calibration on a 80/20 of train
    rng = np.random.default_rng(11)
    n_tr = len(y_tr)
    idx_perm = rng.permutation(n_tr)
    cal_n = max(int(0.2 * n_tr), 200)
    cal_idx = idx_perm[:cal_n]
    p_cal = clf.predict_proba(X_tr.iloc[cal_idx])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal, y_tr[cal_idx])
    p_te_cal = iso.predict(p_te)

    m_raw = compute_metrics(y_te, p_te)
    m_cal = compute_metrics(y_te, p_te_cal)

    # Original model metrics: re-evaluate the stored model on test set
    # Day-2 stored a raw LightGBM Booster, not a sklearn wrapper, so use .predict()
    orig_model = orig_model_pkl["model"]
    orig_iso = orig_model_pkl.get("iso", None)
    X_te_orig = feats.loc[test_idx, orig_features].copy()
    for c in X_te_orig.columns:
        if X_te_orig[c].dtype.name in ("boolean", "Int64", "Int32"):
            X_te_orig[c] = X_te_orig[c].astype(float)
    if hasattr(orig_model, "predict_proba"):
        p_te_orig = orig_model.predict_proba(X_te_orig)[:, 1]
    else:
        # Booster.predict returns probabilities directly for binary objective
        p_te_orig = orig_model.predict(X_te_orig)
    if orig_iso is not None:
        p_te_orig_cal = orig_iso.predict(p_te_orig)
    else:
        p_te_orig_cal = p_te_orig
    m_orig = compute_metrics(y_te, p_te_orig)
    m_orig_cal = compute_metrics(y_te, p_te_orig_cal)

    return {
        "target": target,
        "split_type": split_type,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "n_pos_train": int(y_tr.sum()),
        "n_pos_test": int(y_te.sum()),
        "n_features_orig": len(orig_features),
        "n_features_audited": len(audited_features),
        "dropped_features": dropped,
        "orig_metrics_replayed": m_orig,
        "orig_metrics_calibrated": m_orig_cal,
        "audited_metrics_raw": m_raw,
        "audited_metrics_calibrated": m_cal,
        "delta_auroc": m_cal["auroc"] - m_orig_cal["auroc"],
    }


def null_label_experiment(target: str, split_type: str, seed: int = 42) -> dict:
    """Gold-standard leak test: shuffle train labels, train, eval on real test.

    AUROC should be ~0.5. If >0.55, the feature set contains information about
    the test labels via some path that does not require the train labels (i.e.
    a structural leak in features or split).
    """
    feats, train_idx, test_idx, sp = load_cohort_and_split(target, split_type)
    feats = attach_fold_encodings(feats, sp)
    orig_model_pkl = pickle.load(open(MODEL_DIR / f"lgbm_v2_{target.split('_')[0]}_{split_type}.pkl", "rb"))
    model_features = orig_model_pkl["features"]
    X_tr = feats.loc[train_idx, model_features].copy()
    X_te = feats.loc[test_idx, model_features].copy()
    for df in (X_tr, X_te):
        for c in df.columns:
            if df[c].dtype.name in ("boolean", "Int64", "Int32"):
                df[c] = df[c].astype(float)
    y_tr = feats.loc[train_idx, target].astype(float).to_numpy()
    y_te = feats.loc[test_idx, target].astype(float).to_numpy()
    rng = np.random.default_rng(seed)
    y_tr_shuf = rng.permutation(y_tr)
    params = dict(objective="binary", metric="auc", learning_rate=0.05,
                  num_leaves=63, min_child_samples=30, feature_fraction=0.85,
                  bagging_fraction=0.85, bagging_freq=5, reg_lambda=0.1,
                  verbosity=-1, num_threads=4)
    dtr = lgb.Dataset(X_tr, label=y_tr_shuf)
    model = lgb.train(params, dtr, num_boost_round=300)
    p_te = model.predict(X_te)
    null_auc = float(roc_auc_score(y_te, p_te)) if len(np.unique(y_te)) > 1 else float("nan")
    return {
        "target": target,
        "split_type": split_type,
        "null_shuffled_auroc": null_auc,
        "seed": seed,
        "verdict": ("PASS_no_structural_leakage" if abs(null_auc - 0.5) < 0.05
                    else ("WARN_mild_structural_signal" if abs(null_auc - 0.5) < 0.10
                          else "FAIL_structural_leakage")),
    }


def main():
    log.info("=" * 80)
    log.info("Day-3 Phase B: leakage audit")
    log.info("=" * 80)

    # ===== focus split: p3 / time =====
    target = "p3_success"
    split_type = "time"
    feats, train_idx, test_idx, sp = load_cohort_and_split(target, split_type)
    feats = attach_fold_encodings(feats, sp)
    log.info("Loaded cohort: %d rows, train=%d, test=%d", len(feats), len(train_idx), len(test_idx))

    # original model feature list
    orig_model_pkl = pickle.load(open(MODEL_DIR / f"lgbm_v2_p3_time.pkl", "rb"))
    model_features = orig_model_pkl["features"]
    log.info("Original model uses %d features", len(model_features))

    # B1+B2: univariate AUROCs
    log.info("B1+B2: univariate AUROCs ...")
    uni = b1_b2_univariate(feats, train_idx, test_idx, target, model_features)
    uni.to_csv(OUT_DIR / "B12_univariate.csv", index=False)
    top15 = uni.head(15)
    log.info("Top 15 by |test_AUC-0.5|:")
    for _, r in top15.iterrows():
        log.info("  %-45s  test=%.3f  train=%.3f  delta=%+.3f",
                 r["feature"], r["test_auc"], r["train_auc"], r["delta_test_minus_train"])

    # B3: ChEMBL null-indicator
    log.info("B3: ChEMBL selection bias ...")
    b3 = b3_chembl_null_indicator(feats, train_idx, test_idx, target)
    b3.to_csv(OUT_DIR / "B3_chembl_null.csv", index=False)
    log.info("B3 results:")
    for _, r in b3.iterrows():
        if r.get("present"):
            log.info("  %-30s  train_null=%.2f  test_null=%.2f  null_AUC_test=%s",
                     r["col"], r["train_null_rate"], r["test_null_rate"],
                     f"{r['test_null_indicator_auc']:.3f}" if pd.notna(r["test_null_indicator_auc"]) else "NA")
        else:
            log.info("  %-30s  NOT PRESENT (already dropped)", r["col"])

    # B4: MoA kinase overlap (drug-class split)
    log.info("B4: MoA kinase overlap on drug-class split ...")
    feats_dc, tr_dc, te_dc, sp_dc = load_cohort_and_split(target, "drugclass")
    feats_dc = attach_fold_encodings(feats_dc, sp_dc)
    b4 = b4_moa_kinase_overlap(feats_dc, tr_dc, te_dc)
    with open(OUT_DIR / "B4_kinase_overlap.json", "w") as f:
        json.dump(b4, f, indent=2)
    log.info("B4: train_kinase=%d/%d (%.1f%%), test_kinase=%d/%d (%.1f%%)",
             b4["train_kinase_count"], len(tr_dc), 100*b4["train_kinase_frac"],
             b4["test_kinase_count"], len(te_dc), 100*b4["test_kinase_frac"])
    flagged_buckets = [b for b in b4["per_bucket"] if b["flag_gt_50pct"]]
    log.info("  Buckets flagged >50%% kinase: %d", len(flagged_buckets))
    for b in flagged_buckets:
        log.info("    %s  frac=%.2f  (n=%d)", b["bucket"], b["frac_kinase"], b["n_train"])

    # B5: OOPD edge case
    log.info("B5: OOPD edge case ...")
    b5 = b5_oopd_edge_case(feats)
    with open(OUT_DIR / "B5_oopd_edge.json", "w") as f:
        json.dump(b5, f, indent=2)
    log.info("B5: %s", b5["convention"][:160])

    # B6: leave-one-drug-out target encoding sanity
    log.info("B6: leave-one-DRUG-out target encoding sanity ...")
    b6 = b6_leave_one_drug_out_check(feats, sp, target, target, n_sample=10)
    with open(OUT_DIR / "B6_loo_encoding.json", "w") as f:
        json.dump(b6, f, indent=2, default=str)
    log.info("B6: max_ta_diff=%.5f  max_moa_diff=%.5f  encoder_leak_flag=%s",
             b6["max_ta_diff_abs"], b6["max_moa_diff_abs"], b6["encoder_leak_flag_gt_0p01"])

    # ===== B7: identify confirmed leaks and retrain =====
    log.info("B7: synthesis ...")
    # Decide which features to drop.
    # Criteria for "confirmed leak":
    #   (a) abs(delta_test_minus_train) > 0.10  AND test_auc > 0.65  AND it's a feature
    #       whose post-hoc derivation is suspicious (we'll list the exact names).
    #   OR
    #   (b) is_orphan_at_registration if it shows AUC > 0.65 (post-hoc designation date)
    #
    # Per the dispatch the suspects to investigate explicitly are:
    #   - chembl_resolved      (already dropped)
    #   - drug_has_orphan_ever (already dropped)
    #   - is_orphan_at_registration  (under audit)
    #   - any other feature with massive test-set predictiveness
    suspects = []
    # Explicit suspect features called out in the orchestrator dispatch
    named_suspects = {"is_orphan_at_registration", "drug_has_orphan_ever", "chembl_resolved",
                      "fda_match", "matched_intervention_count"}
    for _, r in uni.iterrows():
        f = r["feature"]
        # heuristic A: features named in the dispatch
        if f in named_suspects:
            # Only drop if it's actually predictive (otherwise no harm)
            if (not np.isnan(r["test_auc"])) and r["test_auc"] > 0.55:
                suspects.append({"feature": f, "test_auc": float(r["test_auc"]),
                                 "train_auc": float(r["train_auc"]),
                                 "delta": float(r["delta_test_minus_train"]),
                                 "reason": "post_hoc_FDA_or_ChEMBL_derived"})
        # heuristic B: extreme train/test AUROC delta
        elif (not np.isnan(r["delta_test_minus_train"])) and abs(r["delta_test_minus_train"]) > 0.10 and r["test_auc"] > 0.65:
            suspects.append({"feature": f, "test_auc": float(r["test_auc"]),
                             "train_auc": float(r["train_auc"]),
                             "delta": float(r["delta_test_minus_train"]),
                             "reason": "large_train_test_AUC_delta"})

    log.info("Confirmed-leak candidates to drop:")
    for s in suspects:
        log.info("  %s  test=%.3f  train=%.3f  delta=%+.3f  reason=%s",
                 s["feature"], s["test_auc"], s["train_auc"], s["delta"], s["reason"])

    confirmed_leak_features = [s["feature"] for s in suspects]
    # Also drop the duration_days if it appears (failed trials terminated early ⇒ short duration ⇒ proxy)
    if "duration_days" in [s["feature"] for s in suspects]:
        pass
    else:
        # check duration explicitly
        dur_row = uni[uni["feature"] == "duration_days"]
        if len(dur_row) and dur_row["test_auc"].iloc[0] > 0.60:
            confirmed_leak_features.append("duration_days")
            confirmed_leak_features.append("duration_days_missing")
            log.info("  + duration_days (test_auc=%.3f) — terminated trials end early, "
                     "this is a label proxy", dur_row["test_auc"].iloc[0])

    # Also drop completion_date / start_date derivatives — they're not in the model
    # features list anyway, just confirm.
    log.info("Final dropped features for audit retrain: %s", confirmed_leak_features)

    # Retrain all 8 splits
    audit_table = []
    for tgt in ["p2_success", "p3_success"]:
        for st in ["time", "drugclass", "indication", "target"]:
            log.info("  retrain %s / %s ...", tgt, st)
            try:
                r = retrain_split(tgt, st, confirmed_leak_features)
                audit_table.append(r)
                log.info("    orig_AUROC=%.4f  audited_AUROC=%.4f  delta=%+.4f",
                         r["orig_metrics_calibrated"]["auroc"],
                         r["audited_metrics_calibrated"]["auroc"],
                         r["delta_auroc"])
            except Exception as e:
                log.exception("    FAILED: %s", e)
                audit_table.append({"target": tgt, "split_type": st, "error": str(e)})

    # === Null-label experiment: train on shuffled labels, eval on test ===
    log.info("Null-label experiment on p3/time (gold-standard leak test) ...")
    null_result = null_label_experiment(target, split_type)
    log.info("  null_shuffled_AUROC=%.4f  (should be ~0.5; >0.55 = leakage)",
             null_result["null_shuffled_auroc"])

    # Final synthesis JSON
    out = {
        "focus_split": "p3_success / time",
        "top15_test_dev_from_half": top15.to_dict(orient="records"),
        "B3_chembl_null_indicator": b3.to_dict(orient="records"),
        "B4_kinase_overlap": b4,
        "B5_oopd_edge_case": b5,
        "B6_leave_one_drug_out": b6,
        "confirmed_leak_features": confirmed_leak_features,
        "audit_retrain_table": audit_table,
        "null_label_experiment_p3_time": null_result,
    }
    with open(OUT_DIR / "leakage_audit.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info("Wrote leakage_audit.json")

    # ===== Verdict =====
    p3_time_orig = next((r for r in audit_table if r.get("target") == "p3_success"
                         and r.get("split_type") == "time"), None)
    if p3_time_orig and "audited_metrics_calibrated" in p3_time_orig:
        orig = p3_time_orig.get("orig_metrics_calibrated", p3_time_orig["orig_metrics_replayed"])["auroc"]
        audited = p3_time_orig["audited_metrics_calibrated"]["auroc"]
        if audited > 0.85:
            verdict = "HEADLINE_REAL"
        elif 0.65 <= audited <= 0.85:
            verdict = "HEADLINE_INFLATED_AUDITED_IN_BAND"
        elif audited < 0.65:
            verdict = "SUBSTANTIAL_LEAKAGE_FOUND"
        else:
            verdict = "UNCLEAR"
        log.info("=" * 60)
        log.info("VERDICT: p3/time original=%.4f  audited=%.4f  -> %s", orig, audited, verdict)
        log.info("=" * 60)
        out["verdict"] = {"orig_auroc": orig, "audited_auroc": audited, "label": verdict}
        with open(OUT_DIR / "leakage_audit.json", "w") as f:
            json.dump(out, f, indent=2, default=str)

    return out


if __name__ == "__main__":
    main()
