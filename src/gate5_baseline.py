"""
Gate 5 — LightGBM tabular baseline across the 8 frozen splits.

For each (target, split_type) pair this script:
  1. Loads the cohort, features (master parquet), and split pickle.
  2. Overwrites the placeholder *_fold columns with the per-split encodings
     stored in the split pickle (Gate 4 already computed these out-of-fold).
  3. Builds X / y for train and test using positional train_idx / test_idx.
  4. Holds out 20% of TRAIN as a calibration slice (stratified). Optuna's
     hyperparameter search uses the remaining 80% (= "tune slice") only.
  5. Runs Optuna (50 trials OR 10 min wall, whichever first) maximising
     mean stratified 5-fold ROC-AUC on the tune slice. LightGBM uses an
     internal early-stopping holdout from each CV fold.
  6. Refits the best params on the FULL tune slice (with a 10% internal
     early-stop holdout) -> 'tuned_model'.
  7. Evaluates 'tuned_model' on TEST: AUROC, PR-AUC, Brier (uncal), ECE10.
  8. Fits IsotonicRegression on the held-out calibration slice using
     tuned_model probabilities, then refits a final model on TRAIN (full)
     and applies isotonic calibration to TEST probabilities.
     Re-reports Brier and ECE10.
  9. Reliability + calibration PNGs, model pickle, predictions parquet.

Run:  python gate5_baseline.py [--targets p3,p2] [--splits time,...]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #
ROOT = Path("work")
FEATURE_PATH = ROOT / "processed" / "features_tabular.parquet"
SCHEMA_PATH = ROOT / "processed" / "features_tabular_schema.json"
SPLITS_DIR = ROOT / "processed" / "splits"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
GATE_LOG = LOGS_DIR / "gates.log"

for d in (RESULTS_DIR, MODELS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Logging                                                                      #
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.FileHandler(LOGS_DIR / "gate5.log", mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("gate5")

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #
TARGETS = ("p3", "p2")
SPLITS = ("time", "drugclass", "indication", "target")
RANDOM_SEED = 17
OPTUNA_TRIALS = 50
OPTUNA_TIMEOUT_S = 600  # 10 minutes
# Host is shared with other long-running jobs. Limit thread count so we do not
# storm the box (16 physical cores; load avg has been ~20 during this run).
LGBM_N_JOBS = 4
# Optuna CV folds. 3 is the literature norm for "lightweight" tuning and
# halves wall-clock vs 5-fold; we still get an unbiased AUROC estimate.
OPTUNA_CV_FOLDS = 3
# Per-fit boost-round cap inside Optuna. Early stopping (patience 50) almost
# always fires well before this; the cap just stops pathological trials.
OPTUNA_MAX_BOOST_ROUNDS = 1200
# Final refit boost-round cap (a little more headroom).
FINAL_MAX_BOOST_ROUNDS = 2000

# Columns we never want as features.
#
# CRITICAL leakage finding (documented during Gate 5 smoke runs):
# Gate 2/3 build drug-level descriptors (drug_type, target_type, action_type,
# moa_bucket, num_prior_trials_same_drug, target_novelty, has_orphan_app,
# is_rare_disease) by joining on `primary_drug_canonical`, which itself
# comes from the CT.gov<->FDA name-matching pipeline. Therefore for ~46%
# of the cohort where the FDA match failed, all these features collapse to
# constant "unknown"/0/missing values; for the remaining 54% they reflect
# information that is only knowable if the drug already has an FDA record
# (often = is/was approved). Univariate AUROCs of 0.24 / 0.68 on the
# *_is_missing indicators confirm direct leakage of the outcome.
#
# Together with `*_success_rate_fold` (same-drug aggregation across the
# bucket), excluding these features brings TEST AUROC from ~0.94 back into
# the literature-plausible 0.65-0.75 range.
#
# These columns MUST be reconstructed in Gate 3 v2 using only the raw
# CT.gov intervention name path (not the FDA-canonicalised name) before
# week 2 architecture work. For Gate 5 we simply drop them.

# Lineage tag: "label/id" — direct label or identifier
# Lineage tag: "string" — raw string column already one-hot encoded elsewhere
# Lineage tag: "fda_leak" — derived through the FDA-match pipeline
# Lineage tag: "fold_leak" — target-encoded with same-drug aggregation
EXCLUDE_ALWAYS = {
    # ---- label / id / metadata -------------------------------------------
    "nct_id",
    "p2_success",
    "p3_success",
    "label_available",
    "termination_reason_raw",
    "registration_date",
    # ---- naive whole-cohort encodings (forbidden by Gate 3 README) -------
    "indication_success_rate_naive",
    "drug_class_success_rate_naive",
    # ---- raw high-cardinality strings (one-hots already exist) -----------
    "phase",
    "primary_drug_canonical",
    "primary_endpoint_type",
    "primary_endpoint_types_all",
    "sponsor_type",
    "sponsor_agency_class",
    "drug_type",
    "mechanism_of_action",
    "moa_bucket",
    "target_type",
    "action_type",
    "therapeutic_area",
    # ---- FDA-match-derived (LEAKY: see header comment) -------------------
    "primary_drug_canonical_is_missing",
    "num_prior_trials_same_drug",
    "num_prior_trials_same_target",
    "target_novelty",
    "has_orphan_app",
    "is_rare_disease",
    # ---- target-encoded with same-drug aggregation (LEAKY) ---------------
    "indication_success_rate_fold",
    "drug_class_success_rate_fold",
    "indication_success_rate_fold_is_missing",
    "drug_class_success_rate_fold_is_missing",
}

# All one-hot families that go through primary_drug_canonical and therefore
# leak. We drop them by prefix.
EXCLUDE_PREFIXES = (
    "oh_drug_type_",
    "oh_moa_bucket_",
    "oh_target_type_",
    "oh_action_type_",
)

# Per-split holdout description for the log line
HOLDOUT_RULES = {
    "time": "registration_date >= 2018-01-01",
    "drugclass": "moa contains 'kinase'",
    "indication": "therapeutic_area == 'oncology'",
    "target": "GPCR family moa regex",
}


# --------------------------------------------------------------------------- #
# Result container                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class SplitResult:
    target: str
    split_type: str
    train_n: int
    test_n: int
    test_pos_rate: float
    train_pos_rate: float
    n_features: int
    best_params: dict[str, Any] = field(default_factory=dict)
    best_n_estimators: int = 0
    auroc: float = float("nan")
    pr_auc: float = float("nan")
    brier_uncal: float = float("nan")
    brier_cal: float = float("nan")
    ece_uncal: float = float("nan")
    ece_cal: float = float("nan")
    top_features: list[tuple[str, float]] = field(default_factory=list)
    optuna_trials_completed: int = 0
    elapsed_s: float = 0.0
    notes: str = ""


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def now_z() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Standard equal-width binned ECE with 10 bins."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(y_prob, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def reliability_data(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    bin_idx = np.digitize(y_prob, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    accs = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = bin_idx == b
        counts[b] = int(mask.sum())
        if mask.any():
            accs[b] = float(y_true[mask].mean())
    return centers, accs, counts


def save_reliability_png(path: Path, y_true, y_prob, title: str, n_bins: int = 10):
    centers, accs, counts = reliability_data(y_true, y_prob, n_bins)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    valid = ~np.isnan(accs)
    ax1.plot(centers[valid], accs[valid], "o-", lw=1.5, label="empirical")
    ax1.set_ylabel("Empirical positive rate")
    ax1.set_title(title)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)
    ax2.bar(centers, counts, width=1.0 / n_bins, alpha=0.7, edgecolor="k")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


def save_calibration_png(path: Path, y_true, y_uncal, y_cal, title: str, n_bins: int = 10):
    cu, au, _ = reliability_data(y_true, y_uncal, n_bins)
    cc, ac, _ = reliability_data(y_true, y_cal, n_bins)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    mu = ~np.isnan(au)
    mc = ~np.isnan(ac)
    ax.plot(cu[mu], au[mu], "o-", lw=1.5, label="uncalibrated")
    ax.plot(cc[mc], ac[mc], "s-", lw=1.5, label="isotonic")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Data assembly                                                                #
# --------------------------------------------------------------------------- #
def load_master_features() -> pd.DataFrame:
    df = pd.read_parquet(FEATURE_PATH)
    log.info("Loaded master features: %s (index=%s)", df.shape, df.index.name)
    return df


def build_split_frame(
    master: pd.DataFrame, target: str, split_type: str
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray, dict, list[str]]:
    """Return X (cohort-ordered), y, train_idx, test_idx, metadata, feature_cols."""
    cohort = pd.read_parquet(SPLITS_DIR / f"cohort_{target}_nct_ids.parquet")["nct_id"].tolist()
    with open(SPLITS_DIR / f"split_{target}_{split_type}.pkl", "rb") as f:
        split = pickle.load(f)
    train_idx = split["train_idx"]
    test_idx = split["test_idx"]
    target_col = split["target_col"]
    metadata = split["metadata"]

    # Cohort-aligned slice of master
    df = master.loc[cohort].copy()
    if len(df) != len(cohort):
        raise RuntimeError(f"cohort/master alignment mismatch for {target}/{split_type}")

    # Gate 4 stashed per-fold target encodings here, but they aggregate
    # outcome over same-drug clusters and so are FDA-pipeline-leaky in this
    # snapshot. They are dropped via EXCLUDE_ALWAYS; do NOT inject them.
    # (Kept the assignment commented in case Gate 3 v2 re-enables them.)
    # ind_series = split["indication_success_rate_fold"]
    # drug_series = split["drug_class_success_rate_fold"]
    # df["indication_success_rate_fold"] = ind_series.reindex(df.index).astype(float).values
    # df["drug_class_success_rate_fold"] = drug_series.reindex(df.index).astype(float).values

    # Build feature column list — keep numeric, boolean, one-hot.
    # Drop direct identifiers, labels, raw strings, and FDA-match-derived
    # features (see EXCLUDE_ALWAYS / EXCLUDE_PREFIXES header comment).
    feature_cols: list[str] = []
    for c in df.columns:
        if c in EXCLUDE_ALWAYS:
            continue
        if any(c.startswith(p) for p in EXCLUDE_PREFIXES):
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            feature_cols.append(c)
    feature_cols = [c for c in feature_cols if c != target_col]

    # Convert pandas nullable types to numpy (LightGBM doesn't love pd.NA)
    X = df[feature_cols].copy()
    # Cast pandas nullable to float (NaN-aware)
    for c in feature_cols:
        if X[c].dtype.name in ("Int64", "Float64", "boolean"):
            X[c] = X[c].astype("float32")
        elif X[c].dtype.name == "bool":
            X[c] = X[c].astype("float32")
        elif X[c].dtype.kind == "i":
            # int8 one-hots can stay
            X[c] = X[c].astype("float32")
        elif X[c].dtype.kind == "f":
            X[c] = X[c].astype("float32")
    y = df[target_col]
    if y.dtype.name in ("Int64", "boolean"):
        y = y.astype("float32")
    y = y.astype("float32")

    log.info(
        "[%s/%s] X=%s  y=%s  feature_cols=%d  train_n=%d test_n=%d",
        target,
        split_type,
        X.shape,
        y.shape,
        len(feature_cols),
        len(train_idx),
        len(test_idx),
    )

    # Validate: y must have no NaN for cohort (label_available filter happened upstream)
    if y.isna().any():
        raise RuntimeError(
            f"[{target}/{split_type}] target {target_col} has {int(y.isna().sum())} NaN inside cohort"
        )

    return X, y, np.asarray(train_idx), np.asarray(test_idx), metadata, feature_cols


# --------------------------------------------------------------------------- #
# Optuna objective                                                             #
# --------------------------------------------------------------------------- #
def make_objective(X_tune: np.ndarray, y_tune: np.ndarray, n_features: int):
    skf = StratifiedKFold(
        n_splits=OPTUNA_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 500, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 1,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            "n_jobs": LGBM_N_JOBS,
            "seed": RANDOM_SEED,
        }
        aucs: list[float] = []
        for fold_i, (tr, va) in enumerate(skf.split(X_tune, y_tune)):
            dtr = lgb.Dataset(X_tune[tr], label=y_tune[tr])
            dva = lgb.Dataset(X_tune[va], label=y_tune[va], reference=dtr)
            booster = lgb.train(
                params,
                dtr,
                num_boost_round=OPTUNA_MAX_BOOST_ROUNDS,
                valid_sets=[dva],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )
            pred = booster.predict(X_tune[va], num_iteration=booster.best_iteration)
            try:
                auc = roc_auc_score(y_tune[va], pred)
            except ValueError:
                auc = 0.5
            aucs.append(auc)
        return float(np.mean(aucs))

    return objective


# --------------------------------------------------------------------------- #
# Fit + evaluate one split                                                     #
# --------------------------------------------------------------------------- #
def run_split(
    master: pd.DataFrame,
    target: str,
    split_type: str,
) -> SplitResult:
    t0 = time.time()
    X_df, y_ser, train_idx, test_idx, metadata, feature_cols = build_split_frame(
        master, target, split_type
    )

    # Numpy materialisation (lighter on memory than passing pandas through lgb)
    X = X_df.values.astype("float32", copy=False)
    y = y_ser.values.astype("float32", copy=False)
    cohort_nct = X_df.index.values

    X_train_full = X[train_idx]
    y_train_full = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    nct_test = cohort_nct[test_idx]

    train_n = int(len(train_idx))
    test_n = int(len(test_idx))
    train_pos_rate = float(y_train_full.mean()) if train_n > 0 else float("nan")
    test_pos_rate = float(y_test.mean()) if test_n > 0 else float("nan")

    result = SplitResult(
        target=target,
        split_type=split_type,
        train_n=train_n,
        test_n=test_n,
        train_pos_rate=train_pos_rate,
        test_pos_rate=test_pos_rate,
        n_features=len(feature_cols),
    )

    # --------------------------------------------------------------------- #
    # Calibration slice — 20% of train, stratified                          #
    # --------------------------------------------------------------------- #
    # If a class is too rare to stratify, fall back to a simple random slice.
    pos_count = int(y_train_full.sum())
    if pos_count >= 10 and (train_n - pos_count) >= 10:
        X_tune, X_calib, y_tune, y_calib = train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.20,
            random_state=RANDOM_SEED,
            stratify=y_train_full,
        )
    else:
        X_tune, X_calib, y_tune, y_calib = train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.20,
            random_state=RANDOM_SEED,
        )
    log.info(
        "[%s/%s] tune_slice=%d  calib_slice=%d  test=%d",
        target,
        split_type,
        len(X_tune),
        len(X_calib),
        test_n,
    )

    # --------------------------------------------------------------------- #
    # Optuna                                                                #
    # --------------------------------------------------------------------- #
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    obj = make_objective(X_tune, y_tune, len(feature_cols))
    log.info(
        "[%s/%s] starting Optuna (max %d trials / %ds, cv=%d folds, lgbm_jobs=%d)",
        target,
        split_type,
        OPTUNA_TRIALS,
        OPTUNA_TIMEOUT_S,
        OPTUNA_CV_FOLDS,
        LGBM_N_JOBS,
    )

    _trial_log_state = {"t0": time.time()}

    def _trial_cb(study_obj, trial):
        n = len(study_obj.trials)
        if n % 5 == 0 or n == 1:
            elapsed = time.time() - _trial_log_state["t0"]
            best = study_obj.best_value if study_obj.best_trial else float("nan")
            log.info(
                "[%s/%s] Optuna trial %d/%d  best_cv_auc=%.4f  elapsed=%.0fs",
                target,
                split_type,
                n,
                OPTUNA_TRIALS,
                best,
                elapsed,
            )

    study.optimize(
        obj,
        n_trials=OPTUNA_TRIALS,
        timeout=OPTUNA_TIMEOUT_S,
        gc_after_trial=True,
        show_progress_bar=False,
        n_jobs=1,  # parallelism is inside LightGBM only
        callbacks=[_trial_cb],
    )
    best_params = study.best_params.copy()
    result.optuna_trials_completed = len(study.trials)
    log.info(
        "[%s/%s] Optuna done: trials=%d best_cv_auc=%.4f params=%s",
        target,
        split_type,
        len(study.trials),
        study.best_value,
        best_params,
    )

    # --------------------------------------------------------------------- #
    # Tuned model — early-stopping holdout taken from the tune slice         #
    # --------------------------------------------------------------------- #
    Xt_fit, Xt_es, yt_fit, yt_es = train_test_split(
        X_tune,
        y_tune,
        test_size=0.10,
        random_state=RANDOM_SEED,
        stratify=y_tune if y_tune.sum() >= 5 else None,
    )
    tuned_params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "bagging_freq": 1,
        "n_jobs": LGBM_N_JOBS,
        "seed": RANDOM_SEED,
        **best_params,
    }
    tuned_booster = lgb.train(
        tuned_params,
        lgb.Dataset(Xt_fit, label=yt_fit),
        num_boost_round=FINAL_MAX_BOOST_ROUNDS,
        valid_sets=[lgb.Dataset(Xt_es, label=yt_es)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    best_iter_tuned = int(tuned_booster.best_iteration or tuned_booster.current_iteration())

    # --------------------------------------------------------------------- #
    # Uncalibrated TEST evaluation                                          #
    # --------------------------------------------------------------------- #
    p_test_uncal = tuned_booster.predict(X_test, num_iteration=best_iter_tuned)
    p_test_uncal = np.clip(p_test_uncal, 1e-6, 1 - 1e-6)
    try:
        result.auroc = float(roc_auc_score(y_test, p_test_uncal))
        result.pr_auc = float(average_precision_score(y_test, p_test_uncal))
    except ValueError:
        result.auroc = float("nan")
        result.pr_auc = float("nan")
    result.brier_uncal = float(brier_score_loss(y_test, p_test_uncal))
    result.ece_uncal = expected_calibration_error(y_test, p_test_uncal, n_bins=10)

    save_reliability_png(
        RESULTS_DIR / f"reliability_{target}_{split_type}.png",
        y_test,
        p_test_uncal,
        f"Reliability  target={target}  split={split_type}  AUROC={result.auroc:.3f}",
    )

    # --------------------------------------------------------------------- #
    # Isotonic calibration                                                  #
    # --------------------------------------------------------------------- #
    p_calib = tuned_booster.predict(X_calib, num_iteration=best_iter_tuned)
    p_calib = np.clip(p_calib, 1e-6, 1 - 1e-6)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_calib, y_calib)
    p_test_cal = iso.transform(p_test_uncal)
    p_test_cal = np.clip(p_test_cal, 1e-6, 1 - 1e-6)
    result.brier_cal = float(brier_score_loss(y_test, p_test_cal))
    result.ece_cal = expected_calibration_error(y_test, p_test_cal, n_bins=10)

    save_calibration_png(
        RESULTS_DIR / f"calibration_{target}_{split_type}.png",
        y_test,
        p_test_uncal,
        p_test_cal,
        f"Calibration  target={target}  split={split_type}  "
        f"ECE {result.ece_uncal:.3f} -> {result.ece_cal:.3f}",
    )

    # --------------------------------------------------------------------- #
    # Refit on full TRAIN for the saved deliverable                          #
    # --------------------------------------------------------------------- #
    Xf_fit, Xf_es, yf_fit, yf_es = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.10,
        random_state=RANDOM_SEED,
        stratify=y_train_full if y_train_full.sum() >= 5 else None,
    )
    final_booster = lgb.train(
        tuned_params,
        lgb.Dataset(Xf_fit, label=yf_fit),
        num_boost_round=FINAL_MAX_BOOST_ROUNDS,
        valid_sets=[lgb.Dataset(Xf_es, label=yf_es)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    result.best_n_estimators = int(final_booster.best_iteration or final_booster.current_iteration())

    # --------------------------------------------------------------------- #
    # Feature importance (gain) — top 10                                    #
    # --------------------------------------------------------------------- #
    gains = final_booster.feature_importance(importance_type="gain")
    pairs = sorted(zip(feature_cols, gains.tolist()), key=lambda kv: kv[1], reverse=True)
    result.top_features = pairs[:10]
    result.best_params = best_params

    # --------------------------------------------------------------------- #
    # Persist artefacts                                                     #
    # --------------------------------------------------------------------- #
    bundle = {
        "tuned_model": tuned_booster.model_to_string(),
        "final_model": final_booster.model_to_string(),
        "best_params": best_params,
        "best_iter_tuned": best_iter_tuned,
        "best_iter_final": result.best_n_estimators,
        "isotonic_calibrator": iso,
        "feature_columns": feature_cols,
        "target_col": metadata["target"],
        "split_type": split_type,
        "metadata": metadata,
        "calibration_choice": "20% stratified hold-out of TRAIN; isotonic fit on tuned_model probs",
        "constructed_at": now_z(),
    }
    with open(MODELS_DIR / f"lgbm_{target}_{split_type}.pkl", "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    preds = pd.DataFrame(
        {
            "nct_id": nct_test,
            "y_true": y_test.astype(int),
            "y_pred_uncal": p_test_uncal.astype("float32"),
            "y_pred_cal": p_test_cal.astype("float32"),
        }
    )
    preds.to_parquet(RESULTS_DIR / f"predictions_{target}_{split_type}.parquet", index=False)

    result.elapsed_s = time.time() - t0
    log.info(
        "[%s/%s] DONE  AUROC=%.4f  PR-AUC=%.4f  Brier %.4f->%.4f  ECE %.4f->%.4f  iters=%d  elapsed=%.1fs",
        target,
        split_type,
        result.auroc,
        result.pr_auc,
        result.brier_uncal,
        result.brier_cal,
        result.ece_uncal,
        result.ece_cal,
        result.best_n_estimators,
        result.elapsed_s,
    )
    return result


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--splits", default=",".join(SPLITS))
    args = parser.parse_args()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    log.info("Gate 5 start  targets=%s  splits=%s", targets, splits)
    master = load_master_features()

    all_results: list[SplitResult] = []
    notify_path = os.path.expanduser("~/bin/notify")

    for target in targets:
        for split_type in splits:
            log.info("===== %s / %s =====", target, split_type)
            try:
                r = run_split(master, target, split_type)
                all_results.append(r)
                if os.path.exists(notify_path):
                    os.system(
                        f'{notify_path} "trial-failure Gate5: {target}/{split_type} done '
                        f'AUROC={r.auroc:.3f} ECE {r.ece_uncal:.3f}->{r.ece_cal:.3f}" '
                        f'"trial-failure" > /dev/null 2>&1'
                    )
            except Exception as exc:
                log.exception("[%s/%s] FAILED: %s", target, split_type, exc)
                raise

    # ---------------- Summary ---------------------------------------------#
    summary_rows = []
    for r in all_results:
        summary_rows.append(
            {
                "target": r.target,
                "split_type": r.split_type,
                "train_n": r.train_n,
                "test_n": r.test_n,
                "test_pos_rate": round(r.test_pos_rate, 6),
                "n_features": r.n_features,
                "auroc": round(r.auroc, 6),
                "pr_auc": round(r.pr_auc, 6),
                "brier_uncal": round(r.brier_uncal, 6),
                "brier_cal": round(r.brier_cal, 6),
                "ece_uncal": round(r.ece_uncal, 6),
                "ece_cal": round(r.ece_cal, 6),
                "best_iter": r.best_n_estimators,
                "optuna_trials": r.optuna_trials_completed,
                "elapsed_s": round(r.elapsed_s, 1),
                "best_params": r.best_params,
                "top_features": r.top_features,
            }
        )

    summary_path = RESULTS_DIR / "gate5_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"constructed_at": now_z(), "results": summary_rows}, f, indent=2, default=str)
    log.info("Wrote %s", summary_path)

    # ---------------- Gate log line ---------------------------------------#
    # Sanity check the time/p3 split
    sanity_msgs = []
    for r in all_results:
        if r.target == "p3" and r.split_type == "time":
            if r.auroc < 0.55:
                sanity_msgs.append(f"P3/time AUROC={r.auroc:.3f} < 0.55 (label-error suspect)")
            elif r.auroc > 0.80:
                sanity_msgs.append(f"P3/time AUROC={r.auroc:.3f} > 0.80 (leakage suspect)")
    gate_status = "PASS" if not sanity_msgs else "PASS_WITH_FLAGS"

    parts = [f"[{now_z()}] GATE_5: {gate_status}"]
    if sanity_msgs:
        parts.append("SANITY=" + "|".join(sanity_msgs))
    for r in all_results:
        parts.append(
            f"{r.target}/{r.split_type}:AUROC={r.auroc:.4f},"
            f"PRAUC={r.pr_auc:.4f},"
            f"Brier_u={r.brier_uncal:.4f},Brier_c={r.brier_cal:.4f},"
            f"ECE_u={r.ece_uncal:.4f},ECE_c={r.ece_cal:.4f},"
            f"trials={r.optuna_trials_completed},iter={r.best_n_estimators}"
        )
    log_line = " ".join(parts)
    with open(GATE_LOG, "a") as f:
        f.write(log_line + "\n")
    log.info("Wrote gate log:\n%s", log_line)

    if os.path.exists(notify_path):
        os.system(
            f'{notify_path} "trial-failure Gate 5 {gate_status}: all 8 splits done" '
            f'"trial-failure" > /dev/null 2>&1'
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
