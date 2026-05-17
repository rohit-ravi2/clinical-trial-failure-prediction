"""Phase D — per-drug framing.

Day 1 is per-trial: 33,526 P3 rows, 11.78% base rate.
Per-drug framing: one row per (chembl_id, therapeutic_area, phase). Label =
whether ANY trial of that compound × indication × phase pair led to FDA
approval. Expected base rate 30-40% (matches Hay 2014 per-drug numbers).

Features at per-drug level are aggregated from the per-trial features:
  - n_trials_this_compound (count)
  - max/sum enrollment
  - latest design (most recent trial)
  - earliest registration
  - indication area, drug type, MoA (modal)
  - chembl_resolved (max)

Output:
- DuckDB table `trials_perdrug_labeled`
- DuckDB table `features_perdrug` (numeric features)
- 4 splits at processed/day2/splits_perdrug/
- LightGBM baseline at models/day2/lgbm_perdrug_<task>_<split>.pkl
- Summary at results/day2/perdrug_summary.json
"""
from __future__ import annotations
import json, logging, pickle, sys, time
from pathlib import Path
from datetime import datetime, timezone

import duckdb
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path("/mnt/ssd/research/trial-failure")
DUCKDB = ROOT / "aact.duckdb"
PARQUET_V2 = ROOT / "processed/day2/features_v2.parquet"
OUTDIR = ROOT / "processed/day2"
SPLITS_DIR = OUTDIR / "splits_perdrug"; SPLITS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = ROOT / "models/day2"; MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = ROOT / "results/day2"; RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG = ROOT / "logs/phaseD.log"

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler(sys.stdout)])
log = logging.getLogger("phaseD")

def ece(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i+1])
        if i == n_bins - 1:
            m |= (p == 1.0)
        if m.sum() > 0:
            out += abs(p[m].mean() - y_true[m].mean()) * (m.sum() / len(p))
    return float(out)

def build_perdrug(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Aggregate trial-level data to per (chembl_id, therapeutic_area, phase) row."""
    log.info("loading features_v2 + labels")
    df = pd.read_parquet(PARQUET_V2)
    # need raw therapeutic_area (one-hot already done). Recompute from base
    # by inverting one-hots:
    ta_cols = [c for c in df.columns if c.startswith("therapeutic_area_")]
    df["therapeutic_area"] = df[ta_cols].idxmax(axis=1).str.replace("therapeutic_area_", "")
    dt_cols = [c for c in df.columns if c.startswith("drug_type_")]
    df["drug_type"] = df[dt_cols].idxmax(axis=1).str.replace("drug_type_", "")
    moa_cols = [c for c in df.columns if c.startswith("moa_bucket_")]
    df["moa_bucket"] = df[moa_cols].idxmax(axis=1).str.replace("moa_bucket_", "")
    # filter to label-available trials with a chembl_id resolved
    sub = df[df["label_available"].astype(bool) & (df["chembl_resolved"] == 1)].copy()
    log.info(f"  label-available chembl-resolved trials: {len(sub)}")
    # decode phase from phase_ord
    sub["phase_str"] = sub["phase_ord"].map({1.0: "P1", 2.0: "P2", 3.0: "P3", 4.0: "P4",
                                              0.5: "EP1", 1.5: "P12", 2.5: "P23"}).fillna("UNK")
    # tasks: per-drug P3 success = did ANY trial of this compound for this indication get the compound FDA-approved?
    # Use p3_success aggregated by chembl_id × therapeutic_area; we approximate by:
    #   "compound × indication × phase → max(p3_success)"
    grp = sub.groupby(["chembl_id", "therapeutic_area", "phase_str"], as_index=False).agg(
        n_trials=("nct_id", "count"),
        any_p2_success=("p2_success", lambda x: int(x.astype(float).max() == 1.0) if x.notna().any() else np.nan),
        any_p3_success=("p3_success", lambda x: int(x.astype(float).max() == 1.0) if x.notna().any() else np.nan),
        enrollment_max=("enrollment", "max"),
        enrollment_log1p_mean=("enrollment_log1p", "mean"),
        duration_days_mean=("duration_days", "mean"),
        is_randomized_max=("is_randomized", "max"),
        any_blinding_max=("any_blinding", "max"),
        has_placebo_max=("has_placebo", "max"),
        is_orphan_at_registration_max=("is_orphan_at_registration", "max"),
        num_prior_trials_same_target_max=("num_prior_trials_same_target", "max"),
        target_novelty_min=("target_novelty", "min"),
        registration_date_first=("registration_date", "min"),
        registration_date_last=("registration_date", "max"),
        drug_type_mode=("drug_type", lambda x: x.mode().iat[0] if len(x.mode()) else "other"),
        moa_bucket_mode=("moa_bucket", lambda x: x.mode().iat[0] if len(x.mode()) else "unknown"),
    )
    log.info(f"  per-drug rows: {len(grp)}")
    # one-hot encode drug_type_mode, moa_bucket_mode, therapeutic_area, phase_str
    for c in ["drug_type_mode", "moa_bucket_mode", "therapeutic_area", "phase_str"]:
        top = grp[c].value_counts().head(30).index.tolist()
        grp[c] = grp[c].where(grp[c].isin(top), "_other")
    cat_cols = ["drug_type_mode", "moa_bucket_mode", "therapeutic_area", "phase_str"]
    dummies = pd.get_dummies(grp[cat_cols], drop_first=False, dtype=np.int8)
    grp = pd.concat([grp.drop(columns=cat_cols), dummies], axis=1)
    return grp

def split_perdrug(grp: pd.DataFrame, task: str) -> dict[str, dict]:
    """Build 4 splits at per-drug level. registration_date_first is the
    "first-trial-registration" anchor for time split."""
    splits = {}
    # filter rows by phase relevance for task
    if task == "p3":
        ph = grp[(grp.get("phase_str_P3", 0) == 1) | (grp.get("phase_str_P23", 0) == 1)]
    else:
        ph = grp[(grp.get("phase_str_P2", 0) == 1) | (grp.get("phase_str_P12", 0) == 1) | (grp.get("phase_str_P23", 0) == 1)]
    yl = ph[f"any_{task}_success"]
    ph = ph[yl.notna()].copy()
    ph[f"any_{task}_success"] = ph[f"any_{task}_success"].astype(int)
    ph = ph.reset_index(drop=True)
    log.info(f"  {task} cohort: {len(ph)} rows, base rate={ph[f'any_{task}_success'].mean():.3f}")

    # time split: first registration < 2018
    rd = pd.to_datetime(ph["registration_date_first"], errors="coerce")
    is_train_t = rd < pd.Timestamp("2018-01-01")
    splits["time"] = {
        "train_idx": ph.index[is_train_t].values,
        "test_idx": ph.index[~is_train_t & rd.notna()].values,
        "rule": "registration_date_first < 2018-01-01",
    }
    # drug class split: hold out a top class - use drug_type_mode_small_molecule == 0 (biologics or other)
    if "drug_type_mode_biologic" in ph.columns:
        is_test_dc = ph["drug_type_mode_biologic"] == 1
    else:
        is_test_dc = np.zeros(len(ph), dtype=bool)
    splits["drugclass"] = {
        "train_idx": ph.index[~is_test_dc].values,
        "test_idx": ph.index[is_test_dc].values,
        "rule": "biologic held out",
    }
    # indication split: hold out oncology
    if "therapeutic_area_oncology" in ph.columns:
        is_test_in = ph["therapeutic_area_oncology"] == 1
    else:
        is_test_in = np.zeros(len(ph), dtype=bool)
    splits["indication"] = {
        "train_idx": ph.index[~is_test_in].values,
        "test_idx": ph.index[is_test_in].values,
        "rule": "oncology held out",
    }
    # target split: hold out moa_bucket_agonist (proxy for GPCR-likes)
    if "moa_bucket_mode_agonist" in ph.columns:
        is_test_tg = ph["moa_bucket_mode_agonist"] == 1
    else:
        is_test_tg = np.zeros(len(ph), dtype=bool)
    splits["target"] = {
        "train_idx": ph.index[~is_test_tg].values,
        "test_idx": ph.index[is_test_tg].values,
        "rule": "agonist held out",
    }
    return ph, splits

def train_perdrug(ph: pd.DataFrame, splits: dict, task: str, n_trials: int = 6) -> dict:
    target = f"any_{task}_success"
    drop_cols = {"chembl_id", "registration_date_first", "registration_date_last",
                 target, f"any_p2_success", f"any_p3_success"}
    feats = [c for c in ph.columns if c not in drop_cols and (
        pd.api.types.is_numeric_dtype(ph[c]) or str(ph[c].dtype) == "bool")]
    X = ph[feats].astype(float).fillna(0)
    y = ph[target].astype(int).values
    out_split = {}
    for name, spl in splits.items():
        tr_i, te_i = spl["train_idx"], spl["test_idx"]
        if len(tr_i) < 100 or len(te_i) < 20:
            out_split[name] = {"error": f"too few rows train={len(tr_i)} test={len(te_i)}"}
            continue
        X_tr, X_te = X.iloc[tr_i], X.iloc[te_i]
        y_tr, y_te = y[tr_i], y[te_i]
        # fixed defaults — fast
        best = {"objective": "binary", "metric": "auc", "verbosity": -1,
            "learning_rate": 0.05, "num_leaves": 31,
            "min_child_samples": 20,
            "feature_fraction": 0.85, "bagging_fraction": 0.85,
            "bagging_freq": 5, "num_threads": 4}
        # holdout-calibration slice
        rng = np.random.default_rng(0); idx = np.arange(len(X_tr)); rng.shuffle(idx)
        n_cal = max(200, int(0.2 * len(idx)))
        ci, fi = idx[:n_cal], idx[n_cal:]
        m = lgb.train(best, lgb.Dataset(X_tr.iloc[fi], label=y_tr[fi]),
                      num_boost_round=1500, valid_sets=[lgb.Dataset(X_tr.iloc[ci], label=y_tr[ci])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        p_cal = m.predict(X_tr.iloc[ci]); p_te = m.predict(X_te)
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal, y_tr[ci])
        p_te_cal = iso.transform(p_te)
        out_split[name] = {
            "n_train": int(len(tr_i)), "n_test": int(len(te_i)),
            "pos_train": float(y_tr.mean()), "pos_test": float(y_te.mean()),
            "AUROC": float(roc_auc_score(y_te, p_te)) if y_te.sum() else None,
            "PRAUC": float(average_precision_score(y_te, p_te)) if y_te.sum() else None,
            "Brier_u": float(brier_score_loss(y_te, p_te)),
            "Brier_c": float(brier_score_loss(y_te, p_te_cal)),
            "ECE_u": ece(y_te, p_te), "ECE_c": ece(y_te, p_te_cal),
            "best_iter": int(m.best_iteration or m.current_iteration()),
        }
        (MODELS_DIR / f"lgbm_perdrug_{task}_{name}.pkl").write_bytes(
            pickle.dumps({"model": m, "iso": iso, "features": list(X.columns)}))
        log.info(f"  {task}/{name}: AUROC={out_split[name]['AUROC']}, n_train={len(tr_i)} n_test={len(te_i)}")
    return out_split

def main():
    t0 = time.time()
    con = duckdb.connect(str(DUCKDB), read_only=False)
    grp = build_perdrug(con)
    # save
    con.execute("DROP TABLE IF EXISTS trials_perdrug_labeled")
    con.register("g", grp)
    # cast object cols to varchar
    con.execute("CREATE TABLE trials_perdrug_labeled AS SELECT * FROM g")
    log.info(f"saved trials_perdrug_labeled: {len(grp)} rows")
    grp.to_parquet(OUTDIR / "features_perdrug.parquet", index=False)
    out = {}
    for task in ["p2", "p3"]:
        ph, splits = split_perdrug(grp, task)
        pickle.dump(splits, open(SPLITS_DIR / f"splits_perdrug_{task}.pkl", "wb"))
        out[task] = {"n_cohort": len(ph), "base_rate": float(ph[f"any_{task}_success"].mean()),
                     "splits": train_perdrug(ph, splits, task, n_trials=20)}
    out["_elapsed_s"] = round(time.time() - t0, 1)
    (RESULTS_DIR / "perdrug_summary.json").write_text(json.dumps(out, indent=2, default=str))
    log.info(f"DONE in {out['_elapsed_s']}s")
    print("\n=== Phase D Per-Drug Summary ===")
    for task, v in out.items():
        if task.startswith("_"): continue
        print(f"\n{task}: cohort={v['n_cohort']}, base={v['base_rate']:.3f}")
        for sp, r in v["splits"].items():
            if "error" in r:
                print(f"  {sp}: {r['error']}")
            else:
                print(f"  {sp}: AUROC={r['AUROC']:.4f} n_test={r['n_test']} pos={r['pos_test']:.3f}")

if __name__ == "__main__":
    main()
