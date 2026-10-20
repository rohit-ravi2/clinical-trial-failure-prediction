"""
Day-3 Phase C4: Refit the fusion model with REAL embeddings from Phase C1-C3.

Replaces the Day-2 classical-fallback SMILES/protein/text embeddings with
ChemBERTa-768 / ESM2-640 / SciBERT-768. Tabular and historical-retrieval
modalities are unchanged from Day-2.

Embedding lookup strategy:
  - smiles_embeddings.npz keyed by chembl_id    → for each trial, use its
    primary chembl_id (from features_v2); missing => zero vector.
  - protein_embeddings.npz keyed by uniprot_id  → for each trial, use the
    primary target's uniprot_id (resolved from target_chembl_id via the
    target_sequences mapping); missing => zero vector.
  - text_embeddings.npz keyed by nct_id         → direct lookup; missing => zero vector.

We also produce 4 fusion variants:
  A. Tab-only (LightGBM on tabular features only — should reproduce Day-2 lgbm_v2)
  B. Tab + Hist (Day-2-equivalent fusion sans transformer modalities)
  C. Tab + Hist + SMILES + Protein + Text (full real-fusion)
  D. C above, plus the AUDITED tabular set (with duration_days etc. dropped, if any)

For each variant, run on all 8 task/split combos. Save metrics.

Outputs:
  work/processed/day3/real_embeddings_per_nct/*.npz
  work/results/day3/real_fusion_summary.json
"""
from __future__ import annotations
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("work/logs/day3_phaseC4.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("phaseC4")

ROOT = Path("work")
DUCKDB = ROOT / "aact.duckdb"
PARQUET_V2 = ROOT / "processed/day2/features_v2.parquet"
SPLITS_DIR = ROOT / "processed/splits"
EMB_DIR = ROOT / "processed/day3"
RESULTS_DIR = ROOT / "results/day3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_npz_dict(path: Path) -> tuple[dict[str, np.ndarray], int]:
    """Return (id->vector dict, dim)."""
    z = np.load(path, allow_pickle=False)
    ids = z["ids"].astype(str)
    emb = z["emb"]
    return {ids[i]: emb[i].astype(np.float32) for i in range(len(ids))}, emb.shape[1]


def build_per_nct_embeddings(task: str) -> dict[str, np.ndarray]:
    """For the given task's cohort, build per-NCT embedding matrices for
    SMILES, Protein, Text (real) and Tab, Hist (recomputed Day-2 style).

    Returns dict with keys: smiles, protein, text, tab, hist, ids
    """
    cohort_path = SPLITS_DIR / f"cohort_{task}_nct_ids.parquet"
    cohort_ncts = pd.read_parquet(cohort_path)["nct_id"].tolist()
    fv2 = pd.read_parquet(PARQUET_V2)
    fv2_c = fv2.set_index("nct_id").loc[cohort_ncts].reset_index()
    log.info("task=%s cohort=%d", task, len(cohort_ncts))

    # ---- load real embeddings ----
    sm_dict, dim_sm = load_npz_dict(EMB_DIR / "smiles_embeddings.npz")
    log.info("  smiles_embeddings: %d compounds, dim=%d", len(sm_dict), dim_sm)
    pr_dict, dim_pr = load_npz_dict(EMB_DIR / "protein_embeddings.npz")
    log.info("  protein_embeddings: %d targets, dim=%d", len(pr_dict), dim_pr)
    tx_dict, dim_tx = load_npz_dict(EMB_DIR / "text_embeddings.npz")
    log.info("  text_embeddings: %d trials, dim=%d", len(tx_dict), dim_tx)

    # SMILES: per-nct via chembl_id
    chembl_ids = fv2_c["chembl_id"].astype(object).where(fv2_c["chembl_id"].notna(), None).tolist()
    Z_sm = np.zeros((len(fv2_c), dim_sm), dtype=np.float32)
    sm_missing = np.ones(len(fv2_c), dtype=np.float32)
    for i, cid in enumerate(chembl_ids):
        if cid is not None and cid in sm_dict:
            Z_sm[i] = sm_dict[cid]
            sm_missing[i] = 0.0
    log.info("  SMILES per-NCT coverage: %d/%d (%.1f%%)",
             int((1 - sm_missing).sum()), len(fv2_c), 100 * (1 - sm_missing).mean())

    # Protein: per-NCT via target_chembl_id -> uniprot via target_sequences cache
    seq_df = pd.read_parquet(EMB_DIR / "cohort_target_sequences.parquet")
    # target_chembl_id may map to multiple uniprots — keep first
    tgt_to_uniprot = (seq_df.drop_duplicates(subset=["target_chembl_id"])
                              .set_index("target_chembl_id")["uniprot_id"].to_dict())
    Z_pr = np.zeros((len(fv2_c), dim_pr), dtype=np.float32)
    pr_missing = np.ones(len(fv2_c), dtype=np.float32)
    tcids = fv2_c["target_chembl_id"].astype(object).where(fv2_c["target_chembl_id"].notna(), None).tolist()
    for i, tcid in enumerate(tcids):
        if tcid is not None and tcid in tgt_to_uniprot:
            uid = tgt_to_uniprot[tcid]
            if uid in pr_dict:
                Z_pr[i] = pr_dict[uid]
                pr_missing[i] = 0.0
    log.info("  Protein per-NCT coverage: %d/%d (%.1f%%)",
             int((1 - pr_missing).sum()), len(fv2_c), 100 * (1 - pr_missing).mean())

    # Text: direct nct_id lookup
    Z_tx = np.zeros((len(fv2_c), dim_tx), dtype=np.float32)
    tx_missing = np.ones(len(fv2_c), dtype=np.float32)
    for i, nct in enumerate(fv2_c["nct_id"].tolist()):
        if nct in tx_dict:
            Z_tx[i] = tx_dict[nct]
            tx_missing[i] = 0.0
    log.info("  Text per-NCT coverage: %d/%d (%.1f%%)",
             int((1 - tx_missing).sum()), len(fv2_c), 100 * (1 - tx_missing).mean())

    # Tab: same 108 features Day-2 model used; we'll match by re-extracting from fv2_c
    drop_cols = {"nct_id", "registration_date", "start_date", "completion_date",
                 "overall_status", "p2_success", "p3_success", "label_available",
                 "primary_drug_canonical", "fda_match",
                 "chembl_resolved", "drug_has_orphan_ever",
                 "chembl_id", "target_chembl_id", "target_pref_name",
                 "mechanism_of_action", "molecule_type", "why_stopped",
                 "matched_intervention_count", "has_exact", "has_fuzzy",
                 "allocation", "intervention_model", "primary_purpose", "masking",
                 "lead_sponsor_class", "label_available_reason"}
    feat_cols = []
    for c in fv2_c.columns:
        if c in drop_cols:
            continue
        if pd.api.types.is_numeric_dtype(fv2_c[c]) or fv2_c[c].dtype.name == "bool":
            feat_cols.append(c)
    Z_tab = fv2_c[feat_cols].copy()
    for c in Z_tab.columns:
        if pd.api.types.is_extension_array_dtype(Z_tab[c]) or str(Z_tab[c].dtype) == "bool":
            Z_tab[c] = Z_tab[c].astype(float)
    Z_tab = Z_tab.astype(np.float32).fillna(0).values
    log.info("  Tab shape: %s", Z_tab.shape)

    # Hist: 2-feature historical retrieval — same as Day-2
    dt_cols = [c for c in fv2_c.columns if c.startswith("drug_type_")]
    ta_cols = [c for c in fv2_c.columns if c.startswith("therapeutic_area_")]
    fv2_c["dt_bucket"] = fv2_c[dt_cols].values.argmax(axis=1) if dt_cols else 0
    fv2_c["ta_bucket"] = fv2_c[ta_cols].values.argmax(axis=1) if ta_cols else 0
    fv2_c["ph_bucket"] = fv2_c["phase_ord"].fillna(0).round(0).astype(int)
    Z_hist = np.zeros((len(fv2_c), 2), dtype=np.float32)
    sub = fv2_c[fv2_c["p3_success"].notna()].copy()
    sub["p3_success"] = sub["p3_success"].astype(float)
    g = sub.groupby(["dt_bucket", "ta_bucket", "ph_bucket"])
    rate_map = {k: (gg["p3_success"].mean(), len(gg)) for k, gg in g}
    for i, row in enumerate(fv2_c.itertuples(index=False)):
        key = (row.dt_bucket, row.ta_bucket, row.ph_bucket)
        rate, n = rate_map.get(key, (np.nan, 0))
        Z_hist[i, 0] = rate if not np.isnan(rate) else 0.0
        Z_hist[i, 1] = n

    return {
        "ids": np.array(fv2_c["nct_id"].tolist()),
        "smiles": Z_sm,
        "smiles_missing": sm_missing.reshape(-1, 1),
        "protein": Z_pr,
        "protein_missing": pr_missing.reshape(-1, 1),
        "text": Z_tx,
        "text_missing": tx_missing.reshape(-1, 1),
        "tab": Z_tab,
        "hist": Z_hist,
        "feat_cols": feat_cols,
    }


def ece(y_true, p, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1])
        if i == n_bins - 1:
            m |= (p == 1.0)
        if m.sum() > 0:
            out += abs(p[m].mean() - y_true[m].mean()) * (m.sum() / len(p))
    return float(out)


def fuse_and_train(emb: dict, task: str, split: str, variant: str) -> dict:
    """variant in {tab_only, tab_hist, full_real}"""
    cohort_ncts = pd.read_parquet(SPLITS_DIR / f"cohort_{task}_nct_ids.parquet")["nct_id"].tolist()
    target = f"{task}_success"
    fv2 = pd.read_parquet(PARQUET_V2)
    fv2_c = fv2.set_index("nct_id").loc[cohort_ncts].reset_index()
    y_full = fv2_c[target].astype(float).values

    pieces = [emb["tab"]]
    if variant == "tab_only":
        pass
    elif variant == "tab_hist":
        pieces.append(emb["hist"])
    elif variant == "full_real":
        pieces.extend([emb["hist"], emb["smiles"], emb["smiles_missing"],
                       emb["protein"], emb["protein_missing"],
                       emb["text"], emb["text_missing"]])
    else:
        raise ValueError(f"unknown variant: {variant}")
    X = np.concatenate(pieces, axis=1).astype(np.float32)

    spl = pickle.loads((SPLITS_DIR / f"split_{task}_{split}.pkl").read_bytes())
    tr_i, te_i = np.asarray(spl["train_idx"]), np.asarray(spl["test_idx"])
    X_tr, X_te = X[tr_i], X[te_i]
    y_tr, y_te = y_full[tr_i], y_full[te_i]
    if X_tr.shape[0] < 100 or X_te.shape[0] < 50:
        return {"error": "too few rows"}

    params = {"objective": "binary", "metric": "auc", "verbosity": -1,
              "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 30,
              "feature_fraction": 0.5, "bagging_fraction": 0.7, "bagging_freq": 5,
              "num_threads": 4}
    rng = np.random.default_rng(0)
    idx = np.arange(len(X_tr))
    rng.shuffle(idx)
    n_cal = max(500, int(0.15 * len(idx)))
    ci, fi = idx[:n_cal], idx[n_cal:]
    m = lgb.train(params, lgb.Dataset(X_tr[fi], label=y_tr[fi]),
                  num_boost_round=800,
                  valid_sets=[lgb.Dataset(X_tr[ci], label=y_tr[ci])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    p_te = m.predict(X_te)
    p_cal = m.predict(X_tr[ci])
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal, y_tr[ci])
    p_te_cal = iso.transform(p_te)

    return {
        "task": task, "split": split, "variant": variant,
        "n_train": int(len(tr_i)), "n_test": int(len(te_i)),
        "pos_test": float(y_te.mean()),
        "fused_dim": int(X.shape[1]),
        "AUROC": float(roc_auc_score(y_te, p_te)),
        "PRAUC": float(average_precision_score(y_te, p_te)),
        "Brier_u": float(brier_score_loss(y_te, p_te)),
        "Brier_c": float(brier_score_loss(y_te, p_te_cal)),
        "ECE_u": ece(y_te, p_te),
        "ECE_c": ece(y_te, p_te_cal),
        "best_iter": int(m.best_iteration or m.current_iteration()),
    }


def main():
    t0 = time.time()
    out = {"summaries": {}, "transformers_used": True}
    for task in ["p2", "p3"]:
        emb = build_per_nct_embeddings(task)
        # cache per-nct emb to disk for future use
        cache_path = EMB_DIR / f"per_nct_emb_{task}.npz"
        np.savez_compressed(cache_path,
                            ids=emb["ids"],
                            smiles=emb["smiles"], smiles_missing=emb["smiles_missing"].squeeze(),
                            protein=emb["protein"], protein_missing=emb["protein_missing"].squeeze(),
                            text=emb["text"], text_missing=emb["text_missing"].squeeze(),
                            tab=emb["tab"], hist=emb["hist"])
        log.info("cached per-nct embeddings → %s", cache_path)
        for split in ["time", "drugclass", "indication", "target"]:
            for variant in ["tab_only", "tab_hist", "full_real"]:
                key = f"{task}/{split}/{variant}"
                try:
                    r = fuse_and_train(emb, task, split, variant)
                    out["summaries"][key] = r
                    log.info("  %s AUROC=%.4f", key, r.get("AUROC", float("nan")))
                except Exception as e:
                    log.exception("%s failed", key)
                    out["summaries"][key] = {"error": str(e)}
        (RESULTS_DIR / "real_fusion_summary.json").write_text(
            json.dumps(out, indent=2, default=str))

    out["_elapsed_s"] = time.time() - t0
    (RESULTS_DIR / "real_fusion_summary.json").write_text(
        json.dumps(out, indent=2, default=str))
    log.info("DONE in %.0fs", out["_elapsed_s"])

    # Print summary
    print("\n=== Phase C4 Real-Fusion Summary ===")
    print(f"{'task/split':<18} {'variant':<12} {'AUROC':>7} {'PRAUC':>7} {'Brier_c':>8} {'ECE_c':>7}")
    for k, r in out["summaries"].items():
        if "error" in r:
            print(f"{k:<31} ERROR: {r['error']}")
            continue
        parts = k.split("/")
        ts = f"{parts[0]}/{parts[1]}"
        print(f"{ts:<18} {parts[2]:<12} {r['AUROC']:>7.4f} {r['PRAUC']:>7.4f} {r['Brier_c']:>8.4f} {r['ECE_c']:>7.4f}")


if __name__ == "__main__":
    main()
