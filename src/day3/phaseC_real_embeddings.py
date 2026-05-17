"""
Day-3 Phase C: Real ChemBERTa + ESM2 + SciBERT embeddings, replacing the
Day-2 classical-fallback embeddings used in the fusion model.

Sub-phases:
  C1 - ChemBERTa SMILES embeddings (per chembl_id, 768-dim CLS)
  C2 - ESM2-150M protein target embeddings (per uniprot_id, 640-dim mean-pool)
  C3 - SciBERT protocol text embeddings (per nct_id, 768-dim CLS)
  C4 - Refit fusion model with real embeddings (deferred to a separate driver)

Outputs:
  /mnt/ssd/research/trial-failure/processed/day3/smiles_embeddings.npz
  /mnt/ssd/research/trial-failure/processed/day3/protein_embeddings.npz
  /mnt/ssd/research/trial-failure/processed/day3/text_embeddings.npz

GPU: 4060 Ti / 8GB. ChemBERTa-zinc-base-v1 is 87M params (≈350MB fp32).
ESM2-150M is ~590MB. SciBERT is ~440MB. Each fits comfortably; embeddings
fp16 to halve memory.
"""
from __future__ import annotations
import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/mnt/ssd/research/trial-failure/logs/day3_phaseC.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("phaseC")

ROOT = Path("/mnt/ssd/research/trial-failure")
OUT_DIR = ROOT / "processed/day3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log.info("Device: %s", DEVICE)


def embed_smiles_chemberta(smiles_df: pd.DataFrame, batch_size: int = 32) -> dict[str, np.ndarray]:
    """Returns dict chembl_id -> 768-dim CLS embedding (fp16)."""
    from transformers import AutoTokenizer, AutoModel
    MODEL = "seyonec/ChemBERTa-zinc-base-v1"
    log.info("Loading %s ...", MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float16 if DEVICE == "cuda" else torch.float32)
    model.to(DEVICE).eval()
    log.info("ChemBERTa loaded. n compounds: %d", len(smiles_df))

    out: dict[str, np.ndarray] = {}
    ids = smiles_df["chembl_id"].tolist()
    smiles = smiles_df["canonical_smiles"].tolist()

    with torch.no_grad():
        for i in tqdm(range(0, len(ids), batch_size), desc="ChemBERTa"):
            chunk_ids = ids[i:i + batch_size]
            chunk_smiles = smiles[i:i + batch_size]
            enc = tok(chunk_smiles, padding=True, truncation=True,
                      max_length=256, return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            out_h = model(**enc).last_hidden_state[:, 0, :]
            out_h = out_h.float().cpu().numpy().astype(np.float16)
            for j, cid in enumerate(chunk_ids):
                out[cid] = out_h[j]
    return out


def get_uniprot_sequences() -> dict[str, str]:
    """Pull unique target uniprot_ids and their sequences from ChEMBL.

    component_sequences holds the protein sequence; target_components links
    target_chembl_id <-> molregno <-> component_id.
    """
    feats = pd.read_parquet(ROOT / "processed/day2/features_v2.parquet")
    tids = feats["target_chembl_id"].dropna().unique().tolist()
    log.info("unique target_chembl_ids in cohort: %d", len(tids))

    con = sqlite3.connect(ROOT / "raw/chembl/chembl_36/chembl_36_sqlite/chembl_36.db")
    # Check schema
    cols_td = pd.read_sql_query("PRAGMA table_info(target_dictionary)", con)
    cols_tc = pd.read_sql_query("PRAGMA table_info(target_components)", con)
    cols_cs = pd.read_sql_query("PRAGMA table_info(component_sequences)", con)
    log.info("target_dictionary cols: %s", cols_td["name"].tolist()[:8])
    log.info("target_components cols: %s", cols_tc["name"].tolist())
    log.info("component_sequences cols: %s", cols_cs["name"].tolist())

    # batched IN-query
    all_rows = []
    bs = 500
    for i in range(0, len(tids), bs):
        batch = tids[i:i + bs]
        placeholders = ",".join(["?"] * len(batch))
        q = f"""SELECT td.chembl_id AS target_chembl_id,
                       td.target_type,
                       cs.accession AS uniprot_id,
                       cs.sequence
                FROM target_dictionary td
                JOIN target_components tc ON tc.tid = td.tid
                JOIN component_sequences cs ON cs.component_id = tc.component_id
                WHERE td.chembl_id IN ({placeholders})
                  AND cs.sequence IS NOT NULL"""
        df = pd.read_sql_query(q, con, params=batch)
        all_rows.append(df)
    con.close()
    seq_df = pd.concat(all_rows, ignore_index=True)
    # Map uniprot_id -> sequence (dedupe)
    seq_df = seq_df.drop_duplicates(subset=["uniprot_id"]).dropna(subset=["uniprot_id", "sequence"])
    log.info("unique target uniprot_ids resolved: %d", len(seq_df))
    # cache
    seq_df.to_parquet(OUT_DIR / "cohort_target_sequences.parquet")
    return dict(zip(seq_df["uniprot_id"], seq_df["sequence"]))


def embed_proteins_esm2(seqs: dict[str, str], batch_size: int = 4, max_len: int = 1022) -> dict[str, np.ndarray]:
    """ESM2-150M (t30) — 640-dim per residue. Mean-pool over residues.

    max_len 1022 = ESM2's max context (1024 - 2 for special tokens).
    """
    from transformers import AutoTokenizer, AutoModel
    MODEL = "facebook/esm2_t30_150M_UR50D"
    log.info("Loading %s ...", MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float16 if DEVICE == "cuda" else torch.float32)
    model.to(DEVICE).eval()
    log.info("ESM2 loaded. n proteins: %d", len(seqs))

    out: dict[str, np.ndarray] = {}
    ids = list(seqs.keys())

    with torch.no_grad():
        for i in tqdm(range(0, len(ids), batch_size), desc="ESM2"):
            chunk_ids = ids[i:i + batch_size]
            chunk_seqs = [seqs[k][:max_len] for k in chunk_ids]
            enc = tok(chunk_seqs, padding=True, truncation=True,
                      max_length=max_len + 2, return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            h = model(**enc).last_hidden_state  # (B, L, 640)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = pooled.float().cpu().numpy().astype(np.float16)
            for j, uid in enumerate(chunk_ids):
                out[uid] = pooled[j]
    return out


def embed_text_scibert(text_df: pd.DataFrame, batch_size: int = 16, max_len: int = 512) -> dict[str, np.ndarray]:
    """SciBERT CLS pooled."""
    from transformers import AutoTokenizer, AutoModel
    MODEL = "allenai/scibert_scivocab_uncased"
    log.info("Loading %s ...", MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float16 if DEVICE == "cuda" else torch.float32)
    model.to(DEVICE).eval()
    log.info("SciBERT loaded. n trials: %d", len(text_df))

    out: dict[str, np.ndarray] = {}
    ids = text_df["nct_id"].tolist()
    texts = text_df["text"].tolist()

    with torch.no_grad():
        for i in tqdm(range(0, len(ids), batch_size), desc="SciBERT"):
            chunk_ids = ids[i:i + batch_size]
            chunk_texts = texts[i:i + batch_size]
            enc = tok(chunk_texts, padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            h = model(**enc).last_hidden_state[:, 0, :]
            h = h.float().cpu().numpy().astype(np.float16)
            for j, nct in enumerate(chunk_ids):
                out[nct] = h[j]
    return out


def get_text_corpus() -> pd.DataFrame:
    """Concatenate title + brief_summary for each cohort NCT, from AACT DuckDB.

    Constrain to cohort NCTs (both p2 and p3 cohorts combined).
    """
    p2 = pd.read_parquet(ROOT / "processed/splits/cohort_p2_nct_ids.parquet")
    p3 = pd.read_parquet(ROOT / "processed/splits/cohort_p3_nct_ids.parquet")
    cohort = set(p2["nct_id"]).union(set(p3["nct_id"]))
    log.info("Combined cohort size for text embeddings: %d", len(cohort))
    con = duckdb.connect(str(ROOT / "aact.duckdb"), read_only=True)
    # Studies table has brief_title and official_title; brief_summaries is a separate table
    placeholders = ",".join([f"'{x}'" for x in list(cohort)[:1]])  # check schema first
    schema_studies = con.execute("DESCRIBE studies").fetchdf()
    title_cols = [c for c in schema_studies["column_name"].tolist() if "title" in c.lower()]
    log.info("studies title cols: %s", title_cols)
    # brief_summaries
    try:
        schema_bs = con.execute("DESCRIBE brief_summaries").fetchdf()
        log.info("brief_summaries cols: %s", schema_bs["column_name"].tolist())
        has_bs = True
    except Exception:
        has_bs = False

    if has_bs:
        q = """
        SELECT s.nct_id,
               COALESCE(s.brief_title, '') AS brief_title,
               COALESCE(s.official_title, '') AS official_title,
               COALESCE(bs.description, '') AS summary
        FROM studies s
        LEFT JOIN brief_summaries bs ON bs.nct_id = s.nct_id
        WHERE s.nct_id IN (SELECT unnest(?))
        """
    else:
        q = """
        SELECT s.nct_id,
               COALESCE(s.brief_title, '') AS brief_title,
               COALESCE(s.official_title, '') AS official_title,
               '' AS summary
        FROM studies s
        WHERE s.nct_id IN (SELECT unnest(?))
        """
    df = con.execute(q, [list(cohort)]).fetchdf()
    con.close()
    df["text"] = (df["brief_title"].fillna("") + " || " +
                  df["official_title"].fillna("") + " || " +
                  df["summary"].fillna("")).str.slice(0, 4000)
    df = df[["nct_id", "text"]]
    log.info("Built text corpus: %d rows, mean_text_len=%.0f",
             len(df), df["text"].str.len().mean())
    return df


def main():
    t0 = time.time()
    log.info("=" * 80)
    log.info("Day-3 Phase C: Real embeddings")
    log.info("=" * 80)

    # ---- C1 ChemBERTa ----
    log.info("=== C1 ChemBERTa SMILES embeddings ===")
    out_smiles_path = OUT_DIR / "smiles_embeddings.npz"
    if out_smiles_path.exists():
        log.info("found existing %s, reusing", out_smiles_path)
    else:
        smiles_df = pd.read_parquet(OUT_DIR / "cohort_smiles.parquet")
        log.info("cohort_smiles loaded: %d rows", len(smiles_df))
        smi_emb = embed_smiles_chemberta(smiles_df, batch_size=64)
        np.savez_compressed(out_smiles_path,
                            ids=np.array(list(smi_emb.keys())),
                            emb=np.stack(list(smi_emb.values()), axis=0))
        log.info("wrote %s", out_smiles_path)

    # ---- C2 ESM2 ----
    log.info("=== C2 ESM2 protein embeddings ===")
    out_prot_path = OUT_DIR / "protein_embeddings.npz"
    if out_prot_path.exists():
        log.info("found existing %s, reusing", out_prot_path)
    else:
        seqs = get_uniprot_sequences()
        log.info("Resolved %d unique target sequences", len(seqs))
        if len(seqs) == 0:
            log.warning("No target sequences resolved — skipping ESM2 embeddings")
        else:
            prot_emb = embed_proteins_esm2(seqs, batch_size=4)
            np.savez_compressed(out_prot_path,
                                ids=np.array(list(prot_emb.keys())),
                                emb=np.stack(list(prot_emb.values()), axis=0))
            log.info("wrote %s", out_prot_path)

    # ---- C3 SciBERT ----
    log.info("=== C3 SciBERT text embeddings ===")
    out_text_path = OUT_DIR / "text_embeddings.npz"
    if out_text_path.exists():
        log.info("found existing %s, reusing", out_text_path)
    else:
        text_df = get_text_corpus()
        text_df.to_parquet(OUT_DIR / "cohort_text_corpus.parquet")
        txt_emb = embed_text_scibert(text_df, batch_size=24)
        np.savez_compressed(out_text_path,
                            ids=np.array(list(txt_emb.keys())),
                            emb=np.stack(list(txt_emb.values()), axis=0))
        log.info("wrote %s", out_text_path)

    elapsed = time.time() - t0
    log.info("Phase C embeddings done in %.0fs (%.1f min)", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
