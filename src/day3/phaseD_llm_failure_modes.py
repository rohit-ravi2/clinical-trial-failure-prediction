"""
Day-3 Phase D: Claude Sonnet failure-mode classification on ~20K unique
why_stopped strings via the Anthropic Message Batches API.

Resumable: persists batch_id and results to disk; re-running picks up where
it stopped.

Outputs:
  /mnt/ssd/research/trial-failure/processed/day3/llm_failure_modes.parquet
  /mnt/ssd/research/trial-failure/processed/day3/llm_batch_state.json
  /mnt/ssd/research/trial-failure/processed/day3/llm_failure_modes_summary.json
"""
from __future__ import annotations
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/mnt/ssd/research/trial-failure/logs/day3_phaseD.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("phaseD")

ROOT = Path("/mnt/ssd/research/trial-failure")
OUT_DIR = ROOT / "processed/day3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """You classify clinical trial termination reasons into ONE of these 6 categories. Respond with ONLY the category name in capital letters, no other text.

Categories:
- EFFICACY: primary endpoint not met, insufficient drug effect, drug worked but not better than control, futility
- SAFETY: adverse events, toxicity, deaths, serious adverse reactions, safety signal
- PK: pharmacokinetics/pharmacodynamics failure, dosing issues, drug interactions, poor bioavailability
- RECRUITMENT: enrollment failure, screen failures, poor retention, slow accrual, insufficient subjects
- OPERATIONAL: sponsor decision, business reasons, funding, IRB issues, strategic priority change, supply issues, COVID
- UNKNOWN: free text doesn't clearly indicate a category"""

CATEGORIES = {"EFFICACY", "SAFETY", "PK", "RECRUITMENT", "OPERATIONAL", "UNKNOWN"}


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def pick_model(client) -> str:
    """Find the most recent Sonnet 4.x model id."""
    try:
        models = client.models.list().data
        names = [m.id for m in models]
        log.info("Available models (first 10): %s", names[:10])
        # prefer claude-sonnet-4-5 if available, else any sonnet-4
        for prefer in ["claude-sonnet-4-5", "claude-sonnet-4"]:
            for m in models:
                if m.id.startswith(prefer):
                    log.info("Selected model: %s", m.id)
                    return m.id
        # fallback: any sonnet
        for m in models:
            if "sonnet" in m.id.lower():
                log.info("Fallback selected: %s", m.id)
                return m.id
        raise RuntimeError("No sonnet model found")
    except Exception as e:
        log.warning("models.list() failed: %s — falling back to 'claude-sonnet-4-5-20250929'", e)
        return "claude-sonnet-4-5-20250929"


def gather_unique_strings(top_n: int = 20000) -> pd.DataFrame:
    """Return DataFrame with one row per UNIQUE non-trivial why_stopped string,
    sorted by frequency descending (most common first).
    """
    con = duckdb.connect(str(ROOT / "aact.duckdb"), read_only=True)
    q = """
        SELECT TRIM(why_stopped) AS why_stopped, COUNT(*) AS n
        FROM failure_modes
        WHERE why_stopped IS NOT NULL
          AND TRIM(why_stopped) != ''
          AND LENGTH(TRIM(why_stopped)) > 3
        GROUP BY TRIM(why_stopped)
        ORDER BY n DESC
        LIMIT ?
    """
    df = con.execute(q, [top_n]).fetchdf()
    con.close()
    df["why_stopped_md5"] = df["why_stopped"].apply(md5)
    log.info("Gathered %d unique why_stopped strings (top %d)", len(df), top_n)
    return df


def submit_batches(client, model: str, df: pd.DataFrame, batch_size: int = 1000) -> list[str]:
    """Submit Message Batch jobs. Returns list of batch ids."""
    from anthropic.types.messages.batch_create_params import Request
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming

    state_path = OUT_DIR / "llm_batch_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        log.info("Loaded existing batch state with %d batches", len(state["batches"]))
    else:
        state = {"batches": [], "model": model, "submitted_at": time.time()}

    submitted_ids = {b["custom_id"] for batch in state["batches"] for b in batch.get("custom_ids", [])}

    pending = df[~df["why_stopped_md5"].isin(submitted_ids)]
    log.info("Pending strings to submit: %d (already submitted: %d)",
             len(pending), len(submitted_ids))

    new_batches = []
    for start in tqdm(range(0, len(pending), batch_size), desc="Submitting batches"):
        chunk = pending.iloc[start:start + batch_size]
        requests = []
        for _, row in chunk.iterrows():
            requests.append(Request(
                custom_id=row["why_stopped_md5"],
                params=MessageCreateParamsNonStreaming(
                    model=model,
                    max_tokens=8,
                    temperature=0,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": row["why_stopped"][:1500]}],
                ),
            ))
        try:
            batch = client.messages.batches.create(requests=requests)
            new_batches.append({
                "batch_id": batch.id,
                "custom_ids": [{"custom_id": r.custom_id} for r in requests],
                "submitted_at": time.time(),
            })
            log.info("Submitted batch %s with %d requests", batch.id, len(requests))
        except Exception as e:
            log.exception("Batch submission failed: %s", e)
            break
        state["batches"].extend(new_batches)
        state_path.write_text(json.dumps(state, indent=2))
        new_batches = []

    return [b["batch_id"] for b in state["batches"]]


def poll_and_collect(client, batch_ids: list[str], poll_interval: int = 30,
                     timeout_s: int = 3600) -> pd.DataFrame:
    """Poll batch endpoints until all complete; collect results."""
    rows = []
    pending = set(batch_ids)
    t0 = time.time()
    while pending and (time.time() - t0) < timeout_s:
        for bid in list(pending):
            try:
                b = client.messages.batches.retrieve(bid)
                status = b.processing_status
                log.info("Batch %s: %s (succeeded=%d, errored=%d)",
                         bid, status,
                         b.request_counts.succeeded if b.request_counts else 0,
                         b.request_counts.errored if b.request_counts else 0)
                if status == "ended":
                    # fetch results
                    for result in client.messages.batches.results(bid):
                        cid = result.custom_id
                        if result.result.type == "succeeded":
                            content = result.result.message.content
                            text = content[0].text.strip().upper() if content else ""
                            cat = text.split()[0] if text else "UNKNOWN"
                            if cat not in CATEGORIES:
                                cat = "UNKNOWN"
                            rows.append({"why_stopped_md5": cid, "llm_category": cat,
                                         "raw_response": text})
                        else:
                            rows.append({"why_stopped_md5": cid, "llm_category": "ERROR",
                                         "raw_response": str(result.result)[:200]})
                    pending.discard(bid)
            except Exception as e:
                log.warning("Batch %s poll failed: %s", bid, e)
        if pending:
            time.sleep(poll_interval)
    return pd.DataFrame(rows)


def merge_with_rule_labels(llm_df: pd.DataFrame) -> dict:
    """Join LLM labels back to failure_modes table via why_stopped_md5.

    Categories LLM and rule-based agree → high confidence.
    Categories disagree → prefer LLM.
    """
    con = duckdb.connect(str(ROOT / "aact.duckdb"))
    # Load failure_modes
    fm = con.execute("SELECT * FROM failure_modes").fetchdf()
    fm["why_stopped_md5"] = fm["why_stopped"].apply(lambda s: md5(s.strip()) if isinstance(s, str) else None)
    # join
    merged = fm.merge(llm_df[["why_stopped_md5", "llm_category"]], on="why_stopped_md5", how="left")
    # category mapping rule -> LLM equivalent
    rule_to_llm = {
        "efficacy": "EFFICACY", "safety": "SAFETY", "pk": "PK",
        "recruitment": "RECRUITMENT", "operational": "OPERATIONAL", "unknown": "UNKNOWN",
    }
    merged["rule_category"] = merged["label_primary"].map(rule_to_llm).fillna("UNKNOWN")
    merged["llm_category"] = merged["llm_category"].fillna("UNKNOWN")
    merged["final_category"] = merged.apply(
        lambda r: r["llm_category"] if r["llm_category"] != "UNKNOWN"
        else r["rule_category"], axis=1)
    merged["agreement"] = merged["llm_category"] == merged["rule_category"]

    # write back to DuckDB as a new table
    out = merged[["nct_id", "why_stopped_md5", "rule_category", "llm_category",
                  "final_category", "agreement"]]
    con.execute("DROP TABLE IF EXISTS failure_modes_llm")
    con.execute("CREATE TABLE failure_modes_llm AS SELECT * FROM out")
    con.close()

    counts = merged["final_category"].value_counts().to_dict()
    agreement_rate = float(merged["agreement"].mean())
    return {
        "final_category_counts": counts,
        "agreement_rate": agreement_rate,
        "n_total": len(merged),
        "llm_efficacy": int(counts.get("EFFICACY", 0)),
        "llm_safety": int(counts.get("SAFETY", 0)),
        "llm_pk": int(counts.get("PK", 0)),
        "llm_recruitment": int(counts.get("RECRUITMENT", 0)),
        "llm_operational": int(counts.get("OPERATIONAL", 0)),
        "llm_unknown": int(counts.get("UNKNOWN", 0)),
    }


def main(n_unique: int = 20000):
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if len(api_key) < 50:
        log.error("ANTHROPIC_API_KEY not set or too short — Phase D cannot run.")
        log.error("Set ANTHROPIC_API_KEY in the shell environment and rerun.")
        sys.exit(2)
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    model = pick_model(client)
    df = gather_unique_strings(top_n=n_unique)
    df.to_parquet(OUT_DIR / "llm_unique_strings.parquet")
    log.info("Wrote %d unique strings to llm_unique_strings.parquet", len(df))

    batch_ids = submit_batches(client, model, df, batch_size=1000)
    log.info("Total batches submitted: %d", len(batch_ids))

    results_df = poll_and_collect(client, batch_ids)
    log.info("Collected %d results", len(results_df))

    # Merge with input strings
    full = df.merge(results_df, on="why_stopped_md5", how="left")
    full["llm_category"] = full["llm_category"].fillna("UNKNOWN")
    full.to_parquet(OUT_DIR / "llm_failure_modes.parquet")

    # Merge with rule-based table & write back
    summary = merge_with_rule_labels(full[["why_stopped_md5", "llm_category"]].drop_duplicates())
    (OUT_DIR / "llm_failure_modes_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("LLM-merged summary: %s", json.dumps(summary, indent=2))
    print("\n=== Phase D Failure-Mode LLM Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
