"""Phase C — failure-mode label construction.

Taxonomy (locked): EFFICACY, SAFETY, PK, RECRUITMENT, OPERATIONAL.
Plus UNKNOWN for unclassifiable.

Sources (priority order):
1. FDA Complete Response Letter reason — not available in Day 1 raw; treated
   here as enhancement deferred. (We have openFDA labels in fda_adcom/ but
   they're approval documents not rejection letters.)
2. CT.gov why_stopped — primary source. Rule-based regex classifier plus
   optional Anthropic Sonnet 4.5 batch refinement when ANTHROPIC_API_KEY is set.
3. FAERS post-market spike — used for already-completed trials to flag
   SAFETY mode based on AE counts within 2 years post-completion.
4. AdCom voting — not directly downloadable as structured data; deferred.

For each terminated/withdrawn/suspended trial, store a 5-bit vector across the
five categories. Single-most-probable label is also stored.

Output:
- DuckDB table `failure_modes` with columns:
    nct_id, label_efficacy, label_safety, label_pk, label_recruitment,
    label_operational, label_primary, source, conf, why_stopped, classified_at
- Coverage / category counts logged.
"""
from __future__ import annotations
import json, logging, os, re, sys, time, hashlib
from pathlib import Path
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path("work")
DUCKDB = ROOT / "aact.duckdb"
LOG = ROOT / "logs/phaseC.log"
OUT_JSON = ROOT / "results/day2/phaseC_summary.json"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler(sys.stdout)])
log = logging.getLogger("phaseC")

CATEGORIES = ["efficacy", "safety", "pk", "recruitment", "operational"]

# ===========================================================
# Rule-based classifier on why_stopped strings
# ===========================================================
# Each pattern: (compiled_regex, category, weight)
# Patterns scored independently; final = argmax(per-category sum).
RULES: list[tuple[re.Pattern, str, float]] = [
    # SAFETY
    (re.compile(r"\b(adverse|toxicit|safety|side effect|sae\b|aes\b|serious adverse|fatal|death(s)? in|hepato|cardiotox|nephrotox|qt prolongat|liver injury|hepatic|liver enzyme|hemolytic|thrombocytopenia|neutropenia|allergic reaction|anaphyl|elevated lft|grade [3-5] toxicity)", re.I), "safety", 1.5),
    (re.compile(r"\b(unfavor[a-z]+ (?:safety|benefit-risk|risk[/-]benefit)|risk[/-]benefit (?:ratio|profile)|risk outweigh|benefit risk profile|imbalance in (?:adverse|safety))", re.I), "safety", 1.7),
    (re.compile(r"\b(dsmb|data safety|safety review|stopped (?:early|prematurely) (?:due|for) safety|safety signal|safety concern|safety hold|safety findings|stopping rule|fda (?:hold|placed)|clinical hold)", re.I), "safety", 1.8),

    # EFFICACY
    (re.compile(r"\b(lack of efficacy|no efficacy|insufficient efficacy|inefficac|did not (?:meet|reach|show)|failure to (?:demonstrate|show)|negative result|primary endpoint not met|missed primary endpoint|did not achieve|futility|futile|interim (?:analysis|review) (?:.{0,30}?)futil)", re.I), "efficacy", 1.8),
    (re.compile(r"\b(efficacy (?:was )?(?:not|insufficient|inadequate)|underwhelming|less effective than|inferior to (?:standard|control)|no clinical benefit)", re.I), "efficacy", 1.5),
    (re.compile(r"\b(superior(?:ity)? not (?:demonstrated|established)|primary endpoint (?:was )?not met|efficacy hurdle)", re.I), "efficacy", 1.4),

    # PK / PHARMACOLOGY
    (re.compile(r"\b(pharmacokinetic|pk profile|bioavailab|exposure (?:was|too) (?:low|insufficient)|drug levels|absorpt|metaboli[sz]m issue|formulation issue|stability)", re.I), "pk", 1.4),
    (re.compile(r"\b(dose escalation (?:stopped|halted)|dlt|dose limiting toxicity|maximum tolerated dose|mtd not reached|drug exposure)", re.I), "pk", 1.2),

    # RECRUITMENT
    (re.compile(r"\b(enrollment|accrual|recruit|enrol(?:l|ling|led|lment)?|participant|subject|patient)\b", re.I), "recruitment", 0.0),  # placeholder; never matches alone
    (re.compile(r"\b(lack of (?:enrollment|accrual|recruit|recruitment|inclusion|patients?|subjects?|participants?|enrolment))", re.I), "recruitment", 2.0),
    (re.compile(r"\b(no (?:enrollment|accrual|recruit(?:ment)?|patients?|subjects?|participants?|enrolment))", re.I), "recruitment", 2.0),
    (re.compile(r"\b(low|slow|poor|under|inadequate|insufficient|limited)\s+(?:enrollment|accrual|recruit(?:ment)?|enrolment)", re.I), "recruitment", 1.8),
    (re.compile(r"\b(enrollment|accrual|recruit(?:ment)?)\s+(?:rate|was|is|too|issues?|problems?|challenges?|difficult)", re.I), "recruitment", 1.6),
    (re.compile(r"\b(failure to (?:enroll|recruit)|unable to (?:enroll|recruit)|too few patient|too few subject)", re.I), "recruitment", 1.8),
    (re.compile(r"\b(no patient(?:s)? enrol|no participant(?:s)? enrol|insufficient patient|insufficient subject|no subject(?:s)? enrol)", re.I), "recruitment", 1.7),
    (re.compile(r"\b(lower than (?:expected|anticipated))\b", re.I), "recruitment", 1.0),
    (re.compile(r"\bstudy never started\b", re.I), "recruitment", 1.4),
    (re.compile(r"\b<\s*75% participation\b", re.I), "recruitment", 1.5),
    (re.compile(r"\b(competing trial|standard of care chang|approval of (?:a |an |new )?(?:alternative|competitor))", re.I), "recruitment", 1.0),

    # OPERATIONAL
    (re.compile(r"\b(business (?:decision|need|reason|objective|consideration)|strategic (?:decision|reason|priorities|consideration)|sponsor(?:'s)? decision|portfolio (?:decision|priorit|review)|company (?:decided|decision|merger|acquisition)|discontinu(?:ed|ation) (?:of|by) sponsor|terminated for business)", re.I), "operational", 1.7),
    (re.compile(r"\b(funding|budget|financial|expensive|economic|lack of fund|not funded|no fund|unfund)", re.I), "operational", 1.1),
    (re.compile(r"\b(logistic|administrative|staff(?:ing)? (?:shortage|issue|change|relocat)|investigator (?:left|relocat|retire|unavailable)|principal investigator|\bpi\b (?:left|decision|request|relocat|retire|unavailable)|change in (?:leadership|management|sponsor))", re.I), "operational", 1.4),
    (re.compile(r"\b(covid|coronavirus|pandemic|sars-cov)", re.I), "operational", 1.5),
    (re.compile(r"\b(supply (?:issue|chain|shortage)|drug supply|product supply|manufactur(?:ing|er)|production halt|raw material)", re.I), "operational", 1.3),
    (re.compile(r"\b(irb|ethics committee|ethical (?:issue|concern)|protocol (?:amendment|modification|change) (?:not )?feasib|regulatory (?:hold|issue|delay)|fda (?:request|requirement)|protocol (?:was )?(?:amended|modified))", re.I), "operational", 1.0),
    (re.compile(r"\b(study (?:was )?(?:closed|terminated|withdrawn|discontinued) (?:due to|for|because) (?:funding|administrative|sponsor|business))", re.I), "operational", 1.6),
    (re.compile(r"\b(see (?:termination reason|detailed description)|see protocol|see record|see comments)", re.I), "operational", 0.5),
]

def classify_reason(text: str) -> tuple[dict, str, float]:
    """Return (scores_dict, primary_category, confidence)."""
    if not isinstance(text, str) or not text.strip():
        return {c: 0.0 for c in CATEGORIES}, "unknown", 0.0
    text = text.strip()
    scores = {c: 0.0 for c in CATEGORIES}
    for pat, cat, w in RULES:
        if pat.search(text):
            scores[cat] += w
    total = sum(scores.values())
    if total == 0:
        return scores, "unknown", 0.0
    primary = max(scores, key=scores.get)
    # confidence: top score / sum
    conf = scores[primary] / total
    return scores, primary, float(conf)

# ===========================================================
# Anthropic API path (only if ANTHROPIC_API_KEY is set)
# ===========================================================
def maybe_anthropic_refine(unknown_strings: list[str], max_calls: int = 200) -> dict[str, dict]:
    """Send a batch of UNKNOWN reasons to Sonnet for classification.

    Returns mapping reason -> {"category": str, "conf": float}.
    Skipped silently if key absent or anthropic not installed.
    """
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        log.info("ANTHROPIC_API_KEY empty; skipping LLM refinement")
        return {}
    try:
        import anthropic
    except ImportError:
        log.warning("anthropic package not installed; skipping LLM refinement")
        return {}
    client = anthropic.Anthropic()
    sample = unknown_strings[:max_calls]
    out = {}
    SYSTEM = ("You classify clinical-trial stop reasons. Reply with one of "
              "EFFICACY, SAFETY, PK, RECRUITMENT, OPERATIONAL, UNKNOWN. No other text.")
    for s in tqdm(sample, desc="anthropic-classify"):
        try:
            r = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=10,
                system=SYSTEM,
                messages=[{"role": "user", "content": s[:1000]}],
            )
            lbl = r.content[0].text.strip().upper()
            if lbl in {"EFFICACY","SAFETY","PK","RECRUITMENT","OPERATIONAL","UNKNOWN"}:
                out[s] = {"category": lbl.lower(), "conf": 0.7}
        except Exception as e:
            log.warning(f"anthropic call failed: {e}")
            break
    return out

# ===========================================================
# FAERS-based safety signal (optional, for completed trials)
# ===========================================================
def build_faers_safety_signal(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Aggregate FAERS report counts per drug per quarter, look for spikes
    within 2 years post-completion. Returns df: nct_id, faers_signal_score.
    Requires FAERS quarterly files to have been ingested. Skipped if not."""
    try:
        n = con.execute("SELECT COUNT(*) FROM faers_demo").fetchone()[0]
    except Exception:
        log.info("FAERS not yet loaded; skipping safety signal")
        return pd.DataFrame(columns=["nct_id", "faers_signal_score"])
    log.info(f"FAERS demo rows: {n}")
    # we'd join FAERS DRUG.drugname to trials primary_drug_canonical,
    # count reports within +24mo of completion_date, and compute z-score
    # against baseline. Implementation deferred until FAERS ingest is done.
    return pd.DataFrame(columns=["nct_id", "faers_signal_score"])

# ===========================================================
# Main
# ===========================================================
def main():
    t0 = time.time()
    con = duckdb.connect(str(DUCKDB), read_only=False)

    # Pull all terminated/withdrawn/suspended trials + why_stopped
    log.info("loading failed trials with reasons")
    df = con.execute("""
        SELECT s.nct_id, s.overall_status, s.why_stopped, t.p2_success, t.p3_success,
               t.phase, t.label_available, t.primary_drug_canonical
        FROM studies s
        LEFT JOIN trials_labeled t ON s.nct_id = t.nct_id
        WHERE s.overall_status IN ('TERMINATED','WITHDRAWN','SUSPENDED')
    """).fetchdf()
    log.info(f"  total failed trials: {len(df)}")
    log.info(f"  with why_stopped: {df['why_stopped'].notna().sum()}")

    # classify
    log.info("running rule-based classifier")
    cats, primary, conf = [], [], []
    for s in tqdm(df["why_stopped"].tolist(), desc="rules"):
        sc, prim, c = classify_reason(s)
        cats.append(sc); primary.append(prim); conf.append(c)
    df["scores"] = cats
    df["label_primary"] = primary
    df["conf"] = conf

    # Per-category indicator (multi-label): >0 in that category
    for c in CATEGORIES:
        df[f"label_{c}"] = df["scores"].apply(lambda s: int(s.get(c, 0.0) > 0))

    cov_with_reason = (df["label_primary"] != "unknown").sum() / max(1, len(df))
    log.info(f"rule-based primary-label coverage: {cov_with_reason:.1%}")
    log.info("category counts (multi-label, can sum > N):")
    cat_counts = {c: int(df[f"label_{c}"].sum()) for c in CATEGORIES}
    log.info(f"  {cat_counts}")
    log.info("primary distribution:")
    log.info("\n" + df["label_primary"].value_counts().to_string())

    # Try Anthropic refinement on a sample of UNKNOWN
    unknown_strings = df[df["label_primary"] == "unknown"]["why_stopped"].dropna().drop_duplicates().tolist()
    log.info(f"unique UNKNOWN reasons available for LLM: {len(unknown_strings)}")
    refined = maybe_anthropic_refine(unknown_strings, max_calls=200)
    if refined:
        log.info(f"LLM resolved {len(refined)} additional reasons")
        mask = (df["label_primary"] == "unknown") & df["why_stopped"].isin(refined.keys())
        for k in df[mask].index:
            txt = df.at[k, "why_stopped"]
            v = refined.get(txt)
            if v:
                df.at[k, "label_primary"] = v["category"]
                df.at[k, "conf"] = v["conf"]
                if v["category"] in CATEGORIES:
                    df.at[k, f"label_{v['category']}"] = 1

    # Save to DuckDB
    df["source"] = "ctgov_why_stopped_rules"
    df["classified_at"] = datetime.now(timezone.utc)
    df["scores_json"] = df["scores"].apply(lambda s: json.dumps({k: float(v) for k, v in s.items()}))
    out_df = df[["nct_id", "label_primary", "conf",
                 "label_efficacy", "label_safety", "label_pk",
                 "label_recruitment", "label_operational",
                 "source", "why_stopped", "scores_json", "classified_at"]]
    con.execute("DROP TABLE IF EXISTS failure_modes")
    con.execute("""
        CREATE TABLE failure_modes (
            nct_id VARCHAR PRIMARY KEY,
            label_primary VARCHAR,
            conf DOUBLE,
            label_efficacy INTEGER,
            label_safety INTEGER,
            label_pk INTEGER,
            label_recruitment INTEGER,
            label_operational INTEGER,
            source VARCHAR,
            why_stopped VARCHAR,
            scores_json VARCHAR,
            classified_at TIMESTAMP
        )
    """)
    con.register("out_df", out_df)
    con.execute("INSERT INTO failure_modes SELECT * FROM out_df")
    n_loaded = con.execute("SELECT COUNT(*) FROM failure_modes").fetchone()[0]
    log.info(f"loaded {n_loaded} rows into failure_modes")

    # Summary JSON
    summary = {
        "total_failed_trials": int(len(df)),
        "with_why_stopped": int(df["why_stopped"].notna().sum()),
        "coverage_primary_label_pct": round(100 * cov_with_reason, 2),
        "category_multilabel_counts": cat_counts,
        "primary_distribution": df["label_primary"].value_counts().to_dict(),
        "anthropic_llm_resolved": len(refined),
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    log.info(f"wrote {OUT_JSON}: {summary}")
    return summary

if __name__ == "__main__":
    main()
