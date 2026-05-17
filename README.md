# Clinical Trial Failure Prediction — Leak-Clean Multimodal Fusion

[![rohitravi.com](https://img.shields.io/badge/project_page-rohitravi.com-blue)](https://rohitravi.com/projects/clinical-trial-failure-prediction)

Audited Phase 3 / time-split AUROC = **0.8603** on the AACT (CT.gov + Drugs@FDA) cohort. The headline isn't the AUROC — it's the **leakage audit methodology** (null-label experiment, per-feature univariate train/test AUROC, cohort-row-ordering bug-catch) that proved the number is signal, not label leakage.

## Headline numbers

| Metric | Value |
|--------|------:|
| **Audited P3 / time-split AUROC** | **0.8603** |
| Real multimodal fusion P3 / time | 0.8746 |
| Best multimodal lift (indication holdout) | **+0.034 over tab-only** |
| Null-label control (target ≤ 0.55) | 0.4818 ✓ |
| 5-fold within-train CV (should match test) | 0.8647 ± 0.005 ✓ |
| ECE on P3 / time | 0.0239 |
| Brier on P3 / time | 0.0749 |

## What's in this repo

```
src/                     # Day 1 (gates 0-5), Day 2 (phases A-E), Day 3 (audit + fusion)
docs/                    # REPORT.md, REPORT_day2.md, REPORT_day3.md — full project narrative
results/                 # JSON metrics, calibration plots, audit outputs
models/                  # 8 LightGBM models (P2/P3 × time/drugclass/indication/target splits)
logs/gates.log           # Append-only gate-progress log across all 3 days
```

## Project narrative (3 days)

- **Day 1** — AACT 2026-05-01 dump (582,806 trials) → DuckDB (1.5GB). FDA Drugs@FDA join: 46.91% P3 match rate. 8 frozen splits. Baseline LightGBM: P3 / time AUROC = 0.7663.
- **Day 2** — Discovered direct FDA-match features encoded the label structurally (matching means the drug got FDA-reviewed → biased toward success). Dropped 506 → 134 features. Leak-clean LGBM v2: P3 / time AUROC = 0.8718.
- **Day 3** — Rigorous leakage audit confirmed the headline. Real multimodal fusion (ChemBERTa + ESM2 + SciBERT) added +0.034 on indication holdout.

Full reports under `docs/`:
- [Day 1 report](docs/REPORT.md)
- [Day 2 supplemental](docs/REPORT_day2.md)
- [Day 3 supplemental](docs/REPORT_day3.md)

## The leakage audit — the differentiated contribution

Published trial-outcome prediction work (HINT NeurIPS 2022, Lo 2023, Hong 2020) reports AUROC 0.7–0.85 routinely **without** these three checks:

1. **Null-label experiment**: train on shuffled train labels, eval on real test. Target AUROC ≤ 0.55. Got 0.4818 — features don't encode label structurally.
2. **5-fold within-train CV**: 0.8647 ± 0.005, matches test 0.8718 → no temporal shortcut.
3. **Per-feature univariate test/train AUC delta**: no feature has |test_AUC − 0.5| > 0.20. Top: `moa_bucket_unknown` (test 0.695) — biologically plausible.

**Audit bug catch.** First audit run produced fake AUROC 0.91+ because `feats[isin(keep)].reset_index()` preserved features_v2 row order, but split-pickle positional indices reference cohort order from `cohort_<task>_nct_ids.parquet`. Fixed via `set_index('nct_id').loc[cohort_ncts]`. Generally applicable warning for anyone working with the same split-pickle convention.

## Reproducing

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download raw data (~30 GB total)
bash scripts/data_download.sh   # AACT, ChEMBL, FDA, UniProt, Open Targets

# Build features
python src/gate1_aact.py
python src/gate2_fda.py
python src/gate3_features.py
python src/gate4_splits.py

# Day 2 leak-clean rebuild
python src/day2/phaseA_downloads.py
python src/day2/phaseB_features_v2.py
python src/day2/phaseB_train.py
python src/day2/phaseE_fusion.py

# Day 3 audit
python src/day3/phaseB_leakage_audit.py
python src/day3/phaseC_real_embeddings.py
python src/day3/phaseC4_real_fusion.py
```

## Honest assessment

The multimodal-fusion approach is on the curve, not above it. HINT (Fu et al. NeurIPS 2022) and similar papers cluster around AUROC 0.80–0.85 on the same kind of cohort. The +0.034 indication-holdout lift is the only architectural differentiator. **The leakage audit is the load-bearing methods contribution** — null-label experiment + cohort-row-ordering bug-catch is not standard practice in this literature.

## Open work

**Phase D (LLM failure-mode classification)** didn't ship. The script `src/day3/phaseD_llm_failure_modes.py` is ready and resumable via the Anthropic Batches API at temperature 0 over 33,042 unique `why_stopped` strings. Blocker was an empty `ANTHROPIC_API_KEY` in the agent shell. This would be the genuinely differentiated contribution — most published work outputs success/fail, not *why*.

## License

MIT. See [LICENSE](LICENSE).
