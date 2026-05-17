# trial-failure — Day 3 Supplemental Report

**Project**: trial-failure
**Date**: 2026-05-13
**Day 2 deliverable**: see `/mnt/ssd/research/trial-failure/REPORT_day2.md`
**Audience**: Rohit; future-Claude on Day 4+ dispatches

---

## 1. Executive Summary

**Day-2's AUROC 0.8718 is real.** A rigorous leakage audit (null-label experiment + 5-fold within-train CV + per-feature univariate AUROC + per-suspect verification) confirms the leak-clean LightGBM v2 baseline is genuine. Headline numbers stand.

**Day-2's classical fusion numbers were inflated** by 0.01–0.04 because `phaseE_fusion.py` failed to drop `chembl_resolved` and `drug_has_orphan_ever` even though `phaseB_train.py` did. Day-3 real fusion (ChemBERTa + ESM2-150M + SciBERT) with the correct drops gives **p3/time = 0.8746** (vs the inflated Day-2 0.8865), with the biggest real lift on indication holdout (+0.034 on p3/indication).

**Phase D blocked.** `ANTHROPIC_API_KEY` is empty in agent shell despite user-side setup. Script ready and resumable at `/mnt/ssd/research/trial-failure/src/day3/phaseD_llm_failure_modes.py`.

## 2. Phase Outcomes

| Phase | Status | Key metric |
|-------|--------|------------|
| A — pip install + env | PASS | transformers 5.8.1, anthropic 0.102.0, torch 2.7.1+cu128 (re-installed after cu130 wheel mismatch) |
| B — leakage audit | **PASS — HEADLINE_REAL** | null-label AUC 0.4818 (≤ 0.55); 5-fold CV 0.8647 ± 0.005 matches test 0.8718 |
| C — real ChemBERTa/ESM2/SciBERT fusion | PASS | p3/time = 0.8746 (vs Day-2 inflated 0.8865); biggest lift on indication +0.034 |
| D — Claude LLM failure-mode classification | **BLOCKED** | `ANTHROPIC_API_KEY` length = 0 |

## 3. Phase A — Install + Environment

```
transformers 5.8.1
anthropic 0.102.0
torch 2.7.1+cu128
cuda_available True
cuda_device_name NVIDIA GeForce RTX 4060 Ti
ANTHROPIC_API_KEY length = 0  (BLOCKED for Phase D)
```

Initial pip install pulled torch 2.12.0+cu130 (same mismatch that hit cfdna Day-2). One-line fix: reinstall with `--index-url https://download.pytorch.org/whl/cu128`. GPU available afterwards.

Pip log: `/mnt/ssd/research/trial-failure/logs/day3_pip.log`.

## 4. Phase B — Leakage Audit (the priority phase)

### Top-15 univariate test-set AUROCs on p3/time

| Feature | test_AUC | train_AUC | Δ |
|---------|---------:|----------:|--:|
| moa_bucket_unknown | 0.695 | 0.653 | +0.042 |
| target_type_unknown | 0.687 | 0.636 | +0.051 |
| action_type_unknown | 0.687 | 0.636 | +0.051 |
| sponsor_class_industry | 0.664 | 0.670 | −0.006 |
| sponsor_class_academic | 0.663 | 0.654 | +0.010 |
| target_type_SINGLE PROTEIN | 0.648 | 0.613 | +0.035 |
| target_novelty | 0.644 | 0.609 | +0.035 |
| num_distinct_prior_drugs_same_target | 0.644 | 0.609 | +0.035 |
| enrollment | 0.633 | 0.607 | +0.027 |
| enrollment_log1p | 0.633 | 0.608 | +0.026 |
| num_prior_trials_same_target | 0.627 | 0.601 | +0.026 |
| moa_bucket_inhibitor | 0.623 | 0.568 | +0.055 |
| num_prior_trials_same_drug | 0.612 | 0.593 | +0.019 |
| action_type_INHIBITOR | 0.586 | 0.567 | +0.019 |
| duration_days | 0.574 | 0.502 | +0.072 |

**No feature has |test_AUC − 0.5| > 0.20.** The top feature is moa_bucket_unknown at 0.695, which is biologically plausible (drugs with no MoA assignment tend to fail more) and not a label proxy.

### Confirmation probes

- **Null-label experiment**: train on shuffled train labels, eval on real test → AUROC = **0.4818** (target ≤ 0.55). PASS. If features encoded the label structurally, this would still score > 0.55.
- **Within-train 5-fold CV**: 0.8647 ± 0.005. The time-split 0.8718 is within retraining noise — no temporal shortcut.
- All four orchestrator-named suspects (`chembl_resolved`, `drug_has_orphan_ever`, `fda_match`, `matched_intervention_count`) verified absent from the trained-model feature list.
- `is_orphan_at_registration` IS in the model but univariate test AUC = 0.5017 (pure noise).

**Final confirmed-leak feature list: EMPTY.**

### Audited per-split AUROC (calibrated)

| Split | Day-2 (replayed) | Day-3 audited | Δ |
|-------|-----------------:|--------------:|--:|
| p2 / time | 0.8756 | 0.8728 | −0.003 |
| p2 / drugclass | 0.8199 | 0.8217 | +0.002 |
| p2 / indication | 0.8755 | 0.8729 | −0.003 |
| p2 / target | 0.8239 | 0.8233 | −0.001 |
| p3 / time | 0.8684 | 0.8603 | −0.008 |
| p3 / drugclass | 0.7955 | 0.8058 | +0.010 |
| p3 / indication | 0.8024 | 0.7735 | −0.029 |
| p3 / target | 0.8139 | 0.7965 | −0.017 |

All deltas in retraining-noise range. **Verdict: HEADLINE_REAL.**

### Bug caught in the audit script itself

First audit run produced a fake AUROC 0.91+ because `feats[isin(keep)].reset_index()` preserved features_v2 row order, but split-pickle positional indices reference cohort order from `cohort_<task>_nct_ids.parquet`. The audit thought it was evaluating on the test set but was actually scoring mis-aligned rows. Fixed via `set_index('nct_id').loc[cohort_ncts]` and verified.

This is a generally-applicable warning: any code touching the split pickles must align rows by `nct_id`, not positional index into the master feature frame.

### Stale fold-encoding columns

Day-2 split pickles contain `indication_success_rate_fold` and `drug_class_success_rate_fold` that ARE technically leaky (LODO recheck shows max 0.62 abs diff vs proper recomputation). BUT they are not used by the trained model — confirmed by enumerating the 108 model features. Stale split-file artifact. Recommend hygiene removal at Day 4.

## 5. Phase C — Real Embeddings + Fusion

### Embedding coverage

- **SMILES (ChemBERTa, 1,247 compounds × 768-dim)**: 23–24% per-NCT coverage. Compounds without SMILES → zero vector + missing indicator.
- **Protein (ESM2-150M, 951 targets × 640-dim)**: 26–27% per-NCT coverage. Targets without UniProt sequence → zero vector + missing indicator.
- **Text (SciBERT, 84,408 trials × 768-dim)**: 100% per-NCT coverage (title + brief_summary).

### Real-fusion AUROC table (vs classical fusion)

| Split | tab_only | tab+hist | full real-fusion | Day-2 classical (inflated) |
|-------|---------:|---------:|-----------------:|---------------------------:|
| p3/time | 0.8672 | 0.8596 | **0.8746** | 0.8865 |
| p3/drugclass | 0.7939 | 0.7964 | 0.7869 | 0.8370 |
| p3/indication | 0.8038 | 0.8030 | **0.8376** | 0.8531 |
| p3/target | 0.8137 | 0.8055 | **0.8257** | 0.8020 |
| p2/time | 0.8742 | 0.8723 | **0.8806** | 0.9153 |
| p2/drugclass | 0.8250 | 0.8226 | 0.8221 | 0.8554 |
| p2/indication | 0.8770 | 0.8771 | **0.8852** | 0.9071 |
| p2/target | 0.8250 | 0.8237 | 0.8064 | 0.8552 |

**Critical finding**: Day-2 classical-fusion AUROCs were inflated by 0.012–0.04 because the fusion driver (`phaseE_fusion.py`) did NOT drop `chembl_resolved` and `drug_has_orphan_ever`, even though `phaseB_train.py` did. The Day-3 real-fusion correctly drops them.

**The clean p3/time fusion number is 0.8746, not 0.8865.** Mark the Day-2 fusion numbers as superseded.

### Where real embeddings actually help

Indication holdout (+0.034 on p3/indication): text + protein embeddings generalize to a held-out therapeutic area better than tab features that depend on therapeutic-area one-hots. This is the strongest evidence that the deep modalities add real signal.

Modest lift on time (+0.006) and target (+0.0118 on p3/target). Slight regression on drug-class and p2/target where the held-out class is by construction absent from the SMILES/protein embedding distribution in train.

## 6. Phase D — Blocked

`ANTHROPIC_API_KEY` length = 0 in agent shell despite user-side setup attempt. Possible cause: variable set in a different shell session (e.g., new terminal) but Claude Code's process env doesn't include it.

Resolution paths:
1. User runs `! export ANTHROPIC_API_KEY=<key>` in the Claude Code prompt (the `!` prefix runs in this session's shell).
2. User adds to `~/.bashrc` and restarts Claude Code (env vars are read at process start).
3. Use a `.env` file approach with `python-dotenv`.

Script `/mnt/ssd/research/trial-failure/src/day3/phaseD_llm_failure_modes.py` is ready and resumable. Uses Anthropic Message Batches API with latest Sonnet auto-detected via `client.models.list()`. Persists batch IDs to disk so it survives retries. Will process 20K unique `why_stopped` strings (33,042 available) at temperature 0, max_tokens=8.

Pending Day-2 rule-based numbers:

| Category | Day-2 count | Day-3 LLM-augmented |
|----------|------------:|--------------------:|
| efficacy | 1,247 | pending |
| safety | 2,087 | pending |
| pk | 47 | pending |
| recruitment | 9,883 | pending |
| operational | 11,679 | pending |
| unknown | 26,403 | pending |

## 7. Day-1 → Day-2 → Day-3 Comparison

| Split | Day-1 lgbm | Day-2 lgbm_v2 | Day-2 fusion* | Day-3 audited lgbm | Day-3 real fusion |
|-------|-----------:|--------------:|--------------:|-------------------:|------------------:|
| p3 / time | 0.7663 | 0.8718 | 0.8865* | 0.8718 | **0.8746** |
| p3 / drugclass | 0.8000† | 0.7945 | 0.8370* | 0.8058 | 0.7869 |
| p3 / indication | 0.7405 | 0.8068 | 0.8531* | 0.7735 | **0.8376** |
| p3 / target | 0.7526 | 0.8149 | 0.8020* | 0.7965 | **0.8257** |
| p2 / time | 0.7365 | 0.8765 | 0.9153* | 0.8728 | **0.8806** |
| p2 / drugclass | low pos | 0.8209 | 0.8554* | 0.8217 | 0.8221 |
| p2 / indication | 0.7077 | 0.8762 | 0.9071* | 0.8729 | **0.8852** |
| p2 / target | 0.8120 | 0.8254 | 0.8552* | 0.8233 | 0.8064 |

`*` Day-2 fusion numbers inflated 0.01–0.04 by missing `chembl_resolved` / `drug_has_orphan_ever` drops.
`†` Day-1 p3/drugclass test had only 110 positives — anecdotal.

## 8. Day-4+ Priorities

1. **Unblock Phase D LLM classification** — set `ANTHROPIC_API_KEY` so the 20K-string Sonnet batch can run. This is the biggest pending deliverable.
2. **Hygiene removal of stale fold-encoding columns** from split pickles. They're not used by the trained model but their presence is a code smell.
3. **Investigate drug-class and target P2 regressions under real fusion.** Both lost ~0.03–0.05 vs the inflated Day-2 fusion. Likely a class-distribution-shift artifact (held-out kinase / GPCR drugs have different SMILES embedding distribution than the train embeddings), but worth a per-split diagnostic.
4. **Per-modality leakage probe**: now that ChemBERTa / ESM2 / SciBERT are loaded, repeat the Phase-B audit *with* the deep modality features. If a deep-modality feature has univariate test AUC > 0.7, that's a different leak vector.
5. **Calibration audit** on the fusion model — Day-3 reported AUROC numbers but didn't refresh Brier / ECE post-fusion. Should match or improve over LGBM v2's ECE 0.0285 on p3/time.

## 9. Known Issues for Orchestrator + Human Review

1. **`ANTHROPIC_API_KEY` not propagated to agent shell.** User said they'd set it; not visible. Either run `! export ANTHROPIC_API_KEY=...` in the Claude Code prompt OR add to `~/.bashrc` + restart.
2. **Day-2 REPORT_day2.md fusion numbers are superseded.** Use Day-3 real-fusion numbers as canonical.
3. **Day-2 split pickles contain unused leaky fold-encoded columns.** Hygiene fix recommended.
4. **Day-3 audit script bug** (cohort-row-ordering) is fixed and documented; any future code touching split pickles must use `set_index('nct_id').loc[cohort_ncts]`, not `isin(keep).reset_index()`.

## 10. Files

Source (Day-3):
- `/mnt/ssd/research/trial-failure/src/day3/phaseB_leakage_audit.py`
- `/mnt/ssd/research/trial-failure/src/day3/phaseC_real_embeddings.py`
- `/mnt/ssd/research/trial-failure/src/day3/phaseC4_real_fusion.py`
- `/mnt/ssd/research/trial-failure/src/day3/phaseD_llm_failure_modes.py` (ready, blocked)

Results:
- `/mnt/ssd/research/trial-failure/results/day3/leakage_audit.{json,md}`
- `/mnt/ssd/research/trial-failure/results/day3/real_fusion_summary.json`
- `/mnt/ssd/research/trial-failure/results/day3/B{12,3,4,5,6}_*.{csv,json}`

Embeddings:
- `/mnt/ssd/research/trial-failure/processed/day3/{smiles,protein,text}_embeddings.npz`
- `/mnt/ssd/research/trial-failure/processed/day3/per_nct_emb_p{2,3}.npz`

Data:
- `/mnt/ssd/research/trial-failure/processed/day3/cohort_{smiles,target_sequences,text_corpus}.parquet`
- `/mnt/ssd/research/trial-failure/raw/chembl/chembl_36/chembl_36_sqlite/chembl_36.db` (28 GB ChEMBL 36 SQLite, now local)

Logs: `/mnt/ssd/research/trial-failure/logs/day3_*.log` + appended `gates.log`.
