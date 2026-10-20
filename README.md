# Clinical trial outcome prediction, with the leakage audit attached

Phase 2 and Phase 3 trial outcome prediction over AACT (the full ClinicalTrials.gov database, 582,806 trials) joined to Drugs@FDA, ChEMBL and Open Targets. A LightGBM model on 108 leak-clean tabular features gets AUROC 0.8718 on the Phase-3 time-held-out split, 0.8603 after the audit retrain; a multimodal fusion over ChemBERTa, ESM2 and SciBERT embeddings adds +0.007 there and +0.034 on the indication holdout.

The part I would hand to someone else is the leakage audit in `src/day3/phaseB_leakage_audit.py`. An early version of this model scored 0.87 for the wrong reason: whether a compound ever appears in Drugs@FDA is close to the label by construction, since drugs in that database are mostly drugs that finished Phase 3. After dropping those features the number came back anyway, so the audit exists to decide whether to believe it.

Project page: https://rohitravi.com/projects/clinical-trial-failure-prediction

## The audit

Everything below is on the Phase-3 success / time split (`results/day3/leakage_audit.json`, `results/day3/B*.csv|json`).

- **Shuffled-label control.** Train on permuted train labels, evaluate on the real test labels. If any feature encodes the label structurally, this still beats chance. It scored AUROC 0.4818.
- **Per-feature univariate train and test AUROC.** No feature sits outside [0.30, 0.70] on test. The strongest is `moa_bucket_unknown` at 0.695 (train 0.653): drugs with no documented mechanism fail more often, which is plausible rather than circular. The largest train-to-test shift is `duration_days` (+0.072), and terminated vs completed trials have near-identical mean durations (~1,050 vs ~1,140 days), so it is a size effect, not a label proxy.
- **Audit retrain.** Replaying the Day-2 feature set under the audit gives 0.8603 against 0.8684 calibrated / 0.8718 uncalibrated for the original, i.e. the headline survives.
- **Row-alignment bug found by the audit.** The first audit run reported 0.91+, which was impossible. The split pickles store positional indices into `cohort_<task>_nct_ids.parquet` row order, not into `features_v2.parquet` row order, so `feats[isin(keep)].reset_index()` silently misaligns rows. The fix is `set_index('nct_id').loc[cohort_ncts]`. Anyone reusing this split convention can hit the same thing.
- **One check that did not come back clean.** The indication and drug-class success-rate columns are target-encoded from training trials. Recomputing them for 10 sampled test trials while excluding every trial of the same drug moved the drug-class rate by up to 0.62 (`results/day3/B6_loo_encoding.json`). Those two columns are stored in the split files but absent from the 108 features the shipped model uses, so they do not affect the numbers here; re-encoding them leave-one-drug-out before any use is the next fix.

## Model results

Leak-clean LightGBM, 108 features (`results/day2/lgbm_v2_summary.json`):

| Target / split | AUROC | PR-AUC | ECE | n train | n test |
|---|---:|---:|---:|---:|---:|
| P3 / time | 0.8718 | 0.452 | 0.026 | 27,834 | 5,692 |
| P3 / indication | 0.8068 | 0.324 | 0.027 | 28,173 | 5,353 |
| P3 / target | 0.8149 | 0.605 | 0.109 | 31,833 | 1,693 |
| P3 / drug class | 0.7945 | 0.754 | 0.241 | 33,304 | 222 |
| P2 / time | 0.8765 | 0.753 | 0.026 | 44,271 | 11,084 |
| P2 / indication | 0.8762 | 0.839 | 0.048 | 34,787 | 20,568 |
| P2 / target | 0.8254 | 0.935 | 0.157 | 53,860 | 1,495 |
| P2 / drug class | 0.8209 | 0.862 | 0.082 | 54,408 | 947 |

The Day-1 baseline on 50 features was 0.7663 on P3 / time (`results/gate5_summary.json`); the lift comes from richer ChEMBL attribute encodings and forward-only prior-trial counts, not from the dropped FDA-match features.

Multimodal fusion, trained in the same pipeline so the comparison is like for like (`results/day3/real_fusion_summary.json`): ChemBERTa-77M over canonical SMILES, ESM2-150M over target protein sequences, SciBERT over title and brief summary, concatenated to a 2,289-dim input.

| Target / split | tabular only | + deep modalities | delta |
|---|---:|---:|---:|
| P3 / time | 0.8672 | 0.8746 | +0.007 |
| P3 / indication | 0.8038 | 0.8376 | +0.034 |
| P3 / target | 0.8137 | 0.8257 | +0.012 |
| P3 / drug class | 0.7939 | 0.7869 | −0.007 |
| P2 / time | 0.8742 | 0.8806 | +0.006 |
| P2 / indication | 0.8770 | 0.8852 | +0.008 |
| P2 / target | 0.8250 | 0.8064 | −0.019 |
| P2 / drug class | 0.8250 | 0.8221 | −0.003 |

Fusion helps on five of eight combinations and hurts on three. The pattern that makes sense is the indication holdout, where the test therapeutic areas never appear in training: one-hot indication features are zero by construction on held-out classes, while text and protein embeddings still carry information. That is where the gain is largest.

For reference, published work on comparable cohorts (HINT, NeurIPS 2022; Lo 2023; Hong 2020) reports AUROC in the 0.70-0.85 range, so these are competitive numbers rather than surprising ones.

## Data and labels

- **AACT** 2026-05-01 snapshot, loaded into DuckDB (`src/gate1_aact.py`). Labels come from trial status and the follow-on-trial / approval record rather than from any FDA-match indicator.
- **Drugs@FDA** for approval records (`src/gate2_fda.py`); 46.91% of Phase-3 interventions matched a compound.
- **ChEMBL** for drug type, target type, action type and mechanism buckets, matched through preferred names and synonyms; **Open Targets** and **UniProt** for target annotation and sequences (`src/day2/phaseA_downloads.py`).
- Eight frozen splits (`src/gate4_splits.py`): time, drug class, indication and target holdouts for each of Phase 2 and Phase 3. Orphan designation is evaluated at registration date, and prior-trial counts are computed forward-only, so nothing after a trial's registration enters its features.

## Repo layout

```
src/gate0_setup.py .. gate5_baseline.py   environment, AACT load, FDA join, feature build,
                                          split construction, Day-1 baseline
src/day2/                                 extra downloads, 108-feature leak-clean rebuild,
                                          training, per-drug cohort, failure-mode parsing, fusion
src/day3/phaseB_leakage_audit.py          the audit
src/day3/phaseC_real_embeddings.py        ChemBERTa / ESM2 / SciBERT embedding extraction
src/day3/phaseC4_real_fusion.py           fusion training and the tabular-only control
src/day3/phaseD_llm_failure_modes.py      failure-mode labelling of why_stopped text (not run)
results/                                  every metrics file quoted above
models/                                   8 LightGBM models (P2/P3 x 4 splits)
```

Scripts read and write under `work/` (`work/raw`, `work/processed`, `work/models`, `work/results`), which is gitignored; point it at wherever the data lives.

## Running it

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
mkdir -p work                       # or: ln -s /path/to/data work

python src/gate1_aact.py            # AACT -> DuckDB
python src/gate2_fda.py
python src/gate3_features.py
python src/gate4_splits.py
python src/gate5_baseline.py        # Day-1 50-feature baseline

python src/day2/phaseA_downloads.py
python src/day2/phaseB_features_v2.py
python src/day2/phaseB_train.py     # 108-feature leak-clean LightGBM

python src/day3/phaseB_leakage_audit.py
python src/day3/phaseC_real_embeddings.py   # GPU
python src/day3/phaseC4_real_fusion.py
```

## Not finished

`src/day3/phaseD_llm_failure_modes.py` classifies the 33,042 unique `why_stopped` strings from terminated trials into efficacy / safety / PK / recruitment / operational buckets through the Anthropic Batches API at temperature 0. It is written and resumable but has not been run, so no failure-mode results are in this repo. Predicting *why* a trial stops, rather than whether it succeeds, is the direction I would take this next.

## License

MIT, see [LICENSE](LICENSE).
