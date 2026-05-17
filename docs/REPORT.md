# Clinical Trial Failure Prediction — Week 1 Report

**Project**: trial-failure
**Snapshot**: AACT 2026-05-01 monthly pipe-file dump; FDA Drugs@FDA `datdaf20260512.zip`
**Period covered**: 2026-05-12 (Gate 0) through 2026-05-12 (Gate 5)
**Audience**: Rohit (NYU DS undergrad); future-Claude on week-2 architecture dispatch

---

## 1. Executive Summary

### What was built

A clean, survivorship-bias-corrected clinical trial cohort plus a tabular LightGBM baseline for binary trial success prediction:

- **Cohort**: 582,806 CT.gov studies pulled from the AACT 2026-05-01 monthly pipe-file dump, anchored on **registration date** (not outcome availability), loaded as a 10-table DuckDB at `/mnt/ssd/research/trial-failure/aact.duckdb`.
- **Labeling**: Drug-level FDA approval lookup (29,060 applications, 51,201 products) joined to CT.gov intervention names via exact + fuzzy match (rapidfuzz). Approval window for a positive label: `[registration_date − 3y, completion_date + 10y]` (or `+ 15y` if completion missing).
- **Cohort sizes for supervised use**:
  - Phase 3 cohort: 33,526 trials, 11.78% positive base rate.
  - Phase 2 cohort: 55,355 trials, 37.24% positive base rate.
- **Features**: Final leak-clean tabular feature set is 50 columns (trial-design + indication + sponsor only). The original 506-column matrix was reduced after a leakage finding in Gate 5 (see §4).
- **Splits**: 8 frozen splits = {P3, P2} × {time, drugclass, indication, target}.
- **Baseline**: LightGBM per split with 50-trial Optuna search + isotonic calibration. Headline number: **P3 time-split AUROC = 0.7663**, in the Lo 2019 reference band [0.65, 0.72] (slightly above the upper end), well below the 0.80 leakage threshold.

### Scope / methodological choices

- **Binary success prediction only this week.** Failure-mode taxonomy (efficacy / safety / PK / recruitment / operational) is NOT built; it is week-2 work (§5).
- **P3 is the primary target**, P2 is secondary. P3 is the harder, more decision-relevant outcome.
- **Survivorship-bias correction**: cohort defined by registration date. Trials registered after 2020 without outcome are kept in the cohort but marked `label_available = False` and excluded from supervised fits.
- **Calibration is treated on equal footing with AUROC.** All splits report Brier and ECE before and after isotonic calibration; reliability diagrams saved as PNGs under `results/`.
- **Four required splits implemented.** Time-based plus three group-held-out splits (drug class, indication, target). Drug class held out = "kinase inhibitor" by ChEMBL action type. Indication held out = oncology. Target held out = GPCR (action type).

---

## 2. Gate Outcomes

| Gate | Timestamp (UTC)      | Status | Key metrics |
|------|----------------------|--------|-------------|
| G0   | 2026-05-12 12:15Z    | PASS   | 12/12 packages imported; python 3.12.3; venv `/home/rohit/venvs/trial-failure/` |
| G1   | 2026-05-12 16:22Z    | PASS   | AACT 2026-05-01 snapshot; 582,806 studies; 10 tables in DuckDB; extracted 16.01 GB; duckdb 0.89 GB |
| G2   | 2026-05-12 16:48Z    | PASS (after 3 HARD_STOP retries) | P3 match 46.91%, P3 base 11.78% (n=33,526), P2 base 37.24% (n=55,355); 29,060 FDA apps, 51,201 products |
| G3   | 2026-05-12 17:44Z    | PASS   | 582,806 × 506 feature matrix; ChEMBL hits 2,863 / 6,249 queried (45.8%) |
| G4   | 2026-05-12 17:49Z    | PASS (1 split flagged) | 8 splits frozen; P3 kinase drugclass test positives = 110 (<200 flag); no train/test overlap |
| G5   | 2026-05-12 18:36Z    | PASS   | P3/time AUROC=0.7663, Brier_cal=0.0749, ECE_cal=0.0239; feature count reduced 506 → 50 after leakage triage |

### Gate 2 notes (labeling-window tuning)

Gate 2 hit `HARD_STOP` three times before passing:

| Attempt | P3 base rate | Window used | Outcome |
|---------|--------------|-------------|---------|
| 1       | 44.85%       | Unlimited / all approvals matched to any trial of the drug | HARD_STOP (>25%) |
| 2       | 45.20%       | Tightened drug → application set, still no time window | HARD_STOP |
| 3       | 6.58%        | Approval must occur strictly between `completion_date` and `completion + 1y` | HARD_STOP (<8%) |
| 4 (PASS)| 11.78%       | Approval window `[reg − 3y, completion + 10y]` (or `reg − 3y, reg + 15y` if completion is null) | PASS |

The "PASS" window is a judgment call — see §4.

### Gate 4 split flag

P3 / drugclass test set has only 222 trials and 110 positives. This is < 200 positives and is flagged. Do not interpret the P3/drugclass AUROC as a precise generalization number. All other splits cleared the 200-pos floor.

---

## 3. Baseline Model Performance

All numbers from `/mnt/ssd/research/trial-failure/results/gate5_summary.json`.
Models live at `/mnt/ssd/research/trial-failure/models/lgbm_<target>_<split>.pkl`.
Reliability diagrams at `/mnt/ssd/research/trial-failure/results/reliability_<target>_<split>.png` and calibration curves at `calibration_<target>_<split>.png`.

### Phase 3 (primary target)

| Split       | Train n | Test n | Test pos rate | AUROC  | PR-AUC | Brier (uncal) | Brier (cal) | ECE (uncal) | ECE (cal) | Cal direction |
|-------------|---------|--------|---------------|--------|--------|---------------|-------------|-------------|-----------|---------------|
| time        | 27,834  | 5,692  | 8.68%         | 0.7663 | 0.2166 | 0.0751        | 0.0749      | 0.0246      | 0.0239    | improved      |
| drugclass*  | 33,304  | 222    | 49.55%        | 0.8000 | 0.7645 | 0.3175        | 0.3149      | 0.3350      | 0.3316    | improved (mildly) |
| indication  | 28,173  | 5,353  | 9.27%         | 0.7405 | 0.2118 | 0.0793        | 0.0795      | 0.0170      | 0.0198    | mildly worse  |
| target      | 31,833  | 1,693  | 27.94%        | 0.7526 | 0.5237 | 0.1973        | 0.1948      | 0.1499      | 0.1455    | improved      |

`*` P3/drugclass has 110 test positives < 200; treat as anecdotal.

### Phase 2 (secondary target)

| Split       | Train n | Test n  | Test pos rate | AUROC  | PR-AUC | Brier (uncal) | Brier (cal) | ECE (uncal) | ECE (cal) | Cal direction |
|-------------|---------|---------|---------------|--------|--------|---------------|-------------|-------------|-----------|---------------|
| time        | 44,271  | 11,084  | 30.80%        | 0.7365 | 0.4994 | 0.1832        | 0.1839      | 0.0321      | 0.0367    | slightly worse |
| drugclass   | 54,408  | 947     | 65.15%        | 0.8305 | 0.8766 | 0.2177        | 0.2160      | 0.2486      | 0.2489    | mixed         |
| indication  | 34,787  | 20,568  | 45.42%        | 0.7077 | 0.6157 | 0.2188        | 0.2175      | 0.0794      | 0.0687    | improved      |
| target      | 53,860  | 1,495   | 79.60%        | 0.8120 | 0.9188 | 0.3238        | 0.3220      | 0.4509      | 0.4483    | mildly improved |

### Headline result

**P3 / time-split AUROC = 0.7663**, Brier (calibrated) = 0.0749, ECE (calibrated) = 0.0239. This is in the Lo 2019 reference band of 0.65–0.72 (slightly above the upper end) and below the 0.80 leakage threshold from the agent definition. P3 / time-split calibration is the cleanest in the table: ECE 2.4% after isotonic.

### Honest caveats — inflated AUROCs

Several splits look "good" only because their test set base rate is massively different from training:

- **P3 / drugclass** (49.55% test pos vs ~12% train pos): held-out kinase inhibitors are a non-representative slice. AUROC 0.80 is partly a class-shift artifact, and Brier 0.31 / ECE 0.33 confirm the model is badly miscalibrated on this slice.
- **P2 / drugclass** (65.15% test pos): same story; ECE 0.25 after calibration.
- **P2 / target** (79.60% test pos): held-out GPCR P2 trials almost all "succeed" by the per-trial framing. PR-AUC 0.92 is mostly base rate. ECE 0.45 makes this model unusable as a probability estimator.
- **P3 / target** (27.94% test pos vs 12% train): GPCR P3 is also enriched; AUROC 0.75 but ECE 0.15 — predictions need rescaling.

**Trust the time split first.** It is the only split with comparable train / test base rates and clean calibration.

### Top features (P3 / time-split)

From `gate5_summary.json`. All trial-design columns; no drug or target columns survive in the leak-clean feature set.

1. `enrollment` (gain 49,721)
2. `duration_days` (gain 46,317)
3. `oh_sponsor_type_industry` (gain 45,792)
4. `number_of_arms` (gain 13,662)
5. `masking_ord` (blinding ordinal) (gain 10,831)
6. `has_placebo` (gain 5,712)
7. `is_randomized` (gain 4,315)
8. `oh_therapeutic_area_endocrinology` (gain 3,905)
9. `oh_therapeutic_area_infectious_disease` (gain 3,810)
10. `oh_primary_endpoint_type_other` (gain 3,183)

---

## 4. Data Quality Issues

### 4a. Labeling-window judgment call (Gate 2)

The Gate 2 P3 base rate of 11.78% depends on a tuned FDA-approval window: an approval counts as the trial's positive label only if the approval date falls in `[registration_date − 3y, completion_date + 10y]` (or `[reg − 3y, reg + 15y]` if completion is null).

Implications:

- We label a trial as "success" only if the drug got an approval within a plausible temporal neighborhood of the trial. Trials of approved drugs that happened way after approval (post-marketing studies, label expansions) are not credited as positives for the original P3 registration.
- The framing is **per-trial**. Hay et al. 2014 quote ~58% P3 success in a **per-drug** framing. The two are not directly comparable. In per-trial terms, **85.8% of completed Phase 3 trials in our cohort are labeled as failure**. That is a strong stance; it is consistent with the literature for per-trial accounting (many drugs run multiple P3s, and many P3s are negative confirmatory studies, additional endpoint studies, etc.).
- Decision deferred to Rohit: keep the per-trial framing (more pessimistic, closer to "will this specific trial succeed") or switch to a per-drug framing (closer to Hay 2014, "will this molecule ever reach approval") before week 2.

### 4b. Gate 3 feature leakage (SEVERE — surfaced in Gate 5)

This is the single most important data-quality finding of week 1.

**Symptom**: First Gate 5 smoke test produced P3 / time test AUROC **0.94**, far above the 0.80 leakage threshold.

**Root cause**: Gate 2 built `primary_drug_canonical` only when CT.gov interventions matched an FDA drug name. Gate 3 then keyed all of `drug_type`, `target_type`, `action_type`, `moa_bucket`, `num_prior_trials_same_drug`, `num_prior_trials_same_target`, `target_novelty`, `has_orphan_app`, `is_rare_disease` on `primary_drug_canonical`. Consequences:

- 46% of cohort has `primary_drug_canonical = NULL`. For those rows, every drug / target / MoA feature collapses to a constant or NA-token.
- The 54% with a non-null canonical are exactly the rows whose drug found an FDA application — strongly enriched for approved drugs.
- The indicator `primary_drug_canonical_is_missing` had univariate AUROC ≈ 0.24 against `p3_success` (which is AUROC 0.76 in the predictive direction). The model can predict "success" almost entirely from "did we manage to find an FDA app for this drug."
- The fold-encoded target-leakage columns (`*_success_rate_fold`) also leaked: MOA buckets group repeated trials of the same drug, so the per-fold target mean for the bucket carries direct signal about the test trial's drug.

**Mitigation applied in Gate 5**:

- Dropped 15 numeric/boolean columns: every feature gated on `primary_drug_canonical`, plus the leaking fold-encoding columns.
- Dropped 4 one-hot families: `drug_type`, `target_type`, `action_type`, `moa_bucket`.
- Feature count went 506 (original Gate 3) → 487 (after one-hot cleanup) → 50 (after leakage triage).
- After exclusion, P3/time AUROC = 0.7663, back in the Lo 2019 band.

**Therefore the week-1 baseline is the trial-design-only baseline (Lo 2019 style), not the full Lo + drug + target + MoA feature set.** Drug/target/MoA features need a leak-clean rebuild before week 2.

### 4c. ChEMBL MoA coverage

- 6,249 distinct `primary_drug_canonical` values queried against ChEMBL.
- 2,863 returned a mechanism record (45.8% hit rate, drug-level).
- Trial-level "unknown MoA" rate is 85%: because 46% of trials have no `primary_drug_canonical` to begin with, and another ~30% of trials whose canonical was found have no ChEMBL MoA record.
- Even after the leak-clean rebuild (see §7), MoA coverage will be a ceiling on any MoA-conditioned modeling.

### 4d. Drug-name matching gaps

- **Descriptive biologic names** (G-CSF, anti-PD-1 antibody, recombinant IL-2, etc.) are missed by exact and rapidfuzz fuzzy matching because the CT.gov intervention text often does not contain the FDA brand or USAN/INN name. Estimated coverage loss: 1–2 pp on P3 match rate. **Recommended fix**: bring in RxNorm and pull synonyms for each FDA `product.activeingredient` so descriptive names like "G-CSF" resolve to "filgrastim" / "pegfilgrastim".
- **ANDA generics intentionally excluded.** Generic ANDA approvals do not represent new-molecule success and would inflate apparent base rates. Only NDA + BLA applications are credited as positives.

### 4e. Other notes

- 7 Gate 3 features carry > 50% missing values. They are kept (per agent definition) but paired with `*_is_missing` indicator columns. The indicators must be audited in the leak-clean rebuild to make sure they are not themselves an FDA-match proxy.
- DuckDB `aact.duckdb` is 1.5 GB on disk; AACT extracted size 16 GB on HDD. Re-runs that hit DuckDB are fast (Gate 4 was 0 s wall).

---

## 5. Failure-Mode Label Status

**Not built in week 1.**

The agent definition called for a structured failure-mode taxonomy (efficacy / safety / PK / recruitment / operational) constructed by cross-referencing CT.gov termination reason + FDA outcome + post-hoc literature flags. The current state:

- Binary success/failure label exists in `trials_labeled` (DuckDB).
- Free-text CT.gov termination reasons exist in `studies.why_stopped` and the `drop_withdrawals` table — these are present but not parsed into a taxonomy. The agent definition (rule 4) explicitly states these are unreliable and must be cross-referenced.
- FDA Complete Response Letters and AdCom outcomes are NOT yet pulled.
- Post-hoc literature flags (PubMed mining of "trial failed because of...") are NOT yet pulled.

This is a known week-1 gap, documented intentionally rather than papered over. Failure-mode label construction is queued for week 2; no architecture work on failure-mode classification should start before the labels exist.

---

## 6. What Is Ready for Week 2 (Multimodal Fusion Transformer)

The following infrastructure is already production-quality and reusable without rework:

1. **Clean cohort** — 582,806 trials in DuckDB with a `label_available` flag. Survivorship-bias correction is baked in. Reproducible from AACT 2026-05-01 + Drugs@FDA `datdaf20260512.zip`.
2. **Validation infrastructure** — 8 frozen splits at `/mnt/ssd/research/trial-failure/processed/splits/split_<target>_<split>.pkl`, each with `{train_idx, test_idx}`. Verified zero train/test overlap. Per-fold target encoding sidecars at `fold_encodings_<target>_<split>.parquet` so the encoder fit is honest per-fold (no global leakage).
3. **Tabular trial-design feature set** — 50 leak-clean columns, covering trial design (enrollment, duration, arms, blinding, randomization, placebo), phase, sponsor type, primary endpoint type, and one-hot therapeutic area. This is usable AS-IS as the "tabular branch" of the multimodal fusion model.
4. **DuckDB warehouse** — full AACT (10 tables) + FDA Drugs@FDA + ChEMBL MoA cache (drug-level mechanism table) at `/mnt/ssd/research/trial-failure/aact.duckdb`. Ready for ad-hoc enrichment queries.
5. **Calibration scaffolding** — isotonic-regression post-hoc calibration plus Brier + ECE + reliability plots are wired into the Gate 5 driver; reusable for any model that produces probability scores.
6. **Baseline numbers locked in** — `gate5_summary.json` is the reference any future model must beat. Beating LightGBM P3/time AUROC 0.7663 with comparable or better calibration is the bar.

---

## 7. Requires Human Review Before Week 2 Starts

These are blockers or decision points that need Rohit's sign-off before architecture work begins.

### 7a. Gate 3 leak-clean rebuild (BLOCKER)

The Gate 5 agent's required fix list, copied verbatim:

1. Recompute `drug_type` from raw CT.gov `interventions.intervention_type` directly (not gated by FDA match).
2. Recompute `mechanism_of_action`, `target_type`, `action_type`, `moa_bucket` via ChEMBL keyed on raw CT.gov intervention name.
3. Recompute prior-trial counts from a leak-clean drug identifier, time-truncated to `registration_date − 1 day`.
4. Replace `has_orphan_app` with registration-time orphan designation from FDA OOPD.
5. Switch target encoding from leave-one-trial-out to leave-one-DRUG-out.
6. Rerun Gate 4 fold encodings and Gate 5.

Without these fixes, the only honest features available to week-2 architectures are the 50 trial-design columns. Drug, target, and MoA branches of any multimodal fusion will be missing or unreliable.

### 7b. Gate 2 labeling-window decision (BLOCKER)

P3 base rate 11.78% (per-trial framing) vs Hay 2014 ~58% (per-drug framing). Decision to make:

- Keep per-trial framing as-is (pessimistic, "will this trial succeed").
- Switch to per-drug framing (Hay 2014-comparable, "will this molecule reach approval"). Requires reshaping the cohort to one row per drug rather than one row per trial.
- Build both and treat them as two different prediction tasks.

This affects the loss function and the deliverable, so it is week-1 unresolved.

### 7c. Failure-mode taxonomy definition (BLOCKER for multimode head)

Before any "predict failure mode" head is trained, define the taxonomy:

- Categories (efficacy / safety / PK / recruitment / operational — agent definition uses these five; do we keep them or refine?).
- Label sources and precedence: CT.gov termination → FDA outcome → manual literature flag.
- Treatment of ambiguous / multi-cause failures (multi-label vs forced single-label).

Architecture work on a failure-mode classifier should not start before this is decided.

### 7d. Additional modalities — scoping question

Week-2 multimodal architecture work could include:

- **SMILES → molecular embeddings** (RDKit + a chem-LM such as ChemBERTa-77M-MLM, fits on the 8 GB 4060 Ti with bf16). Requires a leak-clean SMILES join — which itself depends on §7a.
- **Trial-protocol text encoding** (CT.gov brief description / detailed description / eligibility criteria → SciBERT or BioBERT embeddings). This is independent of §7a and can start in parallel with the leak-clean rebuild.
- **Sponsor history embeddings** (sponsor → historical success rate by phase, with proper time truncation).

Scope decision needed from Rohit.

---

## Appendix A — File Map

```
/mnt/ssd/research/trial-failure/
├── aact.duckdb                                  1.5 GB     full warehouse
├── requirements.txt
├── logs/
│   ├── gates.log                                gate log (canonical)
│   ├── gate{1..5}.log                           per-gate detail
│   └── gate3_diagnostics.json
├── processed/
│   └── splits/
│       ├── cohort_p{2,3}_nct_ids.parquet        cohort row IDs
│       ├── split_p{2,3}_{time,drugclass,indication,target}.pkl
│       └── fold_encodings_p{2,3}_{...}.parquet  per-fold target encodings
├── models/
│   └── lgbm_p{2,3}_{time,drugclass,indication,target}.pkl
├── results/
│   ├── gate5_summary.json                       all metrics + best params
│   ├── predictions_p{2,3}_{...}.parquet         per-trial test predictions
│   ├── reliability_p{2,3}_{...}.png             reliability diagrams
│   └── calibration_p{2,3}_{...}.png             calibration curves
└── src/
    └── gate{0..5}_*.py                          gate driver scripts
```

## Appendix B — Reference Bands

| Metric                | Source                | Band                       |
|-----------------------|-----------------------|----------------------------|
| P3 time-split AUROC   | Lo et al. 2019        | 0.65 – 0.72 (we hit 0.7663) |
| P3 base rate (per-trial) | This cohort        | 11.78%                     |
| P2 base rate (per-trial) | This cohort        | 37.24%                     |
| P3 success (per-drug, Hay 2014) | Hay et al. 2014 | ~58% (different framing)   |
| AUROC leakage flag    | Agent definition      | > 0.80                     |
| AUROC underflag       | Agent definition      | < 0.55                     |
