# Loan Performance Intelligence Engine

**Intain Campus FinTech Challenge 2026 — AI Track, Round 2**

An end-to-end loan performance intelligence system: data quality profiling, calibrated
delinquency / default / prepayment prediction, survival and transition modelling, anomaly and
exception detection, macro stress simulation, SHAP explanation, and a grounded LLM reviewer
copilot.

**The core predictive work is entirely non-LLM.** Every probability, score, rate and ranked
driver traces to a fitted LightGBM, scikit-learn, or lifelines estimator. A language model is
used in exactly one place — narrating model output for human reviewers — and that boundary is
enforced by a test that parses the AST of every modelling module and fails if it can even
*import* an LLM client.

---

## Data

The panel is built from the **Freddie Mac Single-Family Loan-Level Dataset (SFLLD)**, vintages
2019-2023: 250,000 loans and 10,482,492 monthly performance records, sampled at loan level to
16,000 loans / ~670,000 monthly rows.

The raw files are **not committed** — SFLLD is licence-gated, and redistributing it would
breach the source terms that section 13 of the problem statement lists as a disqualification
condition. See **[`dataset/download_sflld.md`](dataset/download_sflld.md)** to obtain them.

The pack is a **hybrid, and the model card says so in section 2**: the loan panel, its
outcomes and the macro history are real; the exception / reconciliation / document-status
layer is fabricated, because SFLLD has no second source, no ingestion timestamps and no
document data. The synthetic generator remains in the repository and produces the identical
33-column panel contract, so the pipeline runs either way.

> **`next_12m_default_flag` is a 90+ DPD proxy.** Realised credit events occur on 14 of the
> 16,000 sampled loans (~0.09%) — not modellable. Every "default" figure in this repository
> refers to that proxy. Stated in full in the model card and the data intelligence report.

---

## Quick start

```bash
pip install -r requirements.txt
```

**With the real SFLLD files in `dataset/`** (see the link above):

```bash
python -m src.data.build_from_sflld   # build the pack from real data  (~1 min)
python -m src.data.macro_real         # real FRED macro history + scenarios
python -m src.pipeline --skip-data    # Tasks 1-8 + submission.csv
```

**Without them**, the synthetic generator produces the same contract:

```bash
python -m src.pipeline                # generates the synthetic pack, then runs everything
```

Either path writes `submission/submission.csv`. Expect roughly 10-20 minutes on the synthetic
pack, longer on the real panel.

```bash
python -m src.pipeline --skip-data          # reuse the existing data pack and feature cache
python -m src.pipeline --stage models       # run up to a stage and stop
python -m src.pipeline --n-loans 400        # smaller synthetic pack for a fast smoke run
python -m pytest tests -q                   # 31 tests, including the leakage guards
```

The feature cache invalidates itself when the raw pack is newer, so switching between the two
sources cannot silently reuse the other one's frame.

### Optional: enable the live LLM copilot

```bash
export ANTHROPIC_API_KEY=sk-ant-...     # or: ant auth login
python -m src.copilot.run_copilot
```

Without a credential the copilot runs in `offline_template` mode. That mode is deterministic
string formatting, is labelled as such everywhere it appears, and is never presented as model
output. See "Known gaps" below.

---

## What each stage does

| Stage | Module | Output |
|---|---|---|
| Data | `src/data/build_from_sflld.py` (real) or `build_dataset.py` (synthetic) | Loan panel, servicer feed, macro history, scenarios, data dictionary, ground-truth defect log |
| Profile | `src/data/report_data_intelligence.py` | `reports/data_intelligence_report.md` + 7 CSV extracts |
| Models | `src/models/run_performance.py` | `reports/model_performance_report.md`, fitted models, metrics, leakage probe |
| Survival | `src/models/run_survival.py` | `reports/survival_report.md`, KM / CIF curves, Cox coefficients, transition matrix |
| Anomaly | `src/models/run_anomaly.py` | `reports/anomaly_report.md`, `reports/anomaly_review_queue.csv` |
| Scenarios | `src/scenarios/run_scenarios.py` | `reports/scenario_report.md`, segment impacts, Markov stress paths |
| Explain | `src/explain/run_explain.py` | `reports/explainability_report.md`, global + local SHAP |
| Copilot | `src/copilot/run_copilot.py` | `reports/copilot_report.md`, `submission/llm_prompt_log.jsonl` |
| Submission | `src/submission.py` | `submission/submission.csv` |
| Model card | `src/report_model_card.py` | `submission/MODEL_CARD.md`, generated from the report artefacts |

---

## Repository layout

```
data/
  raw/            loan_panel.csv, servicer_updates.csv, macro_history.csv,
                  macro_scenarios.csv, data_dictionary.csv, ground_truth_defect_log.csv
  processed/      cached model frame
  samples/        small committed samples of the raw files
src/
  data/           generation, messiness injection, loading, reconciliation, validation, profiling
  features/       leakage-safe feature engineering, dataset assembly
  models/         splits, metrics, prediction, survival, anomaly
  scenarios/      stress and scenario simulation
  explain/        SHAP global/local, uncertainty, error analysis
  copilot/        grounding packs, Anthropic client, output validators
  pipeline.py     end-to-end orchestration
  submission.py   submission.csv builder
reports/          all generated reports and CSV extracts
submission/       submission.csv, MODEL_CARD.md, AI_DEVELOPMENT_LOG.md,
                  llm_prompt_log.jsonl, run_manifest.json
tests/            31 tests, leakage guards and LLM-boundary enforcement
```

---

## Swapping the data source

Both loaders write the same 33-column `data/raw/loan_panel.csv` contract, and nothing
downstream reads either loader directly:

| Source | Builder | Notes |
|---|---|---|
| Real SFLLD | `src/data/build_from_sflld.py` | Needs `dataset/`; layout asserted at load |
| Synthetic | `src/data/build_dataset.py` | No external files needed |
| Your own panel | — | Match `data/raw/data_dictionary.csv` and run `--skip-data` |

Split boundaries are derived from the data's own month range, so a different panel window
needs no code change. `servicer_updates.csv` is optional — the pipeline degrades gracefully
if the second feed is absent.

**On the SFLLD layout.** The sample files carry **31 origination and 35 performance columns**,
not the 32/32 in Freddie Mac's published `file_layout.xlsx` and January 2026 User Guide.
`Servicer Name` is absent from the origination file; the performance file appends
`MI Cancellation Indicator`, `Servicer Name` and a filler column. The mapping was verified
empirically across all five vintages — the evidence is in `src/data/sflld.py` — and
`sflld.verify_layout()` refuses to load anything that deviates rather than mis-mapping
silently.

---

## The four things worth looking at

**1. The validation design (`src/models/splits.py`).** Naive splitting inflates test ROC-AUC
here by **+0.10 to +0.32**. Three problems are handled separately: labels that are
unobservable near the panel end, training rows whose outcome window overlaps the test window,
and right-censoring. The split-sensitivity probe in
`reports/model_performance_report.md` quantifies exactly what each control is worth.

**2. The honest baseline comparison.** LightGBM does not beat a nine-feature logistic
regression on ranking for the delinquency and default targets — it wins on *calibration*, by
2-4x on Brier score. That is stated plainly rather than presented as a clean sweep, because
the dominant delinquency signals are near-monotone in the log-odds and a linear model is hard
to beat there.

**3. The scenario identification problem (`reports/scenario_report.md`).** The model-repricing
engine produces a near-zero adverse-credit impact and a high-prepayment scenario that *raises*
projected default. That is not a bug to tune away: with one realised macro path there is no
cross-sectional variation to identify a macro credit effect from. A second engine — a
macro-conditioned Markov chain — carries the credit stress properly, and the division of
labour is stated explicitly.

**4. The LLM boundary (`src/copilot/`).** The copilot never sees the dataframe or the models.
It receives a JSON grounding pack of already-computed figures. A grounding validator extracts
every number from its output and blocks anything that does not match the pack — including
values helpfully rescaled by 100 or rounded. A six-case self-test proves the validator
actually bites.

---

## Known gaps

- **No live LLM transcripts.** No Anthropic credential was available in the build
  environment, so the adversarial probes in Task 7 ran against the deterministic offline
  template, which cannot hallucinate and therefore cannot demonstrate the failure modes the
  probes are designed to catch. Rather than fabricate transcripts, that section states what is
  missing and how to produce it (`export ANTHROPIC_API_KEY` and re-run the copilot stage).
- **No loss-given-default model**, so nothing converts default probability into a dollar loss.
- **No fair-lending testing.** `state` and `servicer_name` are model inputs and would need
  disparate-impact analysis before any production use.
- Metrics are single-seed point estimates; no repeated-run variance is reported.

---

## Documents

| Document | What it covers |
|---|---|
| `submission/MODEL_CARD.md` | Objective, data, features, validation, metrics, leakage controls, failure modes |
| `submission/AI_DEVELOPMENT_LOG.md` | Tooling, accepted and rejected AI output with reasons, review process, lessons |
| `reports/data_intelligence_report.md` | Profiling, missingness mechanism, validation rules, drift, DQ scoring |
| `reports/model_performance_report.md` | Split design, baseline comparison, calibration, leakage probe, backtest |
| `reports/survival_report.md` | Censoring treatment, KM curves, competing risks, Cox, Markov |
| `reports/anomaly_report.md` | Anomaly scoring, exception models, driver attribution, reviewer queue |
| `reports/scenario_report.md` | Scenario assumptions, dual-engine projections, segment impacts, drivers |
| `reports/explainability_report.md` | Global and local SHAP, uncertainty, false positive / negative analysis |
| `reports/copilot_report.md` | LLM boundary, grounding validator, adversarial probes |
| `reports/demo_video_script.md` | Five-minute demo script and storyboard, mapped to PS section 14 |
| `dataset/download_sflld.md` | How to obtain the licence-gated raw data and refresh the macro series |
| `PROGRESS.md` | Build status, decisions taken, open gaps |

---

## Environment

Python 3.11. Core dependencies: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `shap`,
`lifelines`, `scipy`, `anthropic`. Fixed seed `20260828` throughout.
