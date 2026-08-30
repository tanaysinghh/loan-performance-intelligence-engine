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

**No external test file was issued.** Problem statement section 6 anticipates an organiser-
supplied unlabeled `loan_monthly_performance_test.csv` for final scoring; none was released, so
this project builds its own data pipeline to fill that gap. `submission/submission.csv`
therefore contains held-out predictions on the project's own **time-aware split** — the purged
out-of-time test window defined in `src/models/splits.py` and reported per target in
`reports/split_summary.csv` — not scores against an external file.

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
python -m pytest tests -q                   # 40 tests, including the leakage guards
```

The feature cache invalidates itself when the raw pack is newer, so switching between the two
sources cannot silently reuse the other one's frame.

### The LLM copilot

The copilot calls **Google Gemini** (`gemini-3.5-flash-lite`) and runs live in the committed
artefacts. Gemini was chosen deliberately on cost and availability rather than as a fallback:
the model is free-tier eligible, so every figure in `reports/copilot_report.md` reproduces for
anyone holding a free Google AI Studio key.

```bash
export GEMINI_API_KEY=...          # free key from https://aistudio.google.com/apikey
python -m src.copilot.run_copilot
```

Without a credential the copilot degrades to `offline_template` mode — deterministic string
formatting, labelled as such everywhere it appears, and never presented as model output. The
report states its execution mode in its first line, so which one produced a given run is never
ambiguous.

**Free-tier note.** `gemini-3.6-flash` writes better prose but allows only 20 requests per
*day*, and one Task 7 run issues 15-20 calls, so it is effectively single-shot. The lite model
is the default for that reason.

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
  copilot/        grounding packs, Gemini client, output validators, LaTeX ablation
  pipeline.py     end-to-end orchestration
  submission.py   submission.csv builder
reports/          all generated reports and CSV extracts
submission/       submission.csv, MODEL_CARD.md, AI_DEVELOPMENT_LOG.md,
                  SUBMISSION_FORMAT.md, llm_prompt_log.jsonl,
                  llm_prompt_log_archive.jsonl, run_manifest.json
tests/            40 tests, leakage guards and LLM-boundary enforcement
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

**1. The validation design (`src/models/splits.py`).** A naive random row split inflates test
ROC-AUC on the two twelve-month targets to the point of absurdity — **0.9988 on the default
proxy and 0.9831 on prepayment, against 0.9207 and 0.6259 under the purged split** (+0.08 and
+0.36). The three-month delinquency target barely moves (−0.01), which is the useful part: the
inflation is not uniform, so a single well-behaved target proves nothing. Three problems are
handled separately: labels that are unobservable near the panel end, training rows whose
outcome window overlaps the test window, and right-censoring. `reports/leakage_probe.csv` also
runs a **loan-disjoint** split as a control, and it lands within noise of the reported
numbers.

**2. The honest baseline comparison.** LightGBM does not beat a nine-feature logistic
regression on ranking for the delinquency and default targets — it wins on *calibration*, by
**2x to 12x on Brier score** (0.0091 vs 0.1089 on the default proxy; 0.1367 vs 0.2789 on
prepayment). That is stated plainly rather than presented as a clean sweep, because
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
values helpfully rescaled by 100 or rounded. A second control, the usefulness check, blocks
output that is true but points the reviewer at a field the pack already reports as clean. A
twelve-case self-test, run against a fixed pack so its verdicts cannot drift with the data,
proves both actually bite.

Running it live caught real failures on both sides. Gemini dropped a decimal place restating a
small probability — `0.046` where the pack said `0.0046` — which the validator blocked and the
correction round-trip fixed. It also caught the validator itself flagging *correct* output six
different ways, each now fixed at source and pinned by a self-test case. Both halves are in
`reports/copilot_report.md` section 5, which separates genuine model failures from validator
false positives rather than counting every block against the model.

---

## Known gaps

- **No five-minute demo video yet.** `reports/demo_video_script.md` is complete and mapped
  to the fifteen beats of problem statement section 14, but the recording itself is outstanding.
- **No loss-given-default model**, so nothing converts default probability into a dollar loss.
- **No fair-lending testing.** `state` and `servicer_name` are model inputs and would need
  disparate-impact analysis before any production use.
- Metrics are single-seed point estimates; no repeated-run variance is reported.
- **Copilot output varies run to run.** Gemini is sampled, so re-running may block different
  outputs, or none. The report names which run it is showing, and earlier captured failures are
  retained in `submission/llm_prompt_log_archive.jsonl`.
- **One ablation came out negative and is reported as negative.** The system-prompt rule
  forbidding LaTeX could not be shown to be what stopped the model emitting it
  (`src/copilot/ablation_latex.py`, 0 of 3 in both arms). Detection is the load-bearing
  control, not the prompt.

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
| `reports/copilot_report.md` | LLM boundary, grounding + usefulness validators, adversarial probes, captured failures and corrections |
| `reports/demo_video_script.md` | Five-minute demo script and storyboard, mapped to PS section 14 |
| `dataset/download_sflld.md` | How to obtain the licence-gated raw data and refresh the macro series |
| `PROGRESS.md` | Build status, decisions taken, open gaps |

---

## Environment

Python 3.11. Core dependencies: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `shap`,
`lifelines`, `scipy`, `google-generativeai`. Fixed seed `20260828` throughout. The full pinned
list is in `requirements.txt`.
