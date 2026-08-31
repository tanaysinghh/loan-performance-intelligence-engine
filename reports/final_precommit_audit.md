# Final Pre-Submission Compliance Audit

**Audited state:** `master` @ `d2b7cbe`, verified in sync with `origin/master` on GitHub.
**Source of truth:** `Intain_AI_Track_Problem_Statement.docx1e3a138 (1).pdf`, read directly for
this audit (6 pages, extracted fresh). No earlier audit in this repository was used as input;
where this pass contradicts an earlier one, this pass supersedes it.
**Method:** deliverable presence checked with `git ls-files`, not filesystem existence — the
question is what a judge cloning the repository actually receives. Figures read from committed
artefacts.
**Date:** 2026-08-30. Originally audit-only.

> **Update, same day — P1 and P2 are now RESOLVED.** The audit below is preserved as written,
> including its original verdict, because an audit rewritten after its findings were fixed is
> no longer evidence of anything. See *Resolution* at the end for what changed. The two
> remaining findings, P3 and P4, were reviewed and deliberately skipped as cosmetic.

---

## 1. Minimum Acceptable Solution (PS §9)

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Reproducible data pipeline | **Met** | `src/pipeline.py`, single command `python -m src.pipeline`; `submission/run_manifest.json`; fixed seed `20260828`; `requirements.txt` |
| 2 | Data profiling report | **Met** | `reports/data_intelligence_report.md` — distributions, missingness with chi-square mechanism tests, outliers, invalid date relationships, §5 cross-column relationship breaks, §6 correlation/dependent-field analysis |
| 3 | Feature engineering | **Met** | `src/features/build_features.py`, 81 engineered features; `reports/leakage_probe.csv` guards them |
| 4 | Non-LLM supervised model | **Met** | LightGBM + isotonic calibration, logistic baseline, isolation forest, Cox PH, Markov chain. `artifacts/performance_models.pkl`, `reports/model_metrics.csv` |
| 5 | Time-aware train / validation split | **Met** | `src/models/splits.py::purged_time_split`, chronological with an H-month embargo. `reports/split_summary.csv` reports every window explicitly |
| 6 | Delinquency or default prediction | **Met** | Both. Test ROC-AUC 0.916 (delinq-3m), 0.878 (delinq-6m), 0.921 (default-12m) |
| 7 | Prepayment or next-state prediction | **Met** | Both. Prepayment-12m ROC 0.626; next-state multiclass macro-F1 0.614 vs Markov baseline 0.366 (`reports/next_state_metrics.csv`) |
| 8 | Anomaly or exception detection | **Met** | Isolation forest + rule engine + exception type/probability. `reports/anomaly_report.md`, `reports/anomaly_review_queue.csv` |
| 9 | Explainability output | **Met** | Global + local SHAP for all four targets, `reports/explainability_report.md`, 8 SHAP CSVs |
| 10 | LLM reviewer summary | **Met** | Live on Google Gemini `gemini-3.5-flash-lite`. `reports/copilot_report.md`, `submission/llm_prompt_log.jsonl` |
| 11 | Model card | **Met** | `submission/MODEL_CARD.md` — objective, data, features, model type, validation method, metrics, limitations, leakage controls, known failure modes. All nine PS §11 model-card elements present |
| 12 | AI Development Log | **Met** | `submission/AI_DEVELOPMENT_LOG.md`, 14 sections — tools, verbatim prompts, accepted/rejected outputs, human review process, AI code share, lessons |
| 13 | `submission.csv` | **Met** | 16,000 × 21. Read **from the git index**: zero nulls, no duplicate `loan_id`, all probabilities in [0,1] |

**13 of 13 met.**

### Qualification rule — "a solution that only sends records to an LLM API for classification should not qualify"

**Does not apply, and this is enforced mechanically rather than asserted.**

- The only LLM call site in the repository is `Copilot.ask` (`src/copilot/client.py`), reachable
  only from `src/copilot/run_copilot.py`.
- Every input to it is a **grounding pack** built in `src/copilot/grounding.py` from figures
  that LightGBM, SHAP, the isolation forest, Cox and the Markov chain had *already produced*.
  The pack is constructed after the models have run.
- `tests/test_no_llm_prediction.py` AST-parses every module under `src/data`, `src/features`,
  `src/models`, `src/scenarios`, `src/explain` and fails if any imports an LLM client. The
  guard is written against the **capability**, not one vendor: `anthropic`, `openai`, `google`,
  `cohere`, `mistralai`, `ollama`.
- `src/submission.py` never imports the copilot. Every column in `submission.csv` traces to a
  fitted non-LLM estimator.
- In `src/pipeline.py` the copilot stage returns only `{mode, stats}` — nothing flows back into
  predictions.

**One nuance stated plainly rather than glossed:** loan-record *fields* (loan_id, servicer,
status, DPD, balance) **do** reach the LLM inside `loan_pack`. They are always accompanied by
the model's already-computed prediction, and the task is always "narrate this", never "score
this". That is narration downstream of a trained model, which the PS explicitly permits
("The LLM can help explain, summarize, retrieve definitions, generate reviewer notes").

---

## 2. Advanced Features (PS §10) — optional / bonus

Accuracy matters more than score here. Nothing below is inflated.

| # | Feature | Status | Evidence / what is actually there |
|---|---|---|---|
| 1 | Competing-risk survival model | **Implemented** | Aalen-Johansen cumulative incidence with prepayment as the competing risk. `reports/cumulative_incidence.csv`; survival report §4 explains why 1−KM is the wrong number |
| 2 | Monte Carlo portfolio simulation | **Not attempted** | No sampling-based simulation anywhere. Scenarios are deterministic repricing (Engine A) and a macro-conditioned Markov chain (Engine B) |
| 3 | Drift monitoring dashboard | **Partial** | Drift *is* computed — PSI, KS, missingness delta, severity across 19 columns (`reports/drift_report.csv`, `reports/drift_by_target.csv`). There is **no dashboard**: no Streamlit/HTML/served app in the repo. Reports, not monitoring UI |
| 4 | Segment-level scenario curves | **Implemented** | Six segment CSVs (vintage, credit band, LTV band, state, servicer, rate-incentive bucket) plus `scenario_markov_paths.csv` giving 0–12 month state-distribution curves per scenario |
| 5 | Model calibration by vintage or credit band | **Not attempted** | Calibration is global (isotonic, ECE reported overall). `km_default_by_credit_band.csv` is **survival curves** by credit band, not calibration by band — different thing, and it would be wrong to claim it |
| 6 | MLflow / Weights & Biases tracking | **Not attempted** | No tracking library in `requirements.txt` or anywhere in `src/` |
| 7 | RAG over data dictionary and validation rules | **Partial** | Retrieval-grounded generation over both sources is real and working: `grounding.dictionary_pack` retrieves dictionary entries, `grounding.rule_pack` retrieves rule definitions plus observed firing rates, both feeding live Gemini calls. But it is **filtered lookup, not embedding/vector retrieval** — no index, no embeddings, no similarity search. Honest label: grounded retrieval, not RAG in the usual sense |
| 8 | Agentic experiment runner | **Partial, generously** | `reports/hyperparameter_search.csv` — 20 configurations swept over learning rate, num_leaves, min_child_samples, n_estimators, selected on validation PR-AUC. That is a **hyperparameter sweep**, not an agent choosing experiments. The agentic evidence in this submission is the development log (Task 8), not a runner |
| 9 | Automated feature-store style pipeline | **Partial** | `src/features/dataset.py` provides a single shared feature frame with a parquet cache and `_cache_is_stale()` invalidation keyed to raw-pack mtime, so every stage sees identical rows/repairs/features. That is the *discipline* of a feature store, not a feature store — no registry, no versioned feature definitions, no point-in-time serving API |
| 10 | Bias / fairness analysis | **Not attempted** | Explicitly disclosed as absent. Model card: `state` and `servicer_name` are model inputs and "would need disparate-impact analysis before production use". README lists it under Known gaps. Correctly disclosed, genuinely not done |
| 11 | Counterfactual explanations | **Not attempted** | No counterfactual/what-if machinery anywhere |
| 12 | Stress sensitivity by feature cluster | **Not attempted** | Scenario drivers exist (`scenario_drivers_default.csv`, `scenario_drivers_prepay.csv`) but per-feature, not clustered |
| 13 | Model confidence intervals | **Partial** | `reports/uncertainty_*.csv` give `staged_mean`, `staged_std`, `staged_p10`, `staged_p90` from staged ensemble predictions. `src/explain/shap_explain.py:158` states outright that this "is *not* a statistical confidence interval and is not labelled as one." Uncertainty bands, not CIs — and the code says so |
| 14 | Human-in-the-loop active learning | **Not attempted** | A reviewer queue exists (`anomaly_review_queue.csv`, 40 rows) and the copilot has a correction round-trip, but nothing feeds human labels back into retraining |
| 15 | Synthetic-data stress testing | **Partial** | A full synthetic generator (`src/data/generate_synthetic.py`) exists and still runs — the pipeline builds against either source, and `src/data/messiness.py` injects controlled defects with `ground_truth_defect_log.csv` as the answer key. That is closer to *synthetic-data-driven defect validation* than to stress testing the model under synthetic distribution shift |

**Tally: 2 implemented, 6 partial, 7 not attempted.** This section is bonus, so no gap here is
a problem — but claiming more than 2 as fully implemented would not survive scrutiny.

---

## 3. Disqualification Conditions (PS §13)

Each confirmed with evidence, not assertion.

### 3.1 "Only uses an LLM API for prediction" → **Does not apply**
Four trained model families produce every number: LightGBM (calibrated), isolation forest, Cox
PH, empirical Markov chain. The LLM narrates. Enforced by AST import guard (§1 above), which is
a tracked, passing test — not a claim.

### 3.2 "Does not train a non-LLM model" → **Does not apply**
`artifacts/performance_models.pkl`; `reports/model_metrics.csv` covers four binary targets ×
four model variants (`prior`, `baseline_logistic`, `lgbm_raw`, `lgbm_calibrated`) across valid
and test splits. Multiclass models in `next_state_metrics.csv` and `exception_type_metrics.csv`.

### 3.3 "Random splits that leak the same loan across train and validation without justification" → **Does not apply**
This is the condition most worth scrutinising on panel data, and the repository does more than
avoid it — it **measures** it.

- **No random split is used for any reported metric.** `purged_time_split` is chronological,
  capped at `usable_max = last_month − horizon` so no unobservable label enters, with an
  H-month embargo between fitting data and the test window.
- **The loan overlap is disclosed, not hidden.** `reports/split_summary.csv` carries an explicit
  `loan_overlap_train_test` column (e.g. 11,061 loans for the 3-month target). Same loan at
  different months is inherent to a loan-month panel.
- **Justified by a controlled experiment.** `reports/leakage_probe.csv` runs three splits side
  by side:

  | target | purged_time_split | loan_disjoint_time_split | random_row_split (unsound) | inflation |
  |---|---:|---:|---:|---:|
  | delinquency 3m | 0.9161 | 0.9191 | 0.9064 | −0.0097 |
  | delinquency 6m | 0.8784 | 0.8699 | 0.8920 | +0.0135 |
  | default 12m | 0.9207 | 0.9352 | **0.9988** | +0.0781 |
  | prepayment 12m | 0.6259 | 0.6041 | **0.9831** | +0.3572 |

  A **loan-disjoint** variant is run and reported, and it lands within noise of the headline
  number. The random row split is shown inflating default AUC to 0.999 and prepayment to 0.983
  — demonstrating precisely the failure the condition describes, and that this submission does
  not have it. Reproduced in `MODEL_CARD.md:191`.

### 3.4 "Leaks target labels into features" → **Does not apply**
Targets are forward-looking flags constructed from future months and excluded from the feature
matrix. Horizon purging prevents a training row's outcome window reaching the test period.
`reports/leakage_probe.csv` is the standing check; the near-perfect random-split AUCs above are
the signature of leakage, and the reported splits do not show it.

### 3.5 "Provides no reproducible code" → **Does not apply**
Single-command pipeline with stage selection; `requirements.txt`; `run_manifest.json`; fixed
seed; README quick start. The PS §6 metadata artefacts (`validation_rules.json`,
`data_dictionary.csv`, `macro_scenarios.csv`) are tracked, so a clone can reproduce.

### 3.6 "Provides no evaluation metrics" → **Does not apply**
Every metric PS §8 Task 2 names is present in `reports/model_metrics.csv`: `roc_auc`, `pr_auc`,
`best_f1`, `recall_at_precision_30`, `recall_at_precision_50`, `brier`, `log_loss`, `ks`,
`lift_at_10pct`, `ece`. **macro-F1** is present for both multiclass targets.

### 3.7 "Fabricates results" → **Does not apply**
Every figure traces to a committed artefact. Three places where fabrication was available and
declined, each documented:
- No plausible LLM transcripts were written while waiting for a credential (AI log §12).
- The lost LaTeX transcript is **described, not reconstructed** (copilot report §5) — the raw
  log line was destroyed by the log-deletion bug and is not recreated from memory.
- The LaTeX ablation came out **negative** and is reported as negative; the prompt rule is not
  claimed as the fix.
Counter-evidence is also reported rather than suppressed: the logistic baseline beats LightGBM
on ranking for the default proxy (PR-AUC 0.574 vs 0.532) and prepayment, and the model card
says so.

### 3.8 "Uses public data in violation of source terms" → **Does not apply — re-verified mechanically**

| Check | Result |
|---|---|
| `git ls-files data/raw/loan_panel.csv` | **0** (149 MB, 673k loan-month rows — excluded) |
| `git ls-files data/raw/servicer_updates.csv` | **0** (24 MB, 245k loan-level rows — excluded) |
| Raw SFLLD `.txt` under `dataset/` tracked | **0** |
| Tracked under `dataset/` | `download_sflld.md` only |
| Largest blob ever in history | 2.7 MB, against ~1.2 GB of raw data — never committed |

`.gitignore` now states the reason inline rather than leaving it implicit. Representative
samples (`data/samples/`) ship instead of the licensed panels.

**Download instructions verified working, not just present.** `dataset/download_sflld.md`
cites the exact Freddie Mac URL listed in PS §5
(`freddiemac.com/research/datasets/sf-loanlevel-dataset`) plus the Clarity portal, gives the
required directory layout, and documents the undocumented 31/35 column arrangement. Its
verification command was **executed during this audit** and returned the expected result for
all five vintages:

```
{'2019_origination': 31, '2019_performance': 35, ... '2023_performance': 35}
```

FRED macro series are separately licensed for redistribution and *are* vendored, with a refresh
command.

### 3.9 "Cannot explain model behavior" → **Does not apply**
Global + local SHAP for all four targets; plain-English driver strings; anomaly driver
attribution; false positive / false negative analysis per target with mean feature values by
error class (`explainability_report.md` §§86, 245, 404, 563); uncertainty bands.

### 3.10 "Presents LLM-generated narratives without grounding" → **Does not apply**
The strongest area of the submission. Every LLM call receives a grounding pack and nothing else.
Two automated controls run on output:
- **Grounding validator** — extracts every number from generated text and matches it against the
  pack, including ×100 and rounded forms. Unmatched numbers block the output.
- **Usefulness check** — blocks output that is true but directs the reviewer at a field the pack
  itself reports as clean.
12-case self-test, deterministic against a fixed pack. Blocked output is fed back with the
specific finding and re-judged; both halves are logged. 97 calls retained across
`llm_prompt_log.jsonl` and `llm_prompt_log_archive.jsonl`.

**10 of 10 conditions do not apply.**

---

## 4. Full Guideline Sweep

### §3 Business-Adjacent Context
The "core question" is answered in three parts, each with an artefact: unreliable records
(`anomaly_review_queue.csv`, `record_quality_scores.csv`, `batch_quality_scores.csv`), loans
likely to deteriorate (four calibrated binary models), portfolio under future scenarios
(`scenario_report.md`, both engines). All 20 example fields in §3 are present in the panel.

### §4 Benchmarking Takeaways
| Theme | Addressed |
|---|---|
| Data intelligence before modeling | Yes — profiling, missingness, outliers, relationships, **association rules**, drift all precede training |
| Prediction is multi-outcome | Yes — delinquency 3m/6m, default, prepayment, next-state, exception |
| Time-aware modeling | Yes — purged split **and** Cox/KM/Aalen-Johansen/Markov |
| Scenario analytics table stakes | Yes — base, adverse-credit, high-prepayment |
| LLM copilots need governance | Yes — grounded explanations, full prompt logs, human decision control, rejected-output examples |

### §5 Source Data
Organiser did not supply a dataset. Freddie Mac SFLLD is one of the six listed sources. The PS
anticipated participants would not need to "understand raw mortgage-performance schemas"; this
submission did the schema work anyway and documented a 31/35 layout that contradicts Freddie
Mac's own published `file_layout.xlsx` — verified empirically across five vintages.

### §6 Organiser Data Pack — **residual risk, unchanged**
No organiser pack exists, so file names were chosen locally.

| Expected | Present | Note |
|---|---|---|
| `loan_monthly_performance_train.csv` | Equivalent | `data/raw/loan_panel.csv` — single labelled panel, split internally |
| `loan_monthly_performance_test.csv` | **No** | **See gap R1 below** |
| `loan_static_attributes.csv` | Denormalised | Static attributes are in the panel |
| `servicer_updates.csv` | **Yes** | Name and purpose match |
| `data_dictionary.md` | Partial | Ships as `.csv`, not `.md` |
| `validation_rules.json` | **Yes** | Now tracked; 17 rules |
| `macro_scenarios.csv` | **Yes** | Now tracked |
| `submission_template.csv` | **No** | Column names are this submission's own |

### §7 Example Fields and Targets
All example fields present. All seven named targets modelled: `next_3m_delinquency_flag`,
`next_6m_delinquency_flag`, `next_12m_default_flag`, `next_12m_prepayment_flag`, `next_state`,
`exception_required`, `exception_type`. **`next_12m_default_flag` is a documented 90+ DPD
proxy** — realised credit events occur on 14 of 16,000 loans (~0.09%), disclosed in five places.

### §8 Required Tasks 1–8, in detail

| Task | Sub-requirement | Status |
|---|---|---|
| 1 | Column distributions | Met — `profile_numeric.csv`, `profile_categorical.csv` |
| 1 | Missing-value patterns | Met — with chi-square mechanism tests (`missingness_mechanism_tests.csv`) |
| 1 | Outliers + invalid date relationships | Met |
| 1 | Correlations / highly dependent fields | Met — report §6 |
| 1 | Cross-column relationship breaks | Met — report §5 |
| 1 | Train vs test drift | Met — PSI/KS over 19 columns |
| 1 | Record- **and** batch-level DQ scores | Met — both files present |
| 2 | Non-LLM models for all four outcomes | Met |
| 2 | Time-aware split | Met |
| 2 | Baseline vs improved comparison | Met — `prior`, `baseline_logistic`, `lgbm_raw`, `lgbm_calibrated` |
| 2 | Class imbalance + calibration | Met — isotonic; ECE ≤ 0.004 on three of four targets |
| 2 | ROC-AUC, PR-AUC, F1, recall@fixed precision, Brier, macro-F1 | **All six present** |
| 3 | Survival / hazard / competing-risk / transition | Met — all four |
| 3 | Event curves / cumulative probabilities | Met |
| 3 | Censoring treatment explained | Met — survival report §2, typed censoring table |
| 3 | Compare against simpler baseline | Met — KM concordance 0.50 by construction; Markov transition baseline for next-state |
| 4 | Record-level anomaly score | Met |
| 4 | Exception probability and type | Met |
| 4 | Anomaly drivers explained | Met |
| 4 | **≥20 reviewer-ready examples** | Met — **40 rows**, with drivers, rules violated, DQ score |
| 5 | Base / adverse-credit / high-prepayment | Met |
| 5 | Projected delinquency, default, prepayment rates | Met |
| 5 | Segment impacts by vintage / credit band / state / servicer | Met — all four, plus LTV and rate-incentive |
| 5 | Top scenario drivers explained | Met |
| 6 | Global + local explanations | Met |
| 6 | Drivers of default, delinquency, prepayment **and anomaly** | Met |
| 6 | Model confidence or uncertainty | Met — labelled as bands, not CIs |
| 6 | False positive / false negative analysis | Met — per target, with feature means by error class |
| 7 | Grounded summaries, reviewer notes, dictionary retrieval, rule suggestions, scenario summaries | Met — four use cases live |
| 7 | Log prompt, model, timestamp, output | Met — plus provider, SDK, tokens, finish reason, latency, validator verdict |
| 7 | Label output recommendation-not-decision | Met — per output, with model and timestamp |
| 7 | **Examples where the LLM was wrong, vague, or overconfident** | Met — 10× transcription error `0.046` vs `0.0046` quoted verbatim with its correction; null advice; LaTeX markup |
| 8 | AI Development Log | Met |
| 8 | Tools, prompts, accepted/rejected, review process, code share, lessons | Met — all six, 14 sections |

### §11 Expected Deliverables

| Deliverable | Status |
|---|---|
| GitHub repository | Met — `master` on GitHub, 129 tracked files |
| **Reproducible notebook *or* scripts** | **Met via scripts.** The PS says "notebook **or** scripts". `src/pipeline.py` is an end-to-end runnable workflow. `notebooks/` is empty, and that is **not a gap** under this wording |
| `submission.csv` | Met |
| Model card | Met — all nine named elements |
| Data intelligence report | Met |
| Explainability report | Met |
| Scenario report | Met |
| LLM copilot demo | Met |
| AI Development Log | Met |
| **Five-minute demo video** | **MISSING — not recorded** |

### §12 Judging Criteria (100 pts)

| Criterion | Pts | Assessment |
|---|---:|---|
| Data Intelligence and Profiling | 15 | Strong — every named element present including association rules and both DQ score levels |
| Predictive Modeling | 20 | Strong — four targets, time-aware split, calibration, full metric set. Honest that the logistic baseline wins on ranking for two targets |
| Time-to-Event / Transition | 15 | Strong — KM, Aalen-Johansen competing risk, Cox, Markov, with baseline comparison |
| Anomaly and Exception Intelligence | 10 | Strong — 40 examples, rule+ML combination, drivers |
| Scenario and Stress Simulation | 10 | Strong — two engines, six segment cuts. Engine A's null credit response is reported as null |
| Explainability and Responsible AI | 10 | Strong on explanations/calibration/uncertainty/limitations. **Weakest sub-item: no fairness analysis**, though it is disclosed rather than hidden |
| Smart LLM Usage | 10 | Strong — grounded, logged, two automated controls, real captured failures. **At risk from gap P1 below** |
| ML Engineering and Reproducibility | 5 | Runnable pipeline, clean structure, reproducible submission. **README is stale — gap P1** |
| Agentic Coding Evidence | 5 | Strong — 14-section log, verbatim prompts, rejected outputs, a prompt whose literal instruction was not followed |

### §14 Five-Minute Demo Flow
`reports/demo_video_script.md` maps 1:1 to all 15 beats in the correct order with timings
summing to 5:30, each naming the exact file to have on screen. **The script is ready; the video
is not recorded.**

---

## 5. Findings — things not addressed anywhere before this audit

### P1 — `README.md` is materially stale and understates the submission — **RESOLVED**

The README still describes the pre-Gemini state in five places. This was missed by every prior
audit in this repository.

| Line | Says | Reality |
|---|---|---|
| 76 | `export ANTHROPIC_API_KEY=sk-ant-...` | Wrong variable — the client reads `GEMINI_API_KEY` / `GOOGLE_API_KEY` |
| 117 | "copilot/ … **Anthropic client** …" | It is a Gemini client |
| 184 | "**No live LLM transcripts.** No Anthropic credential was available…" | **False.** The copilot runs live; transcripts exist |
| 188 | "(`export ANTHROPIC_API_KEY` and re-run…)" | Wrong variable |
| 218 | Dependencies "… `anthropic`" | `requirements.txt` ships `google-generativeai>=0.8.5`; `anthropic` is not a dependency |

**Why this is the top finding, not a typo:**

1. **It actively understates a completed deliverable.** README "Known gaps" tells a judge that
   Task 7 has no live LLM transcripts. That is the single item worth up to 10 points under
   *Smart LLM Usage*, and it is done. A judge reading the README first may never look further.
2. **It breaks reproducibility for the copilot.** A judge following the README exports
   `ANTHROPIC_API_KEY`, `credentials_available()` returns false, and the copilot silently runs
   in `offline_template` — appearing to confirm the stale README. Self-reinforcing failure.
3. **README is explicitly named in the §12 rubric** under *ML Engineering and Reproducibility*.
4. **Internal contradiction:** README dependency list vs `requirements.txt`.

### P2 — No code path scores an externally supplied unlabeled test file — **RESOLVED (disclosure)**

PS §6 specifies `loan_monthly_performance_test.csv` as "unlabeled test dataset for final
scoring. Participants submit probabilities, anomaly scores, and reviewer actions." No organiser
pack was issued, so `submission.csv` is built from this project's own internal out-of-time
holdout. **There is no loader or scoring entry point for an externally supplied unlabeled
file.** If organisers release one at judging time, producing a conforming submission would
require new code under time pressure.

This risk is recorded in `reports/compliance_audit.md` (Risk 4b) but has not been mitigated and
does not appear in `PROGRESS.md`'s handover list, so it would be easy to miss.

### P3 — Column names in `submission.csv` are this project's own invention

No `submission_template.csv` was issued. `SUBMISSION_FORMAT.md` maps all seven PS §6 elements to
columns, which is the right defence, but if organisers publish a template with different
headers a rename pass is needed. Low effort, low probability, non-zero.

### P4 — `data_dictionary` ships as `.csv` where PS §6 names `.md`

Content is complete and the copilot retrieves from it successfully. Format differs from the
name in the PS table. Cosmetic; note only.

### Explicitly **not** gaps

- **`notebooks/` is empty.** PS §11 says "Reproducible notebook **or** scripts". Scripts exist
  and run end to end. Earlier handover notes flagged this as a possible gap; reading the PS
  directly, it is not one.
- **Fairness analysis absent.** Bonus-tier (§10) and disclosed in both the model card and the
  README. It costs some *Explainability and Responsible AI* upside but breaches nothing.
- **Advanced features mostly not attempted.** §10 is explicitly optional.

---

## 6. Summary Table

| # | Item | Status | Evidence / Gap |
|---|---|---|---|
| §9.1–13 | Minimum Acceptable Solution, all 13 items | **Met** | See §1 table; all verified against the git index |
| §9 | Qualification rule (not LLM-only classification) | **Passes** | AST import guard test; grounding packs built from trained-model output |
| §10 | Advanced features | **2 implemented, 6 partial, 7 not attempted** | Bonus tier; accurately labelled, nothing inflated |
| §13.1–10 | All ten disqualification conditions | **None apply** | §3; loan-disjoint control and licence checks run during this audit |
| §13.8 | Licensed data + download instructions | **Clean** | 0 licensed files tracked; `verify_layout` executed, returned 31/35 across five vintages |
| §11 | Deliverables | **9 of 10** | **Demo video not recorded** |
| §12 | Judging criteria | **All nine addressed** | Weakest: no fairness analysis (disclosed) |
| §14 | Demo flow | **Script complete, video missing** | 15 beats mapped, figures verified |
| **P1** | **README stale in 5 places** | **MUST FIX** | Wrong env var; falsely claims Task 7 incomplete; contradicts `requirements.txt` |
| **P2** | **No external test-file scoring path** | **Risk** | PS §6 anticipates one; none issued; no loader exists |
| **P3** | Submission column names self-defined | Low risk | Mapped in `SUBMISSION_FORMAT.md` |
| **P4** | `data_dictionary` is `.csv` not `.md` | Cosmetic | Content complete |

---

## Verdict

*(Verdict as originally written, before P1 and P2 were fixed. Superseded by the Resolution
section below.)*

**Not submission-ready without one fix. The work is complete; the README misrepresents it.**

Everything the problem statement requires is built, tested and committed. All 13 Minimum
Acceptable Solution items are met, all 10 disqualification conditions verifiably do not apply,
and 9 of 10 deliverables are present. The one substantive gap in the *work* — the demo video —
is known and requires a human.

But `README.md` — the first file a judge opens, and one explicitly named in the rubric — tells
that judge the LLM copilot never ran live. That is no longer true, and it understates the single
deliverable worth 10 points. It also hands over the wrong environment variable, so a judge
attempting to reproduce the copilot will land in `offline_template` mode and see the stale claim
apparently confirmed.

**Priority order before submitting:**

1. **P1 — Fix `README.md` (required).** Five edits: `ANTHROPIC_API_KEY` → `GEMINI_API_KEY`
   (lines 76, 188), "Anthropic client" → Gemini (117), delete or rewrite the "No live LLM
   transcripts" known-gap bullet (184), and `anthropic` → `google-generativeai` in the
   dependency list (218). Small, mechanical, and it is the difference between a judge reading
   Task 7 as incomplete or complete.
2. **Record the five-minute demo video (required deliverable).** Script is ready to read from
   directly.
3. **P2 — Decide on the external test file (judgment call).** Either add a thin loader +
   scoring entry point defensively, or accept the risk and note it in the README. Currently it
   is recorded only in the compliance audit, where it is easy to miss.
4. **P3/P4 — Optional.** Only act if organisers publish a template or specifically require
   `data_dictionary.md`.

Items 1 and 2 are genuinely required. Item 3 is a considered risk decision. Items 4 are
cosmetic.

---

# Resolution — P1 and P2 closed

## P1 — `README.md` corrected

Eleven fixes, not the five originally flagged. Re-reading the file end to end, as opposed to
grepping for the provider name, turned up six more — including two wrong numbers that no
provider-name search would ever have caught.

| # | Location | Was | Now |
|---|---|---|---|
| 1 | Copilot section | `export ANTHROPIC_API_KEY=sk-ant-...` | `export GEMINI_API_KEY=...`, with the free-key URL |
| 2 | Copilot section | Framed as an optional add-on | States the copilot runs live on `gemini-3.5-flash-lite`, why Gemini was chosen, and the 20-req/day caveat on `gemini-3.6-flash` |
| 3 | Repository layout | "Anthropic client" | "Gemini client, output validators, LaTeX ablation" |
| 4 | **Known gaps** | **"No live LLM transcripts… no Anthropic credential was available"** | **Removed — it was false.** Replaced with the genuine open gap: the demo video is not recorded |
| 5 | Environment | Dependency `anthropic` | `google-generativeai`, matching `requirements.txt` |
| 6 | Quick start | "31 tests" | "40 tests" |
| 7 | Repository layout | "tests/ 31 tests" | "40 tests" |
| 8 | Repository layout | `submission/` listing omitted two tracked files | Adds `SUBMISSION_FORMAT.md` and `llm_prompt_log_archive.jsonl` |
| 9 | LLM boundary | "A six-case self-test" | Twelve-case, fixed-pack, plus the second control and the real captured failures |
| 10 | Validation design | **"inflates test ROC-AUC by +0.10 to +0.32"** | **Wrong.** Actual −0.0097 to +0.3572. Now quotes the real per-target figures and notes the inflation is *not* uniform |
| 11 | Baseline comparison | **"2-4x on Brier score"** | **Wrong, and understated.** Actual 2.0x to 11.9x. Now quotes both endpoints with the underlying Brier values |

Findings 10 and 11 are the ones worth noting. One overstated the low end of a range and the
other understated a result in this submission's own favour, and both had survived every prior
review because reviews checked the generated reports and treated the README as prose.

Also verified during the fix, mechanically: all 33 file paths referenced in the README exist;
all three CLI flags (`--skip-data`, `--stage`, `--n-loans`) exist as documented; and the
headline figures (16,000 loans, 673,242 rows, 33-column contract, 14 realised credit events,
31/35 SFLLD layout) match the committed artefacts and a live `verify_layout()` run.

## P2 — external test file, closed as a disclosure

No organiser pack was issued, so no `loan_monthly_performance_test.csv` exists to score. One
sentence was added to `README.md` (Data section) and to `MODEL_CARD.md` (before §3, via the
generator so it survives regeneration) stating that none was issued, that this project's own
pipeline fills the gap, and that `submission.csv` contains held-out predictions on the purged
out-of-time window from `src/models/splits.py` — reported per target in
`reports/split_summary.csv` — rather than scores against an external file.

**No defensive loader was built.** Writing a scoring path for a file that does not exist, to a
schema nobody has published, would be speculative code carrying a real risk of being wrong in a
way nobody could test.

## Not done, deliberately

- **P3** — submission column names are this project's own, since no `submission_template.csv`
  was issued. All seven PS §6 elements are mapped in `SUBMISSION_FORMAT.md`, which is the
  defence that matters. Cosmetic until a template exists.
- **P4** — `data_dictionary` ships as `.csv` where the PS table names `.md`. Content complete.

## Revised verdict

**Submission-ready, with one outstanding deliverable that requires a human: the five-minute
demo video.**

Every §9 Minimum Acceptable Solution item is met, all ten §13 disqualification conditions
verifiably do not apply, and the README now describes the submission accurately rather than
underselling its strongest-governed component. Tests 40/40; validator self-test 12/12.

---

# Addendum — 2026-08-31: the "no dashboard" line is now out of date

Recorded here rather than edited into section 2, for the same reason P1 and P2 were: the audit
body is pinned to `master` @ `d2b7cbe` and is preserved as written. This addendum supersedes it
on one point of fact.

**Section 2, advanced feature 3 (*Drift monitoring dashboard*) says "There is **no dashboard**:
no Streamlit/HTML/served app in the repo." That was true at `d2b7cbe` and is no longer true.**
`dashboard.py` was added afterwards in `ea23ef6` and is tracked, along with `.streamlit/config.toml`,
with `streamlit>=1.49` in `requirements.txt`. It renders ten sections over the committed
artefacts — including **Train versus test drift**, carrying the PSI panel and the note that loan
age and remaining term drift hardest because the test window sits later in the panel by
construction.

**What is deliberately not changed here:**

- **The section 2 tally ("2 implemented, 6 partial, 7 not attempted") is left as written.**
  Whether a demo dashboard over static generated artefacts promotes feature 3 from *Partial* to
  *Implemented* is a scoring judgment, not a fact. The app displays drift; it does not monitor a
  live feed or re-compute on new data. Re-tallying is the author's call, and section 10 is
  bonus tier either way.
- **No deployment is claimed.** Nothing in this repository records a Streamlit Community Cloud
  URL, and none was verified for this addendum. The app is committed and runnable locally with
  `streamlit run dashboard.py`; whether it is also *deployed* is outstanding.

Nothing else in the audit changed. The demo video remains the one outstanding deliverable.
