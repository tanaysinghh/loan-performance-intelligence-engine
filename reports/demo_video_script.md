# Five-Minute Demo — Script and Storyboard

Maps one-to-one to the Five-Minute Demo Flow in section 14 of the problem statement. Fifteen
beats, 300 seconds. Each beat names the exact file to have on screen, so recording is a matter
of reading the narration while stepping through the listed artefacts.

**Recording setup.** Two windows: a terminal at the repo root, and an editor/preview showing
`reports/`. Open these as tabs before you start, in this order:

1. `reports/data_intelligence_report.md`
2. `reports/model_performance_report.md`
3. `reports/survival_report.md`
4. `reports/anomaly_review_queue.csv`
5. `reports/scenario_report.md`
6. `reports/explainability_report.md`
7. `reports/copilot_report.md`
8. `submission/submission.csv`
9. `submission/AI_DEVELOPMENT_LOG.md`

**Figures below are from the run of 2026-08-30 and match the committed reports.** If you
retrain before recording, re-read them off the screen. Never say a number that is not visible.

---

## Beat 1 — Dataset and targets (0:00–0:25)

> "This is the Loan Performance Intelligence Engine. It runs on real Freddie Mac
> Single-Family Loan-Level data — five vintages, 2019 through 2023, 250,000 loans and 10.5
> million monthly records, sampled to 16,000 loans and 673,000 monthly rows.
>
> One row per loan per month. We predict six things: delinquency at three and six months,
> default and prepayment at twelve, the next servicing state, and whether a record needs an
> operational exception raised.
>
> One deviation up front, and it is in the model card. Real credit events — foreclosure,
> short sale, REO — happen to just **14 of our 16,000 loans**. Under a tenth of a percent.
> That is not modellable, so the twelve-month default target is a documented ninety-plus-days
> past-due proxy. We say so everywhere it appears."

**On screen:** `dataset/download_sflld.md`, then the `data/raw/loan_panel.csv` header.

---

## Beat 2 — Data profiling report (0:25–0:50)

> "Task 1 profiles before anything is trained: distributions for every column, missingness
> with mechanism tests, outliers, invalid date relationships.
>
> Missingness here is not random. A chi-square test against servicer rejects independence, so
> the mechanism is missing-at-random conditional on servicer. That matters — dropping
> incomplete rows would quietly drop those servicers' books and bias every rate downstream."

**On screen:** `reports/data_intelligence_report.md`, sections 1 and 3.

---

## Beat 3 — Top data-quality issues (0:50–1:15)

> "The exception and data-quality layer is the one part that is fabricated, and the model card
> says so plainly — Freddie Mac gives you no second source, no ingestion timestamps and no
> document data. We inject defects at known rates and score the profiler against that ground
> truth, so detection is validated rather than eyeballed.
>
> The conflicts sit on real events, though: **6,903 of the 16,000 loans genuinely change
> servicer** at least once, and the fabricated second feed is anchored on those transfers."

**On screen:** `reports/validation_rule_summary.csv`, then
`data/raw/ground_truth_defect_log.csv`.

---

## Beat 4 — Feature engineering (1:15–1:35)

> "Features use only information available at or before the reporting month: loan age, balance
> trajectory, delinquency run-lengths, rate incentive against the real Freddie Mac survey
> rate, servicer aggregates. The macro series are real — PMMS, BLS unemployment,
> Case-Shiller, all vendored in the repo."

**On screen:** `src/features/build_features.py`, then `data/external/fred_MORTGAGE30US.csv`.

---

## Beat 5 — Time-aware split (1:35–1:55)

> "Splitting is by time, never by row — and every target gets its own boundary, because a
> twelve-month label has to stop training far earlier than a three-month one. The default
> model trains to 2023-03 and tests on 2024-10 through 2025-03. The three-month model trains
> to 2024-09.
>
> Row overlap across the boundary is exactly zero, an embargo longer than the label horizon
> separates the windows, and a leakage probe re-checks it on every run."

**On screen:** `reports/split_summary.csv`, then `reports/leakage_probe.csv`.

---

## Beat 6 — Baseline model performance (1:55–2:15)

> "Baseline first, and it is a real baseline: regularised logistic regression on nine raw
> credit fields. On three-month delinquency it gets ROC-AUC 0.883 and PR-AUC 0.581 against a
> 3.1% base rate."

**On screen:** `reports/model_metrics.csv`, baseline rows.

---

## Beat 7 — Improved model performance (2:15–2:45)

> "Then gradient boosting with class weighting and calibration. Three-month delinquency goes
> from 0.883 to **0.916** ROC-AUC, and PR-AUC from 0.581 to **0.650** — twenty times the base
> rate.
>
> And here is the part I want to be honest about. On the twelve-month default target the
> logistic baseline actually **beats** the boosted model on PR-AUC: 0.574 against 0.532. We
> ship the GBM anyway, for one reason — calibration. Its expected calibration error is 0.004
> against the baseline's 0.223. The baseline can rank a queue; it cannot tell you what the
> probability is, and the action thresholds need a probability."

**On screen:** `reports/model_performance_report.md`, comparison and calibration sections.

---

## Beat 8 — Survival / transition output (2:45–3:15)

> "Task 3 is time-to-event: Cox proportional hazards for serious delinquency and prepayment as
> competing risks, plus a monthly Markov transition model.
>
> Cox test concordance is **0.72** on default and 0.68 on prepayment, and the hazard ratios
> run the right way — debt-to-income 1.11, LTV 1.06, both significant.
>
> Read the transition matrix: 98.2% of current loans stay current, 1.3% prepay each month,
> and about half of thirty-day delinquencies cure. Default and prepaid are absorbing."

**On screen:** `reports/survival_report.md`, `reports/markov_transition_matrix.csv`,
`reports/cumulative_incidence.csv`.

---

## Beat 9 — Anomaly examples (3:15–3:40)

> "Task 4 combines deterministic rules with an isolation forest. Every flagged record carries
> a driver attribution, and the queue is ranked by exception probability and anomaly score
> together. Here are the top twenty — pick one and read its driver and its reconciliation gap
> straight off the row."

**On screen:** `reports/anomaly_review_queue.csv`, top 20 rows.

---

## Beat 10 — Scenario output (3:40–4:05)

> "Three scenarios. Adverse raises unemployment three points and turns house prices negative —
> supervisory severity, and disclosed as constructed rather than observed, because this window
> contains no housing downturn at all.
>
> The transition engine takes the twelve-month cumulative default rate from **1.17% to 1.86%**
> and nearly doubles the delinquent stock. The loan-level engine shows essentially **no**
> credit response — and that is not a bug we hid. One realised macro path gives no
> cross-sectional variation in unemployment, so that channel is unidentified. We report the
> number as zero and size the credit stress from the transition engine instead.
>
> On the prepayment side the loan-level engine adds **5.1 points**, and the response is not
> monotone in incentive: loans already deep in the money are saturated, so the biggest move
> comes from loans just below the refinance threshold that the rate cut pushes across."

**On screen:** `reports/scenario_report.md`, sections 3 and 8, then
`reports/scenario_segment_prepay_by_rate_incentive.csv`.

---

## Beat 11 — Local explanation for one loan (4:05–4:25)

> "Global importance by SHAP, and local explanations per loan. Take the top row of the
> submission and read its default probability, its three drivers and its confidence band
> straight off the file.
>
> Short-horizon risk is led by behavioural signals — current status, recent delinquency.
> Twelve-month risk by structural ones — credit band, debt-to-income. Nobody imposed that
> ordering; both models saw the same features."

**On screen:** `reports/explainability_report.md`, local example section.

---

## Beat 12 — LLM reviewer note (4:25–4:45)

> "The copilot never sees the dataframe or the models. It receives a grounding pack — a JSON
> object of figures a non-LLM model already produced — and turns those into reviewer prose."

**On screen:** `reports/copilot_report.md`, then `submission/llm_prompt_log.jsonl`.

> **Say the execution mode that is on screen.** As committed it reads `offline_template`, so
> say: "No API credential was available in the build environment, so this ran in
> offline-template mode. The report states that in its first line rather than dressing
> deterministic text up as model output." If a key has since been set and the report says
> `live_api`, say that instead.

---

## Beat 13 — Rejected or corrected LLM output (4:45–5:05)

> "This is the part that matters. The grounding validator is ordinary code, not a prompt. It
> pulls every number out of the generated text and checks it against the pack. Invent a
> figure, rescale a real one, or turn an association into a cause, and the output is blocked
> before a reviewer sees it.
>
> Six adversarial cases, and the validator returns the expected verdict on every one — this
> one asserts a 41.7% default probability that appears nowhere in the pack, and is blocked."

**On screen:** `reports/copilot_validator_self_test.csv`, then the blocked cases in
`reports/copilot_report.md`.

---

## Beat 14 — Final submission file (5:05–5:20)

> "Per loan: four probabilities, exception probability and type, predicted next state, anomaly
> score, top drivers, a recommended action with its reason, and a confidence band. Every
> column traces to a fitted non-LLM estimator. The action is a documented rule over those
> numbers — not a model, not a language model — so the reason any loan got any action is
> reconstructible from its own row."

**On screen:** `submission/submission.csv`, then `submission/SUBMISSION_FORMAT.md`.

---

## Beat 15 — AI Development Log (5:20–5:30)

> "And the development log: tools used, what we accepted, what we rejected and why. Including
> the run that completed cleanly and regenerated every report from a stale cache — caught on
> two figures that could not be true, and fixed at the root."

**On screen:** `submission/AI_DEVELOPMENT_LOG.md`, section 12.

---

## Backup answers for likely questions

**"Isn't the exception label synthetic?"**
> Yes, and section 2 of the model card says so with a per-layer table. SFLLD has no second
> source, no ingestion timestamps and no document data. Every delinquency, default,
> prepayment and next-state figure is measured on real outcomes; every exception figure is
> measured against a fabricated label and is a demonstration of method.

**"Why not use real defaults?"**
> Because there are 14 of them in 16,000 loans. We report that number next to the proxy
> rather than hiding behind it.

**"Your boosted model loses to logistic regression."**
> On ranking, for two of the four targets, yes — and it is in the model card. It wins
> decisively on calibration, which is what the action thresholds consume. Reporting the clean
> sweep we did not get would have been the weaker choice.

**"What stops the LLM making things up?"**
> Code, not instructions. Show `src/copilot/validators.py`.

**"Can you reproduce this?"**
> `python -m src.pipeline` runs end to end on the synthetic generator with no external
> dependency. With the licence-gated Freddie Mac files in `dataset/`, the real path is three
> commands in the README.
