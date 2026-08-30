# Five-Minute Demo — Script and Storyboard

Maps one-to-one to the Five-Minute Demo Flow in section 14 of the problem statement. Fifteen
beats, 300 seconds. Each beat names the exact file or command to have on screen, so recording
is a matter of reading the narration while stepping through the listed artefacts.

**Recording setup.** Two windows: a terminal at the repo root, and an editor/preview showing
`reports/`. Have these open in tabs before you start, in this order:

1. `reports/data_intelligence_report.md`
2. `reports/model_performance_report.md`
3. `reports/survival_report.md`
4. `reports/anomaly_report.md` + `reports/anomaly_review_queue.csv`
5. `reports/scenario_report.md`
6. `reports/explainability_report.md`
7. `reports/copilot_report.md`
8. `submission/submission.csv`
9. `reports/ai_development_log.md`

Numbers in square brackets are placeholders — read them off the regenerated reports on the
day, and say the figure that is on screen. Never say a figure that is not visible.

---

## Beat 1 — Dataset and targets (0:00–0:25)

> "This is the Loan Performance Intelligence Engine. It runs on real Freddie Mac
> Single-Family Loan-Level data — five vintages, 2019 through 2023, 250,000 loans and 10.5
> million monthly records, from which we sample [N] loans and [R] monthly rows.
>
> The panel is one row per loan per month. We predict six things: delinquency at three and
> six months, default and prepayment at twelve, the next servicing state, and whether a
> record needs an operational exception raised.
>
> One deviation up front, and it's in the model card. Real credit events — foreclosure, short
> sale, REO — happen to just [14] of our [16,000] loans, under a tenth of a percent. That is
> not modellable. So the twelve-month default target is a documented 90-plus-days-past-due
> proxy. We say so everywhere it appears."

**On screen:** `dataset/download_sflld.md` (top), then `data/raw/loan_panel.csv` header.

---

## Beat 2 — Data profiling report (0:25–0:50)

> "Task 1 profiles before anything is trained. Distributions for every column, missingness
> patterns with mechanism tests, outliers, and invalid date relationships."

**On screen:** `reports/data_intelligence_report.md` — scroll the profile and missingness
sections. Then `reports/profile_numeric.csv`.

---

## Beat 3 — Top data-quality issues (0:50–1:15)

> "The defects aren't guessed at. We inject a known set at known rates and score the profiler
> against that ground truth, so the data-quality layer is validated, not eyeballed.
> [X] defect families, and the profiler recovers them at [Y]."

**On screen:** `reports/validation_rule_summary.csv`, then
`data/raw/ground_truth_defect_log.csv` side by side.

---

## Beat 4 — Feature engineering (1:15–1:35)

> "Features are built strictly from information available at or before the reporting month —
> loan age, balance trajectory, delinquency run-lengths, rate incentive against the real
> Freddie Mac survey rate, and servicer-level aggregates. The macro series are real: PMMS,
> BLS unemployment, Case-Shiller."

**On screen:** `src/features/build_features.py` — the feature list; then
`data/external/fred_MORTGAGE30US.csv`.

---

## Beat 5 — Time-aware split (1:35–1:55)

> "Splitting is by time, never by row. Train ends [2024-06], validation ends [2024-12], test
> is everything after. Loans are purged across the boundary so no loan appears on both sides,
> and a leakage probe re-checks that every run."

**On screen:** `src/models/splits.py`, then `reports/leakage_probe.csv`.

---

## Beat 6 — Baseline model performance (1:55–2:15)

> "Baseline first: regularised logistic regression on the same features."

**On screen:** `reports/model_metrics.csv` filtered to baseline rows.

---

## Beat 7 — Improved model performance (2:15–2:45)

> "Then gradient boosting with class weighting and isotonic calibration. On the
> twelve-month default target, ROC-AUC goes from [B] to [I], PR-AUC from [B] to [I], and the
> Brier score improves to [Z]. PR-AUC is the one to watch — positives are about [1.5%] of
> rows, so ROC-AUC alone would flatter us."

**On screen:** `reports/model_performance_report.md` — the comparison table, then the
calibration plot section.

---

## Beat 8 — Survival / transition output (2:45–3:15)

> "Task 3 is time-to-event. A Cox proportional-hazards model for default and prepayment as
> competing risks, plus a monthly Markov transition model over the seven servicing states.
> Here are the cumulative incidence curves, and here is the transition matrix — read the
> Current row: [p] stays current, [p] rolls to thirty days."

**On screen:** `reports/survival_report.md`, `reports/cumulative_incidence.csv`,
`reports/markov_transition_matrix.csv`.

---

## Beat 9 — Anomaly examples (3:15–3:40)

> "Task 4 combines deterministic rules with an isolation forest. Every flagged record gets a
> driver attribution, and the reviewer queue is ranked. Here are the top twenty — this one is
> flagged because [driver], and the reported balance disagrees with the servicer feed by
> [amount]."

**On screen:** `reports/anomaly_review_queue.csv` — scroll the top 20 rows.

---

## Beat 10 — Scenario output (3:40–4:05)

> "Three scenarios: base, adverse credit, high prepayment. Adverse raises unemployment three
> points and turns house prices negative — supervisory severity, and we disclose that it is
> constructed rather than observed, because this window has no housing downturn in it.
> Portfolio default rate moves from [a] to [b]; the [2022] vintage moves most."

**On screen:** `reports/scenario_report.md` — the segment table.

---

## Beat 11 — Local explanation for one loan (4:05–4:25)

> "Task 6. Global importance by SHAP, and local explanations per loan. This loan scores
> [p] for twelve-month default. The drivers are [d1], [d2], [d3] — and the model's own
> spread across boosting stages gives us a confidence band, here [high/medium/low]."

**On screen:** `reports/explainability_report.md` — the local example section.

---

## Beat 12 — LLM reviewer note (4:25–4:45)

> "Task 7 is the copilot. It never sees the dataframe or the models. It receives a grounding
> pack — a JSON object of figures a non-LLM model already produced — and turns those into
> reviewer prose. Here is a generated note."

**On screen:** `reports/copilot_report.md` — a released reviewer note, plus
`submission/llm_prompt_log.jsonl`.

> **Say the execution mode that is on screen.** If the report says `offline_template`, say:
> "This ran in offline-template mode — no API credential was available in the build
> environment, and the report says so rather than pretending otherwise."

---

## Beat 13 — Rejected or corrected LLM output (4:45–5:05)

> "And this is the part that matters. The grounding validator is ordinary code, not a prompt.
> It pulls every number out of the generated text and checks it against the pack. Invent a
> figure, rescale one, or turn an association into a cause, and the output is blocked before
> a reviewer ever sees it. Here are [N] blocked cases — this one asserted a 41.7% default
> probability that appears nowhere in the pack."

**On screen:** `reports/copilot_validator_self_test.csv`, then the blocked examples in
`reports/copilot_report.md`.

---

## Beat 14 — Final submission file (5:05–5:20)

> "The submission carries, per loan: four probabilities, exception probability and type,
> predicted next state, anomaly score, top drivers, a recommended action with its reason, and
> a confidence band. Every column traces to a fitted non-LLM estimator. The action is a
> documented rule over those numbers — not a model, and not a language model."

**On screen:** `submission/submission.csv`.

---

## Beat 15 — AI Development Log (5:20–5:30)

> "Task 8. The development log records the tools used, representative prompts, what we
> accepted, what we rejected and why, and the share of generated code that survived review."

**On screen:** `reports/ai_development_log.md`.

---

## Backup answers for likely questions

**"Isn't the exception label synthetic?"**
> Yes, and that is stated in the model card. SFLLD has no second source, no ingestion
> timestamps and no document data, so the reconciliation and exception layer is fabricated on
> top of the real panel. We anchored the fabricated servicer feed on real servicing
> transfers — 43% of loans genuinely change servicer — so the conflicts sit on real events.

**"Why not use real defaults?"**
> Because there are 14 of them in 16,000 loans. We report that number rather than hiding
> behind the proxy.

**"What stops the LLM from making things up?"**
> Code, not instructions. Show `src/copilot/validators.py`.
