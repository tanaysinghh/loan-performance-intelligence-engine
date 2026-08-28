# AI Development Log
**Project:** Loan Performance Intelligence Engine
**Event:** Intain Campus FinTech Challenge 2026 — AI Track, Round 2
**Author:** Tanay Singh
**Log started:** 2026-08-28

This log is maintained *during* development, appended at each task boundary. It records
which AI tooling was used, what was accepted or rejected, how outputs were reviewed, and
what was learned.

---

## 0. Tooling and working method

| Tool | Role in this project |
|---|---|
| Claude Code (Opus 5) | Agentic pair-programmer: scaffolding, module implementation, refactors, test authoring, report generation |
| Anthropic Messages API (`claude-sonnet-5`) | Runtime component only — the reviewer copilot in `src/copilot/`. Never used for prediction. |
| LightGBM / scikit-learn / lifelines / SHAP | The actual modelling stack. All predictive numbers come from these. |

**Hard rule enforced throughout:** no LLM produces a predictive number anywhere in this
system. The LLM reads model outputs and writes prose. Every probability, score, rate, and
ranked driver in `submission/submission.csv` traces to a fitted non-LLM estimator. This is
verified by a test (`tests/test_no_llm_leakage.py`) that fails if the copilot module is
importable from any modelling path.

**Human review process.** Every AI-generated module was reviewed under three lenses before
being kept:
1. **Leakage lens** — could this feature or split let the model see the future, or let the
   same `loan_id` sit on both sides of a split?
2. **Numeric lens** — run it, print the distribution, does the number pass a sanity check
   against the data-generating process?
3. **Judge lens** — does this map to a line item in the rubric, or is it decoration?

Rejections are logged below with reasons. Rough AI-generated code share is estimated per
task and totalled in section 9.

---
## Task 0 — Scaffold and synthetic data pack

**Date:** 2026-08-28
**AI-generated code share (est.):** ~90% generated, 100% reviewed, ~20% rewritten by hand after review.

**What was built.** `src/data/generate_synthetic.py` (monthly state machine),
`src/data/messiness.py` (defect injection + second servicer feed),
`src/data/exceptions_label.py` (operational exception labelling + latest-wins reconciliation),
`src/data/build_dataset.py` (entrypoint). Output: 53,972 rows / 1,900 loans / 54 months
(2022-01 to 2026-06), plus `servicer_updates.csv`, `macro_history.csv`, `macro_scenarios.csv`,
`data_dictionary.csv` and a ground-truth defect log.

**Judgment calls flagged.**
- *Panel window vs. vintage window.* Loans originate 2018-01 onward but the panel only
  reports 2022-01 to 2026-06. This gives a realistic mix of seasoned and new loans and means
  `PaidOff` is structurally absent (no 180-month loan originated 2018+ matures inside the
  window). Documented rather than faked.
- *Terminal months are not reported.* A loan's last panel row is the month **before** it
  prepays or defaults; the outcome lands in `next_state`. This matches how a servicing tape
  drops terminated loans and it keeps `next_state` a genuine one-step-ahead target.
- *Pool characterisation.* Tuned hazards give a ~7% 12-month default rate and ~16% 12-month
  prepayment rate. That is a seasoned non-QM / alt-A profile under a rising-unemployment
  macro, not an agency prime pool. Stated in the model card so the numbers are not read as
  agency benchmarks.

**Accepted AI output.** Vectorised month-by-month simulation across all loans at once
(rather than a per-loan Python loop) — 54 iterations instead of 1.9M, runs in seconds.

**Rejected AI output.**
1. *First draft made every forward target a plain `shift(-k)` rolling max.* Rejected: rows in
   the last k months of the panel would have been silently labelled 0 instead of censored,
   manufacturing a fake negative class exactly where the test period sits. Replaced with
   explicit right-censoring — `NaN` unless the window is fully observed or an absorbing state
   closes it. 18.9% of rows are censored for the 12-month targets and are dropped from
   supervised training rather than treated as negatives.
2. *First draft made exceptions a deterministic function of rule breaches.* Rejected: any
   model would have hit AUC 1.0 and the task would have proved nothing. Replaced with
   breach → materiality threshold → per-type escalation probability → ~1.2% reviewer flip
   noise, which is how a real oversight queue behaves.
3. *First draft of loss severity put 83% of defaults in the `60+` band* because the LTV term
   was uncentred. Caught by the numeric lens (printed the distribution), recentred.

**Lesson.** The censoring bug is the one that would have quietly inflated every metric in
Task 2. Worth restating: when an AI assistant writes label construction, check the *edges of
the time axis* first — that is where forward-looking labels break.

---
