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
## Tasks 1-2 — Data intelligence and loan performance prediction

**Date:** 2026-08-28
**AI-generated code share (est.):** ~85% generated, 100% reviewed, ~30% rewritten after review.

### Task 1
Built `loaders.py` (typed loading, latest-wins servicer reconciliation, auditable repair),
`validate.py` (17 named rules across six quality dimensions, severity-weighted record and
batch scoring), `profiling.py` (distributions, missingness mechanism tests, Cramer's V
association, functional dependencies, PSI/KS drift) and the report writer.

**Accepted.** Treating sentinels as *absence* rather than magnitude — masking `days_past_due`
9999 to NaN while keeping a `dpd_repaired` indicator. The indicator turned out to be a
genuinely predictive exception feature, so discarding it would have lost signal.

**Rejected.** The first profiling draft reported a single global missingness rate per column.
Rejected as useless: it hides the mechanism. Replaced with a chi-square test of each
missingness indicator against `servicer_name`, which shows the mechanism is MAR-conditional-
on-servicer for seven fields (Cramer's V up to 0.17). That changes the modelling decision —
row-dropping would have silently deleted two servicers' books.

### Task 2

**A generator flaw the modelling exposed.** The first panel ran 2022-01 to 2026-06 (54
months). With a 12-month horizon, an embargo and a label-observability cap, the 12-month
targets had a *train window containing only rising rates* and a test window in a falling-rate
regime. Prepayment AUC came out at 0.51 — a coin flip. This was not a modelling failure; the
panel was too short to support the validation design. The generator was rebuilt over 2019-01
to 2026-06 (90 months) with a full rate cycle (4.4% → 3.0% → 6.7% → 5.5%) and a COVID-shaped
unemployment path, so training spans both regimes. Prepayment AUC moved to 0.68.

*The lesson is the general one:* when a time-aware split with an embargo leaves a target with
one macro regime in training, the fix is more history, not a looser split.

**Accepted after ablation, not assumption.** Four features/capacity variants were run before
settling the feature set. `vintage_year` was dropped — it is a calendar-time proxy whose
levels are unseen in the test window, and removing it gained 0.004-0.008 test ROC-AUC on
every target. That exclusion and its reason are recorded in
`build_features.EXCLUDED_WITH_REASON`.

**Rejected (twice) — the calibration selection.**
1. First draft applied isotonic regression unconditionally. Rejected: isotonic maps score
   ranges to constants, and the ties measurably depressed PR-AUC (0.645 → 0.586 on the
   12-month default target). Ranking and calibration should not be traded against each other
   silently.
2. Second draft fitted both isotonic and Platt on the validation window and chose between
   them *on that same window*. Rejected: this systematically favours the more flexible
   calibrator, which can bend to validation noise. Replaced with 3-fold cross-validated
   selection inside the validation window. Platt now wins on three of four targets, ranking
   is preserved exactly, and expected calibration error still falls (0.023 → 0.012 on the
   3-month target).

**Rejected — the next-state baseline.** First draft benchmarked the multiclass model only
against persistence ("next state = current state"). Rejected as an unfair and uninformative
comparison: persistence emits no probabilities, so log loss and AUC could not be computed for
it, and it wins on accuracy purely because 95%+ of transitions are Current-to-Current. Added
an empirical Markov transition matrix estimated on the training window as the real baseline.
The covariate model beats it on macro-F1 (0.440 vs 0.375) and macro-AUC (0.886 vs 0.841) —
lift *over already knowing the current state*, which is the number that means something.

**Result reported honestly rather than spun.** LightGBM does **not** dominate the nine-feature
logistic baseline on ranking. It wins PR-AUC on three of four targets and prepayment ROC-AUC
by 0.03, but the baseline is within 0.01 ROC-AUC on delinquency and default and beats it on
prepayment PR-AUC. The dominant delinquency signals are near-monotone in the log-odds, which
is where linear models are hard to beat. The GBM is kept because its Brier score is 2-4x
better and the baseline's expected calibration error runs 0.16-0.27 — usable for ranking a
queue, not for answering "what is the probability". This is stated plainly in the report
rather than presented as a clean win.

**Leakage controls that fired.** The split-sensitivity probe is the headline: refitting the
same model under an unsound random row split lifts test ROC-AUC from 0.89 to **0.99**, and
prepayment from 0.68 to 0.996. That +0.10 to +0.32 gap is the exact amount a naive split
would have flattered this submission by.

---
