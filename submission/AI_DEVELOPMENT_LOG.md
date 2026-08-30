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
| Google Gemini API (`gemini-3.5-flash-lite`) | Runtime component only — the reviewer copilot in `src/copilot/`. Never used for prediction. Switched from the Anthropic Messages API in session 3; see section 14. |
| LightGBM / scikit-learn / lifelines / SHAP | The actual modelling stack. All predictive numbers come from these. |

**Hard rule enforced throughout:** no LLM produces a predictive number anywhere in this
system. The LLM reads model outputs and writes prose. Every probability, score, rate, and
ranked driver in `submission/submission.csv` traces to a fitted non-LLM estimator. This is
verified by a test (`tests/test_no_llm_prediction.py`) that fails if the copilot module is
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
## Tasks 3-5 — Survival, anomaly, scenarios

**Date:** 2026-08-28
**AI-generated code share (est.):** ~85% generated, 100% reviewed, ~25% rewritten.

### Task 3 — Time-to-event
Kaplan-Meier and Cox from `lifelines`, plus an empirical multi-state Markov chain.

**Accepted.** Handling three censoring mechanisms separately rather than as one bucket:
administrative right-censoring, competing risks, and left truncation. Left truncation matters
most here — 66% of loans enter the panel already seasoned, and crediting them with event-free
exposure at ages they were never observed at would have flattened the seasoning ramp.

**Rejected.** First draft reported `1 - KM` as the default probability. Rejected: with a 65%
cumulative prepayment rate, that treats prepaid loans as still at risk. Added Aalen-Johansen
cumulative incidence alongside. At loan age 108 months the naive figure reads 0.418 against a
true cumulative incidence of 0.192 — an overstatement of 0.227 in absolute probability, which
on a 10,000-loan book is provisioning for ~2,265 defaults that cannot happen.

### Task 4 — Anomaly and exception

**Rejected — the entire first anomaly feature set.** The isolation forest was first built on
raw record levels (balance, loan age, term, original balance). It scored a lift of **0.92x**
against the exception label — *worse than random selection* — because a genuinely large,
genuinely seasoned jumbo loan is a statistical outlier and a perfectly correct record. The
feature set was rebuilt around quantities where deviation means a *defect*: residuals against
what the record should say given its own other fields (amortisation vs term elapsed, DPD vs
reported status), servicer-feed disagreements, reporting timeliness and repair indicators.
Same model, new features: lift 2.02x, ROC-AUC 0.615 → 0.845.

**Accepted.** Keeping anomaly score, exception probability and exception type as three
separate models. A blended score would hide both failure modes — the clean-looking record
with a missing document file, and the wild outlier nobody needs to review.

**Instructive negative result kept in the report.** The exception model's logistic baseline is
deliberately the same nine credit fields used in Task 2, and it scores ROC-AUC 0.53. Operational
exceptions are not a credit phenomenon, and reusing a credit feature set for them solves the
wrong problem.

### Task 5 — Scenarios

**The finding that took the most work.** Engine A (model repricing) produced a near-zero
adverse-credit impact (+0.15% relative on 12-month default from a 2.3pp unemployment shock)
and, worse, a high-prepayment scenario that *raised* projected default — sign backwards. This
is not a tuning problem. Macro levels are constant across all loans within a month, so with a
single realised macro path there is no cross-sectional variation to identify them from; the
trees learn a calendar-time proxy, and in this panel's history low rates coincided with the
pandemic unemployment spike.

**Rejected — the first fix.** Perturbing only the cross-sectionally identified
refinance-incentive features while holding macro levels at observed values. It made things
worse: it hands the model a combination that never occurs in training (market rate 5.5%
alongside an incentive computed against 5.74%) and the *base case* prepayment projection —
which should be a no-op — jumped from 0.156 to 0.396. Reverted to internally consistent
shifts plus an explicit statement of what the resulting credit number is worth.

**Accepted.** Engine B, a macro-conditioned Markov chain that regresses transition log-odds on
the macro path and extrapolates through a logistic link. It carries the credit stress
properly: adverse conditions move cumulative 12-month default from 17.4% to 22.7% and roughly
triple the delinquent stock (4.1% → 12.2%). The division of labour is stated plainly in the
report: **Engine B sizes the stress, Engine A says which loans.**

*Lesson:* when a model gives an implausible scenario answer, check identification before
reaching for hyperparameters. No amount of tuning recovers a coefficient the data cannot
identify.

---
## Tasks 6-7 — Explainability and the LLM copilot

**Date:** 2026-08-28
**AI-generated code share (est.):** ~85% generated, 100% reviewed, ~20% rewritten.

### Task 6

**Accepted.** Computing SHAP against the raw LightGBM margin rather than the calibrated
probability. SHAP values are additive in log-odds; explaining the calibrated probability
breaks additivity and the contributions stop summing to anything. Local explanations report
log-odds contributions *and* the calibrated probability, labelled separately.

**Rejected — my own narrative, not the code.** The first draft of the cross-model
observations asserted that "delinquency and default are dominated by payment history, not
origination credit attributes." Checking it against the actual SHAP output showed that is only
true for the 3-month model; the 12-month default model is led by credit band, DTI band and
note rate. Rewritten to the accurate and more interesting finding: **short-horizon risk is
behavioural, long-horizon risk is structural**, and both models saw identical features.

*This is the failure mode to watch for when an AI assistant writes reports: plausible summary
prose that was never checked against the numbers it summarises.*

### Task 7

**Accepted — the grounding validator.** The system prompt tells the model not to invent
numbers. That is not a control, it is a request. The validator extracts every number from the
generated text and matches it against the grounding pack, including values scaled by 100 or
rounded, which are the forms a helpful model reaches for. Unmatched numbers block the output.

**Rejected — reporting an 11/11 validator pass rate as evidence.** The offline template only
ever emits pack numbers, so a 100% pass rate proved nothing at all. Added a six-case self-test
that feeds the validator deliberately bad output: a fabricated 41.7% probability, a rescaled
figure, a causal assertion, an overconfident foreclosure directive, and a note with no
reviewer framing, plus one clean case. All six behave as specified. *A validator that has only
seen well-behaved output is untested.*

**Not done, and stated rather than faked.** No Anthropic credential was available in the build
environment. The five adversarial probes are defined in code and execute, but against the
deterministic offline template — which cannot hallucinate and therefore cannot demonstrate the
failure modes the probes exist to catch. Writing plausible-looking transcripts and presenting
them as captured API output would be fabricating evidence, so that section says what is
missing and exactly how to produce it. This is the one incomplete item in the submission.

---

## Deliverables, model card and the drift problem

**Date:** 2026-08-28

**Rejected — a hand-written model card.** The first model card was written by hand with
metrics copied from the reports. Within one retraining run six figures were already stale
(3-month ROC-AUC quoted 0.892 against an actual 0.900; next-state macro-F1 0.440 against
0.434). Replaced with `src/report_model_card.py`, which authors the narrative but reads every
figure from the pipeline's own report CSVs. Retraining regenerates the card and the numbers
cannot drift away from the models.

**Accepted.** Making `run_rules` idempotent after a test caught that re-running it on an
already-flagged frame raised a column-overlap error. Small bug, found by a test that existed
for a different reason.

---

## 9. Human review process

Three lenses on every AI-generated module, applied before it was kept:

1. **Leakage lens** — could this feature or split let the model see the future, or let the
   same `loan_id` sit on both sides of a boundary? Caught the forward-target censoring bug and
   drove the whole purge/embargo/observability design.
2. **Numeric lens** — run it, print the distribution, sanity-check against the data-generating
   process. Caught the loss-severity skew, the 0.92x anomaly lift, the near-zero adverse
   scenario, and the base-case prepayment jump from 0.156 to 0.396.
3. **Judge lens** — does this map to a rubric line item, or is it decoration?

**What the AI was good at.** Vectorised numerical code, boilerplate across parallel modules
(five report writers with the same shape), test scaffolding, and remembering to handle the
edge cases of an API it had just been told about.

**What it consistently got wrong without review.** The *edges* — the ends of the time axis,
the tails of distributions, and the boundary between correlation and identification. Every
significant defect in this build was at one of those three edges:

| Defect | Edge | Caught by |
|---|---|---|
| Forward targets zero-filled instead of censored | End of time axis | Leakage lens |
| Loss severity 83% in the top band | Tail of a distribution | Numeric lens |
| Isotonic calibration selected on its own fitting data | Model-selection boundary | Numeric lens |
| Anomaly features measuring size, not defect | Tail vs. defect confusion | Numeric lens |
| Macro credit channel unidentified | Correlation vs. identification | Numeric lens |
| Narrative claim contradicted by the SHAP output | Prose vs. evidence | Judge lens |

---

## 10. Rough AI-generated code share

| Component | Generated | Rewritten after review |
|---|---|---|
| Data generation and messiness | ~90% | ~20% |
| Profiling and validation | ~85% | ~15% |
| Features and splits | ~80% | ~35% |
| Models, calibration, metrics | ~85% | ~30% |
| Survival and transitions | ~85% | ~25% |
| Anomaly and exceptions | ~85% | ~30% |
| Scenarios | ~85% | ~40% |
| Explainability | ~85% | ~20% |
| Copilot | ~80% | ~25% |
| Tests | ~90% | ~10% |
| Reports and narrative | ~75% | ~40% |
| **Overall** | **~85%** | **~27%** |

Every line was read before being kept. The rewrite share is the honest measure of how much
review actually changed.

---

## 11. Lessons

1. **Check the edges of the time axis first.** When an assistant writes label construction for
   forward-looking targets, the bug will be at the panel boundary. It was here, and it would
   have inflated every metric in Task 2.
2. **A model giving an implausible answer is a specification problem more often than a tuning
   problem.** The adverse-credit scenario returning +0.15% was not fixable with
   hyperparameters — the macro credit effect is not identified from one macro path, and no
   amount of tuning recovers a coefficient the data cannot support.
3. **Validators need adversarial tests, not happy-path ones.** An 11/11 pass rate on
   well-behaved input is not evidence a guard works.
4. **Generate documents that contain numbers.** Hand-copied metrics were stale within one
   retraining run.
5. **The most dangerous AI output is fluent summary prose.** Code that is wrong usually
   crashes or produces an obviously bad number. A paragraph confidently describing what the
   SHAP values show, written without looking at them, passes review unless you specifically go
   and check.
6. **Report the honest comparison.** LightGBM tying a nine-feature logistic model on ranking
   was not the result I wanted, and the reasoning behind keeping the GBM anyway — calibration,
   not discrimination — is more defensible than a claimed clean sweep would have been.

---

## 12. Session 2 (2026-08-30) — switching from synthetic to real Freddie Mac data

The original build ran on a synthetic generator because the organiser data pack described in
section 6 of the problem statement was never issued. Five Freddie Mac SFLLD vintage sample
folders became available, and this session replaced the panel with real data.

**Why switch at all.** With a self-authored generator, the models partly recover the
generator's own coefficients. That is circular, it is visible to a judge, and it undercuts the
two heaviest rubric blocks — Data Intelligence (15) and Predictive Modelling (20). Real data
removes the objection outright and supplies the scale the problem statement asks for.

### The layout was not what the documentation said

The assistant's first instinct was to map columns using Freddie Mac's published layout. That
would have been wrong. The official `file_layout.xlsx` and the January 2026 General User Guide
both specify **32 origination and 32 performance fields**; the sample files on disk carry
**31 and 35**. Mapping to the stock layout would have silently mis-assigned seven origination
columns and produced plausible-looking garbage — `Super Conforming Flag` read as
`Servicer Name`, and so on.

Resolved empirically instead, by profiling every column's value distribution across all five
vintages. The signatures are unambiguous:

| Column | Observed values | Identifies it as |
|---|---|---|
| orig 26 | 99.9% blank, rest formatted `F08Q20307907` | Pre-HARP Loan Sequence Number |
| orig 28 | `Y` present in 2019, never in 2023 | Relief Refinance Indicator (HARP ended) |
| orig 29 | code `4` appears only from 2022 | Property Valuation Method (ACE+PDR, effective Jul 2022) |
| perf 33 | exactly `7`/`N`/`Y` | MI Cancellation Indicator |
| perf 34 | real servicer names | Servicer Name |

Both relocated fields are time-varying, which explains the move to the monthly file.
`sflld.verify_layout()` now asserts 31/35 and refuses to load anything else, so a
re-download with a different layout fails loudly instead of quietly mis-mapping.

**Lesson:** when documentation and data disagree, the data is right. Verify the schema against
the bytes before writing a loader, not after the metrics look odd.

### Rejected: calibrating the adverse scenario to observed history

The assistant wrote scenario calibration that anchored each shock to the largest comparable
move in the same real series — which is exactly the right instinct, and produced an
unusable result. The largest 12-month unemployment rise in the window is **+11.1pp**: the
COVID spike to 14.8% in April 2020. An "adverse credit" scenario built on a once-in-a-century
labour shock is not a credit scenario.

Excluding COVID (2020-01 to 2021-06) gives the opposite problem — the largest rise is
**+0.7pp**, which stresses nothing. The window simply contains no credit downturn and no
housing downturn.

Rejected both and used supervisory (DFAST/CCAR-style) magnitudes instead: **+3.0pp
unemployment, -10% HPI YoY**. All three figures are written into the scenario file's
assumption note, so a reader sees the observed bounds, the exclusion, and the override rather
than a number presented as if it were empirical. The high-prepayment scenario needed no
override — its -1.19pp rate decline is taken straight from observed history.

### Caught: the pipeline regenerated every report from stale data

The most dangerous defect of the session. `prepare()` caches the engineered feature frame, and
`--skip-data` disabled cache invalidation. The first full run after the switch completed
cleanly, exit code 0, and rewrote every report — all of it from the **August 28 synthetic
frame**. Nothing crashed. Nothing warned. The reports looked entirely current.

Caught by the numeric lens, on two figures that could not be true:

- `train_loans: 1379` in `split_summary.csv`, against 16,000 loans in the new panel.
- a test window ending **2026-06**, when the real data ends 2026-03.

Fixed at the root rather than by deleting the file: `dataset._cache_is_stale()` compares the
cache mtime against `loan_panel.csv`, `servicer_updates.csv` and `macro_history.csv` and
forces a rebuild when the pack is newer. The same guard now protects the model card, which
reads `artifacts/sflld_build_summary.json` but treats it as stale if the panel was written
after it — so a synthetic run cannot inherit real-data prose.

**Lesson, and it generalises past this project:** a cache plus a "skip the slow step" flag is a
silent-staleness machine. Any pipeline offering both needs an invalidation rule, because the
failure mode is not a crash — it is a full set of confident, wrong, freshly-dated reports.

### The default target had to be redefined, and it is disclosed

Realised credit events (zero-balance codes 02/03/09/15) occur on **14 of 16,000 sampled
loans** — under a tenth of a percent, roughly one row in 200,000. These are post-2019 agency
vintages with strong house-price appreciation and pandemic forbearance behind them; the
scarcity is a property of the cohort, not a sampling artefact.

`next_12m_default_flag` is therefore a **90+ DPD proxy**. This is a real deviation from the
problem statement's field list, so it is stated in the module docstring, in the data
intelligence report, and in its own subsection of the model card, with the realised-event
count reported next to it. A serious-delinquency model labelled "default" without that
disclosure would be the kind of quiet redefinition this log exists to prevent.

### Sampling: rejected the rare-event oversample

The instruction was to downsample loans while keeping all rare-event (90+ DPD) loans.
Implemented and then rejected: retaining all 7,878 ever-90+DPD loans from the 250,000-loan
population while sampling the rest down would lift their share from 3.2% to about 40% and
destroy the base rates that calibration and the scenario engine depend on.

Used a plain loan-level stratified random sample instead — 3,200 per vintage, 16,000 loans,
670,548 monthly rows — which preserves true prevalence and still yields 9,989 positive rows on
the 12-month default proxy. Sampling is at loan level only, so no retained loan has its history
truncated and no rare event is cut mid-path. The intent behind the instruction is met; the
literal mechanism was not the way to meet it.

### Not done, and not faked: live LLM copilot

> **Superseded in session 3 (2026-08-30).** A Gemini key became available and the copilot now runs live; see section 14. This entry is left standing as written because it records what was true when it was written, and because the decision it describes — refusing to fabricate transcripts while waiting for a credential — is the reason there was nothing to unwind when the key arrived.

No `ANTHROPIC_API_KEY` is present in the build environment, so `src/copilot/` runs in
`offline_template` mode. The option of writing plausible reviewer transcripts and presenting
them as captured API output was available and was **not taken** — section 13 of the problem
statement lists fabricated results as a disqualification condition, and the distinction between
"the model said this" and "this is what the model would plausibly say" is exactly the one this
whole layer exists to police.

The grounding validator itself is ordinary code, not a prompt, and it is genuinely exercised:
`run_self_test()` feeds deliberately bad outputs through it — a fabricated probability, a
rescaled real figure, a causal assertion, an overconfident decision, missing reviewer framing,
and one clean case — and asserts the expected verdict on each. (Session 3 grew this suite
from six cases to ten.) That evidence is real regardless
of execution mode. The copilot report states its mode in the first line rather than burying it.

### Prepayment degrades out of time, and that is reported

On real data the prepayment model's PR-AUC drops sharply from validation to test while the
positive rate more than doubles. This is the rate cycle doing exactly what it should: the test
window sits in a different rate regime from training. It is a genuine limitation of a
single-realised-macro-path fit, it is recorded in the model card rather than smoothed over,
and it is the strongest argument in the build for the macro-conditioned transition engine over
the loan-level classifier for anything scenario-shaped.

### AI-generated code share, this session

| Component | Generated | Rewritten after review |
|---|---|---|
| SFLLD layout module and loader | ~85% | ~30% |
| Panel derivation and target logic | ~80% | ~35% |
| Real macro sourcing and scenarios | ~85% | ~45% |
| Model card / report updates | ~75% | ~35% |
| **Session overall** | **~82%** | **~35%** |

The rewrite share is higher than session 1 (~27%), and the reason is worth recording: real
data has edges that a generator does not. Sentinel values, an undocumented layout, a target
that barely exists, and a macro window with no downturn in it are all things the assistant
handled reasonably on the first pass and correctly only after the numbers were checked against
the actual bytes.

---

## 13. Representative prompts, verbatim

Section 8 of the problem statement asks for representative prompts, and an earlier compliance
audit correctly flagged that this log described *outcomes* without showing a single *input*.
These are actual prompts issued during development, quoted as sent. Long prompts are excerpted
at the marked points; nothing is paraphrased or reconstructed.

### 13.1 The prompt that produced the data diagnostic

> "Five Freddie Mac SFLLD sample-file folders are now in `dataset/` [...] **Step 2 — Map
> columns using Freddie Mac's official file layout.** SFLLD files are position-based, not
> self-describing. Find Freddie Mac's current SFLLD file layout/glossary (search if needed, or
> check if a layout doc came bundled in the download) and map each column index to its real
> field name for both the origination and performance files."

**Why it worked.** It named the failure mode — position-based, not self-describing — and
required the layout to be *found* rather than recalled. The assistant's first instinct was to
map against the published 32/32 layout, which is wrong for these files. The instruction to go
and check is what surfaced the 31/35 discrepancy before a loader was written, rather than
after the metrics looked strange.

### 13.2 The prompt that set the gate

> "**Hard gate at hour 6:** the switch is only worth continuing if, by then, you have a clean
> `loan_panel.csv` that the existing pipeline (`python -m src.pipeline --skip-data` or
> equivalent) accepts AND the D90+ target has a workable class balance after derivation (not
> still ~1-in-2000). If either condition fails at hour 6, abandon the real-data branch, return
> to the last known-good synthetic commit, and say so plainly rather than continuing to sink
> time into it."

**Why it worked.** Two falsifiable conditions and a named fallback. The value was not that the
gate fired — it passed at roughly one hour, 1.84% against a 1-in-2000 floor — but that the
work was framed as abandonable from the start, which kept the first hour aimed at the two
things that would decide it instead of at polish.

### 13.3 The prompt that prevented a fabrication

> "If `ANTHROPIC_API_KEY` is available in this environment, run it for real against actual
> model output. [...] Do not write plausible-looking transcripts and label them as captured
> API output — if no key is available, say so plainly in the copilot report and model card
> rather than faking it."

**Why it worked.** It pre-committed the honest branch before the answer was known. No
credential existed, and the temptation to generate reviewer prose that would have read
perfectly well is real — an LLM writing plausible LLM transcripts is the single easiest
fabrication available in this project. Naming the prohibition in advance removed the decision
from the moment it would have been made under time pressure.

### 13.4 A prompt whose literal instruction was not followed

> "loan-level downsampling to a workable row count keeping all rare-event (D90+) loans"

Implemented, measured, then rejected. Retaining all 7,878 ever-90+DPD loans from the
250,000-loan population while sampling the remainder down would lift their share from 3.2% to
roughly 40% and destroy the base rates that calibration and the scenario engine consume. A
plain loan-level stratified sample was used instead, preserving true prevalence and still
yielding 9,989 positive rows.

**Recorded because the disagreement is the useful part.** The instruction's intent — do not
lose rare events to downsampling — is met, by sampling at loan level so no retained loan has
its history truncated. The literal mechanism was not the way to meet it. An assistant that had
silently complied would have produced a calibrated model against a fabricated base rate.

### 13.5 The standing instruction that caught the most defects

> "**Model card freshness.** Regenerate from current report CSVs after whichever data path is
> finalized — confirm no stale figures from an earlier run."

**Why it worked.** Applied as a general suspicion rather than a single check, this found five
separate stale-figure defects: the cached feature frame that regenerated every report from the
previous data source; hardcoded scenario figures ("17.4% to 22.7%") surviving regeneration; a
SHAP claim about prepayment buckets that real data contradicts; a left-truncation share of 66%
that is 5% here; and an anomaly comparison quoting pre-change numbers as if freshly measured.

Only the first was a code bug. The other four were **English sentences containing numbers** —
which is the failure mode a test suite does not catch and a reader cannot distinguish from a
computed figure. The rule that came out of it is in section 11: generate documents that
contain numbers, and treat any hand-typed digit in narrative prose as a defect until proven
otherwise.

### 13.6 On prompt style, from this session

What produced good output, consistently:

- **Name the failure mode, not just the task.** "SFLLD files are position-based, not
  self-describing" did more work than "map the columns" would have.
- **Give falsifiable stopping conditions.** "not still ~1-in-2000" is checkable; "reasonable
  class balance" is not.
- **Pre-commit the honest branch.** Deciding what to do if no API key exists, before knowing
  whether one exists, is worth more than any instruction issued afterwards.
- **Ask for the disagreement.** "say so plainly rather than continuing to sink time into it"
  and "recommend one path plainly" both produced direct answers where an open-ended request
  would have produced a survey of options.

---

## 14. Session 3 (2026-08-30) — wiring the copilot to Gemini

A `GEMINI_API_KEY` became available, so the copilot moved off `offline_template` and ran for
real. The instruction was explicit that this was a provider choice, not a rescue, and the
record should reflect that.

### Why Gemini, stated plainly

Cost and availability. `gemini-3.5-flash-lite` is free-tier eligible, so the whole Task 7
deliverable reproduces for anyone holding a free Google AI Studio key. An assessment artefact
that only runs for someone with a paid credential is worth less than one that runs for
anybody. This was a deliberate selection, not a fallback after an Anthropic failure — the
Anthropic path worked, it was simply not the one available here.

The specific model was chosen by measurement rather than from the docs. The first live run
used `gemini-3.6-flash`, which writes noticeably better prose, and it died partway through
with `ResourceExhausted`: the free allowance for that model is **20 requests per day**
(`GenerateRequestsPerDayPerProjectPerModel-FreeTier`, `quota_value: 20`). A single Task 7 run
issues 15-20 calls, so one attempt would have burned the day's quota and left the deliverable
un-rerunnable. The lite tier clears a full run with headroom. Two things came out of that:
the model default changed, and the client learned to tell a per-**day** quota from a
per-minute one, because retrying the former with backoff just burns wall time — the first run
spent 125 seconds per call retrying a cap that was never going to clear.

### What actually changed in the code

Only the client, auth and response-parsing layer. Grounding packs, the system prompt, the
validators and the adversarial probes are the same objects they were under Anthropic, which
was the point of keeping the LLM behind a grounding pack in the first place. The vendor swap
touched `src/copilot/client.py` and nothing in the copilot's design.

The import guard in `tests/test_no_llm_prediction.py` was rewritten to be vendor-neutral in
the same commit. It previously forbade `anthropic` and `openai` by name, which would have
silently stopped enforcing anything the moment the project moved to Google. The constraint is
"no LLM in the modelling path", not "no Anthropic in the modelling path", so it now blocks
`anthropic`, `openai`, `google`, `cohere`, `mistralai` and `ollama` alike.

### The first check was the one that mattered

Before any code changed, the standing instruction was to confirm that no prediction path
sends loan records to an LLM for classification. It held: the only call site is
`Copilot.ask`, reachable only from `run_copilot.py`, and every input to it is a grounding
pack built from figures that LightGBM, SHAP, the isolation forest, Cox and the Markov chain
had already produced. Nothing needed flagging. Worth recording that the check was run against
the *capability* rather than the vendor, because a provider swap is exactly the kind of change
under which a name-based guarantee quietly stops guaranteeing anything.

### What Gemini got wrong

Three genuine failures, all caught and all corrected on a logged round-trip.

1. **A 10x transcription error, twice.** Gemini drops a decimal place restating small
   probabilities: `exception_required` reported as `0.046` where the pack said `0.0046`, and
   on an earlier run `0.042` for `0.0042`. The earlier one is the more interesting: it
   appended its own parenthetical noting that the pack said `0.0042`, and then led with the
   wrong figure anyway. It caught its own error and published it. This is precisely the
   failure a reviewer cannot catch by eye — correctly formatted, plausible, wrong by one
   decimal place — and precisely what a validator comparing against the pack does catch.
2. **Null advice.** A reviewer note whose "check this first" instruction pointed at a
   document status the same pack reported as `complete`. True, well-formed, and it told the
   reviewer to go and look at nothing. Nothing in the existing controls had an opinion about
   it, because they were all truthfulness controls.
3. **LaTeX in a plain-text queue.** A portfolio summary rendered every scientific-notation
   figure as MathJax (`$-2 \times 10^{-5}$`), which the servicing queue does not render.

Failure 2 produced a new control (`usefulness_validator`) rather than a prompt tweak, because
a prompt tweak would not have been checkable. It is deliberately narrow: it fires only when
the text steers the reviewer at a field the grounding pack itself reports as clean, which is
a question the pack can settle. General vagueness is not mechanically detectable and no claim
is made that it is caught.

### Where the validator was wrong, which was more often than Gemini

The uncomfortable finding. Of the outputs blocked across the live runs, most were correct
Gemini output that the validator flagged in error:

- `-2e-05`, copied verbatim from the pack, split into `-2` and `-05` and reported twice as
  ungrounded.
- `next-3m-delinquency` — a field name — read as the number `-3`.
- The credit band `580-619` tokenized as `-619` by `grounding.py` and as `619` by
  `validators.py`, so a figure copied verbatim out of the pack was "ungrounded" purely
  because the two sides of the comparison disagreed about what a number is.
- `15.` and `17.` — markdown ordered-list markers — read as the numbers 15 and 17. The
  rule-suggestion task enumerates the existing rule set, so its output is inherently a
  numbered list and it tripped this every time.
- "three draft candidate rules **for human review**... drafts requiring human review before
  implementation" blocked for missing reviewer framing, because the check demanded the
  literal string `reviewer` and the model wrote `review`.

That last one is the root cause worth naming: **there were two number-parsing regexes that
had to agree and no mechanism making them agree.** They were consolidated into one
(`grounding.NUMBER_TOKEN_RE`), imported by the validator, so they cannot drift apart again.
Each of the three is pinned by a self-test case that fails if it regresses.

These mattered more than they look. A validator that cries wolf on correct output trains a
reviewer to wave blocks through, which costs more than the errors it was built to catch. After
the fixes a live run passes 11 of 12 outputs, against 6 of 12 at the worst point — and the
improvement is almost entirely the validator getting more accurate, not the model getting
better. Two false positives are being kept anyway: a refusal that quotes the blacklisted phrase it is
refusing to use, and a refusal that echoes the question's own "24 months". Narrowing the
check to admit them would open a gap a real failure could walk through — a model can refuse
and still slip a fabricated number into the refusal — so the bias stays toward blocking
correct output rather than releasing incorrect output.

### The self-test was not a regression test

Worth recording separately because it is the kind of defect that makes every other number in
this section untrustworthy. `run_self_test()` was called with whichever loan pack the live run
happened to pick. Its cases assert things like "this figure is ungrounded" and "this figure is
grounded", which are only true relative to a pack — so a case silently flips to a pass the
moment a real loan happens to carry a number near the fabricated one, and flips to a failure
whenever the picked loan lacks the grounded one.

It showed up as the suite reporting **8 of 10 on one run and 10 of 10 on another with no code
change between them**. Two cases were data-dependent. A regression test whose verdict moves
with the input is not a regression test, and worse, it had been cited as evidence in the model
card and the demo script.

Fixed by giving the suite a fixed `SELF_TEST_PACK` containing exactly the figures its cases
reference, and defaulting `run_self_test()` to it. Now deterministic at 12/12. The two cases
that had been quietly wrong are the two that were added most recently — new tests are exactly
where this class of defect hides, because nobody has yet seen them fail for the right reason.

### An ablation that came out negative, and is reported that way

Rule 7 ("write plain prose, no LaTeX") was added to the system prompt after the MathJax
incident, and the next run came back clean. That is not evidence: the model is sampled, and it
might simply not have reached for LaTeX. So the same pack was run three times with the rule
and three times without it (`src/copilot/ablation_latex.py`).

LaTeX did not reappear in either arm. **The ablation gives no evidence that rule 7 is what
fixed anything**, and the copilot report says so. The markup was most likely low-frequency
sampling behaviour these samples did not hit. The rule is kept because it costs nothing, but
it is not claimed as the fix. What is load-bearing is detection, which does not depend on the
model's cooperation: the validator now recognises LaTeX, normalises the figure inside it
*before* checking grounding — so the markup is not additionally mis-reported as a fabricated
number — and blocks with that named as the reason.

### A logging bug that was eating the evidence

`run_copilot` deleted the prompt log at the start of every run. Since captured failures are
the deliverable for this task, each run was destroying the evidence the run before had
collected — which is how the original LaTeX transcript was lost. The log is now rotated into
`submission/llm_prompt_log_archive.jsonl` instead of unlinked.

The 10x error recurred later and is quoted verbatim from the live log, so it needed no
reconstruction. The LaTeX transcript did not recur and its log line is gone; it is described
from the analysis made at the time and deliberately **not** written out as a quoted
transcript, because reproducing a log entry that no longer exists would be fabricating
evidence however accurate the reconstruction. The same reasoning as session 2, applied to a
case where it was tempting to cut a corner because the text had genuinely once existed.

### AI-generated code share, this session

| Area | AI share | Human-directed revision |
|---|---|---|
| Gemini client / auth / response parsing | ~85% | ~30% — quota handling and the empty-response path were rewritten after live failures |
| Validator fixes (regex consolidation, LaTeX, usefulness, list markers, framing) | ~70% | ~45% — each fix was driven by a specific observed failure, not written speculatively |
| Self-test determinism fix | ~60% | ~50% — the defect was found by noticing the same suite report two different scores |
| Report and log prose | ~80% | ~35% |
