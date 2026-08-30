# PROGRESS — Loan Performance Intelligence Engine

**Branch:** `real-data-switch` (9 commits) · **Fallback:** `master` @ `91fe18d` · **Tests:** 40/40 · **Copilot:** live on Gemini

---

## Bottom line

The switch to real Freddie Mac data is **done and the gate passed**, and the LLM copilot is
**live on Google Gemini** — the last code gap is closed. **One deliverable still needs you:
recording the five-minute demo video.** Everything else is complete and verified.

The copilot ran for real against `gemini-3.5-flash-lite`: **15 calls in the current log,
82 in the retained archive, 97 retained in total.** Genuine model failures were
captured and corrected on logged round-trips, and — reported separately rather than counted
against the model — six validator false-positive classes were found and fixed at source.

> **On the prompt-log history, stated precisely.** `llm_prompt_log_archive.jsonl` is **not** a
> complete record of every call ever made. `run_copilot` used to delete the log at the start of
> each run, so roughly **six earlier runs (~87 calls) were destroyed** before that bug was
> found and fixed. The archive holds only calls from the rotation fix onward. That number is
> reconstructed from this session's run outputs, not from an artefact — the artefacts are
> exactly what the bug destroyed. The one piece of evidence genuinely lost with them is the
> original LaTeX transcript, which is described but deliberately not reconstructed.

---

## Phase 0 — Safety ✅ (~6 min)

- `dataset/` gitignored; negation fixed so `download_sflld.md` stays tracked.
- Verified `dataset/` was **never committed**: 0 tracked files, largest blob ever in history
  2.7 MB against ~1.2 GB of raw data. No history rewrite needed.
- Fallback `91fe18d` on `master`.

## Phase 1 — Real data switch ✅ (gate passed at ~1h against a 6h budget)

| Gate condition | Result |
|---|---|
| Pipeline accepts a clean `loan_panel.csv` | **PASS** — 33-column contract, 673,242 rows → 670,548 × 133 feature frame |
| D90+ balance not ~1-in-2000 | **PASS** — 1.81%, 9,989 positive rows |

**Data.** 16,000 loans / 673,242 monthly rows, sampled from 250,000 loans and 10,482,492
records across vintages 2019–2023. Panel 2019-01 to 2026-03.

**Layout.** Sample files carry **31 origination / 35 performance** columns, not the 32/32 in
Freddie Mac's published `file_layout.xlsx` or January 2026 User Guide. Verified empirically
across all five vintages; `verify_layout()` refuses to load anything that deviates.

**Macro is real** — FRED `MORTGAGE30US`, `UNRATE`, `CSUSHPINSA`, vendored in `data/external/`.
Scenario paths are constructed at supervisory severity with all three candidate calibrations
disclosed (COVID +11.1pp, ex-COVID +0.7pp, chosen +3.0pp).

**Default target is a 90+ DPD proxy.** Realised credit events hit 14 of 16,000 loans (0.09%).
Disclosed in the model card (own subsection), data intelligence report (call-out), README,
`SUBMISSION_FORMAT.md` and the data dictionary.

## Results (out-of-time test, one coherent run)

| Target | Base rate | ROC-AUC | PR-AUC | Lift | ECE |
|---|---:|---:|---:|---:|---:|
| Delinquency 3m | 3.1% | 0.916 | 0.650 | 20.7× | 0.003 |
| Delinquency 6m | 4.1% | 0.878 | 0.578 | 14.3× | 0.002 |
| Default 12m (D90+) | 1.5% | 0.921 | 0.532 | 34.6× | 0.004 |
| Prepayment 12m | 7.6% | 0.626 | 0.201 | 2.6× | 0.135 |

Cox concordance 0.72 default / 0.68 prepayment. Markov: Current→Current 0.982, ~50% DQ30 cure.
Scenario adverse credit 1.17% → 1.86% (Engine B); high prepayment +5.1pp (Engine A).

**Reported honestly:** the logistic baseline beats LightGBM on ranking for the default proxy
(PR-AUC 0.574 vs 0.532) and prepayment (0.685 vs 0.626 ROC). The GBM ships for calibration
(ECE 0.004 vs 0.223), and the model card says so rather than claiming a clean sweep.

## Defects found and fixed

1. **Stale feature cache.** A run with `--skip-data` regenerated every report from the
   previous data source, exit 0, no warning. Caught on two impossible figures. Fixed at root
   (`_cache_is_stale`).
2. **Survival stage crash.** `HIGH_OPS_SERVICERS` was a hardcoded synthetic pair → constant
   column → singular Hessian. Now derived from DQ scores, plus a zero-variance guard.
3. **Survival event redefined** to first entry into 90+ DPD. Terminal-state defaults gave 59
   events, 3 in the training window, c-index 0.53. Now 505 events, c-index 0.72 — and
   consistent with the proxy used everywhere else.
4. **Five stale narrative figures**, four of them English sentences containing numbers:
   scenario credit stress (17.4%→22.7% vs real 1.17%→1.86%), prepayment lift (13.0pp vs
   5.1pp), left truncation (66% vs 5.0%), unemployment shock (2.3pp vs 3.0pp), and an anomaly
   comparison quoting pre-change numbers as current. All computed or labelled as history.
5. **A finding that changed, not just a number.** The prepayment response is non-monotone in
   incentive (+0.232 for −0.5 to 0 vs +0.027 for >1.0): deep-in-the-money loans are saturated
   at 84% and have no headroom. Better result than the synthetic claim it replaced.
6. Other synthetic hardcoding purged: data dictionary, orphan feed IDs, missingness narrative.

## Phase 2 — Gap closure

| # | Item | Status |
|---|---|---|
| 1 | LLM copilot made real | ✅ **Closed — live on Google Gemini (`gemini-3.5-flash-lite`).** 14 logged calls this run, 52 retained. Reviewer notes, scenario summary, data-dictionary retrieval and rule suggestion all generated live. 3 genuine model failures captured with corrections; validator self-test now 10/10. Every output labelled *recommendation, not decision*. |
| 2 | submission.csv format | ✅ 16,000 × 21. All seven PS §6 elements mapped in `SUBMISSION_FORMAT.md`, pinned by 9 new tests. |
| 3 | Model card freshness | ✅ Regenerated via `write()` and post-dates every artefact it reads. Date and data source generated, not typed. |
| 4 | Demo video script | ✅ 15 beats, real figures, backup Q&A. |
| 5 | AI Development Log | ✅ §12 this session's work; §13 verbatim prompts. |
| 6 | Compliance re-audit | ✅ Re-audited. Tasks 1, 6, 8 moved PARTIAL → FULLY MET. §3.8 rewritten (was vacuous). Only Task 7 remains partial, on the API key. |

## Phase 3 — LLM copilot on Gemini ✅

**Provider switched from Anthropic to Google Gemini, deliberately.** Cost and availability:
`gemini-3.5-flash-lite` is free-tier eligible, so the whole Task 7 deliverable reproduces for anyone
holding a free API key rather than only for someone with a paid credential. The copilot
design is vendor-neutral — grounding packs, system prompt, validators and adversarial probes
are unchanged; only the client, auth and response-parsing layer differs.

**Step-0 constraint check passed.** No prediction path sends loan records to an LLM for
classification. The only call site is `Copilot.ask`, reachable only from `run_copilot.py`,
and every input is a grounding pack built from figures LightGBM, SHAP, the isolation forest,
Cox and the Markov chain already produced. The import guard in `tests/test_no_llm_prediction.py`
was rewritten to be vendor-neutral in the same change — it previously named `anthropic` and
`openai` only, and would have stopped enforcing anything the moment the project moved to
Google.

### What Gemini got wrong (real, logged, corrected)

| # | Failure | Evidence | Correction |
|---|---|---|---|
| 1 | **10x transcription error** — `exception_required` given as `0.046` where the pack says `0.0046`. Recurred on an earlier run as `0.042` for `0.0042`, where it appended its own note that the pack said `0.0042` and led with the wrong figure anyway. | Quoted verbatim, `copilot_report.md` §5 | Blocked on the number; round-trip returned `0.0046` |
| 2 | **Null advice** — a reviewer note whose "check this first" pointed at a document status the same pack reported as `complete`. True, well-formed, useless. | `llm_prompt_log_archive.jsonl` | New `usefulness_validator`; corrected to "the pack surfaces no specific item to check first" |
| 3 | **LaTeX in a plain-text queue** — portfolio summary rendered figures as `$-2 \times 10^{-5}$`. | Described, not quoted — original log line was destroyed by the pre-rotation delete bug, and reconstructing it would be fabricating evidence | Validator detects and normalises it; ablation reported **negative** (see below) |

### What the validator got wrong

Three false-positive classes, all fixed at source, each pinned by a self-test case:
`-2e-05` split into `-2` and `-05`; `next-3m-delinquency` read as `-3`; and the credit band
`580-619` tokenized as `-619` by `grounding.py` but `619` by `validators.py`. That last was
the root cause — **two number-parsing regexes that had to agree with no mechanism making them
agree**. Consolidated into one shared `grounding.NUMBER_TOKEN_RE`.

Two false positives are **kept deliberately**: a refusal quoting the blacklisted phrase it is
declining to use, and a refusal echoing the question's own "24 months". Narrowing the check
would open a gap a real failure could use, so the bias stays toward blocking correct output
over releasing incorrect output. The correction round-trip clears both automatically.

### A negative result, reported as negative

System-prompt rule 7 ("no LaTeX") was added after the MathJax incident and the next run came
back clean — which proves nothing, since the model is sampled. `src/copilot/ablation_latex.py`
ran the same pack 3× with the rule and 3× without. **LaTeX did not reappear in either arm**,
so there is no evidence rule 7 is what fixed it. The rule is kept because it costs nothing;
it is not claimed as the fix. Detection is the load-bearing part, and it does not depend on
the model's cooperation.

### Also fixed

- **A logging bug that was eating the evidence.** `run_copilot` deleted the prompt log at the
  start of every run, so each run destroyed the failures the one before had captured — which
  is how the LaTeX transcript was lost. Now rotated into
  `submission/llm_prompt_log_archive.jsonl`.
- **Per-day vs per-minute quota.** The client retried an exhausted *daily* cap with backoff,
  burning 125s per call on a limit that was never going to clear. The two are now
  distinguished and a daily exhaustion fails fast.
- Prompt log gained `provider`, `sdk`, `finish_reason`, `response_id` and `latency_seconds`.
- Every LLM output in the copilot report carries an individual **recommendation, not
  decision** label with its model, timestamp and token counts.

---

## Compliance status

- **Tasks 1–8: FULLY MET.** Task 7 closed this session — the copilot runs live on Gemini,
  with real captured failures and corrections rather than an architecture-only demonstration.
- **All 10 disqualification conditions: do not apply.** §3.8 now met on its merits with
  gitignore/history/redistribution evidence, not by "all data is synthetic".
- New deliverable `data/raw/validation_rules.json` (PS §6 named it; none was issued).

---

## Requires your manual action

1. ~~**`ANTHROPIC_API_KEY`**~~ — **closed.** The copilot now runs live on Google Gemini
   (`GEMINI_API_KEY`), which was a deliberate provider choice on cost and availability rather
   than a fallback: the model is free-tier eligible, so this reproduces for anyone with a free
   Google AI Studio key. Re-run with:
   ```bash
   export GEMINI_API_KEY=...
   python -m src.copilot.run_copilot
   python -c "from src.report_model_card import write; write()"
   ```
   Nothing is outstanding here. Two notes if you review it:
   - **Free-tier quota is a real constraint.** `gemini-3.6-flash` writes better prose but
     allows only **20 requests per day**; a Task 7 run issues 15-20, so it is effectively
     single-shot. `gemini-3.5-flash-lite` is the default for that reason and clears a full
     run with headroom.
   - **The validator was wrong more often than Gemini was.** Three false-positive classes
     were found and fixed at source (scientific notation split in two, hyphens in field names
     read as minus signs, and two number-parsing regexes that disagreed with each other).
     `reports/copilot_report.md` §5 separates genuine model failures from validator defects
     rather than reporting every block as model error.

2. **Record the demo video** — `reports/demo_video_script.md`, 15 beats with figures filled
   and the artefacts to have on screen listed in order.
3. **Final review before submitting** — suggest reading `submission/MODEL_CARD.md` §2 first;
   it carries the hybrid-provenance table and the D90+ redefinition, which are the two things
   a judge is most likely to probe.
4. **Merging `real-data-switch`** — I have not merged or pushed. `master` @ `91fe18d` is
   untouched as a fallback.

*No organiser/HackerEarth communication has been needed. The real-data gate passed, so no
go/no-go decision is outstanding from you.*

---

# Final status — 2026-08-30

## Fully done, verified, committed

| Area | State |
|---|---|
| Real-data switch | Gate passed. 16,000 loans / 673,242 rows from Freddie Mac SFLLD, five vintages |
| Tasks 1–6, 8 | Fully met, unchanged this session |
| **Task 7 (LLM copilot)** | **Closed.** Live on `gemini-3.5-flash-lite`; four grounded use cases; real failures captured with corrections; every output labelled *recommendation, not decision* |
| Validators | Two controls (grounding + usefulness). Self-test **12/12, deterministic** against a fixed pack |
| `submission.csv` | 16,000 × 21. Zero nulls, no duplicate `loan_id`, probabilities in [0,1], every action carries a reason. All seven PS §6 elements mapped |
| Model card | Regenerated last, post-dates every artefact it reads. Self-test count now data-driven so it cannot go stale |
| AI Development Log | Current through this session — §14 covers the Gemini switch, the validator defects, the negative ablation and the log-rotation bug |
| Compliance audit | Re-audited end to end post-Gemini. 8/8 judging criteria met; 10/10 disqualification conditions do not apply |
| Demo script | Finalised, 15 beats, mapped to PS §14 flow, every figure verified against current artefacts |
| Tests | 40/40 |

## Needs your eyes before submission

1. **Record the five-minute demo video.** The only hard gap. `reports/demo_video_script.md`
   is ready to read from directly — each beat names the exact file to have on screen, and the
   recording setup lists the tabs to open in order. Two standing rules are written into it:
   read the copilot's execution mode off the screen rather than from the script, and never
   describe a failure that is not visible.
2. **The two judgment calls I did not touch, as instructed.** Both read clearly and need no
   editing — they need your agreement:
   - *90+ DPD proxy* (`MODEL_CARD.md` §2). Realised credit events hit 14 of 16,000 loans, so
     `next_12m_default_flag` is a serious-delinquency proxy, not a loss model. Clearly written
     and disclosed in five places.
   - *Rejected rare-event oversample* (`AI_DEVELOPMENT_LOG.md` §12). Keeping all 7,878
     ever-90+DPD loans would have lifted their share from 3.2% to ~40% and destroyed the base
     rates calibration depends on. One figure a judge may query: the "~40%" is for the
     counterfactual design as scoped; against the final 16,000-loan sample the share would be
     closer to 49%. The argument is unaffected either way.
3. **Merging `real-data-switch`.** Not merged, as instructed. `master` @ `91fe18d` untouched.

## Open risks worth flagging

| Risk | Assessment |
|---|---|
| **Prepayment model is weak out of time** (ROC 0.626, ECE 0.135) | Real and disclosed in model card §8. The macro window spans a full rate cycle, so the regime genuinely shifts. It is reported, not patched over |
| **Copilot run-to-run variance** | Gemini is sampled, so a given run may block little or nothing. The report says which run it is showing and points at the archive for earlier captured failures. A reviewer re-running may see different blocks — this is stated rather than hidden |
| **Free-tier quota** | `gemini-3.5-flash-lite` clears a full run comfortably. `gemini-3.6-flash` does **not** — 20 requests/day against 15–20 per run. If you switch models before recording, check the quota first |
| **The LaTeX ablation is negative** | The no-LaTeX prompt rule could not be shown to be what fixed the markup. Reported as negative. Detection is the load-bearing control and does not depend on the model cooperating |
| **Lost prompt-log history** | ~87 calls from six early runs are unrecoverable. No current claim depends on them; the 10× fabrication recurred and is quoted from a live log, and the LaTeX case is described rather than reconstructed |
| **Exception/DQ layer is fabricated** | Long-standing and disclosed per-layer in model card §2. SFLLD provides no second source, no ingestion timestamps and no operational exception feed |

---

# FINAL — `master` is the submission state

**Branch structure is settled.** `real-data-switch` was merged into `master` with a merge
commit (`--no-ff`), preserving all 17 commits rather than squashing them. Both branches are on
GitHub. **`master` is the default branch and the state to submit.**

| | |
|---|---|
| Repository | `github.com/tanaysinghh/loan-performance-intelligence-engine` |
| Submission branch | **`master`** — default branch |
| Feature branch | `real-data-switch` — retained, identical tree, kept for history |
| Pre-switch fallback | `91fe18d`, now backed up on GitHub, reachable as the merge's first parent |
| Tests | 40/40 · validator self-test 12/12 |

**The merge changed nothing.** `master^{tree}` and `real-data-switch^{tree}` are the same git
object (`8c6c8ce`) and `git diff master real-data-switch` is empty — so no conflict was
silently resolved, and nothing was reverted or left stale. Every deliverable on `master` is
byte-identical to what was verified on the feature branch.

## One real gap was found while finalising, and fixed

`data/raw/*` was gitignored wholesale to keep the licence-gated Freddie Mac panels out of the
repository. That was right for the panels and wrong for the rest of the directory: **three
artefacts named in section 6 of the problem statement were being excluded with them**, so
anyone cloning the repo got no `validation_rules.json`, no `data_dictionary.csv` and no
`macro_scenarios.csv`.

Earlier audits missed it because they tested whether files existed *on disk* rather than
whether they were *tracked*. Fixed by whitelisting those three plus `macro_history.csv` and
`ground_truth_defect_log.csv` — 32 KB total, no loan-level data in any of them. The two
licensed panels (`loan_panel.csv`, 149 MB; `servicer_updates.csv`, 24 MB) stay excluded, with
the reason now written into `.gitignore` instead of left implicit, and representative samples
of both remain tracked under `data/samples/`.

## Final pre-submission audit — P1 and P2 closed

A first-principles audit against the problem statement PDF (`reports/final_precommit_audit.md`)
found two items that every earlier audit in this repository had missed, because they checked
generated artefacts and never re-read `README.md`. Both are now fixed.

**P1 — `README.md` was stale and understated the submission. RESOLVED.** It still described the
pre-Gemini state, and its *Known gaps* section told a judge that Task 7 had no live LLM
transcripts — false, and the single deliverable worth 10 rubric points. It also handed over
`ANTHROPIC_API_KEY`, so anyone following the README to reproduce the copilot would have landed
in `offline_template` mode and seen the stale claim apparently confirmed. Eleven corrections in
all: the five originally flagged, plus a stale test count in two places, a "six-case" self-test
that is now twelve, an incomplete `submission/` layout listing, and — found by re-checking every
figure rather than only the prose — **two wrong numbers**. The split-inflation range was quoted
as "+0.10 to +0.32" against an actual −0.01 to +0.36, and LightGBM's calibration advantage as
"2-4x on Brier" against an actual 2x to 12x. One overstated, one understated; both now read off
the committed artefacts.

**P2 — no external test file. RESOLVED as a disclosure, not a code change.** Problem statement
section 6 anticipates an organiser-supplied unlabeled `loan_monthly_performance_test.csv`. None
was issued, so this project's own pipeline fills that gap. Both `README.md` and
`MODEL_CARD.md` now state plainly that `submission.csv` contains held-out predictions on the
project's own time-aware split — the purged out-of-time window in `src/models/splits.py`,
reported per target in `reports/split_summary.csv` — and that no code path claims to score an
external file. No defensive loader was built for a file that does not exist.

P3 (self-chosen submission column names, mapped in `SUBMISSION_FORMAT.md`) and P4
(`data_dictionary` ships as `.csv` where the PS table names `.md`) were reviewed and
deliberately skipped as cosmetic.

## What is left for you before submitting

1. **Record the five-minute demo video.** The only outstanding deliverable, and the only thing
   that cannot be produced without you. `reports/demo_video_script.md` is ready to read from
   directly: 15 beats mapped to the PS §14 flow, every figure verified against current
   artefacts, and the exact file to have on screen named for each beat. Two rules are written
   into the script — read the copilot's execution mode off the screen rather than from the
   page, and never describe a failure that is not visible.
2. **Attach the video** wherever the HackerEarth portal expects it, and paste the repository
   link.
3. **Optional, only if the portal asks for a runnable notebook.** `notebooks/` is empty. It is
   not a named §11 deliverable and the reports carry the narrative a notebook would, so this
   is a gap only if the portal specifically requires one.

Nothing else is outstanding. Every other deliverable is committed, pushed and verified against
the git index rather than the working directory.

## Open risks, unchanged

| Risk | Assessment |
|---|---|
| **Prepayment weak out of time** (ROC 0.626, ECE 0.135) | Real, disclosed in model card §8. The macro window spans a full rate cycle so the regime genuinely shifts. Reported, not patched over |
| **Copilot run-to-run variance** | Gemini is sampled; a re-run may block different outputs, or none. The report names which run it shows and points at the archive. Stated rather than hidden |
| **Free-tier quota** | `gemini-3.5-flash-lite` clears a full run. `gemini-3.6-flash` does **not** — 20 requests/day against 15–20 per run. Do not switch models before recording |
| **LaTeX ablation is negative** | The no-LaTeX prompt rule could not be shown to be what fixed the markup. Reported as negative; detection is the load-bearing control |
| **~87 calls of prompt-log history unrecoverable** | Destroyed by the log-deletion bug before it was found. No current claim depends on them: the 10× fabrication recurred and is quoted from a live log; the LaTeX case is described, not reconstructed |
| **Exception / DQ layer is fabricated** | Long-standing, disclosed per-layer in model card §2. SFLLD provides no second source, no ingestion timestamps and no operational exception feed |
