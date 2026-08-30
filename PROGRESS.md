# PROGRESS — Loan Performance Intelligence Engine

**Branch:** `real-data-switch` (9 commits) · **Fallback:** `master` @ `91fe18d` · **Tests:** 40/40

---

## Bottom line

The switch to real Freddie Mac data is **done and the gate passed**. Every deliverable is
regenerated from one coherent run on real data. Two things need you: an `ANTHROPIC_API_KEY`,
and recording the demo video. Everything else is closed.

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
| 1 | LLM copilot made real | **Blocked on API key — not faked.** Mode stated in the report's first line; all 11 log entries labelled `offline_template`. Validator self-test 6/6 is genuine non-LLM evidence and runs regardless. Rule-suggestion task added (4 of 6 Task 7 use cases). |
| 2 | submission.csv format | ✅ 16,000 × 21. All seven PS §6 elements mapped in `SUBMISSION_FORMAT.md`, pinned by 9 new tests. |
| 3 | Model card freshness | ✅ Regenerated via `write()` and post-dates every artefact it reads. Date and data source generated, not typed. |
| 4 | Demo video script | ✅ 15 beats, real figures, backup Q&A. |
| 5 | AI Development Log | ✅ §12 this session's work; §13 verbatim prompts. |
| 6 | Compliance re-audit | ✅ Re-audited. Tasks 1, 6, 8 moved PARTIAL → FULLY MET. §3.8 rewritten (was vacuous). Only Task 7 remains partial, on the API key. |

## Compliance status

- **Tasks 1–6, 8: FULLY MET.** Task 7 partial, solely on the missing credential.
- **All 10 disqualification conditions: do not apply.** §3.8 now met on its merits with
  gitignore/history/redistribution evidence, not by "all data is synthetic".
- New deliverable `data/raw/validation_rules.json` (PS §6 named it; none was issued).

---

## Requires your manual action

1. **`ANTHROPIC_API_KEY`** — the one open gap, worth up to 10 rubric points. No credential
   exists here, so the copilot runs `offline_template`. I did not write plausible transcripts
   and label them as API output. To close it:
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...
   python -m src.copilot.run_copilot
   python -c "from src.report_model_card import write; write()"
   ```
   That regenerates the report against live output with real timestamps, token counts and
   validator verdicts, then refreshes the card. Then capture 2–3 cases where the model was
   wrong, vague or overconfident — the five adversarial probes in `run_copilot.py` are built
   to elicit exactly those and currently have nothing to catch.
2. **Record the demo video** — `reports/demo_video_script.md`, 15 beats with figures filled
   and the artefacts to have on screen listed in order.
3. **Final review before submitting** — suggest reading `submission/MODEL_CARD.md` §2 first;
   it carries the hybrid-provenance table and the D90+ redefinition, which are the two things
   a judge is most likely to probe.
4. **Merging `real-data-switch`** — I have not merged or pushed. `master` @ `91fe18d` is
   untouched as a fallback.

*No organiser/HackerEarth communication has been needed. The real-data gate passed, so no
go/no-go decision is outstanding from you.*
