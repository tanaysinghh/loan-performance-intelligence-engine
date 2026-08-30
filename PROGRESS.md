# PROGRESS — Loan Performance Intelligence Engine

Running status file. Updated as work proceeds.

**Deadline:** <40h from 2026-08-30. **Branch:** `real-data-switch`. **Fallback:** `master` @ `91fe18d`.
**Last updated:** during the first clean real-data pipeline run.

---

## Phase 0 — Safety ✅ DONE (~6 min)

- `dataset/` gitignored (licence-gated SFLLD raw files); negation fixed so
  `dataset/download_sflld.md` stays tracked while the raw `.txt` files never are.
- Verified `dataset/` was **never committed**: 0 tracked files, largest blob ever in history
  2.7 MB. No history rewrite needed.
- Fallback commit `91fe18d` on `master`; working branch `real-data-switch`.

## Phase 1 — Real data switch ✅ GATE PASSED (~2h15m elapsed; gate was at 6h)

### Gate result

| Condition | Result |
|---|---|
| Clean `loan_panel.csv` the pipeline accepts | **PASS** — 33-column contract, 673,243 rows; `prepare()` builds a 670,548 × 133 feature frame, 16,000 loans, 2019-01..2026-03 |
| D90+ target class balance not ~1-in-2000 | **PASS** — **1.84% (1-in-54)**, 9,989 positive rows |

Both conditions met roughly 1h into the 6h budget, so the switch continued.

### Built

- **`src/data/sflld.py`** — layout module. Sample files carry **31 origination / 35
  performance** columns, *not* the published 32/32. Mapping verified empirically across all
  five vintages. `verify_layout()` refuses to load a deviating file rather than mis-mapping.
- **`src/data/build_from_sflld.py`** — join, banding, status/ZBC mapping, forward targets with
  censoring at the real end (2026-03), loan-level stratified sampling (3,200/vintage →
  **16,000 loans, 673,243 rows**). Same 33-column contract as the synthetic generator, so no
  downstream stage changed.
- **`src/data/macro_real.py`** — real FRED series vendored under `data/external/`:
  `MORTGAGE30US`, `UNRATE`, `CSUSHPINSA`. History observed; scenario paths constructed at
  supervisory severity (+3.0pp unemployment, −10% HPI YoY) with all three candidate
  calibrations disclosed (COVID +11.1pp, ex-COVID +0.7pp, chosen +3.0pp).
- **`dataset/download_sflld.md`** — how to re-obtain the licence-gated files; why 31/35.
- **`submission/SUBMISSION_FORMAT.md`** — maps every PS §6 required element to a column.
- **`profiling.drift_report_by_target()`** — drift at each target's *actual* purged boundary,
  closing the PARTIAL finding on Task 1 in the prior audit.
- README rewritten so real SFLLD is the documented primary path.

### Decisions worth knowing

- **Default redefined to a 90+ DPD proxy.** Realised credit events hit **14 of 16,000**
  sampled loans (0.09%). Disclosed in the model card (own subsection), the data intelligence
  report (call-out box), the README, and `SUBMISSION_FORMAT.md`.
- **Rejected the rare-event oversample.** Keeping all 7,878 ever-90+DPD loans from the
  250,000-loan population would lift their share from 3.2% to ~40% and wreck the base rates
  calibration and scenarios depend on. Used a plain loan-level stratified sample instead —
  true prevalence preserved, still 9,989 positive rows. Sampling is loan-level only, so no
  retained loan's history is truncated.
- **Adverse scenario is not empirically anchored, deliberately.** The window has no housing
  downturn, and ex-COVID no material labour deterioration. Supervisory magnitudes used, with
  the observed bounds printed next to them.

### Bug found and fixed — stale feature cache

The first full run completed exit 0 and rewrote **every report from the Aug-28 synthetic
frame**. `prepare()` caches the feature frame and `--skip-data` disabled invalidation. Nothing
crashed; the reports looked current. Caught on two impossible figures: `train_loans: 1379`
against 16,000, and a test window ending 2026-06 when real data ends 2026-03.

Fixed at the root — `dataset._cache_is_stale()` compares cache mtime against the raw pack and
forces a rebuild. `config.real_build_summary()` carries the same guard so a synthetic run
cannot inherit real-data prose. **Any model figure quoted before that fix is void.**

### In flight

- First clean end-to-end run on the real panel. Stages `data`→`profile` done; `models` running
  (~30 LightGBM fits over ~500k training rows — the long pole).
- Test suite running.

### Still to do in Phase 1

- Regenerate the `profile` stage (it ran before the provenance / D90+ / per-target-drift
  edits landed).
- Fill real figures into `reports/demo_video_script.md`.

## Phase 2 — Gap closure 🔄

| # | Item | Status |
|---|---|---|
| 1 | LLM copilot made real | **Blocked — no API key.** Not faked. See manual actions. Validator self-test is genuine non-LLM evidence and runs regardless. |
| 2 | submission.csv format | ✅ All seven PS §6 elements present; mapping documented in `submission/SUBMISSION_FORMAT.md`. Re-verify after final run. |
| 3 | Model card freshness | Generator made source-aware; hardcoded `2026-08-28` replaced with a generated timestamp. Regenerates at end of run. |
| 4 | Demo video script | ✅ Drafted, 15 beats mapped to PS §14. Figures pending final run. |
| 5 | AI Development Log currency | ✅ Session-2 section added (layout discovery, rejected COVID calibration, stale-cache defect, rejected oversample, un-faked copilot, per-session code share). |
| 6 | Full compliance re-audit | Pending final run. |

---

## Requires your manual action

1. **`ANTHROPIC_API_KEY` for the LLM copilot** — worth up to 10 rubric points.
   No credential exists in this environment (`ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` and
   `.env` all absent), so `src/copilot/` runs in `offline_template` mode. Per your
   instruction I did **not** write plausible transcripts and label them as captured API
   output. With a key exported:
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...
   python -m src.copilot.run_copilot
   ```
   That regenerates `reports/copilot_report.md` against live output and rewrites
   `submission/llm_prompt_log.jsonl` with real timestamps, token counts and validator
   verdicts. The report already states its mode in its first line either way.
2. **Recording the demo video** — script and storyboard ready at
   `reports/demo_video_script.md`; you record.
3. *(Not yet needed)* Go/no-go on the real-data path — **the gate passed**, so no decision is
   required from you here unless the final run surfaces something.
