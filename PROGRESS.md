# PROGRESS — Loan Performance Intelligence Engine

Running status file. Updated as work proceeds.

**Deadline:** <40h from 2026-08-30. **Branch:** `real-data-switch`. **Fallback:** `master` @ `91fe18d`.

---

## Phase 0 — Safety ✅ DONE (~6 min)

- `dataset/` added to `.gitignore` (license-gated SFLLD raw files). Negation fixed so
  `dataset/download_sflld.md` stays tracked while the raw `.txt` files never are.
- Verified `dataset/` was **never committed**: 0 tracked files, largest blob ever in history
  2.7 MB. No history rewrite needed.
- Fallback commit `91fe18d` on `master`; working branch `real-data-switch`.

## Phase 1 — Real data switch 🔄 IN PROGRESS (~1h40m elapsed, gate at 6h)

### Done

- **`src/data/sflld.py`** — layout module. The sample files carry **31 origination / 35
  performance** columns, *not* the 32/32 in Freddie Mac's published `file_layout.xlsx` and
  January 2026 User Guide. Mapping verified empirically across all five vintages, not
  assumed. `Servicer Name` is absent from origination (positions 25–31 → official 26–32);
  performance appends 33 = `MI Cancellation Indicator`, 34 = `Servicer Name`, 35 = filler.
  `verify_layout()` fails loudly if a re-download deviates.
- **`src/data/build_from_sflld.py`** — origination↔performance join, banding, status and
  zero-balance-code mapping, forward targets with censoring at the real panel end
  (**2026-03**), loan-level stratified sampling (3,200/vintage → **16,000 loans, 673,243
  rows**, inside the PS's 250k–1M range). Writes the **same 33-column contract** as the
  synthetic generator, so no downstream stage changed.
- **Default target redefined** to a documented 90+ DPD proxy. Realised credit events occur on
  **14 of 16,000 sampled loans (0.09%)** — not modellable. Stated in the module docstring;
  still to be surfaced in the model card and data intelligence report.
- **`src/data/macro_real.py`** — real public series vendored under `data/external/`: FRED
  `MORTGAGE30US` (Freddie Mac PMMS 30y), `UNRATE` (BLS), `CSUSHPINSA` (Case-Shiller).
  History is observed; the three scenario paths are constructed and labelled as such.
  Adverse shocks use supervisory (DFAST-style) severity — **+3.0pp unemployment, −10% HPI
  YoY** — because the panel window contains no housing downturn and, excluding COVID, the
  largest 12-month unemployment rise in it is only +0.7pp. All three figures (COVID +11.1pp,
  ex-COVID +0.7pp, chosen +3.0pp) are disclosed in the scenario file.
- **`dataset/download_sflld.md`** — how to re-obtain the license-gated raw files, why 31/35,
  and how to refresh the macro series.
- **`reports/demo_video_script.md`** — 15-beat script mapped to PS section 14 (Phase 2 item,
  done early; figures still to be filled from the final reports).

### Bug found and fixed — stale feature cache

The first full pipeline run completed (exit 0) but **every report it produced was from the
old synthetic data**. `prepare()` reads a cached `model_frame`, and `--skip-data` disabled
cache invalidation, so the run silently reused the Aug-28 synthetic frame (1,379 train loans,
test window running to 2026-06 — impossible for real data ending 2026-03).

Fixed properly rather than by deleting the file: `dataset._cache_is_stale()` now compares the
cache mtime against `loan_panel.csv` / `servicer_updates.csv` / `macro_history.csv` and
rebuilds when the pack is newer. **Any model figures quoted before this point are void.**

### In flight

- Rebuilding the feature frame from the real panel, then a clean end-to-end re-run.

### Gate status (hard gate at hour 6)

| Condition | Status |
|---|---|
| Clean `loan_panel.csv` the pipeline accepts | Pipeline ran end-to-end exit 0, but on the stale cache. **Re-verifying on the real frame.** |
| D90+ target class balance not ~1-in-2000 | **1.84% at build time (1-in-54)**, 9,989 positive rows. Comfortably passes, pending confirmation post-features. |

## Phase 2 — Gap closure ⏳ PENDING

| # | Item | Status |
|---|---|---|
| 1 | LLM copilot made real | **Blocked — no API key in this environment.** Will not fabricate transcripts. See manual actions. |
| 2 | submission.csv format | Verified: all seven PS §6 elements present. No organizer template exists to match names against, so the mapping will be documented explicitly. |
| 3 | Model card freshness | Pending final data path. |
| 4 | Demo video script | ✅ Drafted (`reports/demo_video_script.md`); figures to fill. |
| 5 | AI Development Log currency | Pending. |
| 6 | Full compliance re-audit | Pending. |

---

## Requires your manual action

1. **`ANTHROPIC_API_KEY` for the LLM copilot.** No credential is present in this environment
   (`ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, and `.env` all absent), so the copilot runs
   in `offline_template` mode. Per your instruction I am **not** writing plausible transcripts
   and labelling them as captured API output. With a key exported, `python -m
   src.copilot.run_copilot` regenerates the report against live model output and rewrites
   `submission/llm_prompt_log.jsonl` with real timestamps, token counts and validator
   verdicts. This is worth up to 10 rubric points.
2. **Recording the demo video.** Script is ready; you record.
