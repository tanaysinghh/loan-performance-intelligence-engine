# Model Card — Loan Performance Intelligence Engine

**Submission:** Intain Campus FinTech Challenge 2026, AI Track, Round 2  
**Author:** Tanay Singh  
**Card generated:** 2026-08-30 16:55 UTC  
**Data source:** Freddie Mac SFLLD (real)

> Every figure in this card is generated from the pipeline's own report artefacts by `src/report_model_card.py`. Retraining regenerates the card; the numbers cannot drift away from the models.

---

## 1. Objective

For every loan-month record in a servicing panel, produce decision-support output a servicing-oversight team can act on:

| output | model | horizon |
| --- | --- | --- |
| Delinquency probability | LightGBM binary | 3 and 6 months |
| Default probability | LightGBM binary | 12 months |
| Prepayment probability | LightGBM binary | 12 months |
| Next performance state | LightGBM multiclass | 1 month |
| Exception probability | LightGBM binary | current record |
| Exception type | LightGBM multiclass, 6 classes | current record |
| Record anomaly score | Isolation forest (unsupervised) | current record |
| Time-to-event curves | Kaplan-Meier, Aalen-Johansen, Cox PH | loan lifetime |
| Multi-period state distribution | Empirical Markov chain | 1-12 months |
| Recommended action | Deterministic rule over the above | - |

**No language model produces any of these numbers.** An LLM is used only in `src/copilot/` to narrate model output, and that constraint is enforced by automated tests (section 9), not by convention.

---

## 2. Data

### Source: real Freddie Mac loan-level data

The panel is built from the **Freddie Mac Single-Family Loan-Level Dataset (SFLLD)** sample files for vintages 2019-2023 — a population of 250,000 loans and 10,482,492 monthly performance records — loaded by `src/data/build_from_sflld.py`. The organiser data pack described in section 6 of the problem statement was never issued, so the data was sourced directly rather than simulated.

The raw files are **not committed**: SFLLD is licence-gated and redistributing it would breach the source terms that section 13 of the problem statement lists as a disqualification condition. `dataset/download_sflld.md` documents how to re-obtain them; everything under `data/` regenerates from them.

> **Layout note.** These sample files carry **31 origination and 35 performance columns**, not the 32/32 in Freddie Mac's published `file_layout.xlsx` and January 2026 User Guide. `Servicer Name` is absent from the origination file, and the performance file appends `MI Cancellation Indicator`, `Servicer Name` and a filler column. The mapping was verified empirically against value distributions in all five vintages rather than assumed, and `sflld.verify_layout()` fails loudly if a re-download deviates. Evidence is documented in `src/data/sflld.py`.

### This pack is a hybrid, by necessity

SFLLD supplies no second data source, no ingestion timestamps, no document-custody data and no exception taxonomy. Those are required by sections 6 and 7 of the problem statement, so they are **fabricated on top of the real panel**. This is a documented methodological choice, not an oversight:

| layer | provenance |
| --- | --- |
| Loan / month panel, all origination and performance attributes | **Real** — Freddie Mac SFLLD |
| Delinquency, prepayment, credit-event and servicing-transfer outcomes | **Real** — derived from SFLLD status and zero-balance codes |
| Macro history (mortgage rate, unemployment, HPI) | **Real** — FRED `MORTGAGE30US`, `UNRATE`, `CSUSHPINSA` |
| Forward scenario paths | Constructed assumptions at supervisory severity; disclosed in the scenario report |
| `last_updated_at`, `source_system`, `document_status` | **Fabricated** — no equivalent exists in SFLLD |
| `servicer_updates.csv` second source, reconciliation conflicts | **Fabricated**, but anchored on real servicing transfers (6,903 of the sampled loans genuinely change servicer) |
| `exception_required`, `exception_type`, injected data-quality defects | **Fabricated** at logged rates, for Task 1 and Task 4 |

Every model figure for delinquency, default, prepayment and next-state is therefore trained and evaluated on real outcomes. Every figure for exceptions and data quality is trained on a fabricated label and must be read as a demonstration of method, not as validated real-world performance.

### Default target is a 90+ DPD proxy — read this before any default figure

Realised credit events — third-party sale, short sale, REO disposition, note sale (zero-balance codes 02/03/09/15) — occur on **14 of 16,000 sampled loans**, about one in a thousand, and roughly one row in 200,000. That is not a modellable target at this sample size. These are post-2019 agency vintages that benefited from strong house-price appreciation and pandemic-era forbearance, so the scarcity is a property of the cohort, not of the sample.

**`next_12m_default_flag` is therefore defined as: the loan reaches 90+ days past due, or a realised credit event, within the next 12 months.** Every 'default' figure in this card, in `reports/`, and in `submission.csv` refers to that proxy. It is a serious-delinquency model, not a loss model, and it must not be read as a probability of foreclosure or of loss. The realised-event count above is reported rather than hidden precisely so the gap between the two is visible.

| property | value |
| --- | --- |
| Records (after de-duplication) | 670,548 |
| Loans | 16,000 |
| Reporting window | 2019-01 to 2026-03 (87 months) |
| Servicers / states | 42 / 53 |
| Secondary servicer feed | 244,763 records with balance and status conflicts, duplicates and orphan rows |
| Engineered features | 81 |

**Pool characterisation.** Observed rates over the panel are 1.8% for the 12-month 90+ DPD proxy and 11.3% for 12-month prepayment. This is an **agency prime pool** — Freddie Mac acquisition criteria, mean origination FICO in the 740s across all five vintages — so credit performance is strong and prepayment dominates the outcome mix. Figures should not be extrapolated to non-QM, alt-A or seasoned distressed collateral.

**Macro path.** Real, and it spans a full rate cycle: mean origination rate falls from 4.24% (2019) to 2.97% (2021) and then rises to 6.74% (2023), a 377bp trough-to-peak move. Prepayment tracks it — 71% of the 2019 vintage prepaid versus 19% of the 2021 vintage. That gives genuine, non-simulated regime shift for the drift analysis in Task 1 and the scenarios in Task 5, and it is also the reason the prepayment model degrades out of time (section 8).

**Injected defects.** Missingness (missing-at-random conditional on servicer), sentinel values, invalid date relationships, balance outliers, inconsistent loan ages, duplicate rows and conflicting servicer records — all at logged rates in `data/raw/ground_truth_defect_log.csv`. That log validates detection and is never a model input.

---

**No external test file was issued.** Section 6 of the problem statement anticipates an organiser-supplied unlabeled `loan_monthly_performance_test.csv` for final scoring. None was released, so this project builds its own data pipeline to fill that gap, and `submission/submission.csv` contains held-out predictions on the project's own **time-aware split** — the purged out-of-time test window defined in `src/models/splits.py` and reported per target in `reports/split_summary.csv` — rather than scores against an external file. No code path claims to score one.

---

## 3. Features

81 features from `src/features/build_features.py`, in seven families: static credit attributes; current position; behavioural history (lags, rolling maxima, delinquency counts, clean streaks); macro and refinance incentive; data-quality and repair indicators; servicer-feed reconciliation; and residuals (balance against expected amortisation, days past due against reported status).

**Deliberately excluded, with reasons recorded in code:**

| excluded | reason |
| --- | --- |
| prepayment_flag, default_flag, next_state | Describe month t+1. They are targets, not inputs. |
| loss_severity_band | Populated only after default, so its presence alone is the label. |
| vintage_year | A calendar-time proxy whose levels are unseen in the test window. An ablation showed removing it gained 0.004-0.008 test ROC-AUC on every target. |

---

## 4. Validation method

Time-aware, horizon-purged, and capped at label observability. For horizon H with `U = last_month - H`:

```
[ train .......... | valid (6m) ] [ embargo (H months, dropped) ] [ test (6m) ]
                                                                   ends at U
```

Three distinct problems this solves:

1. **Unobservable labels.** A row within H months of the panel end can only carry a positive H-month label if the event already happened. Keeping such rows turns the test set into a sample of terminated loans. They are excluded, not imputed to zero.
2. **Window overlap.** A training row at month *t* encodes months *t+1..t+H*. An embargo of H months sits between the fitting data and the test window.
3. **Censoring.** Rows whose horizon runs past the panel end are `NaN`, not `0`, and are dropped from supervised training.

Train and validation are contiguous with no internal embargo: validation drives early stopping and calibration only, and both are then assessed out-of-time on the purged test window, which is what is reported.

| target | horizon_months | train_window | valid_window | embargo_window | test_window |
| --- | --- | --- | --- | --- | --- |
| next_3m_delinquency_flag | 3 | 2019-01..2024-09 | 2024-10..2025-03 | 2025-04..2025-06 | 2025-07..2025-12 |
| next_6m_delinquency_flag | 6 | 2019-01..2024-03 | 2024-04..2024-09 | 2024-10..2025-03 | 2025-04..2025-09 |
| next_12m_default_flag | 12 | 2019-01..2023-03 | 2023-04..2023-09 | 2023-10..2024-09 | 2024-10..2025-03 |
| next_12m_prepayment_flag | 12 | 2019-01..2023-03 | 2023-04..2023-09 | 2023-10..2024-09 | 2024-10..2025-03 |
| exception_required | 0 | 2019-01..2025-03 | 2025-04..2025-09 | none | 2025-10..2026-03 |

| target | train_rows | valid_rows | test_rows | rows_dropped_embargo | rows_dropped_unobservable_label | train_positive_rate | test_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| next_3m_delinquency_flag | 471347 | 68984 | 65149 | 33599 | 1348 | 0.0230 | 0.0313 |
| next_6m_delinquency_flag | 399768 | 71579 | 66539 | 68984 | 3946 | 0.0326 | 0.0405 |
| next_12m_default_flag | 263320 | 64660 | 68984 | 143367 | 8889 | 0.0187 | 0.0154 |
| next_12m_prepayment_flag | 263320 | 64660 | 68984 | 143367 | 8333 | 0.1456 | 0.0758 |
| exception_required | 540331 | 66539 | 63678 | 0 | 0 | 0.1398 | 0.1430 |

Hyperparameters are selected per target on **validation** average precision from a five-point grid. The calibrator (Platt vs isotonic) is chosen by 3-fold cross-validation *inside* the validation window — selecting on the full validation window systematically favours isotonic, which bends to noise that does not repeat out of time.

---

## 5. Metrics (purged out-of-time test window)

| target | model | roc_auc | pr_auc | pr_auc_lift_over_base | best_f1 | recall_at_precision_30 | brier | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| next_3m_delinquency_flag | baseline_logistic | 0.8830 | 0.5810 | 18.5460 | 0.6520 | 0.6430 | 0.1110 | 0.2720 |
| next_3m_delinquency_flag | lgbm_calibrated | 0.9160 | 0.6500 | 20.7290 | 0.6720 | 0.7660 | 0.0150 | 0.0020 |
| next_6m_delinquency_flag | baseline_logistic | 0.8380 | 0.5040 | 12.4530 | 0.5580 | 0.5370 | 0.1340 | 0.2960 |
| next_6m_delinquency_flag | lgbm_calibrated | 0.8780 | 0.5780 | 14.2870 | 0.6030 | 0.6490 | 0.0230 | 0.0020 |
| next_12m_default_flag | baseline_logistic | 0.9190 | 0.5740 | 37.3020 | 0.6150 | 0.6600 | 0.1090 | 0.2230 |
| next_12m_default_flag | lgbm_calibrated | 0.9210 | 0.5320 | 34.5940 | 0.5930 | 0.6750 | 0.0090 | 0.0040 |
| next_12m_prepayment_flag | baseline_logistic | 0.6850 | 0.2410 | 3.1810 | 0.2830 | 0.2360 | 0.2790 | 0.4170 |
| next_12m_prepayment_flag | lgbm_calibrated | 0.6260 | 0.2010 | 2.6490 | 0.2160 | 0.0790 | 0.1370 | 0.1350 |
| exception_required | baseline_logistic | 0.5400 | 0.2300 | 1.6080 | 0.2510 |  | 0.2470 | 0.3590 |
| exception_required | lgbm_calibrated | 0.9690 | 0.8290 | 5.7980 | 0.8540 |  | 0.0350 | 0.0020 |

**Read the baseline comparison honestly.** LightGBM does *not* dominate the nine-feature logistic baseline on ranking. The largest ROC-AUC gap on the four performance targets is 0.059, and the baseline wins outright on some of them. The dominant delinquency signals are near-monotone in the log-odds, which is exactly where a linear model is hard to beat on ranking.

Where they separate decisively is **calibration**: the baseline's Brier score is 6.8x worse on average and its expected calibration error runs 0.22-0.42, because `class_weight=balanced` inflates every probability. It can rank a queue; it cannot answer "what is the probability", which is what the submission format asks for.

The exception model is the one case where the gap is total — 0.969 against 0.540. That gap is itself the finding: the baseline is deliberately the same nine *credit* fields, and operational exceptions are not a credit phenomenon.

**Multiclass and time-to-event:**

| model | metric | value | baseline |
| --- | --- | --- | --- |
| Next state (1 month) | macro-F1 | 0.5860 | 0.370 (Markov), 0.473 (persistence) |
| Next state (1 month) | macro-ROC-AUC | 0.9820 | 0.832 (Markov) |
| Exception type (6-class) | macro-F1 | 0.9170 | 0.104 (majority class) |
| Exception type (6-class) | macro-ROC-AUC | 0.9970 | - |
| Markov 12-month projection | MAE vs realised default rate | 0.1264 | - |

Cox proportional-hazards discrimination and the full survival results are in `reports/survival_report.md`. Kaplan-Meier assigns every loan the same curve, so its concordance is 0.50 by construction — that is the baseline Cox is beating.

Persistence ("next state = current state") edges the covariate model on raw accuracy (0.977 against 0.986) and ties it on macro-F1. Reported rather than buried: when 95%+ of transitions are Current-to-Current, a rule that never predicts a transition is hard to beat on accuracy and useless in practice, because it assigns zero probability to every event a servicer cares about. Macro-AUC and log loss are where the difference lives.

---

## 6. Class imbalance and calibration

Positive rates run 5-17%. Two mechanisms, kept deliberately separate:

- **Ranking** — `scale_pos_weight = sqrt(neg/pos)`. The square root rather than the full ratio: full reweighting maximises separation but destroys calibration.
- **Calibration** — Platt or isotonic fitted on validation, chosen by cross-validation inside that window. Platt wins on most targets; being strictly monotone it leaves ROC-AUC and PR-AUC exactly unchanged while cutting expected calibration error.

---

## 7. Leakage controls

| control | catches |
| --- | --- |
| Banned-feature list enforced in `assert_no_leakage`, called inside the design-matrix builder and again in tests | Target columns and post-outcome fields reaching the model |
| Horizon embargo between fitting data and test window | A training row's outcome window overlapping the evaluation period |
| Label-observability cap | Test sets that are secretly samples of terminated loans |
| Censored rows excluded rather than zero-filled | Manufactured negatives at the end of the panel |
| Split-sensitivity probe | Whether the honest split is doing any work at all |

**The split-sensitivity probe is the headline evidence.** Refitting the same model and features under an unsound random row split inflates test ROC-AUC by the amounts below — which is precisely how much a naive split would have flattered this submission.

| target | purged_time_split | loan_disjoint_time_split | random_row_split_unsound | random_split_inflation |
| --- | --- | --- | --- | --- |
| next_3m_delinquency_flag | 0.9160 | 0.9190 | 0.9060 | -0.0100 |
| next_6m_delinquency_flag | 0.8780 | 0.8700 | 0.8920 | 0.0140 |
| next_12m_default_flag | 0.9210 | 0.9350 | 0.9990 | 0.0780 |
| next_12m_prepayment_flag | 0.6260 | 0.6040 | 0.9830 | 0.3570 |

The loan-disjoint column additionally forces no `loan_id` into both the fitting data and the test window. Performance holding there means the model learned loan *characteristics*, not loan *identities*.

---

## 8. Failure modes and limitations

**Model limitations**

- **Prepayment is the weakest model** (test ROC-AUC 0.626 for the calibrated GBM against 0.685 for the logistic baseline — the baseline ranks better here, and that is reported rather than hidden; PR-AUC 0.201 against 0.241). It depends on refinance incentive, which depends on a rate path the panel contains exactly one realisation of. The GBM is still the shipped model because its calibration is usable and the baseline's is not (ECE 0.135 against 0.417), but on ranking alone the simpler model is the better choice, and anything scenario-shaped should use the macro-conditioned transition engine instead of either.
- **Regime change on the 12-month targets.** Train and test sit in different macro regimes; the default rate moves from 1.9% to 1.5% between them. Reported, not corrected — correcting it by reweighting would hide the most useful fact about the model's operating conditions.
- **Data volume on long horizons.** The 12-month targets lose 143,367 rows to the embargo and 8,889 to the observability cap: 263,320 training rows against 471,347 for the 3-month target. Confidence intervals are correspondingly wider.
- **The scenario engine's credit channel is not identified.** Macro levels are constant across loans within a month, so with one realised macro path a loan-level model cannot separate unemployment from calendar time. The symptom is diagnostic: a 2.3pp unemployment shock moves projected 12-month default by -0.13% in relative terms, and the high-prepayment scenario *raises* projected default. **Use Engine B (macro-conditioned Markov) to size credit stress; use Engine A only for which-loans segment detail.** Engine B moves cumulative 12-month default from 0.012 to 0.019 under adverse conditions.
- **The Markov first-order assumption is wrong**, usefully. A loan five months into DQ30 differs from one that entered last month. The covariate model beats the chain on macro-AUC (0.982 against 0.832); the chain is kept for transparency and multi-period projection.
- **Proportional hazards is assumed, not tested.** No Schoenfeld residual test is run.
- **No loss-given-default model.** Nothing here converts default probability into an expected dollar loss.
- **Single-seed point estimates.** No repeated-run variance is reported.

**Operational failure modes**

- **Servicer is a confound.** The two servicers with the worst reporting hygiene also have elevated delinquency, and SHAP cannot separate credit risk from reporting behaviour. A servicer-driven score is a prompt to investigate the servicer, not a statement about the borrower.
- **False negatives concentrate in specific segments**, quantified per segment in `reports/explainability_report.md`. That is a coverage issue, not just an accuracy one.
- **Synthetic-label optimism.** The exception label comes from rule breaches plus a materiality threshold and ~1.2% reviewer noise. Real reviewers are less consistent, so the 0.969 ROC-AUC is an upper bound.
- **Confidence bands are a boosting-stability proxy**, not statistical confidence intervals, and do not capture regime-change risk — the dominant risk on the 12-month targets.

---

## 9. LLM governance

The LLM never predicts. This is enforced, not promised:

1. `tests/test_no_llm_prediction.py::test_no_modelling_module_can_reach_a_language_model` parses the AST of every module under `src/data`, `src/features`, `src/models`, `src/scenarios` and `src/explain` and fails if any imports `anthropic`, `openai`, `google`, `cohere`, `mistralai`, `ollama` or `src.copilot`. The modelling code path *cannot* reach a language model. The guard is written against the capability, not against one vendor, so switching provider does not weaken it — adding a provider costs one line.
2. The **grounding validator** extracts every number from generated text and matches it against the grounding pack, including values scaled by 100 or rounded — the forms a helpful model reaches for. Unmatched numbers block the output. Its 12-case self-test confirms it catches fabricated probabilities, rescaled figures, causal assertions, overconfident decisions, missing reviewer framing and LaTeX markup, and that it does *not* fire on correct output (scientific notation, hyphenated field names, ordered-list markers, a legitimate refusal). Six of those cases were added after live Gemini runs exposed defects in the validator itself, and the suite runs against a fixed pack so its verdicts do not move with the data.
3. A **usefulness check** on per-record reviewer notes. The grounding validator is a truthfulness control and says nothing about output that is true and useless; a live run produced a note telling a reviewer to verify a document status the same pack reported as `complete`. That is now blocked and sent back for correction.

Every prompt, provider, model id, SDK, timestamp, response, token count, finish reason, latency and validator verdict is written to `submission/llm_prompt_log.jsonl`, with prior runs rotated into `submission/llm_prompt_log_archive.jsonl` so captured failures are not overwritten by the next run. All LLM output carries *"RECOMMENDATION, NOT DECISION."*

**Provider.** The copilot calls **Google Gemini** (`gemini-3.5-flash-lite`) through the `google-generativeai` SDK. This was a deliberate choice on cost and availability, not a fallback after a failure: the model is free-tier eligible, so the copilot reproduces end to end for anyone holding a free Google AI Studio key rather than only for someone with a paid credential. The model was picked by measurement — `gemini-3.6-flash` writes better prose but its free allowance is 20 requests per *day*, and one Task 7 run issues 15-20 calls, so it cannot be re-run.

The copilot design is vendor-neutral. Grounding packs, the system prompt, the validators and the adversarial probes are unchanged from the earlier Anthropic wiring; only the client, auth and response-parsing layer differs.

**Status: complete, with real failures captured.** The copilot ran live. Genuine Gemini errors were caught by the validators and corrected on a logged round-trip — most notably a 10x transcription error (`exception_required` reported as `0.046` where the pack says `0.0046`), a reviewer note that directed the reviewer at a field the pack already reported clean, and a portfolio summary rendered in LaTeX for a plain-text queue. `reports/copilot_report.md` section 5 separates genuine model failures from validator false positives rather than reporting all blocks as model error, and records one ablation that came out **negative** — the plain-text prompt rule could not be shown to be what suppressed the LaTeX.

---

## 10. Intended use and out-of-scope use

**Intended:** decision *support* for a servicing-oversight team — prioritising a review queue, sizing a stress scenario, and surfacing records whose data does not hold together.

**Out of scope:** automated adverse action, credit pricing, underwriting, or any use where output reaches a borrower without human review. The models are fitted on synthetic data and have no validated real-world performance. Fair-lending testing has not been performed; `state` and `servicer_name` are model inputs and would require disparate-impact analysis before production use.

---

## 11. Reproducibility

Fixed seed `20260828` throughout. `python -m src.pipeline` runs data generation through submission and writes `submission/run_manifest.json` recording every stage, its status, duration, and the artefacts produced.
