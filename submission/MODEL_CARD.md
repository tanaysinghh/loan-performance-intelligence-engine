# Model Card — Loan Performance Intelligence Engine

**Submission:** Intain Campus FinTech Challenge 2026, AI Track, Round 2  
**Author:** Tanay Singh  
**Date:** 2026-08-28

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

Organiser data was not available, so a synthetic panel is generated from an explicit data-generating process (`src/data/generate_synthetic.py`). It is built to be replaced: drop real CSVs matching the same schema into `data/raw/` and the pipeline runs unchanged.

| property | value |
| --- | --- |
| Records (after de-duplication) | 48,924 |
| Loans | 1,500 |
| Reporting window | 2019-01 to 2026-06 (90 months) |
| Servicers / states | 5 / 15 |
| Secondary servicer feed | 17,651 records with balance and status conflicts, duplicates and orphan rows |
| Engineered features | 81 |

**Pool characterisation.** Observed rates over the panel are 6.2% for 12-month default and 15.5% for 12-month prepayment. That is a **seasoned non-QM / alt-A profile**, not an agency prime pool, and these figures should not be read as agency benchmarks.

**Macro path.** A full rate cycle with a pandemic-shaped unemployment spike. The panel was deliberately lengthened from 54 to 90 months during development, because a 12-month horizon with an embargo left the original window with only one rate regime in training — prepayment ROC-AUC came out at 0.51. See the AI Development Log.

**Injected defects.** Missingness (missing-at-random conditional on servicer), sentinel values, invalid date relationships, balance outliers, inconsistent loan ages, duplicate rows and conflicting servicer records — all at logged rates in `data/raw/ground_truth_defect_log.csv`. That log validates detection and is never a model input.

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
| next_3m_delinquency_flag | 3 | 2019-01..2024-12 | 2025-01..2025-06 | 2025-07..2025-09 | 2025-10..2026-03 |
| next_6m_delinquency_flag | 6 | 2019-01..2024-06 | 2024-07..2024-12 | 2025-01..2025-06 | 2025-07..2025-12 |
| next_12m_default_flag | 12 | 2019-01..2023-06 | 2023-07..2023-12 | 2024-01..2024-12 | 2025-01..2025-06 |
| next_12m_prepayment_flag | 12 | 2019-01..2023-06 | 2023-07..2023-12 | 2024-01..2024-12 | 2025-01..2025-06 |
| exception_required | 0 | 2019-01..2025-06 | 2025-07..2025-12 | none | 2026-01..2026-06 |

| target | train_rows | valid_rows | test_rows | rows_dropped_embargo | rows_dropped_unobservable_label | train_positive_rate | test_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| next_3m_delinquency_flag | 37491 | 4040 | 3719 | 1987 | 185 | 0.0581 | 0.0664 |
| next_6m_delinquency_flag | 33521 | 3970 | 3913 | 4040 | 484 | 0.0742 | 0.0917 |
| next_12m_default_flag | 26257 | 3511 | 4040 | 7723 | 1075 | 0.0512 | 0.0723 |
| next_12m_prepayment_flag | 26257 | 3511 | 4040 | 7723 | 1075 | 0.1521 | 0.1710 |
| exception_required | 41531 | 3913 | 3480 | 0 | 0 | 0.1251 | 0.1305 |

Hyperparameters are selected per target on **validation** average precision from a five-point grid. The calibrator (Platt vs isotonic) is chosen by 3-fold cross-validation *inside* the validation window — selecting on the full validation window systematically favours isotonic, which bends to noise that does not repeat out of time.

---

## 5. Metrics (purged out-of-time test window)

| target | model | roc_auc | pr_auc | pr_auc_lift_over_base | best_f1 | recall_at_precision_30 | brier | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| next_3m_delinquency_flag | baseline_logistic | 0.9060 | 0.7870 | 11.8470 | 0.8140 | 0.8180 | 0.0790 | 0.1930 |
| next_3m_delinquency_flag | lgbm_calibrated | 0.8940 | 0.7850 | 11.8200 | 0.8140 | 0.8020 | 0.0210 | 0.0140 |
| next_6m_delinquency_flag | baseline_logistic | 0.8880 | 0.7360 | 8.0200 | 0.7390 | 0.7990 | 0.1150 | 0.2300 |
| next_6m_delinquency_flag | lgbm_calibrated | 0.8820 | 0.7380 | 8.0430 | 0.7310 | 0.7630 | 0.0390 | 0.0240 |
| next_12m_default_flag | baseline_logistic | 0.9020 | 0.6360 | 8.7930 | 0.6860 | 0.7600 | 0.0840 | 0.1610 |
| next_12m_default_flag | lgbm_calibrated | 0.8970 | 0.5860 | 8.1130 | 0.6240 | 0.7770 | 0.0400 | 0.0230 |
| next_12m_prepayment_flag | baseline_logistic | 0.6510 | 0.3150 | 1.8390 | 0.3700 | 0.4020 | 0.2560 | 0.2740 |
| next_12m_prepayment_flag | lgbm_calibrated | 0.6700 | 0.2610 | 1.5270 | 0.3790 | 0.2170 | 0.1640 | 0.1170 |
| exception_required | baseline_logistic | 0.5330 | 0.1700 | 1.3050 | 0.2400 |  | 0.2440 | 0.3630 |
| exception_required | lgbm_calibrated | 0.9650 | 0.8330 | 6.3850 | 0.8600 |  | 0.0320 | 0.0070 |

**Read the baseline comparison honestly.** LightGBM does *not* dominate the nine-feature logistic baseline on ranking. The largest ROC-AUC gap on the four performance targets is 0.018, and the baseline wins outright on some of them. The dominant delinquency signals are near-monotone in the log-odds, which is exactly where a linear model is hard to beat on ranking.

Where they separate decisively is **calibration**: the baseline's Brier score is 2.6x worse on average and its expected calibration error runs 0.16-0.27, because `class_weight=balanced` inflates every probability. It can rank a queue; it cannot answer "what is the probability", which is what the submission format asks for.

The exception model is the one case where the gap is total — 0.965 against 0.533. That gap is itself the finding: the baseline is deliberately the same nine *credit* fields, and operational exceptions are not a credit phenomenon.

**Multiclass and time-to-event:**

| model | metric | value | baseline |
| --- | --- | --- | --- |
| Next state (1 month) | macro-F1 | 0.4220 | 0.375 (Markov), 0.439 (persistence) |
| Next state (1 month) | macro-ROC-AUC | 0.8900 | 0.841 (Markov) |
| Exception type (6-class) | macro-F1 | 0.8690 | 0.096 (majority class) |
| Exception type (6-class) | macro-ROC-AUC | 0.9970 | - |
| Markov 12-month projection | MAE vs realised default rate | 0.0504 | - |

Cox proportional-hazards discrimination and the full survival results are in `reports/survival_report.md`. Kaplan-Meier assigns every loan the same curve, so its concordance is 0.50 by construction — that is the baseline Cox is beating.

Persistence ("next state = current state") edges the covariate model on raw accuracy (0.959 against 0.956) and ties it on macro-F1. Reported rather than buried: when 95%+ of transitions are Current-to-Current, a rule that never predicts a transition is hard to beat on accuracy and useless in practice, because it assigns zero probability to every event a servicer cares about. Macro-AUC and log loss are where the difference lives.

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
| next_3m_delinquency_flag | 0.8940 | 0.8930 | 0.9910 | 0.0960 |
| next_6m_delinquency_flag | 0.8820 | 0.8660 | 0.9970 | 0.1140 |
| next_12m_default_flag | 0.8970 | 0.8910 | 0.9970 | 0.0990 |
| next_12m_prepayment_flag | 0.6700 | 0.5830 | 0.9970 | 0.3270 |

The loan-disjoint column additionally forces no `loan_id` into both the fitting data and the test window. Performance holding there means the model learned loan *characteristics*, not loan *identities*.

---

## 8. Failure modes and limitations

**Model limitations**

- **Prepayment is the weakest model** (test ROC-AUC 0.670). It depends on refinance incentive, which depends on a rate path the panel contains exactly one realisation of.
- **Regime change on the 12-month targets.** Train and test sit in different macro regimes; the default rate moves from 5.1% to 7.2% between them. Reported, not corrected — correcting it by reweighting would hide the most useful fact about the model's operating conditions.
- **Data volume on long horizons.** The 12-month targets lose 7,723 rows to the embargo and 1,075 to the observability cap: 26,257 training rows against 37,491 for the 3-month target. Confidence intervals are correspondingly wider.
- **The scenario engine's credit channel is not identified.** Macro levels are constant across loans within a month, so with one realised macro path a loan-level model cannot separate unemployment from calendar time. The symptom is diagnostic: a 2.3pp unemployment shock moves projected 12-month default by 0.51% in relative terms, and the high-prepayment scenario *raises* projected default. **Use Engine B (macro-conditioned Markov) to size credit stress; use Engine A only for which-loans segment detail.** Engine B moves cumulative 12-month default from 0.174 to 0.227 under adverse conditions.
- **The Markov first-order assumption is wrong**, usefully. A loan five months into DQ30 differs from one that entered last month. The covariate model beats the chain on macro-AUC (0.890 against 0.841); the chain is kept for transparency and multi-period projection.
- **Proportional hazards is assumed, not tested.** No Schoenfeld residual test is run.
- **No loss-given-default model.** Nothing here converts default probability into an expected dollar loss.
- **Single-seed point estimates.** No repeated-run variance is reported.

**Operational failure modes**

- **Servicer is a confound.** The two servicers with the worst reporting hygiene also have elevated delinquency, and SHAP cannot separate credit risk from reporting behaviour. A servicer-driven score is a prompt to investigate the servicer, not a statement about the borrower.
- **False negatives concentrate in specific segments**, quantified per segment in `reports/explainability_report.md`. That is a coverage issue, not just an accuracy one.
- **Synthetic-label optimism.** The exception label comes from rule breaches plus a materiality threshold and ~1.2% reviewer noise. Real reviewers are less consistent, so the 0.965 ROC-AUC is an upper bound.
- **Confidence bands are a boosting-stability proxy**, not statistical confidence intervals, and do not capture regime-change risk — the dominant risk on the 12-month targets.

---

## 9. LLM governance

The LLM never predicts. This is enforced, not promised:

1. `tests/test_no_llm_prediction.py::test_no_modelling_module_can_reach_a_language_model` parses the AST of every module under `src/data`, `src/features`, `src/models`, `src/scenarios` and `src/explain` and fails if any imports `anthropic`, `openai`, or `src.copilot`. The modelling code path *cannot* reach a language model.
2. The **grounding validator** extracts every number from generated text and matches it against the grounding pack, including values scaled by 100 or rounded — the forms a helpful model reaches for. Unmatched numbers block the output. Its six-case self-test confirms it catches fabricated probabilities, rescaled figures, causal assertions, overconfident decisions, and missing reviewer framing.

Every prompt, model id, timestamp, response, token count and validator verdict is written to `submission/llm_prompt_log.jsonl`. All LLM output carries *"RECOMMENDATION, NOT DECISION."*

**Known gap.** No Anthropic credential was available in the build environment, so the copilot ran in labelled `offline_template` mode and the live LLM failure transcripts the task calls for are **not yet captured**. The offline template cannot hallucinate, so the adversarial probes have nothing real to catch. Presenting invented transcripts as captured API output would be fabricating evidence, so they are absent rather than filled in. Setting `ANTHROPIC_API_KEY` and re-running `python -m src.copilot.run_copilot` executes the five probes against `claude-opus-5` and regenerates that section.

---

## 10. Intended use and out-of-scope use

**Intended:** decision *support* for a servicing-oversight team — prioritising a review queue, sizing a stress scenario, and surfacing records whose data does not hold together.

**Out of scope:** automated adverse action, credit pricing, underwriting, or any use where output reaches a borrower without human review. The models are fitted on synthetic data and have no validated real-world performance. Fair-lending testing has not been performed; `state` and `servicer_name` are model inputs and would require disparate-impact analysis before production use.

---

## 11. Reproducibility

Fixed seed `20260828` throughout. `python -m src.pipeline` runs data generation through submission and writes `submission/run_manifest.json` recording every stage, its status, duration, and the artefacts produced.
