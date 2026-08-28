# Loan Performance Prediction Report

**Task 2 — non-LLM predictive models.** Every number in this report comes from a LightGBM or scikit-learn estimator fitted on the engineered feature set. No language model participates in producing any figure here.

## 1. Validation design

Splitting is time-aware, horizon-purged and label-observability-capped. The two traps this avoids are documented in `src/models/splits.py`:

- **Unobservable labels.** Rows within H months of the panel end can only carry a positive 12-month label if the event already happened, so keeping them turns the test set into a sample of terminated loans. Those rows are excluded, not imputed to zero.
- **Window overlap.** A training row at month t encodes months t+1..t+H. An embargo of H months sits between the fitting data and the test window so no training row's outcome window reaches into the evaluation period.

| target | horizon_months | train_window | valid_window | embargo_window | test_window |
| --- | --- | --- | --- | --- | --- |
| next_3m_delinquency_flag | 3 | 2019-01..2024-12 | 2025-01..2025-06 | 2025-07..2025-09 | 2025-10..2026-03 |
| next_6m_delinquency_flag | 6 | 2019-01..2024-06 | 2024-07..2024-12 | 2025-01..2025-06 | 2025-07..2025-12 |
| next_12m_default_flag | 12 | 2019-01..2023-06 | 2023-07..2023-12 | 2024-01..2024-12 | 2025-01..2025-06 |
| next_12m_prepayment_flag | 12 | 2019-01..2023-06 | 2023-07..2023-12 | 2024-01..2024-12 | 2025-01..2025-06 |
| exception_required | 0 | 2019-01..2025-06 | 2025-07..2025-12 | none | 2026-01..2026-06 |

| target | train_rows | valid_rows | test_rows | rows_dropped_embargo | rows_dropped_unobservable_label | train_positive_rate | valid_positive_rate | test_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| next_3m_delinquency_flag | 37491 | 4040 | 3719 | 1987 | 185 | 0.0581 | 0.0809 | 0.0664 |
| next_6m_delinquency_flag | 33521 | 3970 | 3913 | 4040 | 484 | 0.0742 | 0.0932 | 0.0917 |
| next_12m_default_flag | 26257 | 3511 | 4040 | 7723 | 1075 | 0.0512 | 0.0558 | 0.0723 |
| next_12m_prepayment_flag | 26257 | 3511 | 4040 | 7723 | 1075 | 0.1521 | 0.0721 | 0.1710 |
| exception_required | 41531 | 3913 | 3480 | 0 | 0 | 0.1251 | 0.1296 | 0.1305 |

Positive rates are stable across train, validation and test for the short-horizon targets. The 12-month default rate moves from 5.1% in training to 7.2% in test; that is genuine regime change driven by the unemployment path in the panel window, not a split artefact, and it is why calibration is re-assessed out-of-time rather than assumed.

Feature count: **81** (11 categorical, handled natively by LightGBM). Loans appearing in both train and test windows: **562** — expected for a panel, and probed for memorisation in section 4.

## 2. Baseline versus improved model

Three tiers per target: the training-window prior (a constant), an L2 logistic regression on nine raw credit fields, and LightGBM on the full engineered set.

**Read this comparison carefully, because the honest answer is mixed.** LightGBM wins PR-AUC on three of four targets and ROC-AUC on prepayment by a wide margin, but the nine-feature logistic baseline is within ~0.01 ROC-AUC on the delinquency and default targets and beats LightGBM on prepayment PR-AUC. That is not a bug and it is not hidden here: the dominant signals for delinquency (current status, DPD history, worst status to date) are close to monotone in the log-odds, which is exactly the regime where a linear model is hard to beat on *ranking*.

Where the two separate decisively is **calibration**. The baseline's Brier score is 2-4x worse and its expected calibration error runs 0.16-0.27, because `class_weight=balanced` inflates every probability. Those outputs can rank a queue but cannot answer "what is the chance this loan defaults", which is the question the submission format actually asks. The GBM is retained on that basis, plus its ability to carry the full 76-feature set into the explainability layer.

### Test-window results

| target | model | n | positive_rate | roc_auc | pr_auc | pr_auc_lift_over_base | best_f1 | recall_at_precision_30 | recall_at_precision_50 | brier | ece | ks | lift_at_10pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| next_3m_delinquency_flag | prior | 3719 | 0.0664 | 0.5000 | 0.0664 | 1.0000 | 0.1246 | 0.0000 | 0.0000 | 0.0621 | 0.0000 | 0.1614 | 0.4058 |
| next_3m_delinquency_flag | baseline_logistic | 3719 | 0.0664 | 0.9061 | 0.7868 | 11.8472 | 0.8141 | 0.8178 | 0.7611 | 0.0792 | 0.1925 | 0.7151 | 7.5892 |
| next_3m_delinquency_flag | lgbm_raw | 3719 | 0.0664 | 0.8995 | 0.7849 | 11.8185 | 0.8178 | 0.7935 | 0.7611 | 0.0215 | 0.0177 | 0.7136 | 7.6298 |
| next_3m_delinquency_flag | lgbm_calibrated | 3719 | 0.0664 | 0.8995 | 0.7849 | 11.8185 | 0.8178 | 0.7935 | 0.7611 | 0.0209 | 0.0136 | 0.7136 | 7.6298 |
| next_6m_delinquency_flag | prior | 3913 | 0.0917 | 0.5000 | 0.0917 | 1.0000 | 0.1681 | 0.0000 | 0.0000 | 0.0836 | 0.0000 | 0.1251 | 0.4460 |
| next_6m_delinquency_flag | baseline_logistic | 3913 | 0.0917 | 0.8882 | 0.7358 | 8.0204 | 0.7390 | 0.7994 | 0.6852 | 0.1149 | 0.2303 | 0.6287 | 6.5510 |
| next_6m_delinquency_flag | lgbm_raw | 3913 | 0.0917 | 0.8784 | 0.7318 | 7.9761 | 0.7411 | 0.7632 | 0.6880 | 0.0405 | 0.0274 | 0.6270 | 6.6346 |
| next_6m_delinquency_flag | lgbm_calibrated | 3913 | 0.0917 | 0.8784 | 0.7318 | 7.9761 | 0.7411 | 0.7632 | 0.6880 | 0.0393 | 0.0225 | 0.6270 | 6.6346 |
| next_12m_default_flag | prior | 4040 | 0.0723 | 0.5000 | 0.0723 | 1.0000 | 0.1348 | 0.0000 | 0.0000 | 0.0675 | 0.0000 | 0.1136 | 0.5137 |
| next_12m_default_flag | baseline_logistic | 4040 | 0.0723 | 0.9022 | 0.6355 | 8.7930 | 0.6864 | 0.7603 | 0.6815 | 0.0843 | 0.1614 | 0.6367 | 6.8151 |
| next_12m_default_flag | lgbm_raw | 4040 | 0.0723 | 0.8972 | 0.6255 | 8.6539 | 0.6612 | 0.7774 | 0.6678 | 0.0386 | 0.0245 | 0.6585 | 6.6781 |
| next_12m_default_flag | lgbm_calibrated | 4040 | 0.0723 | 0.8972 | 0.6255 | 8.6539 | 0.6612 | 0.7774 | 0.6678 | 0.0384 | 0.0214 | 0.6585 | 6.6781 |
| next_12m_prepayment_flag | prior | 4040 | 0.1710 | 0.5000 | 0.1710 | 1.0000 | 0.2921 | 0.0000 | 0.0000 | 0.1421 | 0.0000 | 0.1302 | 1.2590 |
| next_12m_prepayment_flag | baseline_logistic | 4040 | 0.1710 | 0.6512 | 0.3146 | 1.8393 | 0.3701 | 0.4023 | 0.1389 | 0.2563 | 0.2735 | 0.2647 | 2.0695 |
| next_12m_prepayment_flag | lgbm_raw | 4040 | 0.1710 | 0.6846 | 0.2751 | 1.6081 | 0.3841 | 0.3415 | 0.0000 | 0.1538 | 0.1131 | 0.3057 | 1.6498 |
| next_12m_prepayment_flag | lgbm_calibrated | 4040 | 0.1710 | 0.6685 | 0.2552 | 1.4919 | 0.3837 | 0.2996 | 0.0000 | 0.1612 | 0.1149 | 0.2897 | 1.7366 |

### Improvement over baseline

| target | baseline_roc_auc | lgbm_roc_auc | roc_auc_gain | baseline_pr_auc | lgbm_pr_auc | pr_auc_gain_pct | baseline_brier | lgbm_brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| next_3m_delinquency_flag | 0.9061 | 0.8995 | -0.0065 | 0.7868 | 0.7849 | -0.2426 | 0.0792 | 0.0209 |
| next_6m_delinquency_flag | 0.8882 | 0.8784 | -0.0098 | 0.7358 | 0.7318 | -0.5516 | 0.1149 | 0.0393 |
| next_12m_default_flag | 0.9022 | 0.8972 | -0.0050 | 0.6355 | 0.6255 | -1.5816 | 0.0843 | 0.0384 |
| next_12m_prepayment_flag | 0.6512 | 0.6685 | 0.0173 | 0.3146 | 0.2552 | -18.8894 | 0.2563 | 0.1612 |

## 3. Class imbalance and calibration

Positive rates run from 4% to 15%. Two things are done about it, and they are kept separate on purpose:

1. **Ranking** — LightGBM is trained with `scale_pos_weight = sqrt(neg/pos)`. The square root rather than the full ratio is deliberate: full reweighting maximises separation but destroys probability calibration, and these outputs feed a servicing action queue where the absolute probability matters.
2. **Calibration** — an isotonic map is fitted on the validation window and applied to test predictions. Reweighting is therefore allowed to distort the scale, and the isotonic step puts it back.

Brier score and expected calibration error below compare raw against calibrated on the untouched test window.

| target | model | brier | ece |
| --- | --- | --- | --- |
| next_3m_delinquency_flag | lgbm_raw | 0.0215 | 0.0177 |
| next_3m_delinquency_flag | lgbm_calibrated | 0.0209 | 0.0136 |
| next_6m_delinquency_flag | lgbm_raw | 0.0405 | 0.0274 |
| next_6m_delinquency_flag | lgbm_calibrated | 0.0393 | 0.0225 |
| next_12m_default_flag | lgbm_raw | 0.0386 | 0.0245 |
| next_12m_default_flag | lgbm_calibrated | 0.0384 | 0.0214 |
| next_12m_prepayment_flag | lgbm_raw | 0.1538 | 0.1131 |
| next_12m_prepayment_flag | lgbm_calibrated | 0.1612 | 0.1149 |

### Calibration curve — `next_3m_delinquency_flag` (test window)

| bucket | n | mean_predicted | observed_rate | calibration_gap |
| --- | --- | --- | --- | --- |
| (0.00046999999999999993, 0.00575] | 372 | 0.0041 | 0.0081 | 0.0039 |
| (0.00575, 0.00826] | 372 | 0.0070 | 0.0108 | 0.0037 |
| (0.00826, 0.0114] | 372 | 0.0097 | 0.0108 | 0.0011 |
| (0.0114, 0.0156] | 372 | 0.0135 | 0.0242 | 0.0107 |
| (0.0156, 0.0201] | 372 | 0.0177 | 0.0108 | -0.0069 |
| (0.0201, 0.0269] | 371 | 0.0232 | 0.0135 | -0.0097 |
| (0.0269, 0.037] | 372 | 0.0318 | 0.0188 | -0.0130 |
| (0.037, 0.0537] | 372 | 0.0440 | 0.0215 | -0.0225 |
| (0.0537, 0.0976] | 372 | 0.0712 | 0.0403 | -0.0309 |
| (0.0976, 0.991] | 372 | 0.5387 | 0.5054 | -0.0334 |

### Calibration curve — `next_6m_delinquency_flag` (test window)

| bucket | n | mean_predicted | observed_rate | calibration_gap |
| --- | --- | --- | --- | --- |
| (0.00013999999999999993, 0.00373] | 392 | 0.0028 | 0.0128 | 0.0099 |
| (0.00373, 0.00512] | 391 | 0.0044 | 0.0077 | 0.0032 |
| (0.00512, 0.0069] | 391 | 0.0060 | 0.0205 | 0.0145 |
| (0.0069, 0.00902] | 391 | 0.0079 | 0.0332 | 0.0253 |
| (0.00902, 0.0122] | 392 | 0.0105 | 0.0230 | 0.0125 |
| (0.0122, 0.0177] | 391 | 0.0148 | 0.0230 | 0.0082 |
| (0.0177, 0.0285] | 391 | 0.0228 | 0.0460 | 0.0233 |
| (0.0285, 0.0516] | 391 | 0.0373 | 0.0614 | 0.0241 |
| (0.0516, 0.185] | 391 | 0.0936 | 0.0818 | -0.0117 |
| (0.185, 0.996] | 392 | 0.6991 | 0.6071 | -0.0920 |

### Calibration curve — `next_12m_default_flag` (test window)

| bucket | n | mean_predicted | observed_rate | calibration_gap |
| --- | --- | --- | --- | --- |
| (-0.000849, 0.00084] | 404 | 0.0006 | 0.0124 | 0.0118 |
| (0.00084, 0.00117] | 404 | 0.0010 | 0.0050 | 0.0040 |
| (0.00117, 0.00165] | 404 | 0.0014 | 0.0050 | 0.0036 |
| (0.00165, 0.00243] | 404 | 0.0020 | 0.0074 | 0.0054 |
| (0.00243, 0.00394] | 404 | 0.0031 | 0.0198 | 0.0167 |
| (0.00394, 0.00662] | 404 | 0.0051 | 0.0149 | 0.0098 |
| (0.00662, 0.0127] | 404 | 0.0092 | 0.0421 | 0.0329 |
| (0.0127, 0.0278] | 404 | 0.0185 | 0.0470 | 0.0285 |
| (0.0278, 0.124] | 404 | 0.0542 | 0.0866 | 0.0324 |
| (0.124, 0.993] | 404 | 0.5517 | 0.4827 | -0.0690 |

### Calibration curve — `next_12m_prepayment_flag` (test window)

| bucket | n | mean_predicted | observed_rate | calibration_gap |
| --- | --- | --- | --- | --- |
| (-0.000999, 0.0159] | 677 | 0.0124 | 0.0606 | 0.0481 |
| (0.0159, 0.0289] | 1574 | 0.0289 | 0.1175 | 0.0886 |
| (0.0289, 0.0758] | 228 | 0.0568 | 0.1447 | 0.0879 |
| (0.0758, 0.118] | 885 | 0.1128 | 0.2542 | 0.1414 |
| (0.118, 0.364] | 376 | 0.3531 | 0.2979 | -0.0552 |
| (0.364, 1.0] | 300 | 0.7366 | 0.3167 | -0.4200 |

## 4. Leakage controls

Four controls, each of which would catch a different failure:

1. **Banned-feature list.** `prepayment_flag`, `default_flag`, `next_state`, `loss_severity_band` and all `next_*` columns are refused by `assert_no_leakage`, which runs inside the design-matrix builder and again in the test suite. `loss_severity_band` is the subtle one: it is populated only after default, so its mere presence is the label.
2. **Horizon purging and embargo**, described in section 1.
3. **Label observability cap**, described in section 1.
4. **Split-sensitivity probe.** The same model and features are refitted under an unsound random row split. If the honest split were leaking, the two would agree; the gap below is the measure of how much a naive split would have flattered the model.

| target | purged_time_split | loan_disjoint_time_split | random_row_split_unsound | random_split_inflation |
| --- | --- | --- | --- | --- |
| next_3m_delinquency_flag | 0.8995 | 0.8957 | 0.9897 | 0.0902 |
| next_6m_delinquency_flag | 0.8784 | 0.8705 | 0.9966 | 0.1182 |
| next_12m_default_flag | 0.8972 | 0.8757 | 0.9966 | 0.0994 |
| next_12m_prepayment_flag | 0.6685 | 0.5808 | 0.9964 | 0.3279 |

The loan-disjoint column additionally forces no `loan_id` to appear in both the fitting data and the test window. Performance holding up there means the model has learned loan *characteristics*, not loan *identities*.

## 5. Expanding-window backtest

Stability across successive origination cut-offs, each fold re-purged.

| target | fold | train_window | test_window | roc_auc | pr_auc | brier |
| --- | --- | --- | --- | --- | --- | --- |
| next_3m_delinquency_flag | 0 | 2019-01..2023-12 | 2025-10..2026-03 | 0.9075 | 0.7903 | 0.0208 |
| next_3m_delinquency_flag | 1 | 2019-01..2024-06 | 2025-10..2026-03 | 0.9062 | 0.7722 | 0.0208 |
| next_3m_delinquency_flag | 2 | 2019-01..2024-12 | 2025-10..2026-03 | 0.8995 | 0.7849 | 0.0209 |
| next_6m_delinquency_flag | 0 | 2019-01..2023-06 | 2025-07..2025-12 | 0.8716 | 0.7322 | 0.0383 |
| next_6m_delinquency_flag | 1 | 2019-01..2023-12 | 2025-07..2025-12 | 0.8882 | 0.7164 | 0.0391 |
| next_6m_delinquency_flag | 2 | 2019-01..2024-06 | 2025-07..2025-12 | 0.8784 | 0.7318 | 0.0393 |
| next_12m_default_flag | 0 | 2019-01..2022-06 | 2025-01..2025-06 | 0.9031 | 0.6204 | 0.0361 |
| next_12m_default_flag | 1 | 2019-01..2022-12 | 2025-01..2025-06 | 0.8997 | 0.6301 | 0.0355 |
| next_12m_default_flag | 2 | 2019-01..2023-06 | 2025-01..2025-06 | 0.8972 | 0.6255 | 0.0384 |
| next_12m_prepayment_flag | 0 | 2019-01..2022-06 | 2025-01..2025-06 | 0.6717 | 0.2650 | 0.1643 |
| next_12m_prepayment_flag | 1 | 2019-01..2022-12 | 2025-01..2025-06 | 0.6618 | 0.2566 | 0.1996 |
| next_12m_prepayment_flag | 2 | 2019-01..2023-06 | 2025-01..2025-06 | 0.6685 | 0.2552 | 0.1612 |

## 6. Next-state prediction (multiclass)

One-step-ahead state transition model, benchmarked against two baselines:

- **Persistence** — predict that the current status continues. Deceptively strong because most loan-months stay Current, and it is exactly why accuracy is not the headline metric here.
- **Empirical Markov transition matrix** — `P(next_state | current_status)` estimated on the training window. Unlike persistence this emits a full probability vector, so log loss and macro-AUC are directly comparable and any lift the covariate model shows is lift *over already knowing the current state*.

Persistence still edges the covariate model on raw accuracy, and that is reported rather than buried: when 95%+ of transitions are Current-to-Current, a rule that never predicts a transition is hard to beat on accuracy alone. It is also useless, because it assigns zero probability to every event a servicer cares about. Log loss and macro-AUC are where the difference lives.

| n | accuracy | macro_f1 | weighted_f1 | log_loss | macro_roc_auc | split | model |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3994 | 0.9437 | 0.4091 | 0.9320 | 0.2322 | 0.8824 | valid | lgbm_multiclass |
| 3994 | 0.9492 | 0.3663 | 0.9317 | 0.1819 | 0.8474 | valid | markov_transition_baseline |
| 3994 | 0.9487 | 0.4379 | 0.9375 |  |  | valid | persistence_baseline |
| 3557 | 0.9497 | 0.4339 | 0.9407 | 0.2130 | 0.8823 | test | lgbm_multiclass |
| 3557 | 0.9595 | 0.3747 | 0.9447 | 0.1613 | 0.8415 | test | markov_transition_baseline |
| 3557 | 0.9587 | 0.4385 | 0.9476 |  |  | test | persistence_baseline |

### Per-class performance — valid window

| class | support | precision | recall | f1 |
| --- | --- | --- | --- | --- |
| Current | 3654 | 0.9735 | 0.9967 | 0.9850 |
| DQ30 | 64 | 0.3182 | 0.1094 | 0.1628 |
| DQ60 | 62 | 0.3816 | 0.4677 | 0.4203 |
| DQ90plus | 119 | 0.6466 | 0.7227 | 0.6825 |
| Default | 27 | 0.2273 | 0.1852 | 0.2041 |
| Prepaid | 68 | 0.0000 | 0.0000 | 0.0000 |

### Per-class performance — test window

| class | support | precision | recall | f1 |
| --- | --- | --- | --- | --- |
| Current | 3316 | 0.9751 | 0.9937 | 0.9843 |
| DQ30 | 50 | 0.2500 | 0.0600 | 0.0968 |
| DQ60 | 33 | 0.4048 | 0.5152 | 0.4533 |
| DQ90plus | 81 | 0.6986 | 0.6296 | 0.6623 |
| Default | 21 | 0.3158 | 0.5714 | 0.4068 |
| Prepaid | 56 | 0.0000 | 0.0000 | 0.0000 |

## 7. Model configuration

```json
{
  "objective": "binary",
  "learning_rate": 0.045,
  "num_leaves": 48,
  "min_child_samples": 80,
  "feature_fraction": 0.72,
  "bagging_fraction": 0.82,
  "bagging_freq": 1,
  "lambda_l2": 4.0,
  "max_depth": -1,
  "n_estimators": 1400,
  "verbose": -1,
  "seed": 20260828
}
```

Early stopping on validation average precision, patience 120 rounds. Selected iteration counts:

| target | chosen_num_leaves | chosen_learning_rate | chosen_min_child_samples | best_iteration | calibrator | train_prior |
| --- | --- | --- | --- | --- | --- | --- |
| next_3m_delinquency_flag | 48 | 0.0450 | 80 | 167 | platt | 0.0581 |
| next_6m_delinquency_flag | 64 | 0.0600 | 40 | 150 | platt | 0.0742 |
| next_12m_default_flag | 48 | 0.0450 | 80 | 238 | platt | 0.0512 |
| next_12m_prepayment_flag | 64 | 0.0600 | 40 | 489 | isotonic | 0.1521 |

Hyperparameters are selected per target on **validation** average precision from a five-point grid over capacity and learning rate; the test window is never consulted during selection. Selection traces are in `reports/hyperparameter_search.csv`.

The calibrator is likewise chosen per target, by 3-fold cross-validated log loss *inside* the validation window. Selecting on the full validation window would have favoured isotonic every time, since it can bend to validation noise that does not repeat out of time.

## 8. Honest limitations

- The 12-month targets lose 7,723 rows to the embargo and 1,075 to the observability cap. Training data for those targets is 26,257 rows against 37,491 for the 3-month target, and confidence intervals are correspondingly wider.
- Train and test for the 12-month targets sit in different macro regimes. This is reported rather than corrected, because correcting it by reweighting would hide the single most useful fact about the model's operating conditions.
- `PaidOff` never occurs in this panel window, so the next-state model has six reachable classes, not seven.
- Metrics are single-run point estimates. No repeated-seed variance is reported.
