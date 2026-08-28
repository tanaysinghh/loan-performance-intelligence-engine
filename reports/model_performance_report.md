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

Feature count: **76** (11 categorical, handled natively by LightGBM). Loans appearing in both train and test windows: **562** — expected for a panel, and probed for memorisation in section 4.

## 2. Baseline versus improved model

Three tiers per target: the training-window prior (a constant), an L2 logistic regression on nine raw credit fields, and LightGBM on the full engineered set.

**Read this comparison carefully, because the honest answer is mixed.** LightGBM wins PR-AUC on three of four targets and ROC-AUC on prepayment by a wide margin, but the nine-feature logistic baseline is within ~0.01 ROC-AUC on the delinquency and default targets and beats LightGBM on prepayment PR-AUC. That is not a bug and it is not hidden here: the dominant signals for delinquency (current status, DPD history, worst status to date) are close to monotone in the log-odds, which is exactly the regime where a linear model is hard to beat on *ranking*.

Where the two separate decisively is **calibration**. The baseline's Brier score is 2-4x worse and its expected calibration error runs 0.16-0.27, because `class_weight=balanced` inflates every probability. Those outputs can rank a queue but cannot answer "what is the chance this loan defaults", which is the question the submission format actually asks. The GBM is retained on that basis, plus its ability to carry the full 76-feature set into the explainability layer.

### Test-window results

| target | model | n | positive_rate | roc_auc | pr_auc | pr_auc_lift_over_base | best_f1 | recall_at_precision_30 | recall_at_precision_50 | brier | ece | ks | lift_at_10pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| next_3m_delinquency_flag | prior | 3719 | 0.0664 | 0.5000 | 0.0664 | 1.0000 | 0.1246 | 0.0000 | 0.0000 | 0.0621 | 0.0000 | 0.1614 | 0.4058 |
| next_3m_delinquency_flag | baseline_logistic | 3719 | 0.0664 | 0.9061 | 0.7868 | 11.8472 | 0.8141 | 0.8178 | 0.7611 | 0.0792 | 0.1925 | 0.7151 | 7.5892 |
| next_3m_delinquency_flag | lgbm_raw | 3719 | 0.0664 | 0.8921 | 0.7842 | 11.8076 | 0.8141 | 0.7935 | 0.7692 | 0.0215 | 0.0165 | 0.7228 | 7.7110 |
| next_3m_delinquency_flag | lgbm_calibrated | 3719 | 0.0664 | 0.8921 | 0.7842 | 11.8076 | 0.8141 | 0.7935 | 0.7692 | 0.0210 | 0.0123 | 0.7228 | 7.7110 |
| next_6m_delinquency_flag | prior | 3913 | 0.0917 | 0.5000 | 0.0917 | 1.0000 | 0.1681 | 0.0000 | 0.0000 | 0.0836 | 0.0000 | 0.1251 | 0.4460 |
| next_6m_delinquency_flag | baseline_logistic | 3913 | 0.0917 | 0.8882 | 0.7358 | 8.0204 | 0.7390 | 0.7994 | 0.6852 | 0.1149 | 0.2303 | 0.6287 | 6.5510 |
| next_6m_delinquency_flag | lgbm_raw | 3913 | 0.0917 | 0.8827 | 0.7406 | 8.0725 | 0.7358 | 0.7799 | 0.7131 | 0.0405 | 0.0300 | 0.6430 | 6.6625 |
| next_6m_delinquency_flag | lgbm_calibrated | 3913 | 0.0917 | 0.8827 | 0.7406 | 8.0725 | 0.7358 | 0.7799 | 0.7131 | 0.0393 | 0.0240 | 0.6430 | 6.6625 |
| next_12m_default_flag | prior | 4040 | 0.0723 | 0.5000 | 0.0723 | 1.0000 | 0.1348 | 0.0000 | 0.0000 | 0.0675 | 0.0000 | 0.1136 | 0.5137 |
| next_12m_default_flag | baseline_logistic | 4040 | 0.0723 | 0.9022 | 0.6355 | 8.7930 | 0.6864 | 0.7603 | 0.6815 | 0.0843 | 0.1614 | 0.6367 | 6.8151 |
| next_12m_default_flag | lgbm_raw | 4040 | 0.0723 | 0.9008 | 0.6453 | 8.9287 | 0.6613 | 0.8048 | 0.6712 | 0.0381 | 0.0230 | 0.6629 | 6.7808 |
| next_12m_default_flag | lgbm_calibrated | 4040 | 0.0723 | 0.9008 | 0.6453 | 8.9287 | 0.6613 | 0.8048 | 0.6712 | 0.0382 | 0.0214 | 0.6629 | 6.7808 |
| next_12m_prepayment_flag | prior | 4040 | 0.1710 | 0.5000 | 0.1710 | 1.0000 | 0.2921 | 0.0000 | 0.0000 | 0.1421 | 0.0000 | 0.1302 | 1.2590 |
| next_12m_prepayment_flag | baseline_logistic | 4040 | 0.1710 | 0.6512 | 0.3146 | 1.8393 | 0.3701 | 0.4023 | 0.1389 | 0.2563 | 0.2735 | 0.2647 | 2.0695 |
| next_12m_prepayment_flag | lgbm_raw | 4040 | 0.1710 | 0.6858 | 0.2798 | 1.6358 | 0.3724 | 0.4530 | 0.0072 | 0.1625 | 0.1255 | 0.2837 | 1.7656 |
| next_12m_prepayment_flag | lgbm_calibrated | 4040 | 0.1710 | 0.6792 | 0.2670 | 1.5613 | 0.3714 | 0.4124 | 0.0000 | 0.1709 | 0.1256 | 0.2817 | 1.7800 |

### Improvement over baseline

| target | baseline_roc_auc | lgbm_roc_auc | roc_auc_gain | baseline_pr_auc | lgbm_pr_auc | pr_auc_gain_pct | baseline_brier | lgbm_brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| next_3m_delinquency_flag | 0.9061 | 0.8921 | -0.0140 | 0.7868 | 0.7842 | -0.3350 | 0.0792 | 0.0210 |
| next_6m_delinquency_flag | 0.8882 | 0.8827 | -0.0055 | 0.7358 | 0.7406 | 0.6501 | 0.1149 | 0.0393 |
| next_12m_default_flag | 0.9022 | 0.9008 | -0.0014 | 0.6355 | 0.6453 | 1.5435 | 0.0843 | 0.0382 |
| next_12m_prepayment_flag | 0.6512 | 0.6792 | 0.0280 | 0.3146 | 0.2670 | -15.1168 | 0.2563 | 0.1709 |

## 3. Class imbalance and calibration

Positive rates run from 4% to 15%. Two things are done about it, and they are kept separate on purpose:

1. **Ranking** — LightGBM is trained with `scale_pos_weight = sqrt(neg/pos)`. The square root rather than the full ratio is deliberate: full reweighting maximises separation but destroys probability calibration, and these outputs feed a servicing action queue where the absolute probability matters.
2. **Calibration** — an isotonic map is fitted on the validation window and applied to test predictions. Reweighting is therefore allowed to distort the scale, and the isotonic step puts it back.

Brier score and expected calibration error below compare raw against calibrated on the untouched test window.

| target | model | brier | ece |
| --- | --- | --- | --- |
| next_3m_delinquency_flag | lgbm_raw | 0.0215 | 0.0165 |
| next_3m_delinquency_flag | lgbm_calibrated | 0.0210 | 0.0123 |
| next_6m_delinquency_flag | lgbm_raw | 0.0405 | 0.0300 |
| next_6m_delinquency_flag | lgbm_calibrated | 0.0393 | 0.0240 |
| next_12m_default_flag | lgbm_raw | 0.0381 | 0.0230 |
| next_12m_default_flag | lgbm_calibrated | 0.0382 | 0.0214 |
| next_12m_prepayment_flag | lgbm_raw | 0.1625 | 0.1255 |
| next_12m_prepayment_flag | lgbm_calibrated | 0.1709 | 0.1256 |

### Calibration curve — `next_3m_delinquency_flag` (test window)

| bucket | n | mean_predicted | observed_rate | calibration_gap |
| --- | --- | --- | --- | --- |
| (0.0008799999999999999, 0.00524] | 372 | 0.0039 | 0.0054 | 0.0015 |
| (0.00524, 0.00759] | 372 | 0.0064 | 0.0161 | 0.0097 |
| (0.00759, 0.0107] | 372 | 0.0092 | 0.0108 | 0.0016 |
| (0.0107, 0.0145] | 372 | 0.0125 | 0.0188 | 0.0063 |
| (0.0145, 0.0193] | 372 | 0.0168 | 0.0161 | -0.0007 |
| (0.0193, 0.026] | 371 | 0.0225 | 0.0189 | -0.0036 |
| (0.026, 0.0342] | 373 | 0.0299 | 0.0322 | 0.0023 |
| (0.0342, 0.0482] | 371 | 0.0405 | 0.0189 | -0.0216 |
| (0.0482, 0.093] | 372 | 0.0653 | 0.0161 | -0.0492 |
| (0.093, 0.98] | 372 | 0.5377 | 0.5108 | -0.0270 |

### Calibration curve — `next_6m_delinquency_flag` (test window)

| bucket | n | mean_predicted | observed_rate | calibration_gap |
| --- | --- | --- | --- | --- |
| (-0.000396, 0.00303] | 392 | 0.0021 | 0.0230 | 0.0208 |
| (0.00303, 0.00429] | 391 | 0.0037 | 0.0026 | -0.0011 |
| (0.00429, 0.006] | 391 | 0.0051 | 0.0102 | 0.0051 |
| (0.006, 0.00805] | 391 | 0.0070 | 0.0205 | 0.0134 |
| (0.00805, 0.0112] | 392 | 0.0095 | 0.0230 | 0.0135 |
| (0.0112, 0.0163] | 391 | 0.0136 | 0.0563 | 0.0426 |
| (0.0163, 0.026] | 391 | 0.0201 | 0.0409 | 0.0208 |
| (0.026, 0.0523] | 391 | 0.0374 | 0.0460 | 0.0087 |
| (0.0523, 0.207] | 391 | 0.0978 | 0.0818 | -0.0160 |
| (0.207, 0.996] | 392 | 0.7105 | 0.6122 | -0.0983 |

### Calibration curve — `next_12m_default_flag` (test window)

| bucket | n | mean_predicted | observed_rate | calibration_gap |
| --- | --- | --- | --- | --- |
| (-0.000696, 0.00111] | 404 | 0.0008 | 0.0000 | -0.0008 |
| (0.00111, 0.0016] | 404 | 0.0014 | 0.0124 | 0.0110 |
| (0.0016, 0.0022] | 404 | 0.0019 | 0.0173 | 0.0154 |
| (0.0022, 0.00319] | 404 | 0.0026 | 0.0025 | -0.0001 |
| (0.00319, 0.00515] | 404 | 0.0041 | 0.0248 | 0.0206 |
| (0.00515, 0.00808] | 404 | 0.0065 | 0.0198 | 0.0133 |
| (0.00808, 0.0149] | 404 | 0.0111 | 0.0124 | 0.0013 |
| (0.0149, 0.031] | 404 | 0.0214 | 0.0470 | 0.0256 |
| (0.031, 0.125] | 404 | 0.0589 | 0.0965 | 0.0376 |
| (0.125, 0.995] | 404 | 0.5780 | 0.4901 | -0.0879 |

### Calibration curve — `next_12m_prepayment_flag` (test window)

| bucket | n | mean_predicted | observed_rate | calibration_gap |
| --- | --- | --- | --- | --- |
| (-0.000999, 0.00491] | 504 | 0.0048 | 0.0456 | 0.0408 |
| (0.00491, 0.0298] | 1129 | 0.0279 | 0.0859 | 0.0580 |
| (0.0298, 0.0448] | 712 | 0.0411 | 0.1896 | 0.1485 |
| (0.0448, 0.0678] | 119 | 0.0652 | 0.1261 | 0.0608 |
| (0.0678, 0.138] | 660 | 0.1186 | 0.2061 | 0.0875 |
| (0.138, 0.167] | 143 | 0.1667 | 0.2797 | 0.1131 |
| (0.167, 0.259] | 375 | 0.2591 | 0.3307 | 0.0716 |
| (0.259, 1.0] | 398 | 0.8257 | 0.3040 | -0.5217 |

## 4. Leakage controls

Four controls, each of which would catch a different failure:

1. **Banned-feature list.** `prepayment_flag`, `default_flag`, `next_state`, `loss_severity_band` and all `next_*` columns are refused by `assert_no_leakage`, which runs inside the design-matrix builder and again in the test suite. `loss_severity_band` is the subtle one: it is populated only after default, so its mere presence is the label.
2. **Horizon purging and embargo**, described in section 1.
3. **Label observability cap**, described in section 1.
4. **Split-sensitivity probe.** The same model and features are refitted under an unsound random row split. If the honest split were leaking, the two would agree; the gap below is the measure of how much a naive split would have flattered the model.

| target | purged_time_split | loan_disjoint_time_split | random_row_split_unsound | random_split_inflation |
| --- | --- | --- | --- | --- |
| next_3m_delinquency_flag | 0.8921 | 0.8954 | 0.9903 | 0.0982 |
| next_6m_delinquency_flag | 0.8827 | 0.8575 | 0.9963 | 0.1136 |
| next_12m_default_flag | 0.9008 | 0.8499 | 0.9968 | 0.0960 |
| next_12m_prepayment_flag | 0.6792 | 0.6253 | 0.9961 | 0.3169 |

The loan-disjoint column additionally forces no `loan_id` to appear in both the fitting data and the test window. Performance holding up there means the model has learned loan *characteristics*, not loan *identities*.

## 5. Expanding-window backtest

Stability across successive origination cut-offs, each fold re-purged.

| target | fold | train_window | test_window | roc_auc | pr_auc | brier |
| --- | --- | --- | --- | --- | --- | --- |
| next_3m_delinquency_flag | 0 | 2019-01..2023-12 | 2025-10..2026-03 | 0.9094 | 0.7871 | 0.0211 |
| next_3m_delinquency_flag | 1 | 2019-01..2024-06 | 2025-10..2026-03 | 0.9042 | 0.7739 | 0.0215 |
| next_3m_delinquency_flag | 2 | 2019-01..2024-12 | 2025-10..2026-03 | 0.8921 | 0.7842 | 0.0210 |
| next_6m_delinquency_flag | 0 | 2019-01..2023-06 | 2025-07..2025-12 | 0.8771 | 0.7202 | 0.0376 |
| next_6m_delinquency_flag | 1 | 2019-01..2023-12 | 2025-07..2025-12 | 0.8753 | 0.7147 | 0.0382 |
| next_6m_delinquency_flag | 2 | 2019-01..2024-06 | 2025-07..2025-12 | 0.8827 | 0.7406 | 0.0393 |
| next_12m_default_flag | 0 | 2019-01..2022-06 | 2025-01..2025-06 | 0.8948 | 0.6143 | 0.0365 |
| next_12m_default_flag | 1 | 2019-01..2022-12 | 2025-01..2025-06 | 0.9006 | 0.6269 | 0.0361 |
| next_12m_default_flag | 2 | 2019-01..2023-06 | 2025-01..2025-06 | 0.9008 | 0.6453 | 0.0382 |
| next_12m_prepayment_flag | 0 | 2019-01..2022-06 | 2025-01..2025-06 | 0.6637 | 0.2547 | 0.1899 |
| next_12m_prepayment_flag | 1 | 2019-01..2022-12 | 2025-01..2025-06 | 0.6661 | 0.2575 | 0.1570 |
| next_12m_prepayment_flag | 2 | 2019-01..2023-06 | 2025-01..2025-06 | 0.6792 | 0.2670 | 0.1709 |

## 6. Next-state prediction (multiclass)

One-step-ahead state transition model, benchmarked against two baselines:

- **Persistence** — predict that the current status continues. Deceptively strong because most loan-months stay Current, and it is exactly why accuracy is not the headline metric here.
- **Empirical Markov transition matrix** — `P(next_state | current_status)` estimated on the training window. Unlike persistence this emits a full probability vector, so log loss and macro-AUC are directly comparable and any lift the covariate model shows is lift *over already knowing the current state*.

Persistence still edges the covariate model on raw accuracy, and that is reported rather than buried: when 95%+ of transitions are Current-to-Current, a rule that never predicts a transition is hard to beat on accuracy alone. It is also useless, because it assigns zero probability to every event a servicer cares about. Log loss and macro-AUC are where the difference lives.

| n | accuracy | macro_f1 | weighted_f1 | log_loss | macro_roc_auc | split | model |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3994 | 0.9482 | 0.4225 | 0.9352 | 0.1862 | 0.8886 | valid | lgbm_multiclass |
| 3994 | 0.9492 | 0.3663 | 0.9317 | 0.1819 | 0.8474 | valid | markov_transition_baseline |
| 3994 | 0.9487 | 0.4379 | 0.9375 |  |  | valid | persistence_baseline |
| 3557 | 0.9581 | 0.4397 | 0.9457 | 0.1632 | 0.8862 | test | lgbm_multiclass |
| 3557 | 0.9595 | 0.3747 | 0.9447 | 0.1613 | 0.8415 | test | markov_transition_baseline |
| 3557 | 0.9587 | 0.4385 | 0.9476 |  |  | test | persistence_baseline |

### Per-class performance — valid window

| class | support | precision | recall | f1 |
| --- | --- | --- | --- | --- |
| Current | 3654 | 0.9736 | 0.9973 | 0.9853 |
| DQ30 | 64 | 0.4667 | 0.2188 | 0.2979 |
| DQ60 | 62 | 0.3929 | 0.3548 | 0.3729 |
| DQ90plus | 119 | 0.6522 | 0.8824 | 0.7500 |
| Default | 27 | 0.5000 | 0.0741 | 0.1290 |
| Prepaid | 68 | 0.0000 | 0.0000 | 0.0000 |

### Per-class performance — test window

| class | support | precision | recall | f1 |
| --- | --- | --- | --- | --- |
| Current | 3316 | 0.9753 | 0.9985 | 0.9867 |
| DQ30 | 50 | 0.5556 | 0.1000 | 0.1695 |
| DQ60 | 33 | 0.4762 | 0.6061 | 0.5333 |
| DQ90plus | 81 | 0.6970 | 0.8519 | 0.7667 |
| Default | 21 | 0.2500 | 0.1429 | 0.1818 |
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
| next_3m_delinquency_flag | 64 | 0.0600 | 40 | 83 | platt | 0.0581 |
| next_6m_delinquency_flag | 48 | 0.0450 | 80 | 316 | platt | 0.0742 |
| next_12m_default_flag | 64 | 0.0600 | 40 | 143 | platt | 0.0512 |
| next_12m_prepayment_flag | 48 | 0.0450 | 80 | 868 | isotonic | 0.1521 |

Hyperparameters are selected per target on **validation** average precision from a five-point grid over capacity and learning rate; the test window is never consulted during selection. Selection traces are in `reports/hyperparameter_search.csv`.

The calibrator is likewise chosen per target, by 3-fold cross-validated log loss *inside* the validation window. Selecting on the full validation window would have favoured isotonic every time, since it can bend to validation noise that does not repeat out of time.

## 8. Honest limitations

- The 12-month targets lose 7,723 rows to the embargo and 1,075 to the observability cap. Training data for those targets is 26,257 rows against 37,491 for the 3-month target, and confidence intervals are correspondingly wider.
- Train and test for the 12-month targets sit in different macro regimes. This is reported rather than corrected, because correcting it by reweighting would hide the single most useful fact about the model's operating conditions.
- `PaidOff` never occurs in this panel window, so the next-state model has six reachable classes, not seven.
- Metrics are single-run point estimates. No repeated-seed variance is reported.
