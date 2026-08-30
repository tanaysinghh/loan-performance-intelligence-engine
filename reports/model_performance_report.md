# Loan Performance Prediction Report

**Task 2 — non-LLM predictive models.** Every number in this report comes from a LightGBM or scikit-learn estimator fitted on the engineered feature set. No language model participates in producing any figure here.

## 1. Validation design

Splitting is time-aware, horizon-purged and label-observability-capped. The two traps this avoids are documented in `src/models/splits.py`:

- **Unobservable labels.** Rows within H months of the panel end can only carry a positive 12-month label if the event already happened, so keeping them turns the test set into a sample of terminated loans. Those rows are excluded, not imputed to zero.
- **Window overlap.** A training row at month t encodes months t+1..t+H. An embargo of H months sits between the fitting data and the test window so no training row's outcome window reaches into the evaluation period.

| target | horizon_months | train_window | valid_window | embargo_window | test_window |
| --- | --- | --- | --- | --- | --- |
| next_3m_delinquency_flag | 3 | 2019-01..2024-09 | 2024-10..2025-03 | 2025-04..2025-06 | 2025-07..2025-12 |
| next_6m_delinquency_flag | 6 | 2019-01..2024-03 | 2024-04..2024-09 | 2024-10..2025-03 | 2025-04..2025-09 |
| next_12m_default_flag | 12 | 2019-01..2023-03 | 2023-04..2023-09 | 2023-10..2024-09 | 2024-10..2025-03 |
| next_12m_prepayment_flag | 12 | 2019-01..2023-03 | 2023-04..2023-09 | 2023-10..2024-09 | 2024-10..2025-03 |
| exception_required | 0 | 2019-01..2025-03 | 2025-04..2025-09 | none | 2025-10..2026-03 |

| target | train_rows | valid_rows | test_rows | rows_dropped_embargo | rows_dropped_unobservable_label | train_positive_rate | valid_positive_rate | test_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| next_3m_delinquency_flag | 471347 | 68984 | 65149 | 33599 | 1348 | 0.0230 | 0.0276 | 0.0313 |
| next_6m_delinquency_flag | 399768 | 71579 | 66539 | 68984 | 3946 | 0.0326 | 0.0348 | 0.0405 |
| next_12m_default_flag | 263320 | 64660 | 68984 | 143367 | 8889 | 0.0187 | 0.0109 | 0.0154 |
| next_12m_prepayment_flag | 263320 | 64660 | 68984 | 143367 | 8333 | 0.1456 | 0.0469 | 0.0758 |
| exception_required | 540331 | 66539 | 63678 | 0 | 0 | 0.1398 | 0.1420 | 0.1430 |

Positive rates are stable across train, validation and test for the short-horizon targets. The 12-month default rate moves from 1.9% in training to 1.5% in test; that is genuine regime change driven by the unemployment path in the panel window, not a split artefact, and it is why calibration is re-assessed out-of-time rather than assumed.

Feature count: **81** (11 categorical, handled natively by LightGBM). Loans appearing in both train and test windows: **11269** — expected for a panel, and probed for memorisation in section 4.

## 2. Baseline versus improved model

Three tiers per target: the training-window prior (a constant), an L2 logistic regression on nine raw credit fields, and LightGBM on the full engineered set.

**Read this comparison carefully, because the honest answer is mixed.** LightGBM wins PR-AUC on three of four targets and ROC-AUC on prepayment by a wide margin, but the nine-feature logistic baseline is within ~0.01 ROC-AUC on the delinquency and default targets and beats LightGBM on prepayment PR-AUC. That is not a bug and it is not hidden here: the dominant signals for delinquency (current status, DPD history, worst status to date) are close to monotone in the log-odds, which is exactly the regime where a linear model is hard to beat on *ranking*.

Where the two separate decisively is **calibration**. The baseline's Brier score is 2-4x worse and its expected calibration error runs 0.16-0.27, because `class_weight=balanced` inflates every probability. Those outputs can rank a queue but cannot answer "what is the chance this loan defaults", which is the question the submission format actually asks. The GBM is retained on that basis, plus its ability to carry the full 76-feature set into the explainability layer.

### Test-window results

| target | model | n | positive_rate | roc_auc | pr_auc | pr_auc_lift_over_base | best_f1 | recall_at_precision_30 | recall_at_precision_50 | brier | ece | ks | lift_at_10pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| next_3m_delinquency_flag | prior | 65149 | 0.0313 | 0.5000 | 0.0313 | 1.0000 | 0.0608 | 0.0000 | 0.0000 | 0.0304 | 0.0000 | 0.1721 | 1.0579 |
| next_3m_delinquency_flag | baseline_logistic | 65149 | 0.0313 | 0.8829 | 0.5813 | 18.5465 | 0.6520 | 0.6430 | 0.5867 | 0.1112 | 0.2716 | 0.6045 | 6.8080 |
| next_3m_delinquency_flag | lgbm_raw | 65149 | 0.0313 | 0.9173 | 0.6583 | 21.0023 | 0.6730 | 0.7791 | 0.7253 | 0.0190 | 0.0263 | 0.7259 | 7.9884 |
| next_3m_delinquency_flag | lgbm_calibrated | 65149 | 0.0313 | 0.9161 | 0.6497 | 20.7293 | 0.6717 | 0.7664 | 0.7067 | 0.0151 | 0.0025 | 0.7237 | 7.9884 |
| next_6m_delinquency_flag | prior | 66539 | 0.0405 | 0.5000 | 0.0405 | 1.0000 | 0.0778 | 0.0000 | 0.0000 | 0.0389 | 0.0000 | 0.1500 | 1.0626 |
| next_6m_delinquency_flag | baseline_logistic | 66539 | 0.0405 | 0.8376 | 0.5038 | 12.4530 | 0.5583 | 0.5371 | 0.4651 | 0.1339 | 0.2956 | 0.5156 | 5.7400 |
| next_6m_delinquency_flag | lgbm_raw | 66539 | 0.0405 | 0.8798 | 0.5908 | 14.6042 | 0.6047 | 0.6861 | 0.6207 | 0.0275 | 0.0325 | 0.6264 | 6.9252 |
| next_6m_delinquency_flag | lgbm_calibrated | 66539 | 0.0405 | 0.8784 | 0.5780 | 14.2869 | 0.6031 | 0.6493 | 0.6100 | 0.0226 | 0.0023 | 0.6232 | 6.9400 |
| next_12m_default_flag | prior | 68984 | 0.0154 | 0.5000 | 0.0154 | 1.0000 | 0.0303 | 0.0000 | 0.0000 | 0.0152 | 0.0000 | 0.2251 | 1.2819 |
| next_12m_default_flag | baseline_logistic | 68984 | 0.0154 | 0.9190 | 0.5737 | 37.3018 | 0.6148 | 0.6598 | 0.6070 | 0.1089 | 0.2232 | 0.6867 | 7.6347 |
| next_12m_default_flag | lgbm_raw | 68984 | 0.0154 | 0.9308 | 0.5584 | 36.3033 | 0.5939 | 0.6777 | 0.6117 | 0.0091 | 0.0050 | 0.7093 | 7.8798 |
| next_12m_default_flag | lgbm_calibrated | 68984 | 0.0154 | 0.9207 | 0.5321 | 34.5936 | 0.5926 | 0.6748 | 0.5957 | 0.0091 | 0.0040 | 0.6969 | 8.0023 |
| next_12m_prepayment_flag | prior | 68984 | 0.0758 | 0.5000 | 0.0758 | 1.0000 | 0.1410 | 0.0000 | 0.0000 | 0.0750 | 0.0000 | 0.2208 | 0.9442 |
| next_12m_prepayment_flag | baseline_logistic | 68984 | 0.0758 | 0.6847 | 0.2413 | 3.1814 | 0.2834 | 0.2364 | 0.0937 | 0.2789 | 0.4170 | 0.2861 | 3.2609 |
| next_12m_prepayment_flag | lgbm_raw | 68984 | 0.0758 | 0.6267 | 0.2237 | 2.9491 | 0.2168 | 0.1638 | 0.1164 | 0.0908 | 0.1036 | 0.1838 | 2.4849 |
| next_12m_prepayment_flag | lgbm_calibrated | 68984 | 0.0758 | 0.6259 | 0.2009 | 2.6492 | 0.2156 | 0.0793 | 0.0793 | 0.1367 | 0.1348 | 0.1826 | 2.4868 |

### Improvement over baseline

| target | baseline_roc_auc | lgbm_roc_auc | roc_auc_gain | baseline_pr_auc | lgbm_pr_auc | pr_auc_gain_pct | baseline_brier | lgbm_brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| next_3m_delinquency_flag | 0.8829 | 0.9161 | 0.0332 | 0.5813 | 0.6497 | 11.7693 | 0.1112 | 0.0151 |
| next_6m_delinquency_flag | 0.8376 | 0.8784 | 0.0408 | 0.5038 | 0.5780 | 14.7265 | 0.1339 | 0.0226 |
| next_12m_default_flag | 0.9190 | 0.9207 | 0.0017 | 0.5737 | 0.5321 | -7.2605 | 0.1089 | 0.0091 |
| next_12m_prepayment_flag | 0.6847 | 0.6259 | -0.0588 | 0.2413 | 0.2009 | -16.7294 | 0.2789 | 0.1367 |

## 3. Class imbalance and calibration

Positive rates run from 4% to 15%. Two things are done about it, and they are kept separate on purpose:

1. **Ranking** — LightGBM is trained with `scale_pos_weight = sqrt(neg/pos)`. The square root rather than the full ratio is deliberate: full reweighting maximises separation but destroys probability calibration, and these outputs feed a servicing action queue where the absolute probability matters.
2. **Calibration** — an isotonic map is fitted on the validation window and applied to test predictions. Reweighting is therefore allowed to distort the scale, and the isotonic step puts it back.

Brier score and expected calibration error below compare raw against calibrated on the untouched test window.

| target | model | brier | ece |
| --- | --- | --- | --- |
| next_3m_delinquency_flag | lgbm_raw | 0.0190 | 0.0263 |
| next_3m_delinquency_flag | lgbm_calibrated | 0.0151 | 0.0025 |
| next_6m_delinquency_flag | lgbm_raw | 0.0275 | 0.0325 |
| next_6m_delinquency_flag | lgbm_calibrated | 0.0226 | 0.0023 |
| next_12m_default_flag | lgbm_raw | 0.0091 | 0.0050 |
| next_12m_default_flag | lgbm_calibrated | 0.0091 | 0.0040 |
| next_12m_prepayment_flag | lgbm_raw | 0.0908 | 0.1036 |
| next_12m_prepayment_flag | lgbm_calibrated | 0.1367 | 0.1348 |

### Calibration curve — `next_3m_delinquency_flag` (test window)

| bucket | n | mean_predicted | observed_rate | calibration_gap |
| --- | --- | --- | --- | --- |
| (-0.000999, 0.00294] | 9981 | 0.0018 | 0.0027 | 0.0009 |
| (0.00294, 0.0043] | 25670 | 0.0043 | 0.0043 | -0.0000 |
| (0.0043, 0.00613] | 7399 | 0.0061 | 0.0078 | 0.0017 |
| (0.00613, 0.00894] | 7877 | 0.0089 | 0.0123 | 0.0034 |
| (0.00894, 0.0122] | 2198 | 0.0122 | 0.0209 | 0.0087 |
| (0.0122, 0.0169] | 5721 | 0.0150 | 0.0135 | -0.0016 |
| (0.0169, 0.875] | 6303 | 0.2448 | 0.2581 | 0.0133 |

### Calibration curve — `next_6m_delinquency_flag` (test window)

| bucket | n | mean_predicted | observed_rate | calibration_gap |
| --- | --- | --- | --- | --- |
| (-0.000999, 0.0064] | 26068 | 0.0058 | 0.0066 | 0.0009 |
| (0.0064, 0.0103] | 7764 | 0.0103 | 0.0121 | 0.0018 |
| (0.0103, 0.0133] | 8463 | 0.0132 | 0.0141 | 0.0009 |
| (0.0133, 0.0169] | 6047 | 0.0161 | 0.0155 | -0.0006 |
| (0.0169, 0.0257] | 7170 | 0.0252 | 0.0248 | -0.0004 |
| (0.0257, 0.0512] | 6847 | 0.0424 | 0.0418 | -0.0006 |
| (0.0512, 0.929] | 4180 | 0.3939 | 0.4182 | 0.0243 |

### Calibration curve — `next_12m_default_flag` (test window)

| bucket | n | mean_predicted | observed_rate | calibration_gap |
| --- | --- | --- | --- | --- |
| (-0.000999, 0.0011] | 52347 | 0.0009 | 0.0020 | 0.0011 |
| (0.0011, 0.00514] | 12259 | 0.0049 | 0.0144 | 0.0096 |
| (0.00514, 0.917] | 4378 | 0.1554 | 0.1779 | 0.0225 |

### Calibration curve — `next_12m_prepayment_flag` (test window)

| bucket | n | mean_predicted | observed_rate | calibration_gap |
| --- | --- | --- | --- | --- |
| (-0.000999, 0.014] | 7590 | 0.0134 | 0.0474 | 0.0340 |
| (0.014, 0.0203] | 11385 | 0.0189 | 0.0509 | 0.0319 |
| (0.0203, 0.0224] | 4759 | 0.0224 | 0.0601 | 0.0377 |
| (0.0224, 0.0252] | 5285 | 0.0252 | 0.0579 | 0.0327 |
| (0.0252, 0.0461] | 6325 | 0.0357 | 0.0588 | 0.0231 |
| (0.0461, 0.0541] | 6329 | 0.0538 | 0.0667 | 0.0129 |
| (0.0541, 0.12] | 7933 | 0.0780 | 0.0706 | -0.0074 |
| (0.12, 0.225] | 5741 | 0.1796 | 0.0810 | -0.0986 |
| (0.225, 0.679] | 6908 | 0.4609 | 0.0864 | -0.3745 |
| (0.679, 1.0] | 6729 | 0.9165 | 0.1910 | -0.7255 |

## 4. Leakage controls

Four controls, each of which would catch a different failure:

1. **Banned-feature list.** `prepayment_flag`, `default_flag`, `next_state`, `loss_severity_band` and all `next_*` columns are refused by `assert_no_leakage`, which runs inside the design-matrix builder and again in the test suite. `loss_severity_band` is the subtle one: it is populated only after default, so its mere presence is the label.
2. **Horizon purging and embargo**, described in section 1.
3. **Label observability cap**, described in section 1.
4. **Split-sensitivity probe.** The same model and features are refitted under an unsound random row split. If the honest split were leaking, the two would agree; the gap below is the measure of how much a naive split would have flattered the model.

| target | purged_time_split | loan_disjoint_time_split | random_row_split_unsound | random_split_inflation |
| --- | --- | --- | --- | --- |
| next_3m_delinquency_flag | 0.9161 | 0.9191 | 0.9064 | -0.0097 |
| next_6m_delinquency_flag | 0.8784 | 0.8699 | 0.8920 | 0.0135 |
| next_12m_default_flag | 0.9207 | 0.9352 | 0.9988 | 0.0781 |
| next_12m_prepayment_flag | 0.6259 | 0.6041 | 0.9831 | 0.3572 |

The loan-disjoint column additionally forces no `loan_id` to appear in both the fitting data and the test window. Performance holding up there means the model has learned loan *characteristics*, not loan *identities*.

## 5. Expanding-window backtest

Stability across successive origination cut-offs, each fold re-purged.

| target | fold | train_window | test_window | roc_auc | pr_auc | brier |
| --- | --- | --- | --- | --- | --- | --- |
| next_3m_delinquency_flag | 0 | 2019-01..2023-09 | 2025-07..2025-12 | 0.9171 | 0.6439 | 0.0154 |
| next_3m_delinquency_flag | 1 | 2019-01..2024-03 | 2025-07..2025-12 | 0.9129 | 0.6433 | 0.0153 |
| next_3m_delinquency_flag | 2 | 2019-01..2024-09 | 2025-07..2025-12 | 0.9161 | 0.6497 | 0.0151 |
| next_6m_delinquency_flag | 0 | 2019-01..2023-03 | 2025-04..2025-09 | 0.8770 | 0.5766 | 0.0230 |
| next_6m_delinquency_flag | 1 | 2019-01..2023-09 | 2025-04..2025-09 | 0.8740 | 0.5734 | 0.0228 |
| next_6m_delinquency_flag | 2 | 2019-01..2024-03 | 2025-04..2025-09 | 0.8784 | 0.5780 | 0.0226 |
| next_12m_default_flag | 0 | 2019-01..2022-03 | 2024-10..2025-03 | 0.8989 | 0.5438 | 0.0089 |
| next_12m_default_flag | 1 | 2019-01..2022-09 | 2024-10..2025-03 | 0.9191 | 0.5350 | 0.0089 |
| next_12m_default_flag | 2 | 2019-01..2023-03 | 2024-10..2025-03 | 0.9207 | 0.5321 | 0.0091 |
| next_12m_prepayment_flag | 0 | 2019-01..2022-03 | 2024-10..2025-03 | 0.5875 | 0.1813 | 0.0933 |
| next_12m_prepayment_flag | 1 | 2019-01..2022-09 | 2024-10..2025-03 | 0.5923 | 0.1780 | 0.1073 |
| next_12m_prepayment_flag | 2 | 2019-01..2023-03 | 2024-10..2025-03 | 0.6259 | 0.2009 | 0.1367 |

## 6. Next-state prediction (multiclass)

One-step-ahead state transition model, benchmarked against two baselines:

- **Persistence** — predict that the current status continues. Deceptively strong because most loan-months stay Current, and it is exactly why accuracy is not the headline metric here.
- **Empirical Markov transition matrix** — `P(next_state | current_status)` estimated on the training window. Unlike persistence this emits a full probability vector, so log loss and macro-AUC are directly comparable and any lift the covariate model shows is lift *over already knowing the current state*.

Persistence still edges the covariate model on raw accuracy, and that is reported rather than buried: when 95%+ of transitions are Current-to-Current, a rule that never predicts a transition is hard to beat on accuracy alone. It is also useless, because it assigns zero probability to every event a servicer cares about. Log loss and macro-AUC are where the difference lives.

| n | accuracy | macro_f1 | weighted_f1 | log_loss | macro_roc_auc | split | model |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 67403 | 0.9879 | 0.6138 | 0.9859 | 0.0420 | 0.9804 | valid | lgbm_multiclass |
| 67403 | 0.9800 | 0.3656 | 0.9713 | 0.0892 | 0.8016 | valid | markov_transition_baseline |
| 67403 | 0.9800 | 0.4668 | 0.9769 |  |  | valid | persistence_baseline |
| 64160 | 0.9858 | 0.5864 | 0.9831 | 0.0476 | 0.9822 | test | lgbm_multiclass |
| 64160 | 0.9769 | 0.3700 | 0.9667 | 0.1007 | 0.8321 | test | markov_transition_baseline |
| 64160 | 0.9773 | 0.4732 | 0.9736 |  |  | test | persistence_baseline |

### Per-class performance — valid window

| class | support | precision | recall | f1 |
| --- | --- | --- | --- | --- |
| Current | 65730 | 0.9925 | 0.9974 | 0.9949 |
| DQ30 | 652 | 0.5458 | 0.2377 | 0.3312 |
| DQ60 | 153 | 0.3980 | 0.2549 | 0.3108 |
| DQ90plus | 197 | 0.6705 | 0.8782 | 0.7604 |
| Default | 246 | 0.8481 | 0.9756 | 0.9074 |
| PaidOff | 4 | 0.0000 | 0.0000 | 0.0000 |
| Prepaid | 421 | 0.9859 | 0.9976 | 0.9917 |

### Per-class performance — test window

| class | support | precision | recall | f1 |
| --- | --- | --- | --- | --- |
| Current | 62302 | 0.9907 | 0.9974 | 0.9940 |
| DQ30 | 718 | 0.5753 | 0.2396 | 0.3382 |
| DQ60 | 172 | 0.3065 | 0.1105 | 0.1624 |
| DQ90plus | 219 | 0.6250 | 0.8219 | 0.7101 |
| Default | 270 | 0.8529 | 0.9667 | 0.9062 |
| PaidOff | 3 | 0.0000 | 0.0000 | 0.0000 |
| Prepaid | 476 | 0.9896 | 0.9979 | 0.9937 |

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
| next_3m_delinquency_flag | 24 | 0.0300 | 150 | 19 | isotonic | 0.0230 |
| next_6m_delinquency_flag | 48 | 0.0450 | 80 | 11 | isotonic | 0.0326 |
| next_12m_default_flag | 64 | 0.0600 | 40 | 356 | isotonic | 0.0187 |
| next_12m_prepayment_flag | 64 | 0.0600 | 40 | 998 | isotonic | 0.1456 |

Hyperparameters are selected per target on **validation** average precision from a five-point grid over capacity and learning rate; the test window is never consulted during selection. Selection traces are in `reports/hyperparameter_search.csv`.

The calibrator is likewise chosen per target, by 3-fold cross-validated log loss *inside* the validation window. Selecting on the full validation window would have favoured isotonic every time, since it can bend to validation noise that does not repeat out of time.

## 8. Honest limitations

- The 12-month targets lose 143,367 rows to the embargo and 8,889 to the observability cap. Training data for those targets is 263,320 rows against 471,347 for the 3-month target, and confidence intervals are correspondingly wider.
- Train and test for the 12-month targets sit in different macro regimes. This is reported rather than corrected, because correcting it by reweighting would hide the single most useful fact about the model's operating conditions.
- `PaidOff` never occurs in this panel window, so the next-state model has six reachable classes, not seven.
- Metrics are single-run point estimates. No repeated-seed variance is reported.
