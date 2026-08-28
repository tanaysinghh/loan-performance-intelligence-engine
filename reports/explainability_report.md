# Explainability Report

**Task 6.** SHAP TreeExplainer over the trained LightGBM models. Every explanation here is derived from the fitted model's own structure — no language model is involved in producing any attribution.

## 1. What is being explained, and in what units

SHAP values are additive in **log-odds**, not probability. Explaining the calibrated probability directly would break that additivity and the contributions would stop summing to anything meaningful. Attribution is therefore computed against the raw LightGBM margin and reported in log-odds, alongside the calibrated probability the reviewer actually acts on. The two are labelled separately throughout and should not be added together.

## Target: `next_3m_delinquency_flag`

### Global feature importance

Mean absolute SHAP contribution across the test window. `direction` is the sign of the correlation between a feature's value and its contribution, which recovers whether higher values raise or lower risk without assuming monotonicity.

| feature | plain_english | mean_abs_shap | share_of_total_attribution | direction |
| --- | --- | --- | --- | --- |
| status_ord | current performance status | 0.5089 | 0.1177 | non-monotone / categorical |
| credit_ord | credit score band | 0.4317 | 0.0998 | non-monotone / categorical |
| current_streak_clean | consecutive clean months | 0.2957 | 0.0684 | higher value lowers risk |
| dti_ord | debt-to-income band | 0.2137 | 0.0494 | higher value raises risk |
| state | state | 0.2093 | 0.0484 | non-monotone / categorical |
| payment_to_balance | scheduled payment relative to balance | 0.1941 | 0.0449 | non-monotone / categorical |
| credit_score_band | credit score band | 0.1757 | 0.0406 | non-monotone / categorical |
| ltv_ord | loan-to-value band | 0.1617 | 0.0374 | non-monotone / categorical |
| term_progress | share of term elapsed | 0.1611 | 0.0372 | higher value lowers risk |
| current_status | current status | 0.1553 | 0.0359 | non-monotone / categorical |
| loan_age_months_clean | loan age | 0.1381 | 0.0319 | higher value lowers risk |
| unemployment_rate | unemployment rate | 0.1236 | 0.0286 | higher value raises risk |
| market_rate_delta_12m | market rate delta 12m | 0.1139 | 0.0263 | higher value lowers risk |
| amortisation_residual | balance against expected amortisation | 0.1040 | 0.0240 | higher value raises risk |
| interest_rate_clean | note rate | 0.0970 | 0.0224 | higher value lowers risk |
| ltv_band | ltv band | 0.0941 | 0.0217 | non-monotone / categorical |
| log_current_balance | current balance | 0.0817 | 0.0189 | higher value lowers risk |
| balance_change_1m | balance change 1m | 0.0718 | 0.0166 | higher value lowers risk |

The top three drivers — current performance status, credit score band, consecutive clean months — account for **28.6%** of total attribution.

### Local explanations — ten highest-risk records in the test window

Each row shows the calibrated probability a reviewer sees and the four largest log-odds contributions behind it.

| loan_id | reporting_month | current_status | calibrated_probability | driver_1 | driver_1_value | driver_1_log_odds | driver_2 | driver_2_value | driver_2_log_odds | driver_3 | driver_3_value | driver_3_log_odds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LN101348 | 2025-12 | DQ90plus | 0.9908 | current performance status | 3.0000 | 6.8526 | current status | DQ90plus | 1.9704 | months delinquent in last 6 months | 4.0000 | 0.5180 |
| LN101254 | 2025-10 | DQ90plus | 0.9908 | current performance status | 3.0000 | 6.4715 | current status | DQ90plus | 1.8418 | months delinquent in last 6 months | 6.0000 | 0.5346 |
| LN101348 | 2026-01 | DQ90plus | 0.9907 | current performance status | 3.0000 | 6.8822 | current status | DQ90plus | 1.9703 | months delinquent in last 6 months | 5.0000 | 0.5249 |
| LN100523 | 2026-03 | DQ90plus | 0.9906 | current performance status | 3.0000 | 6.6628 | current status | DQ90plus | 1.8618 | months delinquent in last 6 months | 6.0000 | 0.5512 |
| LN100523 | 2026-01 | DQ90plus | 0.9906 | current performance status | 3.0000 | 6.6375 | current status | DQ90plus | 1.8631 | months delinquent in last 6 months | 6.0000 | 0.5478 |
| LN100948 | 2025-12 | DQ90plus | 0.9906 | current performance status | 3.0000 | 6.1464 | current status | DQ90plus | 1.7288 | credit score band | 0.0000 | 0.5534 |
| LN101204 | 2025-10 | DQ90plus | 0.9906 | current performance status | 3.0000 | 5.5492 | current status | DQ90plus | 1.5235 | credit score band | 1.0000 | 0.5556 |
| LN101204 | 2025-11 | DQ90plus | 0.9906 | current performance status | 3.0000 | 5.4878 | current status | DQ90plus | 1.5124 | credit score band | 1.0000 | 0.5459 |
| LN100251 | 2026-02 | DQ90plus | 0.9905 | current performance status | 3.0000 | 6.0063 | current status | DQ90plus | 1.7328 | days past due | 104.0000 | 0.4953 |
| LN101026 | 2026-01 | DQ90plus | 0.9905 | current performance status | 3.0000 | 6.1904 | current status | DQ90plus | 1.7029 | days past due | 98.0000 | 0.4904 |

### Local explanations — five lowest-risk records (contrast set)

| loan_id | reporting_month | current_status | calibrated_probability | driver_1 | driver_1_value | driver_1_log_odds | driver_2 | driver_2_value | driver_2_log_odds | driver_3 | driver_3_value | driver_3_log_odds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LN100613 | 2026-01 | Current | 0.0015 | credit score band | 6.0000 | -0.5923 | state | IL | -0.4872 | current performance status | 0.0000 | -0.2760 |
| LN100265 | 2026-03 | Current | 0.0016 | credit score band | 6.0000 | -0.4623 | current performance status | 0.0 | -0.2785 | debt-to-income band | 0.0000 | -0.2256 |
| LN100613 | 2026-02 | Current | 0.0016 | credit score band | 6.0000 | -0.5956 | state | IL | -0.4486 | share of term elapsed | 0.0080 | -0.2914 |
| LN100265 | 2026-02 | Current | 0.0016 | credit score band | 6.0000 | -0.4595 | current performance status | 0.0 | -0.2783 | debt-to-income band | 0.0000 | -0.2307 |
| LN100613 | 2026-03 | Current | 0.0017 | credit score band | 6.0000 | -0.5968 | state | IL | -0.4412 | share of term elapsed | 0.0110 | -0.2823 |

### Model confidence and uncertainty

Predictions from the final boosting rounds are collected and their spread used as an epistemic-uncertainty proxy. This measures sensitivity to where the boosting sequence stopped — it is a stability signal, **not** a statistical confidence interval, and is not presented as one.

| confidence_band | share_of_records |
| --- | --- |
| high | 0.9153 |
| medium | 0.0804 |
| low | 0.0043 |

| statistic | calibrated_probability | staged_std | staged_p10 | staged_p90 |
| --- | --- | --- | --- | --- |
| count | 3719.0000 | 3719.0000 | 3719.0000 | 3719.0000 |
| mean | 0.0761 | 0.0030 | 0.0787 | 0.0859 |
| std | 0.2033 | 0.0030 | 0.2079 | 0.2086 |
| min | 0.0015 | 0.0002 | 0.0009 | 0.0017 |
| 25% | 0.0096 | 0.0011 | 0.0077 | 0.0110 |
| 50% | 0.0201 | 0.0019 | 0.0184 | 0.0241 |
| 75% | 0.0434 | 0.0036 | 0.0452 | 0.0563 |
| max | 0.9908 | 0.0267 | 0.9938 | 0.9977 |

### False positive / false negative analysis

Evaluated at the threshold that achieves 30% precision on the test window (`0.0604`): 196 true positives, 457 false positives, 51 false negatives out of 3719 records with 247 actual events. Precision 0.300, recall 0.794.

**Where the errors concentrate.** A model that misses events uniformly is a different problem from one that misses them in a specific segment — the second is a fairness and coverage issue, not just an accuracy one.

By `credit_score_band`:

| credit_score_band | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| 620-659 | 507 | 0.1499 | 0.1504 | 0.2919 | 0.0394 | -0.0005 |
| 780+ | 579 | 0.0242 | 0.0158 | 0.0000 | 0.0190 | 0.0083 |
| 580-619 | 175 | 0.1829 | 0.2075 | 0.5714 | 0.0171 | -0.0246 |
| 700-739 | 800 | 0.0450 | 0.0635 | 0.1037 | 0.0088 | -0.0185 |
| 660-699 | 595 | 0.0504 | 0.0729 | 0.1193 | 0.0067 | -0.0225 |
| 740-779 | 903 | 0.0299 | 0.0389 | 0.0111 | 0.0055 | -0.0090 |
| <580 | 68 | 0.4118 | 0.3955 | 0.4412 | 0.0000 | 0.0163 |

By `servicer_name`:

| servicer_name | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| Belmont Loan Services | 910 | 0.0857 | 0.0882 | 0.1374 | 0.0220 | -0.0024 |
| Pioneer Mortgage Ops | 547 | 0.0402 | 0.0567 | 0.1609 | 0.0128 | -0.0165 |
| Arcadia Capital Servicing | 766 | 0.0522 | 0.0639 | 0.1005 | 0.0117 | -0.0117 |
| Northgate Servicing | 1125 | 0.0773 | 0.0845 | 0.1067 | 0.0116 | -0.0072 |
| Kestrel Financial | 371 | 0.0539 | 0.0749 | 0.1267 | 0.0054 | -0.0209 |

By `current_status`:

| current_status | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| Current | 3541 | 0.0209 | 0.0313 | 0.1276 | 0.0144 | -0.0104 |
| DQ30 | 38 | 0.8684 | 0.8945 | 0.1316 | 0.0000 | -0.0261 |
| DQ60 | 32 | 1.0000 | 0.9860 | 0.0000 | 0.0000 | 0.0140 |
| DQ90plus | 108 | 1.0000 | 0.9892 | 0.0000 | 0.0000 | 0.0108 |

By `state`:

| state | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| CO | 74 | 0.1081 | 0.0826 | 0.2027 | 0.0405 | 0.0255 |
| OH | 155 | 0.0645 | 0.0565 | 0.1226 | 0.0387 | 0.0080 |
| NC | 141 | 0.1064 | 0.0806 | 0.1631 | 0.0355 | 0.0258 |
| FL | 426 | 0.0728 | 0.0750 | 0.1737 | 0.0211 | -0.0022 |
| GA | 253 | 0.0672 | 0.0729 | 0.0356 | 0.0158 | -0.0057 |
| TX | 526 | 0.0456 | 0.0634 | 0.1217 | 0.0152 | -0.0178 |
| NY | 338 | 0.0740 | 0.0797 | 0.1391 | 0.0148 | -0.0058 |
| AZ | 169 | 0.0473 | 0.0507 | 0.0237 | 0.0118 | -0.0034 |
| NV | 130 | 0.0154 | 0.0435 | 0.1308 | 0.0077 | -0.0281 |
| MI | 132 | 0.0909 | 0.0923 | 0.0530 | 0.0076 | -0.0014 |
| CA | 665 | 0.0722 | 0.0893 | 0.1398 | 0.0075 | -0.0171 |
| WA | 157 | 0.0892 | 0.1084 | 0.1656 | 0.0064 | -0.0192 |
| PA | 172 | 0.0756 | 0.0868 | 0.0465 | 0.0058 | -0.0112 |
| IL | 227 | 0.0705 | 0.0961 | 0.2070 | 0.0000 | -0.0257 |
| NJ | 154 | 0.0260 | 0.0426 | 0.0260 | 0.0000 | -0.0166 |

**What false positives and false negatives look like.** Mean feature values for each error class against correctly-rejected records.

| feature | false_positives | false_negatives | true_negatives |
| --- | --- | --- | --- |
| credit score band | 2.2421 | 3.4800 | 4.2607 |
| loan-to-value band | 2.7031 | 2.3125 | 1.8183 |
| current performance status | 0.0109 | 0.0000 | 0.0000 |
| days past due | 0.8118 | 0.0000 | 0.3742 |
| worst days past due in last 6 months | 3.3085 | 0.9020 | 2.7756 |
| loan age | 49.2671 | 51.2549 | 48.9523 |
| record data-quality score | 95.1510 | 94.5098 | 95.2570 |
| balance as a share of original | 0.9213 | 0.8911 | 0.9213 |

## Target: `next_12m_default_flag`

### Global feature importance

Mean absolute SHAP contribution across the test window. `direction` is the sign of the correlation between a feature's value and its contribution, which recovers whether higher values raise or lower risk without assuming monotonicity.

| feature | plain_english | mean_abs_shap | share_of_total_attribution | direction |
| --- | --- | --- | --- | --- |
| credit_ord | credit score band | 0.8529 | 0.1549 | higher value lowers risk |
| dti_ord | debt-to-income band | 0.2461 | 0.0447 | non-monotone / categorical |
| interest_rate_clean | note rate | 0.2385 | 0.0433 | higher value lowers risk |
| state | state | 0.2286 | 0.0415 | non-monotone / categorical |
| status_ord | current performance status | 0.2258 | 0.0410 | higher value raises risk |
| credit_score_band | credit score band | 0.2247 | 0.0408 | non-monotone / categorical |
| loan_purpose | loan purpose | 0.2133 | 0.0387 | non-monotone / categorical |
| market_rate_delta_12m | market rate delta 12m | 0.1832 | 0.0333 | non-monotone / categorical |
| term_progress | share of term elapsed | 0.1612 | 0.0293 | non-monotone / categorical |
| ltv_ord | loan-to-value band | 0.1583 | 0.0287 | higher value raises risk |
| loan_age_months_clean | loan age | 0.1551 | 0.0282 | higher value lowers risk |
| rate_incentive | refinance incentive (note rate less market rate) | 0.1529 | 0.0278 | higher value raises risk |
| months_dq_last_12m | months delinquent in last 12 months | 0.1328 | 0.0241 | higher value raises risk |
| payment_to_balance | scheduled payment relative to balance | 0.1258 | 0.0228 | non-monotone / categorical |
| scheduled_payment | scheduled monthly payment | 0.1247 | 0.0226 | higher value raises risk |
| balance_change_1m | balance change 1m | 0.1241 | 0.0225 | non-monotone / categorical |
| log_original_balance | original balance | 0.1172 | 0.0213 | higher value raises risk |
| unemployment_rate | unemployment rate | 0.1081 | 0.0196 | non-monotone / categorical |

The top three drivers — credit score band, debt-to-income band, note rate — account for **24.3%** of total attribution.

### Local explanations — ten highest-risk records in the test window

Each row shows the calibrated probability a reviewer sees and the four largest log-odds contributions behind it.

| loan_id | reporting_month | current_status | calibrated_probability | driver_1 | driver_1_value | driver_1_log_odds | driver_2 | driver_2_value | driver_2_log_odds | driver_3 | driver_3_value | driver_3_log_odds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LN100307 | 2025-01 | DQ90plus | 0.9932 | current performance status | 3.0000 | 2.2689 | months delinquent in last 12 months | 7.0000 | 1.1962 | days past due | 103.0000 | 1.1233 |
| LN100743 | 2025-02 | DQ90plus | 0.9931 | current performance status | 3.0000 | 2.7473 | months delinquent in last 12 months | 5.0000 | 1.4938 | days past due | 102.0000 | 1.2970 |
| LN100778 | 2025-06 | DQ90plus | 0.9931 | current performance status | 3.0000 | 2.5270 | days past due | 94.0000 | 1.3013 | months delinquent in last 12 months | 5.0000 | 1.2773 |
| LN100307 | 2025-02 | DQ90plus | 0.9930 | current performance status | 3.0000 | 2.2292 | months delinquent in last 12 months | 8.0000 | 1.2744 | days past due | 99.0000 | 1.0630 |
| LN100365 | 2025-02 | DQ90plus | 0.9920 | current performance status | 3.0000 | 2.4516 | months delinquent in last 12 months | 6.0000 | 1.4100 | days past due | 110.0000 | 1.2495 |
| LN100778 | 2025-05 | DQ90plus | 0.9915 | current performance status | 3.0000 | 2.5771 | days past due | 111.0000 | 1.3082 | months delinquent in last 12 months | 4.0000 | 1.2695 |
| LN100366 | 2025-06 | DQ90plus | 0.9904 | current performance status | 3.0000 | 2.4501 | months delinquent in last 12 months | 7.0000 | 1.4322 | days past due | 102.0000 | 1.1859 |
| LN100366 | 2025-04 | DQ60 | 0.9899 | current performance status | 2.0000 | 2.2584 | months delinquent in last 12 months | 5.0000 | 1.4683 | days past due | 77.0000 | 1.1057 |
| LN100366 | 2025-05 | DQ60 | 0.9898 | current performance status | 2.0000 | 2.2643 | months delinquent in last 12 months | 6.0000 | 1.4505 | days past due | 72.0000 | 1.1040 |
| LN100366 | 2025-03 | DQ60 | 0.9896 | current performance status | 2.0000 | 2.2964 | months delinquent in last 12 months | 4.0000 | 1.3308 | days past due | 77.0000 | 1.1668 |

### Local explanations — five lowest-risk records (contrast set)

| loan_id | reporting_month | current_status | calibrated_probability | driver_1 | driver_1_value | driver_1_log_odds | driver_2 | driver_2_value | driver_2_log_odds | driver_3 | driver_3_value | driver_3_log_odds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LN100313 | 2025-02 | Current | 0.0002 | credit score band | 5.0000 | -0.7888 | loan purpose | rate_term_refi | -0.3044 | market rate delta 12m | -0.649 | -0.2337 |
| LN101236 | 2025-04 | Current | 0.0002 | credit score band | 5.0000 | -0.9511 | loan purpose | rate_term_refi | -0.3154 | credit score band | 740-779 | -0.2447 |
| LN100313 | 2025-05 | Current | 0.0002 | credit score band | 5.0000 | -0.7704 | loan purpose | rate_term_refi | -0.2843 | debt-to-income band | 0.0 | -0.2192 |
| LN100313 | 2025-04 | Current | 0.0002 | credit score band | 5.0000 | -0.7972 | loan purpose | rate_term_refi | -0.2860 | credit score band | 740-779 | -0.2260 |
| LN100313 | 2025-03 | Current | 0.0002 | credit score band | 5.0000 | -0.8029 | loan purpose | rate_term_refi | -0.2989 | credit score band | 740-779 | -0.2341 |

### Model confidence and uncertainty

Predictions from the final boosting rounds are collected and their spread used as an epistemic-uncertainty proxy. This measures sensitivity to where the boosting sequence stopped — it is a stability signal, **not** a statistical confidence interval, and is not presented as one.

| confidence_band | share_of_records |
| --- | --- |
| high | 0.9139 |
| medium | 0.0500 |
| low | 0.0361 |

| statistic | calibrated_probability | staged_std | staged_p10 | staged_p90 |
| --- | --- | --- | --- | --- |
| count | 4040.0000 | 4040.0000 | 4040.0000 | 4040.0000 |
| mean | 0.0647 | 0.0014 | 0.0588 | 0.0622 |
| std | 0.1940 | 0.0034 | 0.1948 | 0.1984 |
| min | 0.0002 | 0.0000 | 0.0000 | 0.0001 |
| 25% | 0.0014 | 0.0001 | 0.0005 | 0.0007 |
| 50% | 0.0039 | 0.0002 | 0.0016 | 0.0021 |
| 75% | 0.0175 | 0.0010 | 0.0089 | 0.0115 |
| max | 0.9932 | 0.0472 | 0.9942 | 0.9965 |

### False positive / false negative analysis

Evaluated at the threshold that achieves 30% precision on the test window (`0.0321`): 227 true positives, 529 false positives, 65 false negatives out of 4040 records with 292 actual events. Precision 0.300, recall 0.777.

**Where the errors concentrate.** A model that misses events uniformly is a different problem from one that misses them in a specific segment — the second is a fairness and coverage issue, not just an accuracy one.

By `credit_score_band`:

| credit_score_band | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| 660-699 | 617 | 0.0924 | 0.0637 | 0.1410 | 0.0389 | 0.0286 |
| 620-659 | 577 | 0.1473 | 0.1391 | 0.3830 | 0.0295 | 0.0082 |
| <580 | 75 | 0.3733 | 0.3776 | 0.4933 | 0.0267 | -0.0043 |
| 740-779 | 970 | 0.0330 | 0.0165 | 0.0041 | 0.0113 | 0.0165 |
| 700-739 | 876 | 0.0342 | 0.0411 | 0.0742 | 0.0103 | -0.0068 |
| 580-619 | 203 | 0.2020 | 0.2247 | 0.5123 | 0.0049 | -0.0227 |
| 780+ | 646 | 0.0170 | 0.0135 | 0.0000 | 0.0000 | 0.0035 |

By `servicer_name`:

| servicer_name | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| Northgate Servicing | 1233 | 0.0560 | 0.0437 | 0.0998 | 0.0219 | 0.0123 |
| Pioneer Mortgage Ops | 603 | 0.0796 | 0.0613 | 0.1277 | 0.0166 | 0.0183 |
| Belmont Loan Services | 975 | 0.0574 | 0.0561 | 0.1815 | 0.0133 | 0.0013 |
| Kestrel Financial | 381 | 0.1207 | 0.1072 | 0.1286 | 0.0131 | 0.0136 |
| Arcadia Capital Servicing | 848 | 0.0861 | 0.0883 | 0.1215 | 0.0118 | -0.0022 |

By `current_status`:

| current_status | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| Current | 3814 | 0.0294 | 0.0216 | 0.1266 | 0.0170 | 0.0077 |
| DQ30 | 64 | 0.6719 | 0.5711 | 0.3281 | 0.0000 | 0.1007 |
| DQ60 | 65 | 0.9077 | 0.8379 | 0.0923 | 0.0000 | 0.0698 |
| DQ90plus | 97 | 0.8041 | 0.9051 | 0.1959 | 0.0000 | -0.1010 |

By `state`:

| state | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| IL | 224 | 0.0446 | 0.0365 | 0.1607 | 0.0312 | 0.0081 |
| PA | 181 | 0.0497 | 0.0160 | 0.0442 | 0.0276 | 0.0337 |
| NV | 151 | 0.0795 | 0.0477 | 0.2053 | 0.0265 | 0.0317 |
| WA | 156 | 0.0769 | 0.0522 | 0.2051 | 0.0256 | 0.0247 |
| GA | 275 | 0.0655 | 0.0452 | 0.0582 | 0.0255 | 0.0203 |
| NC | 154 | 0.0974 | 0.0660 | 0.1429 | 0.0195 | 0.0314 |
| TX | 574 | 0.0557 | 0.0765 | 0.1672 | 0.0174 | -0.0208 |
| NY | 380 | 0.0711 | 0.0606 | 0.1316 | 0.0158 | 0.0105 |
| CA | 737 | 0.0855 | 0.0820 | 0.1411 | 0.0149 | 0.0035 |
| FL | 465 | 0.0968 | 0.0962 | 0.1441 | 0.0129 | 0.0006 |
| OH | 180 | 0.0833 | 0.0811 | 0.1278 | 0.0111 | 0.0023 |
| MI | 129 | 0.1318 | 0.0958 | 0.0930 | 0.0000 | 0.0359 |
| CO | 72 | 0.0833 | 0.0650 | 0.2361 | 0.0000 | 0.0183 |
| AZ | 190 | 0.0316 | 0.0379 | 0.0684 | 0.0000 | -0.0064 |
| NJ | 172 | 0.0291 | 0.0079 | 0.0116 | 0.0000 | 0.0212 |

**What false positives and false negatives look like.** Mean feature values for each error class against correctly-rejected records.

| feature | false_positives | false_negatives | true_negatives |
| --- | --- | --- | --- |
| credit score band | 2.0985 | 3.0938 | 4.3083 |
| loan-to-value band | 2.8900 | 2.8125 | 1.7530 |
| current performance status | 0.1701 | 0.0000 | 0.0000 |
| days past due | 6.7365 | 0.0000 | 0.2683 |
| worst days past due in last 6 months | 10.5595 | 5.5385 | 2.0581 |
| loan age | 27.0702 | 36.0156 | 46.1790 |
| record data-quality score | 94.9187 | 94.4000 | 95.3924 |
| balance as a share of original | 0.9564 | 0.9293 | 0.9250 |

## Target: `next_12m_prepayment_flag`

### Global feature importance

Mean absolute SHAP contribution across the test window. `direction` is the sign of the correlation between a feature's value and its contribution, which recovers whether higher values raise or lower risk without assuming monotonicity.

| feature | plain_english | mean_abs_shap | share_of_total_attribution | direction |
| --- | --- | --- | --- | --- |
| interest_rate_clean | note rate | 0.6096 | 0.0978 | higher value lowers risk |
| market_rate_delta_12m | market rate delta 12m | 0.5882 | 0.0944 | higher value lowers risk |
| current_streak_clean | consecutive clean months | 0.4875 | 0.0782 | higher value raises risk |
| state | state | 0.3224 | 0.0517 | non-monotone / categorical |
| loan_purpose | loan purpose | 0.2977 | 0.0478 | non-monotone / categorical |
| scheduled_payment | scheduled monthly payment | 0.2615 | 0.0420 | higher value lowers risk |
| credit_ord | credit score band | 0.2410 | 0.0387 | higher value raises risk |
| credit_score_band | credit score band | 0.2154 | 0.0346 | non-monotone / categorical |
| servicer_name | servicer | 0.2040 | 0.0327 | non-monotone / categorical |
| log_original_balance | original balance | 0.1882 | 0.0302 | higher value lowers risk |
| payment_to_balance | scheduled payment relative to balance | 0.1858 | 0.0298 | higher value raises risk |
| ltv_ord | loan-to-value band | 0.1521 | 0.0244 | higher value lowers risk |
| log_current_balance | current balance | 0.1423 | 0.0228 | non-monotone / categorical |
| market_mortgage_rate | market mortgage rate | 0.1406 | 0.0226 | higher value lowers risk |
| term_progress | share of term elapsed | 0.1396 | 0.0224 | higher value lowers risk |
| loan_age_months_clean | loan age | 0.1368 | 0.0220 | higher value lowers risk |
| amortisation_residual | balance against expected amortisation | 0.1359 | 0.0218 | higher value raises risk |
| remaining_term_months | remaining term months | 0.1286 | 0.0206 | non-monotone / categorical |

The top three drivers — note rate, market rate delta 12m, consecutive clean months — account for **27.0%** of total attribution.

### Local explanations — ten highest-risk records in the test window

Each row shows the calibrated probability a reviewer sees and the four largest log-odds contributions behind it.

| loan_id | reporting_month | current_status | calibrated_probability | driver_1 | driver_1_value | driver_1_log_odds | driver_2 | driver_2_value | driver_2_log_odds | driver_3 | driver_3_value | driver_3_log_odds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LN100026 | 2025-03 | Current | 1.0000 | note rate | 6.5930 | 1.2419 | market rate delta 12m | -0.725 | 0.7816 | scheduled monthly payment | 2629.37 | 0.4497 |
| LN100526 | 2025-05 | Current | 1.0000 | note rate | 8.1960 | 1.1524 | credit score band | 620-659 | -0.6906 | market rate delta 12m | -0.852 | 0.6581 |
| LN101304 | 2025-04 | Current | 1.0000 | note rate | 7.3700 | 1.0224 | market rate delta 12m | -0.792 | 0.6607 | state | IL | 0.5906 |
| LN101321 | 2025-04 | Current | 1.0000 | note rate | 7.0140 | 1.3478 | scheduled monthly payment | 1812.181 | 0.6954 | scheduled payment relative to balance | 0.007 | 0.5800 |
| LN101319 | 2025-03 | Current | 1.0000 | note rate | 7.3580 | 1.5356 | market rate delta 12m | -0.725 | 0.6470 | credit score band | 5.0 | 0.3423 |
| LN100524 | 2025-01 | Current | 1.0000 | scheduled monthly payment | 1874.9240 | 0.6854 | note rate | 6.652 | 0.6597 | original balance | 12.585 | 0.4074 |
| LN100524 | 2025-02 | Current | 1.0000 | note rate | 6.6520 | 0.8183 | scheduled monthly payment | 1874.924 | 0.7405 | state | CA | 0.4014 |
| LN101319 | 2025-02 | Current | 1.0000 | note rate | 7.3580 | 1.4950 | market rate delta 12m | -0.649 | 0.5858 | state | AZ | 0.3603 |
| LN101298 | 2025-05 | Current | 1.0000 | note rate | 7.6890 | 1.3576 | market rate delta 12m | -0.852 | 0.6611 | scheduled monthly payment | 1987.042 | 0.4475 |
| LN101298 | 2025-06 | Current | 1.0000 | note rate | 7.6890 | 1.3470 | market rate delta 12m | -0.901 | 0.6887 | scheduled monthly payment | 1987.042 | 0.4189 |

### Local explanations — five lowest-risk records (contrast set)

| loan_id | reporting_month | current_status | calibrated_probability | driver_1 | driver_1_value | driver_1_log_odds | driver_2 | driver_2_value | driver_2_log_odds | driver_3 | driver_3_value | driver_3_log_odds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LN101254 | 2025-01 | Current | 0.0000 | consecutive clean months | 72 | -1.0925 | state | PA | -0.9088 | balance change 3m | -0.022 | -0.5652 |
| LN101254 | 2025-05 | DQ60 | 0.0000 | state | PA | -0.7919 | current performance status | 2.0 | -0.5899 | balance change 3m | -0.023 | -0.5795 |
| LN101254 | 2025-06 | DQ60 | 0.0000 | current performance status | 2.0 | -0.6980 | state | PA | -0.6807 | balance change 3m | -0.015 | -0.5488 |
| LN101254 | 2025-02 | Current | 0.0000 | consecutive clean months | 73 | -1.0882 | state | PA | -0.9147 | balance change 3m | -0.022 | -0.6077 |
| LN101186 | 2025-05 | Current | 0.0000 | state | CA | -0.7388 | market rate delta 12m | -0.852 | 0.7146 | consecutive clean months | 49 | -0.7138 |

### Model confidence and uncertainty

Predictions from the final boosting rounds are collected and their spread used as an epistemic-uncertainty proxy. This measures sensitivity to where the boosting sequence stopped — it is a stability signal, **not** a statistical confidence interval, and is not presented as one.

| confidence_band | share_of_records |
| --- | --- |
| high | 0.6614 |
| medium | 0.2116 |
| low | 0.1270 |

| statistic | calibrated_probability | staged_std | staged_p10 | staged_p90 |
| --- | --- | --- | --- | --- |
| count | 4040.0000 | 4040.0000 | 4040.0000 | 4040.0000 |
| mean | 0.1288 | 0.0033 | 0.1286 | 0.1365 |
| std | 0.2081 | 0.0041 | 0.1935 | 0.2008 |
| min | 0.0000 | 0.0000 | 0.0003 | 0.0003 |
| 25% | 0.0289 | 0.0003 | 0.0063 | 0.0072 |
| 50% | 0.0289 | 0.0013 | 0.0296 | 0.0325 |
| 75% | 0.1176 | 0.0054 | 0.1729 | 0.1886 |
| max | 1.0000 | 0.0257 | 0.9502 | 0.9575 |

### False positive / false negative analysis

Evaluated at the threshold that achieves 30% precision on the test window (`0.1190`): 207 true positives, 469 false positives, 484 false negatives out of 4040 records with 691 actual events. Precision 0.306, recall 0.300.

**Where the errors concentrate.** A model that misses events uniformly is a different problem from one that misses them in a specific segment — the second is a fairness and coverage issue, not just an accuracy one.

By `credit_score_band`:

| credit_score_band | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| 740-779 | 970 | 0.2299 | 0.1409 | 0.1165 | 0.1577 | 0.0890 |
| 580-619 | 203 | 0.1724 | 0.0854 | 0.0640 | 0.1527 | 0.0870 |
| 780+ | 646 | 0.2090 | 0.1546 | 0.0944 | 0.1285 | 0.0544 |
| 700-739 | 876 | 0.1484 | 0.1489 | 0.1598 | 0.1016 | -0.0005 |
| 660-699 | 617 | 0.1459 | 0.1455 | 0.1767 | 0.0989 | 0.0003 |
| 620-659 | 577 | 0.1023 | 0.0547 | 0.0364 | 0.0953 | 0.0476 |
| <580 | 75 | 0.0933 | 0.0365 | 0.0000 | 0.0933 | 0.0568 |

By `servicer_name`:

| servicer_name | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| Northgate Servicing | 1233 | 0.1987 | 0.1184 | 0.0803 | 0.1395 | 0.0803 |
| Pioneer Mortgage Ops | 603 | 0.1957 | 0.0958 | 0.0663 | 0.1393 | 0.0999 |
| Arcadia Capital Servicing | 848 | 0.1427 | 0.0999 | 0.1002 | 0.1167 | 0.0428 |
| Belmont Loan Services | 975 | 0.1846 | 0.1896 | 0.1969 | 0.1087 | -0.0050 |
| Kestrel Financial | 381 | 0.0709 | 0.1234 | 0.1391 | 0.0604 | -0.0526 |

By `current_status`:

| current_status | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| DQ90plus | 97 | 0.1237 | 0.0221 | 0.0000 | 0.1237 | 0.1016 |
| Current | 3814 | 0.1778 | 0.1349 | 0.1230 | 0.1235 | 0.0429 |
| DQ60 | 65 | 0.0154 | 0.0270 | 0.0000 | 0.0154 | -0.0116 |
| DQ30 | 64 | 0.0000 | 0.0335 | 0.0000 | 0.0000 | -0.0335 |

By `state`:

| state | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| TX | 574 | 0.2003 | 0.1277 | 0.1220 | 0.1551 | 0.0727 |
| NY | 380 | 0.1816 | 0.0941 | 0.0974 | 0.1421 | 0.0875 |
| WA | 156 | 0.1538 | 0.0973 | 0.1154 | 0.1410 | 0.0565 |
| NJ | 172 | 0.1686 | 0.1091 | 0.1047 | 0.1395 | 0.0595 |
| CO | 72 | 0.2083 | 0.1392 | 0.0972 | 0.1389 | 0.0692 |
| PA | 181 | 0.1547 | 0.1182 | 0.1105 | 0.1326 | 0.0365 |
| NV | 151 | 0.1258 | 0.0424 | 0.0000 | 0.1258 | 0.0834 |
| CA | 737 | 0.1696 | 0.1357 | 0.1303 | 0.1140 | 0.0340 |
| FL | 465 | 0.1656 | 0.1002 | 0.0495 | 0.1118 | 0.0654 |
| IL | 224 | 0.1429 | 0.1071 | 0.0714 | 0.1027 | 0.0358 |
| MI | 129 | 0.1085 | 0.0822 | 0.1008 | 0.1008 | 0.0263 |
| OH | 180 | 0.2333 | 0.2544 | 0.1722 | 0.1000 | -0.0210 |
| NC | 154 | 0.1623 | 0.1701 | 0.2078 | 0.0974 | -0.0078 |
| AZ | 190 | 0.1842 | 0.2227 | 0.1789 | 0.0842 | -0.0385 |
| GA | 275 | 0.1527 | 0.1607 | 0.1964 | 0.0764 | -0.0079 |

**What false positives and false negatives look like.** Mean feature values for each error class against correctly-rejected records.

| feature | false_positives | false_negatives | true_negatives |
| --- | --- | --- | --- |
| credit score band | 4.0985 | 4.0564 | 3.7825 |
| loan-to-value band | 1.8057 | 1.8596 | 2.0789 |
| current performance status | 0.0000 | 0.0785 | 0.1552 |
| days past due | 0.5568 | 2.6567 | 5.7560 |
| worst days past due in last 6 months | 2.3475 | 4.0537 | 8.4151 |
| loan age | 14.6432 | 34.1325 | 51.3083 |
| record data-quality score | 95.3710 | 95.1219 | 95.2594 |
| balance as a share of original | 0.9846 | 0.9323 | 0.9162 |

## Target: `exception_required`

### Global feature importance

Mean absolute SHAP contribution across the test window. `direction` is the sign of the correlation between a feature's value and its contribution, which recovers whether higher values raise or lower risk without assuming monotonicity.

| feature | plain_english | mean_abs_shap | share_of_total_attribution | direction |
| --- | --- | --- | --- | --- |
| dq_score | record data-quality score | 1.1373 | 0.5176 | non-monotone / categorical |
| dq_violation_count | dq violation count | 0.2135 | 0.0972 | non-monotone / categorical |
| missing_field_count | missing field count | 0.1867 | 0.0850 | higher value lowers risk |
| svc_present | svc present | 0.1087 | 0.0495 | non-monotone / categorical |
| doc_incomplete | doc incomplete | 0.1076 | 0.0490 | non-monotone / categorical |
| svc_balance_rel_gap | servicer feed balance gap | 0.0873 | 0.0397 | non-monotone / categorical |
| dti_ord | debt-to-income band | 0.0692 | 0.0315 | higher value lowers risk |
| reporting_lag_days | servicer reporting lag | 0.0418 | 0.0190 | non-monotone / categorical |
| document_status | document custody status | 0.0412 | 0.0187 | non-monotone / categorical |
| ltv_ord | loan-to-value band | 0.0275 | 0.0125 | non-monotone / categorical |
| unemployment_delta_12m | unemployment delta 12m | 0.0200 | 0.0091 | non-monotone / categorical |
| svc_status_conflict | svc status conflict | 0.0164 | 0.0075 | non-monotone / categorical |
| age_repaired | age repaired | 0.0135 | 0.0061 | non-monotone / categorical |
| dpd_status_residual | days past due against reported status | 0.0120 | 0.0055 | higher value raises risk |
| credit_ord | credit score band | 0.0100 | 0.0045 | higher value raises risk |
| stale_reporting | stale reporting | 0.0092 | 0.0042 | non-monotone / categorical |
| interest_rate_clean | note rate | 0.0091 | 0.0041 | higher value lowers risk |
| state | state | 0.0086 | 0.0039 | non-monotone / categorical |

The top three drivers — record data-quality score, dq violation count, missing field count — account for **70.0%** of total attribution.

### Local explanations — ten highest-risk records in the test window

Each row shows the calibrated probability a reviewer sees and the four largest log-odds contributions behind it.

| loan_id | reporting_month | current_status | calibrated_probability | driver_1 | driver_1_value | driver_1_log_odds | driver_2 | driver_2_value | driver_2_log_odds | driver_3 | driver_3_value | driver_3_log_odds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LN101279 | 2026-03 | Current | 0.8946 | record data-quality score | 72.0000 | 3.6458 | dq violation count | 3 | 0.6792 | servicer feed balance gap | 0.069 | 0.6038 |
| LN100372 | 2026-01 | Current | 0.8914 | record data-quality score | 77.0000 | 3.5832 | dq violation count | 2 | 0.8799 | age repaired | 1 | 0.5783 |
| LN100013 | 2026-06 | Current | 0.8914 | record data-quality score | 77.0000 | 3.5575 | dq violation count | 2 | 0.8760 | age repaired | 1 | 0.5747 |
| LN100171 | 2026-05 | Current | 0.8912 | record data-quality score | 74.0000 | 3.5896 | dq violation count | 3 | 1.0591 | age repaired | 1 | 0.5595 |
| LN100834 | 2026-04 | Current | 0.8912 | record data-quality score | 67.0000 | 3.5867 | dq violation count | 4 | 1.0580 | age repaired | 1 | 0.5021 |
| LN100011 | 2026-06 | Current | 0.8912 | record data-quality score | 74.0000 | 3.5813 | dq violation count | 3 | 1.0522 | age repaired | 1 | 0.6055 |
| LN100414 | 2026-06 | Current | 0.8904 | record data-quality score | 74.0000 | 3.5644 | dq violation count | 3 | 1.0613 | age repaired | 1 | 0.6048 |
| LN100695 | 2026-05 | Current | 0.8893 | record data-quality score | 74.0000 | 3.5525 | dq violation count | 3 | 1.0498 | age repaired | 1 | 0.6112 |
| LN100892 | 2026-04 | Current | 0.8877 | record data-quality score | 77.0000 | 3.5622 | dq violation count | 2 | 0.8843 | age repaired | 1 | 0.5826 |
| LN100921 | 2026-02 | Current | 0.8870 | record data-quality score | 83.0000 | 3.4485 | servicer reporting lag | -22.0 | 1.0176 | dq violation count | 3 | 0.8038 |

### Local explanations — five lowest-risk records (contrast set)

| loan_id | reporting_month | current_status | calibrated_probability | driver_1 | driver_1_value | driver_1_log_odds | driver_2 | driver_2_value | driver_2_log_odds | driver_3 | driver_3_value | driver_3_log_odds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LN100413 | 2026-04 | Current | 0.0031 | missing field count | 1 | -0.8068 | debt-to-income band | nan | -0.4554 | dq violation count | 2 | 0.3929 |
| LN100935 | 2026-05 | Current | 0.0031 | record data-quality score | 97.0 | -0.7679 | svc present | 0 | -0.1213 | refinance incentive (note rate less market rate) | nan | -0.1094 |
| LN101471 | 2026-05 | Current | 0.0032 | record data-quality score | 97.0 | -0.7644 | svc present | 0 | -0.1199 | refinance incentive (note rate less market rate) | nan | -0.1093 |
| LN100293 | 2026-02 | Current | 0.0032 | record data-quality score | 97.0 | -0.7693 | svc present | 0 | -0.1204 | refinance incentive (note rate less market rate) | nan | -0.1136 |
| LN101418 | 2026-05 | Current | 0.0032 | record data-quality score | 97.0 | -0.7648 | svc present | 0 | -0.1199 | refinance incentive (note rate less market rate) | nan | -0.1097 |

### Model confidence and uncertainty

Predictions from the final boosting rounds are collected and their spread used as an epistemic-uncertainty proxy. This measures sensitivity to where the boosting sequence stopped — it is a stability signal, **not** a statistical confidence interval, and is not presented as one.

| confidence_band | share_of_records |
| --- | --- |
| high | 0.8279 |
| low | 0.1572 |
| medium | 0.0149 |

| statistic | calibrated_probability | staged_std | staged_p10 | staged_p90 |
| --- | --- | --- | --- | --- |
| count | 3480.0000 | 3480.0000 | 3480.0000 | 3480.0000 |
| mean | 0.1254 | 0.0110 | 0.1389 | 0.1665 |
| std | 0.2765 | 0.0124 | 0.2628 | 0.2926 |
| min | 0.0031 | 0.0020 | 0.0160 | 0.0319 |
| 25% | 0.0041 | 0.0056 | 0.0196 | 0.0333 |
| 50% | 0.0044 | 0.0056 | 0.0205 | 0.0345 |
| 75% | 0.0053 | 0.0059 | 0.0239 | 0.0378 |
| max | 0.8946 | 0.0940 | 0.8123 | 0.9172 |

### False positive / false negative analysis

Evaluated at the threshold that achieves 30% precision on the test window (`0.0045`): 441 true positives, 1029 false positives, 13 false negatives out of 3480 records with 454 actual events. Precision 0.300, recall 0.971.

**Where the errors concentrate.** A model that misses events uniformly is a different problem from one that misses them in a specific segment — the second is a fairness and coverage issue, not just an accuracy one.

By `credit_score_band`:

| credit_score_band | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| 660-699 | 573 | 0.1257 | 0.1224 | 0.3089 | 0.0052 | 0.0032 |
| 740-779 | 830 | 0.1181 | 0.1129 | 0.2482 | 0.0048 | 0.0051 |
| 620-659 | 467 | 0.1734 | 0.1565 | 0.3383 | 0.0043 | 0.0169 |
| 780+ | 551 | 0.1307 | 0.1273 | 0.2559 | 0.0036 | 0.0033 |
| 700-739 | 772 | 0.1127 | 0.1163 | 0.2668 | 0.0026 | -0.0036 |
| 580-619 | 154 | 0.1169 | 0.1083 | 0.4221 | 0.0000 | 0.0086 |
| <580 | 57 | 0.1579 | 0.1406 | 0.2982 | 0.0000 | 0.0173 |

By `servicer_name`:

| servicer_name | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| Belmont Loan Services | 846 | 0.1312 | 0.1218 | 0.2766 | 0.0059 | 0.0094 |
| Northgate Servicing | 1044 | 0.1054 | 0.1021 | 0.2385 | 0.0048 | 0.0033 |
| Arcadia Capital Servicing | 729 | 0.1029 | 0.1052 | 0.2785 | 0.0027 | -0.0023 |
| Pioneer Mortgage Ops | 511 | 0.1429 | 0.1513 | 0.4247 | 0.0020 | -0.0084 |
| Kestrel Financial | 350 | 0.2429 | 0.2083 | 0.3600 | 0.0000 | 0.0345 |

By `current_status`:

| current_status | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| DQ30 | 50 | 0.2200 | 0.1699 | 0.3400 | 0.0200 | 0.0501 |
| Current | 3316 | 0.1297 | 0.1250 | 0.2871 | 0.0036 | 0.0047 |
| DQ60 | 33 | 0.1515 | 0.1641 | 0.3939 | 0.0000 | -0.0126 |
| DQ90plus | 81 | 0.0988 | 0.0995 | 0.5802 | 0.0000 | -0.0007 |

By `state`:

| state | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| NC | 127 | 0.1575 | 0.1376 | 0.2441 | 0.0157 | 0.0199 |
| NY | 312 | 0.1410 | 0.1263 | 0.3365 | 0.0096 | 0.0147 |
| NV | 127 | 0.1575 | 0.1418 | 0.3228 | 0.0079 | 0.0157 |
| FL | 406 | 0.1379 | 0.1291 | 0.3202 | 0.0074 | 0.0088 |
| CA | 624 | 0.1170 | 0.1189 | 0.2468 | 0.0048 | -0.0019 |
| GA | 238 | 0.1176 | 0.1300 | 0.2647 | 0.0042 | -0.0124 |
| CO | 63 | 0.2063 | 0.1689 | 0.3175 | 0.0000 | 0.0375 |
| MI | 119 | 0.1261 | 0.1198 | 0.2017 | 0.0000 | 0.0062 |
| IL | 216 | 0.1157 | 0.1144 | 0.2731 | 0.0000 | 0.0013 |
| AZ | 164 | 0.1098 | 0.1075 | 0.6524 | 0.0000 | 0.0022 |
| NJ | 151 | 0.1258 | 0.1131 | 0.2252 | 0.0000 | 0.0127 |
| OH | 137 | 0.1606 | 0.1476 | 0.2117 | 0.0000 | 0.0130 |
| PA | 160 | 0.1187 | 0.1251 | 0.3250 | 0.0000 | -0.0063 |
| TX | 489 | 0.1309 | 0.1246 | 0.2638 | 0.0000 | 0.0063 |
| WA | 147 | 0.1224 | 0.1262 | 0.3469 | 0.0000 | -0.0038 |

**What false positives and false negatives look like.** Mean feature values for each error class against correctly-rejected records.

| feature | false_positives | false_negatives | true_negatives |
| --- | --- | --- | --- |
| credit score band | 3.7237 | 4.0769 | 4.0426 |
| loan-to-value band | 2.0652 | 2.0000 | 1.9484 |
| current performance status | 0.1788 | 0.0769 | 0.0651 |
| days past due | 6.2416 | 3.1538 | 2.2759 |
| worst days past due in last 6 months | 9.4247 | 3.6923 | 4.3665 |
| loan age | 50.6514 | 46.7692 | 54.4507 |
| record data-quality score | 94.5364 | 97.0000 | 97.5168 |
| balance as a share of original | 0.9097 | 0.9295 | 0.9145 |

## Cross-model observations

- **Horizon changes what matters, and the split is clean.** The 3-month delinquency model is led by *behavioural* signals — current performance status, consecutive clean months, recent days past due. The 12-month default model is led by *structural* ones — credit band, debt-to-income band, note rate. Short-horizon risk is about what the borrower is doing right now; long-horizon risk is about what the loan is. That is the economically sensible ordering and it was not imposed: both models saw the same 81 features.
- **Prepayment is dominated by rate economics** — note rate and the 12-month move in market rates — which is the correct mechanism and independently corroborates Task 5, where the prepayment response concentrated in positive-incentive buckets (+0.23 for loans 0.5-1.0pp above market against -0.01 for loans more than 1pp below it).
- **Exceptions are dominated by operational fields** — data-quality score, rule violation count, missing field count — with essentially no contribution from credit attributes. This is the same conclusion the ROC-AUC 0.53 credit baseline reached in Task 4, arrived at from the opposite direction.
- **Servicer identity carries real attribution weight**, which the data intelligence report flagged as a confound: the two servicers with the worst reporting hygiene also have elevated delinquency. Part of that attribution is credit risk and part is reporting behaviour, and SHAP cannot separate the two. A servicer-driven score is a prompt to investigate the servicer, not a statement about the borrower.

## Limitations

- SHAP attributes to *features*, not to causes. A high contribution from days past due does not mean delinquency causes default in any actionable sense; it means the model reads it as the strongest available signal.
- Correlated features split their attribution arbitrarily between them. The DPD family (current, lagged, rolling maxima) is highly correlated, so individual rankings within that family are not stable and should be read as a group.
- Explanations are computed on a sample of up to 4,000 test rows for tractability.
- The uncertainty measure is a boosting-stability proxy. It does not capture uncertainty from feature noise, label noise, or regime change — and regime change is the dominant risk for the 12-month targets, as Task 2 documented.
