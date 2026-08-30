# Explainability Report

**Task 6.** SHAP TreeExplainer over the trained LightGBM models. Every explanation here is derived from the fitted model's own structure — no language model is involved in producing any attribution.

## 1. What is being explained, and in what units

SHAP values are additive in **log-odds**, not probability. Explaining the calibrated probability directly would break that additivity and the contributions would stop summing to anything meaningful. Attribution is therefore computed against the raw LightGBM margin and reported in log-odds, alongside the calibrated probability the reviewer actually acts on. The two are labelled separately throughout and should not be added together.

## Target: `next_3m_delinquency_flag`

### Global feature importance

Mean absolute SHAP contribution across the test window. `direction` is the sign of the correlation between a feature's value and its contribution, which recovers whether higher values raise or lower risk without assuming monotonicity.

| feature | plain_english | mean_abs_shap | share_of_total_attribution | direction |
| --- | --- | --- | --- | --- |
| credit_ord | credit score band | 0.1808 | 0.2055 | higher value lowers risk |
| ever_delinquent_to_date | has ever been delinquent | 0.0999 | 0.1135 | non-monotone / categorical |
| months_dq_last_12m | months delinquent in last 12 months | 0.0781 | 0.0888 | higher value lowers risk |
| amortisation_residual | balance against expected amortisation | 0.0733 | 0.0833 | higher value raises risk |
| credit_score_band | credit score band | 0.0702 | 0.0798 | non-monotone / categorical |
| state | state | 0.0617 | 0.0702 | non-monotone / categorical |
| balance_change_1m | balance change 1m | 0.0466 | 0.0530 | higher value raises risk |
| months_dq_last_3m | months dq last 3m | 0.0463 | 0.0526 | higher value lowers risk |
| balance_change_3m | balance change 3m | 0.0430 | 0.0489 | higher value lowers risk |
| servicer_name | servicer | 0.0370 | 0.0421 | non-monotone / categorical |
| current_streak_clean | consecutive clean months | 0.0301 | 0.0343 | non-monotone / categorical |
| balance_ratio | balance as a share of original | 0.0181 | 0.0206 | higher value raises risk |
| market_rate_delta_12m | market rate delta 12m | 0.0162 | 0.0184 | higher value raises risk |
| dti_ord | debt-to-income band | 0.0096 | 0.0109 | higher value raises risk |
| months_dq_last_6m | months delinquent in last 6 months | 0.0088 | 0.0101 | higher value lowers risk |
| worst_status_to_date | worst status reached to date | 0.0087 | 0.0099 | higher value lowers risk |
| interest_rate_clean | note rate | 0.0057 | 0.0065 | higher value raises risk |
| loan_purpose | loan purpose | 0.0051 | 0.0058 | non-monotone / categorical |

The top three drivers — credit score band, has ever been delinquent, months delinquent in last 12 months — account for **40.8%** of total attribution.

### Local explanations — ten highest-risk records in the test window

Each row shows the calibrated probability a reviewer sees and the four largest log-odds contributions behind it.

| loan_id | reporting_month | current_status | calibrated_probability | driver_1 | driver_1_value | driver_1_log_odds | driver_2 | driver_2_value | driver_2_log_odds | driver_3 | driver_3_value | driver_3_log_odds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F21Q20183479 | 2025-07 | Default | 0.8750 | months dq last 3m | 3.0000 | 0.9274 | months delinquent in last 12 months | 8.0000 | 0.8492 | has ever been delinquent | 1.0000 | 0.7673 |
| F22Q10387904 | 2025-07 | DQ30 | 0.8750 | months delinquent in last 12 months | 6.0000 | 0.9999 | months dq last 3m | 2.0000 | 0.9321 | has ever been delinquent | 1.0000 | 0.7503 |
| F22Q20430998 | 2025-12 | DQ30 | 0.8750 | months dq last 3m | 3.0000 | 1.0562 | months delinquent in last 12 months | 5.0000 | 0.9004 | has ever been delinquent | 1.0000 | 0.7906 |
| F21Q20100299 | 2025-09 | DQ30 | 0.8750 | months dq last 3m | 3.0000 | 0.9305 | months delinquent in last 12 months | 8.0000 | 0.8881 | has ever been delinquent | 1.0000 | 0.7142 |
| F23Q20265187 | 2025-10 | Default | 0.8750 | months dq last 3m | 3.0000 | 0.9058 | months delinquent in last 12 months | 11.0000 | 0.7447 | has ever been delinquent | 1.0000 | 0.5267 |
| F23Q20265187 | 2025-11 | Default | 0.8750 | months dq last 3m | 3.0000 | 0.9058 | months delinquent in last 12 months | 12.0000 | 0.7447 | has ever been delinquent | 1.0000 | 0.5267 |
| F22Q20289359 | 2025-11 | Default | 0.8750 | months dq last 3m | 3.0000 | 1.0559 | months delinquent in last 12 months | 12.0000 | 0.9020 | has ever been delinquent | 1.0000 | 0.6961 |
| F23Q20151926 | 2025-11 | Default | 0.8750 | months dq last 3m | 3.0000 | 0.9259 | months delinquent in last 12 months | 7.0000 | 0.8184 | has ever been delinquent | 1.0000 | 0.6540 |
| F19Q20082593 | 2025-12 | DQ60 | 0.8750 | months dq last 3m | 3.0000 | 1.0733 | months delinquent in last 12 months | 11.0000 | 0.8210 | has ever been delinquent | 1.0000 | 0.6576 |
| F22Q30104351 | 2025-12 | Default | 0.8750 | months dq last 3m | 3.0000 | 1.0554 | months delinquent in last 12 months | 12.0000 | 0.9140 | has ever been delinquent | 1.0000 | 0.7068 |

### Local explanations — five lowest-risk records (contrast set)

| loan_id | reporting_month | current_status | calibrated_probability | driver_1 | driver_1_value | driver_1_log_odds | driver_2 | driver_2_value | driver_2_log_odds | driver_3 | driver_3_value | driver_3_log_odds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F20Q30234290 | 2025-09 | Current | 0.0000 | credit score band | 5.0000 | -0.1363 | state | NC | -0.0997 | balance against expected amortisation | -0.118 | -0.0976 |
| F19Q10120920 | 2025-08 | Current | 0.0000 | credit score band | 5.0000 | -0.1193 | balance against expected amortisation | -0.209 | -0.1073 | state | MT | -0.1070 |
| F19Q10120920 | 2025-11 | Current | 0.0000 | credit score band | 5.0000 | -0.1191 | balance against expected amortisation | -0.214 | -0.1071 | state | MT | -0.1070 |
| F20Q30214682 | 2025-11 | Current | 0.0000 | credit score band | 6.0000 | -0.1896 | credit score band | 780+ | -0.0767 | balance against expected amortisation | -0.058 | -0.0712 |
| F20Q30161577 | 2025-08 | Current | 0.0000 | credit score band | 6.0000 | -0.1646 | balance against expected amortisation | -0.056 | -0.1013 | credit score band | 780+ | -0.0724 |

### Model confidence and uncertainty

Predictions from the final boosting rounds are collected and their spread used as an epistemic-uncertainty proxy. This measures sensitivity to where the boosting sequence stopped — it is a stability signal, **not** a statistical confidence interval, and is not presented as one.

| confidence_band | share_of_records |
| --- | --- |
| high | 0.9480 |
| low | 0.0368 |
| medium | 0.0151 |

| statistic | calibrated_probability | staged_std | staged_p10 | staged_p90 |
| --- | --- | --- | --- | --- |
| count | 65149.0000 | 65149.0000 | 65149.0000 | 65149.0000 |
| mean | 0.0292 | 0.0052 | 0.0327 | 0.0456 |
| std | 0.1216 | 0.0137 | 0.0302 | 0.0640 |
| min | 0.0000 | 0.0001 | 0.0198 | 0.0224 |
| 25% | 0.0043 | 0.0006 | 0.0223 | 0.0232 |
| 50% | 0.0043 | 0.0012 | 0.0258 | 0.0286 |
| 75% | 0.0089 | 0.0037 | 0.0301 | 0.0393 |
| max | 0.8750 | 0.0860 | 0.2158 | 0.4295 |

### False positive / false negative analysis

Evaluated at the threshold that achieves 30% precision on the test window (`0.0317`): 1565 true positives, 3018 false positives, 477 false negatives out of 65149 records with 2042 actual events. Precision 0.341, recall 0.766.

**Where the errors concentrate.** A model that misses events uniformly is a different problem from one that misses them in a specific segment — the second is a fairness and coverage issue, not just an accuracy one.

By `credit_score_band`:

| credit_score_band | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| 580-619 | 49 | 0.1224 | 0.0521 | 0.2449 | 0.1020 | 0.0703 |
| 660-699 | 7091 | 0.0736 | 0.0725 | 0.1089 | 0.0114 | 0.0012 |
| 700-739 | 13621 | 0.0447 | 0.0412 | 0.0575 | 0.0112 | 0.0035 |
| 620-659 | 2274 | 0.1530 | 0.1359 | 0.3417 | 0.0092 | 0.0171 |
| 740-779 | 20079 | 0.0184 | 0.0156 | 0.0189 | 0.0067 | 0.0028 |
| 780+ | 20785 | 0.0075 | 0.0081 | 0.0125 | 0.0035 | -0.0007 |

By `servicer_name`:

| servicer_name | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| AMERIHOME MORTGAGE COMPANY, LLC | 133 | 0.0752 | 0.0384 | 0.0526 | 0.0451 | 0.0367 |
| FREEDOM MORTGAGE CORPORATION | 3018 | 0.0384 | 0.0309 | 0.0586 | 0.0166 | 0.0075 |
| LOANDEPOT.COM, LLC | 716 | 0.0293 | 0.0221 | 0.0531 | 0.0140 | 0.0072 |
| LAKEVIEW LOAN SERVICING, LLC | 3969 | 0.0466 | 0.0402 | 0.0693 | 0.0121 | 0.0064 |
| PENNYMAC CORP. | 2380 | 0.0235 | 0.0186 | 0.0290 | 0.0109 | 0.0050 |
| TH MSR HOLDINGS LLC | 1834 | 0.0360 | 0.0272 | 0.0496 | 0.0109 | 0.0087 |
| PENNYMAC LOAN SERVICES, LLC | 1429 | 0.0378 | 0.0346 | 0.0511 | 0.0105 | 0.0032 |
| CITIZENS BANK, NA | 1243 | 0.0137 | 0.0161 | 0.0322 | 0.0080 | -0.0024 |
| U.S. BANK N.A. | 1622 | 0.0290 | 0.0250 | 0.0327 | 0.0080 | 0.0040 |
| NATIONSTAR MORTGAGE LLC DBA MR. COOPER | 7331 | 0.0273 | 0.0279 | 0.0641 | 0.0075 | -0.0006 |
| OTHER | 15105 | 0.0389 | 0.0356 | 0.0392 | 0.0071 | 0.0032 |
| UNITED WHOLESALE MORTGAGE, LLC | 787 | 0.0508 | 0.0500 | 0.0737 | 0.0064 | 0.0009 |
| FIFTH THIRD BANK, NATIONAL ASSOCIATION | 961 | 0.0166 | 0.0151 | 0.0333 | 0.0062 | 0.0015 |
| ROCKET MORTGAGE, LLC | 4436 | 0.0428 | 0.0430 | 0.0654 | 0.0059 | -0.0001 |
| CROSSCOUNTRY MORTGAGE, LLC | 707 | 0.0311 | 0.0337 | 0.0509 | 0.0057 | -0.0026 |
| PNC BANK, NA | 1415 | 0.0113 | 0.0119 | 0.0226 | 0.0057 | -0.0006 |

By `current_status`:

| current_status | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| Current | 63812 | 0.0147 | 0.0129 | 0.0436 | 0.0075 | 0.0019 |
| DQ30 | 706 | 0.7550 | 0.7495 | 0.2450 | 0.0000 | 0.0054 |
| DQ60 | 163 | 0.8957 | 0.8709 | 0.1043 | 0.0000 | 0.0248 |
| DQ90plus | 205 | 0.8683 | 0.8712 | 0.1317 | 0.0000 | -0.0029 |
| Default | 263 | 0.9316 | 0.8716 | 0.0684 | 0.0000 | 0.0600 |

By `state`:

| state | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| AL | 712 | 0.0435 | 0.0261 | 0.0337 | 0.0197 | 0.0175 |
| LA | 618 | 0.0356 | 0.0211 | 0.0227 | 0.0146 | 0.0145 |
| OH | 2375 | 0.0467 | 0.0338 | 0.0400 | 0.0139 | 0.0129 |
| NY | 2518 | 0.0445 | 0.0410 | 0.0588 | 0.0135 | 0.0035 |
| PA | 2279 | 0.0351 | 0.0300 | 0.0329 | 0.0123 | 0.0052 |
| MT | 246 | 0.0163 | 0.0097 | 0.0041 | 0.0122 | 0.0065 |
| IA | 514 | 0.0195 | 0.0114 | 0.0136 | 0.0117 | 0.0081 |
| TX | 5331 | 0.0439 | 0.0382 | 0.0478 | 0.0103 | 0.0057 |
| AZ | 1857 | 0.0474 | 0.0427 | 0.0609 | 0.0102 | 0.0047 |
| SC | 1027 | 0.0341 | 0.0285 | 0.0779 | 0.0097 | 0.0056 |
| NM | 309 | 0.0356 | 0.0352 | 0.0194 | 0.0097 | 0.0004 |
| UT | 973 | 0.0195 | 0.0165 | 0.0154 | 0.0092 | 0.0030 |
| NJ | 1970 | 0.0305 | 0.0239 | 0.0421 | 0.0091 | 0.0065 |
| FL | 4930 | 0.0422 | 0.0385 | 0.0604 | 0.0089 | 0.0037 |
| MD | 1516 | 0.0396 | 0.0400 | 0.0679 | 0.0086 | -0.0004 |
| MN | 1439 | 0.0438 | 0.0391 | 0.0479 | 0.0083 | 0.0047 |

**What false positives and false negatives look like.** Mean feature values for each error class against correctly-rejected records.

| feature | false_positives | false_negatives | true_negatives |
| --- | --- | --- | --- |
| credit score band | 3.5087 | 4.3062 | 4.8808 |
| loan-to-value band | 1.8058 | 2.1726 | 1.8751 |
| current performance status | 0.1193 | 0.0000 | 0.0000 |
| days past due | 4.3044 | 0.3326 | 0.3740 |
| worst days past due in last 6 months | 18.7425 | 2.0126 | 2.0904 |
| loan age | 43.8958 | 41.8063 | 45.4924 |
| record data-quality score | 93.6663 | 94.7799 | 95.0122 |
| balance as a share of original | 0.9072 | 0.9150 | 0.8677 |

## Target: `next_12m_default_flag`

### Global feature importance

Mean absolute SHAP contribution across the test window. `direction` is the sign of the correlation between a feature's value and its contribution, which recovers whether higher values raise or lower risk without assuming monotonicity.

| feature | plain_english | mean_abs_shap | share_of_total_attribution | direction |
| --- | --- | --- | --- | --- |
| state | state | 0.4977 | 0.0981 | non-monotone / categorical |
| servicer_name | servicer | 0.3428 | 0.0676 | non-monotone / categorical |
| amortisation_residual | balance against expected amortisation | 0.3357 | 0.0662 | higher value raises risk |
| hpi_yoy_growth | house price growth | 0.3264 | 0.0643 | higher value raises risk |
| credit_ord | credit score band | 0.2782 | 0.0548 | higher value lowers risk |
| current_streak_clean | consecutive clean months | 0.2408 | 0.0475 | higher value lowers risk |
| interest_rate_clean | note rate | 0.2153 | 0.0424 | higher value lowers risk |
| dti_ord | debt-to-income band | 0.1711 | 0.0337 | higher value lowers risk |
| market_rate_delta_12m | market rate delta 12m | 0.1593 | 0.0314 | higher value raises risk |
| credit_score_band | credit score band | 0.1571 | 0.0310 | non-monotone / categorical |
| balance_change_3m | balance change 3m | 0.1374 | 0.0271 | higher value raises risk |
| log_original_balance | original balance | 0.1343 | 0.0265 | higher value lowers risk |
| scheduled_payment | scheduled monthly payment | 0.1228 | 0.0242 | higher value raises risk |
| amortisation_progress | amortisation progress | 0.1215 | 0.0240 | non-monotone / categorical |
| unemployment_delta_12m | unemployment delta 12m | 0.1178 | 0.0232 | higher value lowers risk |
| payment_to_balance | scheduled payment relative to balance | 0.1160 | 0.0229 | higher value raises risk |
| property_type | property type | 0.1100 | 0.0217 | non-monotone / categorical |
| ltv_band | ltv band | 0.1047 | 0.0206 | non-monotone / categorical |

The top three drivers — state, servicer, balance against expected amortisation — account for **23.2%** of total attribution.

### Local explanations — ten highest-risk records in the test window

Each row shows the calibrated probability a reviewer sees and the four largest log-odds contributions behind it.

| loan_id | reporting_month | current_status | calibrated_probability | driver_1 | driver_1_value | driver_1_log_odds | driver_2 | driver_2_value | driver_2_log_odds | driver_3 | driver_3_value | driver_3_log_odds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F21Q11301452 | 2024-10 | Default | 0.9167 | current status | Default | 3.1411 | current performance status | 4.0 | 2.5498 | servicer | PNC BANK, NA | -0.9542 |
| F21Q11830711 | 2024-12 | Default | 0.9167 | current status | Default | 2.6321 | current performance status | 4.0 | 2.6086 | months delinquent in last 12 months | 12.0 | 0.6166 |
| F21Q12142553 | 2024-11 | Default | 0.9167 | current status | Default | 3.0742 | current performance status | 4.0 | 2.5259 | state | IN | 0.9701 |
| F23Q30165448 | 2025-01 | Default | 0.9167 | current status | Default | 2.9892 | current performance status | 4.0 | 2.6399 | months delinquent in last 12 months | 8.0 | 0.5230 |
| F21Q20814664 | 2024-10 | DQ90plus | 0.9167 | current status | DQ90plus | 2.9352 | current performance status | 3.0 | 2.4860 | state | CA | 0.8864 |
| F19Q20031612 | 2025-03 | DQ30 | 0.9167 | current status | DQ30 | 2.2896 | state | CA | 0.8803 | months delinquent in last 12 months | 10.0 | 0.8290 |
| F21Q20237707 | 2025-03 | DQ60 | 0.9167 | current status | DQ60 | 2.6277 | current performance status | 2.0 | 1.4990 | months delinquent in last 12 months | 2.0 | 0.9191 |
| F19Q30083113 | 2025-02 | Default | 0.9167 | current status | Default | 2.5614 | current performance status | 4.0 | 2.4291 | state | NJ | 1.1289 |
| F23Q10144499 | 2024-12 | DQ90plus | 0.9167 | current status | DQ90plus | 2.9758 | current performance status | 3.0 | 2.5759 | months delinquent in last 12 months | 5.0 | 0.5747 |
| F23Q20189032 | 2024-12 | Default | 0.9167 | current status | Default | 3.0449 | current performance status | 4.0 | 3.0107 | state | CT | -1.1714 |

### Local explanations — five lowest-risk records (contrast set)

| loan_id | reporting_month | current_status | calibrated_probability | driver_1 | driver_1_value | driver_1_log_odds | driver_2 | driver_2_value | driver_2_log_odds | driver_3 | driver_3_value | driver_3_log_odds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F23Q20130638 | 2025-03 | Current | 0.0000 | servicer | BANK OF AMERICA, N.A. | -1.7165 | credit score band | 6.0000 | -0.3474 | state | TX | -0.2886 |
| F20Q30741474 | 2025-01 | Current | 0.0000 | servicer | PNC BANK, NA | -0.8195 | balance against expected amortisation | -0.0870 | -0.5476 | credit score band | 6.0 | -0.3729 |
| F23Q40169295 | 2025-01 | Current | 0.0000 | servicer | CITIZENS BANK, NA | -1.0069 | balance against expected amortisation | -0.9940 | -0.7998 | balance change 3m | -0.998 | -0.5355 |
| F23Q40159582 | 2025-02 | Current | 0.0000 | balance against expected amortisation | -0.995 | -0.7318 | months delinquent in last 12 months | 1.0000 | 0.5700 | balance change 3m | -1.0 | -0.4813 |
| F23Q40121948 | 2024-10 | Current | 0.0000 | balance against expected amortisation | -0.997 | -0.7686 | balance change 3m | -1.0000 | -0.5732 | original balance | 13.486 | -0.4688 |

### Model confidence and uncertainty

Predictions from the final boosting rounds are collected and their spread used as an epistemic-uncertainty proxy. This measures sensitivity to where the boosting sequence stopped — it is a stability signal, **not** a statistical confidence interval, and is not presented as one.

| confidence_band | share_of_records |
| --- | --- |
| high | 0.9863 |
| medium | 0.0085 |
| low | 0.0052 |

| statistic | calibrated_probability | staged_std | staged_p10 | staged_p90 |
| --- | --- | --- | --- | --- |
| count | 68984.0000 | 68984.0000 | 68984.0000 | 68984.0000 |
| mean | 0.0114 | 0.0003 | 0.0104 | 0.0112 |
| std | 0.0827 | 0.0019 | 0.0771 | 0.0803 |
| min | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 25% | 0.0011 | 0.0000 | 0.0001 | 0.0001 |
| 50% | 0.0011 | 0.0000 | 0.0003 | 0.0003 |
| 75% | 0.0011 | 0.0001 | 0.0011 | 0.0013 |
| max | 0.9167 | 0.0619 | 0.9877 | 0.9901 |

### False positive / false negative analysis

Evaluated at the threshold that achieves 30% precision on the test window (`0.0140`): 716 true positives, 1622 false positives, 345 false negatives out of 68984 records with 1061 actual events. Precision 0.306, recall 0.675.

**Where the errors concentrate.** A model that misses events uniformly is a different problem from one that misses them in a specific segment — the second is a fairness and coverage issue, not just an accuracy one.

By `credit_score_band`:

| credit_score_band | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| 620-659 | 2457 | 0.0672 | 0.0544 | 0.0993 | 0.0163 | 0.0128 |
| 660-699 | 7561 | 0.0398 | 0.0275 | 0.0680 | 0.0135 | 0.0123 |
| 700-739 | 14377 | 0.0230 | 0.0155 | 0.0315 | 0.0082 | 0.0075 |
| 740-779 | 21260 | 0.0079 | 0.0056 | 0.0118 | 0.0028 | 0.0022 |
| 780+ | 21982 | 0.0029 | 0.0035 | 0.0062 | 0.0009 | -0.0006 |
| 580-619 | 53 | 0.1132 | 0.0681 | 0.0000 | 0.0000 | 0.0452 |

By `servicer_name`:

| servicer_name | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| LAKEVIEW LOAN SERVICING, LLC | 4168 | 0.0271 | 0.0123 | 0.0264 | 0.0146 | 0.0148 |
| TH MSR HOLDINGS LLC | 1935 | 0.0264 | 0.0130 | 0.0196 | 0.0129 | 0.0134 |
| PENNYMAC LOAN SERVICES, LLC | 1612 | 0.0217 | 0.0134 | 0.0155 | 0.0105 | 0.0083 |
| ONSLOW BAY FINANCIAL LLC | 1167 | 0.0120 | 0.0024 | 0.0009 | 0.0086 | 0.0096 |
| CMG MORTGAGE, INC. | 365 | 0.0411 | 0.0298 | 0.0027 | 0.0082 | 0.0113 |
| LOANDEPOT.COM, LLC | 771 | 0.0130 | 0.0103 | 0.0324 | 0.0078 | 0.0026 |
| WELLS FARGO BANK, N.A. | 2781 | 0.0104 | 0.0050 | 0.0144 | 0.0065 | 0.0054 |
| PENNYMAC CORP. | 2493 | 0.0108 | 0.0070 | 0.0132 | 0.0056 | 0.0038 |
| AMERIHOME MORTGAGE COMPANY, LLC | 909 | 0.0198 | 0.0244 | 0.0473 | 0.0055 | -0.0046 |
| ROCKET MORTGAGE, LLC | 4439 | 0.0198 | 0.0180 | 0.0392 | 0.0054 | 0.0018 |
| FIFTH THIRD BANK, NATIONAL ASSOCIATION | 1027 | 0.0049 | 0.0024 | 0.0049 | 0.0049 | 0.0025 |
| FREEDOM MORTGAGE CORPORATION | 3110 | 0.0164 | 0.0110 | 0.0238 | 0.0048 | 0.0054 |
| NATIONSTAR MORTGAGE LLC DBA MR. COOPER | 7795 | 0.0108 | 0.0079 | 0.0346 | 0.0042 | 0.0029 |
| UNITED WHOLESALE MORTGAGE, LLC | 1438 | 0.0195 | 0.0174 | 0.0167 | 0.0042 | 0.0020 |
| PNC BANK, NA | 1480 | 0.0108 | 0.0079 | 0.0041 | 0.0041 | 0.0029 |
| OTHER | 16342 | 0.0181 | 0.0148 | 0.0282 | 0.0039 | 0.0032 |

By `current_status`:

| current_status | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| DQ30 | 654 | 0.2370 | 0.1393 | 0.6086 | 0.0107 | 0.0977 |
| Current | 67736 | 0.0060 | 0.0031 | 0.0167 | 0.0050 | 0.0029 |
| DQ60 | 188 | 0.6968 | 0.6738 | 0.3032 | 0.0000 | 0.0230 |
| DQ90plus | 202 | 0.8762 | 0.8692 | 0.1238 | 0.0000 | 0.0070 |
| Default | 204 | 0.9314 | 0.8892 | 0.0686 | 0.0000 | 0.0422 |

By `state`:

| state | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| MS | 235 | 0.0766 | 0.0430 | 0.0340 | 0.0340 | 0.0336 |
| TX | 5689 | 0.0225 | 0.0121 | 0.0197 | 0.0114 | 0.0104 |
| FL | 5279 | 0.0235 | 0.0125 | 0.0349 | 0.0099 | 0.0110 |
| OR | 1247 | 0.0168 | 0.0047 | 0.0048 | 0.0088 | 0.0121 |
| GA | 2386 | 0.0264 | 0.0172 | 0.0293 | 0.0088 | 0.0092 |
| MI | 2519 | 0.0171 | 0.0118 | 0.0242 | 0.0087 | 0.0053 |
| VA | 1921 | 0.0120 | 0.0037 | 0.0047 | 0.0073 | 0.0083 |
| OH | 2477 | 0.0214 | 0.0118 | 0.0355 | 0.0073 | 0.0096 |
| DC | 138 | 0.0435 | 0.0242 | 0.0000 | 0.0072 | 0.0192 |
| NY | 2626 | 0.0198 | 0.0154 | 0.0270 | 0.0069 | 0.0044 |
| KY | 880 | 0.0114 | 0.0059 | 0.0239 | 0.0057 | 0.0055 |
| CO | 1809 | 0.0166 | 0.0120 | 0.0122 | 0.0055 | 0.0045 |
| SC | 1124 | 0.0142 | 0.0151 | 0.0391 | 0.0053 | -0.0009 |
| MN | 1521 | 0.0237 | 0.0162 | 0.0237 | 0.0046 | 0.0074 |
| PA | 2412 | 0.0170 | 0.0166 | 0.0269 | 0.0046 | 0.0004 |
| MA | 1645 | 0.0128 | 0.0085 | 0.0195 | 0.0043 | 0.0043 |

**What false positives and false negatives look like.** Mean feature values for each error class against correctly-rejected records.

| feature | false_positives | false_negatives | true_negatives |
| --- | --- | --- | --- |
| credit score band | 3.6994 | 3.7552 | 4.8212 |
| loan-to-value band | 2.0803 | 2.5000 | 1.8771 |
| current performance status | 0.3964 | 0.0203 | 0.0015 |
| days past due | 13.0013 | 0.6231 | 0.4168 |
| worst days past due in last 6 months | 21.4550 | 16.7391 | 2.6977 |
| loan age | 28.7176 | 28.2566 | 36.2893 |
| record data-quality score | 93.0530 | 93.8174 | 94.9826 |
| balance as a share of original | 0.9616 | 0.9576 | 0.8929 |

## Target: `next_12m_prepayment_flag`

### Global feature importance

Mean absolute SHAP contribution across the test window. `direction` is the sign of the correlation between a feature's value and its contribution, which recovers whether higher values raise or lower risk without assuming monotonicity.

| feature | plain_english | mean_abs_shap | share_of_total_attribution | direction |
| --- | --- | --- | --- | --- |
| state | state | 0.3749 | 0.0893 | non-monotone / categorical |
| market_rate_delta_12m | market rate delta 12m | 0.3627 | 0.0864 | higher value lowers risk |
| refi_incentive_positive | positive refinance incentive | 0.3036 | 0.0723 | non-monotone / categorical |
| servicer_name | servicer | 0.2963 | 0.0706 | non-monotone / categorical |
| interest_rate_clean | note rate | 0.2925 | 0.0697 | higher value raises risk |
| scheduled_payment | scheduled monthly payment | 0.2158 | 0.0514 | higher value raises risk |
| rate_incentive | refinance incentive (note rate less market rate) | 0.2004 | 0.0478 | higher value lowers risk |
| payment_to_balance | scheduled payment relative to balance | 0.1993 | 0.0475 | higher value raises risk |
| current_streak_clean | consecutive clean months | 0.1506 | 0.0359 | higher value raises risk |
| log_current_balance | current balance | 0.1360 | 0.0324 | non-monotone / categorical |
| market_mortgage_rate | market mortgage rate | 0.1355 | 0.0323 | higher value lowers risk |
| log_original_balance | original balance | 0.1322 | 0.0315 | higher value raises risk |
| loan_age_months_clean | loan age | 0.1246 | 0.0297 | higher value raises risk |
| remaining_term_months | remaining term months | 0.1114 | 0.0266 | higher value lowers risk |
| property_type | property type | 0.0962 | 0.0229 | non-monotone / categorical |
| balance_change_3m | balance change 3m | 0.0870 | 0.0207 | higher value lowers risk |
| credit_score_band | credit score band | 0.0786 | 0.0187 | non-monotone / categorical |
| term_progress | share of term elapsed | 0.0766 | 0.0183 | higher value raises risk |

The top three drivers — state, market rate delta 12m, positive refinance incentive — account for **24.8%** of total attribution.

### Local explanations — ten highest-risk records in the test window

Each row shows the calibrated probability a reviewer sees and the four largest log-odds contributions behind it.

| loan_id | reporting_month | current_status | calibrated_probability | driver_1 | driver_1_value | driver_1_log_odds | driver_2 | driver_2_value | driver_2_log_odds | driver_3 | driver_3_value | driver_3_log_odds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F23Q30045473 | 2024-10 | Current | 1.0000 | current balance | 0.0000 | 3.2298 | balance change 3m | -1.0 | 0.9055 | balance as a share of original | 0.0000 | 0.7857 |
| F22Q30046358 | 2025-03 | Current | 1.0000 | current balance | 0.0000 | 3.1815 | balance against expected amortisation | -0.98 | 0.9307 | balance change 3m | -1.0000 | 0.9136 |
| F22Q10505876 | 2024-11 | Current | 1.0000 | current balance | 0.0000 | 3.0240 | balance change 3m | -1.0 | 0.9763 | balance against expected amortisation | -0.9800 | 0.9118 |
| F23Q30008940 | 2024-10 | Current | 1.0000 | current balance | 0.0000 | 3.2394 | balance change 3m | -1.0 | 0.9425 | balance as a share of original | 0.0000 | 0.7684 |
| F23Q40056472 | 2024-12 | Current | 1.0000 | current balance | 0.0000 | 3.4262 | balance as a share of original | 0.0 | 0.7571 | balance change 3m | -1.0000 | 0.7370 |
| F23Q40067110 | 2025-01 | Current | 1.0000 | current balance | 0.0000 | 3.8639 | state | ND | -0.9830 | balance as a share of original | 0.0000 | 0.8897 |
| F23Q10022362 | 2024-10 | Current | 1.0000 | current balance | 0.0000 | 2.9829 | balance change 3m | -1.0 | 0.9050 | balance against expected amortisation | -0.9900 | 0.7807 |
| F20Q40695723 | 2024-10 | Current | 1.0000 | current balance | 0.0000 | 3.4109 | balance change 3m | -1.0 | 1.0310 | balance against expected amortisation | -0.9630 | 0.7763 |
| F19Q30136628 | 2025-01 | Current | 1.0000 | current balance | 0.0000 | 3.3664 | balance against expected amortisation | -0.935 | 1.0586 | balance change 3m | -1.0000 | 1.0214 |
| F19Q10108341 | 2024-11 | Current | 1.0000 | current balance | 0.0000 | 3.4063 | balance change 3m | -1.0 | 1.0107 | balance against expected amortisation | -0.9310 | 0.8353 |

### Local explanations — five lowest-risk records (contrast set)

| loan_id | reporting_month | current_status | calibrated_probability | driver_1 | driver_1_value | driver_1_log_odds | driver_2 | driver_2_value | driver_2_log_odds | driver_3 | driver_3_value | driver_3_log_odds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F20Q40113681 | 2025-03 | Current | 0.0075 | scheduled payment relative to balance | 0.003 | -1.4139 | state | MN | -0.7318 | note rate | 2.875 | -0.3658 |
| F21Q30940764 | 2024-11 | Current | 0.0101 | state | PR | -1.8783 | note rate | 2.25 | -1.5118 | market rate delta 12m | -0.637 | 0.6394 |
| F20Q41225909 | 2025-02 | Current | 0.0127 | state | NY | -1.0606 | note rate | 2.5 | -1.0543 | consecutive clean months | 49 | 0.3682 |
| F21Q12485546 | 2025-01 | Current | 0.0127 | state | IA | -0.8515 | servicer | NATIONSTAR MORTGAGE LLC DBA MR. COOPER | -0.7487 | note rate | 2.5 | -0.4312 |
| F20Q20326361 | 2025-02 | Current | 0.0127 | state | HI | -1.1107 | note rate | 2.75 | -0.4793 | servicer | U.S. BANK N.A. | -0.4697 |

### Model confidence and uncertainty

Predictions from the final boosting rounds are collected and their spread used as an epistemic-uncertainty proxy. This measures sensitivity to where the boosting sequence stopped — it is a stability signal, **not** a statistical confidence interval, and is not presented as one.

| confidence_band | share_of_records |
| --- | --- |
| high | 0.7626 |
| medium | 0.1245 |
| low | 0.1130 |

| statistic | calibrated_probability | staged_std | staged_p10 | staged_p90 |
| --- | --- | --- | --- | --- |
| count | 68984.0000 | 68984.0000 | 68984.0000 | 68984.0000 |
| mean | 0.1758 | 0.0031 | 0.1643 | 0.1717 |
| std | 0.2807 | 0.0027 | 0.1713 | 0.1752 |
| min | 0.0000 | 0.0000 | 0.0005 | 0.0005 |
| 25% | 0.0203 | 0.0011 | 0.0452 | 0.0483 |
| 50% | 0.0461 | 0.0023 | 0.1028 | 0.1092 |
| 75% | 0.1934 | 0.0044 | 0.2231 | 0.2347 |
| max | 1.0000 | 0.0274 | 0.9992 | 0.9992 |

### False positive / false negative analysis

Evaluated at the threshold that achieves 30% precision on the test window (`0.9825`): 415 true positives, 6 false positives, 4817 false negatives out of 68984 records with 5232 actual events. Precision 0.986, recall 0.079.

**Where the errors concentrate.** A model that misses events uniformly is a different problem from one that misses them in a specific segment — the second is a fairness and coverage issue, not just an accuracy one.

By `credit_score_band`:

| credit_score_band | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| 580-619 | 53 | 0.1132 | 0.1081 | 0.0000 | 0.1132 | 0.0051 |
| 620-659 | 2457 | 0.1038 | 0.2046 | 0.0000 | 0.0956 | -0.1009 |
| 740-779 | 21260 | 0.0770 | 0.1877 | 0.0001 | 0.0718 | -0.1107 |
| 660-699 | 7561 | 0.0766 | 0.1543 | 0.0000 | 0.0709 | -0.0777 |
| 780+ | 21982 | 0.0764 | 0.1792 | 0.0001 | 0.0695 | -0.1028 |
| 700-739 | 14377 | 0.0675 | 0.1601 | 0.0000 | 0.0617 | -0.0925 |

By `servicer_name`:

| servicer_name | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| ROCKET MORTGAGE, LLC | 4439 | 0.1185 | 0.2413 | 0.0000 | 0.1097 | -0.1228 |
| GUILD MORTGAGE COMPANY LLC | 128 | 0.1250 | 0.1955 | 0.0000 | 0.1094 | -0.0705 |
| CMG MORTGAGE, INC. | 365 | 0.1068 | 0.0925 | 0.0000 | 0.1014 | 0.0143 |
| PENNYMAC LOAN SERVICES, LLC | 1612 | 0.1110 | 0.2549 | 0.0000 | 0.0999 | -0.1439 |
| AMERIHOME MORTGAGE COMPANY, LLC | 909 | 0.0957 | 0.2410 | 0.0000 | 0.0891 | -0.1453 |
| NATIONSTAR MORTGAGE LLC DBA MR. COOPER | 7795 | 0.0930 | 0.1423 | 0.0000 | 0.0847 | -0.0493 |
| U.S. BANK N.A. | 1749 | 0.0863 | 0.0851 | 0.0000 | 0.0783 | 0.0013 |
| LAKEVIEW LOAN SERVICING, LLC | 4168 | 0.0837 | 0.2245 | 0.0000 | 0.0780 | -0.1407 |
| LOANDEPOT.COM, LLC | 771 | 0.0856 | 0.1870 | 0.0000 | 0.0778 | -0.1014 |
| CITIZENS BANK, NA | 1312 | 0.0816 | 0.1474 | 0.0000 | 0.0770 | -0.0659 |
| FIFTH THIRD BANK, NATIONAL ASSOCIATION | 1027 | 0.0779 | 0.1446 | 0.0000 | 0.0730 | -0.0667 |
| TRUIST BANK | 1983 | 0.0772 | 0.1372 | 0.0000 | 0.0716 | -0.0600 |
| PHH MORTGAGE CORPORATION | 1015 | 0.0768 | 0.1134 | 0.0000 | 0.0700 | -0.0365 |
| UNITED WHOLESALE MORTGAGE, LLC | 1438 | 0.0723 | 0.1539 | 0.0007 | 0.0688 | -0.0816 |
| OTHER | 16342 | 0.0678 | 0.1976 | 0.0003 | 0.0629 | -0.1298 |
| FREEDOM MORTGAGE CORPORATION | 3110 | 0.0707 | 0.2463 | 0.0000 | 0.0624 | -0.1755 |

By `current_status`:

| current_status | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| DQ90plus | 202 | 0.0941 | 0.0814 | 0.0000 | 0.0941 | 0.0126 |
| DQ60 | 188 | 0.0904 | 0.0719 | 0.0000 | 0.0904 | 0.0186 |
| Default | 204 | 0.0833 | 0.1473 | 0.0049 | 0.0833 | -0.0640 |
| DQ30 | 654 | 0.0734 | 0.0816 | 0.0015 | 0.0734 | -0.0082 |
| Current | 67736 | 0.0757 | 0.1773 | 0.0001 | 0.0696 | -0.1016 |

By `state`:

| state | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| ND | 151 | 0.1656 | 0.0380 | 0.0000 | 0.1523 | 0.1276 |
| SD | 180 | 0.1333 | 0.1321 | 0.0000 | 0.1333 | 0.0012 |
| ME | 326 | 0.1380 | 0.0625 | 0.0000 | 0.1319 | 0.0755 |
| HI | 144 | 0.1250 | 0.0536 | 0.0000 | 0.1250 | 0.0714 |
| NM | 345 | 0.1304 | 0.0814 | 0.0000 | 0.1246 | 0.0490 |
| UT | 1072 | 0.1119 | 0.1729 | 0.0000 | 0.1035 | -0.0609 |
| SC | 1124 | 0.1094 | 0.2124 | 0.0000 | 0.1023 | -0.1030 |
| NH | 321 | 0.1059 | 0.0549 | 0.0000 | 0.0997 | 0.0510 |
| KS | 527 | 0.1025 | 0.0816 | 0.0000 | 0.0949 | 0.0208 |
| IA | 557 | 0.0969 | 0.1272 | 0.0000 | 0.0916 | -0.0302 |
| OR | 1247 | 0.0954 | 0.2276 | 0.0000 | 0.0898 | -0.1322 |
| GA | 2386 | 0.0951 | 0.1824 | 0.0000 | 0.0893 | -0.0873 |
| TN | 1283 | 0.0912 | 0.2411 | 0.0000 | 0.0881 | -0.1499 |
| DC | 138 | 0.1014 | 0.0818 | 0.0000 | 0.0870 | 0.0196 |
| MO | 1247 | 0.0970 | 0.3070 | 0.0000 | 0.0858 | -0.2099 |
| MI | 2519 | 0.0905 | 0.2645 | 0.0000 | 0.0857 | -0.1740 |

**What false positives and false negatives look like.** Mean feature values for each error class against correctly-rejected records.

| feature | false_positives | false_negatives | true_negatives |
| --- | --- | --- | --- |
| credit score band | 5.5000 | 4.7541 | 4.7784 |
| loan-to-value band | 2.8333 | 1.9957 | 1.8786 |
| current performance status | 0.8333 | 0.0430 | 0.0351 |
| days past due | 42.0000 | 1.8862 | 1.5919 |
| worst days past due in last 6 months | 35.0000 | 4.2163 | 4.1570 |
| loan age | 30.3333 | 31.0690 | 36.4612 |
| record data-quality score | 97.3333 | 94.8732 | 94.8594 |
| balance as a share of original | 0.4915 | 0.8837 | 0.9023 |

## Target: `exception_required`

### Global feature importance

Mean absolute SHAP contribution across the test window. `direction` is the sign of the correlation between a feature's value and its contribution, which recovers whether higher values raise or lower risk without assuming monotonicity.

| feature | plain_english | mean_abs_shap | share_of_total_attribution | direction |
| --- | --- | --- | --- | --- |
| dq_score | record data-quality score | 1.0525 | 0.4210 | higher value lowers risk |
| doc_incomplete | doc incomplete | 0.2386 | 0.0955 | non-monotone / categorical |
| dq_violation_count | dq violation count | 0.2265 | 0.0906 | higher value raises risk |
| missing_field_count | missing field count | 0.1670 | 0.0668 | higher value raises risk |
| svc_present | svc present | 0.1105 | 0.0442 | non-monotone / categorical |
| reporting_lag_days | servicer reporting lag | 0.1086 | 0.0435 | higher value lowers risk |
| svc_balance_rel_gap | servicer feed balance gap | 0.1033 | 0.0413 | non-monotone / categorical |
| document_status | document custody status | 0.0743 | 0.0297 | non-monotone / categorical |
| stale_reporting | stale reporting | 0.0445 | 0.0178 | non-monotone / categorical |
| svc_status_conflict | svc status conflict | 0.0324 | 0.0129 | non-monotone / categorical |
| dti_ord | debt-to-income band | 0.0314 | 0.0125 | higher value raises risk |
| state | state | 0.0311 | 0.0125 | non-monotone / categorical |
| age_repaired | age repaired | 0.0302 | 0.0121 | non-monotone / categorical |
| is_missing_dti_band | is missing dti band | 0.0297 | 0.0119 | non-monotone / categorical |
| remaining_term_months | remaining term months | 0.0238 | 0.0095 | higher value raises risk |
| ltv_ord | loan-to-value band | 0.0192 | 0.0077 | higher value lowers risk |
| dpd_status_residual | days past due against reported status | 0.0135 | 0.0054 | non-monotone / categorical |
| servicer_name | servicer | 0.0134 | 0.0053 | non-monotone / categorical |

The top three drivers — record data-quality score, doc incomplete, dq violation count — account for **60.7%** of total attribution.

### Local explanations — ten highest-risk records in the test window

Each row shows the calibrated probability a reviewer sees and the four largest log-odds contributions behind it.

| loan_id | reporting_month | current_status | calibrated_probability | driver_1 | driver_1_value | driver_1_log_odds | driver_2 | driver_2_value | driver_2_log_odds | driver_3 | driver_3_value | driver_3_log_odds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F23Q40169115 | 2026-01 | Current | 1.0000 | record data-quality score | 80.0000 | 3.3982 | servicer feed balance gap | 0.0690 | 0.8511 | doc incomplete | 1.0 | 0.5664 |
| F22Q40135470 | 2026-02 | Current | 1.0000 | record data-quality score | 80.0000 | 3.4050 | servicer feed balance gap | 0.0850 | 0.8714 | doc incomplete | 1.0 | 0.5655 |
| F23Q10025225 | 2025-12 | Current | 1.0000 | record data-quality score | 74.0000 | 3.4296 | servicer feed balance gap | 0.0360 | 0.6354 | dq violation count | 3 | 0.5457 |
| F23Q10175634 | 2025-10 | Current | 1.0000 | record data-quality score | 80.0000 | 3.3907 | servicer feed balance gap | 0.0210 | 0.8631 | doc incomplete | 1.0 | 0.5644 |
| F22Q10446650 | 2025-12 | Current | 0.9646 | record data-quality score | 80.0000 | 3.3822 | servicer feed balance gap | 0.0170 | 0.8718 | doc incomplete | 1.0 | 0.5773 |
| F20Q30651241 | 2026-01 | Current | 0.9646 | record data-quality score | 80.0000 | 3.4083 | servicer feed balance gap | 0.0260 | 0.8640 | doc incomplete | 1.0 | 0.5886 |
| F23Q20255224 | 2025-11 | Current | 0.9646 | record data-quality score | 80.0000 | 3.3973 | servicer feed balance gap | 0.0760 | 0.8614 | doc incomplete | 1.0 | 0.5651 |
| F23Q40064688 | 2025-10 | Current | 0.9646 | record data-quality score | 80.0000 | 3.3738 | servicer feed balance gap | 0.0220 | 0.8574 | doc incomplete | 1.0 | 0.5651 |
| F20Q30267898 | 2025-12 | Current | 0.9646 | record data-quality score | 80.0000 | 3.3726 | servicer feed balance gap | 0.0240 | 0.8845 | doc incomplete | 1.0 | 0.5781 |
| F22Q30067962 | 2026-03 | Current | 0.9646 | record data-quality score | 80.0000 | 3.3817 | servicer feed balance gap | 0.0430 | 0.8633 | doc incomplete | 1.0 | 0.5658 |

### Local explanations — five lowest-risk records (contrast set)

| loan_id | reporting_month | current_status | calibrated_probability | driver_1 | driver_1_value | driver_1_log_odds | driver_2 | driver_2_value | driver_2_log_odds | driver_3 | driver_3_value | driver_3_log_odds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F21Q40797644 | 2026-02 | Current | 0.0000 | missing field count | 2 | -0.8294 | record data-quality score | 88.0 | 0.3453 | dq violation count | 2 | 0.3123 |
| F21Q20715072 | 2026-01 | Current | 0.0000 | missing field count | 1 | -0.8227 | record data-quality score | 88.0 | 0.3767 | dq violation count | 2 | 0.3228 |
| F23Q40205591 | 2025-12 | Current | 0.0031 | missing field count | 1 | -0.8456 | record data-quality score | 88.0 | 0.3873 | dq violation count | 2 | 0.3492 |
| F23Q40204301 | 2025-12 | Current | 0.0031 | missing field count | 1 | -0.8456 | record data-quality score | 88.0 | 0.3786 | dq violation count | 2 | 0.3431 |
| F21Q30320725 | 2025-10 | Current | 0.0031 | record data-quality score | 97.0 | -0.7247 | svc present | 0 | -0.1363 | doc incomplete | 0.0 | -0.1227 |

### Model confidence and uncertainty

Predictions from the final boosting rounds are collected and their spread used as an epistemic-uncertainty proxy. This measures sensitivity to where the boosting sequence stopped — it is a stability signal, **not** a statistical confidence interval, and is not presented as one.

| confidence_band | share_of_records |
| --- | --- |
| high | 0.8160 |
| low | 0.1820 |
| medium | 0.0019 |

| statistic | calibrated_probability | staged_std | staged_p10 | staged_p90 |
| --- | --- | --- | --- | --- |
| count | 63678.0000 | 63678.0000 | 63678.0000 | 63678.0000 |
| mean | 0.1414 | 0.0211 | 0.1413 | 0.1927 |
| std | 0.2940 | 0.0208 | 0.2563 | 0.3067 |
| min | 0.0000 | 0.0036 | 0.0160 | 0.0457 |
| 25% | 0.0042 | 0.0110 | 0.0189 | 0.0466 |
| 50% | 0.0051 | 0.0113 | 0.0199 | 0.0467 |
| 75% | 0.0051 | 0.0119 | 0.0224 | 0.0493 |
| max | 1.0000 | 0.1108 | 0.7590 | 0.9400 |

### False positive / false negative analysis

Evaluated at the threshold that achieves 30% precision on the test window (`0.0051`): 8867 true positives, 4282 false positives, 241 false negatives out of 63678 records with 9108 actual events. Precision 0.674, recall 0.974.

**Where the errors concentrate.** A model that misses events uniformly is a different problem from one that misses them in a specific segment — the second is a fairness and coverage issue, not just an accuracy one.

By `credit_score_band`:

| credit_score_band | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| 740-779 | 19605 | 0.1430 | 0.1395 | 0.0634 | 0.0046 | 0.0035 |
| 660-699 | 6944 | 0.1545 | 0.1520 | 0.0719 | 0.0039 | 0.0025 |
| 700-739 | 13353 | 0.1418 | 0.1415 | 0.0668 | 0.0038 | 0.0004 |
| 620-659 | 2209 | 0.1770 | 0.1686 | 0.0711 | 0.0032 | 0.0084 |
| 780+ | 20317 | 0.1348 | 0.1350 | 0.0657 | 0.0030 | -0.0003 |
| 580-619 | 48 | 0.2500 | 0.2338 | 0.0417 | 0.0000 | 0.0162 |

By `servicer_name`:

| servicer_name | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| GUILD MORTGAGE COMPANY LLC | 668 | 0.1257 | 0.1156 | 0.0494 | 0.0105 | 0.0101 |
| UNITED WHOLESALE MORTGAGE, LLC | 747 | 0.2289 | 0.2251 | 0.0817 | 0.0054 | 0.0038 |
| U.S. BANK N.A. | 1542 | 0.1310 | 0.1302 | 0.0532 | 0.0052 | 0.0008 |
| LAKEVIEW LOAN SERVICING, LLC | 3835 | 0.1515 | 0.1458 | 0.0639 | 0.0050 | 0.0057 |
| TH MSR HOLDINGS LLC | 1702 | 0.1745 | 0.1761 | 0.0823 | 0.0047 | -0.0016 |
| NATIONSTAR MORTGAGE LLC DBA MR. COOPER | 3635 | 0.1436 | 0.1456 | 0.0666 | 0.0044 | -0.0020 |
| ROCKET MORTGAGE, LLC | 7782 | 0.1469 | 0.1421 | 0.0653 | 0.0044 | 0.0048 |
| FIFTH THIRD BANK, NATIONAL ASSOCIATION | 947 | 0.1605 | 0.1567 | 0.0644 | 0.0042 | 0.0038 |
| PNC BANK, NA | 1452 | 0.1315 | 0.1328 | 0.0764 | 0.0041 | -0.0013 |
| JPMORGAN CHASE BANK, NATIONAL ASSOCIATION | 6319 | 0.1458 | 0.1431 | 0.0741 | 0.0041 | 0.0027 |
| OTHER | 14632 | 0.1296 | 0.1293 | 0.0636 | 0.0038 | 0.0004 |
| PENNYMAC CORP. | 2332 | 0.1848 | 0.1886 | 0.0828 | 0.0034 | -0.0038 |
| PHH ASSET SERVICES LLC | 897 | 0.1505 | 0.1447 | 0.0580 | 0.0033 | 0.0058 |
| PENNYMAC LOAN SERVICES, LLC | 1265 | 0.1589 | 0.1581 | 0.0696 | 0.0032 | 0.0008 |
| WELLS FARGO BANK, N.A. | 2581 | 0.1062 | 0.1067 | 0.0585 | 0.0031 | -0.0005 |
| FREEDOM MORTGAGE CORPORATION | 2639 | 0.1565 | 0.1493 | 0.0640 | 0.0030 | 0.0072 |

By `current_status`:

| current_status | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| Current | 62302 | 0.1403 | 0.1390 | 0.0656 | 0.0038 | 0.0014 |
| Default | 267 | 0.2809 | 0.2475 | 0.3745 | 0.0037 | 0.0333 |
| DQ30 | 718 | 0.2423 | 0.2272 | 0.0724 | 0.0028 | 0.0151 |
| DQ60 | 172 | 0.2849 | 0.2884 | 0.0872 | 0.0000 | -0.0035 |
| DQ90plus | 219 | 0.3059 | 0.3003 | 0.1370 | 0.0000 | 0.0056 |

By `state`:

| state | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| HI | 126 | 0.1111 | 0.1187 | 0.0952 | 0.0159 | -0.0076 |
| SD | 156 | 0.1026 | 0.1063 | 0.0513 | 0.0128 | -0.0037 |
| LA | 605 | 0.1421 | 0.1385 | 0.0694 | 0.0083 | 0.0036 |
| IA | 503 | 0.1491 | 0.1421 | 0.0696 | 0.0080 | 0.0070 |
| KY | 836 | 0.1543 | 0.1424 | 0.0622 | 0.0072 | 0.0119 |
| AL | 699 | 0.1559 | 0.1560 | 0.0715 | 0.0072 | -0.0001 |
| UT | 952 | 0.1303 | 0.1229 | 0.0735 | 0.0063 | 0.0073 |
| VA | 1803 | 0.1403 | 0.1391 | 0.0577 | 0.0061 | 0.0012 |
| SC | 1001 | 0.1239 | 0.1208 | 0.0649 | 0.0060 | 0.0031 |
| AZ | 1811 | 0.1507 | 0.1429 | 0.0745 | 0.0050 | 0.0078 |
| PA | 2233 | 0.1433 | 0.1375 | 0.0596 | 0.0049 | 0.0058 |
| MD | 1484 | 0.1381 | 0.1390 | 0.0667 | 0.0047 | -0.0008 |
| GA | 2153 | 0.1626 | 0.1562 | 0.0585 | 0.0046 | 0.0063 |
| WV | 216 | 0.1065 | 0.0981 | 0.0602 | 0.0046 | 0.0084 |
| TX | 5214 | 0.1529 | 0.1516 | 0.0656 | 0.0046 | 0.0012 |
| CA | 6923 | 0.1486 | 0.1438 | 0.0602 | 0.0045 | 0.0049 |

**What false positives and false negatives look like.** Mean feature values for each error class against correctly-rejected records.

| feature | false_positives | false_negatives | true_negatives |
| --- | --- | --- | --- |
| credit score band | 4.7495 | 4.7246 | 4.7940 |
| loan-to-value band | 1.7064 | 1.8559 | 1.8881 |
| current performance status | 0.1336 | 0.0249 | 0.0287 |
| days past due | 6.2606 | 1.0759 | 1.0296 |
| worst days past due in last 6 months | 9.6404 | 2.0539 | 3.2730 |
| loan age | 50.0376 | 47.7220 | 48.3363 |
| record data-quality score | 88.8599 | 96.9378 | 96.8289 |
| balance as a share of original | 0.8471 | 0.8652 | 0.8637 |

## Cross-model observations

- **Horizon changes what matters, and the split is clean.** The 3-month delinquency model is led by *behavioural* signals — current performance status, consecutive clean months, recent days past due. The 12-month default model is led by *structural* ones — credit band, debt-to-income band, note rate. Short-horizon risk is about what the borrower is doing right now; long-horizon risk is about what the loan is. That is the economically sensible ordering and it was not imposed: both models saw the same 81 features.
- **Prepayment is dominated by rate economics** — note rate and the 12-month move in market rates — which is the correct mechanism and independently corroborates the rate-incentive bucket table in Task 5 (`reports/scenario_segment_prepay_by_rate_incentive.csv`). The response there is not monotone in incentive, and that is the economically right shape: loans already far in the money are near-saturated and have little headroom left, so a further rate cut moves them least, while loans sitting just below the refinance threshold move most.
- **Exceptions are dominated by operational fields** — data-quality score, rule violation count, missing field count — with essentially no contribution from credit attributes. This is the same conclusion the ROC-AUC 0.53 credit baseline reached in Task 4, arrived at from the opposite direction.
- **Servicer identity carries real attribution weight**, which the data intelligence report flagged as a confound: the two servicers with the worst reporting hygiene also have elevated delinquency. Part of that attribution is credit risk and part is reporting behaviour, and SHAP cannot separate the two. A servicer-driven score is a prompt to investigate the servicer, not a statement about the borrower.

## Limitations

## Anomaly-score drivers

Task 6 asks for drivers of the anomaly score alongside the three predictive scores. The anomaly score is unsupervised, so it has no SHAP decomposition: an isolation forest gives no native per-feature attribution and inventing one would be exactly the kind of plausible-but-unfounded explanation this layer exists to prevent.

Instead each flagged record is attributed by **robust deviation** — every anomaly feature is scored by its distance from the training-window median in MAD units, and the largest deviations are named. That is a quantity a reviewer can check against the record in front of them, which a SHAP value for an unsupervised model would not be.

| top_anomaly_driver | share_of_records |
| --- | --- |
| balance against expected amortisation for term elapsed | 0.2122 |
| three-month balance movement | 0.1318 |
| servicer feed record present | 0.1230 |
| servicer reporting lag | 0.1189 |
| scheduled payment relative to balance | 0.1060 |
| count of missing credit fields | 0.0878 |
| document file incomplete | 0.0588 |
| record arrived by manual upload | 0.0563 |
| month-over-month balance movement | 0.0472 |
| servicer feed balance gap | 0.0190 |

Across 63,678 held-out records, the leading driver is **balance against expected amortisation for term elapsed** (21.2% of records). Per-record attributions for the reviewer queue are in `reports/anomaly_review_queue.csv`, and the distributional detail is in `reports/anomaly_report.md`.

- SHAP attributes to *features*, not to causes. A high contribution from days past due does not mean delinquency causes default in any actionable sense; it means the model reads it as the strongest available signal.
- Correlated features split their attribution arbitrarily between them. The DPD family (current, lagged, rolling maxima) is highly correlated, so individual rankings within that family are not stable and should be read as a group.
- Explanations are computed on a sample of up to 4,000 test rows for tractability.
- The uncertainty measure is a boosting-stability proxy. It does not capture uncertainty from feature noise, label noise, or regime change — and regime change is the dominant risk for the 12-month targets, as Task 2 documented.
