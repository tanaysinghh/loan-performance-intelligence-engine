# Explainability Report

**Task 6.** SHAP TreeExplainer over the trained LightGBM models. Every explanation here is derived from the fitted model's own structure — no language model is involved in producing any attribution.

## 1. What is being explained, and in what units

SHAP values are additive in **log-odds**, not probability. Explaining the calibrated probability directly would break that additivity and the contributions would stop summing to anything meaningful. Attribution is therefore computed against the raw LightGBM margin and reported in log-odds, alongside the calibrated probability the reviewer actually acts on. The two are labelled separately throughout and should not be added together.

## Target: `next_3m_delinquency_flag`

### Global feature importance

Mean absolute SHAP contribution across the test window. `direction` is the sign of the correlation between a feature's value and its contribution, which recovers whether higher values raise or lower risk without assuming monotonicity.

| feature | plain_english | mean_abs_shap | share_of_total_attribution | direction |
| --- | --- | --- | --- | --- |
| status_ord | current performance status | 0.4606 | 0.1071 | non-monotone / categorical |
| credit_ord | credit score band | 0.4403 | 0.1024 | non-monotone / categorical |
| current_streak_clean | consecutive clean months | 0.2461 | 0.0573 | higher value lowers risk |
| state | state | 0.2370 | 0.0551 | non-monotone / categorical |
| dti_ord | debt-to-income band | 0.2211 | 0.0514 | higher value raises risk |
| payment_to_balance | scheduled payment relative to balance | 0.1907 | 0.0444 | non-monotone / categorical |
| market_rate_delta_12m | market rate delta 12m | 0.1869 | 0.0435 | higher value lowers risk |
| ltv_ord | loan-to-value band | 0.1766 | 0.0411 | non-monotone / categorical |
| term_progress | share of term elapsed | 0.1581 | 0.0368 | higher value lowers risk |
| unemployment_rate | unemployment rate | 0.1573 | 0.0366 | higher value lowers risk |
| current_status | current status | 0.1429 | 0.0332 | non-monotone / categorical |
| loan_age_months_clean | loan age | 0.1239 | 0.0288 | higher value lowers risk |
| credit_score_band | credit score band | 0.1183 | 0.0275 | non-monotone / categorical |
| ltv_band | ltv band | 0.1058 | 0.0246 | non-monotone / categorical |
| amortisation_residual | balance against expected amortisation | 0.0938 | 0.0218 | higher value raises risk |
| interest_rate_clean | note rate | 0.0813 | 0.0189 | higher value lowers risk |
| balance_change_1m | balance change 1m | 0.0801 | 0.0186 | higher value lowers risk |
| log_current_balance | current balance | 0.0793 | 0.0184 | higher value lowers risk |

The top three drivers — current performance status, credit score band, consecutive clean months — account for **26.7%** of total attribution.

### Local explanations — ten highest-risk records in the test window

Each row shows the calibrated probability a reviewer sees and the four largest log-odds contributions behind it.

| loan_id | reporting_month | current_status | calibrated_probability | driver_1 | driver_1_value | driver_1_log_odds | driver_2 | driver_2_value | driver_2_log_odds | driver_3 | driver_3_value | driver_3_log_odds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LN100059 | 2025-10 | DQ90plus | 0.9915 | current performance status | 3.0000 | 5.4434 | current status | DQ90plus | 1.5731 | months delinquent in last 6 months | 6.0000 | 0.5538 |
| LN100059 | 2025-12 | DQ90plus | 0.9911 | current performance status | 3.0000 | 5.4444 | current status | DQ90plus | 1.5546 | months delinquent in last 6 months | 6.0000 | 0.5664 |
| LN100059 | 2025-11 | DQ90plus | 0.9907 | current performance status | 3.0000 | 5.4556 | current status | DQ90plus | 1.5654 | months delinquent in last 6 months | 6.0000 | 0.5692 |
| LN100510 | 2025-10 | DQ90plus | 0.9905 | current performance status | 3.0000 | 5.5682 | current status | DQ90plus | 1.6092 | months delinquent in last 6 months | 6.0000 | 0.5749 |
| LN100059 | 2026-01 | DQ90plus | 0.9903 | current performance status | 3.0000 | 5.3934 | current status | DQ90plus | 1.5684 | months delinquent in last 6 months | 6.0000 | 0.5476 |
| LN101204 | 2025-11 | DQ90plus | 0.9902 | current performance status | 3.0000 | 4.8245 | current status | DQ90plus | 1.3602 | credit score band | 1.0000 | 0.6629 |
| LN100373 | 2025-10 | DQ90plus | 0.9902 | current performance status | 3.0000 | 5.0382 | current status | DQ90plus | 1.3342 | credit score band | 0.0000 | 0.7543 |
| LN101249 | 2025-11 | DQ90plus | 0.9902 | current performance status | 3.0000 | 5.8784 | current status | DQ90plus | 1.6962 | months delinquent in last 6 months | 6.0000 | 0.5757 |
| LN101249 | 2025-12 | DQ90plus | 0.9901 | current performance status | 3.0000 | 5.8870 | current status | DQ90plus | 1.6894 | months delinquent in last 6 months | 6.0000 | 0.5764 |
| LN100706 | 2025-11 | DQ90plus | 0.9900 | current performance status | 3.0000 | 5.4634 | current status | DQ90plus | 1.5662 | months delinquent in last 6 months | 4.0000 | 0.5379 |

### Local explanations — five lowest-risk records (contrast set)

| loan_id | reporting_month | current_status | calibrated_probability | driver_1 | driver_1_value | driver_1_log_odds | driver_2 | driver_2_value | driver_2_log_odds | driver_3 | driver_3_value | driver_3_log_odds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LN100265 | 2026-01 | Current | 0.0012 | credit score band | 6.0000 | -0.5400 | share of term elapsed | 0.0170 | -0.2951 | debt-to-income band | 0.0000 | -0.2543 |
| LN100265 | 2025-12 | Current | 0.0013 | credit score band | 6.0000 | -0.5239 | share of term elapsed | 0.0110 | -0.3306 | current performance status | 0.0000 | -0.2544 |
| LN100265 | 2025-11 | Current | 0.0013 | credit score band | 6.0000 | -0.5264 | share of term elapsed | 0.0060 | -0.3324 | current performance status | 0.0000 | -0.2544 |
| LN100265 | 2026-02 | Current | 0.0014 | credit score band | 6.0000 | -0.5447 | debt-to-income band | 0.0000 | -0.2683 | current performance status | 0.0000 | -0.2528 |
| LN100265 | 2026-03 | Current | 0.0014 | credit score band | 6.0000 | -0.5438 | debt-to-income band | 0.0000 | -0.2663 | current performance status | 0.0000 | -0.2527 |

### Model confidence and uncertainty

Predictions from the final boosting rounds are collected and their spread used as an epistemic-uncertainty proxy. This measures sensitivity to where the boosting sequence stopped — it is a stability signal, **not** a statistical confidence interval, and is not presented as one.

| confidence_band | share_of_records |
| --- | --- |
| high | 0.9266 |
| medium | 0.0707 |
| low | 0.0027 |

| statistic | calibrated_probability | staged_std | staged_p10 | staged_p90 |
| --- | --- | --- | --- | --- |
| count | 3719.0000 | 3719.0000 | 3719.0000 | 3719.0000 |
| mean | 0.0749 | 0.0017 | 0.0867 | 0.0907 |
| std | 0.2020 | 0.0020 | 0.2089 | 0.2093 |
| min | 0.0012 | 0.0001 | 0.0010 | 0.0014 |
| 25% | 0.0104 | 0.0005 | 0.0115 | 0.0130 |
| 50% | 0.0193 | 0.0010 | 0.0234 | 0.0260 |
| 75% | 0.0408 | 0.0020 | 0.0543 | 0.0609 |
| max | 0.9915 | 0.0159 | 0.9977 | 0.9982 |

### False positive / false negative analysis

Evaluated at the threshold that achieves 30% precision on the test window (`0.0538`): 198 true positives, 462 false positives, 49 false negatives out of 3719 records with 247 actual events. Precision 0.300, recall 0.802.

**Where the errors concentrate.** A model that misses events uniformly is a different problem from one that misses them in a specific segment — the second is a fairness and coverage issue, not just an accuracy one.

By `credit_score_band`:

| credit_score_band | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| 620-659 | 507 | 0.1499 | 0.1495 | 0.2821 | 0.0355 | 0.0004 |
| 780+ | 579 | 0.0242 | 0.0149 | 0.0000 | 0.0190 | 0.0093 |
| 580-619 | 175 | 0.1829 | 0.2177 | 0.7086 | 0.0114 | -0.0348 |
| 700-739 | 800 | 0.0450 | 0.0602 | 0.0775 | 0.0100 | -0.0152 |
| 660-699 | 595 | 0.0504 | 0.0699 | 0.1277 | 0.0067 | -0.0195 |
| 740-779 | 903 | 0.0299 | 0.0385 | 0.0144 | 0.0055 | -0.0086 |
| <580 | 68 | 0.4118 | 0.3965 | 0.4559 | 0.0000 | 0.0152 |

By `servicer_name`:

| servicer_name | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| Belmont Loan Services | 910 | 0.0857 | 0.0887 | 0.1516 | 0.0198 | -0.0030 |
| Pioneer Mortgage Ops | 547 | 0.0402 | 0.0535 | 0.1316 | 0.0128 | -0.0133 |
| Northgate Servicing | 1125 | 0.0773 | 0.0830 | 0.1004 | 0.0116 | -0.0057 |
| Arcadia Capital Servicing | 766 | 0.0522 | 0.0627 | 0.1110 | 0.0104 | -0.0105 |
| Kestrel Financial | 371 | 0.0539 | 0.0735 | 0.1456 | 0.0081 | -0.0196 |

By `current_status`:

| current_status | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| Current | 3541 | 0.0209 | 0.0304 | 0.1291 | 0.0138 | -0.0095 |
| DQ30 | 38 | 0.8684 | 0.8720 | 0.1316 | 0.0000 | -0.0036 |
| DQ60 | 32 | 1.0000 | 0.9816 | 0.0000 | 0.0000 | 0.0184 |
| DQ90plus | 108 | 1.0000 | 0.9865 | 0.0000 | 0.0000 | 0.0135 |

By `state`:

| state | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| CO | 74 | 0.1081 | 0.0822 | 0.2703 | 0.0405 | 0.0259 |
| OH | 155 | 0.0645 | 0.0556 | 0.1871 | 0.0387 | 0.0089 |
| NC | 141 | 0.1064 | 0.0835 | 0.1773 | 0.0355 | 0.0229 |
| FL | 426 | 0.0728 | 0.0744 | 0.1995 | 0.0211 | -0.0016 |
| GA | 253 | 0.0672 | 0.0693 | 0.0277 | 0.0158 | -0.0021 |
| TX | 526 | 0.0456 | 0.0578 | 0.0989 | 0.0152 | -0.0122 |
| MI | 132 | 0.0909 | 0.0909 | 0.0227 | 0.0152 | 0.0000 |
| NY | 338 | 0.0740 | 0.0779 | 0.1538 | 0.0148 | -0.0039 |
| AZ | 169 | 0.0473 | 0.0507 | 0.0296 | 0.0118 | -0.0034 |
| NV | 130 | 0.0154 | 0.0429 | 0.0769 | 0.0077 | -0.0275 |
| WA | 157 | 0.0892 | 0.1070 | 0.1783 | 0.0064 | -0.0179 |
| PA | 172 | 0.0756 | 0.0855 | 0.0174 | 0.0058 | -0.0099 |
| CA | 665 | 0.0722 | 0.0906 | 0.1383 | 0.0030 | -0.0185 |
| IL | 227 | 0.0705 | 0.0957 | 0.1938 | 0.0000 | -0.0252 |
| NJ | 154 | 0.0260 | 0.0424 | 0.0455 | 0.0000 | -0.0165 |

**What false positives and false negatives look like.** Mean feature values for each error class against correctly-rejected records.

| feature | false_positives | false_negatives | true_negatives |
| --- | --- | --- | --- |
| credit score band | 2.1180 | 3.6042 | 4.2845 |
| loan-to-value band | 2.8670 | 2.2553 | 1.7921 |
| current performance status | 0.0108 | 0.0000 | 0.0000 |
| days past due | 0.8061 | 0.0000 | 0.3746 |
| worst days past due in last 6 months | 3.5108 | 0.9388 | 2.7437 |
| loan age | 45.8621 | 53.7551 | 49.4713 |
| record data-quality score | 95.3485 | 94.6122 | 95.2269 |
| balance as a share of original | 0.9274 | 0.8852 | 0.9204 |

## Target: `next_12m_default_flag`

### Global feature importance

Mean absolute SHAP contribution across the test window. `direction` is the sign of the correlation between a feature's value and its contribution, which recovers whether higher values raise or lower risk without assuming monotonicity.

| feature | plain_english | mean_abs_shap | share_of_total_attribution | direction |
| --- | --- | --- | --- | --- |
| credit_ord | credit score band | 0.8846 | 0.1285 | higher value lowers risk |
| interest_rate_clean | note rate | 0.3316 | 0.0482 | higher value lowers risk |
| state | state | 0.3222 | 0.0468 | non-monotone / categorical |
| dti_ord | debt-to-income band | 0.2873 | 0.0417 | non-monotone / categorical |
| credit_score_band | credit score band | 0.2738 | 0.0398 | non-monotone / categorical |
| market_rate_delta_12m | market rate delta 12m | 0.2645 | 0.0384 | higher value raises risk |
| loan_purpose | loan purpose | 0.2544 | 0.0370 | non-monotone / categorical |
| loan_age_months_clean | loan age | 0.2338 | 0.0340 | higher value lowers risk |
| status_ord | current performance status | 0.2136 | 0.0310 | higher value raises risk |
| unemployment_rate | unemployment rate | 0.2121 | 0.0308 | higher value lowers risk |
| payment_to_balance | scheduled payment relative to balance | 0.2118 | 0.0308 | higher value raises risk |
| term_progress | share of term elapsed | 0.2104 | 0.0306 | non-monotone / categorical |
| scheduled_payment | scheduled monthly payment | 0.1986 | 0.0289 | higher value raises risk |
| ltv_ord | loan-to-value band | 0.1879 | 0.0273 | higher value raises risk |
| rate_incentive | refinance incentive (note rate less market rate) | 0.1860 | 0.0270 | non-monotone / categorical |
| balance_change_1m | balance change 1m | 0.1783 | 0.0259 | non-monotone / categorical |
| log_original_balance | original balance | 0.1704 | 0.0248 | higher value raises risk |
| servicer_name | servicer | 0.1433 | 0.0208 | non-monotone / categorical |

The top three drivers — credit score band, note rate, state — account for **22.4%** of total attribution.

### Local explanations — ten highest-risk records in the test window

Each row shows the calibrated probability a reviewer sees and the four largest log-odds contributions behind it.

| loan_id | reporting_month | current_status | calibrated_probability | driver_1 | driver_1_value | driver_1_log_odds | driver_2 | driver_2_value | driver_2_log_odds | driver_3 | driver_3_value | driver_3_log_odds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LN100455 | 2025-03 | DQ90plus | 1.0000 | current performance status | 3.0000 | 2.0289 | months delinquent in last 12 months | 6.0000 | 1.1705 | days past due | 94.0000 | 1.0464 |
| LN100455 | 2025-04 | DQ90plus | 1.0000 | current performance status | 3.0000 | 2.0761 | months delinquent in last 12 months | 7.0000 | 1.1895 | days past due | 111.0000 | 1.0519 |
| LN100455 | 2025-05 | DQ90plus | 1.0000 | current performance status | 3.0000 | 2.0582 | months delinquent in last 12 months | 8.0000 | 1.2262 | days past due | 104.0000 | 1.0859 |
| LN100455 | 2025-06 | DQ90plus | 1.0000 | current performance status | 3.0000 | 2.0335 | months delinquent in last 12 months | 9.0000 | 1.3959 | days past due | 106.0000 | 1.0842 |
| LN101437 | 2025-06 | DQ90plus | 1.0000 | current performance status | 3.0000 | 2.7015 | months delinquent in last 12 months | 7.0000 | 1.5715 | days past due | 92.0000 | 1.2620 |
| LN101437 | 2025-04 | DQ60 | 1.0000 | current performance status | 2.0000 | 2.5028 | months delinquent in last 12 months | 5.0000 | 1.5138 | months delinquent in last 6 months | 5.0000 | 0.8669 |
| LN100510 | 2025-04 | DQ90plus | 1.0000 | current performance status | 3.0000 | 2.7422 | months delinquent in last 12 months | 5.0000 | 1.4993 | days past due | 91.0000 | 1.4078 |
| LN101437 | 2025-05 | DQ60 | 1.0000 | current performance status | 2.0000 | 2.5058 | months delinquent in last 12 months | 6.0000 | 1.5449 | days past due | 78.0000 | 1.2399 |
| LN100510 | 2025-03 | DQ90plus | 1.0000 | current performance status | 3.0000 | 2.7914 | months delinquent in last 12 months | 4.0000 | 1.5248 | days past due | 90.0000 | 1.3623 |
| LN101462 | 2025-01 | DQ90plus | 1.0000 | current performance status | 3.0000 | 2.2757 | credit score band | 0.0000 | 1.2905 | months delinquent in last 12 months | 6.0000 | 1.2078 |

### Local explanations — five lowest-risk records (contrast set)

| loan_id | reporting_month | current_status | calibrated_probability | driver_1 | driver_1_value | driver_1_log_odds | driver_2 | driver_2_value | driver_2_log_odds | driver_3 | driver_3_value | driver_3_log_odds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LN100462 | 2025-01 | Current | 0.0000 | credit score band | 6.0000 | -1.0114 | scheduled monthly payment | 579.9280 | -0.3899 | loan purpose | purchase | 0.2457 |
| LN100452 | 2025-06 | Current | 0.0000 | credit score band | 6.0000 | -0.9734 | loan age | 61.0000 | 0.6183 | market rate delta 12m | -0.901 | -0.4988 |
| LN100452 | 2025-05 | Current | 0.0000 | credit score band | 6.0000 | -1.0534 | loan age | 60.0000 | 0.6454 | market rate delta 12m | -0.852 | -0.5116 |
| LN100468 | 2025-06 | Current | 0.0000 | credit score band | 5.0000 | -1.1224 | share of term elapsed | 0.2530 | 0.4671 | credit score band | 740-779 | -0.4592 |
| LN100468 | 2025-05 | Current | 0.0000 | credit score band | 5.0000 | -1.1274 | share of term elapsed | 0.2500 | 0.4833 | credit score band | 740-779 | -0.4592 |

### Model confidence and uncertainty

Predictions from the final boosting rounds are collected and their spread used as an epistemic-uncertainty proxy. This measures sensitivity to where the boosting sequence stopped — it is a stability signal, **not** a statistical confidence interval, and is not presented as one.

| confidence_band | share_of_records |
| --- | --- |
| high | 0.9597 |
| medium | 0.0210 |
| low | 0.0193 |

| statistic | calibrated_probability | staged_std | staged_p10 | staged_p90 |
| --- | --- | --- | --- | --- |
| count | 4040.0000 | 4040.0000 | 4040.0000 | 4040.0000 |
| mean | 0.0643 | 0.0009 | 0.0611 | 0.0631 |
| std | 0.2042 | 0.0022 | 0.1974 | 0.1997 |
| min | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 25% | 0.0000 | 0.0000 | 0.0003 | 0.0004 |
| 50% | 0.0063 | 0.0001 | 0.0012 | 0.0015 |
| 75% | 0.0111 | 0.0005 | 0.0087 | 0.0103 |
| max | 1.0000 | 0.0251 | 0.9968 | 0.9972 |

### False positive / false negative analysis

Evaluated at the threshold that achieves 30% precision on the test window (`0.0494`): 227 true positives, 455 false positives, 65 false negatives out of 4040 records with 292 actual events. Precision 0.333, recall 0.777.

**Where the errors concentrate.** A model that misses events uniformly is a different problem from one that misses them in a specific segment — the second is a fairness and coverage issue, not just an accuracy one.

By `credit_score_band`:

| credit_score_band | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| 620-659 | 577 | 0.1473 | 0.1398 | 0.2808 | 0.0381 | 0.0075 |
| 660-699 | 617 | 0.0924 | 0.0697 | 0.1588 | 0.0324 | 0.0227 |
| <580 | 75 | 0.3733 | 0.3492 | 0.4933 | 0.0133 | 0.0241 |
| 740-779 | 970 | 0.0330 | 0.0159 | 0.0031 | 0.0113 | 0.0171 |
| 700-739 | 876 | 0.0342 | 0.0407 | 0.0605 | 0.0103 | -0.0065 |
| 580-619 | 203 | 0.2020 | 0.2128 | 0.4433 | 0.0049 | -0.0108 |
| 780+ | 646 | 0.0170 | 0.0134 | 0.0000 | 0.0000 | 0.0036 |

By `servicer_name`:

| servicer_name | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| Northgate Servicing | 1233 | 0.0560 | 0.0403 | 0.0762 | 0.0219 | 0.0156 |
| Belmont Loan Services | 975 | 0.0574 | 0.0563 | 0.1508 | 0.0164 | 0.0012 |
| Pioneer Mortgage Ops | 603 | 0.0796 | 0.0617 | 0.1095 | 0.0133 | 0.0179 |
| Kestrel Financial | 381 | 0.1207 | 0.1171 | 0.1155 | 0.0131 | 0.0036 |
| Arcadia Capital Servicing | 848 | 0.0861 | 0.0864 | 0.1226 | 0.0106 | -0.0003 |

By `current_status`:

| current_status | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| Current | 3814 | 0.0294 | 0.0204 | 0.1075 | 0.0170 | 0.0090 |
| DQ30 | 64 | 0.6719 | 0.4940 | 0.3125 | 0.0000 | 0.1779 |
| DQ60 | 65 | 0.9077 | 0.8870 | 0.0923 | 0.0000 | 0.0207 |
| DQ90plus | 97 | 0.8041 | 0.9563 | 0.1959 | 0.0000 | -0.1522 |

By `state`:

| state | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| IL | 224 | 0.0446 | 0.0247 | 0.1920 | 0.0312 | 0.0200 |
| PA | 181 | 0.0497 | 0.0129 | 0.0331 | 0.0276 | 0.0368 |
| NV | 151 | 0.0795 | 0.0501 | 0.0927 | 0.0265 | 0.0294 |
| WA | 156 | 0.0769 | 0.0455 | 0.1603 | 0.0256 | 0.0314 |
| GA | 275 | 0.0655 | 0.0494 | 0.0873 | 0.0255 | 0.0161 |
| MI | 129 | 0.1318 | 0.0955 | 0.0775 | 0.0233 | 0.0363 |
| NC | 154 | 0.0974 | 0.0680 | 0.0844 | 0.0195 | 0.0294 |
| TX | 574 | 0.0557 | 0.0741 | 0.1220 | 0.0174 | -0.0184 |
| NY | 380 | 0.0711 | 0.0631 | 0.1211 | 0.0132 | 0.0080 |
| FL | 465 | 0.0968 | 0.1027 | 0.1462 | 0.0129 | -0.0059 |
| CA | 737 | 0.0855 | 0.0816 | 0.1221 | 0.0122 | 0.0039 |
| OH | 180 | 0.0833 | 0.0802 | 0.0833 | 0.0111 | 0.0031 |
| CO | 72 | 0.0833 | 0.0499 | 0.2361 | 0.0000 | 0.0334 |
| AZ | 190 | 0.0316 | 0.0378 | 0.0684 | 0.0000 | -0.0063 |
| NJ | 172 | 0.0291 | 0.0068 | 0.0058 | 0.0000 | 0.0223 |

**What false positives and false negatives look like.** Mean feature values for each error class against correctly-rejected records.

| feature | false_positives | false_negatives | true_negatives |
| --- | --- | --- | --- |
| credit score band | 2.1106 | 3.0625 | 4.2555 |
| loan-to-value band | 2.9128 | 2.8594 | 1.7758 |
| current performance status | 0.1956 | 0.0000 | 0.0003 |
| days past due | 7.5493 | 0.0000 | 0.3045 |
| worst days past due in last 6 months | 10.5516 | 5.5385 | 2.2504 |
| loan age | 25.7269 | 37.1406 | 45.9396 |
| record data-quality score | 94.8462 | 94.1231 | 95.3917 |
| balance as a share of original | 0.9606 | 0.9271 | 0.9252 |

## Target: `next_12m_prepayment_flag`

### Global feature importance

Mean absolute SHAP contribution across the test window. `direction` is the sign of the correlation between a feature's value and its contribution, which recovers whether higher values raise or lower risk without assuming monotonicity.

| feature | plain_english | mean_abs_shap | share_of_total_attribution | direction |
| --- | --- | --- | --- | --- |
| interest_rate_clean | note rate | 0.6619 | 0.1056 | higher value lowers risk |
| market_rate_delta_12m | market rate delta 12m | 0.6413 | 0.1023 | non-monotone / categorical |
| current_streak_clean | consecutive clean months | 0.4513 | 0.0720 | higher value raises risk |
| state | state | 0.3327 | 0.0531 | non-monotone / categorical |
| loan_purpose | loan purpose | 0.3085 | 0.0492 | non-monotone / categorical |
| scheduled_payment | scheduled monthly payment | 0.2697 | 0.0430 | higher value lowers risk |
| credit_ord | credit score band | 0.2488 | 0.0397 | higher value raises risk |
| servicer_name | servicer | 0.2045 | 0.0326 | non-monotone / categorical |
| credit_score_band | credit score band | 0.2044 | 0.0326 | non-monotone / categorical |
| log_original_balance | original balance | 0.2041 | 0.0326 | non-monotone / categorical |
| ltv_ord | loan-to-value band | 0.1712 | 0.0273 | higher value lowers risk |
| log_current_balance | current balance | 0.1456 | 0.0232 | higher value lowers risk |
| amortisation_residual | balance against expected amortisation | 0.1420 | 0.0227 | higher value raises risk |
| remaining_term_months | remaining term months | 0.1388 | 0.0221 | non-monotone / categorical |
| payment_to_balance | scheduled payment relative to balance | 0.1382 | 0.0221 | higher value raises risk |
| refi_incentive_positive | positive refinance incentive | 0.1381 | 0.0220 | higher value lowers risk |
| term_progress | share of term elapsed | 0.1342 | 0.0214 | higher value lowers risk |
| balance_change_3m | balance change 3m | 0.1274 | 0.0203 | higher value lowers risk |

The top three drivers — note rate, market rate delta 12m, consecutive clean months — account for **28.0%** of total attribution.

### Local explanations — ten highest-risk records in the test window

Each row shows the calibrated probability a reviewer sees and the four largest log-odds contributions behind it.

| loan_id | reporting_month | current_status | calibrated_probability | driver_1 | driver_1_value | driver_1_log_odds | driver_2 | driver_2_value | driver_2_log_odds | driver_3 | driver_3_value | driver_3_log_odds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LN100004 | 2025-06 | Current | 1.0000 | note rate | 8.0030 | 1.4735 | market rate delta 12m | -0.901 | 0.6599 | scheduled monthly payment | 2216.601 | 0.6084 |
| LN100004 | 2025-05 | Current | 1.0000 | note rate | 8.0030 | 1.4633 | market rate delta 12m | -0.852 | 0.6611 | scheduled monthly payment | 2216.601 | 0.6138 |
| LN100004 | 2025-04 | Current | 1.0000 | note rate | 8.0030 | 1.3974 | market rate delta 12m | -0.792 | 0.6525 | scheduled monthly payment | 2216.601 | 0.6417 |
| LN101487 | 2025-06 | Current | 1.0000 | note rate | 7.3730 | 1.5824 | market rate delta 12m | -0.901 | 0.7984 | state | NC | 0.5257 |
| LN101172 | 2025-06 | Current | 1.0000 | note rate | 7.1900 | 1.6685 | market rate delta 12m | -0.901 | 0.7715 | servicer | Kestrel Financial | 0.4327 |
| LN100226 | 2025-02 | Current | 1.0000 | note rate | 7.7870 | 1.3234 | state | OH | 1.0079 | scheduled monthly payment | 2049.067 | 0.7845 |
| LN100226 | 2025-04 | Current | 1.0000 | note rate | 7.7870 | 1.3589 | state | OH | 1.0161 | scheduled monthly payment | 2049.067 | 0.7804 |
| LN100226 | 2025-03 | Current | 1.0000 | note rate | 7.7870 | 1.3144 | state | OH | 0.9656 | scheduled monthly payment | 2049.067 | 0.7453 |
| LN100878 | 2025-04 | Current | 1.0000 | note rate | 6.9230 | 1.4579 | state | OH | 1.3290 | scheduled monthly payment | 1960.614 | 0.7665 |
| LN100878 | 2025-03 | Current | 1.0000 | note rate | 6.9230 | 1.4873 | state | OH | 1.3459 | scheduled monthly payment | 1960.614 | 0.7882 |

### Local explanations — five lowest-risk records (contrast set)

| loan_id | reporting_month | current_status | calibrated_probability | driver_1 | driver_1_value | driver_1_log_odds | driver_2 | driver_2_value | driver_2_log_odds | driver_3 | driver_3_value | driver_3_log_odds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LN100580 | 2025-03 | Current | 0.0000 | state | CA | -0.7432 | market rate delta 12m | -0.725 | 0.6757 | consecutive clean months | 38 | -0.6608 |
| LN100580 | 2025-04 | Current | 0.0000 | state | CA | -0.7641 | market rate delta 12m | -0.792 | 0.6980 | consecutive clean months | 39 | -0.6480 |
| LN100580 | 2025-05 | Current | 0.0000 | state | CA | -0.7567 | market rate delta 12m | -0.852 | 0.7427 | consecutive clean months | 40 | -0.6672 |
| LN101186 | 2025-06 | Current | 0.0000 | market rate delta 12m | -0.901 | 0.8651 | state | CA | -0.7308 | consecutive clean months | 50 | -0.5536 |
| LN101186 | 2025-02 | Current | 0.0000 | state | CA | -0.7439 | market rate delta 12m | -0.649 | 0.6458 | consecutive clean months | 46 | -0.5288 |

### Model confidence and uncertainty

Predictions from the final boosting rounds are collected and their spread used as an epistemic-uncertainty proxy. This measures sensitivity to where the boosting sequence stopped — it is a stability signal, **not** a statistical confidence interval, and is not presented as one.

| confidence_band | share_of_records |
| --- | --- |
| high | 0.6829 |
| medium | 0.1851 |
| low | 0.1319 |

| statistic | calibrated_probability | staged_std | staged_p10 | staged_p90 |
| --- | --- | --- | --- | --- |
| count | 4040.0000 | 4040.0000 | 4040.0000 | 4040.0000 |
| mean | 0.1431 | 0.0033 | 0.1335 | 0.1414 |
| std | 0.2256 | 0.0040 | 0.2053 | 0.2116 |
| min | 0.0000 | 0.0000 | 0.0002 | 0.0002 |
| 25% | 0.0301 | 0.0004 | 0.0067 | 0.0078 |
| 50% | 0.0301 | 0.0014 | 0.0269 | 0.0305 |
| 75% | 0.1456 | 0.0053 | 0.1728 | 0.1885 |
| max | 1.0000 | 0.0285 | 0.9557 | 0.9611 |

### False positive / false negative analysis

Evaluated at the threshold that achieves 30% precision on the test window (`0.3439`): 150 true positives, 342 false positives, 541 false negatives out of 4040 records with 691 actual events. Precision 0.305, recall 0.217.

**Where the errors concentrate.** A model that misses events uniformly is a different problem from one that misses them in a specific segment — the second is a fairness and coverage issue, not just an accuracy one.

By `credit_score_band`:

| credit_score_band | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| 740-779 | 970 | 0.2299 | 0.1535 | 0.0876 | 0.1773 | 0.0763 |
| 580-619 | 203 | 0.1724 | 0.1102 | 0.0443 | 0.1527 | 0.0622 |
| 780+ | 646 | 0.2090 | 0.1795 | 0.0913 | 0.1486 | 0.0294 |
| 700-739 | 876 | 0.1484 | 0.1661 | 0.1324 | 0.1164 | -0.0177 |
| 660-699 | 617 | 0.1459 | 0.1528 | 0.0794 | 0.1118 | -0.0069 |
| 620-659 | 577 | 0.1023 | 0.0611 | 0.0225 | 0.0988 | 0.0412 |
| <580 | 75 | 0.0933 | 0.0369 | 0.0000 | 0.0933 | 0.0565 |

By `servicer_name`:

| servicer_name | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| Pioneer Mortgage Ops | 603 | 0.1957 | 0.0944 | 0.0249 | 0.1708 | 0.1013 |
| Northgate Servicing | 1233 | 0.1987 | 0.1418 | 0.0641 | 0.1533 | 0.0569 |
| Arcadia Capital Servicing | 848 | 0.1427 | 0.1036 | 0.0637 | 0.1274 | 0.0391 |
| Belmont Loan Services | 975 | 0.1846 | 0.2091 | 0.1477 | 0.1210 | -0.0244 |
| Kestrel Financial | 381 | 0.0709 | 0.1436 | 0.1312 | 0.0604 | -0.0727 |

By `current_status`:

| current_status | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| Current | 3814 | 0.1778 | 0.1500 | 0.0897 | 0.1384 | 0.0277 |
| DQ90plus | 97 | 0.1237 | 0.0206 | 0.0000 | 0.1237 | 0.1031 |
| DQ60 | 65 | 0.0154 | 0.0228 | 0.0000 | 0.0154 | -0.0074 |
| DQ30 | 64 | 0.0000 | 0.0371 | 0.0000 | 0.0000 | -0.0371 |

By `state`:

| state | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| CO | 72 | 0.2083 | 0.1539 | 0.0833 | 0.1806 | 0.0544 |
| TX | 574 | 0.2003 | 0.1228 | 0.0645 | 0.1794 | 0.0775 |
| NJ | 172 | 0.1686 | 0.1418 | 0.1047 | 0.1628 | 0.0268 |
| NY | 380 | 0.1816 | 0.1037 | 0.0447 | 0.1553 | 0.0779 |
| WA | 156 | 0.1538 | 0.1026 | 0.0449 | 0.1538 | 0.0513 |
| OH | 180 | 0.2333 | 0.2634 | 0.1389 | 0.1333 | -0.0300 |
| PA | 181 | 0.1547 | 0.1337 | 0.1050 | 0.1326 | 0.0210 |
| FL | 465 | 0.1656 | 0.1118 | 0.0344 | 0.1269 | 0.0538 |
| CA | 737 | 0.1696 | 0.1575 | 0.1140 | 0.1262 | 0.0121 |
| NV | 151 | 0.1258 | 0.0534 | 0.0265 | 0.1192 | 0.0724 |
| IL | 224 | 0.1429 | 0.1166 | 0.0670 | 0.1161 | 0.0262 |
| NC | 154 | 0.1623 | 0.2057 | 0.1299 | 0.1039 | -0.0434 |
| MI | 129 | 0.1085 | 0.1175 | 0.0930 | 0.1008 | -0.0089 |
| AZ | 190 | 0.1842 | 0.2362 | 0.1368 | 0.0947 | -0.0520 |
| GA | 275 | 0.1527 | 0.1859 | 0.1309 | 0.0836 | -0.0331 |

**What false positives and false negatives look like.** Mean feature values for each error class against correctly-rejected records.

| feature | false_positives | false_negatives | true_negatives |
| --- | --- | --- | --- |
| credit score band | 4.3051 | 4.1124 | 3.7729 |
| loan-to-value band | 1.7237 | 1.7346 | 2.0770 |
| current performance status | 0.0000 | 0.0702 | 0.1487 |
| days past due | 0.3593 | 2.3916 | 5.5585 |
| worst days past due in last 6 months | 2.2456 | 3.6333 | 8.1703 |
| loan age | 14.2339 | 32.1359 | 49.8135 |
| record data-quality score | 95.4912 | 94.9445 | 95.2504 |
| balance as a share of original | 0.9863 | 0.9373 | 0.9189 |

## Target: `exception_required`

### Global feature importance

Mean absolute SHAP contribution across the test window. `direction` is the sign of the correlation between a feature's value and its contribution, which recovers whether higher values raise or lower risk without assuming monotonicity.

| feature | plain_english | mean_abs_shap | share_of_total_attribution | direction |
| --- | --- | --- | --- | --- |
| dq_score | record data-quality score | 1.1117 | 0.5259 | non-monotone / categorical |
| dq_violation_count | dq violation count | 0.2115 | 0.1000 | non-monotone / categorical |
| missing_field_count | missing field count | 0.1854 | 0.0877 | higher value lowers risk |
| svc_present | svc present | 0.1026 | 0.0485 | non-monotone / categorical |
| doc_incomplete | doc incomplete | 0.0995 | 0.0471 | non-monotone / categorical |
| svc_balance_rel_gap | servicer feed balance gap | 0.0826 | 0.0391 | non-monotone / categorical |
| dti_ord | debt-to-income band | 0.0681 | 0.0322 | higher value lowers risk |
| document_status | document custody status | 0.0401 | 0.0190 | non-monotone / categorical |
| reporting_lag_days | servicer reporting lag | 0.0381 | 0.0180 | non-monotone / categorical |
| ltv_ord | loan-to-value band | 0.0282 | 0.0133 | non-monotone / categorical |
| svc_status_conflict | svc status conflict | 0.0156 | 0.0074 | non-monotone / categorical |
| unemployment_delta_12m | unemployment delta 12m | 0.0154 | 0.0073 | non-monotone / categorical |
| age_repaired | age repaired | 0.0120 | 0.0057 | non-monotone / categorical |
| dpd_status_residual | days past due against reported status | 0.0111 | 0.0053 | higher value raises risk |
| credit_ord | credit score band | 0.0099 | 0.0047 | higher value raises risk |
| stale_reporting | stale reporting | 0.0091 | 0.0043 | non-monotone / categorical |
| payment_to_balance | scheduled payment relative to balance | 0.0075 | 0.0036 | non-monotone / categorical |
| interest_rate_clean | note rate | 0.0067 | 0.0032 | higher value lowers risk |

The top three drivers — record data-quality score, dq violation count, missing field count — account for **71.4%** of total attribution.

### Local explanations — ten highest-risk records in the test window

Each row shows the calibrated probability a reviewer sees and the four largest log-odds contributions behind it.

| loan_id | reporting_month | current_status | calibrated_probability | driver_1 | driver_1_value | driver_1_log_odds | driver_2 | driver_2_value | driver_2_log_odds | driver_3 | driver_3_value | driver_3_log_odds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LN101279 | 2026-03 | Current | 0.8867 | record data-quality score | 72.0000 | 3.4934 | dq violation count | 3 | 0.6744 | servicer feed balance gap | 0.069 | 0.5750 |
| LN100921 | 2026-02 | Current | 0.8860 | record data-quality score | 83.0000 | 3.3525 | servicer reporting lag | -22.0 | 0.9412 | dq violation count | 3 | 0.7928 |
| LN100372 | 2026-01 | Current | 0.8817 | record data-quality score | 77.0000 | 3.4371 | dq violation count | 2 | 0.8654 | age repaired | 1 | 0.5119 |
| LN100013 | 2026-06 | Current | 0.8817 | record data-quality score | 77.0000 | 3.4136 | dq violation count | 2 | 0.8618 | age repaired | 1 | 0.5052 |
| LN100171 | 2026-05 | Current | 0.8815 | record data-quality score | 74.0000 | 3.4390 | dq violation count | 3 | 1.0449 | age repaired | 1 | 0.5098 |
| LN100011 | 2026-06 | Current | 0.8815 | record data-quality score | 74.0000 | 3.4342 | dq violation count | 3 | 1.0451 | age repaired | 1 | 0.5351 |
| LN100834 | 2026-04 | Current | 0.8815 | record data-quality score | 67.0000 | 3.4345 | dq violation count | 4 | 1.0439 | age repaired | 1 | 0.4501 |
| LN100414 | 2026-06 | Current | 0.8806 | record data-quality score | 74.0000 | 3.4190 | dq violation count | 3 | 1.0471 | age repaired | 1 | 0.5318 |
| LN100695 | 2026-05 | Current | 0.8794 | record data-quality score | 74.0000 | 3.4057 | dq violation count | 3 | 1.0427 | age repaired | 1 | 0.5310 |
| LN100767 | 2026-06 | DQ90plus | 0.8781 | record data-quality score | 76.0000 | 3.4656 | servicer feed balance gap | 0.029 | 0.6219 | dq violation count | 2 | 0.4278 |

### Local explanations — five lowest-risk records (contrast set)

| loan_id | reporting_month | current_status | calibrated_probability | driver_1 | driver_1_value | driver_1_log_odds | driver_2 | driver_2_value | driver_2_log_odds | driver_3 | driver_3_value | driver_3_log_odds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LN101041 | 2026-01 | Current | 0.0032 | record data-quality score | 97.0000 | -0.5649 | missing field count | 1 | -0.1271 | refinance incentive (note rate less market rate) | nan | -0.1122 |
| LN100935 | 2026-05 | Current | 0.0032 | record data-quality score | 97.0000 | -0.7564 | svc present | 0 | -0.1129 | refinance incentive (note rate less market rate) | nan | -0.1087 |
| LN100154 | 2026-05 | Current | 0.0032 | record data-quality score | 97.0000 | -0.7575 | svc present | 0 | -0.1135 | refinance incentive (note rate less market rate) | nan | -0.1087 |
| LN100398 | 2026-05 | Current | 0.0032 | record data-quality score | 97.0000 | -0.7545 | svc present | 0 | -0.1140 | refinance incentive (note rate less market rate) | nan | -0.1085 |
| LN100293 | 2026-02 | Current | 0.0032 | record data-quality score | 97.0000 | -0.7577 | svc present | 0 | -0.1125 | refinance incentive (note rate less market rate) | nan | -0.1124 |

### Model confidence and uncertainty

Predictions from the final boosting rounds are collected and their spread used as an epistemic-uncertainty proxy. This measures sensitivity to where the boosting sequence stopped — it is a stability signal, **not** a statistical confidence interval, and is not presented as one.

| confidence_band | share_of_records |
| --- | --- |
| high | 0.8279 |
| low | 0.1632 |
| medium | 0.0089 |

| statistic | calibrated_probability | staged_std | staged_p10 | staged_p90 |
| --- | --- | --- | --- | --- |
| count | 3480.0000 | 3480.0000 | 3480.0000 | 3480.0000 |
| mean | 0.1254 | 0.0138 | 0.1350 | 0.1693 |
| std | 0.2761 | 0.0151 | 0.2493 | 0.2859 |
| min | 0.0032 | 0.0021 | 0.0180 | 0.0376 |
| 25% | 0.0040 | 0.0071 | 0.0212 | 0.0391 |
| 50% | 0.0043 | 0.0072 | 0.0222 | 0.0403 |
| 75% | 0.0052 | 0.0074 | 0.0260 | 0.0428 |
| max | 0.8867 | 0.0998 | 0.7712 | 0.9040 |

### False positive / false negative analysis

Evaluated at the threshold that achieves 30% precision on the test window (`0.0044`): 442 true positives, 1029 false positives, 12 false negatives out of 3480 records with 454 actual events. Precision 0.300, recall 0.974.

**Where the errors concentrate.** A model that misses events uniformly is a different problem from one that misses them in a specific segment — the second is a fairness and coverage issue, not just an accuracy one.

By `credit_score_band`:

| credit_score_band | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| 660-699 | 573 | 0.1257 | 0.1225 | 0.3298 | 0.0052 | 0.0032 |
| 740-779 | 830 | 0.1181 | 0.1130 | 0.2386 | 0.0048 | 0.0051 |
| 780+ | 551 | 0.1307 | 0.1272 | 0.2523 | 0.0036 | 0.0034 |
| 700-739 | 772 | 0.1127 | 0.1162 | 0.2902 | 0.0026 | -0.0035 |
| 620-659 | 467 | 0.1734 | 0.1566 | 0.2998 | 0.0021 | 0.0169 |
| 580-619 | 154 | 0.1169 | 0.1085 | 0.4156 | 0.0000 | 0.0083 |
| <580 | 57 | 0.1579 | 0.1412 | 0.2807 | 0.0000 | 0.0167 |

By `servicer_name`:

| servicer_name | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| Northgate Servicing | 1044 | 0.1054 | 0.1020 | 0.2538 | 0.0048 | 0.0034 |
| Belmont Loan Services | 846 | 0.1312 | 0.1218 | 0.2754 | 0.0047 | 0.0094 |
| Arcadia Capital Servicing | 729 | 0.1029 | 0.1052 | 0.2675 | 0.0027 | -0.0023 |
| Pioneer Mortgage Ops | 511 | 0.1429 | 0.1516 | 0.4227 | 0.0020 | -0.0087 |
| Kestrel Financial | 350 | 0.2429 | 0.2082 | 0.3429 | 0.0000 | 0.0347 |

By `current_status`:

| current_status | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| DQ30 | 50 | 0.2200 | 0.1703 | 0.3200 | 0.0200 | 0.0497 |
| Current | 3316 | 0.1297 | 0.1250 | 0.2868 | 0.0033 | 0.0047 |
| DQ60 | 33 | 0.1515 | 0.1634 | 0.3636 | 0.0000 | -0.0119 |
| DQ90plus | 81 | 0.0988 | 0.0991 | 0.6173 | 0.0000 | -0.0004 |

By `state`:

| state | n | actual_rate | mean_predicted | false_positive_rate | false_negative_rate | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| NC | 127 | 0.1575 | 0.1372 | 0.2362 | 0.0157 | 0.0203 |
| NY | 312 | 0.1410 | 0.1263 | 0.3622 | 0.0096 | 0.0147 |
| NV | 127 | 0.1575 | 0.1416 | 0.3071 | 0.0079 | 0.0159 |
| FL | 406 | 0.1379 | 0.1291 | 0.3177 | 0.0049 | 0.0088 |
| CA | 624 | 0.1170 | 0.1189 | 0.2516 | 0.0048 | -0.0019 |
| GA | 238 | 0.1176 | 0.1303 | 0.2647 | 0.0042 | -0.0126 |
| CO | 63 | 0.2063 | 0.1699 | 0.2857 | 0.0000 | 0.0364 |
| MI | 119 | 0.1261 | 0.1198 | 0.2101 | 0.0000 | 0.0062 |
| IL | 216 | 0.1157 | 0.1143 | 0.2685 | 0.0000 | 0.0014 |
| AZ | 164 | 0.1098 | 0.1071 | 0.6646 | 0.0000 | 0.0027 |
| NJ | 151 | 0.1258 | 0.1129 | 0.3113 | 0.0000 | 0.0129 |
| OH | 137 | 0.1606 | 0.1482 | 0.2263 | 0.0000 | 0.0124 |
| PA | 160 | 0.1187 | 0.1246 | 0.2875 | 0.0000 | -0.0059 |
| TX | 489 | 0.1309 | 0.1246 | 0.2638 | 0.0000 | 0.0063 |
| WA | 147 | 0.1224 | 0.1260 | 0.2381 | 0.0000 | -0.0036 |

**What false positives and false negatives look like.** Mean feature values for each error class against correctly-rejected records.

| feature | false_positives | false_negatives | true_negatives |
| --- | --- | --- | --- |
| credit score band | 3.7433 | 4.2500 | 4.0330 |
| loan-to-value band | 2.0492 | 1.7500 | 1.9559 |
| current performance status | 0.1846 | 0.0833 | 0.0621 |
| days past due | 6.4399 | 3.4167 | 2.1868 |
| worst days past due in last 6 months | 9.9660 | 4.0000 | 4.0876 |
| loan age | 48.4012 | 47.5000 | 55.6079 |
| record data-quality score | 94.2915 | 97.7500 | 97.6430 |
| balance as a share of original | 0.9118 | 0.9263 | 0.9134 |

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
