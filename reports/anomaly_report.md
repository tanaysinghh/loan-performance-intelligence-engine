# Anomaly and Exception Report

**Task 4.** Isolation forest (unsupervised) and LightGBM (supervised). No language model contributes to any score in this report.

## 1. Three questions, three models

These are kept separate on purpose rather than blended into one number:

| question | model | supervised | why_separate |
| --- | --- | --- | --- |
| Is this record statistically odd? | Isolation forest | no | Catches defect shapes nobody wrote a rule for. A cleanly-formatted record with a missing document file is not statistically odd but is still a control breach. |
| Will a reviewer raise an exception? | LightGBM binary | yes | The actionable number, calibrated against what reviewers actually did rather than against what looks unusual. |
| Which exception is it? | LightGBM multiclass | yes | Routes the item to the right queue. Predicted only where the binary model already says an exception is likely. |

## 2. Record-level anomaly score

Isolation forest over 18 numeric record attributes, 400 trees, fitted on the training window only and applied forward. Scores are min-max mapped to 0-1 against the 0.5th and 99.5th percentiles so the scale is stable against single extreme records.

### Feature selection was a correction, not a first guess

The first feature set for this model used raw record levels — balance, loan age, remaining term, original balance. It scored *below* the base exception rate on its own top decile (lift 0.92x), because a genuinely large, genuinely seasoned jumbo loan is a statistical outlier and an entirely correct record. The feature set was rebuilt around quantities where deviation means a *defect* rather than a large loan: residuals against what the record should say given its own other fields (amortisation against term elapsed, days past due against reported status), disagreements with the second servicer feed, reporting timeliness, and repair indicators.

Rebuilt on defect-shaped features, the same model now reaches 4.12x lift and ROC-AUC 0.893 against the exception label. For reference, the size-shaped feature set this replaced scored 0.92x lift and ROC-AUC 0.615 — those two figures are from the development iteration that motivated the change and are quoted as history, not as a measurement on the current data.

### Does the unsupervised score agree with the reviewer label?

This is the check that tells you whether an unsupervised score is worth anything. It was never shown the exception label.

| score_cutoff | flagged_share | precision_vs_exception_label | recall_vs_exception_label | base_exception_rate | lift_over_base | roc_auc_vs_exception_label |
| --- | --- | --- | --- | --- | --- | --- |
| 0.4467 | 0.0600 | 0.5896 | 0.2474 | 0.1430 | 4.1224 | 0.8934 |

Flagging the top 6.0% of records by anomaly score alone gives **59.0%** precision against the exception label, a lift of **4.12x** over the 14.3% base rate, with ROC-AUC **0.893**. Useful, and clearly weaker than the supervised model below — which is the expected ordering, and the reason the supervised score drives the queue while the anomaly score is kept as a second opinion for defect shapes the label does not cover.

### Anomaly concentration by servicer

| servicer_name | records | mean_anomaly_score | pct_top_decile | mean_exception_probability | actual_exception_rate |
| --- | --- | --- | --- | --- | --- |
| PINGORA LOAN SERVICING, LLC | 2266 | 0.1563 | 0.1465 | 0.1653 | 0.1646 |
| BRANCH BANKING AND TRUST COMPANY | 219 | 0.1528 | 0.1461 | 0.1320 | 0.1370 |
| UNITED WHOLESALE MORTGAGE, LLC | 23897 | 0.1518 | 0.1354 | 0.1826 | 0.1850 |
| TH MSR HOLDINGS LLC | 6606 | 0.1493 | 0.1367 | 0.1742 | 0.1720 |
| PENNYMAC CORP. | 21569 | 0.1473 | 0.1313 | 0.1812 | 0.1811 |
| FIFTH THIRD BANK, NATIONAL ASSOCIATION | 9112 | 0.1457 | 0.1332 | 0.1582 | 0.1610 |
| LOANDEPOT.COM, LLC | 10657 | 0.1432 | 0.1219 | 0.1684 | 0.1702 |
| UNITED SHORE FINANCIAL SERVICES, LLC | 3135 | 0.1431 | 0.1254 | 0.1685 | 0.1684 |
| CALIBER HOME LOANS, INC. | 9159 | 0.1408 | 0.1174 | 0.1778 | 0.1781 |
| PROVIDENT FUNDING ASSOCIATES, L.P. | 686 | 0.1401 | 0.1254 | 0.1460 | 0.1472 |
| CITIZENS BANK, NA | 13577 | 0.1387 | 0.1156 | 0.1567 | 0.1588 |
| AMERIHOME MORTGAGE COMPANY, LLC | 15063 | 0.1359 | 0.1139 | 0.1603 | 0.1595 |
| CROSSCOUNTRY MORTGAGE, LLC | 1519 | 0.1352 | 0.1178 | 0.1413 | 0.1527 |
| MATRIX FINANCIAL SERVICES CORPORATION | 12060 | 0.1343 | 0.1114 | 0.1670 | 0.1686 |
| SPECIALIZED LOAN SERVICING LLC | 374 | 0.1302 | 0.1257 | 0.1266 | 0.1390 |
| NEW RESIDENTIAL MORTGAGE LLC | 28354 | 0.1298 | 0.1056 | 0.1449 | 0.1457 |
| JPMORGAN CHASE BANK, NATIONAL ASSOCIATION | 60672 | 0.1291 | 0.1043 | 0.1512 | 0.1515 |
| FREEDOM MORTGAGE CORPORATION | 17652 | 0.1282 | 0.1041 | 0.1487 | 0.1496 |
| PHH ASSET SERVICES LLC | 1514 | 0.1266 | 0.1057 | 0.1296 | 0.1334 |
| ONSLOW BAY FINANCIAL LLC | 8642 | 0.1246 | 0.0992 | 0.1527 | 0.1504 |
| ROCKET MORTGAGE, LLC | 37632 | 0.1241 | 0.0976 | 0.1415 | 0.1443 |
| SUNTRUST BANK | 289 | 0.1240 | 0.0934 | 0.1234 | 0.1246 |
| PENNYMAC LOAN SERVICES, LLC | 9803 | 0.1239 | 0.0952 | 0.1457 | 0.1468 |
| GUARANTEED RATE, INC. | 331 | 0.1227 | 0.0876 | 0.1384 | 0.1420 |
| LAKEVIEW LOAN SERVICING, LLC | 30384 | 0.1223 | 0.0969 | 0.1369 | 0.1382 |
| PNC BANK, NA | 12829 | 0.1211 | 0.0932 | 0.1295 | 0.1280 |
| GUILD MORTGAGE COMPANY LLC | 1515 | 0.1209 | 0.0865 | 0.1278 | 0.1287 |
| NATIONSTAR MORTGAGE LLC DBA MR. COOPER | 42385 | 0.1189 | 0.0921 | 0.1345 | 0.1347 |
| OTHER | 185467 | 0.1187 | 0.0927 | 0.1275 | 0.1282 |
| QUICKEN LOANS, LLC | 5387 | 0.1184 | 0.0956 | 0.1198 | 0.1192 |
| CMG MORTGAGE, INC. | 2022 | 0.1168 | 0.0841 | 0.1333 | 0.1360 |
| PHH MORTGAGE CORPORATION | 8890 | 0.1161 | 0.0883 | 0.1313 | 0.1295 |
| HOME POINT FINANCIAL CORPORATION | 3488 | 0.1146 | 0.0834 | 0.1443 | 0.1448 |
| BANK OF AMERICA, N.A. | 3531 | 0.1144 | 0.0869 | 0.1298 | 0.1306 |
| TRUIST BANK | 19350 | 0.1143 | 0.0904 | 0.1151 | 0.1150 |
| QUICKEN LOANS INC. | 2169 | 0.1127 | 0.0793 | 0.1215 | 0.1213 |
| MARLIN MORTGAGE CAPITAL, LLC | 512 | 0.1095 | 0.0801 | 0.1168 | 0.1191 |
| FIFTH THIRD BANK | 105 | 0.1091 | 0.0762 | 0.0897 | 0.0952 |
| U.S. BANK N.A. | 20156 | 0.1077 | 0.0821 | 0.1129 | 0.1137 |
| WELLS FARGO BANK, N.A. | 34862 | 0.1074 | 0.0779 | 0.1058 | 0.1055 |

Ranking by mean anomaly score independently recovers the two servicers the data intelligence report identified as having the worst reporting hygiene. The unsupervised model was given no servicer identity at all — it only sees the numeric record profile — so this is corroboration, not circularity.

## 3. Exception probability

The logistic baseline here is deliberately the *same* nine credit fields used in Task 2, and its failure is the point: ROC-AUC 0.53, barely above chance. Whether a record needs an exception has almost nothing to do with borrower credit quality and almost everything to do with reporting hygiene, reconciliation breaks and document custody. Any pipeline that reuses a credit feature set for operational exceptions is solving the wrong problem.

| model | n | positive_rate | roc_auc | pr_auc | best_f1 | recall_at_precision_30 | recall_at_precision_50 | brier | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prior | 63678 | 0.1430 | 0.5000 | 0.1430 | 0.2503 | 0.0000 | 0.0000 | 0.1226 | 0.0000 |
| baseline_logistic | 63678 | 0.1430 | 0.5400 | 0.2301 | 0.2508 | 0.1133 | 0.0825 | 0.2467 | 0.3591 |
| lgbm_raw | 63678 | 0.1430 | 0.9691 | 0.8462 | 0.8539 | 0.9819 | 0.9754 | 0.0369 | 0.0272 |
| lgbm_calibrated | 63678 | 0.1430 | 0.9693 | 0.8293 | 0.8538 | 0.9735 | 0.9735 | 0.0349 | 0.0019 |

## 4. Exception type

Six-way classification over records where an exception is required, benchmarked against always predicting the most common type.

| n | accuracy | macro_f1 | weighted_f1 | log_loss | macro_roc_auc | split | model |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 9446 | 0.9681 | 0.9247 | 0.9692 | 0.1043 | 0.9967 | valid | lgbm_exception_type |
| 9446 | 0.4542 | 0.1041 | 0.2837 |  |  | valid | majority_class_baseline |
| 9108 | 0.9661 | 0.9174 | 0.9676 | 0.1080 | 0.9970 | test | lgbm_exception_type |
| 9108 | 0.4533 | 0.1040 | 0.2828 |  |  | test | majority_class_baseline |

### Per-type performance — valid window

| class | support | precision | recall | f1 |
| --- | --- | --- | --- | --- |
| missing_documentation | 4290 | 0.9917 | 0.9765 | 0.9840 |
| balance_reconciliation_break | 2133 | 0.9864 | 0.9841 | 0.9852 |
| stale_servicer_reporting | 905 | 0.9744 | 0.9260 | 0.9496 |
| invalid_date_relationship | 1498 | 0.9759 | 0.9746 | 0.9753 |
| status_dpd_mismatch | 298 | 0.8435 | 0.8859 | 0.8642 |
| unexpected_balance_movement | 322 | 0.6941 | 0.9161 | 0.7898 |

### Per-type performance — test window

| class | support | precision | recall | f1 |
| --- | --- | --- | --- | --- |
| missing_documentation | 4129 | 0.9924 | 0.9763 | 0.9843 |
| balance_reconciliation_break | 2008 | 0.9860 | 0.9801 | 0.9830 |
| stale_servicer_reporting | 868 | 0.9769 | 0.9274 | 0.9515 |
| invalid_date_relationship | 1504 | 0.9787 | 0.9761 | 0.9774 |
| status_dpd_mismatch | 265 | 0.8538 | 0.8377 | 0.8457 |
| unexpected_balance_movement | 334 | 0.6545 | 0.9132 | 0.7625 |

## 5. Anomaly driver explanation

Isolation forest gives no native per-feature attribution. Rather than invent one, each record's drivers are the features furthest from the training-window distribution measured in robust z-units (median and MAD, so a handful of extreme records cannot move the reference). A reviewer can check the named field against the record in front of them, which a raw path-length score does not allow.

| top_driver | share_of_records |
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

## 6. Reviewer queue

40 reviewer-ready examples from the test window, ranked by a priority score of 0.6 x exception probability + 0.4 x anomaly score, with coverage forced across every predicted exception type so the queue is not monopolised by the single most common defect. Actual labels are shown for assessment only — they are not available at scoring time.

| loan_id | reporting_month | servicer_name | review_priority | exception_probability | anomaly_score | predicted_exception_type | predicted_type_confidence | anomaly_driver_1 | anomaly_driver_1_zscore | anomaly_driver_2 | rules_violated | actual_exception_type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LN-D79DAE9144 | 2025-12 | PHH ASSET SERVICES LLC | 1.0000 | 1.0000 | 1.0000 | invalid_date_relationship | 0.9820 | loan age required repair | 9.4300 | document file incomplete | loan_age_inconsistent_with_dates; remaining_term_inconsistent; document_file_incomplete; servicer_balance_break | invalid_date_relationship |
| LN-5C6F525328 | 2025-11 | CROSSCOUNTRY MORTGAGE, LLC | 1.0000 | 1.0000 | 1.0000 | invalid_date_relationship | 0.9560 | servicer reporting lag | 12.8200 | loan age required repair | last_updated_before_period_end; loan_age_inconsistent_with_dates; balance_exceeds_original; remaining_term_inconsistent; servicer_record_absent | invalid_date_relationship |
| LN-F13353D5BD | 2025-12 | NATIONSTAR MORTGAGE LLC DBA MR. COOPER | 1.0000 | 1.0000 | 1.0000 | invalid_date_relationship | 0.9490 | loan age required repair | 9.4300 | servicer feed balance gap | loan_age_inconsistent_with_dates; balance_exceeds_original; remaining_term_inconsistent; document_file_incomplete; servicer_balance_break | invalid_date_relationship |
| LN-5EB1F1E8EC | 2026-02 | OTHER | 1.0000 | 1.0000 | 1.0000 | invalid_date_relationship | 0.9990 | servicer reporting lag | 12.3100 | loan age required repair | last_updated_before_period_end; loan_age_inconsistent_with_dates; remaining_term_inconsistent | invalid_date_relationship |
| LN-538B437889 | 2026-01 | ONSLOW BAY FINANCIAL LLC | 1.0000 | 1.0000 | 1.0000 | invalid_date_relationship | 0.9780 | loan age required repair | 9.4300 | servicer feed balance gap | loan_age_inconsistent_with_dates; remaining_term_inconsistent; missing_critical_field; document_file_incomplete; servicer_balance_break | balance_reconciliation_break |
| LN-DAD07313C8 | 2026-03 | PENNYMAC CORP. | 1.0000 | 1.0000 | 1.0000 | invalid_date_relationship | 0.9170 | loan age required repair | 9.4300 | servicer feed balance gap | loan_age_inconsistent_with_dates; remaining_term_inconsistent; document_file_incomplete; servicer_balance_break | invalid_date_relationship |
| LN-806B31FFFD | 2026-02 | OTHER | 1.0000 | 1.0000 | 1.0000 | invalid_date_relationship | 0.9880 | loan age required repair | 9.4300 | servicer reporting lag | last_updated_before_period_end; loan_age_inconsistent_with_dates; balance_exceeds_original; remaining_term_inconsistent; document_file_incomplete; servicer_balance_break | invalid_date_relationship |
| LN-1C4269B1E2 | 2025-10 | OTHER | 1.0000 | 1.0000 | 1.0000 | balance_reconciliation_break | 0.8940 | servicer reporting lag | 21.2500 | month-over-month balance movement | document_file_incomplete; stale_servicer_reporting; servicer_balance_break | none |
| LN-72F6856AAE | 2025-12 | JPMORGAN CHASE BANK, NATIONAL ASSOCIATION | 1.0000 | 1.0000 | 1.0000 | balance_reconciliation_break | 0.9100 | balance against expected amortisation for term elapsed | 20.6800 | servicer reporting lag | document_file_incomplete; stale_servicer_reporting; servicer_balance_break | balance_reconciliation_break |
| LN-F61C164687 | 2026-01 | PENNYMAC CORP. | 1.0000 | 1.0000 | 1.0000 | invalid_date_relationship | 1.0000 | month-over-month change in days past due | 11.6100 | loan age required repair | origination_after_reporting; last_updated_before_period_end; loan_age_inconsistent_with_dates; servicer_record_absent | invalid_date_relationship |
| LN-A7F6F25616 | 2025-12 | TH MSR HOLDINGS LLC | 0.9942 | 1.0000 | 0.9854 | invalid_date_relationship | 0.9980 | servicer reporting lag | 11.4700 | loan age required repair | last_updated_before_period_end; loan_age_inconsistent_with_dates; remaining_term_inconsistent | none |
| LN-74CE67155C | 2026-03 | OTHER | 0.9788 | 0.9646 | 1.0000 | balance_reconciliation_break | 0.9660 | servicer feed balance gap | 3.8800 | record arrived by manual upload | document_file_incomplete; servicer_balance_break | balance_reconciliation_break |
| LN-BB849D45D7 | 2025-10 | FIFTH THIRD BANK, NATIONAL ASSOCIATION | 0.9707 | 0.9512 | 1.0000 | balance_reconciliation_break | 0.9210 | servicer reporting lag | 26.8100 | servicer feed balance gap | document_file_incomplete; stale_servicer_reporting; servicer_balance_break | balance_reconciliation_break |
| LN-2F6C0D79C6 | 2026-01 | U.S. BANK N.A. | 0.9684 | 0.9474 | 1.0000 | invalid_date_relationship | 0.9750 | loan age required repair | 9.4300 | balance against expected amortisation for term elapsed | loan_age_inconsistent_with_dates; balance_exceeds_original; remaining_term_inconsistent; document_file_incomplete | invalid_date_relationship |
| LN-FF39F0A9B2 | 2026-01 | OTHER | 0.9684 | 0.9474 | 1.0000 | invalid_date_relationship | 0.9840 | month-over-month balance movement | 56.8400 | three-month balance movement | loan_age_inconsistent_with_dates; balance_exceeds_original; remaining_term_inconsistent; servicer_record_absent | invalid_date_relationship |
| LN-2E66A42EB6 | 2025-12 | OTHER | 0.9684 | 0.9474 | 1.0000 | invalid_date_relationship | 0.9950 | three-month balance movement | 51.3200 | loan age required repair | loan_age_inconsistent_with_dates; balance_exceeds_original; remaining_term_inconsistent | invalid_date_relationship |
| LN-A43B603760 | 2025-10 | ROCKET MORTGAGE, LLC | 0.9684 | 0.9474 | 1.0000 | invalid_date_relationship | 0.9530 | month-over-month balance movement | 65.4100 | three-month balance movement | loan_age_inconsistent_with_dates; balance_exceeds_original; remaining_term_inconsistent; servicer_record_absent | invalid_date_relationship |
| LN-EBFAA3ABD2 | 2025-11 | LAKEVIEW LOAN SERVICING, LLC | 0.9684 | 0.9474 | 1.0000 | invalid_date_relationship | 0.8560 | month-over-month balance movement | 38.2600 | three-month balance movement | loan_age_inconsistent_with_dates; balance_exceeds_original; remaining_term_inconsistent; servicer_record_absent | invalid_date_relationship |
| LN-05E2DF2646 | 2026-01 | UNITED WHOLESALE MORTGAGE, LLC | 0.9684 | 0.9474 | 1.0000 | invalid_date_relationship | 0.9570 | loan age required repair | 9.4300 | month-over-month change in days past due | loan_age_inconsistent_with_dates; balance_exceeds_original; remaining_term_inconsistent | invalid_date_relationship |
| LN-240AD09CD2 | 2025-11 | NEW RESIDENTIAL MORTGAGE LLC | 0.9684 | 0.9474 | 1.0000 | invalid_date_relationship | 0.9600 | month-over-month balance movement | 41.8700 | three-month balance movement | loan_age_inconsistent_with_dates; balance_exceeds_original; remaining_term_inconsistent; servicer_record_absent | invalid_date_relationship |
| LN-DDAB322709 | 2026-03 | PENNYMAC CORP. | 0.9684 | 0.9474 | 1.0000 | invalid_date_relationship | 0.8900 | month-over-month balance movement | 57.6000 | three-month balance movement | loan_age_inconsistent_with_dates; balance_exceeds_original; remaining_term_inconsistent; servicer_record_absent | invalid_date_relationship |
| LN-2E66A42EB6 | 2025-11 | OTHER | 0.9684 | 0.9474 | 1.0000 | invalid_date_relationship | 0.9510 | month-over-month balance movement | 109.2200 | three-month balance movement | loan_age_inconsistent_with_dates; balance_exceeds_original; remaining_term_inconsistent | invalid_date_relationship |
| LN-05E2DF2646 | 2025-12 | UNITED WHOLESALE MORTGAGE, LLC | 0.9684 | 0.9474 | 1.0000 | invalid_date_relationship | 0.9410 | days past due against reported status | 14.6000 | loan age required repair | loan_age_inconsistent_with_dates; balance_exceeds_original; status_dpd_mismatch; remaining_term_inconsistent; servicer_record_absent | invalid_date_relationship |
| LN-2F6C0D79C6 | 2026-02 | U.S. BANK N.A. | 0.9684 | 0.9474 | 1.0000 | invalid_date_relationship | 0.9610 | loan age required repair | 9.4300 | balance against expected amortisation for term elapsed | loan_age_inconsistent_with_dates; balance_exceeds_original; remaining_term_inconsistent; document_file_incomplete; servicer_record_absent | invalid_date_relationship |
| LN-5CCE1027C3 | 2026-02 | OTHER | 0.9684 | 0.9474 | 1.0000 | invalid_date_relationship | 0.8760 | month-over-month balance movement | 18.2900 | three-month balance movement | loan_age_inconsistent_with_dates; balance_exceeds_original; remaining_term_inconsistent | invalid_date_relationship |
| LN-AC4B749119 | 2025-12 | OTHER | 0.9480 | 0.9133 | 1.0000 | status_dpd_mismatch | 0.8190 | days past due against reported status | 19.4600 | month-over-month change in days past due | status_dpd_mismatch; missing_critical_field; servicer_balance_break | balance_reconciliation_break |
| LN-820F8B9235 | 2025-10 | FIFTH THIRD BANK, NATIONAL ASSOCIATION | 0.9480 | 0.9133 | 1.0000 | status_dpd_mismatch | 0.5180 | days past due against reported status | 19.4600 | month-over-month balance movement | status_dpd_mismatch; missing_critical_field; servicer_balance_break | balance_reconciliation_break |
| LN-1DE24E7AFA | 2026-02 | FREEDOM MORTGAGE CORPORATION | 0.9480 | 0.9133 | 1.0000 | missing_documentation | 0.5350 | servicer reporting lag | 19.9000 | balance required repair | balance_exceeds_original; balance_increase_month_over_month; document_file_incomplete; stale_servicer_reporting | unexpected_balance_movement |
| LN-6DC044F621 | 2025-10 | JPMORGAN CHASE BANK, NATIONAL ASSOCIATION | 0.9305 | 0.9133 | 0.9564 | status_dpd_mismatch | 0.6010 | days past due required repair | 19.1300 | servicer feed record present | dpd_sentinel_value; servicer_balance_break | balance_reconciliation_break |
| LN-27816CFD42 | 2026-01 | LOANDEPOT.COM, LLC | 0.9183 | 0.8639 | 1.0000 | missing_documentation | 0.5850 | servicer reporting lag | 26.4700 | servicer feed days-past-due gap | missing_critical_field; document_file_incomplete; stale_servicer_reporting | missing_documentation |
| LN-3EA84D5CE0 | 2026-01 | FIFTH THIRD BANK, NATIONAL ASSOCIATION | 0.9183 | 0.8639 | 1.0000 | missing_documentation | 0.6810 | servicer reporting lag | 14.1600 | count of missing credit fields | missing_critical_field; document_file_incomplete; stale_servicer_reporting | missing_documentation |
| LN-820F8B9235 | 2026-03 | FIFTH THIRD BANK, NATIONAL ASSOCIATION | 0.9183 | 0.8639 | 1.0000 | unexpected_balance_movement | 0.6530 | balance required repair | 15.4700 | document file incomplete | balance_exceeds_original; balance_increase_month_over_month; document_file_incomplete | missing_documentation |
| LN-EF46CA02A8 | 2025-12 | JPMORGAN CHASE BANK, NATIONAL ASSOCIATION | 0.9183 | 0.8639 | 1.0000 | stale_servicer_reporting | 0.7470 | servicer reporting lag | 23.4400 | count of missing credit fields | missing_critical_field; document_file_incomplete; stale_servicer_reporting; servicer_record_absent | stale_servicer_reporting |
| LN-F6550307A8 | 2025-12 | OTHER | 0.9148 | 0.8580 | 1.0000 | stale_servicer_reporting | 0.5610 | three-month balance movement | 26.9000 | servicer reporting lag | document_file_incomplete; stale_servicer_reporting; servicer_record_absent | missing_documentation |
| LN-006012E8B5 | 2025-11 | OTHER | 0.9056 | 0.8427 | 1.0000 | unexpected_balance_movement | 0.5110 | balance required repair | 15.4700 | document file incomplete | negative_balance; document_file_incomplete; servicer_record_absent | missing_documentation |
| LN-5E1F928FB7 | 2025-11 | PNC BANK, NA | 0.8892 | 0.9133 | 0.8531 | unexpected_balance_movement | 0.5830 | balance required repair | 15.4700 | servicer feed balance gap | negative_balance; servicer_balance_break | balance_reconciliation_break |
| LN-6C0DFE98D3 | 2026-02 | OTHER | 0.8388 | 0.8580 | 0.8100 | stale_servicer_reporting | 0.5870 | servicer reporting lag | 23.9400 | record arrived by manual upload | document_file_incomplete; stale_servicer_reporting; servicer_record_absent | missing_documentation |
| LN-E937C14551 | 2025-12 | OTHER | 0.6194 | 0.4444 | 0.8819 | none | 0.8230 | days past due against reported status | 14.6000 | month-over-month change in days past due | status_dpd_mismatch | status_dpd_mismatch |
| LN-CE9B98FD26 | 2026-01 | OTHER | 0.5912 | 0.4595 | 0.7887 | none | 0.9940 | three-month balance movement | 19.6400 | servicer reporting lag | stale_servicer_reporting | stale_servicer_reporting |
| LN-A87F226C09 | 2025-12 | TH MSR HOLDINGS LLC | 0.5813 | 0.4595 | 0.7640 | none | 0.9970 | three-month balance movement | 6.4000 | month-over-month balance movement | missing_critical_field; document_file_incomplete | none |

Full queue with all evidence columns: `reports/anomaly_review_queue.csv`.

## 7. Limitations

- Isolation forest contamination is set to 6%, close to the observed exception rate. That is a prior, not an estimate; a genuine deployment would tune it against reviewer capacity rather than against the label.
- The exception label in this synthetic pack is generated from rule breaches plus a materiality threshold and ~1.2% reviewer noise. Real reviewer behaviour is less consistent, so the supervised ceiling here is optimistic.
- Driver attribution is univariate. A record can be anomalous through an *interaction* of two individually unremarkable fields, and this method will not name it.
