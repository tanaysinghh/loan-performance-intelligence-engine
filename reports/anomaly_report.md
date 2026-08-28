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

Rebuilt on defect-shaped features, the same model moved from 0.92x lift to 2.02x and from ROC-AUC 0.615 to 0.845 against the exception label.

### Does the unsupervised score agree with the reviewer label?

This is the check that tells you whether an unsupervised score is worth anything. It was never shown the exception label.

| score_cutoff | flagged_share | precision_vs_exception_label | recall_vs_exception_label | base_exception_rate | lift_over_base | roc_auc_vs_exception_label |
| --- | --- | --- | --- | --- | --- | --- |
| 0.6534 | 0.0601 | 0.2632 | 0.1211 | 0.1305 | 2.0172 | 0.8453 |

Flagging the top 6.0% of records by anomaly score alone gives **26.3%** precision against the exception label, a lift of **2.02x** over the 13.0% base rate, with ROC-AUC **0.845**. Useful, and clearly weaker than the supervised model below — which is the expected ordering, and the reason the supervised score drives the queue while the anomaly score is kept as a second opinion for defect shapes the label does not cover.

### Anomaly concentration by servicer

| servicer_name | records | mean_anomaly_score | pct_top_decile | mean_exception_probability | actual_exception_rate |
| --- | --- | --- | --- | --- | --- |
| Pioneer Mortgage Ops | 7595 | 0.2094 | 0.1548 | 0.1640 | 0.1637 |
| Kestrel Financial | 4980 | 0.2002 | 0.1386 | 0.2049 | 0.2056 |
| Arcadia Capital Servicing | 10493 | 0.1578 | 0.0954 | 0.1074 | 0.1071 |
| Belmont Loan Services | 11346 | 0.1487 | 0.0804 | 0.1138 | 0.1150 |
| Northgate Servicing | 14510 | 0.1447 | 0.0768 | 0.1018 | 0.1006 |

Ranking by mean anomaly score independently recovers the two servicers the data intelligence report identified as having the worst reporting hygiene. The unsupervised model was given no servicer identity at all — it only sees the numeric record profile — so this is corroboration, not circularity.

## 3. Exception probability

The logistic baseline here is deliberately the *same* nine credit fields used in Task 2, and its failure is the point: ROC-AUC 0.53, barely above chance. Whether a record needs an exception has almost nothing to do with borrower credit quality and almost everything to do with reporting hygiene, reconciliation breaks and document custody. Any pipeline that reuses a credit feature set for operational exceptions is solving the wrong problem.

| model | n | positive_rate | roc_auc | pr_auc | best_f1 | recall_at_precision_30 | recall_at_precision_50 | brier | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prior | 3480 | 0.1305 | 0.5000 | 0.1305 | 0.2308 | 0.0000 | 0.0000 | 0.1135 | 0.0000 |
| baseline_logistic | 3480 | 0.1305 | 0.5326 | 0.1702 | 0.2396 | 0.0463 | 0.0220 | 0.2442 | 0.3631 |
| lgbm_raw | 3480 | 0.1305 | 0.9642 | 0.8327 | 0.8631 | 0.9714 | 0.9626 | 0.0329 | 0.0236 |
| lgbm_calibrated | 3480 | 0.1305 | 0.9642 | 0.8327 | 0.8631 | 0.9714 | 0.9626 | 0.0318 | 0.0064 |

## 4. Exception type

Six-way classification over records where an exception is required, benchmarked against always predicting the most common type.

| n | accuracy | macro_f1 | weighted_f1 | log_loss | macro_roc_auc | split | model |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 507 | 0.9665 | 0.9175 | 0.9701 | 0.1295 | 0.9980 | valid | lgbm_exception_type |
| 507 | 0.4043 | 0.0960 | 0.2328 |  |  | valid | majority_class_baseline |
| 454 | 0.9515 | 0.8691 | 0.9540 | 0.1449 | 0.9972 | test | lgbm_exception_type |
| 454 | 0.4031 | 0.0958 | 0.2316 |  |  | test | majority_class_baseline |

### Per-type performance — valid window

| class | support | precision | recall | f1 |
| --- | --- | --- | --- | --- |
| missing_documentation | 205 | 0.9900 | 0.9659 | 0.9778 |
| balance_reconciliation_break | 141 | 0.9927 | 0.9645 | 0.9784 |
| stale_servicer_reporting | 59 | 0.9825 | 0.9492 | 0.9655 |
| invalid_date_relationship | 81 | 0.9875 | 0.9753 | 0.9814 |
| status_dpd_mismatch | 11 | 0.9167 | 1.0000 | 0.9565 |
| unexpected_balance_movement | 10 | 0.4762 | 1.0000 | 0.6452 |

### Per-type performance — test window

| class | support | precision | recall | f1 |
| --- | --- | --- | --- | --- |
| missing_documentation | 183 | 0.9836 | 0.9836 | 0.9836 |
| balance_reconciliation_break | 117 | 0.9829 | 0.9829 | 0.9829 |
| stale_servicer_reporting | 56 | 0.9464 | 0.9464 | 0.9464 |
| invalid_date_relationship | 66 | 1.0000 | 0.9242 | 0.9606 |
| status_dpd_mismatch | 19 | 0.9167 | 0.5789 | 0.7097 |
| unexpected_balance_movement | 13 | 0.4800 | 0.9231 | 0.6316 |

## 5. Anomaly driver explanation

Isolation forest gives no native per-feature attribution. Rather than invent one, each record's drivers are the features furthest from the training-window distribution measured in robust z-units (median and MAD, so a handful of extreme records cannot move the reference). A reviewer can check the named field against the record in front of them, which a raw path-length score does not allow.

| top_driver | share_of_records |
| --- | --- |
| scheduled payment relative to balance | 0.1853 |
| balance against expected amortisation for term elapsed | 0.1695 |
| three-month balance movement | 0.1563 |
| servicer feed record present | 0.1115 |
| servicer reporting lag | 0.0931 |
| count of missing credit fields | 0.0862 |
| record arrived by manual upload | 0.0506 |
| document file incomplete | 0.0477 |
| month-over-month balance movement | 0.0425 |
| servicer feed balance gap | 0.0221 |

## 6. Reviewer queue

36 reviewer-ready examples from the test window, ranked by a priority score of 0.6 x exception probability + 0.4 x anomaly score, with coverage forced across every predicted exception type so the queue is not monopolised by the single most common defect. Actual labels are shown for assessment only — they are not available at scoring time.

| loan_id | reporting_month | servicer_name | review_priority | exception_probability | anomaly_score | predicted_exception_type | predicted_type_confidence | anomaly_driver_1 | anomaly_driver_1_zscore | anomaly_driver_2 | rules_violated | actual_exception_type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LN100308 | 2026-06 | Pioneer Mortgage Ops | 0.9228 | 0.8713 | 1.0000 | invalid_date_relationship | 0.9950 | month-over-month balance movement | 19.2700 | three-month balance movement | loan_age_inconsistent_with_dates; remaining_term_inconsistent | none |
| LN100627 | 2026-04 | Kestrel Financial | 0.9207 | 0.8679 | 1.0000 | balance_reconciliation_break | 0.6100 | days past due against reported status | 17.1800 | month-over-month change in days past due | status_dpd_mismatch; document_file_incomplete; servicer_balance_break | balance_reconciliation_break |
| LN100308 | 2026-05 | Pioneer Mortgage Ops | 0.9121 | 0.8535 | 1.0000 | invalid_date_relationship | 0.9810 | three-month balance movement | 18.9700 | month-over-month balance movement | last_updated_before_period_end; missing_critical_field; document_file_incomplete; servicer_record_absent | invalid_date_relationship |
| LN100814 | 2026-01 | Arcadia Capital Servicing | 0.9092 | 0.8487 | 1.0000 | balance_reconciliation_break | 0.9760 | three-month balance movement | 17.5000 | month-over-month balance movement | servicer_balance_break | balance_reconciliation_break |
| LN101299 | 2026-04 | Kestrel Financial | 0.8812 | 0.8545 | 0.9212 | invalid_date_relationship | 0.9780 | servicer reporting lag | 25.6300 | loan age required repair | loan_age_inconsistent_with_dates; remaining_term_inconsistent; missing_critical_field; stale_servicer_reporting; servicer_record_absent | invalid_date_relationship |
| LN101444 | 2026-04 | Kestrel Financial | 0.8715 | 0.8419 | 0.9158 | balance_reconciliation_break | 0.8710 | servicer feed balance gap | 5.7300 | document file incomplete | missing_critical_field; document_file_incomplete; servicer_balance_break | balance_reconciliation_break |
| LN100884 | 2026-03 | Pioneer Mortgage Ops | 0.8579 | 0.7631 | 1.0000 | status_dpd_mismatch | 0.9930 | days past due against reported status | 17.1800 | three-month balance movement | status_dpd_mismatch; servicer_record_absent | status_dpd_mismatch |
| LN100853 | 2026-03 | Pioneer Mortgage Ops | 0.8564 | 0.7606 | 1.0000 | missing_documentation | 0.9860 | three-month balance movement | 18.0300 | month-over-month balance movement | document_file_incomplete; servicer_record_absent | missing_documentation |
| LN100785 | 2026-04 | Arcadia Capital Servicing | 0.8556 | 0.7594 | 1.0000 | missing_documentation | 0.9870 | three-month balance movement | 19.4100 | month-over-month balance movement | document_file_incomplete; servicer_record_absent | missing_documentation |
| LN101279 | 2026-03 | Belmont Loan Services | 0.8551 | 0.8946 | 0.7958 | invalid_date_relationship | 0.9600 | loan age required repair | 10.7000 | servicer feed balance gap | loan_age_inconsistent_with_dates; remaining_term_inconsistent; servicer_balance_break | invalid_date_relationship |
| LN100918 | 2026-06 | Pioneer Mortgage Ops | 0.8550 | 0.7583 | 1.0000 | missing_documentation | 0.9910 | three-month balance movement | 16.5000 | month-over-month balance movement | document_file_incomplete; servicer_record_absent | missing_documentation |
| LN100273 | 2026-04 | Belmont Loan Services | 0.8529 | 0.7549 | 1.0000 | missing_documentation | 0.9900 | three-month balance movement | 18.7900 | month-over-month balance movement | document_file_incomplete; servicer_record_absent | missing_documentation |
| LN100719 | 2026-03 | Arcadia Capital Servicing | 0.8524 | 0.7540 | 1.0000 | missing_documentation | 0.9910 | three-month balance movement | 14.3500 | month-over-month balance movement | document_file_incomplete; servicer_record_absent | missing_documentation |
| LN101468 | 2026-01 | Pioneer Mortgage Ops | 0.8514 | 0.7523 | 1.0000 | missing_documentation | 0.9960 | three-month balance movement | 17.6500 | month-over-month balance movement | document_file_incomplete; servicer_record_absent | missing_documentation |
| LN100918 | 2026-05 | Pioneer Mortgage Ops | 0.8486 | 0.7476 | 1.0000 | missing_documentation | 0.9890 | three-month balance movement | 16.2700 | month-over-month balance movement | document_file_incomplete | none |
| LN100786 | 2026-03 | Northgate Servicing | 0.8482 | 0.7581 | 0.9834 | missing_documentation | 0.9990 | three-month balance movement | 10.0500 | month-over-month balance movement | document_file_incomplete | missing_documentation |
| LN101038 | 2026-03 | Pioneer Mortgage Ops | 0.8475 | 0.8475 | 0.8475 | invalid_date_relationship | 0.9990 | loan age required repair | 10.7000 | record arrived by manual upload | loan_age_inconsistent_with_dates; remaining_term_inconsistent; servicer_record_absent | invalid_date_relationship |
| LN100432 | 2026-02 | Northgate Servicing | 0.8432 | 0.8711 | 0.8013 | invalid_date_relationship | 0.9990 | loan age required repair | 10.7000 | record arrived by manual upload | loan_age_inconsistent_with_dates; remaining_term_inconsistent | invalid_date_relationship |
| LN100633 | 2026-05 | Belmont Loan Services | 0.8306 | 0.8502 | 0.8011 | balance_reconciliation_break | 0.9980 | record arrived by manual upload | 3.7300 | days past due against reported status | servicer_balance_break | balance_reconciliation_break |
| LN101194 | 2026-03 | Arcadia Capital Servicing | 0.8259 | 0.8701 | 0.7595 | invalid_date_relationship | 0.9990 | loan age required repair | 10.7000 | record arrived by manual upload | loan_age_inconsistent_with_dates; remaining_term_inconsistent; servicer_record_absent | invalid_date_relationship |
| LN100013 | 2026-06 | Arcadia Capital Servicing | 0.8256 | 0.8914 | 0.7269 | invalid_date_relationship | 1.0000 | loan age required repair | 10.7000 | servicer feed days-past-due gap | origination_after_reporting; loan_age_inconsistent_with_dates | invalid_date_relationship |
| LN101473 | 2026-06 | Pioneer Mortgage Ops | 0.8242 | 0.7704 | 0.9048 | missing_documentation | 0.9990 | count of missing credit fields | 7.1800 | document file incomplete | missing_critical_field; document_file_incomplete | missing_documentation |
| LN101222 | 2026-06 | Northgate Servicing | 0.8238 | 0.8523 | 0.7810 | balance_reconciliation_break | 0.9950 | three-month balance movement | 15.2600 | month-over-month balance movement | servicer_balance_break | balance_reconciliation_break |
| LN100910 | 2026-02 | Northgate Servicing | 0.8201 | 0.8163 | 0.8258 | balance_reconciliation_break | 0.9940 | three-month balance movement | 12.1700 | month-over-month balance movement | missing_critical_field; servicer_balance_break | balance_reconciliation_break |
| LN101410 | 2026-04 | Kestrel Financial | 0.8189 | 0.8161 | 0.8231 | invalid_date_relationship | 0.9950 | three-month balance movement | 15.1500 | month-over-month balance movement | last_updated_before_period_end; missing_critical_field; servicer_record_absent | invalid_date_relationship |
| LN100250 | 2026-02 | Kestrel Financial | 0.8103 | 0.7296 | 0.9314 | status_dpd_mismatch | 0.9860 | days past due against reported status | 17.1800 | month-over-month change in days past due | status_dpd_mismatch; missing_critical_field; servicer_record_absent | status_dpd_mismatch |
| LN100968 | 2026-04 | Northgate Servicing | 0.7897 | 0.7301 | 0.8791 | unexpected_balance_movement | 0.9940 | balance required repair | 15.7500 | count of missing credit fields | balance_exceeds_original; balance_increase_month_over_month; missing_critical_field | none |
| LN100342 | 2026-06 | Northgate Servicing | 0.7831 | 0.7074 | 0.8967 | unexpected_balance_movement | 0.9980 | balance required repair | 15.7500 | record arrived by manual upload | balance_exceeds_original; balance_increase_month_over_month | unexpected_balance_movement |
| LN101000 | 2026-03 | Arcadia Capital Servicing | 0.7796 | 0.6670 | 0.9484 | status_dpd_mismatch | 0.9950 | days past due required repair | 19.9100 | scheduled payment relative to balance | dpd_sentinel_value; servicer_record_absent | status_dpd_mismatch |
| LN100585 | 2026-01 | Belmont Loan Services | 0.7575 | 0.8011 | 0.6922 | stale_servicer_reporting | 0.6210 | servicer reporting lag | 21.4200 | balance against expected amortisation for term elapsed | document_file_incomplete; stale_servicer_reporting | missing_documentation |
| LN100059 | 2026-06 | Belmont Loan Services | 0.7294 | 0.6261 | 0.8843 | stale_servicer_reporting | 0.8340 | servicer reporting lag | 12.1400 | three-month balance movement | stale_servicer_reporting; servicer_record_absent | none |
| LN100599 | 2026-05 | Pioneer Mortgage Ops | 0.6964 | 0.7900 | 0.5561 | unexpected_balance_movement | 0.9990 | balance required repair | 15.7500 | servicer feed record present | balance_exceeds_original; balance_increase_month_over_month | unexpected_balance_movement |
| LN100719 | 2026-02 | Arcadia Capital Servicing | 0.6705 | 0.5647 | 0.8291 | stale_servicer_reporting | 0.9930 | servicer reporting lag | 20.0700 | three-month balance movement | stale_servicer_reporting; servicer_record_absent | none |
| LN100093 | 2026-01 | Northgate Servicing | 0.5249 | 0.3754 | 0.7492 | none | 0.2410 | note rate required repair | 22.0000 | three-month balance movement | interest_rate_out_of_range; servicer_record_absent | none |
| LN100620 | 2026-06 | Belmont Loan Services | 0.5093 | 0.1822 | 1.0000 | none | 0.3040 | note rate required repair | 22.0000 | three-month balance movement | interest_rate_out_of_range | none |
| LN100458 | 2026-05 | Northgate Servicing | 0.5075 | 0.3665 | 0.7189 | none | 0.2950 | note rate required repair | 22.0000 | balance against expected amortisation for term elapsed | interest_rate_out_of_range; servicer_record_absent | none |

Full queue with all evidence columns: `reports/anomaly_review_queue.csv`.

## 7. Limitations

- Isolation forest contamination is set to 6%, close to the observed exception rate. That is a prior, not an estimate; a genuine deployment would tune it against reviewer capacity rather than against the label.
- The exception label in this synthetic pack is generated from rule breaches plus a materiality threshold and ~1.2% reviewer noise. Real reviewer behaviour is less consistent, so the supervised ceiling here is optimistic.
- Driver attribution is univariate. A record can be anomalous through an *interaction* of two individually unremarkable fields, and this method will not name it.
