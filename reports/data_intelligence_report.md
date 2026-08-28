# Data Intelligence Report

**Loan Performance Intelligence Engine — Task 1**  
Generated from `loan_panel.csv` and `servicer_updates.csv`.

## 1. Scope

- Records after de-duplication: **48,924**
- Distinct loans: **1,500**
- Reporting months: **2019-01 to 2026-06** (90 months)
- Servicers: **5**; states: **15**
- Secondary servicer feed: **1,014** duplicate loan-month records resolved latest-wins, **66** orphan records referencing loan-months absent from the panel.

## 2. Column distribution profiling

### Numeric fields

| column | missing_pct | mean | std | min | p01 | median | p99 | max | skew | negatives | iqr_outlier_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original_balance | 0.0000 | 252532.3154 | 122073.2688 | 53000.0000 | 81000.0000 | 227000.0000 | 640000.0000 | 1074000.0000 | 1.4089 | 0 | 0.0017 |
| current_balance | 0.0000 | 285080.2039 | 1076385.5693 | -673410.1900 | 72136.7968 | 216389.5800 | 650557.4296 | 60575229.2613 | 30.1534 | 56 | 0.0055 |
| interest_rate | 0.0138 | 5.1990 | 2.9605 | -1.0000 | 3.1070 | 4.9080 | 7.9770 | 99.9900 | 27.2447 | 34 | 0.0113 |
| loan_age_months | 0.0000 | 35.9008 | 28.2031 | -5.0000 | 0.0000 | 30.0000 | 115.0000 | 138.0000 | 0.9016 | 31 | 0.0000 |
| remaining_term_months | 0.0000 | 291.7516 | 70.1825 | 43.0000 | 94.0000 | 319.0000 | 360.0000 | 360.0000 | -1.3020 | 0 | 0.0000 |
| days_past_due | 0.0259 | 16.8971 | 363.5979 | -1.0000 | 0.0000 | 0.0000 | 106.0000 | 9999.0000 | 27.3545 | 56 | 0.0513 |
| reporting_lag_days | 0.0000 | 12.7825 | 20.8661 | -69.0000 | 1.0000 | 9.0000 | 134.0000 | 192.0000 | 4.5862 | 470 | 0.0430 |

### Categorical fields

| column | missing_pct | distinct | mode | mode_share | normalised_entropy | top_values |
| --- | --- | --- | --- | --- | --- | --- |
| credit_score_band | 0.0195 | 7 | 740-779 | 0.2305 | 0.9280 | 740-779=0.230; 700-739=0.220; 780+=0.174; 660-699=0.144; 620-659=0.143 |
| ltv_band | 0.0299 | 6 | 60-70 | 0.2781 | 0.9467 | 60-70=0.278; 70-80=0.198; 80-90=0.184; <=60=0.177; 90-95=0.105 |
| dti_band | 0.0631 | 5 | 20-30 | 0.2873 | 0.9603 | 20-30=0.287; 30-36=0.226; <=20=0.224; 36-43=0.179; >43=0.084 |
| state | 0.0000 | 15 | CA | 0.1825 | 0.9302 | CA=0.183; TX=0.136; FL=0.115; NY=0.093; IL=0.064 |
| loan_purpose | 0.0000 | 3 | purchase | 0.5960 | 0.8593 | purchase=0.596; rate_term_refi=0.248; cash_out_refi=0.156 |
| occupancy_type | 0.0213 | 3 | primary | 0.8044 | 0.5726 | primary=0.804; investment=0.106; second_home=0.089 |
| property_type | 0.0401 | 5 | single_family | 0.6728 | 0.6414 | single_family=0.673; condo=0.143; pud=0.107; 2-4_unit=0.048; manufactured=0.029 |
| servicer_name | 0.0000 | 5 | Northgate Servicing | 0.2966 | 0.9639 | Northgate Servicing=0.297; Belmont Loan Services=0.232; Arcadia Capital Servicing=0.214; Pioneer Mortgage Ops=0.155; Kestrel Financial=0.102 |
| current_status | 0.0000 | 4 | Current | 0.9562 | 0.1632 | Current=0.956; DQ90plus=0.020; DQ30=0.014; DQ60=0.010 |
| document_status | 0.0000 | 4 | complete | 0.8819 | 0.3511 | complete=0.882; missing=0.054; pending=0.041; exception=0.023 |
| source_system | 0.0000 | 3 | core_servicing | 0.7221 | 0.6879 | core_servicing=0.722; investor_feed=0.200; manual_upload=0.078 |
| loss_severity_band | 0.9954 | 5 | 0-10 | 0.3128 | 0.9659 | 0-10=0.313; 10-25=0.203; 40-60=0.198; 25-40=0.181; 60+=0.106 |

## 3. Missingness patterns

- Rows with at least one missing profiled field: **99.6%**
- Mean missing fields per row: **1.209**

Missingness is not random. A chi-square test of each field's missingness indicator against `servicer_name` rejects independence for the fields below, so the mechanism is **missing-at-random conditional on servicer**, not MCAR. Two servicers (Kestrel Financial, Pioneer Mortgage Ops) account for most of the gap. The practical consequence: dropping incomplete rows would silently drop those servicers' books and bias every downstream rate. Models therefore consume missingness natively and carry explicit missing-indicator features.

| column | chi2_vs_servicer | p_value | cramers_v | verdict |
| --- | --- | --- | --- | --- |
| dti_band | 1531.0896 | 0.0000 | 0.1769 | MAR (depends on servicer) |
| property_type | 823.3665 | 0.0000 | 0.1297 | MAR (depends on servicer) |
| ltv_band | 736.9079 | 0.0000 | 0.1227 | MAR (depends on servicer) |
| days_past_due | 519.1399 | 0.0000 | 0.1030 | MAR (depends on servicer) |
| credit_score_band | 418.8129 | 0.0000 | 0.0925 | MAR (depends on servicer) |
| occupancy_type | 358.9984 | 0.0000 | 0.0857 | MAR (depends on servicer) |
| interest_rate | 276.1161 | 0.0000 | 0.0751 | MAR (depends on servicer) |
| loss_severity_band | 1.9869 | 0.7382 | 0.0064 | consistent with MCAR |

### Co-missingness (fields that go missing together)

| field_a | field_b | missingness_correlation |
| --- | --- | --- |
| days_past_due | dti_band | 0.0255 |
| interest_rate | dti_band | 0.0242 |
| dti_band | property_type | 0.0232 |
| ltv_band | dti_band | 0.0185 |
| days_past_due | property_type | 0.0172 |
| credit_score_band | dti_band | 0.0144 |
| credit_score_band | property_type | 0.0141 |
| occupancy_type | property_type | 0.0138 |

### Missingness by servicer

| servicer_name | interest_rate | days_past_due | credit_score_band | ltv_band | dti_band | occupancy_type | property_type | loss_severity_band |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Arcadia Capital Servicing | 0.0096 | 0.0185 | 0.0115 | 0.0177 | 0.0395 | 0.0156 | 0.0268 | 0.9957 |
| Belmont Loan Services | 0.0102 | 0.0212 | 0.0178 | 0.0246 | 0.0559 | 0.0198 | 0.0352 | 0.9950 |
| Kestrel Financial | 0.0351 | 0.0647 | 0.0500 | 0.0821 | 0.1620 | 0.0484 | 0.1004 | 0.9946 |
| Northgate Servicing | 0.0076 | 0.0126 | 0.0095 | 0.0151 | 0.0273 | 0.0098 | 0.0193 | 0.9958 |
| Pioneer Mortgage Ops | 0.0226 | 0.0429 | 0.0323 | 0.0490 | 0.1099 | 0.0355 | 0.0662 | 0.9951 |

## 4. Outliers, sentinels and invalid dates

Sentinel values are treated as *absence of information*, not as extreme numbers. `days_past_due` of 9999 or -1, note rates of 0 / 99.99 / -1, and balances above 3x original are masked to missing with a `*_repaired` indicator retained, so the fact that a repair happened stays available as a feature.

| repair | rows | rate |
| --- | --- | --- |
| days_past_due sentinel/out-of-range masked | 119 | 0.0024 |
| interest_rate out-of-range masked | 103 | 0.0021 |
| current_balance implausible masked | 195 | 0.0040 |
| loan_age_months recomputed from dates | 451 | 0.0092 |

### Recovery against the injected ground truth

The synthetic generator logs every defect it injects. Comparing detection against that log is how this rule set was validated rather than merely asserted.

| defect | rows_affected | rate |
| --- | --- | --- |
| missing_dti_band | 3086 | 0.0631 |
| missing_ltv_band | 1465 | 0.0299 |
| missing_property_type | 1963 | 0.0401 |
| missing_credit_score_band | 955 | 0.0195 |
| missing_occupancy_type | 1042 | 0.0213 |
| missing_interest_rate | 674 | 0.0138 |
| missing_days_past_due | 1287 | 0.0263 |
| outlier_balance_inflated | 139 | 0.0028 |
| outlier_balance_negative | 56 | 0.0011 |
| outlier_interest_rate | 103 | 0.0021 |
| sentinel_days_past_due | 119 | 0.0024 |
| status_dpd_mismatch | 372 | 0.0076 |
| invalid_origination_after_reporting | 196 | 0.0040 |
| inconsistent_loan_age | 278 | 0.0057 |
| invalid_last_updated_before_period | 470 | 0.0096 |
| duplicate_rows | 192 | 0.0039 |

## 5. Cross-column relationship breaks

17 named rules run over every record, grouped into completeness, validity, consistency, plausibility, timeliness and reconciliation dimensions.

| rule | dimension | severity | violations | violation_rate | description |
| --- | --- | --- | --- | --- | --- |
| servicer_record_absent | reconciliation | 3.0000 | 33136 | 0.6773 | No servicer feed record exists for this loan month. |
| missing_critical_field | completeness | 9.0000 | 5257 | 0.1075 | A field required for credit assessment is missing. |
| document_file_incomplete | completeness | 7.0000 | 3775 | 0.0772 | Document custody status is missing or in exception. |
| servicer_balance_break | reconciliation | 13.0000 | 1805 | 0.0369 | Servicer feed balance differs from the panel by >1% and >$500. |
| stale_servicer_reporting | timeliness | 6.0000 | 1322 | 0.0270 | Record last updated more than 75 days after the period closed. |
| servicer_status_conflict | reconciliation | 11.0000 | 606 | 0.0124 | Servicer feed reports a different performance status than the panel. |
| last_updated_before_period_end | validity | 8.0000 | 470 | 0.0096 | Servicing record was last written before the reporting period closed. |
| loan_age_inconsistent_with_dates | consistency | 9.0000 | 451 | 0.0092 | Reported loan age disagrees with reporting minus origination month by >2 months. |
| remaining_term_inconsistent | consistency | 6.0000 | 277 | 0.0057 | Remaining term plus loan age is not a standard contractual term. |
| origination_after_reporting | validity | 14.0000 | 196 | 0.0040 | Origination month is later than the reporting month. |
| balance_increase_month_over_month | consistency | 7.0000 | 185 | 0.0038 | Unpaid principal balance rose month over month on a non-modified loan. |
| status_dpd_mismatch | consistency | 11.0000 | 142 | 0.0029 | Days past due is inconsistent with the reported performance status. |
| balance_exceeds_original | plausibility | 12.0000 | 139 | 0.0028 | Current balance exceeds original balance by more than 2%. |
| dpd_sentinel_value | validity | 10.0000 | 119 | 0.0024 | Days past due carries a sentinel value (9999, -1). |
| interest_rate_out_of_range | validity | 10.0000 | 103 | 0.0021 | Note rate outside a plausible 0.5%-25% range. |
| negative_balance | validity | 16.0000 | 56 | 0.0011 | Current balance is negative. |
| terminal_status_with_balance | consistency | 12.0000 | 0 | 0.0000 | Loan is in a terminal status but still carries a material balance. |

## 6. Correlation and dependent-field analysis

### Numeric (Spearman)

| field | original_balance | current_balance | interest_rate | loan_age_months | remaining_term_months | days_past_due | reporting_lag_days |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original_balance | 1.0000 | 0.9820 | -0.0350 | -0.0040 | -0.0140 | 0.0070 | 0.0040 |
| current_balance | 0.9820 | 1.0000 | -0.0080 | -0.0890 | 0.0720 | 0.0030 | 0.0050 |
| interest_rate | -0.0350 | -0.0080 | 1.0000 | -0.1640 | 0.0890 | 0.0430 | 0.0080 |
| loan_age_months | -0.0040 | -0.0890 | -0.1640 | 1.0000 | -0.6460 | 0.0730 | -0.0080 |
| remaining_term_months | -0.0140 | 0.0720 | 0.0890 | -0.6460 | 1.0000 | -0.0610 | 0.0030 |
| days_past_due | 0.0070 | 0.0030 | 0.0430 | 0.0730 | -0.0610 | 1.0000 | -0.0040 |
| reporting_lag_days | 0.0040 | 0.0050 | 0.0080 | -0.0080 | 0.0030 | -0.0040 | 1.0000 |

### Categorical association (bias-corrected Cramer's V, top pairs)

| field_a | field_b | cramers_v |
| --- | --- | --- |
| credit_score_band | dti_band | 0.1960 |
| credit_score_band | ltv_band | 0.1648 |
| dti_band | state | 0.1314 |
| ltv_band | state | 0.1276 |
| state | occupancy_type | 0.1245 |
| state | loan_purpose | 0.1187 |
| state | property_type | 0.1172 |
| state | servicer_name | 0.1119 |
| credit_score_band | state | 0.1102 |
| credit_score_band | property_type | 0.0991 |
| credit_score_band | current_status | 0.0987 |
| ltv_band | loan_purpose | 0.0886 |

### Functional dependencies

A loan's static attributes must not change across its reporting months. Violations here are true data-integrity breaks rather than statistical noise.

| determinant | dependent | groups | violating_groups | holds | violation_rate |
| --- | --- | --- | --- | --- | --- |
| loan_id | origination_month | 1500 | 180 | False | 0.1200 |
| loan_id | credit_score_band | 1499 | 0 | True | 0.0000 |
| loan_id | original_balance | 1500 | 0 | True | 0.0000 |
| loan_id | state | 1500 | 0 | True | 0.0000 |
| loan_id | servicer_name | 1500 | 0 | True | 0.0000 |
| current_status | expected_dpd | 4 | 0 | True | 0.0000 |

## 7. Train / test drift

Split at `2024-06`, matching the time-aware modelling split used in Task 2. PSI below 0.10 is stable, 0.10-0.25 moderate, above 0.25 severe.

| column | psi | ks_statistic | train_missing_pct | test_missing_pct | severity |
| --- | --- | --- | --- | --- | --- |
| interest_rate | 0.7329 | 0.3568 | 0.0142 | 0.0129 | severe |
| loan_age_months | 0.1663 | 0.1798 | 0.0000 | 0.0000 | moderate |
| remaining_term_months | 0.1479 | 0.1354 | 0.0000 | 0.0000 | moderate |
| loss_severity_band | 0.0313 |  | 0.9960 | 0.9940 | stable |
| current_balance | 0.0198 | 0.0574 | 0.0000 | 0.0000 | stable |
| credit_score_band | 0.0177 |  | 0.0192 | 0.0203 | stable |
| state | 0.0161 |  | 0.0000 | 0.0000 | stable |
| original_balance | 0.0064 | 0.0292 | 0.0000 | 0.0000 | stable |
| ltv_band | 0.0041 |  | 0.0291 | 0.0317 | stable |
| current_status | 0.0039 |  | 0.0000 | 0.0000 | stable |
| dti_band | 0.0028 |  | 0.0640 | 0.0611 | stable |
| loan_purpose | 0.0025 |  | 0.0000 | 0.0000 | stable |
| servicer_name | 0.0021 |  | 0.0000 | 0.0000 | stable |
| reporting_lag_days | 0.0018 | 0.0123 | 0.0000 | 0.0000 | stable |
| property_type | 0.0015 |  | 0.0402 | 0.0399 | stable |
| source_system | 0.0002 |  | 0.0000 | 0.0000 | stable |
| days_past_due | 0.0002 | 0.0122 | 0.0234 | 0.0312 | stable |
| occupancy_type | 0.0002 |  | 0.0213 | 0.0212 | stable |
| document_status | 0.0000 |  | 0.0000 | 0.0000 | stable |

### Target stability across months

| target | overall_rate | min_month_rate | max_month_rate | std_across_months | censored_rows |
| --- | --- | --- | --- | --- | --- |
| next_3m_delinquency_flag | 0.0639 | 0.0201 | 0.7347 | 0.1096 | 1502 |
| next_6m_delinquency_flag | 0.0852 | 0.0436 | 0.7347 | 0.1316 | 2996 |
| next_12m_default_flag | 0.0616 | 0.0328 | 0.3023 | 0.0727 | 6318 |
| next_12m_prepayment_flag | 0.1549 | 0.0341 | 0.7750 | 0.2172 | 6318 |
| exception_required | 0.1258 | 0.0880 | 0.1670 | 0.0164 | 0 |

## 8. Data quality scoring

Record score = 100 minus the severity-weighted sum of rule violations, floored at 0. Batch score aggregates the same violations to the (reporting month x servicer) grain, which is the level an oversight team can act on.

| dq_band | records | share |
| --- | --- | --- |
| clean | 39500 | 0.8074 |
| watch | 9085 | 0.1857 |
| poor | 339 | 0.0069 |
| critical | 0 | 0.0000 |

- Mean record DQ score: **95.28**
- Median record DQ score: **97.00**
- Records with at least one violation: **78.6%**

### Batch grades by servicer

| servicer_name | records | mean_dq_score | violations_per_record |
| --- | --- | --- | --- |
| Kestrel Financial | 4980 | 92.7878 | 1.2859 |
| Pioneer Mortgage Ops | 7595 | 94.1609 | 1.1211 |
| Belmont Loan Services | 11346 | 95.5492 | 0.9486 |
| Arcadia Capital Servicing | 10493 | 95.8437 | 0.9099 |
| Northgate Servicing | 14510 | 96.0924 | 0.8828 |

### Ten worst batches

| reporting_month | servicer_name | records | mean_dq_score | pct_critical | top_failing_rule | top_failing_rule_rate | batch_grade |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2021-07 | Kestrel Financial | 43 | 90.3256 | 0.0000 | servicer_record_absent | 0.6977 | B |
| 2020-12 | Kestrel Financial | 50 | 91.4400 | 0.0000 | servicer_record_absent | 0.7000 | A |
| 2023-08 | Kestrel Financial | 57 | 91.5088 | 0.0000 | servicer_record_absent | 0.7193 | A |
| 2024-11 | Kestrel Financial | 67 | 91.5821 | 0.0000 | servicer_record_absent | 0.7313 | A |
| 2020-07 | Kestrel Financial | 55 | 91.6545 | 0.0000 | servicer_record_absent | 0.6727 | A |
| 2024-12 | Kestrel Financial | 66 | 91.7121 | 0.0000 | servicer_record_absent | 0.7121 | A |
| 2019-12 | Kestrel Financial | 53 | 91.7547 | 0.0000 | servicer_record_absent | 0.6981 | A |
| 2019-02 | Kestrel Financial | 59 | 91.9153 | 0.0000 | servicer_record_absent | 0.6610 | A |
| 2024-04 | Kestrel Financial | 64 | 91.9219 | 0.0000 | servicer_record_absent | 0.6719 | A |
| 2023-12 | Kestrel Financial | 59 | 92.0000 | 0.0000 | servicer_record_absent | 0.7119 | A |

## 9. What this means for modelling

1. **Servicer is a confound, not just a feature.** Kestrel Financial and Pioneer Mortgage Ops have both the worst data quality *and* elevated delinquency. A model given raw servicer identity will partly learn reporting behaviour rather than credit risk. Servicer is retained but its SHAP contribution is inspected separately in the explainability report.
2. **Censoring is real and material.** Forward-looking targets are undefined for rows whose horizon runs past the panel end. These are `NaN`, not `0`, and are excluded from supervised training rather than counted as non-events.
3. **Repairs are features.** Whether a record needed repair is predictive of whether it needs an exception, so repair indicators are carried forward rather than discarded.
4. **Drift is concentrated in macro-sensitive fields**, which is expected given the rate path in the panel window and is handled by time-aware validation rather than by reweighting.
