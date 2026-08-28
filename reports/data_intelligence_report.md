# Data Intelligence Report

**Loan Performance Intelligence Engine — Task 1**  
Generated from `loan_panel.csv` and `servicer_updates.csv`.

## 1. Scope

- Records after de-duplication: **53,756**
- Distinct loans: **1,900**
- Reporting months: **2022-01 to 2026-06** (54 months)
- Servicers: **5**; states: **15**
- Secondary servicer feed: **1,131** duplicate loan-month records resolved latest-wins, **74** orphan records referencing loan-months absent from the panel.

## 2. Column distribution profiling

### Numeric fields

| column | missing_pct | mean | std | min | p01 | median | p99 | max | skew | negatives | iqr_outlier_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original_balance | 0.0000 | 253605.6626 | 118613.4405 | 51000.0000 | 85000.0000 | 231000.0000 | 640000.0000 | 1074000.0000 | 1.5222 | 0 | 0.0058 |
| current_balance | 0.0000 | 291338.7658 | 1072119.6973 | -742233.9100 | 79148.1380 | 223473.9950 | 670572.2120 | 58589923.0031 | 27.4975 | 66 | 0.0081 |
| interest_rate | 0.0144 | 4.5413 | 2.6764 | -1.0000 | 3.0490 | 4.0530 | 7.2700 | 99.9900 | 29.1711 | 31 | 0.0012 |
| loan_age_months | 0.0000 | 33.1874 | 22.6899 | -5.0000 | 0.0000 | 30.0000 | 89.0000 | 103.0000 | 0.5210 | 37 | 0.0000 |
| remaining_term_months | 0.0000 | 292.4428 | 69.3165 | 79.0000 | 113.0000 | 319.0000 | 360.0000 | 360.0000 | -1.2618 | 0 | 0.0039 |
| days_past_due | 0.0261 | 16.9815 | 365.6727 | -1.0000 | 0.0000 | 0.0000 | 107.0000 | 9999.0000 | 27.2007 | 67 | 0.0509 |
| reporting_lag_days | 0.0000 | 12.8725 | 21.0359 | -69.0000 | 1.0000 | 9.0000 | 135.4500 | 196.0000 | 4.5807 | 517 | 0.0439 |

### Categorical fields

| column | missing_pct | distinct | mode | mode_share | normalised_entropy | top_values |
| --- | --- | --- | --- | --- | --- | --- |
| credit_score_band | 0.0189 | 7 | 740-779 | 0.2224 | 0.9324 | 740-779=0.222; 700-739=0.214; 660-699=0.171; 780+=0.162; 620-659=0.138 |
| ltv_band | 0.0295 | 6 | 60-70 | 0.2234 | 0.9614 | 60-70=0.223; 70-80=0.214; 80-90=0.191; <=60=0.191; 90-95=0.118 |
| dti_band | 0.0628 | 5 | 20-30 | 0.2798 | 0.9685 | 20-30=0.280; 30-36=0.218; <=20=0.214; 36-43=0.194; >43=0.094 |
| state | 0.0000 | 15 | CA | 0.1633 | 0.9329 | CA=0.163; TX=0.139; FL=0.117; NY=0.105; NC=0.062 |
| loan_purpose | 0.0000 | 3 | purchase | 0.5723 | 0.8834 | purchase=0.572; rate_term_refi=0.257; cash_out_refi=0.171 |
| occupancy_type | 0.0214 | 3 | primary | 0.8244 | 0.5304 | primary=0.824; investment=0.106; second_home=0.070 |
| property_type | 0.0418 | 5 | single_family | 0.6882 | 0.6291 | single_family=0.688; condo=0.134; pud=0.096; 2-4_unit=0.046; manufactured=0.036 |
| servicer_name | 0.0000 | 5 | Northgate Servicing | 0.3201 | 0.9560 | Northgate Servicing=0.320; Belmont Loan Services=0.241; Arcadia Capital Servicing=0.192; Pioneer Mortgage Ops=0.140; Kestrel Financial=0.107 |
| current_status | 0.0000 | 4 | Current | 0.9566 | 0.1626 | Current=0.957; DQ90plus=0.018; DQ30=0.014; DQ60=0.011 |
| document_status | 0.0000 | 4 | complete | 0.8859 | 0.3423 | complete=0.886; missing=0.051; pending=0.040; exception=0.023 |
| source_system | 0.0000 | 3 | core_servicing | 0.7185 | 0.6952 | core_servicing=0.719; investor_feed=0.200; manual_upload=0.082 |
| loss_severity_band | 0.9951 | 5 | 0-10 | 0.2443 | 0.9894 | 0-10=0.244; 25-40=0.233; 40-60=0.202; 10-25=0.176; 60+=0.145 |

## 3. Missingness patterns

- Rows with at least one missing profiled field: **99.6%**
- Mean missing fields per row: **1.210**

Missingness is not random. A chi-square test of each field's missingness indicator against `servicer_name` rejects independence for the fields below, so the mechanism is **missing-at-random conditional on servicer**, not MCAR. Two servicers (Kestrel Financial, Pioneer Mortgage Ops) account for most of the gap. The practical consequence: dropping incomplete rows would silently drop those servicers' books and bias every downstream rate. Models therefore consume missingness natively and carry explicit missing-indicator features.

| column | chi2_vs_servicer | p_value | cramers_v | verdict |
| --- | --- | --- | --- | --- |
| dti_band | 1573.9729 | 0.0000 | 0.1711 | MAR (depends on servicer) |
| property_type | 954.9636 | 0.0000 | 0.1333 | MAR (depends on servicer) |
| ltv_band | 817.9155 | 0.0000 | 0.1234 | MAR (depends on servicer) |
| days_past_due | 669.3321 | 0.0000 | 0.1116 | MAR (depends on servicer) |
| occupancy_type | 495.0254 | 0.0000 | 0.0960 | MAR (depends on servicer) |
| credit_score_band | 470.5454 | 0.0000 | 0.0936 | MAR (depends on servicer) |
| interest_rate | 365.2695 | 0.0000 | 0.0824 | MAR (depends on servicer) |
| loss_severity_band | 6.0418 | 0.1960 | 0.0106 | consistent with MCAR |

### Co-missingness (fields that go missing together)

| field_a | field_b | missingness_correlation |
| --- | --- | --- |
| credit_score_band | dti_band | 0.0254 |
| ltv_band | dti_band | 0.0247 |
| dti_band | occupancy_type | 0.0242 |
| dti_band | property_type | 0.0238 |
| ltv_band | occupancy_type | 0.0206 |
| days_past_due | occupancy_type | 0.0184 |
| interest_rate | dti_band | 0.0177 |
| credit_score_band | property_type | 0.0174 |

### Missingness by servicer

| servicer_name | interest_rate | days_past_due | credit_score_band | ltv_band | dti_band | occupancy_type | property_type | loss_severity_band |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Arcadia Capital Servicing | 0.0086 | 0.0176 | 0.0106 | 0.0193 | 0.0397 | 0.0149 | 0.0289 | 0.9945 |
| Belmont Loan Services | 0.0118 | 0.0213 | 0.0176 | 0.0237 | 0.0542 | 0.0183 | 0.0358 | 0.9964 |
| Kestrel Financial | 0.0344 | 0.0682 | 0.0473 | 0.0791 | 0.1556 | 0.0525 | 0.1021 | 0.9944 |
| Northgate Servicing | 0.0069 | 0.0126 | 0.0089 | 0.0139 | 0.0304 | 0.0101 | 0.0203 | 0.9948 |
| Pioneer Mortgage Ops | 0.0284 | 0.0450 | 0.0338 | 0.0512 | 0.1126 | 0.0380 | 0.0728 | 0.9952 |

## 4. Outliers, sentinels and invalid dates

Sentinel values are treated as *absence of information*, not as extreme numbers. `days_past_due` of 9999 or -1, note rates of 0 / 99.99 / -1, and balances above 3x original are masked to missing with a `*_repaired` indicator retained, so the fact that a repair happened stays available as a feature.

| repair | rows | rate |
| --- | --- | --- |
| days_past_due sentinel/out-of-range masked | 137 | 0.0025 |
| interest_rate out-of-range masked | 83 | 0.0015 |
| current_balance implausible masked | 221 | 0.0041 |
| loan_age_months recomputed from dates | 513 | 0.0095 |

### Recovery against the injected ground truth

The synthetic generator logs every defect it injects. Comparing detection against that log is how this rule set was validated rather than merely asserted.

| defect | rows_affected | rate |
| --- | --- | --- |
| missing_dti_band | 3376 | 0.0628 |
| missing_ltv_band | 1584 | 0.0295 |
| missing_property_type | 2245 | 0.0418 |
| missing_credit_score_band | 1018 | 0.0189 |
| missing_occupancy_type | 1152 | 0.0214 |
| missing_interest_rate | 772 | 0.0144 |
| missing_days_past_due | 1423 | 0.0265 |
| outlier_balance_inflated | 155 | 0.0029 |
| outlier_balance_negative | 66 | 0.0012 |
| outlier_interest_rate | 83 | 0.0015 |
| sentinel_days_past_due | 138 | 0.0026 |
| status_dpd_mismatch | 403 | 0.0075 |
| invalid_origination_after_reporting | 216 | 0.0040 |
| inconsistent_loan_age | 316 | 0.0059 |
| invalid_last_updated_before_period | 517 | 0.0096 |
| duplicate_rows | 216 | 0.0040 |

## 5. Cross-column relationship breaks

17 named rules run over every record, grouped into completeness, validity, consistency, plausibility, timeliness and reconciliation dimensions.

| rule | dimension | severity | violations | violation_rate | description |
| --- | --- | --- | --- | --- | --- |
| servicer_record_absent | reconciliation | 3.0000 | 36119 | 0.6719 | No servicer feed record exists for this loan month. |
| missing_critical_field | completeness | 9.0000 | 5671 | 0.1055 | A field required for credit assessment is missing. |
| document_file_incomplete | completeness | 7.0000 | 3979 | 0.0740 | Document custody status is missing or in exception. |
| servicer_balance_break | reconciliation | 13.0000 | 2025 | 0.0377 | Servicer feed balance differs from the panel by >1% and >$500. |
| stale_servicer_reporting | timeliness | 6.0000 | 1494 | 0.0278 | Record last updated more than 75 days after the period closed. |
| servicer_status_conflict | reconciliation | 11.0000 | 708 | 0.0132 | Servicer feed reports a different performance status than the panel. |
| last_updated_before_period_end | validity | 8.0000 | 517 | 0.0096 | Servicing record was last written before the reporting period closed. |
| loan_age_inconsistent_with_dates | consistency | 9.0000 | 513 | 0.0095 | Reported loan age disagrees with reporting minus origination month by >2 months. |
| remaining_term_inconsistent | consistency | 6.0000 | 310 | 0.0058 | Remaining term plus loan age is not a standard contractual term. |
| origination_after_reporting | validity | 14.0000 | 216 | 0.0040 | Origination month is later than the reporting month. |
| balance_increase_month_over_month | consistency | 7.0000 | 213 | 0.0040 | Unpaid principal balance rose month over month on a non-modified loan. |
| status_dpd_mismatch | consistency | 11.0000 | 172 | 0.0032 | Days past due is inconsistent with the reported performance status. |
| balance_exceeds_original | plausibility | 12.0000 | 155 | 0.0029 | Current balance exceeds original balance by more than 2%. |
| dpd_sentinel_value | validity | 10.0000 | 137 | 0.0025 | Days past due carries a sentinel value (9999, -1). |
| interest_rate_out_of_range | validity | 10.0000 | 83 | 0.0015 | Note rate outside a plausible 0.5%-25% range. |
| negative_balance | validity | 16.0000 | 66 | 0.0012 | Current balance is negative. |
| terminal_status_with_balance | consistency | 12.0000 | 0 | 0.0000 | Loan is in a terminal status but still carries a material balance. |

## 6. Correlation and dependent-field analysis

### Numeric (Spearman)

| field | original_balance | current_balance | interest_rate | loan_age_months | remaining_term_months | days_past_due | reporting_lag_days |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original_balance | 1.0000 | 0.9880 | -0.0280 | 0.0090 | 0.0360 | -0.0060 | -0.0040 |
| current_balance | 0.9880 | 1.0000 | -0.0050 | -0.0420 | 0.0910 | -0.0080 | -0.0030 |
| interest_rate | -0.0280 | -0.0050 | 1.0000 | -0.5940 | 0.3580 | 0.0550 | 0.0010 |
| loan_age_months | 0.0090 | -0.0420 | -0.5940 | 1.0000 | -0.6250 | 0.0500 | 0.0040 |
| remaining_term_months | 0.0360 | 0.0910 | 0.3580 | -0.6250 | 1.0000 | -0.0210 | -0.0020 |
| days_past_due | -0.0060 | -0.0080 | 0.0550 | 0.0500 | -0.0210 | 1.0000 | 0.0070 |
| reporting_lag_days | -0.0040 | -0.0030 | 0.0010 | 0.0040 | -0.0020 | 0.0070 | 1.0000 |

### Categorical association (bias-corrected Cramer's V, top pairs)

| field_a | field_b | cramers_v |
| --- | --- | --- |
| credit_score_band | dti_band | 0.2011 |
| credit_score_band | ltv_band | 0.1956 |
| credit_score_band | current_status | 0.1131 |
| state | property_type | 0.1023 |
| credit_score_band | state | 0.1017 |
| ltv_band | dti_band | 0.0997 |
| state | occupancy_type | 0.0957 |
| ltv_band | state | 0.0945 |
| state | loan_purpose | 0.0928 |
| state | servicer_name | 0.0911 |
| dti_band | state | 0.0908 |
| occupancy_type | servicer_name | 0.0868 |

### Functional dependencies

A loan's static attributes must not change across its reporting months. Violations here are true data-integrity breaks rather than statistical noise.

| determinant | dependent | groups | violating_groups | holds | violation_rate |
| --- | --- | --- | --- | --- | --- |
| loan_id | origination_month | 1900 | 201 | False | 0.1058 |
| loan_id | credit_score_band | 1899 | 0 | True | 0.0000 |
| loan_id | original_balance | 1900 | 0 | True | 0.0000 |
| loan_id | state | 1900 | 0 | True | 0.0000 |
| loan_id | servicer_name | 1900 | 0 | True | 0.0000 |
| current_status | expected_dpd | 4 | 0 | True | 0.0000 |

## 7. Train / test drift

Split at `2025-03`, matching the time-aware modelling split used in Task 2. PSI below 0.10 is stable, 0.10-0.25 moderate, above 0.25 severe.

| column | psi | ks_statistic | train_missing_pct | test_missing_pct | severity |
| --- | --- | --- | --- | --- | --- |
| interest_rate | 0.5415 | 0.3020 | 0.0144 | 0.0143 | severe |
| loss_severity_band | 0.1998 |  | 0.9960 | 0.9929 | moderate |
| loan_age_months | 0.1887 | 0.1741 | 0.0000 | 0.0000 | moderate |
| remaining_term_months | 0.1378 | 0.1311 | 0.0000 | 0.0000 | moderate |
| current_balance | 0.0172 | 0.0395 | 0.0000 | 0.0000 | stable |
| credit_score_band | 0.0162 |  | 0.0192 | 0.0182 | stable |
| state | 0.0122 |  | 0.0000 | 0.0000 | stable |
| current_status | 0.0095 |  | 0.0000 | 0.0000 | stable |
| ltv_band | 0.0064 |  | 0.0297 | 0.0290 | stable |
| dti_band | 0.0057 |  | 0.0635 | 0.0609 | stable |
| original_balance | 0.0031 | 0.0148 | 0.0000 | 0.0000 | stable |
| loan_purpose | 0.0030 |  | 0.0000 | 0.0000 | stable |
| property_type | 0.0018 |  | 0.0420 | 0.0411 | stable |
| servicer_name | 0.0017 |  | 0.0000 | 0.0000 | stable |
| reporting_lag_days | 0.0017 | 0.0091 | 0.0000 | 0.0000 | stable |
| source_system | 0.0004 |  | 0.0000 | 0.0000 | stable |
| occupancy_type | 0.0003 |  | 0.0216 | 0.0209 | stable |
| document_status | 0.0001 |  | 0.0000 | 0.0000 | stable |
| days_past_due | 0.0000 | 0.0202 | 0.0217 | 0.0378 | stable |

### Target stability across months

| target | overall_rate | min_month_rate | max_month_rate | std_across_months | censored_rows |
| --- | --- | --- | --- | --- | --- |
| next_3m_delinquency_flag | 0.0670 | 0.0162 | 0.8730 | 0.1714 | 2420 |
| next_6m_delinquency_flag | 0.0933 | 0.0304 | 0.8730 | 0.2057 | 4819 |
| next_12m_default_flag | 0.0697 | 0.0172 | 0.4138 | 0.1248 | 10175 |
| next_12m_prepayment_flag | 0.1566 | 0.1126 | 0.7273 | 0.2149 | 10175 |
| exception_required | 0.1249 | 0.0975 | 0.1515 | 0.0104 | 0 |

## 8. Data quality scoring

Record score = 100 minus the severity-weighted sum of rule violations, floored at 0. Batch score aggregates the same violations to the (reporting month x servicer) grain, which is the level an oversight team can act on.

| dq_band | records | share |
| --- | --- | --- |
| clean | 43488 | 0.8090 |
| watch | 9879 | 0.1838 |
| poor | 388 | 0.0072 |
| critical | 1 | 0.0000 |

- Mean record DQ score: **95.30**
- Median record DQ score: **97.00**
- Records with at least one violation: **78.1%**

### Batch grades by servicer

| servicer_name | records | mean_dq_score | violations_per_record |
| --- | --- | --- | --- |
| Kestrel Financial | 5752 | 92.9444 | 1.2615 |
| Pioneer Mortgage Ops | 7505 | 94.0412 | 1.1294 |
| Belmont Loan Services | 12954 | 95.5368 | 0.9515 |
| Arcadia Capital Servicing | 10336 | 95.8637 | 0.9003 |
| Northgate Servicing | 17209 | 96.1306 | 0.8725 |

### Ten worst batches

| reporting_month | servicer_name | records | mean_dq_score | pct_critical | top_failing_rule | top_failing_rule_rate | batch_grade |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-04 | Kestrel Financial | 116 | 91.4483 | 0.0000 | servicer_record_absent | 0.7241 | A |
| 2022-02 | Kestrel Financial | 104 | 91.6250 | 0.0000 | servicer_record_absent | 0.7115 | A |
| 2025-12 | Kestrel Financial | 97 | 91.8660 | 0.0000 | servicer_record_absent | 0.6598 | A |
| 2023-10 | Kestrel Financial | 110 | 92.0727 | 0.0000 | servicer_record_absent | 0.7182 | A |
| 2023-12 | Kestrel Financial | 112 | 92.1339 | 0.0000 | servicer_record_absent | 0.6161 | A |
| 2025-03 | Kestrel Financial | 112 | 92.1518 | 0.0000 | servicer_record_absent | 0.6964 | A |
| 2026-03 | Kestrel Financial | 95 | 92.3053 | 0.0000 | servicer_record_absent | 0.6632 | A |
| 2025-04 | Kestrel Financial | 109 | 92.3119 | 0.0000 | servicer_record_absent | 0.6514 | A |
| 2025-09 | Kestrel Financial | 106 | 92.3302 | 0.0000 | servicer_record_absent | 0.6509 | A |
| 2024-10 | Kestrel Financial | 117 | 92.3932 | 0.0000 | servicer_record_absent | 0.6410 | A |

## 9. What this means for modelling

1. **Servicer is a confound, not just a feature.** Kestrel Financial and Pioneer Mortgage Ops have both the worst data quality *and* elevated delinquency. A model given raw servicer identity will partly learn reporting behaviour rather than credit risk. Servicer is retained but its SHAP contribution is inspected separately in the explainability report.
2. **Censoring is real and material.** Forward-looking targets are undefined for rows whose horizon runs past the panel end. These are `NaN`, not `0`, and are excluded from supervised training rather than counted as non-events.
3. **Repairs are features.** Whether a record needed repair is predictive of whether it needs an exception, so repair indicators are carried forward rather than discarded.
4. **Drift is concentrated in macro-sensitive fields**, which is expected given the rate path in the panel window and is handled by time-aware validation rather than by reweighting.
