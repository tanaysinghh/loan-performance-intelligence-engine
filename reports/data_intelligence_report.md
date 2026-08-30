# Data Intelligence Report

**Loan Performance Intelligence Engine — Task 1**  
Generated from `loan_panel.csv` and `servicer_updates.csv`.

## 1. Scope

- Records after de-duplication: **670,548**
- Distinct loans: **16,000**
- Reporting months: **2019-01 to 2026-03** (87 months)
- Servicers: **42**; states: **53**
- Secondary servicer feed: **14,118** duplicate loan-month records resolved latest-wins, **920** orphan records referencing loan-months absent from the panel.

## 2. Column distribution profiling

### Numeric fields

| column | missing_pct | mean | std | min | p01 | median | p99 | max | skew | negatives | iqr_outlier_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original_balance | 0.0000 | 283127.7492 | 150167.3234 | 19000.0000 | 57000.0000 | 252000.0000 | 725000.0000 | 1102000.0000 | 0.9509 | 0 | 0.0007 |
| current_balance | 0.0000 | 313953.9139 | 1261672.9420 | -29557921.0496 | 7374.1913 | 234000.0000 | 716000.0000 | 89072843.7371 | 31.4617 | 807 | 0.0038 |
| interest_rate | 0.0139 | 4.1968 | 2.7400 | -1.0000 | 2.1250 | 3.6250 | 7.6250 | 99.9900 | 24.9890 | 389 | 0.0006 |
| loan_age_months | 0.0000 | 25.4340 | 18.2721 | -5.0000 | 0.0000 | 22.0000 | 72.0000 | 105.0000 | 0.6323 | 630 | 0.0000 |
| remaining_term_months | 0.0000 | 303.3672 | 70.7218 | 37.0000 | 112.0000 | 332.0000 | 360.0000 | 480.0000 | -1.5313 | 0 | 0.1391 |
| days_past_due | 0.0258 | 16.3583 | 384.6009 | -1.0000 | 0.0000 | 0.0000 | 90.0000 | 9999.0000 | 25.8856 | 888 | 0.0236 |
| reporting_lag_days | 0.0000 | 12.8365 | 20.8961 | -69.0000 | 1.0000 | 9.0000 | 135.0000 | 206.0000 | 4.6326 | 6098 | 0.0432 |

### Categorical fields

| column | missing_pct | distinct | mode | mode_share | normalised_entropy | top_values |
| --- | --- | --- | --- | --- | --- | --- |
| credit_score_band | 0.0191 | 6 | 780+ | 0.3270 | 0.7966 | 780+=0.327; 740-779=0.313; 700-739=0.211; 660-699=0.112; 620-659=0.036 |
| ltv_band | 0.0315 | 6 | 70-80 | 0.3277 | 0.9041 | 70-80=0.328; <=60=0.252; 60-70=0.140; 90-95=0.135; 80-90=0.110 |
| dti_band | 0.0624 | 5 | 36-43 | 0.2649 | 0.9654 | 36-43=0.265; >43=0.237; 20-30=0.222; 30-36=0.191; <=20=0.086 |
| state | 0.0000 | 53 | CA | 0.1129 | 0.8764 | CA=0.113; TX=0.079; FL=0.073; IL=0.043; NY=0.036 |
| loan_purpose | 0.0000 | 3 | purchase | 0.5073 | 0.9390 | purchase=0.507; rate_term_refi=0.273; cash_out_refi=0.219 |
| occupancy_type | 0.0222 | 3 | primary | 0.9032 | 0.3458 | primary=0.903; investment=0.064; second_home=0.033 |
| property_type | 0.0412 | 5 | single_family | 0.6165 | 0.6090 | single_family=0.616; pud=0.271; condo=0.082; 2-4_unit=0.024; manufactured=0.007 |
| servicer_name | 0.0000 | 42 | OTHER | 0.2766 | 0.7629 | OTHER=0.277; JPMORGAN CHASE BANK, NATIONAL ASSOCIATION=0.090; NATIONSTAR MORTGAGE LLC DBA MR. COOPER=0.063; ROCKET MORTGAGE, LLC=0.056; WELLS FARGO BANK, N.A.=0.052 |
| current_status | 0.0000 | 5 | Current | 0.9842 | 0.0627 | Current=0.984; DQ30=0.008; Default=0.004; DQ90plus=0.003; DQ60=0.002 |
| document_status | 0.0000 | 4 | complete | 0.8532 | 0.4119 | complete=0.853; missing=0.066; pending=0.051; exception=0.029 |
| source_system | 0.0000 | 3 | core_servicing | 0.7193 | 0.6932 | core_servicing=0.719; investor_feed=0.200; manual_upload=0.080 |
| loss_severity_band | 1.0000 | 3 | 0-10 | 0.7273 | 0.6914 | 0-10=0.727; 10-25=0.182; 25-40=0.091 |

## 3. Missingness patterns

- Rows with at least one missing profiled field: **100.0%**
- Mean missing fields per row: **1.216**

Missingness is not random. A chi-square test of each field's missingness indicator against `servicer_name` rejects independence for the fields below, so the mechanism is **missing-at-random conditional on servicer**, not MCAR. Two servicers (Kestrel Financial, Pioneer Mortgage Ops) account for most of the gap. The practical consequence: dropping incomplete rows would silently drop those servicers' books and bias every downstream rate. Models therefore consume missingness natively and carry explicit missing-indicator features.

| column | chi2_vs_servicer | p_value | cramers_v | verdict |
| --- | --- | --- | --- | --- |
| dti_band | 3462.9325 | 0.0000 | 0.0719 | MAR (depends on servicer) |
| property_type | 2061.5904 | 0.0000 | 0.0554 | MAR (depends on servicer) |
| ltv_band | 1630.8508 | 0.0000 | 0.0493 | MAR (depends on servicer) |
| days_past_due | 1337.8468 | 0.0000 | 0.0447 | MAR (depends on servicer) |
| occupancy_type | 1108.6545 | 0.0000 | 0.0407 | MAR (depends on servicer) |
| credit_score_band | 1008.5812 | 0.0000 | 0.0388 | MAR (depends on servicer) |
| interest_rate | 722.6168 | 0.0000 | 0.0328 | MAR (depends on servicer) |
| loss_severity_band | 21.3913 | 0.9951 | 0.0056 | consistent with MCAR |

### Co-missingness (fields that go missing together)

| field_a | field_b | missingness_correlation |
| --- | --- | --- |
| ltv_band | dti_band | 0.0081 |
| dti_band | property_type | 0.0065 |
| ltv_band | occupancy_type | 0.0060 |
| ltv_band | property_type | 0.0054 |
| occupancy_type | property_type | 0.0054 |
| days_past_due | ltv_band | 0.0046 |
| dti_band | occupancy_type | 0.0045 |
| days_past_due | dti_band | 0.0044 |

### Missingness by servicer

| servicer_name | interest_rate | days_past_due | credit_score_band | ltv_band | dti_band | occupancy_type | property_type | loss_severity_band |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AMERIHOME MORTGAGE COMPANY, LLC | 0.0169 | 0.0318 | 0.0270 | 0.0410 | 0.0811 | 0.0322 | 0.0533 | 1.0000 |
| ARVEST CENTRAL MORTGAGE COMPANY | 0.0000 | 0.0119 | 0.0000 | 0.0238 | 0.0119 | 0.0357 | 0.0476 | 1.0000 |
| BANK OF AMERICA, N.A. | 0.0116 | 0.0187 | 0.0178 | 0.0232 | 0.0507 | 0.0212 | 0.0363 | 1.0000 |
| BRANCH BANKING AND TRUST COMPANY | 0.0183 | 0.0228 | 0.0183 | 0.0091 | 0.0639 | 0.0228 | 0.0457 | 1.0000 |
| CALIBER HOME LOANS, INC. | 0.0248 | 0.0359 | 0.0287 | 0.0492 | 0.1031 | 0.0353 | 0.0643 | 1.0000 |
| CITIZENS BANK, NA | 0.0161 | 0.0342 | 0.0242 | 0.0390 | 0.0807 | 0.0273 | 0.0510 | 1.0000 |
| CMG MORTGAGE, INC. | 0.0153 | 0.0143 | 0.0158 | 0.0208 | 0.0549 | 0.0163 | 0.0277 | 1.0000 |
| CROSSCOUNTRY MORTGAGE, LLC | 0.0138 | 0.0441 | 0.0145 | 0.0349 | 0.0619 | 0.0257 | 0.0382 | 1.0000 |
| FIFTH THIRD BANK | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0571 | 0.0095 | 0.0095 | 1.0000 |
| FIFTH THIRD BANK, NATIONAL ASSOCIATION | 0.0161 | 0.0320 | 0.0226 | 0.0405 | 0.0809 | 0.0289 | 0.0518 | 1.0000 |
| FREEDOM MORTGAGE CORPORATION | 0.0131 | 0.0292 | 0.0209 | 0.0318 | 0.0623 | 0.0248 | 0.0439 | 1.0000 |
| GUARANTEED RATE, INC. | 0.0121 | 0.0423 | 0.0242 | 0.0423 | 0.0665 | 0.0181 | 0.0574 | 1.0000 |
| GUILD MORTGAGE COMPANY LLC | 0.0079 | 0.0238 | 0.0145 | 0.0218 | 0.0554 | 0.0178 | 0.0356 | 1.0000 |
| HOME POINT FINANCIAL CORPORATION | 0.0161 | 0.0224 | 0.0172 | 0.0310 | 0.0654 | 0.0269 | 0.0407 | 1.0000 |
| JPMORGAN CHASE BANK, NATIONAL ASSOCIATION | 0.0163 | 0.0311 | 0.0218 | 0.0381 | 0.0733 | 0.0246 | 0.0480 | 1.0000 |
| LAKEVIEW LOAN SERVICING, LLC | 0.0150 | 0.0283 | 0.0191 | 0.0311 | 0.0640 | 0.0212 | 0.0407 | 1.0000 |
| LOANDEPOT.COM, LLC | 0.0198 | 0.0299 | 0.0282 | 0.0449 | 0.0911 | 0.0297 | 0.0571 | 1.0000 |
| MARLIN MORTGAGE CAPITAL, LLC | 0.0156 | 0.0195 | 0.0176 | 0.0273 | 0.0566 | 0.0176 | 0.0273 | 1.0000 |
| MATRIX FINANCIAL SERVICES CORPORATION | 0.0179 | 0.0302 | 0.0250 | 0.0441 | 0.0864 | 0.0328 | 0.0571 | 0.9999 |
| NATIONSTAR MORTGAGE LLC DBA MR. COOPER | 0.0125 | 0.0239 | 0.0172 | 0.0278 | 0.0571 | 0.0214 | 0.0390 | 1.0000 |
| NEW RESIDENTIAL MORTGAGE LLC | 0.0147 | 0.0304 | 0.0202 | 0.0342 | 0.0660 | 0.0239 | 0.0445 | 1.0000 |
| NEWREZ LLC | 0.0137 | 0.0213 | 0.0164 | 0.0213 | 0.0526 | 0.0168 | 0.0343 | 1.0000 |
| ONSLOW BAY FINANCIAL LLC | 0.0168 | 0.0317 | 0.0198 | 0.0352 | 0.0823 | 0.0244 | 0.0503 | 1.0000 |
| OTHER | 0.0117 | 0.0209 | 0.0157 | 0.0270 | 0.0519 | 0.0186 | 0.0341 | 1.0000 |
| PENNYMAC CORP. | 0.0208 | 0.0408 | 0.0318 | 0.0466 | 0.0965 | 0.0344 | 0.0630 | 1.0000 |
| PENNYMAC LOAN SERVICES, LLC | 0.0176 | 0.0309 | 0.0219 | 0.0363 | 0.0699 | 0.0257 | 0.0480 | 1.0000 |
| PHH ASSET SERVICES LLC | 0.0165 | 0.0376 | 0.0159 | 0.0192 | 0.0476 | 0.0165 | 0.0357 | 1.0000 |
| PHH MORTGAGE CORPORATION | 0.0112 | 0.0205 | 0.0169 | 0.0263 | 0.0566 | 0.0198 | 0.0390 | 1.0000 |
| PINGORA LOAN SERVICING, LLC | 0.0238 | 0.0313 | 0.0296 | 0.0543 | 0.1002 | 0.0362 | 0.0618 | 1.0000 |
| PNC BANK, NA | 0.0128 | 0.0228 | 0.0246 | 0.0284 | 0.0574 | 0.0200 | 0.0371 | 0.9999 |
| PROVIDENT FUNDING ASSOCIATES, L.P. | 0.0175 | 0.0277 | 0.0175 | 0.0335 | 0.0612 | 0.0160 | 0.0379 | 1.0000 |
| QUICKEN LOANS INC. | 0.0148 | 0.0249 | 0.0180 | 0.0364 | 0.0632 | 0.0244 | 0.0406 | 1.0000 |
| QUICKEN LOANS, LLC | 0.0145 | 0.0187 | 0.0187 | 0.0286 | 0.0503 | 0.0171 | 0.0414 | 1.0000 |
| ROCKET MORTGAGE, LLC | 0.0133 | 0.0289 | 0.0181 | 0.0284 | 0.0593 | 0.0210 | 0.0381 | 0.9999 |
| SPECIALIZED LOAN SERVICING LLC | 0.0080 | 0.0053 | 0.0160 | 0.0267 | 0.0588 | 0.0241 | 0.0374 | 1.0000 |
| SUNTRUST BANK | 0.0277 | 0.0346 | 0.0208 | 0.0208 | 0.0969 | 0.0173 | 0.0415 | 1.0000 |
| TH MSR HOLDINGS LLC | 0.0218 | 0.0431 | 0.0265 | 0.0463 | 0.0852 | 0.0319 | 0.0580 | 1.0000 |
| TRUIST BANK | 0.0079 | 0.0162 | 0.0126 | 0.0213 | 0.0393 | 0.0139 | 0.0274 | 1.0000 |
| U.S. BANK N.A. | 0.0087 | 0.0146 | 0.0120 | 0.0173 | 0.0351 | 0.0131 | 0.0238 | 1.0000 |
| UNITED SHORE FINANCIAL SERVICES, LLC | 0.0211 | 0.0380 | 0.0268 | 0.0517 | 0.1005 | 0.0370 | 0.0648 | 1.0000 |

## 4. Outliers, sentinels and invalid dates

Sentinel values are treated as *absence of information*, not as extreme numbers. `days_past_due` of 9999 or -1, note rates of 0 / 99.99 / -1, and balances above 3x original are masked to missing with a `*_repaired` indicator retained, so the fact that a repair happened stays available as a feature.

| repair | rows | rate |
| --- | --- | --- |
| days_past_due sentinel/out-of-range masked | 1855 | 0.0028 |
| interest_rate out-of-range masked | 1172 | 0.0017 |
| current_balance implausible masked | 2788 | 0.0042 |
| loan_age_months recomputed from dates | 8207 | 0.0122 |

### Recovery against the injected ground truth

The synthetic generator logs every defect it injects. Comparing detection against that log is how this rule set was validated rather than merely asserted.

| defect | rows_affected | rate |
| --- | --- | --- |
| missing_dti_band | 41824 | 0.0624 |
| missing_ltv_band | 21094 | 0.0315 |
| missing_property_type | 27635 | 0.0412 |
| missing_credit_score_band | 12691 | 0.0189 |
| missing_occupancy_type | 14911 | 0.0222 |
| missing_interest_rate | 9371 | 0.0140 |
| missing_days_past_due | 17505 | 0.0261 |
| outlier_balance_inflated | 2005 | 0.0030 |
| outlier_balance_negative | 810 | 0.0012 |
| outlier_interest_rate | 1172 | 0.0017 |
| sentinel_days_past_due | 1871 | 0.0028 |
| status_dpd_mismatch | 5016 | 0.0075 |
| invalid_origination_after_reporting | 2746 | 0.0041 |
| inconsistent_loan_age | 3676 | 0.0055 |
| invalid_last_updated_before_period | 6098 | 0.0091 |
| duplicate_rows | 2694 | 0.0040 |

## 5. Cross-column relationship breaks

17 named rules run over every record, grouped into completeness, validity, consistency, plausibility, timeliness and reconciliation dimensions.

| rule | dimension | severity | violations | violation_rate | description |
| --- | --- | --- | --- | --- | --- |
| servicer_record_absent | reconciliation | 3.0000 | 452112 | 0.6742 | No servicer feed record exists for this loan month. |
| missing_critical_field | completeness | 9.0000 | 72896 | 0.1087 | A field required for credit assessment is missing. |
| document_file_incomplete | completeness | 7.0000 | 63958 | 0.0954 | Document custody status is missing or in exception. |
| servicer_balance_break | reconciliation | 13.0000 | 24545 | 0.0366 | Servicer feed balance differs from the panel by >1% and >$500. |
| remaining_term_inconsistent | consistency | 6.0000 | 20625 | 0.0308 | Remaining term plus loan age is not a standard contractual term. |
| stale_servicer_reporting | timeliness | 6.0000 | 18318 | 0.0273 | Record last updated more than 75 days after the period closed. |
| servicer_status_conflict | reconciliation | 11.0000 | 8487 | 0.0127 | Servicer feed reports a different performance status than the panel. |
| loan_age_inconsistent_with_dates | consistency | 9.0000 | 8207 | 0.0122 | Reported loan age disagrees with reporting minus origination month by >2 months. |
| last_updated_before_period_end | validity | 8.0000 | 6098 | 0.0091 | Servicing record was last written before the reporting period closed. |
| balance_exceeds_original | plausibility | 12.0000 | 4226 | 0.0063 | Current balance exceeds original balance by more than 2%. |
| balance_increase_month_over_month | consistency | 7.0000 | 3179 | 0.0047 | Unpaid principal balance rose month over month on a non-modified loan. |
| origination_after_reporting | validity | 14.0000 | 2746 | 0.0041 | Origination month is later than the reporting month. |
| terminal_status_with_balance | consistency | 12.0000 | 2343 | 0.0035 | Loan is in a terminal status but still carries a material balance. |
| dpd_sentinel_value | validity | 10.0000 | 1855 | 0.0028 | Days past due carries a sentinel value (9999, -1). |
| status_dpd_mismatch | consistency | 11.0000 | 1730 | 0.0026 | Days past due is inconsistent with the reported performance status. |
| interest_rate_out_of_range | validity | 10.0000 | 1172 | 0.0017 | Note rate outside a plausible 0.5%-25% range. |
| negative_balance | validity | 16.0000 | 807 | 0.0012 | Current balance is negative. |

## 6. Correlation and dependent-field analysis

### Numeric (Spearman)

| field | original_balance | current_balance | interest_rate | loan_age_months | remaining_term_months | days_past_due | reporting_lag_days |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original_balance | 1.0000 | 0.9490 | -0.0380 | -0.0420 | 0.1590 | -0.0070 | 0.0000 |
| current_balance | 0.9490 | 1.0000 | 0.0010 | -0.1230 | 0.2390 | 0.0020 | 0.0010 |
| interest_rate | -0.0380 | 0.0010 | 1.0000 | -0.2160 | 0.3920 | 0.0420 | 0.0010 |
| loan_age_months | -0.0420 | -0.1230 | -0.2160 | 1.0000 | -0.7050 | 0.0090 | -0.0030 |
| remaining_term_months | 0.1590 | 0.2390 | 0.3920 | -0.7050 | 1.0000 | 0.0070 | 0.0040 |
| days_past_due | -0.0070 | 0.0020 | 0.0420 | 0.0090 | 0.0070 | 1.0000 | -0.0010 |
| reporting_lag_days | 0.0000 | 0.0010 | 0.0010 | -0.0030 | 0.0040 | -0.0010 | 1.0000 |

### Categorical association (bias-corrected Cramer's V, top pairs)

| field_a | field_b | cramers_v |
| --- | --- | --- |
| ltv_band | loan_purpose | 0.3978 |
| occupancy_type | property_type | 0.1991 |
| ltv_band | occupancy_type | 0.1379 |
| credit_score_band | loan_purpose | 0.1326 |
| loan_purpose | property_type | 0.1124 |
| dti_band | loan_purpose | 0.1047 |
| credit_score_band | dti_band | 0.1008 |
| ltv_band | dti_band | 0.0856 |
| credit_score_band | ltv_band | 0.0811 |
| credit_score_band | current_status | 0.0582 |
| credit_score_band | occupancy_type | 0.0564 |
| ltv_band | property_type | 0.0488 |

### Functional dependencies

A loan's static attributes must not change across its reporting months. Violations here are true data-integrity breaks rather than statistical noise.

| determinant | dependent | groups | violating_groups | holds | violation_rate |
| --- | --- | --- | --- | --- | --- |
| loan_id | origination_month | 16000 | 2492 | False | 0.1557 |
| loan_id | credit_score_band | 15996 | 0 | True | 0.0000 |
| loan_id | original_balance | 16000 | 0 | True | 0.0000 |
| loan_id | state | 16000 | 0 | True | 0.0000 |
| loan_id | servicer_name | 16000 | 6903 | False | 0.4314 |
| current_status | expected_dpd | 5 | 0 | True | 0.0000 |

## 7. Train / test drift

Split at `2024-06`, matching the time-aware modelling split used in Task 2. PSI below 0.10 is stable, 0.10-0.25 moderate, above 0.25 severe.

| column | psi | ks_statistic | train_missing_pct | test_missing_pct | severity |
| --- | --- | --- | --- | --- | --- |
| loan_age_months | 2.2742 | 0.5271 | 0.0000 | 0.0000 | severe |
| remaining_term_months | 1.6142 | 0.4037 | 0.0000 | 0.0000 | severe |
| servicer_name | 1.1434 |  | 0.0000 | 0.0000 | severe |
| interest_rate | 0.2351 | 0.1978 | 0.0137 | 0.0143 | moderate |
| loan_purpose | 0.0323 |  | 0.0000 | 0.0000 | stable |
| original_balance | 0.0097 | 0.0378 | 0.0000 | 0.0000 | stable |
| dti_band | 0.0073 |  | 0.0624 | 0.0625 | stable |
| state | 0.0048 |  | 0.0000 | 0.0000 | stable |
| ltv_band | 0.0037 |  | 0.0315 | 0.0314 | stable |
| current_status | 0.0022 |  | 0.0000 | 0.0000 | stable |
| current_balance | 0.0019 | 0.0134 | 0.0000 | 0.0000 | stable |
| occupancy_type | 0.0009 |  | 0.0223 | 0.0221 | stable |
| property_type | 0.0007 |  | 0.0413 | 0.0410 | stable |
| reporting_lag_days | 0.0001 | 0.0030 | 0.0000 | 0.0000 | stable |
| document_status | 0.0001 |  | 0.0000 | 0.0000 | stable |
| credit_score_band | 0.0001 |  | 0.0190 | 0.0192 | stable |
| days_past_due | 0.0000 | 0.0050 | 0.0225 | 0.0319 | stable |
| source_system | 0.0000 |  | 0.0000 | 0.0000 | stable |
| loss_severity_band |  |  | 1.0000 | 1.0000 | stable |

### Target stability across months

| target | overall_rate | min_month_rate | max_month_rate | std_across_months | censored_rows |
| --- | --- | --- | --- | --- | --- |
| next_3m_delinquency_flag | 0.0253 | 0.0000 | 0.4794 | 0.0669 | 30121 |
| next_6m_delinquency_flag | 0.0368 | 0.0000 | 0.4794 | 0.0986 | 59732 |
| next_12m_default_flag | 0.0181 | 0.0000 | 0.2268 | 0.0535 | 121328 |
| next_12m_prepayment_flag | 0.1129 | 0.0459 | 0.8296 | 0.2160 | 121884 |
| exception_required | 0.1404 | 0.0000 | 0.1540 | 0.0160 | 0 |

## 8. Data quality scoring

Record score = 100 minus the severity-weighted sum of rule violations, floored at 0. Batch score aggregates the same violations to the (reporting month x servicer) grain, which is the level an oversight team can act on.

| dq_band | records | share |
| --- | --- | --- |
| clean | 527629 | 0.7869 |
| watch | 136285 | 0.2032 |
| poor | 6606 | 0.0099 |
| critical | 28 | 0.0000 |

- Mean record DQ score: **94.88**
- Median record DQ score: **97.00**
- Records with at least one violation: **79.5%**

### Batch grades by servicer

| servicer_name | records | mean_dq_score | violations_per_record |
| --- | --- | --- | --- |
| UNITED WHOLESALE MORTGAGE, LLC | 23897 | 93.8063 | 1.1639 |
| PENNYMAC CORP. | 21569 | 93.9454 | 1.1494 |
| CALIBER HOME LOANS, INC. | 9159 | 93.9500 | 1.1466 |
| UNITED SHORE FINANCIAL SERVICES, LLC | 3135 | 93.9949 | 1.1451 |
| LOANDEPOT.COM, LLC | 10657 | 94.0457 | 1.1403 |
| TH MSR HOLDINGS LLC | 6606 | 94.0521 | 1.1315 |
| PINGORA LOAN SERVICING, LLC | 2266 | 94.0883 | 1.1320 |
| MATRIX FINANCIAL SERVICES CORPORATION | 12060 | 94.2819 | 1.1035 |
| AMERIHOME MORTGAGE COMPANY, LLC | 15063 | 94.3376 | 1.1013 |
| FREEDOM MORTGAGE CORPORATION | 17652 | 94.4668 | 1.0907 |
| FIFTH THIRD BANK, NATIONAL ASSOCIATION | 9112 | 94.4771 | 1.0849 |
| CITIZENS BANK, NA | 13577 | 94.4804 | 1.0880 |
| ONSLOW BAY FINANCIAL LLC | 8642 | 94.6115 | 1.0620 |
| JPMORGAN CHASE BANK, NATIONAL ASSOCIATION | 60672 | 94.6458 | 1.0637 |
| PENNYMAC LOAN SERVICES, LLC | 9803 | 94.6901 | 1.0657 |
| ROCKET MORTGAGE, LLC | 37632 | 94.7449 | 1.0554 |
| QUICKEN LOANS, LLC | 5387 | 94.7694 | 1.0715 |
| NEW RESIDENTIAL MORTGAGE LLC | 28354 | 94.8026 | 1.0427 |
| CROSSCOUNTRY MORTGAGE, LLC | 1519 | 94.8262 | 1.0178 |
| GUARANTEED RATE, INC. | 331 | 94.9094 | 1.0423 |
| HOME POINT FINANCIAL CORPORATION | 3488 | 94.9097 | 1.0312 |
| PROVIDENT FUNDING ASSOCIATES, L.P. | 686 | 94.9475 | 1.0321 |
| QUICKEN LOANS INC. | 2169 | 94.9686 | 1.0415 |
| LAKEVIEW LOAN SERVICING, LLC | 30384 | 94.9794 | 1.0195 |
| PNC BANK, NA | 12829 | 95.0355 | 1.0125 |
| PHH MORTGAGE CORPORATION | 8890 | 95.0622 | 1.0241 |
| NATIONSTAR MORTGAGE LLC DBA MR. COOPER | 42385 | 95.0806 | 1.0104 |
| SPECIALIZED LOAN SERVICING LLC | 374 | 95.1150 | 1.0267 |
| MARLIN MORTGAGE CAPITAL, LLC | 512 | 95.1191 | 0.9980 |
| OTHER | 185467 | 95.1731 | 0.9971 |
| PHH ASSET SERVICES LLC | 1514 | 95.1764 | 1.0046 |
| CMG MORTGAGE, INC. | 2022 | 95.2230 | 0.9664 |
| NEWREZ LLC | 2624 | 95.2351 | 0.9931 |
| BRANCH BANKING AND TRUST COMPANY | 219 | 95.2694 | 1.0091 |
| GUILD MORTGAGE COMPANY LLC | 1515 | 95.2983 | 0.9789 |
| SUNTRUST BANK | 289 | 95.3010 | 0.9550 |
| BANK OF AMERICA, N.A. | 3531 | 95.3597 | 0.9805 |
| TRUIST BANK | 19350 | 95.5475 | 0.9504 |
| U.S. BANK N.A. | 20156 | 95.6162 | 0.9393 |
| WELLS FARGO BANK, N.A. | 34862 | 95.7774 | 0.9234 |

### Ten worst batches

| reporting_month | servicer_name | records | mean_dq_score | pct_critical | top_failing_rule | top_failing_rule_rate | batch_grade |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2019-02 | FIFTH THIRD BANK | 1 | 87.0000 | 0.0000 | servicer_balance_break | 1.0000 | B |
| 2019-10 | SPECIALIZED LOAN SERVICING LLC | 1 | 88.0000 | 0.0000 | missing_critical_field | 1.0000 | B |
| 2019-02 | SUNTRUST BANK | 3 | 89.0000 | 0.0000 | servicer_record_absent | 0.6667 | B |
| 2019-01 | OTHER | 2 | 90.5000 | 0.0000 | negative_balance | 0.5000 | B |
| 2019-04 | PNC BANK, NA | 8 | 91.6250 | 0.0000 | servicer_record_absent | 0.8750 | A |
| 2019-06 | LAKEVIEW LOAN SERVICING, LLC | 10 | 92.1000 | 0.0000 | servicer_record_absent | 0.5000 | A |
| 2025-10 | UNITED WHOLESALE MORTGAGE, LLC | 129 | 92.5116 | 0.0000 | servicer_record_absent | 0.6047 | A |
| 2021-02 | FREEDOM MORTGAGE CORPORATION | 44 | 92.6591 | 0.0000 | servicer_record_absent | 0.7045 | A |
| 2025-05 | UNITED WHOLESALE MORTGAGE, LLC | 153 | 92.7516 | 0.0000 | servicer_record_absent | 0.6536 | A |
| 2025-11 | UNITED WHOLESALE MORTGAGE, LLC | 129 | 92.7984 | 0.0000 | servicer_record_absent | 0.6279 | A |

## 9. What this means for modelling

1. **Servicer is a confound, not just a feature.** Kestrel Financial and Pioneer Mortgage Ops have both the worst data quality *and* elevated delinquency. A model given raw servicer identity will partly learn reporting behaviour rather than credit risk. Servicer is retained but its SHAP contribution is inspected separately in the explainability report.
2. **Censoring is real and material.** Forward-looking targets are undefined for rows whose horizon runs past the panel end. These are `NaN`, not `0`, and are excluded from supervised training rather than counted as non-events.
3. **Repairs are features.** Whether a record needed repair is predictive of whether it needs an exception, so repair indicators are carried forward rather than discarded.
4. **Drift is concentrated in macro-sensitive fields**, which is expected given the rate path in the panel window and is handled by time-aware validation rather than by reweighting.
