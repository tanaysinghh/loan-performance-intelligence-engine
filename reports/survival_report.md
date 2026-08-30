# Time-to-Event and Transition Report

**Task 3.** Two model families, both non-LLM: Kaplan-Meier / Cox proportional hazards from `lifelines`, and an empirical multi-state Markov chain estimated from the training window.

## 1. Why two models rather than one

A survival model answers *when* and *how much a covariate moves the timing*. It cannot express "the loan is 60 days down today, where is it in twelve months", because it collapses intermediate states into a single absorbing event. A Markov chain answers exactly that but has no covariates beyond the current state. They are reported together because a servicing team needs both questions answered.

## 2. Censoring treatment

Three distinct reasons an outcome is unobserved, each handled differently rather than lumped together:

| mechanism | loans | share | treatment |
| --- | --- | --- | --- |
| Administrative right-censoring | 10008 | 0.6255 | Still performing at panel end. Duration = final observed age, event = 0. Contributes exposure to the risk set up to that age and nothing after. |
| Competing risk (prepayment before default) | 5437 | 0.3398 | Censored in the default model, giving the cause-specific hazard. Cumulative incidence is computed separately by Aalen-Johansen; see section 4. |
| Left truncation (loan originated before the panel opens) | 795 | 0.0497 | Entry age passed as truncation time, so ages before panel entry are excluded from the risk set instead of counted as event-free exposure. |
| Observed default | 505 | 0.0316 | Event = 1 at the loan age of the transition into Default. |

Loan-level survival frame: **16,000** loans, **505** defaults, **5437** prepayments, **10058** censored. Median observed duration: **40** months.

Ignoring left truncation would be the expensive mistake here: **5%** of loans enter the panel already seasoned. Crediting them with event-free exposure at ages they were never observed at would flatten the early hazard and understate the seasoning ramp.

## 3. Event curves

Cumulative event probability by loan age, Kaplan-Meier, left-truncation aware.

### Default

| group | 12 | 24 | 36 | 60 | 84 |
| --- | --- | --- | --- | --- | --- |
| all | 0.0113 | 0.0249 | 0.0334 | 0.0395 | 0.0523 |

### Prepayment

| group | 12 | 24 | 36 | 60 | 84 |
| --- | --- | --- | --- | --- | --- |
| all | 0.1018 | 0.2294 | 0.3002 | 0.3701 | 0.4447 |

### Default by credit band

| group | 12 | 24 | 36 | 60 | 84 |
| --- | --- | --- | --- | --- | --- |
| 620-659 | 0.0274 | 0.0758 | 0.1026 | 0.1286 | 0.1578 |
| 660-699 | 0.0209 | 0.0505 | 0.0710 | 0.0804 | 0.1366 |
| 700-739 | 0.0128 | 0.0350 | 0.0486 | 0.0593 | 0.0718 |
| 740-779 | 0.0120 | 0.0207 | 0.0262 | 0.0311 | 0.0346 |
| 780+ | 0.0043 | 0.0071 | 0.0089 | 0.0105 | 0.0122 |

### Default by LTV band

| group | 12 | 24 | 36 | 60 | 84 |
| --- | --- | --- | --- | --- | --- |
| 60-70 | 0.0110 | 0.0232 | 0.0330 | 0.0403 | 0.0466 |
| 70-80 | 0.0113 | 0.0253 | 0.0320 | 0.0399 | 0.0598 |
| 80-90 | 0.0120 | 0.0236 | 0.0344 | 0.0428 | 0.0428 |
| 90-95 | 0.0155 | 0.0376 | 0.0493 | 0.0590 | 0.0749 |
| <=60 | 0.0080 | 0.0157 | 0.0213 | 0.0233 | 0.0344 |
| >95 | 0.0143 | 0.0395 | 0.0607 | 0.0607 | 0.0702 |

### Default by servicer

| group | 12 | 24 | 36 | 60 | 84 |
| --- | --- | --- | --- | --- | --- |
| AMERIHOME MORTGAGE COMPANY, LLC | 0.0041 | 0.0201 | 0.0307 | 0.0477 | 0.0477 |
| BANK OF AMERICA, N.A. | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1111 |
| BRANCH BANKING AND TRUST COMPANY | 0.0000 | 0.0597 | 0.0597 | 0.0597 | 0.0597 |
| CALIBER HOME LOANS, INC. | 0.0300 | 0.0410 | 0.0451 | 0.0547 | 0.2910 |
| CITIZENS BANK, NA | 0.0034 | 0.0071 | 0.0114 | 0.0172 | 0.0341 |
| FIFTH THIRD BANK, NATIONAL ASSOCIATION | 0.0000 | 0.0000 | 0.0097 | 0.0097 | 0.0097 |
| FREEDOM MORTGAGE CORPORATION | 0.0258 | 0.0584 | 0.1015 | 0.1245 | 0.1245 |
| HOME POINT FINANCIAL CORPORATION | 0.0065 | 0.0139 | 0.0214 | 0.0312 | 0.0312 |
| JPMORGAN CHASE BANK, NATIONAL ASSOCIATION | 0.0185 | 0.0426 | 0.0484 | 0.0565 | 0.0650 |
| LAKEVIEW LOAN SERVICING, LLC | 0.0126 | 0.0274 | 0.0333 | 0.0432 | 0.0432 |
| LOANDEPOT.COM, LLC | 0.0073 | 0.0192 | 0.0323 | 0.0392 | 0.0392 |
| MATRIX FINANCIAL SERVICES CORPORATION | 0.0226 | 0.0327 | 0.0386 | 0.0451 | 0.0451 |

## 4. Competing risks: why 1 - KM is the wrong number

The naive complement of a cause-specific Kaplan-Meier curve treats prepaid loans as if they remained at risk of default. They did not — prepayment removes the loan permanently. Aalen-Johansen cumulative incidence accounts for the competing hazard. The gap below is the amount by which the naive figure overstates default risk, and it grows with age because prepayment accumulates.

| loan_age_months | at_risk | cif_default | cif_prepay | naive_1_minus_km_default | km_overstatement | event_free_survival |
| --- | --- | --- | --- | --- | --- | --- |
| 12.0000 | 14384 | 0.0108 | 0.1013 | 0.0113 | 0.0005 | 0.8880 |
| 24.0000 | 12124 | 0.0222 | 0.2265 | 0.0249 | 0.0026 | 0.7513 |
| 36.0000 | 8963 | 0.0285 | 0.2954 | 0.0334 | 0.0049 | 0.6762 |
| 48.0000 | 6088 | 0.0312 | 0.3304 | 0.0374 | 0.0061 | 0.6384 |
| 60.0000 | 3297 | 0.0326 | 0.3626 | 0.0395 | 0.0069 | 0.6048 |

Maximum overstatement across the observed age range: **0.0121** in absolute probability. On a book of 10,000 loans that is the difference between provisioning for 121 extra defaults that will not happen.

## 5. Cox proportional hazards

Penalised Cox (ridge, 0.08) with robust standard errors, fitted on the training-window loans and scored out-of-sample on the remainder. Hazard ratio above 1 means the covariate accelerates the event.

### Default hazard

| feature | coef | hazard_ratio | se(coef) | p |
| --- | --- | --- | --- | --- |
| rate_incentive_at_entry | 0.1057 | 1.1115 | 0.0381 | 0.0056 |
| dti_ord | 0.1036 | 1.1092 | 0.0179 | 0.0000 |
| is_cash_out | 0.0990 | 1.1041 | 0.0617 | 0.1086 |
| log_original_balance | 0.0825 | 1.0860 | 0.0460 | 0.0729 |
| interest_rate_clean | 0.0646 | 1.0668 | 0.0246 | 0.0085 |
| ltv_ord | 0.0625 | 1.0645 | 0.0184 | 0.0007 |
| is_investment | 0.0420 | 1.0429 | 0.1130 | 0.7100 |
| is_high_ops_servicer | 0.0391 | 1.0399 | 0.0620 | 0.5283 |
| credit_ord | -0.0911 | 0.9129 | 0.0215 | 0.0000 |

### Prepayment hazard

| feature | coef | hazard_ratio | se(coef) | p |
| --- | --- | --- | --- | --- |
| log_original_balance | 0.4139 | 1.5126 | 0.0313 | 0.0000 |
| interest_rate_clean | 0.1309 | 1.1398 | 0.0241 | 0.0000 |
| is_high_ops_servicer | 0.0582 | 1.0599 | 0.0355 | 0.1010 |
| credit_ord | 0.0308 | 1.0313 | 0.0140 | 0.0281 |
| is_cash_out | 0.0295 | 1.0299 | 0.0390 | 0.4489 |
| dti_ord | -0.0120 | 0.9881 | 0.0127 | 0.3441 |
| ltv_ord | -0.0704 | 0.9320 | 0.0119 | 0.0000 |
| is_investment | -0.0962 | 0.9083 | 0.0592 | 0.1044 |
| rate_incentive_at_entry | -0.1556 | 0.8559 | 0.0369 | 0.0000 |

### Discrimination against the covariate-free baseline

Kaplan-Meier assigns every loan the same survival curve, so its concordance is 0.50 by construction. That is the baseline the Cox models are beating.

| model | n_train | events_train | n_test | events_test | concordance_train | concordance_test |
| --- | --- | --- | --- | --- | --- | --- |
| Cox — default | 3845 | 123 | 12149 | 382 | 0.6940 | 0.7169 |
| Cox — prepayment | 3845 | 3679 | 12149 | 1755 | 0.6036 | 0.6811 |

## 6. Multi-state Markov transition model

Monthly one-step transition matrix estimated on the training window with Laplace smoothing. `Default` and `Prepaid` are absorbing by construction.

| from_state | Current | DQ30 | DQ60 | DQ90plus | Default | Prepaid |
| --- | --- | --- | --- | --- | --- | --- |
| Current | 0.9824 | 0.0048 | 0.0000 | 0.0000 | 0.0000 | 0.0128 |
| DQ30 | 0.5040 | 0.2456 | 0.2487 | 0.0014 | 0.0002 | 0.0002 |
| DQ60 | 0.1772 | 0.0464 | 0.1118 | 0.6636 | 0.0006 | 0.0006 |
| DQ90plus | 0.1206 | 0.0051 | 0.0170 | 0.6435 | 0.2135 | 0.0003 |
| Default | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| Prepaid | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

### 12-month projection by starting state

Raising the matrix to the 12th power gives the state distribution a year out. This is the number a servicer wants when triaging a delinquent loan.

| start_state | horizon_month | p_Current | p_DQ30 | p_DQ60 | p_DQ90plus | p_Default | p_Prepaid |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Current | 12 | 0.8425 | 0.0055 | 0.0016 | 0.0032 | 0.0049 | 0.1422 |
| DQ30 | 12 | 0.7257 | 0.0049 | 0.0016 | 0.0082 | 0.1574 | 0.1023 |
| DQ60 | 12 | 0.4555 | 0.0031 | 0.0012 | 0.0110 | 0.4704 | 0.0587 |
| DQ90plus | 12 | 0.3267 | 0.0022 | 0.0009 | 0.0086 | 0.6201 | 0.0414 |
| Default | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| Prepaid | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

### Cumulative default probability path

| horizon_month | Current | DQ30 | DQ60 | DQ90plus |
| --- | --- | --- | --- | --- |
| 1 | 0.0000 | 0.0002 | 0.0006 | 0.2135 |
| 2 | 0.0000 | 0.0006 | 0.1423 | 0.3509 |
| 3 | 0.0000 | 0.0362 | 0.2494 | 0.4418 |
| 4 | 0.0002 | 0.0717 | 0.3233 | 0.5022 |
| 5 | 0.0006 | 0.0990 | 0.3733 | 0.5426 |
| 6 | 0.0010 | 0.1183 | 0.4070 | 0.5696 |
| 7 | 0.0016 | 0.1318 | 0.4297 | 0.5877 |
| 8 | 0.0022 | 0.1410 | 0.4450 | 0.5999 |
| 9 | 0.0029 | 0.1474 | 0.4553 | 0.6081 |
| 10 | 0.0036 | 0.1519 | 0.4623 | 0.6137 |
| 11 | 0.0042 | 0.1551 | 0.4671 | 0.6175 |
| 12 | 0.0049 | 0.1574 | 0.4704 | 0.6201 |

## 7. Validation against realised outcomes

The projection is compared against what actually happened to test-window rows over the following twelve months. This is the check that separates a plausible-looking matrix from a correct one.

| start_state | n_test_rows | markov_predicted_default_12m | observed_default_12m | markov_predicted_prepay_12m | observed_prepay_12m | default_abs_error | prepay_abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Current | 67736 | 0.0049 | 0.0060 | 0.1422 | 0.0757 | 0.0011 | 0.0665 |
| DQ30 | 654 | 0.1574 | 0.2370 | 0.1023 | 0.0734 | 0.0796 | 0.0289 |
| DQ60 | 188 | 0.4705 | 0.6968 | 0.0587 | 0.0904 | 0.2264 | 0.0317 |
| DQ90plus | 202 | 0.6201 | 0.8762 | 0.0414 | 0.0941 | 0.2561 | 0.0526 |
| Default | 204 | 1.0000 | 0.9314 | 0.0000 | 0.0833 | 0.0686 | 0.0833 |

Mean absolute error on 12-month default probability: **0.1264**; on prepayment: **0.0526**.

## 8. Limitations

- **The Markov assumption is wrong, usefully.** A first-order chain assumes the next state depends only on the current one. It does not: a loan that has been in DQ30 for five months differs from one that entered last month. The LightGBM next-state model in Task 2 uses that history and beats this matrix on macro-AUC (0.886 vs 0.841). The chain is kept because it is transparent, cheap to re-estimate under a stress scenario, and gives full multi-period state distributions the classifier does not.
- **Proportional hazards is assumed, not tested here.** Schoenfeld residual tests are not run; with a macro path this pronounced, time-varying effects are likely for the rate-sensitive prepayment covariates in particular.
- Cox covariates are fixed at loan entry. Time-varying covariates would fit better but would need care to avoid conditioning on post-entry information.
- Loss severity is modelled only as an observed band on defaulted loans; no LGD model is fitted, so nothing here converts default probability into expected loss.
