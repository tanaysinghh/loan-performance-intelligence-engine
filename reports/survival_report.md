# Time-to-Event and Transition Report

**Task 3.** Two model families, both non-LLM: Kaplan-Meier / Cox proportional hazards from `lifelines`, and an empirical multi-state Markov chain estimated from the training window.

## 1. Why two models rather than one

A survival model answers *when* and *how much a covariate moves the timing*. It cannot express "the loan is 60 days down today, where is it in twelve months", because it collapses intermediate states into a single absorbing event. A Markov chain answers exactly that but has no covariates beyond the current state. They are reported together because a servicing team needs both questions answered.

## 2. Censoring treatment

Three distinct reasons an outcome is unobserved, each handled differently rather than lumped together:

| mechanism | loans | share | treatment |
| --- | --- | --- | --- |
| Administrative right-censoring | 534 | 0.3560 | Still performing at panel end. Duration = final observed age, event = 0. Contributes exposure to the risk set up to that age and nothing after. |
| Competing risk (prepayment before default) | 739 | 0.4927 | Censored in the default model, giving the cause-specific hazard. Cumulative incidence is computed separately by Aalen-Johansen; see section 4. |
| Left truncation (loan originated before the panel opens) | 540 | 0.3600 | Entry age passed as truncation time, so ages before panel entry are excluded from the risk set instead of counted as event-free exposure. |
| Observed default | 227 | 0.1513 | Event = 1 at the loan age of the transition into Default. |

Loan-level survival frame: **1,500** loans, **227** defaults, **739** prepayments, **534** censored. Median observed duration: **22** months.

Ignoring left truncation would be the expensive mistake here: **36%** of loans enter the panel already seasoned. Crediting them with event-free exposure at ages they were never observed at would flatten the early hazard and understate the seasoning ramp.

## 3. Event curves

Cumulative event probability by loan age, Kaplan-Meier, left-truncation aware.

### Default

| group | 12 | 24 | 36 | 60 | 84 |
| --- | --- | --- | --- | --- | --- |
| all | 0.0227 | 0.0766 | 0.1387 | 0.2772 | 0.3706 |

### Prepayment

| group | 12 | 24 | 36 | 60 | 84 |
| --- | --- | --- | --- | --- | --- |
| all | 0.2032 | 0.3691 | 0.5002 | 0.6448 | 0.7099 |

### Default by credit band

| group | 12 | 24 | 36 | 60 | 84 |
| --- | --- | --- | --- | --- | --- |
| 580-619 | 0.0585 | 0.2650 | 0.4506 | 0.6925 | 0.7804 |
| 620-659 | 0.0458 | 0.1314 | 0.2193 | 0.4575 | 0.5950 |
| 660-699 | 0.0295 | 0.0981 | 0.0981 | 0.2751 | 0.4066 |
| 700-739 | 0.0055 | 0.0184 | 0.0562 | 0.1635 | 0.2254 |
| 740-779 | 0.0057 | 0.0125 | 0.0278 | 0.0624 | 0.1233 |
| 780+ | 0.0068 | 0.0068 | 0.0168 | 0.0285 | 0.0741 |
| <580 | 0.0835 | 0.2928 | 0.6129 | 0.8434 | 0.9720 |

### Default by LTV band

| group | 12 | 24 | 36 | 60 | 84 |
| --- | --- | --- | --- | --- | --- |
| 60-70 | 0.0092 | 0.0248 | 0.0439 | 0.1564 | 0.2231 |
| 70-80 | 0.0172 | 0.0442 | 0.0819 | 0.2564 | 0.3869 |
| 80-90 | 0.0602 | 0.1390 | 0.2126 | 0.3722 | 0.4621 |
| 90-95 | 0.0282 | 0.1255 | 0.2746 | 0.4560 | 0.5134 |
| <=60 | 0.0000 | 0.0000 | 0.0189 | 0.0654 | 0.1909 |
| >95 | 0.0315 | 0.2518 | 0.4380 | 0.6132 | 0.7242 |

### Default by servicer

| group | 12 | 24 | 36 | 60 | 84 |
| --- | --- | --- | --- | --- | --- |
| Arcadia Capital Servicing | 0.0384 | 0.0739 | 0.1261 | 0.2458 | 0.3407 |
| Belmont Loan Services | 0.0226 | 0.0769 | 0.1467 | 0.3147 | 0.4024 |
| Kestrel Financial | 0.0103 | 0.0922 | 0.1684 | 0.3538 | 0.4025 |
| Northgate Servicing | 0.0078 | 0.0681 | 0.1339 | 0.2452 | 0.3293 |
| Pioneer Mortgage Ops | 0.0424 | 0.0872 | 0.1348 | 0.2746 | 0.4105 |

## 4. Competing risks: why 1 - KM is the wrong number

The naive complement of a cause-specific Kaplan-Meier curve treats prepaid loans as if they remained at risk of default. They did not — prepayment removes the loan permanently. Aalen-Johansen cumulative incidence accounts for the competing hazard. The gap below is the amount by which the naive figure overstates default risk, and it grows with age because prepayment accumulates.

| loan_age_months | at_risk | cif_default | cif_prepay | naive_1_minus_km_default | km_overstatement | event_free_survival |
| --- | --- | --- | --- | --- | --- | --- |
| 12.0000 | 823 | 0.0195 | 0.2021 | 0.0227 | 0.0032 | 0.7784 |
| 24.0000 | 675 | 0.0577 | 0.3606 | 0.0766 | 0.0189 | 0.5817 |
| 36.0000 | 570 | 0.0919 | 0.4788 | 0.1387 | 0.0467 | 0.4293 |
| 48.0000 | 485 | 0.1182 | 0.5500 | 0.1967 | 0.0785 | 0.3318 |
| 60.0000 | 335 | 0.1489 | 0.5957 | 0.2772 | 0.1283 | 0.2554 |
| 84.0000 | 158 | 0.1784 | 0.6401 | 0.3706 | 0.1921 | 0.1815 |
| 108.0000 | 71 | 0.1919 | 0.6521 | 0.4183 | 0.2265 | 0.1560 |

Maximum overstatement across the observed age range: **0.2497** in absolute probability. On a book of 10,000 loans that is the difference between provisioning for 2497 extra defaults that will not happen.

## 5. Cox proportional hazards

Penalised Cox (ridge, 0.08) with robust standard errors, fitted on the training-window loans and scored out-of-sample on the remainder. Hazard ratio above 1 means the covariate accelerates the event.

### Default hazard

| feature | coef | hazard_ratio | se(coef) | p |
| --- | --- | --- | --- | --- |
| rate_incentive_at_entry | 0.5321 | 1.7025 | 0.1675 | 0.0015 |
| is_investment | 0.2658 | 1.3045 | 0.2142 | 0.2146 |
| dti_ord | 0.1976 | 1.2185 | 0.0495 | 0.0001 |
| ltv_ord | 0.1658 | 1.1804 | 0.0427 | 0.0001 |
| is_high_ops_servicer | 0.1582 | 1.1714 | 0.1327 | 0.2335 |
| log_original_balance | 0.0290 | 1.0295 | 0.1281 | 0.8206 |
| interest_rate_clean | 0.0257 | 1.0260 | 0.1463 | 0.8605 |
| credit_ord | -0.1824 | 0.8333 | 0.0335 | 0.0000 |
| is_cash_out | -0.2428 | 0.7844 | 0.1683 | 0.1491 |

### Prepayment hazard

| feature | coef | hazard_ratio | se(coef) | p |
| --- | --- | --- | --- | --- |
| credit_ord | 0.2099 | 1.2336 | 0.0284 | 0.0000 |
| interest_rate_clean | 0.1494 | 1.1611 | 0.0802 | 0.0624 |
| rate_incentive_at_entry | 0.1446 | 1.1555 | 0.1569 | 0.3569 |
| is_cash_out | 0.1321 | 1.1412 | 0.1045 | 0.2065 |
| is_investment | 0.1257 | 1.1339 | 0.1273 | 0.3233 |
| log_original_balance | 0.0397 | 1.0405 | 0.0819 | 0.6282 |
| is_high_ops_servicer | 0.0057 | 1.0057 | 0.0903 | 0.9495 |
| dti_ord | -0.0008 | 0.9992 | 0.0320 | 0.9799 |
| ltv_ord | -0.0637 | 0.9382 | 0.0351 | 0.0697 |

### Discrimination against the covariate-free baseline

Kaplan-Meier assigns every loan the same survival curve, so its concordance is 0.50 by construction. That is the baseline the Cox models are beating.

| model | n_train | events_train | n_test | events_test | concordance_train | concordance_test |
| --- | --- | --- | --- | --- | --- | --- |
| Cox — default | 634 | 135 | 857 | 92 | 0.8043 | 0.8236 |
| Cox — prepayment | 634 | 499 | 857 | 231 | 0.5800 | 0.6967 |

## 6. Multi-state Markov transition model

Monthly one-step transition matrix estimated on the training window with Laplace smoothing. `Default` and `Prepaid` are absorbing by construction.

| from_state | Current | DQ30 | DQ60 | DQ90plus | Default | Prepaid |
| --- | --- | --- | --- | --- | --- | --- |
| Current | 0.9751 | 0.0063 | 0.0000 | 0.0000 | 0.0000 | 0.0186 |
| DQ30 | 0.0904 | 0.4910 | 0.4163 | 0.0008 | 0.0008 | 0.0008 |
| DQ60 | 0.0010 | 0.0569 | 0.4202 | 0.5200 | 0.0010 | 0.0010 |
| DQ90plus | 0.0005 | 0.0005 | 0.0189 | 0.7559 | 0.2175 | 0.0066 |
| Default | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| Prepaid | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

### 12-month projection by starting state

Raising the matrix to the 12th power gives the state distribution a year out. This is the number a servicer wants when triaging a delinquent loan.

| start_state | horizon_month | p_Current | p_DQ30 | p_DQ60 | p_DQ90plus | p_Default | p_Prepaid |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Current | 12 | 0.7484 | 0.0106 | 0.0084 | 0.0165 | 0.0194 | 0.1965 |
| DQ30 | 12 | 0.1560 | 0.0052 | 0.0123 | 0.1248 | 0.6471 | 0.0547 |
| DQ60 | 12 | 0.0201 | 0.0019 | 0.0069 | 0.0911 | 0.8487 | 0.0313 |
| DQ90plus | 12 | 0.0034 | 0.0008 | 0.0035 | 0.0521 | 0.9118 | 0.0285 |
| Default | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| Prepaid | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

### Cumulative default probability path

| horizon_month | Current | DQ30 | DQ60 | DQ90plus |
| --- | --- | --- | --- | --- |
| 1 | 0.0000 | 0.0008 | 0.0010 | 0.2175 |
| 2 | 0.0000 | 0.0017 | 0.1145 | 0.3819 |
| 3 | 0.0001 | 0.0495 | 0.2478 | 0.5084 |
| 4 | 0.0004 | 0.1286 | 0.3723 | 0.6065 |
| 5 | 0.0012 | 0.2194 | 0.4801 | 0.6831 |
| 6 | 0.0026 | 0.3089 | 0.5704 | 0.7430 |
| 7 | 0.0045 | 0.3907 | 0.6445 | 0.7901 |
| 8 | 0.0068 | 0.4619 | 0.7049 | 0.8271 |
| 9 | 0.0096 | 0.5222 | 0.7535 | 0.8563 |
| 10 | 0.0127 | 0.5724 | 0.7925 | 0.8793 |
| 11 | 0.0160 | 0.6135 | 0.8238 | 0.8975 |
| 12 | 0.0194 | 0.6471 | 0.8487 | 0.9118 |

## 7. Validation against realised outcomes

The projection is compared against what actually happened to test-window rows over the following twelve months. This is the check that separates a plausible-looking matrix from a correct one.

| start_state | n_test_rows | markov_predicted_default_12m | observed_default_12m | markov_predicted_prepay_12m | observed_prepay_12m | default_abs_error | prepay_abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Current | 3814 | 0.0194 | 0.0294 | 0.1965 | 0.1778 | 0.0099 | 0.0188 |
| DQ30 | 64 | 0.6471 | 0.6719 | 0.0547 | 0.0000 | 0.0248 | 0.0547 |
| DQ60 | 65 | 0.8487 | 0.9077 | 0.0313 | 0.0154 | 0.0590 | 0.0159 |
| DQ90plus | 97 | 0.9118 | 0.8041 | 0.0285 | 0.1237 | 0.1077 | 0.0952 |

Mean absolute error on 12-month default probability: **0.0504**; on prepayment: **0.0461**.

## 8. Limitations

- **The Markov assumption is wrong, usefully.** A first-order chain assumes the next state depends only on the current one. It does not: a loan that has been in DQ30 for five months differs from one that entered last month. The LightGBM next-state model in Task 2 uses that history and beats this matrix on macro-AUC (0.886 vs 0.841). The chain is kept because it is transparent, cheap to re-estimate under a stress scenario, and gives full multi-period state distributions the classifier does not.
- **Proportional hazards is assumed, not tested here.** Schoenfeld residual tests are not run; with a macro path this pronounced, time-varying effects are likely for the rate-sensitive prepayment covariates in particular.
- Cox covariates are fixed at loan entry. Time-varying covariates would fit better but would need care to avoid conditioning on post-entry information.
- Loss severity is modelled only as an observed band on defaulted loans; no LGD model is fitted, so nothing here converts default probability into expected loss.
