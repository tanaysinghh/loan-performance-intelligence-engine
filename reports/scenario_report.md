# Scenario and Stress Simulation Report

**Task 5.** All projections come from the trained LightGBM models and an empirical Markov chain. No language model produces any number in this report.

## 1. Scenario assumptions

Defined in `data/raw/macro_scenarios.csv`, twelve monthly steps each. Stated here explicitly because a stress result is only as meaningful as the assumption behind it.

| scenario_name | market_rate_month_12 | market_rate_shift | unemployment_month_12 | unemployment_shift | hpi_growth_month_12 | assumption |
| --- | --- | --- | --- | --- | --- | --- |
| adverse_credit | 6.1775 | 0.0000 | 7.3000 | 3.0000 | -0.1000 | Unemployment rises 3.00pp over 12 months and HPI growth falls to -10% YoY; rates held flat. CONSTRUCTED ASSUMPTION at supervisory (DFAST/CCAR-style) severity, NOT taken from the panel window. Disclosed basis: the largest 12-month unemployment rise observed in UNRATE over the window is 11.1pp, but that is the COVID one-off (peak 14.8% in 2020-04); excluding 2020-01..2021-06 the largest rise is only 1.4pp, and the weakest HPI growth observed is -0.29%. The window contains no housing downturn, so an empirically-anchored adverse case would not stress the book; supervisory magnitudes are used instead. |
| base | 6.1775 | 0.0000 | 4.3000 | 0.0000 | 0.0078 | Observed conditions at 2026-03 held flat for 12 months (rate 6.18%, unemployment 4.30%, HPI YoY 0.78%). No shock applied. CONSTRUCTED ASSUMPTION; history is observed. |
| high_prepayment | 4.9875 | -1.1900 | 4.3000 | 0.0000 | 0.0078 | Market mortgage rate falls 1.19pp over 12 months, opening refinance incentive on seasoned high-coupon loans. CONSTRUCTED ASSUMPTION, calibrated to the largest 12-month rate decline actually observed in MORTGAGE30US over the panel window. |

Starting point (latest observed month, 2026-03): market rate 6.18%, unemployment 4.30%, HPI growth 0.008.

## 2. Two engines, deliberately

| engine | method | strength | weakness |
| --- | --- | --- | --- |
| A — model repricing | Overwrite every macro-derived feature on the latest snapshot of each live loan, re-score the Task 2 models. | Uses the full covariate set, so segment detail is real and actionable. Its refinance-incentive channel is cross-sectionally identified. | Its credit channel is NOT identified from a single macro path (section 3). It cannot size a credit stress. |
| B — macro-conditioned Markov | Regress monthly transition log-odds on the macro path, shift inputs to scenario values, rebuild the matrix and roll forward twelve months. | Extrapolates smoothly through a logistic link; gives a full multi-period portfolio path. | Conditions only on current state — no borrower covariates at all. |

They are not redundant. Engine A answers *which loans*; Engine B answers *how bad*. Section 3 shows why neither can be asked to do the other's job here.

## 3. Why Engine A cannot size a credit stress

Engine A's **credit** channel is not identified from this data, and the evidence is in its own output rather than in an argument about it.

Macro levels are constant across every loan within a reporting month. With one realised macro path and 90 monthly observations there is no cross-sectional variation in unemployment or HPI growth at all, so a loan-level model cannot separate the effect of unemployment from the effect of calendar time — they are collinear by construction. What the trees learn is a time proxy, and in this panel's history the low-rate period coincided with the pandemic unemployment spike.

The consequence shows up as two specific wrong answers, both visible in the tables below. The adverse-credit shock moves Engine A's projected 12-month default rate by essentially nothing — a delta indistinguishable from zero, and of a sign that carries no information — which is not a credible stress result for a scenario that raises unemployment by three percentage points and turns house prices negative. And the high-prepayment scenario nudges projected default *upward*, which has the sign backwards. Both are reported in section 8 as computed rather than being corrected.

`rate_incentive` is the exception. It is a loan's own note rate minus the prevailing market rate, so it does vary across loans within a month and its effect is identified cross-sectionally. That is why Engine A's prepayment response is trustworthy and its credit response is not, and why Engine B exists.

An earlier iteration tried to fix this by perturbing only the identified refinance-incentive features and leaving macro levels at their observed values. That was worse, not better: it hands the model a feature combination that never occurs in training (a market rate of 5.5% alongside an incentive computed against 5.74%) and the base-case prepayment projection jumped from 0.156 to 0.396 on a scenario that is supposed to be a no-op. Internally consistent shifts plus an honest statement of what the resulting credit number is worth beats a surgical restriction that breaks the input distribution.

## 4. Engine A — portfolio-level projections

| scenario_name | loans | projected_next_6m_delinquency_flag | projected_next_12m_default_flag | projected_next_12m_prepayment_flag |
| --- | --- | --- | --- | --- |
| adverse_credit | 16000 | 0.0328 | 0.0085 | 0.4485 |
| base | 16000 | 0.0328 | 0.0085 | 0.4463 |
| high_prepayment | 16000 | 0.0327 | 0.0086 | 0.4972 |

| scenario_name | delta_next_6m_delinquency_flag | relative_next_6m_delinquency_flag | delta_next_12m_default_flag | relative_next_12m_default_flag | delta_next_12m_prepayment_flag | relative_next_12m_prepayment_flag |
| --- | --- | --- | --- | --- | --- | --- |
| adverse_credit | -0.0000 | -0.0006 | -0.0000 | -0.0013 | 0.0022 | 0.0049 |
| base | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| high_prepayment | -0.0001 | -0.0043 | 0.0000 | 0.0025 | 0.0509 | 0.1140 |

## 5. Engine A — segment-level impacts

### 12-month default probability by credit band

| credit_score_band | loans | adverse_credit | base | high_prepayment | delta_adverse_credit | delta_high_prepayment |
| --- | --- | --- | --- | --- | --- | --- |
| 780+ | 5086 | 0.0020 | 0.0020 | 0.0019 | 0.0000 | -0.0001 |
| 740-779 | 4975 | 0.0055 | 0.0055 | 0.0055 | 0.0000 | 0.0001 |
| 700-739 | 3274 | 0.0123 | 0.0123 | 0.0127 | -0.0000 | 0.0004 |
| 660-699 | 1754 | 0.0191 | 0.0191 | 0.0191 | -0.0000 | -0.0001 |
| 580-619 | 10 | 0.0022 | 0.0023 | 0.0023 | -0.0001 | 0.0000 |
| 620-659 | 604 | 0.0390 | 0.0396 | 0.0385 | -0.0005 | -0.0011 |

### 12-month default probability by ltv band

| ltv_band | loans | adverse_credit | base | high_prepayment | delta_adverse_credit | delta_high_prepayment |
| --- | --- | --- | --- | --- | --- | --- |
| >95 | 557 | 0.0117 | 0.0115 | 0.0115 | 0.0002 | 0.0000 |
| 90-95 | 2205 | 0.0144 | 0.0143 | 0.0148 | 0.0001 | 0.0005 |
| 80-90 | 1789 | 0.0082 | 0.0081 | 0.0081 | 0.0001 | -0.0000 |
| 70-80 | 5138 | 0.0075 | 0.0075 | 0.0075 | -0.0000 | -0.0000 |
| <=60 | 3695 | 0.0058 | 0.0059 | 0.0056 | -0.0001 | -0.0002 |
| 60-70 | 2118 | 0.0081 | 0.0083 | 0.0083 | -0.0002 | 0.0001 |

### 12-month default probability by vintage

| vintage_year | loans | adverse_credit | base | high_prepayment | delta_adverse_credit | delta_high_prepayment |
| --- | --- | --- | --- | --- | --- | --- |
| 2023 | 3237 | 0.0136 | 0.0133 | 0.0137 | 0.0002 | 0.0003 |
| 2027 | 13 | 0.0027 | 0.0027 | 0.0031 | 0.0000 | 0.0004 |
| 2022 | 3155 | 0.0141 | 0.0142 | 0.0139 | -0.0000 | -0.0002 |
| 2024 | 226 | 0.0073 | 0.0073 | 0.0076 | -0.0001 | 0.0003 |
| 2020 | 3190 | 0.0038 | 0.0039 | 0.0040 | -0.0001 | 0.0001 |
| 2021 | 3226 | 0.0052 | 0.0052 | 0.0052 | -0.0001 | -0.0001 |
| 2019 | 2927 | 0.0055 | 0.0056 | 0.0055 | -0.0001 | -0.0001 |
| 2025 | 7 | 0.0056 | 0.0064 | 0.0069 | -0.0008 | 0.0005 |
| 2026 | 19 | 0.0620 | 0.0635 | 0.0665 | -0.0016 | 0.0030 |

### 12-month default probability by state

| state | loans | adverse_credit | base | high_prepayment | delta_adverse_credit | delta_high_prepayment |
| --- | --- | --- | --- | --- | --- | --- |
| MN | 354 | 0.0121 | 0.0115 | 0.0122 | 0.0006 | 0.0007 |
| MD | 352 | 0.0043 | 0.0037 | 0.0045 | 0.0006 | 0.0008 |
| VT | 31 | 0.0122 | 0.0118 | 0.0173 | 0.0004 | 0.0055 |
| UT | 270 | 0.0034 | 0.0031 | 0.0036 | 0.0003 | 0.0004 |
| OH | 570 | 0.0149 | 0.0146 | 0.0147 | 0.0003 | 0.0000 |
| FL | 1186 | 0.0096 | 0.0093 | 0.0092 | 0.0002 | -0.0001 |
| NV | 183 | 0.0119 | 0.0117 | 0.0123 | 0.0002 | 0.0005 |
| AZ | 538 | 0.0130 | 0.0129 | 0.0131 | 0.0001 | 0.0002 |
| CA | 1795 | 0.0071 | 0.0070 | 0.0071 | 0.0001 | 0.0001 |
| MO | 296 | 0.0061 | 0.0060 | 0.0064 | 0.0001 | 0.0004 |
| WV | 53 | 0.0007 | 0.0007 | 0.0008 | 0.0001 | 0.0001 |
| CO | 463 | 0.0096 | 0.0096 | 0.0097 | 0.0000 | 0.0001 |
| IA | 126 | 0.0004 | 0.0004 | 0.0004 | 0.0000 | 0.0000 |
| PR | 2 | 0.0006 | 0.0006 | 0.0006 | 0.0000 | 0.0000 |
| VI | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| KS | 126 | 0.0032 | 0.0032 | 0.0032 | -0.0000 | 0.0000 |
| VA | 436 | 0.0074 | 0.0075 | 0.0077 | -0.0000 | 0.0002 |
| DE | 63 | 0.0008 | 0.0008 | 0.0008 | -0.0000 | 0.0000 |
| OR | 303 | 0.0078 | 0.0079 | 0.0070 | -0.0000 | -0.0009 |
| NE | 84 | 0.0082 | 0.0082 | 0.0069 | -0.0000 | -0.0013 |

### 12-month default probability by servicer

| servicer_name | loans | adverse_credit | base | high_prepayment | delta_adverse_credit | delta_high_prepayment |
| --- | --- | --- | --- | --- | --- | --- |
| PENNYMAC LOAN SERVICES, LLC | 277 | 0.0106 | 0.0100 | 0.0110 | 0.0006 | 0.0010 |
| ONSLOW BAY FINANCIAL LLC | 345 | 0.0072 | 0.0068 | 0.0073 | 0.0004 | 0.0005 |
| FREEDOM MORTGAGE CORPORATION | 506 | 0.0125 | 0.0123 | 0.0128 | 0.0002 | 0.0005 |
| LAKEVIEW LOAN SERVICING, LLC | 852 | 0.0143 | 0.0141 | 0.0131 | 0.0002 | -0.0010 |
| WELLS FARGO BANK, N.A. | 730 | 0.0026 | 0.0025 | 0.0023 | 0.0001 | -0.0002 |
| U.S. BANK N.A. | 408 | 0.0108 | 0.0106 | 0.0109 | 0.0001 | 0.0002 |
| CMG MORTGAGE, INC. | 10 | 0.0010 | 0.0009 | 0.0009 | 0.0001 | 0.0000 |
| JPMORGAN CHASE BANK, NATIONAL ASSOCIATION | 1480 | 0.0029 | 0.0028 | 0.0030 | 0.0001 | 0.0002 |
| UNITED WHOLESALE MORTGAGE, LLC | 296 | 0.0174 | 0.0174 | 0.0179 | 0.0000 | 0.0005 |
| SUNTRUST BANK | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| MARLIN MORTGAGE CAPITAL, LLC | 2 | 0.0006 | 0.0006 | 0.0006 | 0.0000 | 0.0000 |
| GUARANTEED RATE, INC. | 3 | 0.0011 | 0.0011 | 0.0011 | 0.0000 | 0.0000 |
| BRANCH BANKING AND TRUST COMPANY | 2 | 0.0004 | 0.0004 | 0.0000 | 0.0000 | -0.0004 |
| MATRIX FINANCIAL SERVICES CORPORATION | 105 | 0.0110 | 0.0110 | 0.0111 | -0.0000 | 0.0001 |
| LOANDEPOT.COM, LLC | 216 | 0.0048 | 0.0048 | 0.0049 | -0.0000 | 0.0001 |
| PINGORA LOAN SERVICING, LLC | 50 | 0.0009 | 0.0010 | 0.0009 | -0.0000 | -0.0001 |
| NATIONSTAR MORTGAGE LLC DBA MR. COOPER | 295 | 0.0008 | 0.0008 | 0.0008 | -0.0000 | -0.0000 |
| PHH MORTGAGE CORPORATION | 61 | 0.0024 | 0.0024 | 0.0017 | -0.0000 | -0.0008 |
| PHH ASSET SERVICES LLC | 145 | 0.0049 | 0.0050 | 0.0039 | -0.0000 | -0.0010 |
| QUICKEN LOANS, LLC | 153 | 0.0014 | 0.0014 | 0.0014 | -0.0000 | -0.0001 |

### 12-month prepayment probability by refinance incentive

Incentive is the loan's note rate minus the prevailing market rate. Positive means the borrower is paying above market and has something to gain by refinancing.

| incentive_bucket | loans | adverse_credit | base | high_prepayment | delta_adverse_credit | delta_high_prepayment |
| --- | --- | --- | --- | --- | --- | --- |
| 0 to 0.5 | 1559 | 0.5924 | 0.5869 | 0.6940 | 0.0055 | 0.1071 |
| -0.5 to 0 | 1278 | 0.4652 | 0.4604 | 0.6923 | 0.0048 | 0.2319 |
| -1.0 to -0.5 | 768 | 0.4208 | 0.4168 | 0.6019 | 0.0040 | 0.1851 |
| 0.5 to 1.0 | 1745 | 0.7711 | 0.7678 | 0.8136 | 0.0033 | 0.0458 |
| >1.0 | 2097 | 0.8459 | 0.8444 | 0.8718 | 0.0015 | 0.0274 |
| <-1.0 | 8280 | 0.2537 | 0.2527 | 0.2602 | 0.0010 | 0.0075 |

## 6. Top scenario drivers

Each macro input is shifted to its scenario value in isolation while everything else stays at base. The interaction residual is the gap between the sum of the isolated shifts and the full joint shift — reported rather than dropped, because it is exactly what an additive attribution cannot represent.

### Drivers of the 12-month default delta

| scenario_name | macro_input | isolated_contribution | share_of_total |
| --- | --- | --- | --- |
| adverse_credit | unemployment_rate | 0.0001 | -10.3539 |
| adverse_credit | market_mortgage_rate | 0.0000 | -0.0000 |
| adverse_credit | hpi_yoy_growth | 0.0000 | -0.0000 |
| adverse_credit | rate_incentive | 0.0000 | -0.0000 |
| adverse_credit | refi_incentive_positive | 0.0000 | -0.0000 |
| adverse_credit | market_rate_delta_12m | 0.0000 | -0.0000 |
| adverse_credit | interaction_residual | -0.0000 | 1.4095 |
| adverse_credit | unemployment_delta_12m | -0.0001 | 9.9444 |
| high_prepayment | market_rate_delta_12m | 0.0001 | 3.2769 |
| high_prepayment | interaction_residual | 0.0001 | 2.6883 |
| high_prepayment | rate_incentive | 0.0000 | 1.5885 |
| high_prepayment | unemployment_rate | 0.0000 | 0.0000 |
| high_prepayment | hpi_yoy_growth | 0.0000 | 0.0000 |
| high_prepayment | unemployment_delta_12m | 0.0000 | 0.0000 |
| high_prepayment | refi_incentive_positive | -0.0001 | -3.1419 |
| high_prepayment | market_mortgage_rate | -0.0001 | -3.4119 |

### Drivers of the 12-month prepayment delta

| scenario_name | macro_input | isolated_contribution | share_of_total |
| --- | --- | --- | --- |
| adverse_credit | unemployment_rate | 0.0016 | 0.7254 |
| adverse_credit | unemployment_delta_12m | 0.0006 | 0.2653 |
| adverse_credit | interaction_residual | 0.0000 | 0.0093 |
| adverse_credit | market_mortgage_rate | 0.0000 | 0.0000 |
| adverse_credit | hpi_yoy_growth | 0.0000 | 0.0000 |
| adverse_credit | rate_incentive | 0.0000 | 0.0000 |
| adverse_credit | refi_incentive_positive | 0.0000 | 0.0000 |
| adverse_credit | market_rate_delta_12m | 0.0000 | 0.0000 |
| high_prepayment | refi_incentive_positive | 0.0187 | 0.3677 |
| high_prepayment | rate_incentive | 0.0104 | 0.2040 |
| high_prepayment | market_rate_delta_12m | 0.0087 | 0.1709 |
| high_prepayment | interaction_residual | 0.0086 | 0.1693 |
| high_prepayment | market_mortgage_rate | 0.0045 | 0.0882 |
| high_prepayment | unemployment_rate | 0.0000 | 0.0000 |
| high_prepayment | hpi_yoy_growth | 0.0000 | 0.0000 |
| high_prepayment | unemployment_delta_12m | 0.0000 | 0.0000 |

## 7. Engine B — macro-conditioned transition model

Sensitivity of each origin state's monthly deterioration and prepayment rate to the macro path, fitted across the panel history on training-window months only.

| origin_state | transition | months_fitted | r_squared | historical_mean_rate | beta_unemployment_rate | beta_hpi_yoy_growth | beta_rate_incentive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Current | deteriorate | 50 | 0.1434 | 0.0051 | 0.0822 | -0.8174 | -0.0416 |
| Current | prepay | 50 | 0.7980 | 0.0133 | 0.1104 | 3.1851 | 0.3307 |
| DQ30 | deteriorate | 32 | 0.7941 | 0.2265 | 0.2004 | -2.5701 | 0.1391 |
| DQ30 | prepay | 32 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### 12-month portfolio state distribution

**adverse_credit**

| horizon_month | Current | DQ30 | DQ60 | DQ90plus | Default | Prepaid |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.9859 | 0.0063 | 0.0020 | 0.0023 | 0.0034 | 0.0000 |
| 3 | 0.9405 | 0.0080 | 0.0041 | 0.0051 | 0.0054 | 0.0369 |
| 6 | 0.8998 | 0.0077 | 0.0040 | 0.0070 | 0.0092 | 0.0722 |
| 9 | 0.8615 | 0.0074 | 0.0039 | 0.0074 | 0.0139 | 0.1060 |
| 12 | 0.8249 | 0.0071 | 0.0037 | 0.0073 | 0.0186 | 0.1383 |

**base**

| horizon_month | Current | DQ30 | DQ60 | DQ90plus | Default | Prepaid |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.9859 | 0.0063 | 0.0020 | 0.0023 | 0.0034 | 0.0000 |
| 3 | 0.9460 | 0.0062 | 0.0018 | 0.0032 | 0.0052 | 0.0375 |
| 6 | 0.9081 | 0.0060 | 0.0018 | 0.0034 | 0.0073 | 0.0734 |
| 9 | 0.8718 | 0.0057 | 0.0017 | 0.0033 | 0.0095 | 0.1079 |
| 12 | 0.8369 | 0.0055 | 0.0016 | 0.0032 | 0.0117 | 0.1411 |

**high_prepayment**

| horizon_month | Current | DQ30 | DQ60 | DQ90plus | Default | Prepaid |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.9859 | 0.0063 | 0.0020 | 0.0023 | 0.0034 | 0.0000 |
| 3 | 0.9460 | 0.0062 | 0.0018 | 0.0032 | 0.0052 | 0.0375 |
| 6 | 0.9081 | 0.0060 | 0.0018 | 0.0034 | 0.0073 | 0.0734 |
| 9 | 0.8718 | 0.0057 | 0.0017 | 0.0033 | 0.0095 | 0.1079 |
| 12 | 0.8369 | 0.0055 | 0.0016 | 0.0032 | 0.0117 | 0.1411 |

### Engine B summary

| scenario_name | cumulative_default_12m | cumulative_prepay_12m | delinquent_12m | delta_cumulative_default_12m | delta_cumulative_prepay_12m | delta_delinquent_12m |
| --- | --- | --- | --- | --- | --- | --- |
| adverse_credit | 0.0186 | 0.1383 | 0.0181 | 0.0070 | -0.0027 | 0.0077 |
| base | 0.0117 | 0.1411 | 0.0104 | 0.0000 | 0.0000 | 0.0000 |
| high_prepayment | 0.0117 | 0.1411 | 0.0104 | 0.0000 | 0.0000 | 0.0000 |

## 8. Do the two engines agree?

| scenario_name | engine_a_default_delta | engine_b_default_delta | engine_a_prepay_delta | engine_b_prepay_delta |
| --- | --- | --- | --- | --- |
| adverse_credit | -0.0000 | 0.0070 | 0.0022 | -0.0027 |
| base | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| high_prepayment | 0.0000 | 0.0000 | 0.0509 | 0.0000 |

The two engines answer different questions and the table above should be read that way. **Engine B carries the credit stress**: adverse conditions move the 12-month cumulative default rate from 1.17% to 1.86%, and the delinquent stock from 1.04% to 1.81% (1.7x). **Engine A carries the refinance response**: the high-prepayment scenario lifts projected 12-month prepayment by 5.1 percentage points. The lift is **not** monotone in incentive, and that is the economically correct shape rather than a defect: loans already deep in the money are near-saturated and have little headroom left, so the largest response comes from loans sitting just below the refinance threshold that the rate cut pushes across it. See the incentive-bucket table in section 5.

Engine A's adverse-credit default delta is -0.001 percentage points — effectively zero, and the sign is not meaningful. That is the identification failure of section 3 showing up in the output rather than being argued about, and it is why the credit stress above is quoted from Engine B and not from Engine A.

**For sizing a credit stress, use Engine B. For deciding which loans to act on, use Engine A's segment detail.** Reporting a single blended number would hide that each engine is only trustworthy on one of the two questions.

## 9. Limitations

- **The credit channel is not identified in Engine A.** This is a property of the data, not a tuning failure: one realised macro path gives no cross-sectional variation in unemployment or HPI. Fixing it properly needs either multiple geographies with differing macro paths (state-level unemployment would do it) or an explicitly specified structural macro-to-hazard link, which is what Engine B provides.
- Scenario paths are illustrative and internally specified, not sourced from a published supervisory scenario. Swapping in a real CCAR or IFRS 9 path means replacing `macro_scenarios.csv`; no code changes are needed.
- Engine A holds every loan-level attribute fixed. In reality a twelve-month horizon would season each loan, amortise its balance and change its status; this is a point-in-time repricing, not a full cashflow projection.
- Engine B's macro sensitivities are fitted on a small number of monthly observations per origin state, so the deep-delinquency coefficients in particular are imprecise.
- No loss-given-default model is fitted, so none of this converts to a dollar loss.
