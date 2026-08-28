# Scenario and Stress Simulation Report

**Task 5.** All projections come from the trained LightGBM models and an empirical Markov chain. No language model produces any number in this report.

## 1. Scenario assumptions

Defined in `data/raw/macro_scenarios.csv`, twelve monthly steps each. Stated here explicitly because a stress result is only as meaningful as the assumption behind it.

| scenario_name | market_rate_month_12 | market_rate_shift | unemployment_month_12 | unemployment_shift | hpi_growth_month_12 | assumption |
| --- | --- | --- | --- | --- | --- | --- |
| adverse_credit | 5.7760 | 0.3600 | 7.1320 | 2.2800 | -0.0651 | Unemployment rises ~2.3pp over 12 months and HPI turns negative; rates broadly flat. Stresses default and delinquency transitions. |
| base | 5.4160 | 0.0000 | 5.0920 | 0.2400 | 0.0129 | Macro path continues its current trajectory; no shock applied. |
| high_prepayment | 3.7360 | -1.6800 | 4.9720 | 0.1200 | 0.0405 | Market mortgage rate falls ~1.7pp over 12 months, opening refinance incentive across seasoned high-coupon loans. |

Starting point (latest observed month, 2026-06): market rate 5.42%, unemployment 4.85%, HPI growth 0.007.

## 2. Two engines, deliberately

| engine | method | strength | weakness |
| --- | --- | --- | --- |
| A — model repricing | Overwrite every macro-derived feature on the latest snapshot of each live loan, re-score the Task 2 models. | Uses the full covariate set, so segment detail is real and actionable. Its refinance-incentive channel is cross-sectionally identified. | Its credit channel is NOT identified from a single macro path (section 3). It cannot size a credit stress. |
| B — macro-conditioned Markov | Regress monthly transition log-odds on the macro path, shift inputs to scenario values, rebuild the matrix and roll forward twelve months. | Extrapolates smoothly through a logistic link; gives a full multi-period portfolio path. | Conditions only on current state — no borrower covariates at all. |

They are not redundant. Engine A answers *which loans*; Engine B answers *how bad*. Section 3 shows why neither can be asked to do the other's job here.

## 3. Why Engine A cannot size a credit stress

Engine A's **credit** channel is not identified from this data, and the evidence is in its own output rather than in an argument about it.

Macro levels are constant across every loan within a reporting month. With one realised macro path and 90 monthly observations there is no cross-sectional variation in unemployment or HPI growth at all, so a loan-level model cannot separate the effect of unemployment from the effect of calendar time — they are collinear by construction. What the trees learn is a time proxy, and in this panel's history the low-rate period coincided with the pandemic unemployment spike.

The consequence shows up as two specific wrong answers, both visible in the tables below. A 2.3pp unemployment shock moves the projected 12-month default rate by roughly 0.15% in relative terms, which is not a credible stress result. And the high-prepayment scenario *raises* projected default, which has the sign backwards.

`rate_incentive` is the exception. It is a loan's own note rate minus the prevailing market rate, so it does vary across loans within a month and its effect is identified cross-sectionally. That is why Engine A's prepayment response is trustworthy and its credit response is not, and why Engine B exists.

An earlier iteration tried to fix this by perturbing only the identified refinance-incentive features and leaving macro levels at their observed values. That was worse, not better: it hands the model a feature combination that never occurs in training (a market rate of 5.5% alongside an incentive computed against 5.74%) and the base-case prepayment projection jumped from 0.156 to 0.396 on a scenario that is supposed to be a no-op. Internally consistent shifts plus an honest statement of what the resulting credit number is worth beats a surgical restriction that breaks the input distribution.

## 4. Engine A — portfolio-level projections

| scenario_name | loans | projected_next_6m_delinquency_flag | projected_next_12m_default_flag | projected_next_12m_prepayment_flag |
| --- | --- | --- | --- | --- |
| adverse_credit | 1500 | 0.1969 | 0.1816 | 0.1638 |
| base | 1500 | 0.1943 | 0.1814 | 0.1560 |
| high_prepayment | 1500 | 0.2099 | 0.1902 | 0.2861 |

| scenario_name | delta_next_6m_delinquency_flag | relative_next_6m_delinquency_flag | delta_next_12m_default_flag | relative_next_12m_default_flag | delta_next_12m_prepayment_flag | relative_next_12m_prepayment_flag |
| --- | --- | --- | --- | --- | --- | --- |
| adverse_credit | 0.0026 | 0.0134 | 0.0003 | 0.0015 | 0.0078 | 0.0498 |
| base | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| high_prepayment | 0.0156 | 0.0804 | 0.0088 | 0.0486 | 0.1301 | 0.8338 |

## 5. Engine A — segment-level impacts

### 12-month default probability by credit band

| credit_score_band | loans | adverse_credit | base | high_prepayment | delta_adverse_credit | delta_high_prepayment |
| --- | --- | --- | --- | --- | --- | --- |
| 620-659 | 196 | 0.3832 | 0.3816 | 0.4078 | 0.0015 | 0.0262 |
| 660-699 | 220 | 0.1660 | 0.1653 | 0.1747 | 0.0006 | 0.0093 |
| 700-739 | 308 | 0.1135 | 0.1130 | 0.1152 | 0.0004 | 0.0022 |
| 780+ | 243 | 0.0318 | 0.0316 | 0.0316 | 0.0002 | -0.0000 |
| 580-619 | 100 | 0.5391 | 0.5390 | 0.5645 | 0.0001 | 0.0255 |
| 740-779 | 334 | 0.0451 | 0.0452 | 0.0456 | -0.0001 | 0.0005 |
| <580 | 63 | 0.6819 | 0.6852 | 0.7216 | -0.0032 | 0.0364 |

### 12-month default probability by ltv band

| ltv_band | loans | adverse_credit | base | high_prepayment | delta_adverse_credit | delta_high_prepayment |
| --- | --- | --- | --- | --- | --- | --- |
| >95 | 100 | 0.4408 | 0.4383 | 0.4534 | 0.0025 | 0.0152 |
| 80-90 | 271 | 0.2654 | 0.2644 | 0.2785 | 0.0009 | 0.0141 |
| 70-80 | 287 | 0.1641 | 0.1638 | 0.1715 | 0.0003 | 0.0078 |
| <=60 | 267 | 0.0498 | 0.0496 | 0.0521 | 0.0003 | 0.0025 |
| 60-70 | 368 | 0.1008 | 0.1008 | 0.1053 | 0.0001 | 0.0045 |
| 90-95 | 156 | 0.3123 | 0.3129 | 0.3305 | -0.0005 | 0.0176 |

### 12-month default probability by vintage

| vintage_year | loans | adverse_credit | base | high_prepayment | delta_adverse_credit | delta_high_prepayment |
| --- | --- | --- | --- | --- | --- | --- |
| 2021 | 124 | 0.1804 | 0.1765 | 0.1730 | 0.0039 | -0.0035 |
| 2020 | 132 | 0.2456 | 0.2442 | 0.2446 | 0.0014 | 0.0004 |
| 2024 | 176 | 0.1620 | 0.1607 | 0.1724 | 0.0014 | 0.0117 |
| 2015 | 125 | 0.1776 | 0.1767 | 0.1847 | 0.0009 | 0.0080 |
| 2016 | 133 | 0.1703 | 0.1697 | 0.1829 | 0.0007 | 0.0133 |
| 2019 | 144 | 0.1898 | 0.1893 | 0.1949 | 0.0005 | 0.0056 |
| 2017 | 151 | 0.1738 | 0.1735 | 0.1808 | 0.0003 | 0.0073 |
| 2018 | 123 | 0.2536 | 0.2534 | 0.2605 | 0.0003 | 0.0071 |
| 2027 | 3 | 0.0042 | 0.0041 | 0.0089 | 0.0002 | 0.0048 |
| 2022 | 142 | 0.1664 | 0.1665 | 0.1760 | -0.0001 | 0.0095 |
| 2026 | 2 | 0.0016 | 0.0018 | 0.0027 | -0.0002 | 0.0009 |
| 2023 | 125 | 0.2187 | 0.2211 | 0.2331 | -0.0025 | 0.0120 |
| 2025 | 120 | 0.0712 | 0.0757 | 0.1015 | -0.0045 | 0.0258 |

### 12-month default probability by state

| state | loans | adverse_credit | base | high_prepayment | delta_adverse_credit | delta_high_prepayment |
| --- | --- | --- | --- | --- | --- | --- |
| WA | 77 | 0.1774 | 0.1754 | 0.1852 | 0.0020 | 0.0099 |
| AZ | 70 | 0.1611 | 0.1593 | 0.1610 | 0.0018 | 0.0017 |
| TX | 207 | 0.1629 | 0.1612 | 0.1736 | 0.0016 | 0.0124 |
| GA | 90 | 0.1583 | 0.1568 | 0.1660 | 0.0014 | 0.0092 |
| FL | 188 | 0.2139 | 0.2130 | 0.2218 | 0.0010 | 0.0089 |
| NC | 60 | 0.1515 | 0.1508 | 0.1598 | 0.0007 | 0.0089 |
| PA | 64 | 0.1480 | 0.1480 | 0.1535 | 0.0000 | 0.0055 |
| MI | 45 | 0.1512 | 0.1512 | 0.1605 | -0.0000 | 0.0093 |
| CO | 35 | 0.1971 | 0.1972 | 0.2037 | -0.0001 | 0.0065 |
| CA | 271 | 0.2296 | 0.2298 | 0.2387 | -0.0002 | 0.0090 |
| IL | 92 | 0.2004 | 0.2009 | 0.2014 | -0.0005 | 0.0006 |
| OH | 63 | 0.1302 | 0.1312 | 0.1482 | -0.0010 | 0.0169 |
| NY | 125 | 0.1522 | 0.1536 | 0.1648 | -0.0014 | 0.0112 |
| NV | 62 | 0.2223 | 0.2240 | 0.2303 | -0.0017 | 0.0063 |
| NJ | 51 | 0.1062 | 0.1084 | 0.1165 | -0.0022 | 0.0081 |

### 12-month default probability by servicer

| servicer_name | loans | adverse_credit | base | high_prepayment | delta_adverse_credit | delta_high_prepayment |
| --- | --- | --- | --- | --- | --- | --- |
| Northgate Servicing | 462 | 0.1606 | 0.1594 | 0.1678 | 0.0012 | 0.0083 |
| Arcadia Capital Servicing | 297 | 0.1880 | 0.1870 | 0.1934 | 0.0010 | 0.0064 |
| Pioneer Mortgage Ops | 215 | 0.1973 | 0.1976 | 0.2082 | -0.0003 | 0.0106 |
| Belmont Loan Services | 365 | 0.1884 | 0.1889 | 0.1977 | -0.0005 | 0.0088 |
| Kestrel Financial | 161 | 0.1940 | 0.1951 | 0.2073 | -0.0010 | 0.0123 |

### 12-month prepayment probability by refinance incentive

Incentive is the loan's note rate minus the prevailing market rate. Positive means the borrower is paying above market and has something to gain by refinancing.

| incentive_bucket | loans | adverse_credit | base | high_prepayment | delta_adverse_credit | delta_high_prepayment |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 to 1.0 | 223 | 0.2659 | 0.2491 | 0.4807 | 0.0168 | 0.2316 |
| 0 to 0.5 | 132 | 0.3290 | 0.3173 | 0.4654 | 0.0117 | 0.1482 |
| >1.0 | 612 | 0.1444 | 0.1360 | 0.3331 | 0.0084 | 0.1971 |
| <-1.0 | 260 | 0.1088 | 0.1029 | 0.0926 | 0.0059 | -0.0103 |
| -1.0 to -0.5 | 136 | 0.0801 | 0.0754 | 0.0889 | 0.0046 | 0.0135 |
| -0.5 to 0 | 112 | 0.0884 | 0.0974 | 0.1201 | -0.0090 | 0.0227 |

## 6. Top scenario drivers

Each macro input is shifted to its scenario value in isolation while everything else stays at base. The interaction residual is the gap between the sum of the isolated shifts and the full joint shift — reported rather than dropped, because it is exactly what an additive attribution cannot represent.

### Drivers of the 12-month default delta

| scenario_name | macro_input | isolated_contribution | share_of_total |
| --- | --- | --- | --- |
| adverse_credit | unemployment_delta_12m | 0.0007 | 2.4674 |
| adverse_credit | hpi_yoy_growth | 0.0005 | 1.7372 |
| adverse_credit | market_rate_delta_12m | 0.0001 | 0.2568 |
| adverse_credit | market_mortgage_rate | 0.0001 | 0.1879 |
| adverse_credit | unemployment_rate | 0.0000 | 0.0145 |
| adverse_credit | interaction_residual | -0.0000 | -0.0473 |
| adverse_credit | rate_incentive | -0.0002 | -0.7102 |
| adverse_credit | refi_incentive_positive | -0.0008 | -2.9061 |
| high_prepayment | market_mortgage_rate | 0.0025 | 0.2861 |
| high_prepayment | refi_incentive_positive | 0.0023 | 0.2598 |
| high_prepayment | interaction_residual | 0.0020 | 0.2216 |
| high_prepayment | rate_incentive | 0.0016 | 0.1791 |
| high_prepayment | hpi_yoy_growth | 0.0004 | 0.0400 |
| high_prepayment | market_rate_delta_12m | 0.0001 | 0.0134 |
| high_prepayment | unemployment_rate | 0.0000 | 0.0000 |
| high_prepayment | unemployment_delta_12m | 0.0000 | 0.0000 |

### Drivers of the 12-month prepayment delta

| scenario_name | macro_input | isolated_contribution | share_of_total |
| --- | --- | --- | --- |
| adverse_credit | hpi_yoy_growth | 0.0119 | 1.5368 |
| adverse_credit | market_rate_delta_12m | 0.0019 | 0.2383 |
| adverse_credit | unemployment_rate | 0.0008 | 0.1076 |
| adverse_credit | market_mortgage_rate | 0.0001 | 0.0188 |
| adverse_credit | unemployment_delta_12m | -0.0011 | -0.1356 |
| adverse_credit | interaction_residual | -0.0014 | -0.1777 |
| adverse_credit | rate_incentive | -0.0022 | -0.2822 |
| adverse_credit | refi_incentive_positive | -0.0024 | -0.3059 |
| high_prepayment | market_mortgage_rate | 0.0408 | 0.3134 |
| high_prepayment | interaction_residual | 0.0285 | 0.2189 |
| high_prepayment | refi_incentive_positive | 0.0279 | 0.2142 |
| high_prepayment | rate_incentive | 0.0270 | 0.2073 |
| high_prepayment | market_rate_delta_12m | 0.0032 | 0.0249 |
| high_prepayment | hpi_yoy_growth | 0.0028 | 0.0212 |
| high_prepayment | unemployment_rate | 0.0000 | 0.0000 |
| high_prepayment | unemployment_delta_12m | 0.0000 | 0.0000 |

## 7. Engine B — macro-conditioned transition model

Sensitivity of each origin state's monthly deterioration and prepayment rate to the macro path, fitted across the panel history on training-window months only.

| origin_state | transition | months_fitted | r_squared | historical_mean_rate | beta_unemployment_rate | beta_hpi_yoy_growth | beta_rate_incentive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Current | deteriorate | 54 | 0.2079 | 0.0063 | 0.4139 | -8.9177 | 0.4296 |
| Current | prepay | 54 | 0.4812 | 0.0186 | 0.0704 | -4.8513 | 0.7766 |

### 12-month portfolio state distribution

**adverse_credit**

| horizon_month | Current | DQ30 | DQ60 | DQ90plus | Default | Prepaid |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.8253 | 0.0080 | 0.0047 | 0.1620 | 0.0000 | 0.0000 |
| 3 | 0.6908 | 0.0416 | 0.0253 | 0.0841 | 0.0841 | 0.0740 |
| 6 | 0.5836 | 0.0410 | 0.0323 | 0.0721 | 0.1354 | 0.1356 |
| 9 | 0.4939 | 0.0360 | 0.0302 | 0.0697 | 0.1822 | 0.1879 |
| 12 | 0.4183 | 0.0308 | 0.0265 | 0.0650 | 0.2270 | 0.2324 |

**base**

| horizon_month | Current | DQ30 | DQ60 | DQ90plus | Default | Prepaid |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.8253 | 0.0080 | 0.0047 | 0.1620 | 0.0000 | 0.0000 |
| 3 | 0.7678 | 0.0110 | 0.0105 | 0.0798 | 0.0839 | 0.0470 |
| 6 | 0.7147 | 0.0108 | 0.0100 | 0.0471 | 0.1277 | 0.0897 |
| 9 | 0.6654 | 0.0101 | 0.0089 | 0.0319 | 0.1547 | 0.1290 |
| 12 | 0.6194 | 0.0094 | 0.0080 | 0.0241 | 0.1736 | 0.1655 |

**high_prepayment**

| horizon_month | Current | DQ30 | DQ60 | DQ90plus | Default | Prepaid |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.8253 | 0.0080 | 0.0047 | 0.1620 | 0.0000 | 0.0000 |
| 3 | 0.7771 | 0.0087 | 0.0094 | 0.0795 | 0.0839 | 0.0414 |
| 6 | 0.7318 | 0.0084 | 0.0082 | 0.0452 | 0.1272 | 0.0792 |
| 9 | 0.6892 | 0.0078 | 0.0071 | 0.0289 | 0.1526 | 0.1144 |
| 12 | 0.6489 | 0.0073 | 0.0063 | 0.0207 | 0.1695 | 0.1473 |

### Engine B summary

| scenario_name | cumulative_default_12m | cumulative_prepay_12m | delinquent_12m | delta_cumulative_default_12m | delta_cumulative_prepay_12m | delta_delinquent_12m |
| --- | --- | --- | --- | --- | --- | --- |
| adverse_credit | 0.2270 | 0.2324 | 0.1223 | 0.0534 | 0.0669 | 0.0809 |
| base | 0.1736 | 0.1655 | 0.0414 | 0.0000 | 0.0000 | 0.0000 |
| high_prepayment | 0.1695 | 0.1473 | 0.0342 | -0.0042 | -0.0181 | -0.0072 |

## 8. Do the two engines agree?

| scenario_name | engine_a_default_delta | engine_b_default_delta | engine_a_prepay_delta | engine_b_prepay_delta |
| --- | --- | --- | --- | --- |
| adverse_credit | 0.0003 | 0.0534 | 0.0078 | 0.0669 |
| base | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| high_prepayment | 0.0088 | -0.0042 | 0.1301 | -0.0181 |

The two engines answer different questions and the table above should be read that way. Engine B carries the credit stress: adverse conditions move the 12-month cumulative default rate from 17.4% to 22.7% and roughly triple the delinquent stock (4.1% to 12.2%). Engine A carries the refinance response: the high-prepayment scenario lifts projected 12-month prepayment by 13.0 percentage points, concentrated exactly where theory says it should be — see the incentive-bucket table in section 5.

**For sizing a credit stress, use Engine B. For deciding which loans to act on, use Engine A's segment detail.** Reporting a single blended number would hide that each engine is only trustworthy on one of the two questions.

## 9. Limitations

- **The credit channel is not identified in Engine A.** This is a property of the data, not a tuning failure: one realised macro path gives no cross-sectional variation in unemployment or HPI. Fixing it properly needs either multiple geographies with differing macro paths (state-level unemployment would do it) or an explicitly specified structural macro-to-hazard link, which is what Engine B provides.
- Scenario paths are illustrative and internally specified, not sourced from a published supervisory scenario. Swapping in a real CCAR or IFRS 9 path means replacing `macro_scenarios.csv`; no code changes are needed.
- Engine A holds every loan-level attribute fixed. In reality a twelve-month horizon would season each loan, amortise its balance and change its status; this is a point-in-time repricing, not a full cashflow projection.
- Engine B's macro sensitivities are fitted on a small number of monthly observations per origin state, so the deep-delinquency coefficients in particular are imprecise.
- No loss-given-default model is fitted, so none of this converts to a dollar loss.
