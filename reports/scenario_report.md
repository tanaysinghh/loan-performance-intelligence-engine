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
| adverse_credit | 1500 | 0.1961 | 0.1837 | 0.1600 |
| base | 1500 | 0.1957 | 0.1828 | 0.1609 |
| high_prepayment | 1500 | 0.2117 | 0.1912 | 0.2802 |

| scenario_name | delta_next_6m_delinquency_flag | relative_next_6m_delinquency_flag | delta_next_12m_default_flag | relative_next_12m_default_flag | delta_next_12m_prepayment_flag | relative_next_12m_prepayment_flag |
| --- | --- | --- | --- | --- | --- | --- |
| adverse_credit | 0.0004 | 0.0019 | 0.0009 | 0.0051 | -0.0010 | -0.0060 |
| base | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| high_prepayment | 0.0160 | 0.0815 | 0.0084 | 0.0459 | 0.1192 | 0.7410 |

## 5. Engine A — segment-level impacts

### 12-month default probability by credit band

| credit_score_band | loans | adverse_credit | base | high_prepayment | delta_adverse_credit | delta_high_prepayment |
| --- | --- | --- | --- | --- | --- | --- |
| 580-619 | 100 | 0.5283 | 0.5220 | 0.5457 | 0.0063 | 0.0237 |
| 620-659 | 196 | 0.3826 | 0.3790 | 0.3983 | 0.0036 | 0.0192 |
| 700-739 | 308 | 0.1178 | 0.1173 | 0.1235 | 0.0005 | 0.0062 |
| 740-779 | 334 | 0.0480 | 0.0480 | 0.0496 | -0.0000 | 0.0016 |
| 780+ | 243 | 0.0319 | 0.0320 | 0.0333 | -0.0000 | 0.0014 |
| 660-699 | 220 | 0.1704 | 0.1706 | 0.1796 | -0.0001 | 0.0090 |
| <580 | 63 | 0.6957 | 0.6962 | 0.7192 | -0.0005 | 0.0231 |

### 12-month default probability by ltv band

| ltv_band | loans | adverse_credit | base | high_prepayment | delta_adverse_credit | delta_high_prepayment |
| --- | --- | --- | --- | --- | --- | --- |
| 90-95 | 156 | 0.3174 | 0.3122 | 0.3296 | 0.0052 | 0.0174 |
| 80-90 | 271 | 0.2682 | 0.2662 | 0.2776 | 0.0021 | 0.0114 |
| 70-80 | 287 | 0.1612 | 0.1603 | 0.1675 | 0.0009 | 0.0072 |
| <=60 | 267 | 0.0519 | 0.0512 | 0.0545 | 0.0007 | 0.0033 |
| 60-70 | 368 | 0.1050 | 0.1049 | 0.1091 | 0.0001 | 0.0042 |
| >95 | 100 | 0.4430 | 0.4465 | 0.4656 | -0.0035 | 0.0191 |

### 12-month default probability by vintage

| vintage_year | loans | adverse_credit | base | high_prepayment | delta_adverse_credit | delta_high_prepayment |
| --- | --- | --- | --- | --- | --- | --- |
| 2021 | 124 | 0.1832 | 0.1777 | 0.1779 | 0.0054 | 0.0001 |
| 2024 | 176 | 0.1588 | 0.1556 | 0.1713 | 0.0032 | 0.0157 |
| 2020 | 132 | 0.2568 | 0.2546 | 0.2578 | 0.0022 | 0.0032 |
| 2015 | 125 | 0.1788 | 0.1769 | 0.1833 | 0.0020 | 0.0064 |
| 2016 | 133 | 0.1744 | 0.1730 | 0.1815 | 0.0014 | 0.0085 |
| 2022 | 142 | 0.1681 | 0.1671 | 0.1770 | 0.0011 | 0.0100 |
| 2018 | 123 | 0.2643 | 0.2639 | 0.2695 | 0.0004 | 0.0056 |
| 2019 | 144 | 0.1935 | 0.1933 | 0.1988 | 0.0003 | 0.0055 |
| 2017 | 151 | 0.1767 | 0.1766 | 0.1820 | 0.0000 | 0.0053 |
| 2027 | 3 | 0.0042 | 0.0042 | 0.0058 | 0.0000 | 0.0016 |
| 2026 | 2 | 0.0032 | 0.0032 | 0.0032 | 0.0000 | 0.0000 |
| 2025 | 120 | 0.0668 | 0.0700 | 0.0899 | -0.0032 | 0.0199 |
| 2023 | 125 | 0.2119 | 0.2154 | 0.2261 | -0.0035 | 0.0108 |

### 12-month default probability by state

| state | loans | adverse_credit | base | high_prepayment | delta_adverse_credit | delta_high_prepayment |
| --- | --- | --- | --- | --- | --- | --- |
| TX | 207 | 0.1641 | 0.1593 | 0.1751 | 0.0048 | 0.0158 |
| AZ | 70 | 0.1603 | 0.1574 | 0.1626 | 0.0030 | 0.0053 |
| CA | 271 | 0.2342 | 0.2330 | 0.2427 | 0.0013 | 0.0097 |
| MI | 45 | 0.1581 | 0.1572 | 0.1649 | 0.0009 | 0.0077 |
| CO | 35 | 0.1957 | 0.1951 | 0.1974 | 0.0007 | 0.0024 |
| NV | 62 | 0.2253 | 0.2249 | 0.2279 | 0.0004 | 0.0030 |
| OH | 63 | 0.1236 | 0.1233 | 0.1300 | 0.0003 | 0.0067 |
| FL | 188 | 0.2179 | 0.2179 | 0.2282 | -0.0000 | 0.0103 |
| NJ | 51 | 0.1097 | 0.1098 | 0.1163 | -0.0001 | 0.0065 |
| IL | 92 | 0.2019 | 0.2021 | 0.2041 | -0.0002 | 0.0020 |
| GA | 90 | 0.1571 | 0.1575 | 0.1625 | -0.0003 | 0.0051 |
| PA | 64 | 0.1555 | 0.1558 | 0.1592 | -0.0003 | 0.0034 |
| NC | 60 | 0.1580 | 0.1584 | 0.1643 | -0.0004 | 0.0059 |
| WA | 77 | 0.1729 | 0.1734 | 0.1851 | -0.0004 | 0.0118 |
| NY | 125 | 0.1533 | 0.1543 | 0.1613 | -0.0010 | 0.0070 |

### 12-month default probability by servicer

| servicer_name | loans | adverse_credit | base | high_prepayment | delta_adverse_credit | delta_high_prepayment |
| --- | --- | --- | --- | --- | --- | --- |
| Arcadia Capital Servicing | 297 | 0.1895 | 0.1866 | 0.1940 | 0.0029 | 0.0075 |
| Northgate Servicing | 462 | 0.1625 | 0.1607 | 0.1681 | 0.0017 | 0.0073 |
| Belmont Loan Services | 365 | 0.1911 | 0.1905 | 0.2001 | 0.0006 | 0.0096 |
| Pioneer Mortgage Ops | 215 | 0.1991 | 0.1998 | 0.2092 | -0.0007 | 0.0094 |
| Kestrel Financial | 161 | 0.1966 | 0.1988 | 0.2078 | -0.0022 | 0.0090 |

### 12-month prepayment probability by refinance incentive

Incentive is the loan's note rate minus the prevailing market rate. Positive means the borrower is paying above market and has something to gain by refinancing.

| incentive_bucket | loans | adverse_credit | base | high_prepayment | delta_adverse_credit | delta_high_prepayment |
| --- | --- | --- | --- | --- | --- | --- |
| <-1.0 | 260 | 0.1095 | 0.1057 | 0.0955 | 0.0038 | -0.0102 |
| 0.5 to 1.0 | 223 | 0.2549 | 0.2521 | 0.4875 | 0.0028 | 0.2354 |
| -1.0 to -0.5 | 136 | 0.0770 | 0.0760 | 0.0872 | 0.0011 | 0.0112 |
| 0 to 0.5 | 132 | 0.3054 | 0.3050 | 0.4404 | 0.0004 | 0.1354 |
| -0.5 to 0 | 112 | 0.0940 | 0.0982 | 0.1163 | -0.0042 | 0.0181 |
| >1.0 | 612 | 0.1438 | 0.1482 | 0.3221 | -0.0045 | 0.1739 |

## 6. Top scenario drivers

Each macro input is shifted to its scenario value in isolation while everything else stays at base. The interaction residual is the gap between the sum of the isolated shifts and the full joint shift — reported rather than dropped, because it is exactly what an additive attribution cannot represent.

### Drivers of the 12-month default delta

| scenario_name | macro_input | isolated_contribution | share_of_total |
| --- | --- | --- | --- |
| adverse_credit | unemployment_delta_12m | 0.0011 | 1.1264 |
| adverse_credit | market_rate_delta_12m | 0.0003 | 0.3178 |
| adverse_credit | hpi_yoy_growth | 0.0003 | 0.2975 |
| adverse_credit | rate_incentive | 0.0001 | 0.1102 |
| adverse_credit | market_mortgage_rate | 0.0000 | 0.0000 |
| adverse_credit | unemployment_rate | -0.0002 | -0.2189 |
| adverse_credit | interaction_residual | -0.0002 | -0.2440 |
| adverse_credit | refi_incentive_positive | -0.0004 | -0.3890 |
| high_prepayment | refi_incentive_positive | 0.0026 | 0.3071 |
| high_prepayment | market_mortgage_rate | 0.0021 | 0.2447 |
| high_prepayment | interaction_residual | 0.0018 | 0.2166 |
| high_prepayment | rate_incentive | 0.0013 | 0.1531 |
| high_prepayment | market_rate_delta_12m | 0.0006 | 0.0715 |
| high_prepayment | hpi_yoy_growth | 0.0001 | 0.0069 |
| high_prepayment | unemployment_rate | 0.0000 | 0.0000 |
| high_prepayment | unemployment_delta_12m | 0.0000 | 0.0000 |

### Drivers of the 12-month prepayment delta

| scenario_name | macro_input | isolated_contribution | share_of_total |
| --- | --- | --- | --- |
| adverse_credit | hpi_yoy_growth | 0.0096 | -10.0227 |
| adverse_credit | interaction_residual | 0.0032 | -3.3421 |
| adverse_credit | market_mortgage_rate | -0.0000 | 0.0020 |
| adverse_credit | unemployment_rate | -0.0009 | 0.9086 |
| adverse_credit | unemployment_delta_12m | -0.0016 | 1.6198 |
| adverse_credit | rate_incentive | -0.0028 | 2.9380 |
| adverse_credit | market_rate_delta_12m | -0.0040 | 4.1850 |
| adverse_credit | refi_incentive_positive | -0.0045 | 4.7113 |
| high_prepayment | market_mortgage_rate | 0.0358 | 0.3002 |
| high_prepayment | refi_incentive_positive | 0.0301 | 0.2521 |
| high_prepayment | interaction_residual | 0.0284 | 0.2381 |
| high_prepayment | rate_incentive | 0.0213 | 0.1783 |
| high_prepayment | hpi_yoy_growth | 0.0038 | 0.0322 |
| high_prepayment | unemployment_rate | 0.0000 | 0.0000 |
| high_prepayment | unemployment_delta_12m | 0.0000 | 0.0000 |
| high_prepayment | market_rate_delta_12m | -0.0001 | -0.0008 |

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
| adverse_credit | 0.0009 | 0.0534 | -0.0010 | 0.0669 |
| base | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| high_prepayment | 0.0084 | -0.0042 | 0.1192 | -0.0181 |

The two engines answer different questions and the table above should be read that way. Engine B carries the credit stress: adverse conditions move the 12-month cumulative default rate from 17.4% to 22.7% and roughly triple the delinquent stock (4.1% to 12.2%). Engine A carries the refinance response: the high-prepayment scenario lifts projected 12-month prepayment by 13.0 percentage points, concentrated exactly where theory says it should be — see the incentive-bucket table in section 5.

**For sizing a credit stress, use Engine B. For deciding which loans to act on, use Engine A's segment detail.** Reporting a single blended number would hide that each engine is only trustworthy on one of the two questions.

## 9. Limitations

- **The credit channel is not identified in Engine A.** This is a property of the data, not a tuning failure: one realised macro path gives no cross-sectional variation in unemployment or HPI. Fixing it properly needs either multiple geographies with differing macro paths (state-level unemployment would do it) or an explicitly specified structural macro-to-hazard link, which is what Engine B provides.
- Scenario paths are illustrative and internally specified, not sourced from a published supervisory scenario. Swapping in a real CCAR or IFRS 9 path means replacing `macro_scenarios.csv`; no code changes are needed.
- Engine A holds every loan-level attribute fixed. In reality a twelve-month horizon would season each loan, amortise its balance and change its status; this is a point-in-time repricing, not a full cashflow projection.
- Engine B's macro sensitivities are fitted on a small number of monthly observations per origin state, so the deep-delinquency coefficients in particular are imprecise.
- No loss-given-default model is fitted, so none of this converts to a dollar loss.
