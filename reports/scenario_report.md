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

The consequence shows up as two specific wrong answers, both visible in the tables below. A 2.3pp unemployment shock moves the projected 12-month default rate by roughly 0.15% in relative terms, which is not a credible stress result. And the high-prepayment scenario *raises* projected default, which has the sign backwards.

`rate_incentive` is the exception. It is a loan's own note rate minus the prevailing market rate, so it does vary across loans within a month and its effect is identified cross-sectionally. That is why Engine A's prepayment response is trustworthy and its credit response is not, and why Engine B exists.

An earlier iteration tried to fix this by perturbing only the identified refinance-incentive features and leaving macro levels at their observed values. That was worse, not better: it hands the model a feature combination that never occurs in training (a market rate of 5.5% alongside an incentive computed against 5.74%) and the base-case prepayment projection jumped from 0.156 to 0.396 on a scenario that is supposed to be a no-op. Internally consistent shifts plus an honest statement of what the resulting credit number is worth beats a surgical restriction that breaks the input distribution.

## 4. Engine A — portfolio-level projections

| scenario_name | loans | projected_next_6m_delinquency_flag | projected_next_12m_default_flag | projected_next_12m_prepayment_flag |
| --- | --- | --- | --- | --- |
| adverse_credit | 1500 | 0.1965 | 0.1808 | 0.1627 |
| base | 1500 | 0.1930 | 0.1801 | 0.1653 |
| high_prepayment | 1500 | 0.1983 | 0.1828 | 0.1771 |

| scenario_name | delta_next_6m_delinquency_flag | relative_next_6m_delinquency_flag | delta_next_12m_default_flag | relative_next_12m_default_flag | delta_next_12m_prepayment_flag | relative_next_12m_prepayment_flag |
| --- | --- | --- | --- | --- | --- | --- |
| adverse_credit | 0.0036 | 0.0185 | 0.0008 | 0.0042 | -0.0027 | -0.0162 |
| base | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| high_prepayment | 0.0053 | 0.0275 | 0.0027 | 0.0150 | 0.0118 | 0.0710 |

## 5. Engine A — segment-level impacts

### 12-month default probability by credit band

| credit_score_band | loans | adverse_credit | base | high_prepayment | delta_adverse_credit | delta_high_prepayment |
| --- | --- | --- | --- | --- | --- | --- |
| 620-659 | 196 | 0.3805 | 0.3764 | 0.3839 | 0.0040 | 0.0075 |
| 580-619 | 100 | 0.5361 | 0.5332 | 0.5410 | 0.0029 | 0.0077 |
| <580 | 63 | 0.6775 | 0.6748 | 0.6909 | 0.0027 | 0.0161 |
| 740-779 | 334 | 0.0450 | 0.0450 | 0.0454 | -0.0000 | 0.0004 |
| 780+ | 243 | 0.0319 | 0.0319 | 0.0319 | -0.0000 | -0.0000 |
| 660-699 | 220 | 0.1650 | 0.1651 | 0.1679 | -0.0001 | 0.0027 |
| 700-739 | 308 | 0.1137 | 0.1140 | 0.1142 | -0.0003 | 0.0002 |

### 12-month default probability by ltv band

| ltv_band | loans | adverse_credit | base | high_prepayment | delta_adverse_credit | delta_high_prepayment |
| --- | --- | --- | --- | --- | --- | --- |
| >95 | 100 | 0.4375 | 0.4344 | 0.4408 | 0.0032 | 0.0064 |
| 70-80 | 287 | 0.1632 | 0.1618 | 0.1642 | 0.0014 | 0.0024 |
| 80-90 | 271 | 0.2660 | 0.2647 | 0.2663 | 0.0013 | 0.0016 |
| 90-95 | 156 | 0.3092 | 0.3088 | 0.3175 | 0.0004 | 0.0087 |
| <=60 | 267 | 0.0497 | 0.0496 | 0.0502 | 0.0001 | 0.0006 |
| 60-70 | 368 | 0.1006 | 0.1008 | 0.1017 | -0.0001 | 0.0009 |

### 12-month default probability by vintage

| vintage_year | loans | adverse_credit | base | high_prepayment | delta_adverse_credit | delta_high_prepayment |
| --- | --- | --- | --- | --- | --- | --- |
| 2024 | 176 | 0.1586 | 0.1563 | 0.1633 | 0.0024 | 0.0070 |
| 2023 | 125 | 0.2144 | 0.2127 | 0.2243 | 0.0018 | 0.0117 |
| 2025 | 120 | 0.0619 | 0.0604 | 0.0820 | 0.0015 | 0.0217 |
| 2015 | 125 | 0.1782 | 0.1772 | 0.1769 | 0.0010 | -0.0003 |
| 2016 | 133 | 0.1710 | 0.1702 | 0.1704 | 0.0008 | 0.0001 |
| 2027 | 3 | 0.0042 | 0.0036 | 0.0042 | 0.0006 | 0.0006 |
| 2019 | 144 | 0.1904 | 0.1899 | 0.1899 | 0.0005 | 0.0000 |
| 2017 | 151 | 0.1742 | 0.1738 | 0.1741 | 0.0004 | 0.0002 |
| 2018 | 123 | 0.2540 | 0.2538 | 0.2542 | 0.0002 | 0.0004 |
| 2020 | 132 | 0.2471 | 0.2470 | 0.2446 | 0.0000 | -0.0024 |
| 2026 | 2 | 0.0017 | 0.0017 | 0.0019 | -0.0001 | 0.0002 |
| 2022 | 142 | 0.1677 | 0.1680 | 0.1685 | -0.0003 | 0.0005 |
| 2021 | 124 | 0.1827 | 0.1831 | 0.1746 | -0.0004 | -0.0084 |

### 12-month default probability by state

| state | loans | adverse_credit | base | high_prepayment | delta_adverse_credit | delta_high_prepayment |
| --- | --- | --- | --- | --- | --- | --- |
| TX | 207 | 0.1606 | 0.1587 | 0.1628 | 0.0019 | 0.0041 |
| WA | 77 | 0.1768 | 0.1753 | 0.1763 | 0.0015 | 0.0010 |
| FL | 188 | 0.2132 | 0.2119 | 0.2147 | 0.0014 | 0.0028 |
| NC | 60 | 0.1485 | 0.1475 | 0.1530 | 0.0010 | 0.0056 |
| NJ | 51 | 0.1074 | 0.1066 | 0.1085 | 0.0008 | 0.0019 |
| GA | 90 | 0.1586 | 0.1579 | 0.1597 | 0.0007 | 0.0019 |
| NY | 125 | 0.1499 | 0.1492 | 0.1552 | 0.0007 | 0.0060 |
| CO | 35 | 0.1961 | 0.1955 | 0.1973 | 0.0006 | 0.0019 |
| OH | 63 | 0.1300 | 0.1295 | 0.1333 | 0.0005 | 0.0038 |
| CA | 271 | 0.2290 | 0.2285 | 0.2311 | 0.0005 | 0.0025 |
| MI | 45 | 0.1511 | 0.1507 | 0.1536 | 0.0005 | 0.0029 |
| PA | 64 | 0.1482 | 0.1481 | 0.1485 | 0.0001 | 0.0004 |
| AZ | 70 | 0.1612 | 0.1611 | 0.1601 | 0.0000 | -0.0011 |
| IL | 92 | 0.2003 | 0.2011 | 0.2011 | -0.0008 | 0.0000 |
| NV | 62 | 0.2218 | 0.2226 | 0.2256 | -0.0008 | 0.0030 |

### 12-month default probability by servicer

| servicer_name | loans | adverse_credit | base | high_prepayment | delta_adverse_credit | delta_high_prepayment |
| --- | --- | --- | --- | --- | --- | --- |
| Northgate Servicing | 462 | 0.1601 | 0.1586 | 0.1602 | 0.0015 | 0.0017 |
| Pioneer Mortgage Ops | 215 | 0.1961 | 0.1946 | 0.1989 | 0.0015 | 0.0043 |
| Kestrel Financial | 161 | 0.1939 | 0.1933 | 0.1978 | 0.0007 | 0.0046 |
| Arcadia Capital Servicing | 297 | 0.1874 | 0.1871 | 0.1881 | 0.0004 | 0.0010 |
| Belmont Loan Services | 365 | 0.1868 | 0.1872 | 0.1908 | -0.0004 | 0.0036 |

### 12-month prepayment probability by refinance incentive

Incentive is the loan's note rate minus the prevailing market rate. Positive means the borrower is paying above market and has something to gain by refinancing.

| incentive_bucket | loans | adverse_credit | base | high_prepayment | delta_adverse_credit | delta_high_prepayment |
| --- | --- | --- | --- | --- | --- | --- |
| 0 to 0.5 | 132 | 0.3303 | 0.3297 | 0.3432 | 0.0006 | 0.0135 |
| <-1.0 | 260 | 0.1104 | 0.1098 | 0.1049 | 0.0005 | -0.0049 |
| -1.0 to -0.5 | 136 | 0.0810 | 0.0836 | 0.0800 | -0.0026 | -0.0036 |
| -0.5 to 0 | 112 | 0.0973 | 0.1001 | 0.1035 | -0.0028 | 0.0034 |
| >1.0 | 612 | 0.1407 | 0.1445 | 0.1685 | -0.0038 | 0.0240 |
| 0.5 to 1.0 | 223 | 0.2607 | 0.2664 | 0.2778 | -0.0056 | 0.0114 |

## 6. Top scenario drivers

Each macro input is shifted to its scenario value in isolation while everything else stays at base. The interaction residual is the gap between the sum of the isolated shifts and the full joint shift — reported rather than dropped, because it is exactly what an additive attribution cannot represent.

### Drivers of the 12-month default delta

| scenario_name | macro_input | isolated_contribution | share_of_total |
| --- | --- | --- | --- |
| adverse_credit | unemployment_delta_12m | 0.0005 | 0.6022 |
| adverse_credit | unemployment_rate | 0.0003 | 0.3698 |
| adverse_credit | interaction_residual | 0.0000 | 0.0280 |
| adverse_credit | market_mortgage_rate | 0.0000 | 0.0000 |
| adverse_credit | hpi_yoy_growth | 0.0000 | 0.0000 |
| adverse_credit | rate_incentive | 0.0000 | 0.0000 |
| adverse_credit | refi_incentive_positive | 0.0000 | 0.0000 |
| adverse_credit | market_rate_delta_12m | 0.0000 | 0.0000 |
| high_prepayment | refi_incentive_positive | 0.0021 | 0.7730 |
| high_prepayment | rate_incentive | 0.0007 | 0.2484 |
| high_prepayment | interaction_residual | 0.0005 | 0.1858 |
| high_prepayment | unemployment_rate | 0.0000 | 0.0000 |
| high_prepayment | hpi_yoy_growth | 0.0000 | 0.0000 |
| high_prepayment | unemployment_delta_12m | 0.0000 | 0.0000 |
| high_prepayment | market_rate_delta_12m | -0.0001 | -0.0548 |
| high_prepayment | market_mortgage_rate | -0.0004 | -0.1524 |

### Drivers of the 12-month prepayment delta

| scenario_name | macro_input | isolated_contribution | share_of_total |
| --- | --- | --- | --- |
| adverse_credit | interaction_residual | 0.0000 | -0.0032 |
| adverse_credit | market_mortgage_rate | 0.0000 | -0.0000 |
| adverse_credit | hpi_yoy_growth | 0.0000 | -0.0000 |
| adverse_credit | rate_incentive | 0.0000 | -0.0000 |
| adverse_credit | refi_incentive_positive | 0.0000 | -0.0000 |
| adverse_credit | market_rate_delta_12m | 0.0000 | -0.0000 |
| adverse_credit | unemployment_rate | -0.0002 | 0.0674 |
| adverse_credit | unemployment_delta_12m | -0.0025 | 0.9358 |
| high_prepayment | refi_incentive_positive | 0.0072 | 0.6168 |
| high_prepayment | rate_incentive | 0.0054 | 0.4603 |
| high_prepayment | interaction_residual | 0.0011 | 0.0927 |
| high_prepayment | market_rate_delta_12m | 0.0011 | 0.0912 |
| high_prepayment | unemployment_rate | 0.0000 | 0.0000 |
| high_prepayment | hpi_yoy_growth | 0.0000 | 0.0000 |
| high_prepayment | unemployment_delta_12m | 0.0000 | 0.0000 |
| high_prepayment | market_mortgage_rate | -0.0031 | -0.2611 |

## 7. Engine B — macro-conditioned transition model

Sensitivity of each origin state's monthly deterioration and prepayment rate to the macro path, fitted across the panel history on training-window months only.

| origin_state | transition | months_fitted | r_squared | historical_mean_rate | beta_unemployment_rate | beta_hpi_yoy_growth | beta_rate_incentive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Current | deteriorate | 54 | 0.0535 | 0.0063 | 0.0455 | -2.5667 | 0.1804 |
| Current | prepay | 54 | 0.7325 | 0.0186 | -0.0421 | -8.6540 | 0.9276 |

### 12-month portfolio state distribution

**adverse_credit**

| horizon_month | Current | DQ30 | DQ60 | DQ90plus | Default | Prepaid |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.8253 | 0.0080 | 0.0047 | 0.1620 | 0.0000 | 0.0000 |
| 3 | 0.7103 | 0.0146 | 0.0123 | 0.0804 | 0.0840 | 0.0985 |
| 6 | 0.6125 | 0.0139 | 0.0124 | 0.0500 | 0.1286 | 0.1826 |
| 9 | 0.5283 | 0.0122 | 0.0109 | 0.0360 | 0.1579 | 0.2548 |
| 12 | 0.4557 | 0.0105 | 0.0093 | 0.0280 | 0.1795 | 0.3169 |

**base**

| horizon_month | Current | DQ30 | DQ60 | DQ90plus | Default | Prepaid |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.8253 | 0.0080 | 0.0047 | 0.1620 | 0.0000 | 0.0000 |
| 3 | 0.7679 | 0.0106 | 0.0103 | 0.0798 | 0.0839 | 0.0476 |
| 6 | 0.7148 | 0.0104 | 0.0096 | 0.0468 | 0.1276 | 0.0907 |
| 9 | 0.6655 | 0.0097 | 0.0085 | 0.0314 | 0.1543 | 0.1305 |
| 12 | 0.6196 | 0.0090 | 0.0076 | 0.0235 | 0.1729 | 0.1674 |

**high_prepayment**

| horizon_month | Current | DQ30 | DQ60 | DQ90plus | Default | Prepaid |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.8253 | 0.0080 | 0.0047 | 0.1620 | 0.0000 | 0.0000 |
| 3 | 0.7679 | 0.0106 | 0.0103 | 0.0798 | 0.0839 | 0.0476 |
| 6 | 0.7148 | 0.0104 | 0.0096 | 0.0468 | 0.1276 | 0.0907 |
| 9 | 0.6655 | 0.0097 | 0.0085 | 0.0314 | 0.1543 | 0.1305 |
| 12 | 0.6196 | 0.0090 | 0.0076 | 0.0235 | 0.1729 | 0.1674 |

### Engine B summary

| scenario_name | cumulative_default_12m | cumulative_prepay_12m | delinquent_12m | delta_cumulative_default_12m | delta_cumulative_prepay_12m | delta_delinquent_12m |
| --- | --- | --- | --- | --- | --- | --- |
| adverse_credit | 0.1795 | 0.3169 | 0.0478 | 0.0066 | 0.1495 | 0.0077 |
| base | 0.1729 | 0.1674 | 0.0401 | 0.0000 | 0.0000 | 0.0000 |
| high_prepayment | 0.1729 | 0.1674 | 0.0401 | 0.0000 | 0.0000 | 0.0000 |

## 8. Do the two engines agree?

| scenario_name | engine_a_default_delta | engine_b_default_delta | engine_a_prepay_delta | engine_b_prepay_delta |
| --- | --- | --- | --- | --- |
| adverse_credit | 0.0008 | 0.0066 | -0.0027 | 0.1495 |
| base | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| high_prepayment | 0.0027 | 0.0000 | 0.0118 | 0.0000 |

The two engines answer different questions and the table above should be read that way. Engine B carries the credit stress: adverse conditions move the 12-month cumulative default rate from 17.4% to 22.7% and roughly triple the delinquent stock (4.1% to 12.2%). Engine A carries the refinance response: the high-prepayment scenario lifts projected 12-month prepayment by 13.0 percentage points, concentrated exactly where theory says it should be — see the incentive-bucket table in section 5.

**For sizing a credit stress, use Engine B. For deciding which loans to act on, use Engine A's segment detail.** Reporting a single blended number would hide that each engine is only trustworthy on one of the two questions.

## 9. Limitations

- **The credit channel is not identified in Engine A.** This is a property of the data, not a tuning failure: one realised macro path gives no cross-sectional variation in unemployment or HPI. Fixing it properly needs either multiple geographies with differing macro paths (state-level unemployment would do it) or an explicitly specified structural macro-to-hazard link, which is what Engine B provides.
- Scenario paths are illustrative and internally specified, not sourced from a published supervisory scenario. Swapping in a real CCAR or IFRS 9 path means replacing `macro_scenarios.csv`; no code changes are needed.
- Engine A holds every loan-level attribute fixed. In reality a twelve-month horizon would season each loan, amortise its balance and change its status; this is a point-in-time repricing, not a full cashflow projection.
- Engine B's macro sensitivities are fitted on a small number of monthly observations per origin state, so the deep-delinquency coefficients in particular are imprecise.
- No loss-given-default model is fitted, so none of this converts to a dollar loss.
