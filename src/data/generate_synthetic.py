"""Generates a realistic synthetic loan performance panel.

The panel is produced by an explicit monthly state machine over loan status, driven by
borrower credit attributes, loan seasoning, a macroeconomic path and a servicer effect.
Targets are derived from the realised forward path, then feature-side messiness is injected
so that data-quality work has something real to find. Swapping in organiser data means
replacing the CSVs in data/raw with files matching the same schema.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C


def month_range(start: str, end: str) -> pd.PeriodIndex:
    return pd.period_range(start=start, end=end, freq="M")


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def build_macro_history(months: pd.PeriodIndex, rng: np.random.Generator) -> pd.DataFrame:
    n = len(months)
    t = np.arange(n)
    market_rate = (4.40 - 1.60 * logistic((t - 18) / 4.0) + 4.40 * logistic((t - 44) / 4.0)
                   - 1.80 * logistic((t - 74) / 5.0))
    market_rate = market_rate + rng.normal(0, 0.04, n).cumsum() * 0.08
    unemployment = (3.70 + 2.90 * np.exp(-((t - 15) / 3.5) ** 2)
                    + 1.20 * logistic((t - 64) / 8.0) + rng.normal(0, 0.03, n).cumsum() * 0.06)
    hpi_growth = (0.045 + 0.130 * np.exp(-((t - 28) / 8.0) ** 2)
                  - 0.070 * logistic((t - 52) / 6.0) + 0.030 * logistic((t - 76) / 7.0)
                  + rng.normal(0, 0.003, n))
    return pd.DataFrame({
        "reporting_month": months.astype(str),
        "market_mortgage_rate": np.round(market_rate, 3),
        "unemployment_rate": np.round(unemployment, 3),
        "hpi_yoy_growth": np.round(hpi_growth, 4),
    })


def build_loan_book(n_loans: int, months: pd.PeriodIndex, rng: np.random.Generator) -> pd.DataFrame:
    credit_p = np.array([0.04, 0.06, 0.12, 0.16, 0.20, 0.22, 0.20])
    credit = rng.choice(C.CREDIT_BANDS, n_loans, p=credit_p)
    credit_ord = pd.Categorical(credit, categories=C.CREDIT_BANDS, ordered=True).codes.astype(float)

    ltv_logit = -0.35 * (credit_ord - 3) + rng.normal(0, 1.2, n_loans)
    ltv_idx = np.clip(np.digitize(ltv_logit, [-1.4, -0.5, 0.2, 0.9, 1.7]), 0, 5)
    ltv = np.array(C.LTV_BANDS)[ltv_idx]

    dti_logit = -0.30 * (credit_ord - 3) + rng.normal(0, 1.1, n_loans)
    dti_idx = np.clip(np.digitize(dti_logit, [-1.2, -0.3, 0.4, 1.3]), 0, 4)
    dti = np.array(C.DTI_BANDS)[dti_idx]

    orig_start = pd.Period("2015-01", freq="M")
    span = (months[-1] - orig_start).n - 6
    orig_offsets = rng.integers(0, span, n_loans)
    origination = [orig_start + int(o) for o in orig_offsets]
    orig_month_idx = np.array([(p - months[0]).n for p in origination])

    orig_bal = np.round(np.exp(rng.normal(12.35, 0.45, n_loans)) / 1000.0) * 1000.0
    orig_bal = np.clip(orig_bal, 45_000, 1_250_000)

    macro_rate_at_orig = (4.40 - 1.60 * logistic((orig_month_idx - 18) / 4.0)
                          + 4.40 * logistic((orig_month_idx - 44) / 4.0)
                          - 1.80 * logistic((orig_month_idx - 74) / 5.0))
    note_rate = (macro_rate_at_orig + 0.55 - 0.11 * (credit_ord - 3)
                 + 0.09 * ltv_idx + rng.normal(0, 0.22, n_loans))
    note_rate = np.round(np.clip(note_rate, 2.25, 11.5), 3)

    term = rng.choice([180, 240, 360], n_loans, p=[0.12, 0.10, 0.78])
    servicer = rng.choice(C.SERVICERS, n_loans, p=[0.30, 0.24, 0.20, 0.16, 0.10])
    state = rng.choice(C.STATES_US, n_loans,
                       p=np.array([18, 14, 12, 9, 5, 4, 6, 4, 6, 5, 5, 3, 4, 3, 2]) / 100)
    purpose = rng.choice(C.LOAN_PURPOSES, n_loans, p=[0.55, 0.27, 0.18])
    occupancy = rng.choice(C.OCCUPANCY_TYPES, n_loans, p=[0.82, 0.07, 0.11])
    prop_type = rng.choice(C.PROPERTY_TYPES, n_loans, p=[0.68, 0.13, 0.11, 0.05, 0.03])
    frailty = rng.normal(0, 0.55, n_loans)

    return pd.DataFrame({
        "loan_id": [f"LN{100000 + i}" for i in range(n_loans)],
        "origination_month": [str(p) for p in origination],
        "orig_period_idx": orig_month_idx,
        "original_balance": orig_bal,
        "interest_rate": note_rate,
        "original_term_months": term,
        "credit_score_band": credit,
        "credit_ord": credit_ord,
        "ltv_band": ltv,
        "ltv_ord": ltv_idx.astype(float),
        "dti_band": dti,
        "dti_ord": dti_idx.astype(float),
        "state": state,
        "loan_purpose": purpose,
        "occupancy_type": occupancy,
        "property_type": prop_type,
        "servicer_name": servicer,
        "frailty": frailty,
    })


SERVICER_OPS_NOISE = {
    "Northgate Servicing": 0.05,
    "Belmont Loan Services": 0.09,
    "Arcadia Capital Servicing": 0.07,
    "Pioneer Mortgage Ops": 0.18,
    "Kestrel Financial": 0.26,
}
SERVICER_DQ_BIAS = {
    "Northgate Servicing": -0.10,
    "Belmont Loan Services": 0.00,
    "Arcadia Capital Servicing": 0.05,
    "Pioneer Mortgage Ops": 0.22,
    "Kestrel Financial": 0.34,
}
STATE_STRESS = {"NV": 0.30, "FL": 0.22, "AZ": 0.18, "CA": 0.05, "TX": 0.02,
                "GA": 0.12, "OH": 0.08, "MI": 0.10, "IL": 0.06}


def simulate_panel(loans: pd.DataFrame, macro: pd.DataFrame, months: pd.PeriodIndex,
                   rng: np.random.Generator) -> pd.DataFrame:
    n = len(loans)
    n_months = len(months)
    mkt_rate = macro["market_mortgage_rate"].to_numpy()
    unemp = macro["unemployment_rate"].to_numpy()
    hpi = macro["hpi_yoy_growth"].to_numpy()

    credit_ord = loans["credit_ord"].to_numpy()
    ltv_ord = loans["ltv_ord"].to_numpy()
    dti_ord = loans["dti_ord"].to_numpy()
    frailty = loans["frailty"].to_numpy()
    note_rate = loans["interest_rate"].to_numpy()
    orig_bal = loans["original_balance"].to_numpy()
    term = loans["original_term_months"].to_numpy()
    orig_idx = loans["orig_period_idx"].to_numpy()
    servicer = loans["servicer_name"].to_numpy()
    us_state = loans["state"].to_numpy()
    loan_ids = loans["loan_id"].to_numpy()
    is_refi = (loans["loan_purpose"].to_numpy() != "purchase").astype(float)

    svc_bias = np.array([SERVICER_DQ_BIAS[s] for s in servicer])
    state_bias = np.array([STATE_STRESS.get(s, 0.0) for s in us_state])
    static_risk = (-0.46 * (credit_ord - 3.0) + 0.20 * (ltv_ord - 2.0)
                   + 0.22 * (dti_ord - 2.0) + frailty + svc_bias + state_bias)

    monthly_rate = note_rate / 1200.0
    pow_term = (1 + monthly_rate) ** term
    pmt = orig_bal * monthly_rate * pow_term / (pow_term - 1)

    status = np.full(n, "Current", dtype=object)
    balance = orig_bal.copy()
    modified = np.zeros(n, dtype=int)
    active = np.zeros(n, dtype=bool)
    months_in_dq = np.zeros(n, dtype=int)
    burnout = np.zeros(n)
    loss_sev = np.full(n, "", dtype=object)
    dpd_map = {"Current": 0, "DQ30": 30, "DQ60": 60, "DQ90plus": 90,
               "Default": 180, "Prepaid": 0, "PaidOff": 0}

    records = []
    for m in range(n_months):
        age = m - orig_idx
        active = active | ((age >= 0) & ~np.isin(status, list(C.TERMINAL_STATES)))
        live = active & (age >= 0) & (~np.isin(status, list(C.TERMINAL_STATES)))
        if not live.any():
            continue
        idx = np.where(live)[0]
        a = age[idx].astype(float)

        seasoning = 1.35 * logistic((a - 16) / 9.0) - 0.55 * logistic((a - 74) / 18.0)
        macro_stress = 0.45 * (unemp[m] - 3.8) - 3.4 * hpi[m]
        rate_incentive = note_rate[idx] - mkt_rate[m]
        cur = status[idx].copy()

        base_dq = -6.10 + static_risk[idx] + seasoning + macro_stress
        prepay_ramp = np.minimum(a / 30.0, 1.0)
        prepay_logit = (-5.05 + 1.30 * np.maximum(rate_incentive, 0)
                        + 0.30 * np.minimum(rate_incentive, 0)
                        + 1.15 * prepay_ramp - 0.90 * burnout[idx]
                        + 0.16 * (credit_ord[idx] - 3.0) - 0.10 * (ltv_ord[idx] - 2.0)
                        + 0.30 * is_refi[idx] + 0.35 * frailty[idx])
        p_prepay = logistic(prepay_logit)

        new_status = cur.copy()
        u = rng.random(len(idx))

        is_cur = cur == "Current"
        if is_cur.any():
            p_dq = logistic(base_dq[is_cur])
            pp = p_prepay[is_cur]
            sel = u[is_cur]
            new_status[is_cur] = np.where(sel < pp, "Prepaid",
                                          np.where(sel < pp + p_dq, "DQ30", "Current"))

        for src, worse, better in (("DQ30", "DQ60", "Current"),
                                   ("DQ60", "DQ90plus", "DQ30"),
                                   ("DQ90plus", "Default", "DQ60")):
            mask = cur == src
            if not mask.any():
                continue
            sub = idx[mask]
            roll_base = {"DQ30": -0.60, "DQ60": -0.25, "DQ90plus": -1.55}[src]
            cure_base = {"DQ30": -0.65, "DQ60": -1.30, "DQ90plus": -2.30}[src]
            p_roll = logistic(roll_base + 0.45 * static_risk[sub] + 0.30 * macro_stress)
            p_cure = logistic(cure_base - 0.35 * static_risk[sub]
                              + 0.75 * modified[sub] - 0.20 * months_in_dq[sub]) * (1 - p_roll)
            sel = u[mask]
            out = np.where(sel < p_roll, worse, np.where(sel < p_roll + p_cure, better, src))
            if src == "DQ90plus":
                liq = rng.random(int(mask.sum())) < 0.015
                out = np.where((out == src) & liq, "Prepaid", out)
            new_status[mask] = out

        scheduled_end = a >= term[idx]
        new_status = np.where(scheduled_end & (new_status == "Current"), "PaidOff", new_status)

        entering_deep = np.isin(new_status, ["DQ60", "DQ90plus"]) & (modified[idx] == 0)
        newly_modified = entering_deep & (rng.random(len(idx)) < 0.055)
        modified[idx] = np.where(newly_modified, 1, modified[idx])

        interest = balance[idx] * monthly_rate[idx]
        principal = np.clip(pmt[idx] - interest, 0, None)
        paying = np.isin(cur, ["Current", "DQ30"])
        new_balance = np.clip(balance[idx] - np.where(paying, principal, 0.0), 0.0, None)
        new_balance = np.where(np.isin(new_status, ["Prepaid", "PaidOff"]), 0.0, new_balance)

        defaulted = new_status == "Default"
        if defaulted.any():
            sev_logit = (0.72 * ltv_ord[idx][defaulted] - 1.85 - 2.2 * hpi[m]
                         - 0.25 * (credit_ord[idx][defaulted] - 3)
                         + rng.normal(0, 0.7, int(defaulted.sum())))
            sev_idx = np.clip(np.digitize(sev_logit, [-0.4, 0.4, 1.2, 2.1]), 0, 4)
            loss_sev[idx[defaulted]] = np.array(C.LOSS_SEVERITY_BANDS)[sev_idx]

        dpd_now = np.array([dpd_map[s] for s in cur], dtype=float)
        dpd_now = dpd_now + np.where(np.isin(cur, ["DQ30", "DQ60", "DQ90plus"]),
                                     rng.integers(0, 25, len(idx)), 0)

        records.append(pd.DataFrame({
            "loan_id": loan_ids[idx],
            "month_index": m,
            "reporting_month": str(months[m]),
            "loan_age_months": a.astype(int),
            "remaining_term_months": np.clip(term[idx] - a, 0, None).astype(int),
            "current_balance": np.round(balance[idx], 2),
            "current_status": cur,
            "days_past_due": dpd_now.astype(int),
            "modification_flag": modified[idx].copy(),
            "status_next": new_status,
            "loss_severity_band": loss_sev[idx].copy(),
            "market_mortgage_rate": mkt_rate[m],
            "unemployment_rate": unemp[m],
            "hpi_yoy_growth": hpi[m],
        }))

        burnout[idx] = np.where(rate_incentive > 0.35, burnout[idx] + 0.12, burnout[idx] * 0.97)
        months_in_dq[idx] = np.where(np.isin(new_status, ["DQ30", "DQ60", "DQ90plus"]),
                                     months_in_dq[idx] + 1, 0)
        balance[idx] = new_balance
        status[idx] = new_status

    panel = pd.concat(records, ignore_index=True)
    return panel.sort_values(["loan_id", "month_index"], kind="mergesort").reset_index(drop=True)


def attach_forward_targets(panel: pd.DataFrame, last_month_index: int) -> pd.DataFrame:
    """Derives forward-looking labels from the realised path with explicit right-censoring.

    A horizon-k label is 1 if the event occurs within the next k months, 0 if the full
    k-month window is observed with no event or the loan reaches an absorbing state that
    rules the event out, and NaN when the window runs past the end of the panel with no
    event yet seen. NaN rows are censored and are excluded from supervised training.
    """
    panel = panel.sort_values(["loan_id", "month_index"], kind="mergesort").reset_index(drop=True)
    g = panel.groupby("loan_id", sort=False)

    nxt = g["status_next"].shift(0)
    panel["next_state"] = nxt

    dq_now = panel["status_next"].isin(list(C.DELINQUENT_STATES)).astype(float)
    is_default = (panel["status_next"] == "Default").astype(float)
    is_prepay = (panel["status_next"] == "Prepaid").astype(float)

    def forward_any(flag: pd.Series, horizon: int, absorbing_close: pd.Series) -> pd.Series:
        frame = pd.DataFrame({"loan_id": panel["loan_id"], "f": flag.to_numpy(),
                              "close": absorbing_close.to_numpy(),
                              "mi": panel["month_index"].to_numpy()})
        gg = frame.groupby("loan_id", sort=False)
        hit = np.zeros(len(frame))
        avail = np.zeros(len(frame))
        for k in range(horizon):
            hit = np.maximum(hit, gg["f"].shift(-k).fillna(0.0).to_numpy())
            avail = avail + gg["f"].shift(-k).notna().to_numpy().astype(float)
        closed = gg["close"].transform("max").to_numpy()
        last_obs_gap = last_month_index - frame["mi"].to_numpy()
        window_complete = (avail >= horizon) | (closed == 1)
        out = np.where(hit > 0, 1.0, np.where(window_complete, 0.0, np.nan))
        out = np.where((hit == 0) & (last_obs_gap < horizon) & (closed == 0), np.nan, out)
        return pd.Series(out, index=panel.index)

    terminal_seen = panel["status_next"].isin(list(C.TERMINAL_STATES)).astype(float)
    closed_flag = panel.groupby("loan_id", sort=False)["status_next"].transform(
        lambda s: float(s.isin(list(C.TERMINAL_STATES)).any()))

    panel["next_3m_delinquency_flag"] = forward_any(dq_now, 3, closed_flag)
    panel["next_6m_delinquency_flag"] = forward_any(dq_now, 6, closed_flag)
    panel["next_12m_default_flag"] = forward_any(is_default, 12, closed_flag)
    panel["next_12m_prepayment_flag"] = forward_any(is_prepay, 12, closed_flag)

    panel["prepayment_flag"] = is_prepay.astype(int)
    panel["default_flag"] = is_default.astype(int)
    panel["terminal_next"] = terminal_seen
    return panel
