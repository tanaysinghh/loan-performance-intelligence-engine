from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index

from src import config as C

SURVIVAL_COVARIATES = ["credit_ord", "ltv_ord", "dti_ord", "log_original_balance",
                       "interest_rate_clean", "rate_incentive_at_entry", "is_investment",
                       "is_cash_out", "is_high_ops_servicer"]

HIGH_OPS_SERVICER_QUANTILE = 0.25


def high_ops_servicers(df: pd.DataFrame) -> set:
    if "dq_score" not in df.columns or "servicer_name" not in df.columns:
        return set()
    by_servicer = df.groupby("servicer_name")["dq_score"].mean()
    if by_servicer.nunique() < 2:
        return set()
    cutoff = by_servicer.quantile(HIGH_OPS_SERVICER_QUANTILE)
    return set(by_servicer[by_servicer <= cutoff].index)


def build_survival_frame(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values(["loan_id", "month_index"], kind="mergesort")
    g = d.groupby("loan_id", sort=False)

    first = g.first()
    last = g.last()
    out = pd.DataFrame(index=first.index)

    out["entry_age"] = g["loan_age_months_clean"].min().clip(lower=0)
    last_age = g["loan_age_months_clean"].max().clip(lower=0)
    out["last_month_index"] = last["month_index"]

    serious = d["current_status"].isin(["DQ90plus", "Default"])
    serious_age = d["loan_age_months_clean"].where(serious)
    t_serious = serious_age.groupby(d["loan_id"], sort=False).min()

    prepaid_loan = (last["next_state"] == "Prepaid")
    t_prepay = last_age.where(prepaid_loan)

    t_serious = t_serious.reindex(out.index)
    t_prepay = t_prepay.reindex(out.index)

    default_first = t_serious.notna() & (t_prepay.isna() | (t_serious <= t_prepay))
    prepay_first = t_prepay.notna() & ~default_first

    out["event_default"] = default_first.astype(int)
    out["event_prepay"] = prepay_first.astype(int)
    out["event_any"] = out["event_default"] | out["event_prepay"]
    out["event_type"] = np.where(out["event_default"] == 1, "default",
                                 np.where(out["event_prepay"] == 1, "prepay", "censored"))
    out["exit_age"] = np.where(default_first, t_serious,
                               np.where(prepay_first, t_prepay, last_age))

    panel_end = int(df["month_index"].max())
    out["administratively_censored"] = ((out["event_any"] == 0)
                                        & (out["last_month_index"] >= panel_end)).astype(int)
    out["left_truncated"] = (out["entry_age"] > 0).astype(int)

    out["credit_ord"] = g["credit_ord"].median()
    out["ltv_ord"] = g["ltv_ord"].median()
    out["dti_ord"] = g["dti_ord"].median()
    out["log_original_balance"] = np.log1p(first["original_balance"])
    out["interest_rate_clean"] = g["interest_rate_clean"].median()
    out["rate_incentive_at_entry"] = first["rate_incentive"]
    out["is_investment"] = (first["occupancy_type"] == "investment").astype(float)
    out["is_cash_out"] = (first["loan_purpose"] == "cash_out_refi").astype(float)
    out["is_high_ops_servicer"] = first["servicer_name"].isin(
        high_ops_servicers(df)).astype(float)
    out["credit_score_band"] = first["credit_score_band"]
    out["ltv_band"] = first["ltv_band"]
    out["servicer_name"] = first["servicer_name"]
    out["state"] = first["state"]
    out["vintage_year"] = first["origination_period"].dt.year

    out["duration"] = (out["exit_age"] - out["entry_age"]).clip(lower=0.5)
    out["exit_age"] = out["exit_age"].clip(lower=out["entry_age"] + 0.5)
    return out.reset_index()


def kaplan_meier_curves(surv: pd.DataFrame, event_col: str = "event_default",
                        by: str | None = None, max_age: int = 120) -> pd.DataFrame:
    rows = []
    groups = [("all", surv)] if by is None else list(surv.groupby(by, observed=True))
    grid = np.arange(0, max_age + 1)
    for name, sub in groups:
        if len(sub) < 40:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(durations=sub["exit_age"], event_observed=sub[event_col],
                entry=sub["entry_age"], label=str(name))
        sf = kmf.survival_function_at_times(grid).to_numpy()
        rows.append(pd.DataFrame({"group": str(name), "loan_age_months": grid,
                                  "survival": sf, "cumulative_event_prob": 1 - sf,
                                  "n_loans": len(sub), "events": int(sub[event_col].sum())}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def cumulative_incidence(surv: pd.DataFrame, max_age: int = 120) -> pd.DataFrame:
    grid = np.arange(0, max_age + 1)
    entry = surv["entry_age"].to_numpy()
    exit_ = surv["exit_age"].to_numpy()
    d_evt = surv["event_default"].to_numpy()
    p_evt = surv["event_prepay"].to_numpy()

    times = np.unique(exit_[(d_evt == 1) | (p_evt == 1)])
    times = times[times <= max_age]
    overall_surv = 1.0
    cif_d, cif_p = 0.0, 0.0
    records = []
    for t in times:
        at_risk = int(((entry < t) & (exit_ >= t)).sum())
        if at_risk == 0:
            continue
        nd = int(((exit_ == t) & (d_evt == 1)).sum())
        npz = int(((exit_ == t) & (p_evt == 1)).sum())
        cif_d += overall_surv * nd / at_risk
        cif_p += overall_surv * npz / at_risk
        overall_surv *= (1 - (nd + npz) / at_risk)
        records.append({"loan_age_months": float(t), "at_risk": at_risk,
                        "cif_default": cif_d, "cif_prepay": cif_p,
                        "event_free_survival": overall_surv})
    cif = pd.DataFrame(records)
    if cif.empty:
        return cif
    km_d = kaplan_meier_curves(surv, "event_default", max_age=max_age)
    km_side = (km_d[["loan_age_months", "cumulative_event_prob"]]
               .rename(columns={"cumulative_event_prob": "naive_1_minus_km_default"})
               .astype({"loan_age_months": float})
               .sort_values("loan_age_months"))
    merged = pd.merge_asof(cif.astype({"loan_age_months": float}).sort_values("loan_age_months"),
                           km_side, on="loan_age_months", direction="backward")
    merged["km_overstatement"] = merged["naive_1_minus_km_default"] - merged["cif_default"]
    return merged


def fit_cox(surv: pd.DataFrame, event_col: str, train_mask: np.ndarray,
            covariates: list[str] | None = None) -> dict:
    covariates = covariates or SURVIVAL_COVARIATES
    cols = covariates + ["entry_age", "exit_age", event_col]
    train = surv.loc[train_mask, cols].dropna()
    test = surv.loc[~train_mask, cols].dropna()

    dropped = [c for c in covariates if float(train[c].std()) == 0.0 or train[c].nunique() < 2]
    if dropped:
        covariates = [c for c in covariates if c not in dropped]
        cols = covariates + ["entry_age", "exit_age", event_col]
        train, test = train[cols], test[cols]
        print(f"  [cox] dropped zero-variance covariate(s): {', '.join(dropped)}", flush=True)

    cph = CoxPHFitter(penalizer=0.08)
    cph.fit(train, duration_col="exit_age", event_col=event_col,
            entry_col="entry_age", robust=True)

    summary = cph.summary[["coef", "exp(coef)", "se(coef)", "p"]].reset_index()
    summary = summary.rename(columns={"covariate": "feature", "exp(coef)": "hazard_ratio"})
    summary = summary.sort_values("hazard_ratio", ascending=False).reset_index(drop=True)

    def _ci(frame):
        if frame.empty or frame[event_col].sum() < 5:
            return float("nan")
        risk = -cph.predict_partial_hazard(frame).to_numpy()
        return float(concordance_index(frame["exit_age"], risk, frame[event_col]))

    baseline_rate = float(train[event_col].mean())
    return {"model": cph, "summary": summary,
            "concordance_train": _ci(train), "concordance_test": _ci(test),
            "n_train": len(train), "n_test": len(test),
            "covariates_used": covariates, "covariates_dropped": dropped,
            "events_train": int(train[event_col].sum()),
            "events_test": int(test[event_col].sum()),
            "baseline_event_rate": baseline_rate}


def transition_matrix(df: pd.DataFrame, mask: np.ndarray | None = None,
                      states: list[str] | None = None) -> pd.DataFrame:
    states = states or ["Current", "DQ30", "DQ60", "DQ90plus", "Default", "Prepaid"]
    sub = df if mask is None else df.loc[mask]
    sub = sub[["current_status", "next_state"]].dropna()
    tab = pd.crosstab(sub["current_status"], sub["next_state"])
    tab = tab.reindex(index=states, columns=states, fill_value=0).astype(float)
    for absorbing in ("Default", "Prepaid"):
        tab.loc[absorbing] = 0.0
        tab.loc[absorbing, absorbing] = 1.0
    tab = tab + 0.25
    for absorbing in ("Default", "Prepaid"):
        tab.loc[absorbing] = 0.0
        tab.loc[absorbing, absorbing] = 1.0
    return tab.div(tab.sum(axis=1), axis=0)


def project_states(P: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    states = list(P.index)
    M = P.to_numpy()
    rows = []
    acc = np.eye(len(states))
    for k in range(1, horizon + 1):
        acc = acc @ M
        for i, s in enumerate(states):
            rows.append({"start_state": s, "horizon_month": k,
                         **{f"p_{t}": acc[i, j] for j, t in enumerate(states)}})
    return pd.DataFrame(rows)


def markov_vs_observed(df: pd.DataFrame, P: pd.DataFrame, test_mask: np.ndarray,
                       horizon: int = 12) -> pd.DataFrame:
    proj = project_states(P, horizon)
    at_h = proj[proj["horizon_month"] == horizon].set_index("start_state")
    sub = df.loc[test_mask & df["next_12m_default_flag"].notna()]
    rows = []
    for state, grp in sub.groupby("current_status", observed=True):
        if state not in at_h.index or len(grp) < 30:
            continue
        rows.append({
            "start_state": state, "n_test_rows": len(grp),
            "markov_predicted_default_12m": float(at_h.loc[state, "p_Default"]),
            "observed_default_12m": float(grp["next_12m_default_flag"].mean()),
            "markov_predicted_prepay_12m": float(at_h.loc[state, "p_Prepaid"]),
            "observed_prepay_12m": float(grp["next_12m_prepayment_flag"].mean()),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["default_abs_error"] = (out["markov_predicted_default_12m"]
                                    - out["observed_default_12m"]).abs()
        out["prepay_abs_error"] = (out["markov_predicted_prepay_12m"]
                                   - out["observed_prepay_12m"]).abs()
    return out
