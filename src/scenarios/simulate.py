from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.data import loaders
from src.models import survival as S

MACRO_FEATURES = ["market_mortgage_rate", "unemployment_rate", "hpi_yoy_growth",
                  "rate_incentive", "refi_incentive_positive", "unemployment_delta_12m",
                  "market_rate_delta_12m"]

IDENTIFICATION_NOTE = (
    "Engine A's **credit** channel is not identified from this data, and the evidence is in "
    "its own output rather than in an argument about it.\n\n"
    "Macro levels are constant across every loan within a reporting month. With one realised "
    "macro path and 90 monthly observations there is no cross-sectional variation in "
    "unemployment or HPI growth at all, so a loan-level model cannot separate the effect of "
    "unemployment from the effect of calendar time — they are collinear by construction. What "
    "the trees learn is a time proxy, and in this panel's history the low-rate period "
    "coincided with the pandemic unemployment spike.\n\n"
    "The consequence shows up as two specific wrong answers, both visible in the tables "
    "below. The adverse-credit shock moves Engine A's projected 12-month default rate by "
    "essentially nothing — a delta indistinguishable from zero, and of a sign that carries no "
    "information — which is not a credible stress result for a scenario that raises "
    "unemployment by three percentage points and turns house prices negative. And the "
    "high-prepayment scenario nudges projected default *upward*, which has the sign "
    "backwards. Both are reported in section 8 as computed rather than being corrected.\n\n"
    "`rate_incentive` is the exception. It is a loan's own note rate minus the prevailing "
    "market rate, so it does vary across loans within a month and its effect is identified "
    "cross-sectionally. That is why Engine A's prepayment response is trustworthy and its "
    "credit response is not, and why Engine B exists."
)

SCENARIO_TARGETS = ["next_6m_delinquency_flag", "next_12m_default_flag",
                    "next_12m_prepayment_flag"]


def latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values(["loan_id", "month_index"], kind="mergesort")
    return d.groupby("loan_id", sort=False).tail(1).reset_index(drop=True)


def apply_scenario(snapshot: pd.DataFrame, scenario: pd.DataFrame,
                   horizon_month: int = 12, channel: str = "all") -> pd.DataFrame:
    row = scenario[scenario["horizon_month"] == horizon_month].iloc[0]
    out = snapshot.copy()
    base_rate = snapshot["market_mortgage_rate"].iloc[0]
    base_unemp = snapshot["unemployment_rate"].iloc[0]

    out["rate_incentive"] = out["interest_rate_clean"] - row["market_mortgage_rate"]
    out["refi_incentive_positive"] = out["rate_incentive"].clip(lower=0)
    if channel == "all":
        out["market_mortgage_rate"] = row["market_mortgage_rate"]
        out["unemployment_rate"] = row["unemployment_rate"]
        out["hpi_yoy_growth"] = row["hpi_yoy_growth"]
        out["market_rate_delta_12m"] = row["market_mortgage_rate"] - base_rate
        out["unemployment_delta_12m"] = row["unemployment_rate"] - base_unemp
    return out


def score_scenarios(snapshot: pd.DataFrame, scenarios: pd.DataFrame, models: dict,
                    horizon_month: int = 12, channel: str = "all") -> pd.DataFrame:
    rows = []
    for name, grp in scenarios.groupby("scenario_name"):
        frame = apply_scenario(snapshot, grp, horizon_month, channel)
        rec = {"scenario_name": name, "loans": len(frame)}
        for target in SCENARIO_TARGETS:
            if target not in models:
                continue
            rec[f"projected_{target}"] = float(models[target].predict_proba(frame).mean())
        rows.append(rec)
    out = pd.DataFrame(rows)
    base = out[out["scenario_name"] == "base"].iloc[0]
    for target in SCENARIO_TARGETS:
        col = f"projected_{target}"
        if col in out.columns:
            out[f"delta_{target}"] = out[col] - base[col]
            out[f"relative_{target}"] = out[col] / base[col] - 1.0
    return out


def segment_impacts(snapshot: pd.DataFrame, scenarios: pd.DataFrame, models: dict,
                    target: str, by: str, horizon_month: int = 12,
                    channel: str = "all") -> pd.DataFrame:
    frames = {}
    for name, grp in scenarios.groupby("scenario_name"):
        frame = apply_scenario(snapshot, grp, horizon_month, channel)
        frames[name] = models[target].predict_proba(frame)
    out = pd.DataFrame({by: snapshot[by].to_numpy(),
                        **{n: p for n, p in frames.items()}})
    agg = out.groupby(by, observed=True).agg(["mean", "size"])
    result = pd.DataFrame({by: agg.index})
    result["loans"] = agg[("base", "size")].to_numpy()
    for name in frames:
        result[f"{name}"] = agg[(name, "mean")].to_numpy()
    for name in frames:
        if name != "base":
            result[f"delta_{name}"] = result[name] - result["base"]
    return result.sort_values(f"delta_{[n for n in frames if n != 'base'][0]}",
                              ascending=False).reset_index(drop=True)


def driver_decomposition(snapshot: pd.DataFrame, scenarios: pd.DataFrame, models: dict,
                         target: str, horizon_month: int = 12) -> pd.DataFrame:
    base_grp = scenarios[scenarios["scenario_name"] == "base"]
    base_frame = apply_scenario(snapshot, base_grp, horizon_month, channel="all")
    base_p = float(models[target].predict_proba(base_frame).mean())

    rows = []
    for name, grp in scenarios.groupby("scenario_name"):
        if name == "base":
            continue
        full_frame = apply_scenario(snapshot, grp, horizon_month, channel="all")
        full_p = float(models[target].predict_proba(full_frame).mean())
        total = full_p - base_p

        contribs = {}
        for feat in MACRO_FEATURES:
            partial = base_frame.copy()
            partial[feat] = full_frame[feat]
            if feat == "market_mortgage_rate":
                partial["rate_incentive"] = base_frame["rate_incentive"]
            contribs[feat] = float(models[target].predict_proba(partial).mean()) - base_p

        explained = sum(contribs.values())
        for feat, val in contribs.items():
            rows.append({"scenario_name": name, "macro_input": feat,
                         "isolated_contribution": val,
                         "share_of_total": val / total if abs(total) > 1e-12 else np.nan})
        rows.append({"scenario_name": name, "macro_input": "interaction_residual",
                     "isolated_contribution": total - explained,
                     "share_of_total": ((total - explained) / total
                                        if abs(total) > 1e-12 else np.nan)})
    out = pd.DataFrame(rows)
    return out.sort_values(["scenario_name", "isolated_contribution"],
                           ascending=[True, False]).reset_index(drop=True)


def fit_macro_transition_model(df: pd.DataFrame, macro: pd.DataFrame,
                               train_mask: np.ndarray) -> dict:
    from sklearn.linear_model import LinearRegression

    sub = df.loc[train_mask, ["reporting_month", "current_status", "next_state",
                              "rate_incentive"]].dropna(subset=["current_status", "next_state"])
    worse = {"Current": ["DQ30", "DQ60", "DQ90plus", "Default"],
             "DQ30": ["DQ60", "DQ90plus", "Default"],
             "DQ60": ["DQ90plus", "Default"],
             "DQ90plus": ["Default"]}

    macro_idx = macro.set_index("reporting_month")
    fits = {}
    for state, targets in worse.items():
        grp = sub[sub["current_status"] == state]
        rates = grp.groupby("reporting_month").apply(
            lambda g: pd.Series({"deteriorate": g["next_state"].isin(targets).mean(),
                                 "prepay": (g["next_state"] == "Prepaid").mean(),
                                 "n": len(g),
                                 "rate_incentive": g["rate_incentive"].mean()}),
            include_groups=False)
        rates = rates[rates["n"] >= 25]
        if len(rates) < 12:
            continue
        X = pd.DataFrame({
            "unemployment_rate": macro_idx["unemployment_rate"].reindex(rates.index),
            "hpi_yoy_growth": macro_idx["hpi_yoy_growth"].reindex(rates.index),
            "rate_incentive": rates["rate_incentive"],
        }).dropna()
        rates = rates.loc[X.index]
        fits[state] = {}
        for kind in ("deteriorate", "prepay"):
            p = rates[kind].clip(1e-4, 1 - 1e-4)
            y = np.log(p / (1 - p))
            reg = LinearRegression().fit(X, y, sample_weight=rates["n"])
            fits[state][kind] = {
                "model": reg, "columns": list(X.columns),
                "r2": float(reg.score(X, y, sample_weight=rates["n"])),
                "coefficients": dict(zip(X.columns, np.round(reg.coef_, 4))),
                "months": int(len(X)),
                "historical_mean_rate": float(rates[kind].mean()),
            }
    return fits


def stressed_transition_matrix(base_P: pd.DataFrame, fits: dict, macro_row: pd.Series,
                               mean_rate_incentive: float,
                               baseline_macro: pd.Series) -> pd.DataFrame:
    P = base_P.copy()
    worse = {"Current": ["DQ30", "DQ60", "DQ90plus", "Default"],
             "DQ30": ["DQ60", "DQ90plus", "Default"],
             "DQ60": ["DQ90plus", "Default"],
             "DQ90plus": ["Default"]}

    for state, fit in fits.items():
        if state not in P.index:
            continue
        for kind in ("deteriorate", "prepay"):
            f = fit[kind]
            x_scn = pd.DataFrame([{
                "unemployment_rate": macro_row["unemployment_rate"],
                "hpi_yoy_growth": macro_row["hpi_yoy_growth"],
                "rate_incentive": mean_rate_incentive,
            }])[f["columns"]]
            x_base = pd.DataFrame([{
                "unemployment_rate": baseline_macro["unemployment_rate"],
                "hpi_yoy_growth": baseline_macro["hpi_yoy_growth"],
                "rate_incentive": mean_rate_incentive,
            }])[f["columns"]]
            shift = float(f["model"].predict(x_scn)[0] - f["model"].predict(x_base)[0])

            cols = worse[state] if kind == "deteriorate" else ["Prepaid"]
            cols = [c for c in cols if c in P.columns]
            cur = P.loc[state, cols].sum()
            if cur <= 0 or cur >= 1:
                continue
            new = 1 / (1 + np.exp(-(np.log(cur / (1 - cur)) + shift)))
            new = float(np.clip(new, 1e-5, 0.95))
            P.loc[state, cols] = P.loc[state, cols] * (new / cur)

        stay = [c for c in P.columns if c not in
                (worse[state] + ["Prepaid"]) and c in P.columns]
        residual = 1.0 - P.loc[state, [c for c in P.columns if c not in stay]].sum()
        if stay and residual > 0:
            cur_stay = P.loc[state, stay].sum()
            if cur_stay > 0:
                P.loc[state, stay] = P.loc[state, stay] * (residual / cur_stay)
    return P.clip(lower=0).div(P.clip(lower=0).sum(axis=1), axis=0)


def portfolio_projection(df: pd.DataFrame, P: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    snap = latest_snapshot(df)
    states = list(P.index)
    start = (snap["current_status"].value_counts(normalize=True)
             .reindex(states, fill_value=0.0).to_numpy())
    M = P.to_numpy()
    v = start.copy()
    rows = [{"horizon_month": 0, **{s: start[i] for i, s in enumerate(states)}}]
    for k in range(1, horizon + 1):
        v = v @ M
        rows.append({"horizon_month": k, **{s: v[i] for i, s in enumerate(states)}})
    return pd.DataFrame(rows)
