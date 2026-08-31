from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C

DPD_SENTINELS = {9999.0, -1.0, 999.0}
RATE_SENTINELS = {0.0, 99.99, -1.0}


_PANEL_STRING_COLS = {"loss_severity_band": "object", "credit_score_band": "object",
                      "ltv_band": "object", "dti_band": "object",
                      "document_status": "object", "source_system": "object"}


def load_panel(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or C.LOAN_PANEL, dtype=_PANEL_STRING_COLS)
    df["reporting_period"] = pd.PeriodIndex(df["reporting_month"], freq="M")
    df["origination_period"] = pd.to_datetime(df["origination_month"], format="%Y-%m",
                                              errors="coerce").dt.to_period("M")
    df["last_updated_at"] = pd.to_datetime(df["last_updated_at"], errors="coerce")
    df["period_end"] = df["reporting_period"].dt.to_timestamp(how="end").dt.normalize()
    for col in ("days_past_due", "interest_rate", "current_balance", "original_balance"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_servicer_updates(path=None) -> pd.DataFrame:
    u = pd.read_csv(path or C.SERVICER_UPDATES)
    u["update_received_at"] = pd.to_datetime(u["update_received_at"], errors="coerce")
    return u


def load_macro(path=None) -> pd.DataFrame:
    return pd.read_csv(path or C.MACRO_HISTORY)


def load_scenarios(path=None) -> pd.DataFrame:
    return pd.read_csv(path or C.MACRO_SCENARIOS)


def load_dictionary(path=None) -> pd.DataFrame:
    return pd.read_csv(path or C.DATA_DICTIONARY)


def reconcile(df: pd.DataFrame, updates: pd.DataFrame) -> pd.DataFrame:
    u = updates.sort_values("update_received_at", kind="mergesort")
    dup_count = int(u.duplicated(subset=["loan_id", "reporting_month"]).sum())
    u = u.drop_duplicates(subset=["loan_id", "reporting_month"], keep="last")

    panel_keys = set(zip(df["loan_id"], df["reporting_month"]))
    orphan_mask = ~pd.Series(list(zip(u["loan_id"], u["reporting_month"])),
                             index=u.index).isin(panel_keys)
    orphan_count = int(orphan_mask.sum())

    u = u.loc[~orphan_mask, ["loan_id", "reporting_month", "reported_balance",
                             "reported_status", "reported_dpd", "update_received_at"]]
    u = u.rename(columns={"reported_balance": "svc_balance", "reported_status": "svc_status",
                          "reported_dpd": "svc_dpd", "update_received_at": "svc_received_at"})
    out = df.merge(u, on=["loan_id", "reporting_month"], how="left")

    out["svc_present"] = out["svc_balance"].notna().astype(int)
    out["svc_balance_abs_gap"] = (out["svc_balance"] - out["current_balance"]).abs()
    denom = out["current_balance"].abs().replace(0, np.nan)
    out["svc_balance_rel_gap"] = out["svc_balance_abs_gap"] / denom
    out["svc_status_conflict"] = (out["svc_present"].eq(1)
                                  & out["svc_status"].ne(out["current_status"])).astype(int)
    out["svc_dpd_gap"] = (out["svc_dpd"] - out["days_past_due"]).abs()
    out.attrs["feed_duplicate_records"] = dup_count
    out.attrs["feed_orphan_records"] = orphan_count
    return out


def clean_panel(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    dpd = out["days_past_due"]
    bad_dpd = dpd.isin(DPD_SENTINELS) | (dpd < 0) | (dpd > 400)
    out["dpd_repaired"] = bad_dpd.astype(int)
    out["days_past_due_clean"] = dpd.mask(bad_dpd)

    rate = out["interest_rate"]
    bad_rate = rate.isin(RATE_SENTINELS) | (rate <= 0.5) | (rate > 25)
    out["rate_repaired"] = bad_rate.astype(int)
    out["interest_rate_clean"] = rate.mask(bad_rate)

    bal = out["current_balance"]
    bad_bal = (bal < 0) | (bal > out["original_balance"] * 3)
    out["balance_repaired"] = bad_bal.astype(int)
    out["current_balance_clean"] = bal.mask(bad_bal)

    implied_age = (out["reporting_period"] - out["origination_period"]).map(
        lambda x: x.n if pd.notna(x) else np.nan).astype(float)
    out["implied_loan_age_months"] = implied_age
    bad_age = (out["loan_age_months"] - implied_age).abs() > 2
    out["age_repaired"] = bad_age.fillna(False).astype(int)
    out["loan_age_months_clean"] = out["loan_age_months"].mask(bad_age.fillna(False), implied_age)
    out.loc[out["loan_age_months_clean"] < 0, "loan_age_months_clean"] = np.nan

    out["reporting_lag_days"] = (out["last_updated_at"] - out["period_end"]).dt.days
    out["stale_reporting"] = (out["reporting_lag_days"] > 75).astype(int)

    out = out.drop_duplicates(subset=["loan_id", "reporting_month"], keep="first")
    return out.reset_index(drop=True)
