"""Derives the operational exception label from observable record conditions.

Exceptions are what a servicing-oversight reviewer would actually raise. A breach alone is
not an exception: it must be material. Materiality thresholds plus a small reviewer-noise
term keep the label learnable but not trivially separable, which is the realistic case.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C

PRIORITY = [
    "invalid_date_relationship",
    "balance_reconciliation_break",
    "missing_documentation",
    "status_dpd_mismatch",
    "unexpected_balance_movement",
    "stale_servicer_reporting",
]


def build_exception_labels(df: pd.DataFrame, recon: pd.DataFrame,
                           rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    rep = pd.PeriodIndex(df["reporting_month"], freq="M")
    orig = pd.PeriodIndex(df["origination_month"], freq="M")
    period_end = rep.to_timestamp(how="end").normalize()
    last_upd = pd.to_datetime(df["last_updated_at"], errors="coerce")

    implied_age = np.asarray((rep - orig).map(lambda x: x.n), dtype=float)
    age_gap = np.abs(df["loan_age_months"].to_numpy(dtype=float) - implied_age)

    breach_date = (np.asarray(orig > rep) | np.asarray(last_upd < period_end)
                   | (age_gap > 2))

    bal = df["current_balance"].to_numpy(dtype=float)
    rec_bal = df["recon_reported_balance"].to_numpy(dtype=float)
    abs_gap = np.abs(rec_bal - bal)
    rel_gap = np.where(np.abs(bal) > 1, abs_gap / np.abs(bal), np.nan)
    breach_recon = (rel_gap > 0.01) & (abs_gap > 500)

    breach_doc = df["document_status"].isin(["missing", "exception"]).to_numpy()

    dpd = df["days_past_due"].to_numpy(dtype=float)
    expected_dpd = df["current_status"].map(
        {"Current": 0, "DQ30": 30, "DQ60": 60, "DQ90plus": 90, "Default": 180}).to_numpy(dtype=float)
    dpd_gap = np.abs(np.nan_to_num(dpd, nan=expected_dpd) - expected_dpd)
    breach_dpd = (dpd_gap > 29) | (dpd > 900) | (dpd < 0)

    prev_bal = df.groupby("loan_id", sort=False)["current_balance"].shift(1).to_numpy(dtype=float)
    growth = np.where(np.abs(prev_bal) > 1, (bal - prev_bal) / np.abs(prev_bal), 0.0)
    breach_move = (growth > 0.005) | (bal > df["original_balance"].to_numpy() * 1.02) | (bal < 0)

    staleness_days = (last_upd - period_end).dt.days.to_numpy(dtype=float)
    breach_stale = staleness_days > 75

    breaches = {
        "invalid_date_relationship": breach_date,
        "balance_reconciliation_break": np.nan_to_num(breach_recon, nan=0).astype(bool),
        "missing_documentation": breach_doc,
        "status_dpd_mismatch": np.nan_to_num(breach_dpd, nan=0).astype(bool),
        "unexpected_balance_movement": np.nan_to_num(breach_move, nan=0).astype(bool),
        "stale_servicer_reporting": np.nan_to_num(breach_stale, nan=0).astype(bool),
    }
    for k, v in breaches.items():
        df[f"breach_{k}"] = v.astype(int)

    escalation_p = {
        "invalid_date_relationship": 0.93,
        "balance_reconciliation_break": 0.88,
        "missing_documentation": 0.72,
        "status_dpd_mismatch": 0.80,
        "unexpected_balance_movement": 0.68,
        "stale_servicer_reporting": 0.55,
    }
    n = len(df)
    exc_type = np.full(n, "none", dtype=object)
    for name in reversed(PRIORITY):
        hit = breaches[name] & (rng.random(n) < escalation_p[name])
        exc_type = np.where(hit, name, exc_type)

    flip = rng.random(n) < 0.012
    exc_type = np.where(flip & (exc_type != "none"), "none", exc_type)
    false_raise = (rng.random(n) < 0.004) & (exc_type == "none")
    exc_type = np.where(false_raise, rng.choice(PRIORITY, n), exc_type)

    df["exception_type"] = exc_type
    df["exception_required"] = (df["exception_type"] != "none").astype(int)
    return df


def reconcile_servicer_feed(panel: pd.DataFrame, updates: pd.DataFrame) -> pd.DataFrame:
    """Latest-wins reconciliation of the secondary servicer feed onto the panel."""
    u = updates.copy()
    u["update_received_at"] = pd.to_datetime(u["update_received_at"], errors="coerce")
    u = u.sort_values("update_received_at", kind="mergesort")
    u = u.drop_duplicates(subset=["loan_id", "reporting_month"], keep="last")
    u = u.rename(columns={
        "reported_balance": "recon_reported_balance",
        "reported_status": "recon_reported_status",
        "reported_dpd": "recon_reported_dpd",
        "update_received_at": "recon_received_at",
    })[["loan_id", "reporting_month", "recon_reported_balance", "recon_reported_status",
        "recon_reported_dpd", "recon_received_at"]]
    out = panel.merge(u, on=["loan_id", "reporting_month"], how="left")
    out["recon_record_present"] = out["recon_reported_balance"].notna().astype(int)
    return out
