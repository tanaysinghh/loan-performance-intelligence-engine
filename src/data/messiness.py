from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.data.generate_synthetic import SERVICER_OPS_NOISE

DEFECT_RATES = {
    "missing_dti_band": 0.062,
    "missing_ltv_band": 0.031,
    "missing_property_type": 0.041,
    "missing_credit_score_band": 0.019,
    "missing_interest_rate": 0.014,
    "missing_days_past_due": 0.026,
    "missing_occupancy_type": 0.022,
    "outlier_balance_inflated": 0.0030,
    "outlier_balance_negative": 0.0012,
    "outlier_interest_rate": 0.0018,
    "sentinel_days_past_due": 0.0028,
    "invalid_origination_after_reporting": 0.0040,
    "invalid_last_updated_before_period": 0.0090,
    "inconsistent_loan_age": 0.0055,
    "status_dpd_mismatch": 0.0075,
    "duplicate_rows": 0.0040,
}


def _mask(rng, n, rate, weights=None):
    if weights is None:
        return rng.random(n) < rate
    w = weights / weights.mean()
    return rng.random(n) < np.clip(rate * w, 0, 0.95)


def inject(panel: pd.DataFrame, loans: pd.DataFrame, rng: np.random.Generator):
    df = panel.merge(
        loans[["loan_id", "origination_month", "original_balance", "interest_rate",
               "credit_score_band", "ltv_band", "dti_band", "state", "loan_purpose",
               "occupancy_type", "property_type", "servicer_name"]],
        on="loan_id", how="left")

    n = len(df)
    ops_noise = df["servicer_name"].map(SERVICER_OPS_NOISE).to_numpy()
    late_period = (df["month_index"].to_numpy() >= df["month_index"].max() - 8).astype(float)
    defect_log = []

    def log(name, m):
        defect_log.append({"defect": name, "rows_affected": int(m.sum()),
                           "rate": float(m.mean())})

    for col, key in (("dti_band", "missing_dti_band"),
                     ("ltv_band", "missing_ltv_band"),
                     ("property_type", "missing_property_type"),
                     ("credit_score_band", "missing_credit_score_band"),
                     ("occupancy_type", "missing_occupancy_type")):
        m = _mask(rng, n, DEFECT_RATES[key], ops_noise)
        df.loc[m, col] = np.nan
        log(key, m)

    m = _mask(rng, n, DEFECT_RATES["missing_interest_rate"], ops_noise)
    df.loc[m, "interest_rate"] = np.nan
    log("missing_interest_rate", m)

    m = _mask(rng, n, DEFECT_RATES["missing_days_past_due"], ops_noise * (1 + late_period))
    df["days_past_due"] = df["days_past_due"].astype(float)
    df.loc[m, "days_past_due"] = np.nan
    log("missing_days_past_due", m)

    m = _mask(rng, n, DEFECT_RATES["outlier_balance_inflated"], ops_noise)
    df.loc[m, "current_balance"] = df.loc[m, "current_balance"] * rng.uniform(9, 130, int(m.sum()))
    log("outlier_balance_inflated", m)

    m = _mask(rng, n, DEFECT_RATES["outlier_balance_negative"])
    df.loc[m, "current_balance"] = -df.loc[m, "current_balance"].abs()
    log("outlier_balance_negative", m)

    m = _mask(rng, n, DEFECT_RATES["outlier_interest_rate"])
    df.loc[m, "interest_rate"] = rng.choice([0.0, 99.99, -1.0], int(m.sum()))
    log("outlier_interest_rate", m)

    m = _mask(rng, n, DEFECT_RATES["sentinel_days_past_due"])
    df.loc[m, "days_past_due"] = rng.choice([9999.0, -1.0], int(m.sum()))
    log("sentinel_days_past_due", m)

    m = _mask(rng, n, DEFECT_RATES["status_dpd_mismatch"], ops_noise)
    df.loc[m, "days_past_due"] = rng.choice([0.0, 15.0, 120.0], int(m.sum()))
    log("status_dpd_mismatch", m)

    rep = pd.PeriodIndex(df["reporting_month"], freq="M")
    orig = pd.PeriodIndex(df["origination_month"], freq="M")
    m = _mask(rng, n, DEFECT_RATES["invalid_origination_after_reporting"])
    shifted = orig.astype(str).to_numpy().copy()
    bad_idx = np.where(m)[0]
    shifted[bad_idx] = (rep[bad_idx] + rng.integers(1, 14, len(bad_idx))).astype(str)
    df["origination_month"] = shifted
    log("invalid_origination_after_reporting", m)

    m = _mask(rng, n, DEFECT_RATES["inconsistent_loan_age"], ops_noise)
    df.loc[m, "loan_age_months"] = (df.loc[m, "loan_age_months"]
                                    + rng.integers(-30, 30, int(m.sum()))).clip(lower=-5)
    log("inconsistent_loan_age", m)

    period_end = rep.to_timestamp(how="end").normalize()
    lag_days = np.clip(rng.gamma(2.2, 4.0, n), 0, 190) + 1
    stale_boost = _mask(rng, n, 0.035, ops_noise)
    lag_days = lag_days + np.where(stale_boost, rng.uniform(40, 160, n), 0.0)
    last_updated = period_end + pd.to_timedelta(lag_days.round(), unit="D")
    m = _mask(rng, n, DEFECT_RATES["invalid_last_updated_before_period"], ops_noise)
    last_updated = last_updated.to_series().reset_index(drop=True)
    early = period_end.to_series().reset_index(drop=True) - pd.to_timedelta(
        rng.integers(5, 70, n), unit="D")
    last_updated[m] = early[m]
    df["last_updated_at"] = last_updated.dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()
    log("invalid_last_updated_before_period", m)

    df["source_system"] = rng.choice(C.SOURCE_SYSTEMS, n, p=[0.72, 0.20, 0.08])
    doc_logit = (-2.6 + 4.5 * ops_noise
                 + 0.55 * df["current_status"].isin(list(C.DELINQUENT_STATES)).to_numpy()
                 + 0.40 * (df["source_system"].to_numpy() == "manual_upload"))
    doc_p = 1.0 / (1.0 + np.exp(-doc_logit))
    draw = rng.random(n)
    df["document_status"] = np.where(draw < doc_p * 0.45, "missing",
                             np.where(draw < doc_p * 0.80, "pending",
                             np.where(draw < doc_p, "exception", "complete")))

    dup_m = _mask(rng, n, DEFECT_RATES["duplicate_rows"])
    dups = df.loc[dup_m].copy()
    log("duplicate_rows", dup_m)
    df = pd.concat([df, dups], ignore_index=True)

    return df, pd.DataFrame(defect_log)


def build_servicer_updates(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    take = rng.random(len(df)) < 0.34
    s = df.loc[take, ["loan_id", "reporting_month", "servicer_name", "current_balance",
                      "current_status", "days_past_due"]].copy()
    s = s.rename(columns={"current_balance": "reported_balance",
                          "current_status": "reported_status",
                          "days_past_due": "reported_dpd"})
    m = len(s)

    bal_conflict = rng.random(m) < 0.085
    s.loc[bal_conflict, "reported_balance"] = (
        s.loc[bal_conflict, "reported_balance"] * rng.uniform(0.90, 1.12, int(bal_conflict.sum())))

    status_conflict = rng.random(m) < 0.052
    s.loc[status_conflict, "reported_status"] = rng.choice(
        ["Current", "DQ30", "DQ60", "DQ90plus"], int(status_conflict.sum()))

    dpd_conflict = rng.random(m) < 0.04
    s.loc[dpd_conflict, "reported_dpd"] = rng.integers(0, 200, int(dpd_conflict.sum())).astype(float)

    miss = rng.random(m) < 0.05
    s.loc[miss, "reported_balance"] = np.nan

    rep_end = pd.PeriodIndex(s["reporting_month"], freq="M").to_timestamp(how="end").normalize()
    s["update_received_at"] = (rep_end + pd.to_timedelta(
        np.clip(rng.gamma(2.0, 5.0, m), 0, 120).round(), unit="D")).strftime("%Y-%m-%d %H:%M:%S")
    s["file_batch_id"] = [f"BATCH-{h}" for h in
                          pd.PeriodIndex(s["reporting_month"], freq="M").astype(str)]

    dup = s.sample(frac=0.06, random_state=7).copy()
    dup["reported_balance"] = dup["reported_balance"] * rng.uniform(0.97, 1.03, len(dup))
    dup["update_received_at"] = (pd.to_datetime(dup["update_received_at"])
                                 + pd.to_timedelta(rng.integers(1, 20, len(dup)), unit="D")
                                 ).dt.strftime("%Y-%m-%d %H:%M:%S")
    dup["file_batch_id"] = dup["file_batch_id"] + "-R1"

    orphans = s.sample(n=max(40, int(0.004 * m)), random_state=11).copy()
    known = set(df["loan_id"].astype(str))
    template = str(s["loan_id"].iloc[0]) if len(s) else "LN100000"
    head = template.rstrip("0123456789")
    width = len(template) - len(head)
    made, n = [], 0
    while len(made) < len(orphans):
        candidate = f"{head}{(9_000_000 + n) % (10 ** width):0{width}d}"
        if candidate not in known:
            made.append(candidate)
        n += 1
    orphans["loan_id"] = made
    orphans["file_batch_id"] = orphans["file_batch_id"] + "-ORPHAN"

    out = pd.concat([s, dup, orphans], ignore_index=True)
    return out.sample(frac=1.0, random_state=3).reset_index(drop=True)
