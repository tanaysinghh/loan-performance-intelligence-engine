from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C

BANNED_FEATURES = {
    "prepayment_flag", "default_flag", "next_state", "loss_severity_band",
    "exception_required", "exception_type", "terminal_next", "status_next",
    *C.BINARY_TARGETS,
}

STATUS_ORDER = {"Current": 0, "DQ30": 1, "DQ60": 2, "DQ90plus": 3, "Default": 4,
                "Prepaid": -1, "PaidOff": -1}

CATEGORICAL_FEATURES = [
    "state", "loan_purpose", "occupancy_type", "property_type", "servicer_name",
    "current_status", "document_status", "source_system", "credit_score_band",
    "ltv_band", "dti_band",
]

EXCLUDED_WITH_REASON = {
    "vintage_year": "Origination-year categorical acts as a calendar-time proxy. Its levels "
                    "in the test window are unseen during training, and an ablation showed it "
                    "cost 0.004-0.008 test ROC-AUC on every target. Seasoning is already "
                    "carried by loan_age_months_clean and seasoning_bucket.",
}


def assert_no_leakage(feature_names) -> None:
    bad = sorted(set(feature_names) & BANNED_FEATURES)
    if bad:
        raise ValueError(f"Leaking features present in the design matrix: {bad}")
    forward = sorted(f for f in feature_names if f.startswith("next_"))
    if forward:
        raise ValueError(f"Forward-looking features present in the design matrix: {forward}")


def _ordinal(series: pd.Series, categories: list[str]) -> pd.Series:
    codes = pd.Categorical(series, categories=categories, ordered=True).codes.astype(float)
    return pd.Series(np.where(codes < 0, np.nan, codes), index=series.index)


def build(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["loan_id", "month_index"], kind="mergesort").reset_index(drop=True)
    g = out.groupby("loan_id", sort=False)

    out["credit_ord"] = _ordinal(out["credit_score_band"], C.CREDIT_BANDS)
    out["ltv_ord"] = _ordinal(out["ltv_band"], C.LTV_BANDS)
    out["dti_ord"] = _ordinal(out["dti_band"], C.DTI_BANDS)
    out["status_ord"] = out["current_status"].map(STATUS_ORDER).astype(float)
    out["vintage_year"] = out["origination_period"].dt.year.astype("Int64").astype(str)

    out["log_original_balance"] = np.log1p(out["original_balance"].clip(lower=0))
    bal = out["current_balance_clean"]
    out["balance_ratio"] = bal / out["original_balance"].replace(0, np.nan)
    out["log_current_balance"] = np.log1p(bal.clip(lower=0))
    out["amortisation_progress"] = 1.0 - out["balance_ratio"]
    out["term_progress"] = out["loan_age_months_clean"] / (
        out["loan_age_months_clean"] + out["remaining_term_months"]).replace(0, np.nan)

    macro_idx = macro.set_index("reporting_month")
    for col in ("market_mortgage_rate", "unemployment_rate", "hpi_yoy_growth"):
        out[col] = out["reporting_month"].map(macro_idx[col])
    out["rate_incentive"] = out["interest_rate_clean"] - out["market_mortgage_rate"]
    out["refi_incentive_positive"] = out["rate_incentive"].clip(lower=0)
    out["unemployment_delta_12m"] = out["reporting_month"].map(
        macro_idx["unemployment_rate"].diff(12))
    out["market_rate_delta_12m"] = out["reporting_month"].map(
        macro_idx["market_mortgage_rate"].diff(12))

    for lag in (1, 3, 6):
        out[f"status_ord_lag{lag}"] = g["status_ord"].shift(lag)
        out[f"dpd_lag{lag}"] = g["days_past_due_clean"].shift(lag)
    out["status_ord_delta_1m"] = out["status_ord"] - out["status_ord_lag1"]
    out["dpd_delta_1m"] = out["days_past_due_clean"] - out["dpd_lag1"]

    for w in (3, 6, 12):
        out[f"max_dpd_last_{w}m"] = g["days_past_due_clean"].transform(
            lambda s: s.rolling(w, min_periods=1).max())
        out[f"months_dq_last_{w}m"] = g["status_ord"].transform(
            lambda s: (s > 0).rolling(w, min_periods=1).sum())
    out["ever_delinquent_to_date"] = g["status_ord"].transform(
        lambda s: (s > 0).cummax()).astype(float)
    out["worst_status_to_date"] = g["status_ord"].transform("cummax")

    dq_now = (out["status_ord"] > 0).astype(int)
    out["current_streak_clean"] = dq_now.groupby(
        [out["loan_id"], dq_now.groupby(out["loan_id"], sort=False).cumsum()]).cumcount()

    bal_lag1 = g["current_balance_clean"].shift(1)
    bal_lag3 = g["current_balance_clean"].shift(3)
    out["balance_change_1m"] = (bal - bal_lag1) / bal_lag1.abs().replace(0, np.nan)
    out["balance_change_3m"] = (bal - bal_lag3) / bal_lag3.abs().replace(0, np.nan)
    out["balance_flat_1m"] = (out["balance_change_1m"].abs() < 1e-6).astype(float)

    out["scheduled_payment"] = _scheduled_payment(out)
    out["payment_to_balance"] = out["scheduled_payment"] / bal.replace(0, np.nan)

    out["missing_field_count"] = out[["credit_score_band", "ltv_band", "dti_band",
                                      "property_type", "occupancy_type"]].isna().sum(axis=1)
    for c in ("credit_score_band", "ltv_band", "dti_band", "property_type",
              "occupancy_type", "interest_rate_clean", "days_past_due_clean"):
        out[f"is_missing_{c}"] = out[c].isna().astype(int)

    out["svc_balance_rel_gap"] = out["svc_balance_rel_gap"].fillna(0.0)
    out["svc_dpd_gap"] = out["svc_dpd_gap"].fillna(0.0)
    out["reporting_lag_days"] = out["reporting_lag_days"].astype(float)

    expected_ratio = 1.0 - out["term_progress"].clip(0, 1) ** 1.6
    out["amortisation_residual"] = out["balance_ratio"] - expected_ratio
    expected_dpd = out["current_status"].map(
        {"Current": 0.0, "DQ30": 30.0, "DQ60": 60.0, "DQ90plus": 90.0, "Default": 180.0,
         "Prepaid": 0.0, "PaidOff": 0.0})
    out["dpd_status_residual"] = out["days_past_due_clean"] - expected_dpd
    out["balance_growth_excess"] = out["balance_change_1m"].clip(lower=0)
    out["doc_incomplete"] = out["document_status"].isin(["missing", "exception"]).astype(float)
    out["manual_upload"] = (out["source_system"] == "manual_upload").astype(float)

    out["seasoning_bucket"] = pd.cut(out["loan_age_months_clean"],
                                     [-1, 6, 12, 24, 36, 60, 120, 1000],
                                     labels=False).astype(float)
    out["calendar_month"] = out["reporting_period"].dt.month.astype(float)

    return out


def _scheduled_payment(df: pd.DataFrame) -> pd.Series:
    r = df["interest_rate_clean"] / 1200.0
    n = (df["loan_age_months_clean"] + df["remaining_term_months"]).clip(lower=1)
    factor = (1 + r) ** n
    pmt = df["original_balance"] * r * factor / (factor - 1)
    return pmt.replace([np.inf, -np.inf], np.nan)


NUMERIC_FEATURES = [
    "credit_ord", "ltv_ord", "dti_ord", "status_ord", "log_original_balance",
    "log_current_balance", "balance_ratio", "amortisation_progress", "term_progress",
    "loan_age_months_clean", "remaining_term_months", "interest_rate_clean",
    "days_past_due_clean", "modification_flag", "market_mortgage_rate", "unemployment_rate",
    "hpi_yoy_growth", "rate_incentive", "refi_incentive_positive", "unemployment_delta_12m",
    "market_rate_delta_12m", "status_ord_lag1", "status_ord_lag3", "status_ord_lag6",
    "dpd_lag1", "dpd_lag3", "dpd_lag6", "status_ord_delta_1m", "dpd_delta_1m",
    "max_dpd_last_3m", "max_dpd_last_6m", "max_dpd_last_12m", "months_dq_last_3m",
    "months_dq_last_6m", "months_dq_last_12m", "ever_delinquent_to_date",
    "worst_status_to_date", "current_streak_clean", "balance_change_1m", "balance_change_3m",
    "balance_flat_1m", "scheduled_payment", "payment_to_balance", "missing_field_count",
    "is_missing_credit_score_band", "is_missing_ltv_band", "is_missing_dti_band",
    "is_missing_property_type", "is_missing_occupancy_type",
    "is_missing_interest_rate_clean", "is_missing_days_past_due_clean",
    "dq_score", "dq_violation_count", "dpd_repaired", "rate_repaired", "balance_repaired",
    "age_repaired", "stale_reporting", "reporting_lag_days", "svc_present",
    "svc_balance_rel_gap", "svc_status_conflict", "svc_dpd_gap",
    "seasoning_bucket", "calendar_month", "amortisation_residual", "dpd_status_residual",
    "balance_growth_excess", "doc_incomplete", "manual_upload",
]

BASELINE_FEATURES = [
    "credit_ord", "ltv_ord", "dti_ord", "status_ord", "loan_age_months_clean",
    "balance_ratio", "interest_rate_clean", "days_past_due_clean", "modification_flag",
]


def feature_columns(df: pd.DataFrame, include_categoricals: bool = True) -> list[str]:
    cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    if include_categoricals:
        cols = cols + [c for c in CATEGORICAL_FEATURES if c in df.columns]
    assert_no_leakage(cols)
    return cols


def design_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    assert_no_leakage(cols)
    X = df[cols].copy()
    for c in CATEGORICAL_FEATURES:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X
