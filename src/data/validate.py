"""Named cross-column validation rules and data-quality scoring.

Each rule is a pure function returning a boolean Series marking violating rows. Rules carry a
severity weight; record-level scores are 100 minus the weighted sum of violations, floored at
zero. Batch scores aggregate the same violations by reporting month and servicer, which is
the grain an oversight team actually acts on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src import config as C


@dataclass(frozen=True)
class Rule:
    name: str
    dimension: str
    severity: float
    description: str
    fn: Callable[[pd.DataFrame], pd.Series]


def _s(df, expr):
    return expr.reindex(df.index).fillna(False).astype(bool)


RULES: list[Rule] = [
    Rule("origination_after_reporting", "validity", 14.0,
         "Origination month is later than the reporting month.",
         lambda d: _s(d, d["origination_period"] > d["reporting_period"])),
    Rule("last_updated_before_period_end", "validity", 8.0,
         "Servicing record was last written before the reporting period closed.",
         lambda d: _s(d, d["last_updated_at"] < d["period_end"])),
    Rule("loan_age_inconsistent_with_dates", "consistency", 9.0,
         "Reported loan age disagrees with reporting minus origination month by >2 months.",
         lambda d: _s(d, (d["loan_age_months"] - d["implied_loan_age_months"]).abs() > 2)),
    Rule("negative_balance", "validity", 16.0,
         "Current balance is negative.",
         lambda d: _s(d, d["current_balance"] < 0)),
    Rule("balance_exceeds_original", "plausibility", 12.0,
         "Current balance exceeds original balance by more than 2%.",
         lambda d: _s(d, d["current_balance"] > d["original_balance"] * 1.02)),
    Rule("balance_increase_month_over_month", "consistency", 7.0,
         "Unpaid principal balance rose month over month on a non-modified loan.",
         lambda d: _s(d, (d["balance_mom_growth"] > 0.005) & d["modification_flag"].eq(0))),
    Rule("dpd_sentinel_value", "validity", 10.0,
         "Days past due carries a sentinel value (9999, -1).",
         lambda d: _s(d, d["days_past_due"].isin([9999, -1, 999]))),
    Rule("status_dpd_mismatch", "consistency", 11.0,
         "Days past due is inconsistent with the reported performance status.",
         lambda d: _s(d, (d["days_past_due_clean"] - d["expected_dpd"]).abs() > 29)),
    Rule("interest_rate_out_of_range", "validity", 10.0,
         "Note rate outside a plausible 0.5%-25% range.",
         lambda d: _s(d, (d["interest_rate"] <= 0.5) | (d["interest_rate"] > 25))),
    Rule("remaining_term_inconsistent", "consistency", 6.0,
         "Remaining term plus loan age is not a standard contractual term.",
         lambda d: _s(d, ~(d["loan_age_months"] + d["remaining_term_months"]
                           ).isin([180, 240, 360]))),
    Rule("missing_critical_field", "completeness", 9.0,
         "A field required for credit assessment is missing.",
         lambda d: _s(d, d[["credit_score_band", "ltv_band", "dti_band"]].isna().any(axis=1))),
    Rule("document_file_incomplete", "completeness", 7.0,
         "Document custody status is missing or in exception.",
         lambda d: _s(d, d["document_status"].isin(["missing", "exception"]))),
    Rule("stale_servicer_reporting", "timeliness", 6.0,
         "Record last updated more than 75 days after the period closed.",
         lambda d: _s(d, d["reporting_lag_days"] > 75)),
    Rule("servicer_balance_break", "reconciliation", 13.0,
         "Servicer feed balance differs from the panel by >1% and >$500.",
         lambda d: _s(d, (d["svc_balance_rel_gap"] > 0.01) & (d["svc_balance_abs_gap"] > 500))),
    Rule("servicer_status_conflict", "reconciliation", 11.0,
         "Servicer feed reports a different performance status than the panel.",
         lambda d: _s(d, d["svc_status_conflict"].eq(1))),
    Rule("servicer_record_absent", "reconciliation", 3.0,
         "No servicer feed record exists for this loan month.",
         lambda d: _s(d, d["svc_present"].eq(0))),
    Rule("terminal_status_with_balance", "consistency", 12.0,
         "Loan is in a terminal status but still carries a material balance.",
         lambda d: _s(d, d["current_status"].isin(list(C.TERMINAL_STATES))
                     & (d["current_balance"] > 1000))),
]


def prepare_for_rules(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["loan_id", "month_index"], kind="mergesort")
    prev = out.groupby("loan_id", sort=False)["current_balance"].shift(1)
    out["balance_mom_growth"] = np.where(prev.abs() > 1,
                                         (out["current_balance"] - prev) / prev.abs(), 0.0)
    out["expected_dpd"] = out["current_status"].map(
        {"Current": 0, "DQ30": 30, "DQ60": 60, "DQ90plus": 90, "Default": 180,
         "Prepaid": 0, "PaidOff": 0}).astype(float)
    return out


def run_rules(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = prepare_for_rules(df)
    flags = pd.DataFrame(index=prepared.index)
    for rule in RULES:
        flags[f"vr_{rule.name}"] = rule.fn(prepared).astype(int)

    summary = pd.DataFrame([{
        "rule": r.name,
        "dimension": r.dimension,
        "severity": r.severity,
        "description": r.description,
        "violations": int(flags[f"vr_{r.name}"].sum()),
        "violation_rate": float(flags[f"vr_{r.name}"].mean()),
    } for r in RULES]).sort_values("violations", ascending=False).reset_index(drop=True)
    return prepared.join(flags), summary


def score_records(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    penalty = np.zeros(len(out))
    for rule in RULES:
        penalty = penalty + out[f"vr_{rule.name}"].to_numpy() * rule.severity
    out["dq_penalty"] = penalty
    out["dq_score"] = np.clip(100.0 - penalty, 0.0, 100.0)
    out["dq_violation_count"] = out[[f"vr_{r.name}" for r in RULES]].sum(axis=1)
    out["dq_band"] = pd.cut(out["dq_score"], [-0.1, 50, 75, 90, 100],
                            labels=["critical", "poor", "watch", "clean"])
    return out


def score_batches(df: pd.DataFrame) -> pd.DataFrame:
    cols = [f"vr_{r.name}" for r in RULES]
    g = df.groupby(["reporting_month", "servicer_name"], observed=True)
    batch = g.agg(records=("loan_id", "size"),
                  mean_dq_score=("dq_score", "mean"),
                  min_dq_score=("dq_score", "min"),
                  pct_critical=("dq_band", lambda s: float((s == "critical").mean())),
                  violations_per_record=("dq_violation_count", "mean")).reset_index()
    top = g[cols].mean().reset_index()
    top["top_failing_rule"] = top[cols].idxmax(axis=1).str.replace("vr_", "", regex=False)
    top["top_failing_rule_rate"] = top[cols].max(axis=1)
    batch = batch.merge(top[["reporting_month", "servicer_name", "top_failing_rule",
                             "top_failing_rule_rate"]],
                        on=["reporting_month", "servicer_name"], how="left")
    batch["batch_grade"] = pd.cut(batch["mean_dq_score"], [-0.1, 70, 82, 91, 100],
                                  labels=["D", "C", "B", "A"])
    return batch.sort_values(["reporting_month", "servicer_name"]).reset_index(drop=True)
