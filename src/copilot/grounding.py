from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import config as C
from src import ids


def _num(x, digits=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), digits)


def loan_pack(row: pd.Series, predictions: dict, drivers: str,
              anomaly: dict | None = None) -> dict:
    pack = {
        "record": {
            # Masked before the pack is built, so no real Loan Sequence Number is sent to the
            # model or written into the prompt log on any future run. The mask is stable, so a
            # logged id still matches the same loan in the reporting artefacts.
            "loan_id": ids.hash_loan_id(row["loan_id"]),
            "reporting_month": str(row["reporting_month"]),
            "servicer_name": str(row["servicer_name"]),
            "current_status": str(row["current_status"]),
            "days_past_due": _num(row.get("days_past_due_clean"), 0),
            "loan_age_months": _num(row.get("loan_age_months_clean"), 0),
            "credit_score_band": (None if pd.isna(row.get("credit_score_band"))
                                  else str(row.get("credit_score_band"))),
            "ltv_band": (None if pd.isna(row.get("ltv_band")) else str(row.get("ltv_band"))),
            "current_balance": _num(row.get("current_balance_clean"), 2),
            "modification_flag": int(row.get("modification_flag", 0)),
            "document_status": str(row.get("document_status")),
            "data_quality_score": _num(row.get("dq_score"), 1),
        },
        "model_predictions": {k: _num(v) for k, v in predictions.items()},
        "top_drivers_from_shap": drivers,
        "provenance": {
            "predictions_produced_by": "LightGBM gradient-boosted trees, calibrated",
            "drivers_produced_by": "SHAP TreeExplainer over the same models",
            "llm_role": "narration only; produced no number in this pack",
        },
    }
    if anomaly:
        pack["anomaly"] = {k: (_num(v) if isinstance(v, (int, float, np.floating)) else str(v))
                           for k, v in anomaly.items()}
    return pack


def portfolio_pack(headline: pd.DataFrame, segments: dict, metrics: pd.DataFrame) -> dict:
    test = metrics[(metrics["split"] == "test") & (metrics["model"] == "lgbm_calibrated")]
    return {
        "scenario_projections": json.loads(headline.round(5).to_json(orient="records")),
        "worst_segments_by_adverse_default_delta": json.loads(
            segments["credit_score_band"].round(5).head(5).to_json(orient="records")),
        "prepayment_by_incentive_bucket": json.loads(
            segments["prepay_by_rate_incentive"].round(5).to_json(orient="records")),
        "model_test_metrics": json.loads(
            test[["target", "roc_auc", "pr_auc", "brier", "ece"]].round(4)
            .to_json(orient="records")),
        "provenance": {
            "projections_produced_by": "LightGBM repricing (Engine A) and macro-conditioned "
                                       "Markov chain (Engine B)",
            "llm_role": "narration only; produced no number in this pack",
        },
    }


def dictionary_pack(dictionary: pd.DataFrame, fields: list[str]) -> dict:
    sub = dictionary[dictionary["field"].isin(fields)]
    return {"data_dictionary_entries": json.loads(sub.to_json(orient="records")),
            "provenance": {"source": "data/raw/data_dictionary.csv",
                           "llm_role": "retrieval and phrasing only"}}


def extract_numbers(pack: dict) -> set[float]:
    found = set()

    def walk(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)
        elif isinstance(obj, bool):
            return
        elif isinstance(obj, (int, float, np.integer, np.floating)):
            if np.isfinite(float(obj)):
                found.add(float(obj))
        elif isinstance(obj, str):
            for token in _NUM_RE.findall(obj):
                try:
                    found.add(float(token.rstrip("%").replace(",", "")))
                except ValueError:
                    pass

    walk(pack)
    derived = set()
    for v in found:
        derived.add(round(v * 100, 6))
        derived.add(round(v / 100, 6))
        derived.add(round(v, 0))
        derived.add(round(v, 1))
        derived.add(round(v, 2))
        derived.add(round(v, 3))
    return found | derived


import re

NUMBER_TOKEN_RE = re.compile(r"(?<![\w.])-?\d[\d,]*\.?\d*(?:[eE][+-]?\d+)?%?")
_NUM_RE = NUMBER_TOKEN_RE


def rule_pack(rule_summary: pd.DataFrame, rules_json: dict,
              worst_batches: pd.DataFrame | None = None) -> dict:
    cols = [c for c in ("rule", "dimension", "severity", "rows_flagged", "flag_rate")
            if c in rule_summary.columns]
    pack = {
        "existing_rules": rules_json.get("rules", []),
        "dimensions_covered": rules_json.get("dimensions", []),
        "observed_violation_rates": json.loads(rule_summary[cols].to_json(orient="records")),
        "provenance": {
            "rules": "data/raw/validation_rules.json",
            "rates": "reports/validation_rule_summary.csv",
            "llm_role": ("draft candidate rules for human review; the LLM does not add rules "
                         "to the engine and cannot execute one"),
        },
    }
    if worst_batches is not None and len(worst_batches):
        keep = [c for c in ("reporting_month", "servicer_name", "mean_dq_score",
                            "batch_grade") if c in worst_batches.columns]
        pack["worst_scoring_batches"] = json.loads(
            worst_batches[keep].head(5).to_json(orient="records"))
    return pack
