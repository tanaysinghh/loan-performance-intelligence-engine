"""Builds grounding packs: the only facts the copilot is permitted to speak about.

The copilot never touches the model objects or the dataframe. It receives a JSON pack of
numbers that a non-LLM model already produced, and its job is to turn those numbers into
prose. Anything not in the pack is, by construction, something it must refuse to assert.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import config as C


def _num(x, digits=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), digits)


def loan_pack(row: pd.Series, predictions: dict, drivers: str,
              anomaly: dict | None = None) -> dict:
    """Everything the copilot may say about one loan-month record."""
    pack = {
        "record": {
            "loan_id": str(row["loan_id"]),
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
    """Every numeric value anywhere in the pack, for the grounding validator."""
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
                    found.add(float(token))
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
_NUM_RE = re.compile(r"-?\d+\.?\d*")
