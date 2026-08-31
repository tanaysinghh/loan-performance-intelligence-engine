from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.explain import shap_explain as E
from src.features import build_features as F
from src.models import anomaly as A
from src.models import performance as P
from src.models.splits import purged_time_split

ACTION_RULES = [
    ("raise_exception_for_review", "exception_probability", 0.50,
     "Operational exception is more likely than not; route to the data-quality queue."),
    ("escalate_loss_mitigation", "prob_default_12m", 0.35,
     "Twelve-month default probability is materially elevated; refer to loss mitigation."),
    ("early_stage_collections_outreach", "prob_delinquency_3m", 0.30,
     "Near-term delinquency risk is elevated; contact before the loan rolls."),
    ("watchlist_monitor", "prob_delinquency_6m", 0.25,
     "Medium-term delinquency risk is elevated; add to watchlist."),
    ("retention_review", "prob_prepayment_12m", 0.45,
     "Prepayment risk is elevated; refer for retention pricing review."),
]


def _confidence(prob: np.ndarray, spread: np.ndarray) -> np.ndarray:
    return E.confidence_band(prob, spread)


def build(df: pd.DataFrame | None = None, models: dict | None = None,
          scope: str = "latest_per_loan") -> pd.DataFrame:
    from src.features.dataset import prepare
    df = prepare() if df is None else df
    models = P.load() if models is None else models
    features = F.feature_columns(df)

    if scope == "latest_per_loan":
        d = df.sort_values(["loan_id", "month_index"], kind="mergesort")
        rows = d.groupby("loan_id", sort=False).tail(1).index
        mask = df.index.isin(rows)
    else:
        mask = purged_time_split(df, "next_3m_delinquency_flag").test
    sub = df.loc[mask].copy()

    out = pd.DataFrame({
        "loan_id": sub["loan_id"].to_numpy(),
        "reporting_month": sub["reporting_month"].to_numpy(),
        "servicer_name": sub["servicer_name"].to_numpy(),
        "current_status": sub["current_status"].to_numpy(),
    })

    name_map = {"next_3m_delinquency_flag": "prob_delinquency_3m",
                "next_6m_delinquency_flag": "prob_delinquency_6m",
                "next_12m_default_flag": "prob_default_12m",
                "next_12m_prepayment_flag": "prob_prepayment_12m",
                "exception_required": "exception_probability"}
    for target, col in name_map.items():
        if target in models:
            out[col] = np.round(models[target].predict_proba(sub), 6)
        else:
            out[col] = np.nan

    state = models["next_state"]
    state_proba = state["model"].predict_proba(F.design_matrix(sub, state["features"]))
    state_idx = np.argmax(state_proba, axis=1)
    out["predicted_next_state"] = np.array(state["labels"])[state_idx]
    out["next_state_confidence"] = np.round(state_proba[np.arange(len(sub)), state_idx], 6)

    exc_split = purged_time_split(df, "exception_required")
    iso, iso_cols = A.fit_isolation_forest(df, exc_split.train)
    out["anomaly_score"] = np.round(A.anomaly_scores(iso, sub, iso_cols).to_numpy(), 6)
    anomaly_drivers = A.anomaly_drivers(sub, iso_cols, reference=df.loc[exc_split.train])
    out["top_anomaly_driver"] = anomaly_drivers["anomaly_driver_1"].to_numpy()

    exc_models = getattr(build, "_exception_type_model", None)
    if exc_models is None:
        exc_models = A.train_exception_models(df, features)
        build._exception_type_model = exc_models
    type_proba = exc_models["type_model"].predict_proba(F.design_matrix(sub, features))
    type_idx = np.argmax(type_proba, axis=1)
    predicted_type = np.array(exc_models["type_labels"])[type_idx]
    out["predicted_exception_type"] = np.where(out["exception_probability"] >= 0.50,
                                               predicted_type, "none")
    out["exception_type_confidence"] = np.round(
        type_proba[np.arange(len(sub)), type_idx], 6)

    driver_target = "next_12m_default_flag"
    exp = E.explain(models[driver_target], df, mask, max_rows=len(sub) + 1)
    drivers = E.top_drivers_for_rows(exp)
    out["top_drivers_default_model"] = drivers.reindex(sub.index).to_numpy()

    unc = E.uncertainty(models[driver_target], df, mask)
    out["confidence"] = _confidence(out["prob_default_12m"].to_numpy(),
                                    unc["staged_std"].to_numpy())
    out["prediction_spread"] = np.round(unc["staged_std"].to_numpy(), 6)

    action = np.full(len(out), "monitor_no_action", dtype=object)
    reason = np.full(len(out), "All model scores below action thresholds.", dtype=object)
    for name, col, threshold, why in reversed(ACTION_RULES):
        hit = out[col].to_numpy() >= threshold
        action = np.where(hit, name, action)
        reason = np.where(hit, why, reason)
    out["recommended_action"] = action
    out["action_reason"] = reason
    out["action_is_recommendation_not_decision"] = True

    return out.sort_values("prob_default_12m", ascending=False).reset_index(drop=True)


def write(out: pd.DataFrame | None = None) -> pd.DataFrame:
    out = build() if out is None else out
    path = C.SUBMISSION / "submission.csv"
    out.to_csv(path, index=False)
    return out


if __name__ == "__main__":
    o = write()
    print(o.shape)
    print(o["recommended_action"].value_counts().to_string())
    print(o["confidence"].value_counts().to_string())
