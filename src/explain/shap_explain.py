from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from src import config as C
from src.features import build_features as F
from src.models import metrics as M

FRIENDLY = {
    "credit_ord": "credit score band",
    "ltv_ord": "loan-to-value band",
    "dti_ord": "debt-to-income band",
    "status_ord": "current performance status",
    "days_past_due_clean": "days past due",
    "max_dpd_last_3m": "worst days past due in last 3 months",
    "max_dpd_last_6m": "worst days past due in last 6 months",
    "max_dpd_last_12m": "worst days past due in last 12 months",
    "months_dq_last_6m": "months delinquent in last 6 months",
    "months_dq_last_12m": "months delinquent in last 12 months",
    "worst_status_to_date": "worst status reached to date",
    "ever_delinquent_to_date": "has ever been delinquent",
    "current_streak_clean": "consecutive clean months",
    "rate_incentive": "refinance incentive (note rate less market rate)",
    "refi_incentive_positive": "positive refinance incentive",
    "interest_rate_clean": "note rate",
    "loan_age_months_clean": "loan age",
    "balance_ratio": "balance as a share of original",
    "amortisation_residual": "balance against expected amortisation",
    "amortisation_progress": "amortisation progress",
    "term_progress": "share of term elapsed",
    "log_current_balance": "current balance",
    "log_original_balance": "original balance",
    "unemployment_rate": "unemployment rate",
    "market_mortgage_rate": "market mortgage rate",
    "hpi_yoy_growth": "house price growth",
    "modification_flag": "loss-mitigation modification applied",
    "dq_score": "record data-quality score",
    "svc_balance_rel_gap": "servicer feed balance gap",
    "reporting_lag_days": "servicer reporting lag",
    "document_status": "document custody status",
    "servicer_name": "servicer",
    "state": "state",
    "payment_to_balance": "scheduled payment relative to balance",
    "scheduled_payment": "scheduled monthly payment",
    "dpd_status_residual": "days past due against reported status",
}


def friendly(name: str) -> str:
    return FRIENDLY.get(name, name.replace("_", " "))


def explain(model, df: pd.DataFrame, sample_mask: np.ndarray,
            max_rows: int = 4000, seed: int = C.RANDOM_SEED):
    idx = np.where(sample_mask)[0]
    if len(idx) > max_rows:
        idx = np.random.default_rng(seed).choice(idx, max_rows, replace=False)
        idx.sort()
    sub = df.iloc[idx]
    X = F.design_matrix(sub, model.features)
    explainer = shap.TreeExplainer(model.booster)
    values = explainer.shap_values(X)
    if isinstance(values, list):
        values = values[1]
    return {"shap_values": np.asarray(values), "X": X, "rows": sub,
            "expected_value": float(np.ravel(explainer.expected_value)[-1]),
            "index": idx}


def global_importance(exp: dict, top_n: int = 25) -> pd.DataFrame:
    sv = exp["shap_values"]
    X = exp["X"]
    mean_abs = np.abs(sv).mean(axis=0)
    signed = sv.mean(axis=0)
    corr = []
    for j, col in enumerate(X.columns):
        v = X[col]
        if pd.api.types.is_numeric_dtype(v) and v.notna().sum() > 30 and v.nunique() > 2:
            corr.append(float(pd.Series(sv[:, j]).corr(v.astype(float))))
        else:
            corr.append(np.nan)
    out = pd.DataFrame({
        "feature": X.columns, "plain_english": [friendly(c) for c in X.columns],
        "mean_abs_shap": mean_abs, "mean_signed_shap": signed,
        "direction_corr_with_value": corr,
    })
    out["share_of_total_attribution"] = out["mean_abs_shap"] / out["mean_abs_shap"].sum()
    out["direction"] = np.where(out["direction_corr_with_value"] > 0.05, "higher value raises risk",
                        np.where(out["direction_corr_with_value"] < -0.05, "higher value lowers risk",
                                 "non-monotone / categorical"))
    return out.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)


def local_explanations(exp: dict, model, probs: np.ndarray, n: int = 12,
                       top_k: int = 4, select: str = "highest") -> pd.DataFrame:
    sv = exp["shap_values"]
    X = exp["X"]
    rows = exp["rows"]
    p = probs[exp["index"]]
    order = np.argsort(-p) if select == "highest" else np.argsort(p)
    picks = order[:n]

    records = []
    for i in picks:
        contrib = sv[i]
        top = np.argsort(-np.abs(contrib))[:top_k]
        rec = {
            "loan_id": rows["loan_id"].iloc[i],
            "reporting_month": rows["reporting_month"].iloc[i],
            "current_status": rows["current_status"].iloc[i],
            "calibrated_probability": round(float(p[i]), 4),
            "baseline_log_odds": round(exp["expected_value"], 4),
            "total_shap_log_odds": round(float(contrib.sum()), 4),
        }
        for k, j in enumerate(top, start=1):
            col = X.columns[j]
            val = X[col].iloc[i]
            rec[f"driver_{k}"] = friendly(col)
            rec[f"driver_{k}_value"] = (round(float(val), 3)
                                        if isinstance(val, (int, float, np.floating))
                                        and pd.notna(val) else str(val))
            rec[f"driver_{k}_log_odds"] = round(float(contrib[j]), 4)
        records.append(rec)
    return pd.DataFrame(records)


def top_drivers_for_rows(exp: dict, top_k: int = 3) -> pd.DataFrame:
    sv = exp["shap_values"]
    X = exp["X"]
    order = np.argsort(-np.abs(sv), axis=1)[:, :top_k]
    out = []
    for i in range(len(sv)):
        parts = []
        for j in order[i]:
            sign = "+" if sv[i, j] >= 0 else "-"
            parts.append(f"{friendly(X.columns[j])} ({sign}{abs(sv[i, j]):.2f})")
        out.append(" | ".join(parts))
    return pd.Series(out, index=exp["rows"].index)


def uncertainty(model, df: pd.DataFrame, mask: np.ndarray, n_rounds: int = 12) -> pd.DataFrame:
    """Prediction interval from staged boosting-round predictions.

    Predictions from the last `n_rounds` boosting iterations are collected and their spread
    used as an epistemic-uncertainty proxy. This is cheap and honest about what it is: it
    captures sensitivity to where the boosting sequence was stopped, not full model
    uncertainty. It is *not* a statistical confidence interval and is not labelled as one.
    """
    X = F.design_matrix(df.loc[mask], model.features)
    best = model.best_iteration or model.booster.n_estimators_
    starts = np.linspace(max(best - n_rounds * 4, 5), best, n_rounds).astype(int)
    preds = np.column_stack([model.booster.predict_proba(X, num_iteration=int(s))[:, 1]
                             for s in starts])
    cal = model.predict_proba(df.loc[mask])
    return pd.DataFrame({
        "calibrated_probability": cal,
        "staged_mean": preds.mean(axis=1),
        "staged_std": preds.std(axis=1),
        "staged_p10": np.percentile(preds, 10, axis=1),
        "staged_p90": np.percentile(preds, 90, axis=1),
    }, index=df.index[mask])


def confidence_band(prob: np.ndarray, spread: np.ndarray) -> np.ndarray:
    decisive = np.minimum(prob, 1 - prob)
    score = decisive + 3.0 * spread
    return np.where(score < 0.10, "high", np.where(score < 0.25, "medium", "low"))


def error_analysis(df: pd.DataFrame, mask: np.ndarray, y: np.ndarray, probs: np.ndarray,
                   threshold: float, segments=("credit_score_band", "servicer_name",
                                               "current_status", "state")) -> dict:
    pred = (probs >= threshold).astype(int)
    frame = df.loc[mask].copy()
    frame["_y"] = y
    frame["_p"] = probs
    frame["_pred"] = pred
    frame["_fp"] = ((pred == 1) & (y == 0)).astype(int)
    frame["_fn"] = ((pred == 0) & (y == 1)).astype(int)

    overall = {
        "threshold": float(threshold),
        "n": int(len(frame)),
        "positives": int(y.sum()),
        "predicted_positives": int(pred.sum()),
        "true_positives": int(((pred == 1) & (y == 1)).sum()),
        "false_positives": int(frame["_fp"].sum()),
        "false_negatives": int(frame["_fn"].sum()),
        "precision": float(y[pred == 1].mean()) if pred.sum() else 0.0,
        "recall": float(pred[y == 1].mean()) if y.sum() else 0.0,
    }

    seg_tables = {}
    for s in segments:
        if s not in frame.columns:
            continue
        g = frame.groupby(s, observed=True).agg(
            n=("_y", "size"), actual_rate=("_y", "mean"), mean_predicted=("_p", "mean"),
            false_positive_rate=("_fp", "mean"), false_negative_rate=("_fn", "mean")).reset_index()
        g["calibration_gap"] = g["actual_rate"] - g["mean_predicted"]
        seg_tables[s] = g.sort_values("false_negative_rate", ascending=False).reset_index(drop=True)

    fp_profile = frame[frame["_fp"] == 1]
    fn_profile = frame[frame["_fn"] == 1]
    tn_profile = frame[(frame["_pred"] == 0) & (frame["_y"] == 0)]
    compare_cols = ["credit_ord", "ltv_ord", "status_ord", "days_past_due_clean",
                    "max_dpd_last_6m", "loan_age_months_clean", "dq_score", "balance_ratio"]
    compare_cols = [c for c in compare_cols if c in frame.columns]
    profile = pd.DataFrame({
        "feature": [friendly(c) for c in compare_cols],
        "false_positives": [fp_profile[c].mean() for c in compare_cols],
        "false_negatives": [fn_profile[c].mean() for c in compare_cols],
        "true_negatives": [tn_profile[c].mean() for c in compare_cols],
    })
    return {"overall": overall, "segments": seg_tables, "profile": profile,
            "fp_rows": fp_profile, "fn_rows": fn_profile}
