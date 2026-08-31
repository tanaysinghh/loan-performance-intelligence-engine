from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src import config as C
from src.data import validate
from src.features import build_features as F
from src.models import metrics as M
from src.models.splits import purged_time_split

ANOMALY_FEATURES = [
    "amortisation_residual", "dpd_status_residual", "balance_change_1m", "balance_change_3m",
    "balance_growth_excess", "svc_balance_rel_gap", "svc_dpd_gap", "reporting_lag_days",
    "missing_field_count", "doc_incomplete", "manual_upload", "payment_to_balance",
    "dpd_delta_1m", "dpd_repaired", "rate_repaired", "balance_repaired", "age_repaired",
    "svc_present",
]

ANOMALY_FEATURE_RATIONALE = (
    "The first feature set for this model used raw record levels — balance, loan age, "
    "remaining term, original balance. It scored *below* the base exception rate on its own "
    "top decile (lift 0.92x), because a genuinely large, genuinely seasoned jumbo loan is a "
    "statistical outlier and an entirely correct record. The feature set was rebuilt around "
    "quantities where deviation means a *defect* rather than a large loan: residuals against "
    "what the record should say given its own other fields (amortisation against term "
    "elapsed, days past due against reported status), disagreements with the second servicer "
    "feed, reporting timeliness, and repair indicators."
)

DRIVER_LABELS = {
    "amortisation_residual": "balance against expected amortisation for term elapsed",
    "dpd_status_residual": "days past due against reported status",
    "balance_growth_excess": "upward balance movement",
    "doc_incomplete": "document file incomplete",
    "manual_upload": "record arrived by manual upload",
    "dpd_repaired": "days past due required repair",
    "rate_repaired": "note rate required repair",
    "balance_repaired": "balance required repair",
    "age_repaired": "loan age required repair",
    "svc_present": "servicer feed record present",
    "current_balance_clean": "unpaid principal balance",
    "balance_ratio": "balance as a share of original",
    "balance_change_1m": "month-over-month balance movement",
    "balance_change_3m": "three-month balance movement",
    "interest_rate_clean": "note rate",
    "days_past_due_clean": "days past due",
    "loan_age_months_clean": "loan age",
    "remaining_term_months": "remaining term",
    "reporting_lag_days": "servicer reporting lag",
    "svc_balance_rel_gap": "servicer feed balance gap",
    "svc_dpd_gap": "servicer feed days-past-due gap",
    "payment_to_balance": "scheduled payment relative to balance",
    "amortisation_progress": "amortisation progress",
    "term_progress": "share of term elapsed",
    "missing_field_count": "count of missing credit fields",
    "status_ord": "performance status severity",
    "dpd_delta_1m": "month-over-month change in days past due",
    "log_original_balance": "original balance",
}


def fit_isolation_forest(df: pd.DataFrame, train_mask: np.ndarray,
                         contamination: float = 0.06, seed: int = C.RANDOM_SEED):
    cols = [c for c in ANOMALY_FEATURES if c in df.columns]
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", RobustScaler()),
                     ("iso", IsolationForest(n_estimators=400, contamination=contamination,
                                             max_samples=0.7, random_state=seed, n_jobs=-1))])
    pipe.fit(df.loc[train_mask, cols])
    return pipe, cols


def anomaly_scores(pipe, df: pd.DataFrame, cols: list[str]) -> pd.Series:
    raw = -pipe.named_steps["iso"].score_samples(
        pipe.named_steps["scale"].transform(pipe.named_steps["impute"].transform(df[cols])))
    lo, hi = np.percentile(raw, [0.5, 99.5])
    return pd.Series(np.clip((raw - lo) / max(hi - lo, 1e-9), 0, 1), index=df.index)


def anomaly_drivers(df: pd.DataFrame, cols: list[str], reference_mask=None,
                    top_k: int = 3, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    ref = (reference[cols] if reference is not None else df.loc[reference_mask, cols])
    med = ref.median()
    mad = (ref - med).abs().median() * 1.4826
    mad = mad.replace(0, np.nan).fillna(ref.std().replace(0, 1.0)).fillna(1.0)

    z = ((df[cols] - med) / mad).abs().fillna(0.0)
    order = np.argsort(-z.to_numpy(), axis=1)[:, :top_k]
    arr = z.to_numpy()
    out = {}
    for k in range(top_k):
        idx = order[:, k]
        names = np.array(cols)[idx]
        vals = arr[np.arange(len(df)), idx]
        out[f"anomaly_driver_{k + 1}"] = [DRIVER_LABELS.get(n, n) for n in names]
        out[f"anomaly_driver_{k + 1}_zscore"] = np.round(vals, 2)
        out[f"anomaly_driver_{k + 1}_field"] = names
    return pd.DataFrame(out, index=df.index)


def train_exception_models(df: pd.DataFrame, features: list[str]) -> dict:
    import lightgbm as lgb
    from src.models import performance as P

    binary_model, binary_metrics = P.train_binary(df, "exception_required", features)

    split = purged_time_split(df, "exception_type")
    types = [t for t in C.EXCEPTION_TYPES if t != "none"]
    is_exc = df["exception_required"].eq(1).to_numpy()
    mapping = {t: i for i, t in enumerate(types)}

    tr = split.train & is_exc
    va = split.valid & is_exc
    te = split.test & is_exc

    y_tr = df.loc[tr, "exception_type"].map(mapping).astype(int)
    y_va = df.loc[va, "exception_type"].map(mapping).astype(int)
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=len(types),
                             learning_rate=0.06, num_leaves=32, min_child_samples=30,
                             feature_fraction=0.75, bagging_fraction=0.85, bagging_freq=1,
                             lambda_l2=3.0, n_estimators=700, verbose=-1,
                             seed=C.RANDOM_SEED, class_weight="balanced")
    clf.fit(F.design_matrix(df.loc[tr], features), y_tr,
            eval_set=[(F.design_matrix(df.loc[va], features), y_va)],
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])

    rows, reports = [], {}
    for name, mask in (("valid", va), ("test", te)):
        if mask.sum() < 50:
            continue
        y = df.loc[mask, "exception_type"].map(mapping).astype(int).to_numpy()
        proba = clf.predict_proba(F.design_matrix(df.loc[mask], features))
        pred = np.argmax(proba, axis=1)
        m = M.multiclass_metrics(y, pred, proba, labels=range(len(types)))
        m.update({"split": name, "model": "lgbm_exception_type"})
        rows.append(m)

        majority = np.full(len(y), int(pd.Series(y_tr).mode().iloc[0]))
        mb = M.multiclass_metrics(y, majority)
        mb.update({"split": name, "model": "majority_class_baseline"})
        rows.append(mb)
        reports[name] = M.per_class_report([types[i] for i in y],
                                           [types[i] for i in pred], types)

    return {"binary_model": binary_model, "binary_metrics": binary_metrics,
            "type_model": clf, "type_labels": types, "type_mapping": mapping,
            "type_metrics": pd.DataFrame(rows), "type_reports": reports,
            "type_split": split}


def anomaly_vs_exception(df: pd.DataFrame, score: pd.Series, top_pct: float = 0.06) -> dict:
    cutoff = score.quantile(1 - top_pct)
    flagged = score >= cutoff
    y = df["exception_required"].to_numpy()
    return {
        "score_cutoff": float(cutoff),
        "flagged_share": float(flagged.mean()),
        "precision_vs_exception_label": float(y[flagged].mean()) if flagged.any() else 0.0,
        "recall_vs_exception_label": float(y[flagged].sum() / max(y.sum(), 1)),
        "base_exception_rate": float(y.mean()),
        "lift_over_base": float((y[flagged].mean() / max(y.mean(), 1e-9))
                                if flagged.any() else 0.0),
        "roc_auc_vs_exception_label": float(M.binary_metrics(y, score.to_numpy())["roc_auc"]),
    }


def build_review_queue(df: pd.DataFrame, score: pd.Series, exc_prob: np.ndarray,
                       exc_type: np.ndarray, exc_type_conf: np.ndarray,
                       drivers: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    rule_cols = [f"vr_{r.name}" for r in validate.RULES if f"vr_{r.name}" in df.columns]
    fired = df[rule_cols].apply(
        lambda row: "; ".join(c.replace("vr_", "") for c in rule_cols if row[c] == 1), axis=1)

    q = pd.DataFrame({
        "loan_id": df["loan_id"], "reporting_month": df["reporting_month"],
        "servicer_name": df["servicer_name"], "current_status": df["current_status"],
        "anomaly_score": score.round(4),
        "exception_probability": np.round(exc_prob, 4),
        "predicted_exception_type": exc_type,
        "predicted_type_confidence": np.round(exc_type_conf, 3),
        "dq_score": df["dq_score"].round(1),
        "rules_violated": fired,
        "current_balance": df["current_balance"].round(0),
        "servicer_balance_gap_pct": (df["svc_balance_rel_gap"] * 100).round(2),
        "reporting_lag_days": df["reporting_lag_days"],
        "document_status": df["document_status"],
        "actual_exception_required": df["exception_required"],
        "actual_exception_type": df["exception_type"],
    }).join(drivers[[c for c in drivers.columns if not c.endswith("_field")]])

    q["review_priority"] = (0.6 * q["exception_probability"] + 0.4 * q["anomaly_score"]).round(4)
    q = q.sort_values("review_priority", ascending=False)

    diverse = pd.concat([
        q[q["predicted_exception_type"] == t].head(max(3, n // 7))
        for t in pd.unique(q["predicted_exception_type"])
    ]).drop_duplicates(subset=["loan_id", "reporting_month"])
    return pd.concat([q.head(n), diverse]).drop_duplicates(
        subset=["loan_id", "reporting_month"]).sort_values(
        "review_priority", ascending=False).reset_index(drop=True)
