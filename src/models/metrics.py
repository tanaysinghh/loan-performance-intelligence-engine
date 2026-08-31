from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, log_loss, precision_recall_curve, roc_auc_score)


def recall_at_precision(y_true, y_prob, target_precision: float) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ok = precision[:-1] >= target_precision
    if not ok.any():
        return {"recall": 0.0, "threshold": float("nan"),
                "achieved_precision": float(precision[:-1].max())}
    best = int(np.argmax(recall[:-1] * ok))
    return {"recall": float(recall[best]), "threshold": float(thresholds[best]),
            "achieved_precision": float(precision[best])}


def best_f1(y_true, y_prob) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    denom = precision[:-1] + recall[:-1]
    f1 = np.where(denom > 0, 2 * precision[:-1] * recall[:-1] / np.where(denom > 0, denom, 1), 0)
    if len(f1) == 0:
        return {"f1": 0.0, "threshold": 0.5, "precision": 0.0, "recall": 0.0}
    i = int(np.argmax(f1))
    return {"f1": float(f1[i]), "threshold": float(thresholds[i]),
            "precision": float(precision[i]), "recall": float(recall[i])}


def ks_statistic(y_true, y_prob) -> float:
    order = np.argsort(y_prob)
    y = np.asarray(y_true)[order]
    pos = np.cumsum(y) / max(y.sum(), 1)
    neg = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return float(np.max(np.abs(pos - neg)))


def lift_at_k(y_true, y_prob, k: float = 0.10) -> float:
    n = max(int(len(y_prob) * k), 1)
    top = np.argsort(-np.asarray(y_prob))[:n]
    base = np.mean(y_true)
    return float(np.mean(np.asarray(y_true)[top]) / base) if base > 0 else float("nan")


def binary_metrics(y_true, y_prob, prefix: str = "") -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return {f"{prefix}n": len(y_true), f"{prefix}positive_rate": float(y_true.mean())}
    f1 = best_f1(y_true, y_prob)
    r30 = recall_at_precision(y_true, y_prob, 0.30)
    r50 = recall_at_precision(y_true, y_prob, 0.50)
    return {
        f"{prefix}n": int(len(y_true)),
        f"{prefix}positive_rate": float(y_true.mean()),
        f"{prefix}roc_auc": float(roc_auc_score(y_true, y_prob)),
        f"{prefix}pr_auc": float(average_precision_score(y_true, y_prob)),
        f"{prefix}pr_auc_lift_over_base": float(average_precision_score(y_true, y_prob) / max(y_true.mean(), 1e-9)),
        f"{prefix}best_f1": f1["f1"],
        f"{prefix}best_f1_threshold": f1["threshold"],
        f"{prefix}precision_at_best_f1": f1["precision"],
        f"{prefix}recall_at_best_f1": f1["recall"],
        f"{prefix}recall_at_precision_30": r30["recall"],
        f"{prefix}threshold_at_precision_30": r30["threshold"],
        f"{prefix}recall_at_precision_50": r50["recall"],
        f"{prefix}threshold_at_precision_50": r50["threshold"],
        f"{prefix}brier": float(brier_score_loss(y_true, y_prob)),
        f"{prefix}log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7))),
        f"{prefix}ks": ks_statistic(y_true, y_prob),
        f"{prefix}lift_at_10pct": lift_at_k(y_true, y_prob, 0.10),
    }


def multiclass_metrics(y_true, y_pred, y_proba=None, labels=None, prefix: str = "") -> dict:
    out = {
        f"{prefix}n": int(len(y_true)),
        f"{prefix}accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        f"{prefix}macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f"{prefix}weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_proba is not None and labels is not None:
        try:
            out[f"{prefix}log_loss"] = float(log_loss(y_true, y_proba, labels=list(labels)))
        except ValueError:
            pass
        try:
            out[f"{prefix}macro_roc_auc"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro",
                              labels=list(labels)))
        except ValueError:
            pass
    return out


def per_class_report(y_true, y_pred, labels) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for i, lab in enumerate(labels):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({"class": lab, "support": int(cm[i].sum()), "precision": prec,
                     "recall": rec, "f1": f1})
    return pd.DataFrame(rows)


def calibration_table(y_true, y_prob, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y_true).astype(int), "p": np.asarray(y_prob, dtype=float)})
    try:
        df["bucket"] = pd.qcut(df["p"], bins, duplicates="drop")
    except ValueError:
        df["bucket"] = pd.cut(df["p"], bins)
    g = df.groupby("bucket", observed=True).agg(n=("y", "size"), mean_predicted=("p", "mean"),
                                                observed_rate=("y", "mean")).reset_index()
    g["calibration_gap"] = g["observed_rate"] - g["mean_predicted"]
    g["bucket"] = g["bucket"].astype(str)
    return g


def expected_calibration_error(y_true, y_prob, bins: int = 10) -> float:
    t = calibration_table(y_true, y_prob, bins)
    w = t["n"] / t["n"].sum()
    return float((w * t["calibration_gap"].abs()).sum())
