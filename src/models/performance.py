from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config as C
from src.features import build_features as F
from src.models import metrics as M
from src.models.splits import Split, purged_time_split, loan_disjoint_time_split

LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.045,
    num_leaves=48,
    min_child_samples=80,
    feature_fraction=0.72,
    bagging_fraction=0.82,
    bagging_freq=1,
    lambda_l2=4.0,
    max_depth=-1,
    n_estimators=1400,
    verbose=-1,
    seed=C.RANDOM_SEED,
)

PARAM_GRID = [
    dict(learning_rate=0.030, num_leaves=12, min_child_samples=250, n_estimators=2000),
    dict(learning_rate=0.030, num_leaves=24, min_child_samples=150, n_estimators=2000),
    dict(learning_rate=0.045, num_leaves=24, min_child_samples=80, n_estimators=1400),
    dict(learning_rate=0.045, num_leaves=48, min_child_samples=80, n_estimators=1400),
    dict(learning_rate=0.060, num_leaves=64, min_child_samples=40, n_estimators=1000),
]


@dataclass
class TrainedBinaryModel:
    target: str
    booster: lgb.LGBMClassifier
    calibrator: IsotonicRegression | None
    features: list[str]
    baseline: Pipeline
    baseline_features: list[str]
    prior: float
    best_iteration: int
    thresholds: dict = field(default_factory=dict)
    calibrator_name: str = "isotonic"
    calibrator_scores: dict = field(default_factory=dict)
    chosen_params: dict = field(default_factory=dict)
    search: object = None

    def predict_proba(self, df: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
        X = F.design_matrix(df, self.features)
        p = self.booster.predict_proba(X)[:, 1]
        if calibrated and self.calibrator is not None:
            p = self.calibrator.predict(p)
        return np.clip(p, 1e-6, 1 - 1e-6)


def _fit_baseline(df, split, target, cols):
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=2000, C=0.5,
                                                class_weight="balanced"))])
    pipe.fit(df.loc[split.train, cols], df.loc[split.train, target].astype(int))
    return pipe


def _fit_one(X_tr, y_tr, X_va, y_va, params):
    p = dict(LGB_PARAMS)
    p.update(params)
    pos = max(int(y_tr.sum()), 1)
    neg = max(int((1 - y_tr).sum()), 1)
    p["scale_pos_weight"] = float(np.sqrt(neg / pos))
    clf = lgb.LGBMClassifier(**p)
    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="average_precision",
            callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)])
    return clf


def _make_calibrator(name, raw, y):
    if name == "isotonic":
        return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y)
    logit = _to_logit(raw)
    return _PlattWrapper(LogisticRegression(max_iter=1000).fit(logit.reshape(-1, 1), y))


def _to_logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _fit_calibrator(raw_va, y_va, seed: int = C.RANDOM_SEED):
    from sklearn.metrics import log_loss as _ll
    from sklearn.model_selection import StratifiedKFold

    y_va = np.asarray(y_va).astype(int)
    raw_va = np.asarray(raw_va, dtype=float)
    scores = {}
    if y_va.sum() >= 15 and (1 - y_va).sum() >= 15:
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        for name in ("platt", "isotonic"):
            losses = []
            for tr, te in skf.split(raw_va, y_va):
                if y_va[tr].sum() == 0 or y_va[te].sum() == 0:
                    continue
                cal = _make_calibrator(name, raw_va[tr], y_va[tr])
                losses.append(_ll(y_va[te], np.clip(cal.predict(raw_va[te]), 1e-7, 1 - 1e-7)))
            scores[name] = float(np.mean(losses)) if losses else np.inf
    else:
        scores = {"platt": 0.0, "isotonic": np.inf}

    best_name = min(scores, key=scores.get)
    return _make_calibrator(best_name, raw_va, y_va), best_name, scores


class _PlattWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, p):
        return self.model.predict_proba(_to_logit(p).reshape(-1, 1))[:, 1]


def train_binary(df: pd.DataFrame, target: str, features: list[str],
                 split: Split | None = None, params: dict | None = None,
                 tune: bool = True) -> tuple:
    split = split or purged_time_split(df, target)

    y_tr = df.loc[split.train, target].astype(int)
    y_va = df.loc[split.valid, target].astype(int)
    X_tr = F.design_matrix(df.loc[split.train], features)
    X_va = F.design_matrix(df.loc[split.valid], features)

    from sklearn.metrics import average_precision_score
    grid = PARAM_GRID if (tune and params is None) else [params or {}]
    search = []
    best_clf, best_score, best_params = None, -np.inf, {}
    for cand in grid:
        clf = _fit_one(X_tr, y_tr, X_va, y_va, cand)
        score = (float(average_precision_score(y_va, clf.predict_proba(X_va)[:, 1]))
                 if y_va.nunique() > 1 else 0.0)
        search.append({**cand, "valid_pr_auc": score,
                       "best_iteration": int(getattr(clf, "best_iteration_", 0) or 0)})
        if score > best_score:
            best_clf, best_score, best_params = clf, score, cand
    clf = best_clf

    raw_va = clf.predict_proba(X_va)[:, 1]
    if y_va.nunique() > 1:
        calibrator, calibrator_name, calibrator_scores = _fit_calibrator(raw_va, y_va)
    else:
        calibrator, calibrator_name, calibrator_scores = None, "none", {}

    baseline_cols = [c for c in F.BASELINE_FEATURES if c in df.columns]
    baseline = _fit_baseline(df, split, target, baseline_cols)
    prior = float(y_tr.mean())

    model = TrainedBinaryModel(
        target=target, booster=clf, calibrator=calibrator, features=features,
        baseline=baseline, baseline_features=baseline_cols, prior=prior,
        best_iteration=int(getattr(clf, "best_iteration_", 0) or 0),
    )
    model.calibrator_name = calibrator_name
    model.calibrator_scores = calibrator_scores
    model.search = pd.DataFrame(search)
    model.chosen_params = best_params

    rows = []
    for name, mask in (("valid", split.valid), ("test", split.test)):
        y = df.loc[mask, target].astype(int)
        if y.nunique() < 2:
            continue
        Xm = F.design_matrix(df.loc[mask], features)
        raw = clf.predict_proba(Xm)[:, 1]
        cal = (np.clip(calibrator.predict(raw), 1e-6, 1 - 1e-6)
               if calibrator is not None else raw)
        base = baseline.predict_proba(df.loc[mask, baseline_cols])[:, 1]
        prior_p = np.full(len(y), prior)

        for model_name, probs in (("prior", prior_p), ("baseline_logistic", base),
                                  ("lgbm_raw", raw), ("lgbm_calibrated", cal)):
            m = M.binary_metrics(y, probs)
            m.update({"target": target, "split": name, "model": model_name,
                      "ece": M.expected_calibration_error(y, probs)})
            rows.append(m)

    y_te = df.loc[split.test, target].astype(int)
    if y_te.nunique() > 1:
        cal_te = model.predict_proba(df.loc[split.test])
        model.thresholds = {
            "precision_30": M.recall_at_precision(y_te, cal_te, 0.30)["threshold"],
            "precision_50": M.recall_at_precision(y_te, cal_te, 0.50)["threshold"],
            "best_f1": M.best_f1(y_te, cal_te)["threshold"],
        }
    return model, pd.DataFrame(rows)


def _markov_baseline(df, split, mapping, labels):
    tr = df.loc[split.train, ["current_status", "next_state"]].dropna()
    tab = pd.crosstab(tr["current_status"], tr["next_state"])
    tab = tab.reindex(columns=labels, fill_value=0)
    prior = tr["next_state"].value_counts().reindex(labels, fill_value=0)
    prior = ((prior + 1) / (prior.sum() + len(labels))).to_numpy()
    probs = ((tab + 0.5).div((tab + 0.5).sum(axis=1), axis=0))
    return probs, prior


def _markov_predict(states, probs, prior, labels):
    out = np.tile(prior, (len(states), 1))
    known = probs.index
    for i, s in enumerate(states):
        if s in known:
            out[i] = probs.loc[s].to_numpy()
    return out


def train_next_state(df: pd.DataFrame, features: list[str]) -> tuple:
    target = "next_state"
    split = purged_time_split(df, target)
    labels = sorted(df.loc[split.train, target].dropna().unique())
    mapping = {l: i for i, l in enumerate(labels)}

    y_tr = df.loc[split.train, target].map(mapping)
    y_va = df.loc[split.valid, target].map(mapping)
    keep_tr = y_tr.notna().to_numpy()
    keep_va = y_va.notna().to_numpy()

    X_tr = F.design_matrix(df.loc[split.train][keep_tr], features)
    X_va = F.design_matrix(df.loc[split.valid][keep_va], features)
    y_tr_v = y_tr[keep_tr].astype(int)
    y_va_v = y_va[keep_va].astype(int)

    from sklearn.metrics import f1_score
    best, best_score, best_weight = None, -np.inf, None
    for weight in (None, "balanced"):
        cand = lgb.LGBMClassifier(objective="multiclass", num_class=len(labels),
                                  learning_rate=0.05, num_leaves=40, min_child_samples=60,
                                  feature_fraction=0.7, bagging_fraction=0.85, bagging_freq=1,
                                  lambda_l2=4.0, n_estimators=900, verbose=-1,
                                  seed=C.RANDOM_SEED, class_weight=weight)
        cand.fit(X_tr, y_tr_v, eval_set=[(X_va, y_va_v)],
                 callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        score = f1_score(y_va_v, cand.predict(X_va), average="macro", zero_division=0)
        if score > best_score:
            best, best_score, best_weight = cand, score, weight
    clf = best

    probs, prior = _markov_baseline(df, split, mapping, labels)

    rows, reports = [], {}
    for name, mask in (("valid", split.valid), ("test", split.test)):
        y = df.loc[mask, target].map(mapping)
        keep = y.notna().to_numpy()
        if keep.sum() == 0:
            continue
        sub = df.loc[mask][keep]
        proba = clf.predict_proba(F.design_matrix(sub, features))
        pred = np.argmax(proba, axis=1)
        yv = y[keep].astype(int).to_numpy()

        persistence = (sub["current_status"].map(mapping)
                       .fillna(mapping.get("Current", 0)).astype(int).to_numpy())
        markov_proba = _markov_predict(sub["current_status"].to_numpy(), probs, prior, labels)
        markov_pred = np.argmax(markov_proba, axis=1)

        for model_name, pr, pb in (("lgbm_multiclass", pred, proba),
                                   ("markov_transition_baseline", markov_pred, markov_proba),
                                   ("persistence_baseline", persistence, None)):
            m = M.multiclass_metrics(yv, pr, pb, labels=range(len(labels)))
            m.update({"split": name, "model": model_name})
            rows.append(m)
        reports[name] = M.per_class_report(
            [labels[i] for i in yv], [labels[i] for i in pred], labels)

    return ({"model": clf, "labels": labels, "mapping": mapping, "features": features,
             "split": split, "class_weight": best_weight,
             "markov_probs": probs, "markov_prior": prior},
            pd.DataFrame(rows), reports)


def _unused_leakage_probe(df: pd.DataFrame, target: str, features: list[str]) -> dict:
    honest = purged_time_split(df, target)
    _, honest_metrics = train_binary(df, target, features, split=honest)
    honest_auc = honest_metrics.query("split == 'test' and model == 'lgbm_calibrated'")["roc_auc"]

    rng = np.random.default_rng(C.RANDOM_SEED)
    labelled = df[target].notna().to_numpy()
    r = rng.random(len(df))
    leaky = Split(train=labelled & (r < 0.7), valid=labelled & (r >= 0.7) & (r < 0.85),
                  test=labelled & (r >= 0.85), horizon=0, purged_rows=0,
                  train_end="random", valid_end="random")
    _, leaky_metrics = train_binary(df, target, features, split=leaky)
    leaky_auc = leaky_metrics.query("split == 'test' and model == 'lgbm_calibrated'")["roc_auc"]

    disjoint = loan_disjoint_time_split(df, target)
    _, dis_metrics = train_binary(df, target, features, split=disjoint)
    dis_auc = dis_metrics.query("split == 'test' and model == 'lgbm_calibrated'")["roc_auc"]

    return {
        "target": target,
        "purged_time_split_auc": float(honest_auc.iloc[0]) if len(honest_auc) else np.nan,
        "loan_disjoint_time_split_auc": float(dis_auc.iloc[0]) if len(dis_auc) else np.nan,
        "random_row_split_auc": float(leaky_auc.iloc[0]) if len(leaky_auc) else np.nan,
    }


def save(models: dict, path=None) -> None:
    path = path or (C.ARTIFACTS / "performance_models.pkl")
    with open(path, "wb") as fh:
        pickle.dump(models, fh)


def load(path=None) -> dict:
    path = path or (C.ARTIFACTS / "performance_models.pkl")
    with open(path, "rb") as fh:
        return pickle.load(fh)
