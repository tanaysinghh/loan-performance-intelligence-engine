"""Column profiling, missingness structure, dependency analysis and drift measurement."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src import config as C

NUMERIC_PROFILE_COLS = [
    "original_balance", "current_balance", "interest_rate", "loan_age_months",
    "remaining_term_months", "days_past_due", "reporting_lag_days",
]
CATEGORICAL_PROFILE_COLS = [
    "credit_score_band", "ltv_band", "dti_band", "state", "loan_purpose", "occupancy_type",
    "property_type", "servicer_name", "current_status", "document_status", "source_system",
    "loss_severity_band",
]


def profile_numeric(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    cols = cols or [c for c in NUMERIC_PROFILE_COLS if c in df.columns]
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        v = s.dropna()
        q = v.quantile([0.01, 0.25, 0.5, 0.75, 0.99]) if len(v) else pd.Series(dtype=float)
        iqr = (q.get(0.75, np.nan) - q.get(0.25, np.nan)) if len(v) else np.nan
        lo, hi = q.get(0.25, np.nan) - 3 * iqr, q.get(0.75, np.nan) + 3 * iqr
        rows.append({
            "column": c, "n": int(s.shape[0]), "missing": int(s.isna().sum()),
            "missing_pct": float(s.isna().mean()),
            "mean": float(v.mean()) if len(v) else np.nan,
            "std": float(v.std()) if len(v) else np.nan,
            "min": float(v.min()) if len(v) else np.nan,
            "p01": float(q.get(0.01, np.nan)), "p25": float(q.get(0.25, np.nan)),
            "median": float(q.get(0.5, np.nan)), "p75": float(q.get(0.75, np.nan)),
            "p99": float(q.get(0.99, np.nan)),
            "max": float(v.max()) if len(v) else np.nan,
            "skew": float(v.skew()) if len(v) > 2 else np.nan,
            "zeros": int((v == 0).sum()),
            "negatives": int((v < 0).sum()),
            "iqr_outliers": int(((v < lo) | (v > hi)).sum()) if len(v) else 0,
            "iqr_outlier_pct": float(((v < lo) | (v > hi)).mean()) if len(v) else 0.0,
            "distinct": int(v.nunique()),
        })
    return pd.DataFrame(rows)


def profile_categorical(df: pd.DataFrame, cols=None, top_k: int = 5) -> pd.DataFrame:
    cols = cols or [c for c in CATEGORICAL_PROFILE_COLS if c in df.columns]
    rows = []
    for c in cols:
        s = df[c]
        vc = s.value_counts(normalize=True)
        top = "; ".join(f"{k}={v:.3f}" for k, v in vc.head(top_k).items())
        counts = s.value_counts()
        p = counts / counts.sum() if counts.sum() else counts
        entropy = float(-(p * np.log(p.replace(0, np.nan))).sum()) if len(p) else np.nan
        rows.append({
            "column": c, "n": int(len(s)), "missing": int(s.isna().sum()),
            "missing_pct": float(s.isna().mean()), "distinct": int(s.nunique()),
            "mode": vc.index[0] if len(vc) else None,
            "mode_share": float(vc.iloc[0]) if len(vc) else np.nan,
            "normalised_entropy": entropy / np.log(len(p)) if len(p) > 1 else 0.0,
            "top_values": top,
        })
    return pd.DataFrame(rows)


def missingness_structure(df: pd.DataFrame, cols=None) -> dict:
    cols = cols or [c for c in NUMERIC_PROFILE_COLS + CATEGORICAL_PROFILE_COLS
                    if c in df.columns and df[c].isna().any()]
    ind = df[cols].isna().astype(int)
    co = ind.corr().fillna(0.0)

    by_servicer = df.groupby("servicer_name")[cols].apply(lambda g: g.isna().mean())
    by_month = df.groupby("reporting_month")[cols].apply(lambda g: g.isna().mean())

    tests = []
    for c in cols:
        tab = pd.crosstab(df["servicer_name"], df[c].isna())
        if tab.shape[1] == 2 and tab.to_numpy().min() >= 0:
            chi2, p, _, _ = stats.chi2_contingency(tab)
            n = tab.to_numpy().sum()
            cramers_v = float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))
            tests.append({"column": c, "chi2_vs_servicer": float(chi2), "p_value": float(p),
                          "cramers_v": cramers_v,
                          "verdict": "MAR (depends on servicer)" if p < 0.01 else "consistent with MCAR"})
    return {
        "co_missingness": co,
        "by_servicer": by_servicer,
        "by_month": by_month,
        "mechanism_tests": pd.DataFrame(tests).sort_values("cramers_v", ascending=False),
        "rows_with_any_missing": float(ind.any(axis=1).mean()),
        "mean_missing_fields_per_row": float(ind.sum(axis=1).mean()),
    }


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    tab = pd.crosstab(a, b)
    if tab.size == 0 or min(tab.shape) < 2:
        return np.nan
    chi2 = stats.chi2_contingency(tab)[0]
    n = tab.to_numpy().sum()
    phi2 = chi2 / n
    r, k = tab.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else np.nan


def dependency_analysis(df: pd.DataFrame) -> dict:
    num_cols = [c for c in NUMERIC_PROFILE_COLS if c in df.columns]
    numeric_corr = df[num_cols].corr(method="spearman")

    cat_cols = [c for c in CATEGORICAL_PROFILE_COLS if c in df.columns and df[c].nunique() < 40]
    pairs = []
    for i, a in enumerate(cat_cols):
        for b in cat_cols[i + 1:]:
            sub = df[[a, b]].dropna()
            if len(sub) < 500:
                continue
            v = cramers_v(sub[a], sub[b])
            if v is not None and not np.isnan(v):
                pairs.append({"field_a": a, "field_b": b, "cramers_v": v})
    assoc = pd.DataFrame(pairs).sort_values("cramers_v", ascending=False).reset_index(drop=True)

    fd = []
    for det, dep in [("loan_id", "origination_month"), ("loan_id", "credit_score_band"),
                     ("loan_id", "original_balance"), ("loan_id", "state"),
                     ("loan_id", "servicer_name"), ("current_status", "expected_dpd")]:
        if det not in df.columns or dep not in df.columns:
            continue
        g = df.dropna(subset=[dep]).groupby(det, observed=True)[dep].nunique()
        violating = int((g > 1).sum())
        fd.append({"determinant": det, "dependent": dep,
                   "groups": int(len(g)), "violating_groups": violating,
                   "holds": violating == 0,
                   "violation_rate": float(violating / max(len(g), 1))})
    return {"numeric_corr": numeric_corr, "categorical_association": assoc,
            "functional_dependencies": pd.DataFrame(fd)}


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    e, a = expected.dropna(), actual.dropna()
    if len(e) < 50 or len(a) < 50:
        return np.nan
    if pd.api.types.is_numeric_dtype(e):
        edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return 0.0
        e_pct = np.histogram(e, bins=edges)[0] / len(e)
        a_pct = np.histogram(a, bins=edges)[0] / len(a)
    else:
        cats = e.astype(str).value_counts().index
        e_pct = e.astype(str).value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
        a_pct = a.astype(str).value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
    e_pct = np.clip(e_pct, 1e-6, None)
    a_pct = np.clip(a_pct, 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def drift_report(df: pd.DataFrame, train_end: str = C.TRAIN_END,
                 cols=None) -> pd.DataFrame:
    cols = cols or [c for c in NUMERIC_PROFILE_COLS + CATEGORICAL_PROFILE_COLS
                    if c in df.columns]
    train = df[df["reporting_month"] <= train_end]
    test = df[df["reporting_month"] > train_end]
    rows = []
    for c in cols:
        val = psi(train[c], test[c])
        ks = np.nan
        if pd.api.types.is_numeric_dtype(df[c]):
            a, b = train[c].dropna(), test[c].dropna()
            if len(a) > 50 and len(b) > 50:
                ks = float(stats.ks_2samp(a, b).statistic)
        rows.append({
            "column": c, "psi": val, "ks_statistic": ks,
            "train_missing_pct": float(train[c].isna().mean()),
            "test_missing_pct": float(test[c].isna().mean()),
            "severity": ("severe" if (val or 0) >= 0.25 else
                         "moderate" if (val or 0) >= 0.10 else "stable"),
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def target_stability(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in C.BINARY_TARGETS + ["exception_required"]:
        if t not in df.columns:
            continue
        g = df.groupby("reporting_month")[t].mean()
        rows.append({"target": t, "overall_rate": float(df[t].mean(skipna=True)),
                     "min_month_rate": float(g.min()), "max_month_rate": float(g.max()),
                     "std_across_months": float(g.std()),
                     "censored_rows": int(df[t].isna().sum())})
    return pd.DataFrame(rows)
