"""Time-aware, horizon-purged splitting.

Two failures make naive splitting of a loan panel look far better than it is.

**Unobservable labels.** A 12-month default label on a row dated three months before the
panel ends can only ever be a 1 if the default already happened; a genuine 0 is unobservable.
Keeping such rows makes the evaluation window a sample of loans that terminated. Every split
here is therefore capped at `usable_max = last_month - horizon`, so only rows whose full
outcome window is observed are eligible for training or evaluation.

**Window overlap.** A training row dated month t carries information about months t+1..t+H.
If any of those months fall inside the evaluation window, the same loan's future has been
seen during training even though the row is chronologically earlier. An embargo of H months
is therefore dropped between the fitting data and the test window.

Layout produced for horizon H, with `U = last_month - H`:

    [ train .......... | valid (6m) ] [ embargo (H months, dropped) ] [ test (6m) ]
                                                                      ends at U

Train and validation are contiguous with no embargo between them. That is deliberate: the
validation window drives early stopping and the isotonic calibration map only, and both are
subsequently assessed out-of-time on the purged test window, which is the number reported.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config as C

HORIZONS = {
    "next_3m_delinquency_flag": 3,
    "next_6m_delinquency_flag": 6,
    "next_12m_default_flag": 12,
    "next_12m_prepayment_flag": 12,
    "next_state": 1,
    "exception_required": 0,
    "exception_type": 0,
}

N_TEST_MONTHS = 6
N_VALID_MONTHS = 6


@dataclass
class Split:
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray
    horizon: int
    embargo_rows: int
    unobservable_rows: int
    windows: dict

    def describe(self) -> dict:
        return {"train_rows": int(self.train.sum()), "valid_rows": int(self.valid.sum()),
                "test_rows": int(self.test.sum()), "horizon_months": self.horizon,
                "rows_dropped_embargo": self.embargo_rows,
                "rows_dropped_unobservable_label": self.unobservable_rows,
                **self.windows}


def _month_label(df: pd.DataFrame, mi: int) -> str:
    hit = df.loc[df["month_index"] == mi, "reporting_month"]
    return str(hit.iloc[0]) if len(hit) else ""


def purged_time_split(df: pd.DataFrame, target: str, n_test: int = N_TEST_MONTHS,
                      n_valid: int = N_VALID_MONTHS,
                      train_end_index: int | None = None) -> Split:
    horizon = HORIZONS[target]
    mi = df["month_index"].to_numpy()
    last = int(mi.max())
    usable_max = last - horizon

    labelled = df[target].notna().to_numpy()
    observable = mi <= usable_max
    unobservable_rows = int((labelled & ~observable).sum())

    test_start = usable_max - n_test + 1
    embargo_start = test_start - horizon
    valid_end = embargo_start - 1
    valid_start = valid_end - n_valid + 1
    train_end = valid_start - 1
    if train_end_index is not None:
        train_end = min(train_end, train_end_index)
        valid_start = min(valid_start, train_end + 1)

    eligible = labelled & observable
    train = eligible & (mi <= train_end)
    valid = eligible & (mi >= valid_start) & (mi <= valid_end)
    test = eligible & (mi >= test_start) & (mi <= usable_max)
    embargo_rows = int((eligible & (mi >= embargo_start) & (mi < test_start)).sum())

    windows = {
        "train_window": f"{_month_label(df, 0)}..{_month_label(df, train_end)}",
        "valid_window": f"{_month_label(df, valid_start)}..{_month_label(df, valid_end)}",
        "embargo_window": (f"{_month_label(df, embargo_start)}..{_month_label(df, test_start - 1)}"
                           if horizon > 0 else "none"),
        "test_window": f"{_month_label(df, test_start)}..{_month_label(df, usable_max)}",
    }
    return Split(train=train, valid=valid, test=test, horizon=horizon,
                 embargo_rows=embargo_rows, unobservable_rows=unobservable_rows,
                 windows=windows)


def loan_disjoint_time_split(df: pd.DataFrame, target: str, seed: int = C.RANDOM_SEED,
                             holdout_share: float = 0.35, **kwargs) -> Split:
    """Stricter variant: no loan_id appears in both the fitting data and the test window.

    Used as a memorisation probe. If test performance holds up under loan disjointness, the
    model is learning loan characteristics rather than individual loan identities.
    """
    base = purged_time_split(df, target, **kwargs)
    rng = np.random.default_rng(seed)
    unique_loans = np.unique(df["loan_id"].to_numpy())
    holdout = set(rng.permutation(unique_loans)[: int(holdout_share * len(unique_loans))])
    in_holdout = df["loan_id"].isin(holdout).to_numpy()

    train = base.train & ~in_holdout
    valid = base.valid & ~in_holdout
    test = base.test & in_holdout
    return Split(train=train, valid=valid, test=test, horizon=base.horizon,
                 embargo_rows=base.embargo_rows + int((base.train & in_holdout).sum()),
                 unobservable_rows=base.unobservable_rows,
                 windows={**base.windows, "test_window": base.windows["test_window"] + " (held-out loans only)"})


def random_row_split(df: pd.DataFrame, target: str, seed: int = C.RANDOM_SEED) -> Split:
    """Deliberately unsound split kept only as a leakage control in the report."""
    rng = np.random.default_rng(seed)
    labelled = df[target].notna().to_numpy()
    r = rng.random(len(df))
    return Split(train=labelled & (r < 0.70), valid=labelled & (r >= 0.70) & (r < 0.85),
                 test=labelled & (r >= 0.85), horizon=0, embargo_rows=0,
                 unobservable_rows=0,
                 windows={"train_window": "random rows", "valid_window": "random rows",
                          "embargo_window": "none", "test_window": "random rows"})


def expanding_window_folds(df: pd.DataFrame, target: str, n_folds: int = 3) -> list[Split]:
    """Expanding-window backtest. Each fold keeps the same purge and embargo rules."""
    base = purged_time_split(df, target)
    full_train_end = int(np.max(np.where(base.train)[0])) if base.train.any() else 0
    train_end_month = int(df["month_index"].to_numpy()[base.train].max()) if base.train.any() else 0
    folds = []
    for k in range(n_folds):
        cut = train_end_month - (n_folds - 1 - k) * 6
        folds.append(purged_time_split(df, target, train_end_index=cut))
    return folds


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in C.BINARY_TARGETS + ["exception_required"]:
        s = purged_time_split(df, t)
        d = s.describe()
        d["target"] = t
        for name, mask in (("train", s.train), ("valid", s.valid), ("test", s.test)):
            d[f"{name}_positive_rate"] = (float(df.loc[mask, t].mean())
                                          if mask.sum() else float("nan"))
        d["train_loans"] = int(df.loc[s.train, "loan_id"].nunique())
        d["test_loans"] = int(df.loc[s.test, "loan_id"].nunique())
        d["loan_overlap_train_test"] = int(len(
            set(df.loc[s.train, "loan_id"]) & set(df.loc[s.test, "loan_id"])))
        rows.append(d)
    cols = ["target", "horizon_months", "train_window", "valid_window", "embargo_window",
            "test_window", "train_rows", "valid_rows", "test_rows", "rows_dropped_embargo",
            "rows_dropped_unobservable_label", "train_positive_rate", "valid_positive_rate",
            "test_positive_rate", "train_loans", "test_loans", "loan_overlap_train_test"]
    return pd.DataFrame(rows)[cols]
