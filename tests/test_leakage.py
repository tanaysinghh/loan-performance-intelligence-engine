"""Leakage guards. These are the tests that would catch the failures that matter most."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.features import build_features as F
from src.features.dataset import prepare
from src.models.splits import HORIZONS, purged_time_split


@pytest.fixture(scope="module")
def df():
    return prepare()


def test_banned_features_are_refused():
    for bad in ("default_flag", "prepayment_flag", "next_state", "loss_severity_band",
                "next_12m_default_flag"):
        with pytest.raises(ValueError):
            F.assert_no_leakage(["credit_ord", bad])


def test_design_matrix_contains_no_target_or_forward_column(df):
    cols = F.feature_columns(df)
    F.assert_no_leakage(cols)
    assert not set(cols) & set(C.ALL_TARGETS)
    assert not any(c.startswith("next_") for c in cols)
    assert "loss_severity_band" not in cols


def test_loss_severity_is_excluded_because_it_encodes_the_outcome(df):
    """loss_severity_band is populated only after default, so its presence is the label."""
    populated = df["loss_severity_band"].notna()
    if populated.sum() > 0:
        assert df.loc[populated, "next_state"].eq("Default").mean() > 0.9
    assert "loss_severity_band" not in F.feature_columns(df)


def test_training_labels_never_reach_into_the_test_window(df):
    """The embargo must guarantee no training row's outcome window touches the test window."""
    for target, horizon in HORIZONS.items():
        if target not in C.BINARY_TARGETS:
            continue
        s = purged_time_split(df, target)
        if not s.train.any() or not s.test.any():
            continue
        max_train_month = int(df.loc[s.train, "month_index"].max())
        min_test_month = int(df.loc[s.test, "month_index"].min())
        assert max_train_month + horizon < min_test_month, (
            f"{target}: training row at month {max_train_month} with horizon {horizon} "
            f"reaches month {max_train_month + horizon}, but the test window opens at "
            f"{min_test_month}")


def test_no_row_has_an_unobservable_label_in_any_split(df):
    """A horizon-H label is only usable if the panel actually contains H more months."""
    last = int(df["month_index"].max())
    for target, horizon in HORIZONS.items():
        if target not in C.BINARY_TARGETS:
            continue
        s = purged_time_split(df, target)
        for mask in (s.train, s.valid, s.test):
            if not mask.any():
                continue
            assert int(df.loc[mask, "month_index"].max()) <= last - horizon


def test_splits_do_not_overlap(df):
    for target in C.BINARY_TARGETS:
        s = purged_time_split(df, target)
        assert not (s.train & s.valid).any()
        assert not (s.train & s.test).any()
        assert not (s.valid & s.test).any()


def test_censored_rows_are_excluded_rather_than_treated_as_negatives(df):
    for target in C.BINARY_TARGETS:
        s = purged_time_split(df, target)
        for mask in (s.train, s.valid, s.test):
            assert df.loc[mask, target].notna().all()


def test_history_features_use_only_past_information(df):
    """A lagged feature at month t must equal the base feature at an earlier month."""
    d = df.sort_values(["loan_id", "month_index"], kind="mergesort")
    g = d.groupby("loan_id", sort=False)
    recomputed = g["status_ord"].shift(1)
    both = d["status_ord_lag1"].notna() & recomputed.notna()
    assert both.sum() > 1000
    assert np.allclose(d.loc[both, "status_ord_lag1"], recomputed[both])


def test_rolling_windows_are_backward_looking(df):
    d = df.sort_values(["loan_id", "month_index"], kind="mergesort")
    g = d.groupby("loan_id", sort=False)
    manual = g["days_past_due_clean"].transform(lambda s: s.rolling(6, min_periods=1).max())
    both = d["max_dpd_last_6m"].notna() & manual.notna()
    assert np.allclose(d.loc[both, "max_dpd_last_6m"], manual[both])
