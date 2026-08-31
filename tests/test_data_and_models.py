from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.data import loaders, profiling, validate
from src.features.dataset import prepare
from src.models import metrics as M
from src.models import survival as S
from src.models.splits import purged_time_split


@pytest.fixture(scope="module")
def df():
    return prepare()


def test_data_pack_exists_and_has_expected_schema():
    panel = loaders.load_panel()
    for col in C.RAW_COLUMNS:
        assert col in panel.columns, f"missing raw column {col}"
    for col in C.BINARY_TARGETS + ["next_state", "exception_required", "exception_type"]:
        assert col in panel.columns


def test_loan_month_key_is_unique_after_cleaning(df):
    assert not df.duplicated(subset=["loan_id", "reporting_month"]).any()


def test_cleaning_masks_sentinels_rather_than_imputing_them(df):
    assert not df["days_past_due_clean"].isin([9999, -1]).any()
    assert not df["interest_rate_clean"].isin([99.99, 0.0, -1.0]).any()
    assert df["dpd_repaired"].sum() > 0
    assert df.loc[df["dpd_repaired"] == 1, "days_past_due_clean"].isna().all()


def test_validation_rules_detect_the_injected_defects(df):
    flagged, summary = validate.run_rules(df)
    fired = summary[summary["violations"] > 0]["rule"].tolist()
    for expected in ("origination_after_reporting", "negative_balance",
                     "dpd_sentinel_value", "missing_critical_field",
                     "servicer_balance_break"):
        assert expected in fired, f"rule {expected} never fires"


def test_dq_scores_are_bounded_and_discriminating(df):
    flagged, _ = validate.run_rules(df)
    scored = validate.score_records(flagged)
    assert scored["dq_score"].between(0, 100).all()
    assert scored["dq_score"].std() > 1.0
    clean = scored.loc[scored["dq_violation_count"] == 0, "dq_score"]
    dirty = scored.loc[scored["dq_violation_count"] >= 2, "dq_score"]
    if len(dirty):
        assert clean.mean() > dirty.mean()


def test_missingness_is_not_random_with_respect_to_servicer(df):
    result = profiling.missingness_structure(df)
    tests = result["mechanism_tests"]
    assert (tests["verdict"] == "MAR (depends on servicer)").any()


def test_psi_is_zero_for_an_identical_distribution():
    s = pd.Series(np.random.default_rng(0).normal(size=5000))
    assert profiling.psi(s, s.copy()) == pytest.approx(0.0, abs=1e-6)


def test_psi_detects_a_shifted_distribution():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 5000))
    b = pd.Series(rng.normal(2, 1, 5000))
    assert profiling.psi(a, b) > 0.25


def test_binary_metrics_on_a_perfect_and_a_random_classifier():
    y = np.array([0] * 500 + [1] * 100)
    perfect = y.astype(float) * 0.98 + 0.01
    assert M.binary_metrics(y, perfect)["roc_auc"] == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    assert 0.4 < M.binary_metrics(y, rng.random(len(y)))["roc_auc"] < 0.6


def test_recall_at_precision_is_monotone_in_the_precision_target():
    rng = np.random.default_rng(2)
    y = (rng.random(4000) < 0.15).astype(int)
    p = np.clip(y * 0.4 + rng.random(4000) * 0.6, 0, 1)
    r30 = M.recall_at_precision(y, p, 0.30)["recall"]
    r50 = M.recall_at_precision(y, p, 0.50)["recall"]
    assert r30 >= r50


def test_survival_frame_censoring_is_internally_consistent(df):
    surv = S.build_survival_frame(df)
    assert (surv["exit_age"] > surv["entry_age"]).all()
    assert not (surv["event_default"] & surv["event_prepay"]).any()
    assert surv["left_truncated"].sum() > 0
    censored = surv["event_type"] == "censored"
    assert surv.loc[censored, "event_default"].sum() == 0


def test_transition_matrix_rows_sum_to_one(df):
    P = S.transition_matrix(df)
    assert np.allclose(P.sum(axis=1), 1.0)
    for absorbing in ("Default", "Prepaid"):
        assert P.loc[absorbing, absorbing] == pytest.approx(1.0)


def test_markov_projection_is_a_valid_distribution(df):
    P = S.transition_matrix(df)
    proj = S.project_states(P, horizon=12)
    prob_cols = [c for c in proj.columns if c.startswith("p_")]
    assert np.allclose(proj[prob_cols].sum(axis=1), 1.0)
    assert (proj[prob_cols] >= -1e-9).all().all()


def test_cumulative_incidence_never_exceeds_the_naive_km_curve(df):
    surv = S.build_survival_frame(df)
    cif = S.cumulative_incidence(surv)
    if len(cif):
        assert (cif["km_overstatement"] >= -1e-6).all()
        assert (cif["cif_default"] + cif["cif_prepay"] <= 1.0 + 1e-6).all()


def test_calibration_table_buckets_cover_every_record():
    rng = np.random.default_rng(3)
    y = (rng.random(2000) < 0.2).astype(int)
    p = rng.random(2000)
    t = M.calibration_table(y, p)
    assert t["n"].sum() == len(y)
