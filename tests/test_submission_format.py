from __future__ import annotations

import pandas as pd
import pytest

from src import config as C

SUBMISSION = C.SUBMISSION / "submission.csv"

REQUIRED_ELEMENTS = {
    "probabilities": ["prob_delinquency_3m", "prob_delinquency_6m", "prob_default_12m",
                      "prob_prepayment_12m", "exception_probability"],
    "next state": ["predicted_next_state"],
    "exception type": ["predicted_exception_type"],
    "anomaly score": ["anomaly_score"],
    "top drivers": ["top_drivers_default_model"],
    "action": ["recommended_action"],
    "confidence": ["confidence"],
}

PROBABILITY_COLUMNS = REQUIRED_ELEMENTS["probabilities"] + ["next_state_confidence",
                                                           "exception_type_confidence"]


@pytest.fixture(scope="module")
def sub() -> pd.DataFrame:
    if not SUBMISSION.exists():
        pytest.skip("submission.csv not built yet; run the pipeline first")
    return pd.read_csv(SUBMISSION)


def test_every_required_element_is_present(sub):
    missing = {element: [c for c in cols if c not in sub.columns]
               for element, cols in REQUIRED_ELEMENTS.items()}
    missing = {k: v for k, v in missing.items() if v}
    assert not missing, f"PS section 6 elements with no column: {missing}"


def test_probabilities_are_in_range(sub):
    for col in PROBABILITY_COLUMNS:
        if col not in sub.columns:
            continue
        values = sub[col].dropna()
        assert ((values >= 0.0) & (values <= 1.0)).all(), f"{col} outside [0, 1]"


def test_one_row_per_loan(sub):
    assert not sub["loan_id"].duplicated().any(), "submission must carry one row per loan"


def test_no_null_ids_or_actions(sub):
    assert sub["loan_id"].notna().all()
    assert sub["recommended_action"].notna().all()


def test_predicted_states_are_in_the_declared_vocabulary(sub):
    unknown = set(sub["predicted_next_state"].dropna()) - set(C.STATES)
    assert not unknown, f"predicted_next_state outside declared vocabulary: {unknown}"


def test_predicted_exception_types_are_in_the_declared_vocabulary(sub):
    unknown = set(sub["predicted_exception_type"].dropna()) - set(C.EXCEPTION_TYPES)
    assert not unknown, f"predicted_exception_type outside declared vocabulary: {unknown}"


def test_exception_type_is_none_when_probability_is_below_threshold(sub):
    low = sub[sub["exception_probability"] < 0.50]
    assert (low["predicted_exception_type"] == "none").all()


def test_output_is_labelled_as_recommendation_not_decision(sub):
    assert "action_is_recommendation_not_decision" in sub.columns
    assert bool(sub["action_is_recommendation_not_decision"].all())


def test_documented_format_matches_the_file(sub):
    doc = (C.SUBMISSION / "SUBMISSION_FORMAT.md").read_text(encoding="utf-8")
    undocumented = [c for c in sub.columns if f"`{c}`" not in doc]
    assert not undocumented, f"columns absent from SUBMISSION_FORMAT.md: {undocumented}"
