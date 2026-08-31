from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from src import config as C
from src.copilot.grounding import loan_pack
from src.copilot.validators import SELF_TEST_CASES, grounding_validator, run_self_test

MODELLING_PACKAGES = ["src/data", "src/features", "src/models", "src/scenarios", "src/explain"]
LLM_CLIENT_ROOTS = {"anthropic", "openai", "google", "cohere", "mistralai", "ollama"}


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_no_modelling_module_can_reach_a_language_model():
    offenders = []
    for pkg in MODELLING_PACKAGES:
        for path in (C.ROOT / pkg).rglob("*.py"):
            for mod in _imported_modules(path):
                root = mod.split(".")[0]
                if root in LLM_CLIENT_ROOTS or mod.startswith("src.copilot"):
                    offenders.append(f"{path.relative_to(C.ROOT)} imports {mod}")
    assert not offenders, "Modelling code must not import an LLM client: " + "; ".join(offenders)


def test_submission_columns_all_originate_from_non_llm_models():
    path = C.SUBMISSION / "submission.csv"
    if not path.exists():
        pytest.skip("submission.csv not built yet")
    sub = pd.read_csv(path)
    numeric = [c for c in sub.columns if pd.api.types.is_numeric_dtype(sub[c])]
    assert {"prob_delinquency_3m", "prob_default_12m", "prob_prepayment_12m",
            "exception_probability", "anomaly_score"} <= set(numeric)
    for col in ("prob_delinquency_3m", "prob_delinquency_6m", "prob_default_12m",
                "prob_prepayment_12m", "exception_probability", "anomaly_score",
                "next_state_confidence"):
        assert sub[col].between(0, 1).all(), f"{col} outside [0, 1]"
    assert sub["action_is_recommendation_not_decision"].all()


def test_grounding_validator_blocks_fabricated_numbers():
    pack = {"model_predictions": {"prob_default_12m": 0.1234}}
    bad = "The model gives a 41.7% chance of default. Reviewer to confirm; model output."
    verdict = grounding_validator(bad, pack)
    assert not verdict["passed"]
    assert "41.7%" in verdict["ungrounded_numbers"]


def test_grounding_validator_allows_grounded_numbers():
    pack = {"model_predictions": {"prob_default_12m": 0.1234}}
    good = ("The model scores this record at 0.1234 for twelve-month default. "
            "This is a recommendation for the reviewer.")
    assert grounding_validator(good, pack)["passed"]


def test_grounding_validator_blocks_causal_and_overconfident_language():
    pack = {"model_predictions": {"p": 0.5}}
    for text in ("This loan will default. Model output for the reviewer.",
                 "Delinquency is caused by the high LTV band. Model output, reviewer to act."):
        assert not grounding_validator(text, pack)["passed"]


def test_validator_self_test_behaves_as_specified():
    pack = {"model_predictions": {"prob_default_12m": 0.1234, "prob_delinquency_3m": 0.55}}
    rows = run_self_test(pack)
    assert len(rows) == len(SELF_TEST_CASES)
    failures = [r["case"] for r in rows if not r["correct"]]
    assert not failures, f"validator self-test regressions: {failures}"


def test_every_prompt_log_entry_is_complete():
    path = C.SUBMISSION / "llm_prompt_log.jsonl"
    if not path.exists():
        pytest.skip("prompt log not generated yet")
    required = {"timestamp_utc", "task", "mode", "model", "system_prompt", "user_prompt",
                "response", "grounding_validator", "disclaimer"}
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert lines
    for rec in lines:
        assert required <= set(rec), f"prompt log entry missing {required - set(rec)}"
        assert "RECOMMENDATION, NOT DECISION" in rec["disclaimer"]
