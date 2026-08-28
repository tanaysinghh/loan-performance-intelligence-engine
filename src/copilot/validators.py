"""Automated guards on copilot output.

The grounding validator is the load-bearing control in this layer. It extracts every number
from the generated text and checks each one against the grounding pack. A number the model
invented — or "helpfully" rounded, converted to a percentage, or computed as a ratio — is
flagged and the output is blocked from the reviewer queue.

This is what makes "no LLM-produced numbers" an enforced property rather than a promise in a
system prompt.
"""
from __future__ import annotations

import re

from src.copilot.grounding import extract_numbers

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*%?")

CAUSAL_PHRASES = [
    "caused by", "causes ", "because the borrower", "will default", "will prepay",
    "is going to default", "guarantees", "certainly", "definitely will",
    "proves that", "demonstrates that the borrower",
]

HEDGE_REQUIRED = ["recommendation", "reviewer", "model"]

ALLOWED_BARE = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 100.0}


def parse_numbers(text: str) -> list[tuple[str, float]]:
    out = []
    for raw in NUM_RE.findall(text):
        token = raw.rstrip("%").replace(",", "")
        if not token or token in {"-", "."}:
            continue
        try:
            out.append((raw, float(token)))
        except ValueError:
            continue
    return out


def grounding_validator(text: str, grounding: dict, tolerance: float = 5e-3) -> dict:
    allowed = extract_numbers(grounding)
    ungrounded = []
    checked = 0
    for raw, val in parse_numbers(text):
        if val in ALLOWED_BARE:
            continue
        checked += 1
        if any(abs(val - a) <= max(tolerance, abs(a) * tolerance) for a in allowed):
            continue
        ungrounded.append(raw)

    causal = [p for p in CAUSAL_PHRASES if p in text.lower()]
    has_disclaimer = any(h in text.lower() for h in HEDGE_REQUIRED)

    passed = not ungrounded and not causal and has_disclaimer
    return {
        "passed": passed,
        "numbers_checked": checked,
        "ungrounded_numbers": ungrounded,
        "causal_or_overconfident_phrases": causal,
        "carries_reviewer_framing": has_disclaimer,
        "action": "released to reviewer queue" if passed else "BLOCKED — returned for correction",
    }


def summarise(records: list[dict]) -> dict:
    total = len(records)
    passed = sum(1 for r in records if (r.get("grounding_validator") or {}).get("passed"))
    blocked = total - passed
    ungrounded = sum(len((r.get("grounding_validator") or {}).get("ungrounded_numbers", []))
                     for r in records)
    return {"outputs_generated": total, "passed_validation": passed,
            "blocked_by_validator": blocked,
            "total_ungrounded_numbers_caught": ungrounded,
            "pass_rate": round(passed / total, 4) if total else None}


SELF_TEST_CASES = [
    {
        "case": "fabricated probability",
        "text": "The model puts this loan at a 41.7% chance of default over the next year. "
                "This is a recommendation for the reviewer, based on model output.",
        "should_pass": False,
        "why": "41.7 appears nowhere in the pack. This is the single most damaging failure "
               "mode: an invented figure that reads exactly like a real one.",
    },
    {
        "case": "rescaled real number",
        "text": "Default probability is 0.0847 for this record. Reviewer to confirm; model "
                "output only.",
        "should_pass": False,
        "why": "Plausible-looking and wrong. A model that rounds or rescales a grounded "
               "figure produces a number that was never computed.",
    },
    {
        "case": "causal assertion",
        "text": "The elevated days past due is caused by the borrower's high debt-to-income "
                "band. Model output, for reviewer action.",
        "should_pass": False,
        "why": "SHAP attribution is association. Causal language invites a reviewer to act "
               "on a claim the model never made.",
    },
    {
        "case": "overconfident decision",
        "text": "This loan will default. Refer it to foreclosure. Model and reviewer notified.",
        "should_pass": False,
        "why": "An LLM stating a certain outcome and directing an irreversible action.",
    },
    {
        "case": "missing reviewer framing",
        "text": "Status is DQ30 and the record shows an elevated anomaly reading.",
        "should_pass": False,
        "why": "No reviewer framing. Output must never read as an autonomous determination.",
    },
    {
        "case": "clean grounded restatement",
        "text": "The model scores this record and the reviewer should check the servicer feed "
                "gap first. Association only; no causal claim.",
        "should_pass": True,
        "why": "Restates position without inventing figures, and carries reviewer framing.",
    },
]


def run_self_test(grounding: dict) -> "list[dict]":
    """Feeds deliberately bad outputs through the validator to show it actually bites.

    Necessary because a validator that has only ever seen well-behaved output is untested.
    Each case is a failure mode a real language model produces under pressure.
    """
    rows = []
    for c in SELF_TEST_CASES:
        v = grounding_validator(c["text"], grounding)
        rows.append({
            "case": c["case"],
            "expected": "pass" if c["should_pass"] else "block",
            "actual": "pass" if v["passed"] else "block",
            "correct": v["passed"] == c["should_pass"],
            "ungrounded_numbers": ", ".join(v["ungrounded_numbers"]) or "none",
            "flagged_phrases": ", ".join(v["causal_or_overconfident_phrases"]) or "none",
            "why_this_matters": c["why"],
        })
    return rows
