from __future__ import annotations

import re

from src.copilot.grounding import NUMBER_TOKEN_RE, extract_numbers

NUM_RE = NUMBER_TOKEN_RE


LATEX_MARKERS = [r"\times", r"\frac", r"\cdot", r"\text{", r"^{", "$$"]

_LATEX_SCI_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*\\?times\s*10\s*\^\s*\{?\s*(-?\d+)\s*\}?")
_LATEX_POW_RE = re.compile(r"(?<![\d.])10\s*\^\s*\{?\s*(-?\d+)\s*\}?")


_LIST_MARKER_RE = re.compile(r"(?m)^([ \t]{0,8})\d{1,3}[.)](?=\s)")


def strip_list_markers(text: str) -> str:
    return _LIST_MARKER_RE.sub(r"\1", text)


def has_latex_markup(text: str) -> bool:
    if any(m in text for m in LATEX_MARKERS):
        return True
    return bool(re.search(r"\$[^$\n]*\d[^$\n]*\$", text))


def normalise_latex(text: str) -> str:
    out = _LATEX_SCI_RE.sub(lambda m: f"{m.group(1)}e{int(m.group(2)):+03d}", text)
    out = _LATEX_POW_RE.sub(lambda m: f"1e{int(m.group(1)):+03d}", out)
    out = out.replace("\\times", " ").replace("\\cdot", " ")
    return out.replace("$", "")


CAUSAL_PHRASES = [
    "caused by", "causes ", "because the borrower", "will default", "will prepay",
    "is going to default", "guarantees", "certainly", "definitely will",
    "proves that", "demonstrates that the borrower",
]

HEDGE_REQUIRED = ["recommendation", "reviewer", "review", "model", "draft"]

REFUSAL_MARKERS = [
    "does not contain", "is not in the grounding pack", "not present in the pack",
    "cannot answer", "no such figure", "not available in the pack",
    "the pack does not", "grounding pack does not",
]

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
    latex = has_latex_markup(text)
    scanned = normalise_latex(text) if latex else text
    scanned = strip_list_markers(scanned)
    ungrounded = []
    checked = 0
    for raw, val in parse_numbers(scanned):
        if val in ALLOWED_BARE:
            continue
        checked += 1
        if any(abs(val - a) <= max(tolerance, abs(a) * tolerance) for a in allowed):
            continue
        ungrounded.append(raw)

    low = text.lower()
    causal = [p for p in CAUSAL_PHRASES if p in low]
    is_refusal = any(m in low for m in REFUSAL_MARKERS)
    has_disclaimer = any(h in low for h in HEDGE_REQUIRED) or is_refusal

    passed = not ungrounded and not causal and has_disclaimer and not latex
    return {
        "passed": passed,
        "numbers_checked": checked,
        "ungrounded_numbers": ungrounded,
        "causal_or_overconfident_phrases": causal,
        "carries_reviewer_framing": has_disclaimer,
        "framing_satisfied_by_refusal": is_refusal,
        "contains_latex_markup": latex,
        "action": ("released to reviewer queue" if passed else
                   "BLOCKED — LaTeX markup in plain-text output; returned for correction"
                   if latex and not ungrounded and not causal and has_disclaimer else
                   "BLOCKED — returned for correction"),
    }


def summarise(records: list[dict]) -> dict:
    total = len(records)
    passed = sum(1 for r in records if (r.get("grounding_validator") or {}).get("passed"))
    blocked = total - passed
    ungrounded = sum(len((r.get("grounding_validator") or {}).get("ungrounded_numbers", []))
                     for r in records)
    null_advice = sum(
        1 for r in records
        if ((r.get("grounding_validator") or {}).get("usefulness") or {}).get(
            "null_advice_targets"))
    return {"outputs_generated": total, "passed_validation": passed,
            "blocked_by_validator": blocked,
            "total_ungrounded_numbers_caught": ungrounded,
            "blocked_for_null_advice": null_advice,
            "pass_rate": round(passed / total, 4) if total else None}


ACTION_VERBS = ["check", "verify", "confirm", "review", "escalate", "contact", "investigate",
                "reconcile", "prioritise", "prioritize", "flag", "examine"]

BENIGN_WHEN = {
    "data_quality_score": lambda v: isinstance(v, (int, float)) and v >= 95,
    "document_status": lambda v: str(v).strip().lower() in {"complete", "completed"},
    "modification_flag": lambda v: v in (0, 0.0, "0"),
}


def usefulness_validator(text: str, grounding: dict) -> dict:
    record = (grounding or {}).get("record") or {}
    low = text.lower()
    null_targets = []
    for field, is_benign in BENIGN_WHEN.items():
        if field not in record:
            continue
        value = record.get(field)
        if value is None or not is_benign(value):
            continue
        readable = field.replace("_", " ")
        if field not in low and readable not in low:
            continue
        window_hit = any(
            verb in low[max(0, idx - 120):idx]
            for key in (field, readable) if key in low
            for idx in [low.index(key)]
            for verb in ACTION_VERBS)
        if window_hit:
            null_targets.append(f"{field}={value}")

    has_action = any(v in low for v in ACTION_VERBS)
    passed = not null_targets and has_action
    return {
        "passed": passed,
        "null_advice_targets": null_targets,
        "contains_an_action": has_action,
        "action": ("useful" if passed else
                   "BLOCKED — true but directs the reviewer at nothing actionable"),
    }


def combined_validator(text: str, grounding: dict) -> dict:
    g = grounding_validator(text, grounding)
    if "record" not in (grounding or {}):
        return g
    u = usefulness_validator(text, grounding)
    merged = dict(g)
    merged["usefulness"] = u
    merged["passed"] = bool(g["passed"] and u["passed"])
    if not u["passed"]:
        merged["action"] = ("BLOCKED — returned for correction"
                            if g["passed"] else merged["action"])
    return merged


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
        "case": "grounded figure in scientific notation",
        "text": "The adverse-credit delta on twelve-month default is -1e-05 against base. "
                "Model output for the reviewer.",
        "should_pass": True,
        "why": "Caught in the first live Gemini run: the model quoted `-2e-05` verbatim from "
               "the pack and the validator split it into `-2` and `-05`, blocking correct "
               "output. A validator that cries wolf gets ignored.",
    },
    {
        "case": "LaTeX markup in plain-text reviewer prose",
        "text": "The adverse-credit delta is $-2 \\times 10^{-5}$ against base. "
                "Recommendation for the reviewer.",
        "should_pass": False,
        "why": "Caught in a live Gemini run. The servicing queue renders no markup, so the "
               "reviewer sees raw MathJax source. Blocked as a formatting defect — but the "
               "figure inside it is normalised first, so it is not additionally mis-reported "
               "as an ungrounded number.",
    },
    {
        "case": "hyphenated field name read as a negative number",
        "text": "The model projects a next-3m-delinquency probability of 0.1234. "
                "Recommendation for the reviewer.",
        "should_pass": True,
        "why": "Caught in a live Gemini run: `next-3m-delinquency` was parsed as the number "
               "-3 and blocked. Hyphens in field names are not minus signs.",
    },
    {
        "case": "numbered list markers read as figures",
        "text": "Draft candidate rules for human review:\n"
                "15. Flag balance increases on non-modified loans.\n"
                "17. Flag sentinel values in days past due.",
        "should_pass": True,
        "why": "Caught in a live Gemini run on the rule-suggestion task, whose output is "
               "inherently a numbered list. `15.` and `17.` were parsed as figures and "
               "reported as ungrounded.",
    },
    {
        "case": "framing expressed as 'human review' rather than 'reviewer'",
        "text": "Three draft candidate rules are proposed for human review before "
                "implementation.",
        "should_pass": True,
        "why": "Also caught live. The framing could not have been clearer, and it was blocked "
               "for using the word `review` instead of `reviewer`.",
    },
    {
        "case": "correct refusal on an out-of-scope question",
        "text": "The grounding pack does not contain the borrower's employment status or "
                "monthly income.",
        "should_pass": True,
        "why": "Also caught live. Refusing is the specified behaviour, and a refusal asserts "
               "nothing a reviewer could act on, so demanding hedge vocabulary from it "
               "penalised the model for being right.",
    },
    {
        "case": "clean grounded restatement",
        "text": "The model scores this record and the reviewer should check the servicer feed "
                "gap first. Association only; no causal claim.",
        "should_pass": True,
        "why": "Restates position without inventing figures, and carries reviewer framing.",
    },
]


SELF_TEST_PACK = {
    "record": {
        "loan_id": "SELFTEST0001",
        "credit_score_band": "700-739",
        "document_status": "incomplete",
        "data_quality_score": 61.0,
    },
    "model_predictions": {
        "prob_default_12m": 0.1234,
        "prob_delinquency_3m": 0.55,
    },
    "scenario_projections": [
        {"delta_adverse_credit": -1e-05, "delta_high_prepayment": -2e-05},
    ],
}


def run_self_test(grounding: dict | None = None) -> "list[dict]":
    grounding = SELF_TEST_PACK if grounding is None else grounding
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
