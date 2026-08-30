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

from src.copilot.grounding import NUMBER_TOKEN_RE, extract_numbers

# The number tokenizer lives in grounding.py and is imported here so both sides of the
# comparison agree by construction. Three live-run defects are pinned in that one pattern:
# scientific notation split into two tokens (`-2e-05` -> `-2`, `-05`), hyphens in field names
# read as minus signs (`next-3m-delinquency` -> `-3`), and credit-band labels tokenized
# differently on each side (`580-619` -> `-619` there, `619` here). All three flagged correct
# Gemini output. A validator that cries wolf trains a reviewer to ignore it.
NUM_RE = NUMBER_TOKEN_RE

# ---------------------------------------------------------------------------------------
# LaTeX markup.
#
# A live Gemini run rendered every scientific-notation figure in the portfolio summary as
# MathJax — `$-2\times 10^{-5}$` instead of `-2e-05`. Two things were wrong with that, and
# they need separating because they have different fixes.
#
#   1. It is a genuine model failure. These notes are read in a plain-text servicing queue
#      that renders no markup, so the reviewer sees the raw `$...$` source. Rule 7 of the
#      system prompt now forbids it, and it is blocked and sent back for correction here.
#   2. It also broke the number check. `10^{-5}` tokenizes as `10` and `-5`, so a figure the
#      model had copied correctly out of the pack was additionally reported as ungrounded.
#      That is a false accusation layered on top of a real defect, and it made the real
#      defect harder to see.
#
# So the text is normalised back to plain notation *before* the numbers are extracted — the
# grounding check then judges the figures the model actually meant — while the presence of
# the markup is reported separately as its own named finding. The output is still blocked;
# it is blocked for the right reason, with the right evidence.
# ---------------------------------------------------------------------------------------

LATEX_MARKERS = [r"\times", r"\frac", r"\cdot", r"\text{", r"^{", "$$"]

# `-2 \times 10^{-5}`  ->  `-2e-05`
_LATEX_SCI_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*\\?times\s*10\s*\^\s*\{?\s*(-?\d+)\s*\}?")
# bare `10^{-5}` with no mantissa
_LATEX_POW_RE = re.compile(r"(?<![\d.])10\s*\^\s*\{?\s*(-?\d+)\s*\}?")


def has_latex_markup(text: str) -> bool:
    if any(m in text for m in LATEX_MARKERS):
        return True
    # a `$...$` span containing a digit is MathJax, not a dollar amount
    return bool(re.search(r"\$[^$\n]*\d[^$\n]*\$", text))


def normalise_latex(text: str) -> str:
    """Rewrite LaTeX numeric forms to the plain notation used in the grounding pack.

    Used only to make the number comparison honest. It does not repair the output — the
    reviewer-facing text is still rejected and regenerated.
    """
    out = _LATEX_SCI_RE.sub(lambda m: f"{m.group(1)}e{int(m.group(2)):+03d}", text)
    out = _LATEX_POW_RE.sub(lambda m: f"1e{int(m.group(1)):+03d}", out)
    out = out.replace("\\times", " ").replace("\\cdot", " ")
    return out.replace("$", "")


CAUSAL_PHRASES = [
    "caused by", "causes ", "because the borrower", "will default", "will prepay",
    "is going to default", "guarantees", "certainly", "definitely will",
    "proves that", "demonstrates that the borrower",
]

HEDGE_REQUIRED = ["recommendation", "reviewer", "model"]

# A refusal is the correct behaviour on an out-of-scope question, and it is definitionally
# not an autonomous determination. The framing check exists to stop output reading as a
# decision; blocking "the pack does not contain that" punished the model for doing exactly
# what the system prompt demands. Recognising refusals narrows the check to its purpose
# rather than weakening it — a refusal asserts nothing to defer on.
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
    # judge the figures the model meant, not the markup it wrapped them in
    scanned = normalise_latex(text) if latex else text
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


# ---------------------------------------------------------------------------------------
# Usefulness check.
#
# The grounding validator is a *truthfulness* control: it stops the model saying something
# false. It has nothing to say about output that is entirely true and entirely useless, and
# the first live Gemini run produced exactly that — a reviewer note whose "check this first"
# instruction was to verify a data-quality score the same pack reported as 100.0 and a
# document status the same pack reported as complete. Every guard passed. The note told a
# reviewer to go and look at nothing.
#
# That is the failure mode the `invites_a_vague_non_answer` probe was written for, and it
# turned up in production output rather than under the probe. So it gets its own control.
# ---------------------------------------------------------------------------------------

ACTION_VERBS = ["check", "verify", "confirm", "review", "escalate", "contact", "investigate",
                "reconcile", "prioritise", "prioritize", "flag", "examine"]

# Fields whose benign values make an instruction to inspect them a null instruction.
BENIGN_WHEN = {
    "data_quality_score": lambda v: isinstance(v, (int, float)) and v >= 95,
    "document_status": lambda v: str(v).strip().lower() in {"complete", "completed"},
    "modification_flag": lambda v: v in (0, 0.0, "0"),
}


def usefulness_validator(text: str, grounding: dict) -> dict:
    """Flags reviewer-facing output that is true but directs the reviewer at nothing.

    Deliberately narrow. It fires only when the text steers the reviewer at a named field
    that the grounding pack itself reports as clean — a claim the pack can settle, not a
    matter of taste. Vagueness in general is not mechanically detectable and is not claimed
    to be; this catches the one form of it that is.
    """
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
        # only a null instruction if the text actually directs an action at it
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
    """Truthfulness and usefulness together. Both must hold for release.

    Kept as a separate entry point so `grounding_validator` — the load-bearing, self-tested
    control — is unchanged and independently exercised.
    """
    g = grounding_validator(text, grounding)
    if "record" not in (grounding or {}):
        return g  # usefulness is defined for per-record reviewer notes only
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
