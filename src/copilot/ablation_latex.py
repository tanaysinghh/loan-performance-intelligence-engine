"""Ablation: does system-prompt rule 7 actually stop Gemini emitting LaTeX?

A live run produced a portfolio summary in which every scientific-notation figure was
rendered as MathJax — `$-2 \\times 10^{-5}$` instead of `-2e-05`. Rule 7 was added to the
system prompt in response, and the next run came back clean. One clean run is not evidence
that the rule did anything; the model is sampled, and it might simply not have reached for
LaTeX that time.

So this runs the same grounding pack twice against the same model at the same temperature,
once with rule 7 present and once with it stripped, and records whether the markup comes
back. Both calls are logged to the prompt log like any other. The result is written to
`reports/copilot_latex_ablation.csv`.

Run: python -m src.copilot.ablation_latex
"""
from __future__ import annotations

import pandas as pd

from src import config as C
from src.copilot import client as CL
from src.copilot import grounding as G
from src.copilot.validators import grounding_validator, has_latex_markup

PROMPT = ("Write a portfolio scenario summary for a credit committee. State what each "
          "scenario projects, which segments move most, and what the model quality figures "
          "imply about how much weight to place on these projections.")


def _pack() -> dict:
    headline = pd.read_csv(C.REPORTS / "scenario_headline.csv")
    segs = {
        "credit_score_band": pd.read_csv(
            C.REPORTS / "scenario_segment_credit_score_band.csv"),
        "prepay_by_rate_incentive": pd.read_csv(
            C.REPORTS / "scenario_segment_prepay_by_rate_incentive.csv"),
    }
    metrics = pd.read_csv(C.REPORTS / "model_metrics.csv")
    return G.portfolio_pack(headline, segs, metrics)


def _strip_rule_7(system_prompt: str) -> str:
    lines = [ln for ln in system_prompt.splitlines()
             if not ln.strip().startswith("7. Write plain prose")
             and "never `$-2" not in ln
             and "plain-text servicing queue that renders none of it" not in ln]
    return "\n".join(lines)


def run(samples: int = 3) -> pd.DataFrame:
    pack = _pack()
    rows = []
    original = CL.SYSTEM_PROMPT

    conditions = [("rule_7_present", original),
                  ("rule_7_removed", _strip_rule_7(original))]
    for label, prompt_text in [(l, p) for l, p in conditions for _ in range(samples)]:
        # The client builds its GenerativeModel from the module-level SYSTEM_PROMPT, so the
        # ablation swaps it, constructs a fresh copilot, and restores it afterwards.
        CL.SYSTEM_PROMPT = prompt_text
        try:
            cop = CL.Copilot()
            rec = cop.ask(f"ablation_latex__{label}", PROMPT, pack,
                          validator=grounding_validator,
                          purpose=("Ablation: measuring whether the plain-text rule is what "
                                   "suppresses LaTeX markup, or whether the clean run was "
                                   "chance."))
        finally:
            CL.SYSTEM_PROMPT = original

        v = rec["grounding_validator"] or {}
        text = rec["response"]
        rows.append({
            "condition": label,
            "sample_index": sum(1 for r in rows if r["condition"] == label) + 1,
            "mode": rec["mode"],
            "model": rec["model"],
            "contains_latex_markup": has_latex_markup(text),
            "validator_passed": bool(v.get("passed")),
            "ungrounded_numbers": ", ".join(v.get("ungrounded_numbers", [])) or "none",
            "response_chars": len(text),
            "sample": text[:180].replace("\n", " "),
        })

    df = pd.DataFrame(rows)
    df.to_csv(C.REPORTS / "copilot_latex_ablation.csv", index=False)
    return df


if __name__ == "__main__":
    out = run()
    print(out[["condition", "sample_index", "contains_latex_markup",
               "validator_passed", "ungrounded_numbers"]].to_string(index=False))
    print()
    print(out.groupby("condition")["contains_latex_markup"].agg(["sum", "count"]))
