"""Anthropic Messages API client with full prompt logging and an offline fallback.

Every call — live or offline — appends one JSON line to `submission/llm_prompt_log.jsonl`
recording timestamp, model, mode, the exact system and user prompts, the response, token
usage and the grounding-validator verdict. The log is a deliverable, not a debug artefact.

If no credential is available the client runs in `offline_template` mode. That mode is
clearly labelled everywhere it appears and produces deterministic template text — it is not
a language model and is never presented as one.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

from src import config as C

MODEL = "claude-opus-5"
MAX_TOKENS = 4000
PROMPT_LOG = C.SUBMISSION / "llm_prompt_log.jsonl"

DISCLAIMER = ("RECOMMENDATION, NOT DECISION. Generated narrative over model output. "
              "A human reviewer owns the outcome.")

SYSTEM_PROMPT = """You are a servicing-oversight reviewer copilot for a loan performance \
intelligence engine.

You will be given a JSON grounding pack. It contains figures produced by trained statistical \
models (LightGBM, isolation forest, Cox proportional hazards, Markov chains). Your job is to \
turn those figures into clear reviewer-facing prose.

Absolute rules:
1. NEVER produce a number that is not in the grounding pack. Do not estimate, interpolate, \
round to a different precision, compute a ratio, or infer a figure. If a number you want is \
absent, say the pack does not contain it.
2. NEVER predict anything yourself. The models make predictions; you describe them.
3. NEVER assert causation. The models measure association. Write "is associated with" or \
"the model weights heavily", not "caused by" or "because of".
4. State uncertainty where the pack gives it. If a confidence band or calibration figure is \
present, reflect it.
5. Every output you write is a recommendation for a human reviewer, never a decision.
6. Be concise and specific. A reviewer reads dozens of these.

If asked something the grounding pack cannot answer, say so plainly and stop. Refusing to \
answer is correct behaviour, not failure."""


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def credentials_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


class Copilot:
    def __init__(self, model: str = MODEL, force_offline: bool = False):
        self.model = model
        self.mode = "live_api"
        self.client = None
        if force_offline or not credentials_available():
            self.mode = "offline_template"
            return
        try:
            import anthropic
            self.client = anthropic.Anthropic()
        except Exception as exc:
            self.mode = "offline_template"
            self.init_error = str(exc)

    def ask(self, task: str, user_prompt: str, grounding: dict,
            validator=None, purpose: str = "") -> dict:
        payload = f"{user_prompt}\n\nGROUNDING PACK (the only facts you may use):\n" \
                  f"{json.dumps(grounding, indent=2, default=str)}"
        started = datetime.now(timezone.utc)

        if self.mode == "live_api":
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": payload}],
                )
                text = "".join(b.text for b in resp.content if b.type == "text")
                usage = {"input_tokens": resp.usage.input_tokens,
                         "output_tokens": resp.usage.output_tokens}
                request_id = getattr(resp, "_request_id", None)
                error = None
            except Exception as exc:
                text = f"[LIVE API CALL FAILED: {type(exc).__name__}: {exc}]"
                usage, request_id, error = {}, None, str(exc)
        else:
            text = offline_narrative(task, grounding)
            usage, request_id, error = {}, None, None

        verdict = validator(text, grounding) if validator else None
        record = {
            "timestamp_utc": started.isoformat(),
            "task": task,
            "purpose": purpose,
            "mode": self.mode,
            "model": self.model if self.mode == "live_api" else "none (deterministic template)",
            "system_prompt_sha256_16": _hash(SYSTEM_PROMPT),
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": payload,
            "user_prompt_sha256_16": _hash(payload),
            "response": text,
            "usage": usage,
            "request_id": request_id,
            "error": error,
            "grounding_validator": verdict,
            "disclaimer": DISCLAIMER,
        }
        with open(PROMPT_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        return record


def offline_narrative(task: str, g: dict) -> str:
    """Deterministic template text. NOT a language model — labelled as such everywhere."""
    head = "[OFFLINE TEMPLATE OUTPUT — no language model was called. " \
           "Set ANTHROPIC_API_KEY and re-run for the live copilot.]\n\n"

    if task == "reviewer_note":
        r = g["record"]
        p = g["model_predictions"]
        lines = [
            f"Loan {r['loan_id']}, reporting month {r['reporting_month']}, serviced by "
            f"{r['servicer_name']}.",
            f"Status at month end is {r['current_status']} with "
            f"{r['days_past_due']} days past due.",
            "Model output for this record:",
        ]
        for k, v in p.items():
            if v is not None:
                lines.append(f"  - {k}: {v}")
        lines.append(f"Leading model drivers: {g['top_drivers_from_shap']}.")
        if "anomaly" in g:
            lines.append(f"Anomaly score {g['anomaly'].get('anomaly_score')}, "
                         f"predicted exception type "
                         f"{g['anomaly'].get('predicted_exception_type')}.")
        lines.append("Association only; no causal claim is made. " + DISCLAIMER)
        return head + "\n".join(lines)

    if task == "scenario_summary":
        lines = ["Scenario projections as produced by the simulation engines:"]
        for row in g.get("scenario_projections", []):
            lines.append(f"  - {row.get('scenario_name')}: "
                         f"12-month default {row.get('projected_next_12m_default_flag')}, "
                         f"12-month prepayment {row.get('projected_next_12m_prepayment_flag')}")
        lines.append("Segment and driver detail is in reports/scenario_report.md. " + DISCLAIMER)
        return head + "\n".join(lines)

    if task == "data_dictionary":
        lines = ["Data dictionary entries retrieved:"]
        for e in g.get("data_dictionary_entries", []):
            lines.append(f"  - {e['field']} ({e['dtype']}): {e['description']} "
                         f"Allowed: {e['allowed_values']}. Source: {e['source_system']}.")
        return head + "\n".join(lines)

    return head + json.dumps(g, indent=2, default=str)
