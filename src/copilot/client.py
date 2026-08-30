"""Google Gemini API client with full prompt logging and an offline fallback.

Every call — live or offline — appends one JSON line to `submission/llm_prompt_log.jsonl`
recording timestamp, provider, model, mode, the exact system and user prompts, the response,
token usage, finish reason and the grounding-validator verdict. The log is a deliverable, not
a debug artefact.

Provider choice is deliberate, not a fallback born of failure. Gemini was chosen for cost and
availability: `gemini-3.5-flash-lite` is reachable on Google AI Studio's free tier with a
daily allowance that clears a full Task 7 run, so this
deliverable reproduces end to end for a reviewer holding nothing but a free API key. The
copilot design is vendor-neutral by construction — grounding packs, the system prompt, the
grounding validator and the adversarial probes are unchanged from the previous Anthropic
wiring. Only the client, auth and response-parsing layer differs.

If no credential is available the client runs in `offline_template` mode. That mode is
clearly labelled everywhere it appears and produces deterministic template text — it is not
a language model and is never presented as one.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone

from src import config as C

PROVIDER = "google-gemini"

# Model choice is a free-tier *quota* decision, not a quality ranking, and it was settled by
# measurement rather than by reading the pricing page. `gemini-3.6-flash` produces the better
# prose, but its free allowance is 20 requests per **day**
# (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`, quota_value 20) — a single Task 7 run
# issues 15-20 calls, so the deliverable would be un-rerunnable for a day after one attempt.
# The lite tier carries a far larger daily allowance and clears a full run with headroom.
MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODEL = "gemini-3.6-flash"
# Gemini 3.x spends part of this budget on an internal thinking phase before emitting a
# single visible token. At 4000 the first live run burned ~6.3k total tokens thinking on
# the portfolio summary and hit MAX_TOKENS mid-sentence, returning a truncated fragment.
# The cap has to clear thinking *and* the answer, so it is set well above what the prose
# itself needs.
MAX_TOKENS = 12000
TEMPERATURE = 0.2

# Free-tier request-per-minute allowances are low. A fixed inter-call pause plus bounded
# retry on 429 keeps a full Task 7 run inside the quota without hand-holding.
MIN_SECONDS_BETWEEN_CALLS = 4.5
RATE_LIMIT_RETRIES = 4
RATE_LIMIT_BACKOFF = 20.0

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
7. Write plain prose. No LaTeX, no MathJax, no markup around numbers. Write a figure exactly as it appears in the pack — `-2e-05`, never `$-2 \\times 10^{-5}$`. These notes are read in a plain-text servicing queue that renders none of it.

If asked something the grounding pack cannot answer, say so plainly and stop. Refusing to \
answer is correct behaviour, not failure."""


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def credentials_available() -> bool:
    return bool(_api_key())


def _is_rate_limit(exc: Exception) -> bool:
    name = type(exc).__name__
    return name in {"ResourceExhausted", "TooManyRequests"} or "429" in str(exc)


def _is_daily_quota(exc: Exception) -> bool:
    """A per-day cap does not clear on backoff; retrying it just burns wall time.

    The first live run spent 125s per call retrying an exhausted daily quota four times
    before giving up. Separating the two 429 flavours turns that into an immediate, readable
    failure.
    """
    return "PerDay" in str(exc) or "per day" in str(exc).lower()


class Copilot:
    def __init__(self, model: str = MODEL, force_offline: bool = False):
        self.model = model
        self.provider = PROVIDER
        self.mode = "live_api"
        self.client = None
        self.sdk = None
        self.init_error = None
        self.rate_limit_events = 0
        self.daily_quota_exhausted = False
        self._last_call_at = 0.0
        if force_offline or not credentials_available():
            self.mode = "offline_template"
            return
        try:
            import google.generativeai as genai
            genai.configure(api_key=_api_key())
            self._genai = genai
            self.sdk = f"google-generativeai {getattr(genai, '__version__', 'unknown')}"
            self.client = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=SYSTEM_PROMPT,
            )
        except Exception as exc:
            self.mode = "offline_template"
            self.init_error = f"{type(exc).__name__}: {exc}"

    def _throttle(self):
        wait = MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - self._last_call_at)
        if wait > 0:
            time.sleep(wait)

    def _generate(self, payload: str):
        """One Gemini call with bounded retry on free-tier rate limiting."""
        cfg = self._genai.types.GenerationConfig(
            max_output_tokens=MAX_TOKENS, temperature=TEMPERATURE)
        last = None
        for attempt in range(RATE_LIMIT_RETRIES):
            self._throttle()
            try:
                resp = self.client.generate_content(payload, generation_config=cfg)
                self._last_call_at = time.monotonic()
                return resp
            except Exception as exc:
                self._last_call_at = time.monotonic()
                last = exc
                if _is_daily_quota(exc):
                    self.daily_quota_exhausted = True
                    raise
                if not _is_rate_limit(exc) or attempt == RATE_LIMIT_RETRIES - 1:
                    raise
                self.rate_limit_events += 1
                time.sleep(RATE_LIMIT_BACKOFF * (attempt + 1))
        raise last

    @staticmethod
    def _extract_text(resp) -> str:
        """Gemini returns parts, and returns none at all when it stops early.

        `resp.text` raises rather than returning empty when the candidate carries no text
        part (safety block, or MAX_TOKENS reached while still in the thinking phase), so the
        parts are walked directly and the stop condition is surfaced instead of swallowed.
        """
        cands = getattr(resp, "candidates", None) or []
        if not cands:
            return ""
        parts = getattr(getattr(cands[0], "content", None), "parts", None) or []
        return "".join(getattr(p, "text", "") or "" for p in parts)

    @staticmethod
    def _usage(resp) -> dict:
        u = getattr(resp, "usage_metadata", None)
        if u is None:
            return {}
        out = {
            "prompt_tokens": getattr(u, "prompt_token_count", None),
            "output_tokens": getattr(u, "candidates_token_count", None),
            "total_tokens": getattr(u, "total_token_count", None),
        }
        thoughts = getattr(u, "thoughts_token_count", None)
        if thoughts:
            out["thinking_tokens"] = thoughts
        return {k: v for k, v in out.items() if v is not None}

    def ask(self, task: str, user_prompt: str, grounding: dict,
            validator=None, purpose: str = "") -> dict:
        payload = f"{user_prompt}\n\nGROUNDING PACK (the only facts you may use):\n" \
                  f"{json.dumps(grounding, indent=2, default=str)}"
        started = datetime.now(timezone.utc)
        finish_reason = response_id = None
        latency = None

        if self.mode == "live_api":
            t0 = time.monotonic()
            try:
                resp = self._generate(payload)
                text = self._extract_text(resp)
                usage = self._usage(resp)
                cands = getattr(resp, "candidates", None) or []
                finish_reason = str(getattr(cands[0], "finish_reason", "")) if cands else None
                response_id = getattr(resp, "response_id", None)
                error = None
                if not text.strip():
                    error = f"empty response (finish_reason={finish_reason})"
                    text = f"[LIVE API CALL RETURNED NO TEXT: {error}]"
            except Exception as exc:
                text = f"[LIVE API CALL FAILED: {type(exc).__name__}: {exc}]"
                usage, error = {}, f"{type(exc).__name__}: {exc}"
            latency = round(time.monotonic() - t0, 2)
        else:
            text = offline_narrative(task, grounding)
            usage, error = {}, None

        verdict = validator(text, grounding) if validator else None
        live = self.mode == "live_api"
        record = {
            "timestamp_utc": started.isoformat(),
            "task": task,
            "purpose": purpose,
            "mode": self.mode,
            "provider": self.provider if live else "none (deterministic template)",
            "model": self.model if live else "none (deterministic template)",
            "sdk": self.sdk,
            "system_prompt_sha256_16": _hash(SYSTEM_PROMPT),
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": payload,
            "user_prompt_sha256_16": _hash(payload),
            "response": text,
            "usage": usage,
            "finish_reason": finish_reason,
            "response_id": response_id,
            "latency_seconds": latency,
            "error": error,
            "grounding_validator": verdict,
            "disclaimer": DISCLAIMER,
        }
        with open(PROMPT_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        return record

    def correct(self, original: dict, grounding: dict, validator=None) -> dict:
        """Feed a blocked output back to the model with the validator's specific findings.

        This is the correction half of the control loop. The validator does not merely flag
        and discard — it names the exact violation, and the model gets one bounded attempt to
        produce something releasable. Both the rejected output and the retry are logged, so
        the failure is evidence rather than something quietly overwritten.
        """
        v = original.get("grounding_validator") or {}
        issues = []
        if v.get("ungrounded_numbers"):
            issues.append(
                f"- These figures appear nowhere in the grounding pack: "
                f"{', '.join(v['ungrounded_numbers'])}. You either invented them or rescaled a "
                f"grounded figure into a different form. Quote the pack's value exactly as it "
                f"is written, or omit the claim.")
        if v.get("causal_or_overconfident_phrases"):
            issues.append(
                f"- This wording asserts causation or certainty the models do not support: "
                f"{', '.join(v['causal_or_overconfident_phrases'])}. Restate as association.")
        if not v.get("carries_reviewer_framing", True):
            issues.append(
                "- The output does not read as a recommendation to a human reviewer. It must "
                "never read as an autonomous determination.")
        if v.get("contains_latex_markup"):
            issues.append(
                "- The answer contains LaTeX/MathJax markup. These notes are read in a "
                "plain-text servicing queue that renders none of it, so the reviewer sees "
                "the raw source. Write figures exactly as the pack writes them, e.g. "
                "`-2e-05`, with no `$`, no `\\times` and no `^{}`.")
        u = v.get("usefulness") or {}
        if u.get("null_advice_targets"):
            issues.append(
                f"- Your 'check this first' instruction points the reviewer at "
                f"{', '.join(u['null_advice_targets'])} — fields this same pack already "
                f"reports as clean, so the instruction asks them to go and look at nothing. "
                f"Direct them at whatever in the pack actually carries risk, or say plainly "
                f"that the pack surfaces no specific item to check first.")
        elif u and not u.get("contains_an_action", True):
            issues.append(
                "- The output contains no next step for the reviewer at all. It restates "
                "figures and stops.")
        if not issues:
            issues.append(
                "- The output was rejected by an automated control but no specific finding "
                "was recorded. Rewrite it to be more precise and better grounded.")

        prompt = (
            "Your previous answer was REJECTED by an automated grounding validator and was "
            "not released to the reviewer queue.\n\n"
            "YOUR PREVIOUS ANSWER:\n"
            f"{original.get('response', '').strip()}\n\n"
            "VALIDATOR FINDINGS:\n" + "\n".join(issues) + "\n\n"
            "Rewrite the answer so it passes. Same task, same length limit. Use only figures "
            "written verbatim in the grounding pack below.")

        return self.ask(
            f"{original['task']}__correction", prompt, grounding, validator=validator,
            purpose=f"Correction round-trip after validator blocked `{original['task']}`.")


def offline_narrative(task: str, g: dict) -> str:
    """Deterministic template text. NOT a language model — labelled as such everywhere."""
    head = "[OFFLINE TEMPLATE OUTPUT — no language model was called. " \
           "Set GEMINI_API_KEY and re-run for the live copilot.]\n\n"

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
