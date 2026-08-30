# `submission.csv` — format and provenance

Section 6 of the problem statement lists a `submission_template.csv` among the files the
organiser was to provide. It was never issued, so there is no template to match column names
against. The required *elements* are named in section 6 — "probabilities, next state,
exception type, anomaly score, top drivers, action, and confidence" — and every one is
present. This file maps them, so a judge can verify coverage without reading the builder.

One row per loan, at that loan's most recent reporting month.

## Coverage of the required elements

| Required element (PS §6) | Column(s) in `submission.csv` |
|---|---|
| Probabilities | `prob_delinquency_3m`, `prob_delinquency_6m`, `prob_default_12m`, `prob_prepayment_12m`, `exception_probability` |
| Next state | `predicted_next_state` (+ `next_state_confidence`) |
| Exception type | `predicted_exception_type` (+ `exception_type_confidence`) |
| Anomaly score | `anomaly_score` (+ `top_anomaly_driver`) |
| Top drivers | `top_drivers_default_model` |
| Action | `recommended_action` (+ `action_reason`) |
| Confidence | `confidence` (+ `prediction_spread`) |

## Every column

| Column | Type | Produced by |
|---|---|---|
| `loan_id` | string | Panel key |
| `reporting_month` | `YYYY-MM` | Latest observed month for the loan |
| `servicer_name` | string | Panel |
| `current_status` | category | Panel |
| `prob_delinquency_3m` | float 0-1 | LightGBM binary, isotonic/sigmoid calibrated |
| `prob_delinquency_6m` | float 0-1 | LightGBM binary, calibrated |
| `prob_default_12m` | float 0-1 | LightGBM binary, calibrated — **90+ DPD proxy, see below** |
| `prob_prepayment_12m` | float 0-1 | LightGBM binary, calibrated |
| `exception_probability` | float 0-1 | LightGBM binary, calibrated |
| `predicted_next_state` | category | LightGBM multiclass, arg-max over 7 states |
| `next_state_confidence` | float 0-1 | Max class probability |
| `anomaly_score` | float | Isolation forest, unsupervised |
| `top_anomaly_driver` | string | Largest standardised contributor vs the training reference |
| `predicted_exception_type` | category | LightGBM multiclass; forced to `none` when `exception_probability < 0.50` |
| `exception_type_confidence` | float 0-1 | Max class probability |
| `top_drivers_default_model` | string | Top-3 SHAP contributions for that row, signed |
| `confidence` | high/medium/low | Banded from the model's own dispersion |
| `prediction_spread` | float | Std. dev. of the staged boosting predictions |
| `recommended_action` | category | Deterministic rule over the columns above |
| `action_reason` | string | The rule that fired, in plain English |
| `action_is_recommendation_not_decision` | bool | Always `True` |

## Two things a reader should know

**`prob_default_12m` is a 90+ DPD probability.** Realised credit events occur on 14 of the
16,000 sampled loans, so a literal default target is not modellable. The column is the
probability the loan reaches 90+ days past due, or a realised credit event, within 12 months.
Full disclosure in section 2 of `MODEL_CARD.md`.

**No language model produces any value in this file.** Every column traces to a fitted
LightGBM, scikit-learn or lifelines estimator, or to a documented deterministic rule over
those outputs. `recommended_action` is a rule, not a learned policy and not an LLM judgement,
so the reason any loan received any action is reconstructible from the numbers in its own row.
The boundary is enforced by `tests/test_no_llm_leakage.py`, which fails if any modelling
module can even import an LLM client.

## Action thresholds

Rules are evaluated in priority order; the first match wins.

| Action | Condition |
|---|---|
| `raise_exception_for_review` | `exception_probability >= 0.50` |
| `escalate_loss_mitigation` | `prob_default_12m >= 0.35` |
| `early_stage_collections_outreach` | `prob_delinquency_3m >= 0.30` |
| `watchlist_monitor` | `prob_delinquency_6m >= 0.25` |
| `retention_review` | `prob_prepayment_12m >= 0.45` |
| `monitor_no_action` | none of the above |
