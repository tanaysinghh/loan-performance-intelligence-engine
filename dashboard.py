from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# Loan-id masking is defined once, in src/ids.py, and shared with the writer of
# reports/record_quality_scores.csv. Two copies of the same hash would be free to drift.
from src.ids import mask_loan_ids

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
SUBMISSION = ROOT / "submission"

TARGET_LABELS = {
    "next_3m_delinquency_flag": "Delinquency 3m",
    "next_6m_delinquency_flag": "Delinquency 6m",
    "next_12m_default_flag": "Default 12m (90+ DPD proxy)",
    "next_12m_prepayment_flag": "Prepayment 12m",
    "exception_required": "Exception required",
}

MODEL_LABELS = {
    "baseline_logistic": "Logistic baseline (9 features)",
    "lgbm_calibrated": "LightGBM calibrated (shipped)",
}

st.set_page_config(
    page_title="Loan Performance Intelligence Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.4rem; max-width: 1180px;}
      h1 {font-size: 2.0rem;}
      h2 {margin-top: 2.6rem; padding-top: 1.0rem; border-top: 1px solid #e3e8ee;}
      h3 {margin-top: 1.4rem; font-size: 1.05rem;}
      [data-testid="stMetricValue"] {font-size: 1.45rem;}
      .caption {color: #6b7683; font-size: 0.86rem;}
      blockquote {border-left: 3px solid #2f6f9f; padding-left: 0.9rem; background: #f4f6f8;
                  padding-top: 0.6rem; padding-bottom: 0.6rem; border-radius: 0 4px 4px 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_csv(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(REPORTS / name, **kwargs)


@st.cache_data
def load_submission_head(rows: int) -> pd.DataFrame:
    return pd.read_csv(SUBMISSION / "submission.csv", nrows=rows)


@st.cache_data
def submission_columns() -> list[str]:
    return list(pd.read_csv(SUBMISSION / "submission.csv", nrows=1).columns)


@st.cache_data
def markdown_table(report: str, header_contains: str) -> pd.DataFrame:
    lines = (REPORTS / report).read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.startswith("|") and header_contains in line:
            block = []
            for candidate in lines[i:]:
                if not candidate.startswith("|"):
                    break
                block.append(candidate)
            cells = [[c.strip() for c in row.strip("|").split("|")] for row in block]
            header, body = cells[0], [r for r in cells[2:] if len(r) == len(cells[0])]
            return pd.DataFrame(body, columns=header)
    return pd.DataFrame()


@st.cache_data
def copilot_log() -> list[dict]:
    path = SUBMISSION / "llm_prompt_log_archive.jsonl"
    if not path.exists():
        path = SUBMISSION / "llm_prompt_log.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def figure(name: str, caption: str | None = None) -> None:
    path = FIGURES / name
    if path.exists():
        st.image(str(path), width="stretch")
        if caption:
            st.markdown(f"<p class='caption'>{caption}</p>", unsafe_allow_html=True)
    else:
        st.info(f"Figure `{name}` not found. Run `python -m src.make_dashboard_figures`.")


def metrics_table() -> pd.DataFrame:
    frames = [load_csv("model_metrics.csv")]
    exceptions = REPORTS / "exception_binary_metrics.csv"
    if exceptions.exists():
        frames.append(load_csv("exception_binary_metrics.csv"))
    metrics = pd.concat(frames, ignore_index=True)
    metrics = metrics[
        (metrics["split"] == "test") & (metrics["model"].isin(MODEL_LABELS))
    ].copy()
    metrics["Target"] = metrics["target"].map(TARGET_LABELS).fillna(metrics["target"])
    metrics["Model"] = metrics["model"].map(MODEL_LABELS)
    metrics = metrics.set_index("target").loc[list(TARGET_LABELS)].reset_index()
    out = metrics[
        ["Target", "Model", "roc_auc", "pr_auc", "pr_auc_lift_over_base", "best_f1", "brier", "ece", "n"]
    ].rename(
        columns={
            "roc_auc": "ROC-AUC",
            "pr_auc": "PR-AUC",
            "pr_auc_lift_over_base": "PR-AUC lift",
            "best_f1": "Best F1",
            "brier": "Brier",
            "ece": "ECE",
            "n": "Test rows",
        }
    )
    return out.round({"ROC-AUC": 4, "PR-AUC": 4, "PR-AUC lift": 2, "Best F1": 4, "Brier": 4, "ECE": 4})


with st.sidebar:
    st.markdown("### Loan Performance\n### Intelligence Engine")
    st.caption("Intain Campus FinTech Challenge 2026 — AI Track, Round 2")
    st.markdown(
        "\n".join(
            [
                "1. [Overview](#overview)",
                "2. [Data intelligence](#data-intelligence)",
                "3. [Prediction performance](#prediction-performance)",
                "4. [Survival and transition](#survival-and-transition-modelling)",
                "5. [Anomaly detection](#anomaly-detection)",
                "6. [Scenario simulation](#scenario-simulation)",
                "7. [Explainability](#explainability)",
                "8. [LLM copilot](#llm-copilot)",
                "9. [Submission output](#submission-output)",
            ]
        )
    )
    st.divider()
    st.caption(
        "Every figure on this page is read from the committed report artefacts in `reports/` "
        "and `submission/`. Nothing is recomputed and no licensed loan-level data is loaded."
    )


st.title("Loan Performance Intelligence Engine")
st.markdown(
    "<p class='caption'>A demo view over the generated artefacts. "
    "Source of truth remains <code>submission/MODEL_CARD.md</code> and the reports in "
    "<code>reports/</code>.</p>",
    unsafe_allow_html=True,
)

st.header("Overview")

st.markdown(
    """
The engine turns a real Freddie Mac servicing panel into decision support for a servicing
oversight team. It profiles the data, predicts delinquency, serious delinquency and prepayment
on a purged out-of-time split, models time to event and state transitions, scores records whose
data does not hold together, projects the book under macro scenarios, explains every score with
SHAP, and hands a reviewer a ranked queue.

**The predictive work is entirely non-LLM.** Every probability, rate and ranked driver traces to
a fitted LightGBM, scikit-learn or lifelines estimator. A language model is used in exactly one
place — narrating already-computed figures for human reviewers — and a test parses the AST of
every modelling module and fails if it can even import an LLM client.
"""
)

row = st.columns(6)
row[0].metric("Loans", "16,000")
row[1].metric("Loan-months", "670,548")
row[2].metric("Vintages", "2019-2023")
row[3].metric("Panel window", "87 months")
row[4].metric("Rate cycle spread", "377 bp")
row[5].metric("Engineered features", "81")

st.markdown(
    """
Sampled at loan level from a population of 250,000 loans and 10,482,492 monthly performance
records. The reporting window runs 2019-01 to 2026-03 across 42 servicers and 53 states. Mean
origination rate falls from 4.24% (2019) to 2.97% (2021) and rises to 6.74% (2023) — a genuine,
non-simulated rate cycle, which is why 71% of the 2019 vintage prepaid against 19% of 2021.

> **`next_12m_default_flag` is a 90+ DPD proxy.** Realised credit events occur on 14 of the
> 16,000 sampled loans (~0.09%) — not modellable at this sample size. Every "default" figure
> below refers to that proxy, not to foreclosure or to loss.

The panel, its outcomes and the macro history are real. The exception, reconciliation and
document-status layer is fabricated on top, because the source dataset has no second feed, no
ingestion timestamps and no document data — disclosed in section 2 of the model card.
"""
)

st.header("Data intelligence")

left, right = st.columns([1, 1])
with left:
    st.subheader("Validation rules fired")
    rules = load_csv("validation_rule_summary.csv")
    st.dataframe(
        rules[["rule", "dimension", "severity", "violations", "violation_rate"]]
        .sort_values("violations", ascending=False)
        .head(10)
        .round({"violation_rate": 4}),
        hide_index=True,
        width="stretch",
    )
with right:
    st.subheader("Missingness mechanism")
    miss = load_csv("missingness_mechanism_tests.csv")
    st.dataframe(
        miss[["column", "cramers_v", "p_value", "verdict"]].round({"cramers_v": 4, "p_value": 4}),
        hide_index=True,
        width="stretch",
    )
    st.markdown(
        "<p class='caption'>Missingness is not random: it depends on which servicer reported "
        "the record, so it is MAR conditional on servicer and repairable rather than ignorable.</p>",
        unsafe_allow_html=True,
    )

st.subheader("Train versus test drift")
figure(
    "drift_psi.png",
    "PSI above 0.25 is severe. Loan age and remaining term drift hardest because the test window "
    "sits later in the panel by construction — the drift is the split working, not a defect.",
)

st.header("Prediction performance")

st.dataframe(metrics_table(), hide_index=True, width="stretch")
st.markdown(
    "<p class='caption'>Purged out-of-time test window. Shipped model is the calibrated LightGBM; "
    "the baseline is a nine-feature logistic regression on credit fields only.</p>",
    unsafe_allow_html=True,
)

figure("model_comparison.png")

st.markdown(
    """
**Read the baseline comparison honestly.** LightGBM does not dominate on ranking. The baseline
wins outright on prepayment (0.685 against 0.626 ROC-AUC) and is within noise on the default
proxy. The dominant delinquency signals are near-monotone in the log-odds, which is exactly
where a linear model is hard to beat. Where the two separate is calibration — see
[Explainability](#explainability). The one total gap is the exception model (0.969 against
0.540), and that gap is itself the finding: the baseline is deliberately the same nine *credit*
fields, and operational exceptions are not a credit phenomenon.
"""
)

st.subheader("Leakage ablation — the strongest evidence here")
figure("leakage_ablation.png")

probe = load_csv("leakage_probe.csv")
probe_display = probe.copy()
probe_display["target"] = probe_display["target"].map(TARGET_LABELS).fillna(probe_display["target"])
st.dataframe(
    probe_display.rename(
        columns={
            "target": "Target",
            "purged_time_split": "Purged time split",
            "loan_disjoint_time_split": "Loan-disjoint control",
            "random_row_split_unsound": "Random row split (unsound)",
            "random_split_inflation": "Inflation",
        }
    ).round(4),
    hide_index=True,
    width="stretch",
)

st.markdown(
    """
Refitting the same model and the same features under a naive random row split inflates test
ROC-AUC to **0.9988 on the default proxy** and **0.9831 on prepayment** — against 0.9207 and
0.6259 under the purged split. That is precisely how much a careless split would have flattered
this submission.

The useful part is that **the inflation is not uniform**: three-month delinquency actually moves
*down* (−0.0097). A single well-behaved target proves nothing. The loan-disjoint control forces
no `loan_id` into both the fitting data and the test window and lands within noise of the
reported numbers, so the model learned loan characteristics rather than loan identities.
"""
)

with st.expander("Split design — windows, embargo and row counts"):
    splits = load_csv("split_summary.csv")
    st.dataframe(
        splits[
            [
                "target",
                "horizon_months",
                "train_window",
                "valid_window",
                "embargo_window",
                "test_window",
                "train_rows",
                "test_rows",
                "rows_dropped_embargo",
                "rows_dropped_unobservable_label",
                "test_positive_rate",
            ]
        ].round({"test_positive_rate": 4}),
        hide_index=True,
        width="stretch",
    )
    st.markdown(
        "Three problems handled separately: labels that are unobservable near the panel end, "
        "training rows whose outcome window overlaps the test window, and right-censoring. "
        "Censored rows are `NaN` and dropped, never zero-filled."
    )

st.header("Survival and transition modelling")

figure("survival_curves.png")

concordance = markdown_table("survival_report.md", "concordance_test")
left, right = st.columns([1.05, 1])
with left:
    st.subheader("Cox discrimination")
    if not concordance.empty:
        st.dataframe(concordance, hide_index=True, width="stretch")
    st.markdown(
        """
Kaplan-Meier assigns every loan the same curve, so its concordance is **0.50 by construction** —
that is the baseline Cox is beating. Test concordance is **0.7169** for default and **0.6811**
for prepayment.

Proportional hazards is assumed, not tested: no Schoenfeld residual test is run, and that is
stated in the model card rather than glossed.
"""
    )
with right:
    figure("transition_matrix.png")

st.markdown(
    """
The first-order Markov assumption is **wrong, usefully**. A loan five months into DQ30 differs
from one that entered last month, and the covariate model beats the chain on macro-AUC (0.982
against 0.832). The chain is kept because it is transparent and because it projects multiple
periods ahead, which the point-in-time classifier cannot.
"""
)

with st.expander("Markov backtest against realised 12-month outcomes"):
    st.dataframe(load_csv("markov_validation.csv").round(4), hide_index=True, width="stretch")

st.header("Anomaly detection")

figure("anomaly_distribution.png")

st.markdown(
    """
The isolation forest never saw the exception label. Flagging the top **6.0%** of records by
anomaly score alone gives **59.0% precision** against that label — a **4.12x lift** over the
14.3% base rate, at ROC-AUC **0.893**. Weaker than the supervised model, which is the expected
ordering and the reason the supervised score drives the queue while the anomaly score is kept as
a second opinion for defect shapes the label does not cover.
"""
)

st.subheader("Reviewer-ready queue")
queue = load_csv("anomaly_review_queue.csv")
st.dataframe(
    mask_loan_ids(queue)[
        [
            "loan_id",
            "reporting_month",
            "servicer_name",
            "review_priority",
            "exception_probability",
            "anomaly_score",
            "predicted_exception_type",
            "anomaly_driver_1",
            "anomaly_driver_1_zscore",
            "rules_violated",
        ]
    ]
    .head(10)
    .round({"review_priority": 4, "exception_probability": 4, "anomaly_score": 4, "anomaly_driver_1_zscore": 2}),
    hide_index=True,
    width="stretch",
)
st.markdown(
    f"<p class='caption'>Top 10 of {len(queue)} reviewer-ready examples, ranked by "
    "0.6 x exception probability + 0.4 x anomaly score, with coverage forced across every "
    "predicted exception type so one common defect cannot monopolise the queue. "
    "Loan identifiers are hashed for display; the committed artefacts keep the real ones.</p>",
    unsafe_allow_html=True,
)

with st.expander("Anomaly concentration by servicer"):
    by_servicer = load_csv("anomaly_by_servicer.csv")
    st.dataframe(
        by_servicer.sort_values("mean_anomaly_score", ascending=False).head(10).round(4),
        hide_index=True,
        width="stretch",
    )
    st.markdown(
        "Ranking by mean anomaly score independently recovers the two servicers the data "
        "intelligence report identified as having the worst reporting hygiene. The unsupervised "
        "model was given no servicer identity at all — it sees only the numeric record profile — "
        "so this is corroboration, not circularity."
    )

st.header("Scenario simulation")

figure("scenario_projections.png")

headline = load_csv("scenario_headline.csv")
st.dataframe(
    headline[
        [
            "scenario_name",
            "loans",
            "projected_next_6m_delinquency_flag",
            "projected_next_12m_default_flag",
            "projected_next_12m_prepayment_flag",
            "delta_next_12m_default_flag",
            "delta_next_12m_prepayment_flag",
        ]
    ].round(5),
    hide_index=True,
    width="stretch",
)

st.markdown(
    """
**The credit channel is not identified, and that is reported rather than tuned away.** Macro
levels are constant across loans within a month, so with one realised macro path a loan-level
model cannot separate unemployment from calendar time. The symptom is visible above: the
adverse-credit scenario moves projected 12-month default by −0.13% in relative terms, and the
high-prepayment scenario *raises* it.

A second engine carries the stress properly. The macro-conditioned Markov chain moves cumulative
12-month default from **1.17% to 1.86%** under adverse conditions and nearly doubles the
delinquent stock. Use Engine B to size credit stress; use Engine A only for which-loans segment
detail.
"""
)

st.subheader("The most interesting result: prepayment response is not monotone in incentive")
figure("prepay_non_monotone.png")

st.markdown(
    """
Under the high-prepayment scenario the loan-level engine adds 5.1 points overall, but the
response is **not monotone in refinance incentive**. Loans already deep in the money are
saturated — they barely move (**+2.7pp** above 1.0 incentive, **+4.6pp** at 0.5-1.0). The
largest move comes from loans sitting just below the refinance threshold that the rate cut
pushes across (**+23.2pp** at −0.5 to 0, **+18.5pp** at −1.0 to −0.5), while loans far out of
the money stay put (**+0.8pp**).

The behavioural reading is that a rate cut does not accelerate the people already refinancing;
it recruits the marginal borrower. A model that assumed a monotone incentive response would put
the sensitivity in the wrong segment of the book.
"""
)

with st.expander("Segment detail by refinance incentive"):
    st.dataframe(
        load_csv("scenario_segment_prepay_by_rate_incentive.csv").round(5),
        hide_index=True,
        width="stretch",
    )

st.header("Explainability")

figure("shap_global_default.png")

st.markdown(
    """
Direction is recovered from the correlation between a feature's value and its contribution, so
nothing assumes monotonicity. `state` and `servicer_name` carry real weight — which is also a
stated limitation: **servicer is a confound.** The two servicers with the worst reporting hygiene
also have elevated delinquency, and SHAP cannot separate credit risk from reporting behaviour. A
servicer-driven score is a prompt to investigate the servicer, not a statement about the borrower.
"""
)

st.subheader("Calibration: where the gradient boosting actually earns its place")
figure("calibration_comparison.png")

st.markdown(
    """
**Stated plainly: on ranking, the simpler model wins on prepayment and ties on default.** What
separates them is calibration, by **2x to 12x on Brier score** — 0.0091 against 0.1089 on the
default proxy, 0.1367 against 0.2789 on prepayment, with the baseline's expected calibration
error running 0.22 to 0.42 because `class_weight=balanced` inflates every probability.

The baseline can rank a queue. It cannot answer "what is the probability", which is what the
submission format asks for. That is the whole reason the calibrated GBM ships.
"""
)

with st.expander("Confidence bands are a stability proxy, not a confidence interval"):
    st.markdown(
        "Prediction spread is measured across boosting stages. It captures model instability, not "
        "statistical uncertainty, and it does **not** capture regime-change risk — which is the "
        "dominant risk on the twelve-month targets, where train and test sit in different macro "
        "regimes and the default rate moves from 1.9% to 1.5% between them."
    )

st.header("LLM copilot")

st.markdown(
    """
The copilot never sees the dataframe or the models. It receives a JSON grounding pack of
already-computed figures and turns them into reviewer prose. A grounding validator extracts every
number from its output and blocks anything that does not match the pack, including values
helpfully rescaled by 100 or rounded. A second control, the usefulness check, blocks output that
is true but points the reviewer at a field the pack already reports as clean.

Both examples below are read live from `submission/llm_prompt_log_archive.jsonl` — actual logged
Gemini calls, not written for this page.
"""
)

log = copilot_log()
passed_note = next(
    (r for r in log if r.get("task") == "reviewer_note" and (r.get("grounding_validator") or {}).get("passed")),
    None,
)
blocked_index, blocked = next(
    (
        (i, r)
        for i, r in enumerate(log)
        if (r.get("grounding_validator") or {}).get("ungrounded_numbers")
        and r.get("task") == "reviewer_note"
    ),
    (None, None),
)
correction = None
if blocked_index is not None:
    correction = next(
        (r for r in log[blocked_index:] if r.get("task") == "reviewer_note__correction"),
        None,
    )

if passed_note:
    st.subheader("A grounded reviewer summary that passed")
    meta = passed_note.get("grounding_validator") or {}
    st.markdown(f"> {passed_note['response'].strip()}")
    cols = st.columns(4)
    cols[0].metric("Numbers checked", meta.get("numbers_checked", "-"))
    cols[1].metric("Ungrounded", len(meta.get("ungrounded_numbers") or []))
    cols[2].metric("Model", passed_note.get("model", "-"))
    cols[3].metric("Latency", f"{passed_note.get('latency_seconds', 0):.1f}s")
    st.markdown(
        f"<p class='caption'>Verdict: <strong>{meta.get('action', '-')}</strong> · "
        f"logged {passed_note.get('timestamp_utc', '')} · every prompt, response, token count, "
        "finish reason and validator verdict is written to the prompt log.</p>",
        unsafe_allow_html=True,
    )

if blocked:
    st.subheader("A real fabrication the validator caught")
    meta = blocked.get("grounding_validator") or {}
    st.error(
        f"BLOCKED — ungrounded number(s): {', '.join(meta.get('ungrounded_numbers') or [])}. "
        f"{meta.get('numbers_checked', 0)} numbers checked against the pack."
    )
    st.markdown(f"> {blocked['response'].strip()}")
    st.markdown(
        "Gemini dropped a decimal place restating a small probability — it wrote the exception "
        "figure as **0.046** where the grounding pack says **0.0046**, a 10x error on a number "
        "that would have gone in front of a reviewer. Nothing about the sentence looks wrong; "
        "only a check against the pack catches it."
    )
    if correction:
        st.success("Correction round-trip — released after the model was sent back with the mismatch.")
        st.markdown(f"> {correction['response'].strip()}")

st.markdown(
    """
Running it live caught failures on **both** sides. The validator also flagged *correct* output six
different ways, each now fixed at source and pinned by a case in a twelve-case self-test that runs
against a fixed pack so its verdicts cannot drift with the data.
`reports/copilot_report.md` section 5 separates genuine model failures from validator false
positives rather than counting every block against the model. One ablation came out **negative**
and is reported as negative: the system-prompt rule forbidding LaTeX could not be shown to be what
stopped the model emitting it (0 of 3 in both arms). Detection is the load-bearing control, not
the prompt.

All LLM output carries **"RECOMMENDATION, NOT DECISION."** and nothing it produces enters
`submission.csv`.
"""
)

with st.expander("Validator self-test results"):
    st.dataframe(load_csv("copilot_validator_self_test.csv"), hide_index=True, width="stretch")

st.header("Submission output")

preview = mask_loan_ids(load_submission_head(8))
st.dataframe(preview, hide_index=True, width="stretch")
st.markdown(
    f"<p class='caption'>First 8 rows of <code>submission/submission.csv</code> "
    f"({len(submission_columns())} columns, one row per loan at its latest scored month). "
    "Loan identifiers are hashed for display only — the delivered file is unchanged. "
    "Full file not loaded here.</p>",
    unsafe_allow_html=True,
)

st.markdown(
    """
Each row carries the four performance probabilities, the exception probability and type, the
predicted next state, the anomaly score and its top driver, the SHAP drivers behind the default
score, a confidence band, and a recommended action with its reason — flagged
`action_is_recommendation_not_decision = True`.

**No external test file was issued.** Section 6 of the problem statement anticipates an
organiser-supplied unlabeled test file for final scoring; none was released, so this project
builds its own data pipeline to fill that gap. These are held-out predictions on the project's
own time-aware split — the purged out-of-time test window — not scores against an external file.
No code path claims otherwise.
"""
)

st.divider()
st.markdown(
    "<p class='caption'>Built from the committed artefacts in <code>reports/</code> and "
    "<code>submission/</code>. The licence-gated Freddie Mac source files are neither committed "
    "nor required to run this dashboard. Full method, metrics and limitations: "
    "<code>submission/MODEL_CARD.md</code>.</p>",
    unsafe_allow_html=True,
)
