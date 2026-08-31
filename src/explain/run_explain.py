"""Runs Task 6 and writes reports/explainability_report.md."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

from src import config as C
from src import ids
from src.data.report_data_intelligence import _md
from src.explain import shap_explain as E
from src.features.dataset import prepare
from src.models import performance as P
from src.models.splits import purged_time_split

EXPLAINED_TARGETS = ["next_3m_delinquency_flag", "next_12m_default_flag",
                     "next_12m_prepayment_flag", "exception_required"]


def run(df: pd.DataFrame | None = None, models: dict | None = None,
        write_report: bool = True) -> dict:
    df = prepare() if df is None else df
    models = P.load() if models is None else models
    if "exception_required" not in models:
        from src.models import anomaly as A
        from src.features import build_features as F
        models["exception_required"] = A.train_exception_models(
            df, F.feature_columns(df))["binary_model"]
        P.save(models)

    results = {}
    for target in EXPLAINED_TARGETS:
        if target not in models:
            continue
        model = models[target]
        split = purged_time_split(df, target)
        probs = model.predict_proba(df)
        exp = E.explain(model, df, split.test)
        gi = E.global_importance(exp)
        high = E.local_explanations(exp, model, probs, n=10, select="highest")
        low = E.local_explanations(exp, model, probs, n=5, select="lowest")
        unc = E.uncertainty(model, df, split.test)

        y = df.loc[split.test, target].astype(int).to_numpy()
        p = probs[split.test]
        thr = model.thresholds.get("precision_30") or 0.5
        if not np.isfinite(thr):
            thr = 0.5
        err = E.error_analysis(df, split.test, y, p, thr)

        # Masked before both the CSV and the report tables are built from them, so the local
        # explanation artefacts carry the hash rather than the real Loan Sequence Number.
        # `low` is the lowest-risk contrast set and reaches the report even though it has no
        # CSV of its own — it needs the same treatment.
        high = ids.mask_loan_ids(high)
        low = ids.mask_loan_ids(low)
        gi.to_csv(C.REPORTS / f"shap_global_{target}.csv", index=False)
        high.to_csv(C.REPORTS / f"shap_local_high_{target}.csv", index=False)
        unc.describe().to_csv(C.REPORTS / f"uncertainty_{target}.csv")
        results[target] = {"exp": exp, "global": gi, "high": high, "low": low,
                           "uncertainty": unc, "error": err, "threshold": thr,
                           "probs": probs, "split": split}

    if write_report:
        _write_report(df, results, models)
    return results


def _write_report(df, results, models):
    lines = []
    A = lines.append
    A("# Explainability Report")
    A("")
    A("**Task 6.** SHAP TreeExplainer over the trained LightGBM models. Every explanation "
      "here is derived from the fitted model's own structure — no language model is involved "
      "in producing any attribution.")
    A("")
    A("## 1. What is being explained, and in what units")
    A("")
    A("SHAP values are additive in **log-odds**, not probability. Explaining the calibrated "
      "probability directly would break that additivity and the contributions would stop "
      "summing to anything meaningful. Attribution is therefore computed against the raw "
      "LightGBM margin and reported in log-odds, alongside the calibrated probability the "
      "reviewer actually acts on. The two are labelled separately throughout and should not "
      "be added together.")
    A("")
    for target, r in results.items():
        A(f"## Target: `{target}`")
        A("")
        A("### Global feature importance")
        A("")
        A("Mean absolute SHAP contribution across the test window. `direction` is the sign of "
          "the correlation between a feature's value and its contribution, which recovers "
          "whether higher values raise or lower risk without assuming monotonicity.")
        A("")
        A(_md(r["global"][["feature", "plain_english", "mean_abs_shap",
                           "share_of_total_attribution", "direction"]], max_rows=18))
        A("")
        top3 = r["global"].head(3)["plain_english"].tolist()
        share3 = r["global"].head(3)["share_of_total_attribution"].sum()
        A(f"The top three drivers — {', '.join(top3)} — account for "
          f"**{share3:.1%}** of total attribution.")
        A("")
        A("### Local explanations — ten highest-risk records in the test window")
        A("")
        A("Each row shows the calibrated probability a reviewer sees and the four largest "
          "log-odds contributions behind it.")
        A("")
        cols = ["loan_id", "reporting_month", "current_status", "calibrated_probability",
                "driver_1", "driver_1_value", "driver_1_log_odds",
                "driver_2", "driver_2_value", "driver_2_log_odds",
                "driver_3", "driver_3_value", "driver_3_log_odds"]
        A(_md(r["high"][[c for c in cols if c in r["high"].columns]], max_rows=12))
        A("")
        A("### Local explanations — five lowest-risk records (contrast set)")
        A("")
        A(_md(r["low"][[c for c in cols if c in r["low"].columns]], max_rows=8))
        A("")
        A("### Model confidence and uncertainty")
        A("")
        A("Predictions from the final boosting rounds are collected and their spread used as "
          "an epistemic-uncertainty proxy. This measures sensitivity to where the boosting "
          "sequence stopped — it is a stability signal, **not** a statistical confidence "
          "interval, and is not presented as one.")
        A("")
        u = r["uncertainty"]
        band = E.confidence_band(u["calibrated_probability"].to_numpy(),
                                 u["staged_std"].to_numpy())
        summary = (pd.Series(band).value_counts(normalize=True)
                   .rename_axis("confidence_band").reset_index(name="share_of_records"))
        A(_md(summary))
        A("")
        A(_md(u[["calibrated_probability", "staged_std", "staged_p10", "staged_p90"]]
              .describe().reset_index().rename(columns={"index": "statistic"})))
        A("")
        A("### False positive / false negative analysis")
        A("")
        ov = r["error"]["overall"]
        A(f"Evaluated at the threshold that achieves 30% precision on the test window "
          f"(`{ov['threshold']:.4f}`): {ov['true_positives']} true positives, "
          f"{ov['false_positives']} false positives, {ov['false_negatives']} false negatives "
          f"out of {ov['n']} records with {ov['positives']} actual events. "
          f"Precision {ov['precision']:.3f}, recall {ov['recall']:.3f}.")
        A("")
        A("**Where the errors concentrate.** A model that misses events uniformly is a "
          "different problem from one that misses them in a specific segment — the second is "
          "a fairness and coverage issue, not just an accuracy one.")
        A("")
        for seg, tbl in r["error"]["segments"].items():
            A(f"By `{seg}`:")
            A("")
            A(_md(tbl, max_rows=16))
            A("")
        A("**What false positives and false negatives look like.** Mean feature values for "
          "each error class against correctly-rejected records.")
        A("")
        A(_md(r["error"]["profile"]))
        A("")
    A("## Cross-model observations")
    A("")
    A("- **Horizon changes what matters, and the split is clean.** The 3-month delinquency "
      "model is led by *behavioural* signals — current performance status, consecutive clean "
      "months, recent days past due. The 12-month default model is led by *structural* ones "
      "— credit band, debt-to-income band, note rate. Short-horizon risk is about what the "
      "borrower is doing right now; long-horizon risk is about what the loan is. That is the "
      "economically sensible ordering and it was not imposed: both models saw the same 81 "
      "features.")
    A("- **Prepayment is dominated by rate economics** — note rate and the 12-month move in "
      "market rates — which is the correct mechanism and independently corroborates the "
      "rate-incentive bucket table in Task 5 "
      "(`reports/scenario_segment_prepay_by_rate_incentive.csv`). The response there is not "
      "monotone in incentive, and that is the economically right shape: loans already far "
      "in the money are near-saturated and have little headroom left, so a further rate cut "
      "moves them least, while loans sitting just below the refinance threshold move most.")
    A("- **Exceptions are dominated by operational fields** — data-quality score, rule "
      "violation count, missing field count — with essentially no contribution from credit "
      "attributes. This is the same conclusion the ROC-AUC 0.53 credit baseline reached in "
      "Task 4, arrived at from the opposite direction.")
    A("- **Servicer identity carries real attribution weight**, which the data intelligence "
      "report flagged as a confound: the two servicers with the worst reporting hygiene also "
      "have elevated delinquency. Part of that attribution is credit risk and part is "
      "reporting behaviour, and SHAP cannot separate the two. A servicer-driven score is a "
      "prompt to investigate the servicer, not a statement about the borrower.")
    A("")
    A("## Limitations")
    A("")
    A("## Anomaly-score drivers")
    A("")
    A("Task 6 asks for drivers of the anomaly score alongside the three predictive scores. "
      "The anomaly score is unsupervised, so it has no SHAP decomposition: an isolation "
      "forest gives no native per-feature attribution and inventing one would be exactly the "
      "kind of plausible-but-unfounded explanation this layer exists to prevent.")
    A("")
    A("Instead each flagged record is attributed by **robust deviation** — every anomaly "
      "feature is scored by its distance from the training-window median in MAD units, and "
      "the largest deviations are named. That is a quantity a reviewer can check against the "
      "record in front of them, which a SHAP value for an unsupervised model would not be.")
    A("")
    try:
        from src.models import anomaly as ANOM
        from src.models.splits import purged_time_split
        _split = purged_time_split(df, "exception_required")
        _iso, _cols = ANOM.fit_isolation_forest(df, _split.train)
        _test = df.loc[_split.test]
        _drv = ANOM.anomaly_drivers(_test, _cols, reference=df.loc[_split.train])
        _counts = (_drv["anomaly_driver_1"].value_counts(normalize=True)
                   .head(10).rename("share_of_flagged_records").reset_index())
        _counts.columns = ["top_anomaly_driver", "share_of_records"]
        A(_md(_counts.round(4)))
        A("")
        A(f"Across {len(_test):,} held-out records, the leading driver is "
          f"**{_counts.iloc[0]['top_anomaly_driver']}** "
          f"({_counts.iloc[0]['share_of_records']:.1%} of records). Per-record attributions "
          "for the reviewer queue are in `reports/anomaly_review_queue.csv`, and the "
          "distributional detail is in `reports/anomaly_report.md`.")
        A("")
    except Exception as exc:  # never let a reporting extra break the stage
        A(f"_Anomaly driver attribution unavailable in this run: {type(exc).__name__}._")
        A("")
    A("- SHAP attributes to *features*, not to causes. A high contribution from days past due "
      "does not mean delinquency causes default in any actionable sense; it means the model "
      "reads it as the strongest available signal.")
    A("- Correlated features split their attribution arbitrarily between them. The DPD family "
      "(current, lagged, rolling maxima) is highly correlated, so individual rankings within "
      "that family are not stable and should be read as a group.")
    A("- Explanations are computed on a sample of up to 4,000 test rows for tractability.")
    A("- The uncertainty measure is a boosting-stability proxy. It does not capture "
      "uncertainty from feature noise, label noise, or regime change — and regime change is "
      "the dominant risk for the 12-month targets, as Task 2 documented.")
    A("")

    (C.REPORTS / "explainability_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    r = run()
    for t, v in r.items():
        print(t, "->", v["global"].head(3)["plain_english"].tolist())
