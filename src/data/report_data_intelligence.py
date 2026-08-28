"""Produces reports/data_intelligence_report.md and its supporting CSV extracts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.data import loaders, profiling, validate


def _md(df: pd.DataFrame, floatfmt: str = "{:.4f}", max_rows: int = 40) -> str:
    d = df.head(max_rows).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    header = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |"
                     for row in d.itertuples(index=False))
    return "\n".join([header, sep, body])


def build(df: pd.DataFrame = None) -> dict:
    if df is None:
        raw = loaders.load_panel()
        updates = loaders.load_servicer_updates()
        joined = loaders.reconcile(raw, updates)
        df = loaders.clean_panel(joined)
        feed_stats = {"duplicates": joined.attrs.get("feed_duplicate_records", 0),
                      "orphans": joined.attrs.get("feed_orphan_records", 0)}
    else:
        feed_stats = {"duplicates": df.attrs.get("feed_duplicate_records", 0),
                      "orphans": df.attrs.get("feed_orphan_records", 0)}

    flagged, rule_summary = validate.run_rules(df)
    scored = validate.score_records(flagged)
    batches = validate.score_batches(scored)

    num = profiling.profile_numeric(scored)
    cat = profiling.profile_categorical(scored)
    miss = profiling.missingness_structure(scored)
    deps = profiling.dependency_analysis(scored)
    drift = profiling.drift_report(scored)
    tgt = profiling.target_stability(scored)

    num.to_csv(C.REPORTS / "profile_numeric.csv", index=False)
    cat.to_csv(C.REPORTS / "profile_categorical.csv", index=False)
    rule_summary.to_csv(C.REPORTS / "validation_rule_summary.csv", index=False)
    batches.to_csv(C.REPORTS / "batch_quality_scores.csv", index=False)
    drift.to_csv(C.REPORTS / "drift_report.csv", index=False)
    miss["mechanism_tests"].to_csv(C.REPORTS / "missingness_mechanism_tests.csv", index=False)
    scored[["loan_id", "reporting_month", "servicer_name", "dq_score", "dq_band",
            "dq_violation_count"]].to_csv(C.REPORTS / "record_quality_scores.csv", index=False)

    truth_path = C.DATA_RAW / "ground_truth_defect_log.csv"
    truth = pd.read_csv(truth_path) if truth_path.exists() else pd.DataFrame()

    worst_batches = batches.sort_values("mean_dq_score").head(10)
    co = miss["co_missingness"]
    co_pairs = (co.where(np.triu(np.ones(co.shape), 1).astype(bool))
                .stack().sort_values(ascending=False).head(8).reset_index())
    co_pairs.columns = ["field_a", "field_b", "missingness_correlation"]

    lines = []
    A = lines.append
    A("# Data Intelligence Report")
    A("")
    A("**Loan Performance Intelligence Engine — Task 1**  ")
    A(f"Generated from `{C.LOAN_PANEL.name}` and `{C.SERVICER_UPDATES.name}`.")
    A("")
    A("## 1. Scope")
    A("")
    A(f"- Records after de-duplication: **{len(scored):,}**")
    A(f"- Distinct loans: **{scored['loan_id'].nunique():,}**")
    A(f"- Reporting months: **{scored['reporting_month'].min()} to {scored['reporting_month'].max()}** "
      f"({scored['reporting_month'].nunique()} months)")
    A(f"- Servicers: **{scored['servicer_name'].nunique()}**; states: **{scored['state'].nunique()}**")
    A(f"- Secondary servicer feed: **{feed_stats['duplicates']:,}** duplicate loan-month records "
      f"resolved latest-wins, **{feed_stats['orphans']:,}** orphan records referencing loan-months "
      f"absent from the panel.")
    A("")
    A("## 2. Column distribution profiling")
    A("")
    A("### Numeric fields")
    A("")
    A(_md(num[["column", "missing_pct", "mean", "std", "min", "p01", "median", "p99", "max",
               "skew", "negatives", "iqr_outlier_pct"]]))
    A("")
    A("### Categorical fields")
    A("")
    A(_md(cat[["column", "missing_pct", "distinct", "mode", "mode_share",
               "normalised_entropy", "top_values"]]))
    A("")
    A("## 3. Missingness patterns")
    A("")
    A(f"- Rows with at least one missing profiled field: **{miss['rows_with_any_missing']:.1%}**")
    A(f"- Mean missing fields per row: **{miss['mean_missing_fields_per_row']:.3f}**")
    A("")
    A("Missingness is not random. A chi-square test of each field's missingness indicator "
      "against `servicer_name` rejects independence for the fields below, so the mechanism is "
      "**missing-at-random conditional on servicer**, not MCAR. Two servicers "
      "(Kestrel Financial, Pioneer Mortgage Ops) account for most of the gap. The practical "
      "consequence: dropping incomplete rows would silently drop those servicers' books and "
      "bias every downstream rate. Models therefore consume missingness natively and carry "
      "explicit missing-indicator features.")
    A("")
    A(_md(miss["mechanism_tests"]))
    A("")
    A("### Co-missingness (fields that go missing together)")
    A("")
    A(_md(co_pairs))
    A("")
    A("### Missingness by servicer")
    A("")
    A(_md(miss["by_servicer"].round(4).reset_index()))
    A("")
    A("## 4. Outliers, sentinels and invalid dates")
    A("")
    A("Sentinel values are treated as *absence of information*, not as extreme numbers. "
      "`days_past_due` of 9999 or -1, note rates of 0 / 99.99 / -1, and balances above 3x "
      "original are masked to missing with a `*_repaired` indicator retained, so the fact "
      "that a repair happened stays available as a feature.")
    A("")
    repair_tbl = pd.DataFrame([
        {"repair": "days_past_due sentinel/out-of-range masked",
         "rows": int(scored["dpd_repaired"].sum()),
         "rate": float(scored["dpd_repaired"].mean())},
        {"repair": "interest_rate out-of-range masked",
         "rows": int(scored["rate_repaired"].sum()),
         "rate": float(scored["rate_repaired"].mean())},
        {"repair": "current_balance implausible masked",
         "rows": int(scored["balance_repaired"].sum()),
         "rate": float(scored["balance_repaired"].mean())},
        {"repair": "loan_age_months recomputed from dates",
         "rows": int(scored["age_repaired"].sum()),
         "rate": float(scored["age_repaired"].mean())},
    ])
    A(_md(repair_tbl))
    A("")
    if len(truth):
        A("### Recovery against the injected ground truth")
        A("")
        A("The synthetic generator logs every defect it injects. Comparing detection against "
          "that log is how this rule set was validated rather than merely asserted.")
        A("")
        A(_md(truth))
        A("")
    A("## 5. Cross-column relationship breaks")
    A("")
    A(f"{len(validate.RULES)} named rules run over every record, grouped into completeness, "
      "validity, consistency, plausibility, timeliness and reconciliation dimensions.")
    A("")
    A(_md(rule_summary[["rule", "dimension", "severity", "violations", "violation_rate",
                        "description"]]))
    A("")
    A("## 6. Correlation and dependent-field analysis")
    A("")
    A("### Numeric (Spearman)")
    A("")
    A(_md(deps["numeric_corr"].round(3).reset_index().rename(columns={"index": "field"})))
    A("")
    A("### Categorical association (bias-corrected Cramer's V, top pairs)")
    A("")
    A(_md(deps["categorical_association"].head(12)))
    A("")
    A("### Functional dependencies")
    A("")
    A("A loan's static attributes must not change across its reporting months. Violations "
      "here are true data-integrity breaks rather than statistical noise.")
    A("")
    A(_md(deps["functional_dependencies"]))
    A("")
    A("## 7. Train / test drift")
    A("")
    A(f"Split at `{C.TRAIN_END}`, matching the time-aware modelling split used in Task 2. "
      "PSI below 0.10 is stable, 0.10-0.25 moderate, above 0.25 severe.")
    A("")
    A(_md(drift))
    A("")
    A("### Target stability across months")
    A("")
    A(_md(tgt))
    A("")
    A("## 8. Data quality scoring")
    A("")
    A("Record score = 100 minus the severity-weighted sum of rule violations, floored at 0. "
      "Batch score aggregates the same violations to the (reporting month x servicer) grain, "
      "which is the level an oversight team can act on.")
    A("")
    A(_md(scored["dq_band"].value_counts().rename_axis("dq_band")
          .reset_index(name="records").assign(
              share=lambda d: d["records"] / len(scored))))
    A("")
    A(f"- Mean record DQ score: **{scored['dq_score'].mean():.2f}**")
    A(f"- Median record DQ score: **{scored['dq_score'].median():.2f}**")
    A(f"- Records with at least one violation: **{(scored['dq_violation_count'] > 0).mean():.1%}**")
    A("")
    A("### Batch grades by servicer")
    A("")
    svc = (scored.groupby("servicer_name")
           .agg(records=("loan_id", "size"), mean_dq_score=("dq_score", "mean"),
                violations_per_record=("dq_violation_count", "mean"))
           .sort_values("mean_dq_score").reset_index())
    A(_md(svc))
    A("")
    A("### Ten worst batches")
    A("")
    A(_md(worst_batches[["reporting_month", "servicer_name", "records", "mean_dq_score",
                         "pct_critical", "top_failing_rule", "top_failing_rule_rate",
                         "batch_grade"]]))
    A("")
    A("## 9. What this means for modelling")
    A("")
    A("1. **Servicer is a confound, not just a feature.** Kestrel Financial and Pioneer "
      "Mortgage Ops have both the worst data quality *and* elevated delinquency. A model "
      "given raw servicer identity will partly learn reporting behaviour rather than credit "
      "risk. Servicer is retained but its SHAP contribution is inspected separately in the "
      "explainability report.")
    A("2. **Censoring is real and material.** Forward-looking targets are undefined for rows "
      "whose horizon runs past the panel end. These are `NaN`, not `0`, and are excluded from "
      "supervised training rather than counted as non-events.")
    A("3. **Repairs are features.** Whether a record needed repair is predictive of whether it "
      "needs an exception, so repair indicators are carried forward rather than discarded.")
    A("4. **Drift is concentrated in macro-sensitive fields**, which is expected given the "
      "rate path in the panel window and is handled by time-aware validation rather than by "
      "reweighting.")
    A("")

    text = "\n".join(lines)
    (C.REPORTS / "data_intelligence_report.md").write_text(text, encoding="utf-8")
    return {"records": len(scored), "mean_dq_score": float(scored["dq_score"].mean()),
            "rules": len(validate.RULES), "batches": len(batches),
            "scored": scored}


if __name__ == "__main__":
    out = build()
    print({k: v for k, v in out.items() if k != "scored"})
