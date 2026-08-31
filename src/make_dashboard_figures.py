
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
SUBMISSION = ROOT / "submission"

INK = "#1f2933"
MUTED = "#8b97a5"
ACCENT = "#2f6f9f"
WARN = "#c25b3c"
GOOD = "#3f8f6f"

TARGET_LABELS = {
    "next_3m_delinquency_flag": "Delinquency 3m",
    "next_6m_delinquency_flag": "Delinquency 6m",
    "next_12m_default_flag": "Default 12m",
    "next_12m_prepayment_flag": "Prepayment 12m",
    "exception_required": "Exception",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 140,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "axes.edgecolor": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "grid.color": "#e3e8ee",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    path = FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def figure_drift() -> None:
    drift = pd.read_csv(REPORTS / "drift_report.csv")
    drift = drift.dropna(subset=["psi"]).sort_values("psi", ascending=False).head(12)
    colours = [
        WARN if s == "severe" else ACCENT if s == "moderate" else MUTED
        for s in drift["severity"]
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.barh(drift["column"][::-1], drift["psi"][::-1], color=colours[::-1])
    ax.axvline(0.25, color=INK, linestyle="--", linewidth=0.9)
    ax.axvline(0.10, color=MUTED, linestyle=":", linewidth=0.9)
    ax.set_xlabel("Population stability index (train vs test)")
    ax.set_title("Feature drift across the purged out-of-time split")
    ax.grid(axis="x", linewidth=0.6)
    ax.set_axisbelow(True)
    _save(fig, "drift_psi.png")


def figure_leakage() -> None:
    probe = pd.read_csv(REPORTS / "leakage_probe.csv")
    probe["label"] = probe["target"].map(TARGET_LABELS).fillna(probe["target"])
    x = range(len(probe))
    width = 0.27
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar([i - width for i in x], probe["purged_time_split"], width,
           label="Purged time split (reported)", color=ACCENT)
    ax.bar(list(x), probe["loan_disjoint_time_split"], width,
           label="Loan-disjoint control", color=GOOD)
    ax.bar([i + width for i in x], probe["random_row_split_unsound"], width,
           label="Random row split (unsound)", color=WARN)
    for i, row in probe.reset_index(drop=True).iterrows():
        gap = row["random_split_inflation"]
        if abs(gap) > 0.05:
            ax.annotate(
                f"+{gap:.2f}",
                xy=(i + width, row["random_row_split_unsound"]),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=WARN,
                fontweight="bold",
            )
    ax.set_xticks(list(x))
    ax.set_xticklabels(probe["label"])
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("Leakage ablation: what an unsound split would have claimed")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(axis="y", linewidth=0.6)
    ax.set_axisbelow(True)
    _save(fig, "leakage_ablation.png")


def _test_frame() -> pd.DataFrame:
    metrics = pd.concat(
        [
            pd.read_csv(REPORTS / "model_metrics.csv"),
            pd.read_csv(REPORTS / "exception_binary_metrics.csv"),
        ],
        ignore_index=True,
    )
    return metrics[
        (metrics["split"] == "test")
        & (metrics["model"].isin(["baseline_logistic", "lgbm_calibrated"]))
    ]


def _paired(test: pd.DataFrame, target: str, model: str, metric: str) -> float:
    row = test[(test["target"] == target) & (test["model"] == model)]
    return float(row[metric].iloc[0])


def figure_model_comparison() -> None:
    test = _test_frame()
    order = list(TARGET_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for ax, metric, title in zip(axes, ["roc_auc", "pr_auc"], ["ROC-AUC", "PR-AUC"]):
        base = [_paired(test, t, "baseline_logistic", metric) for t in order]
        lgbm = [_paired(test, t, "lgbm_calibrated", metric) for t in order]
        x = range(len(order))
        ax.bar([i - 0.2 for i in x], base, 0.4, label="Logistic baseline", color=MUTED)
        ax.bar([i + 0.2 for i in x], lgbm, 0.4, label="LightGBM calibrated", color=ACCENT)
        ax.set_xticks(list(x))
        ax.set_xticklabels([TARGET_LABELS[t] for t in order], rotation=25, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Out-of-time test performance against the nine-feature baseline", fontweight="bold")
    _save(fig, "model_comparison.png")


def figure_calibration() -> None:
    test = _test_frame()
    order = list(TARGET_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    pairs = [("brier", "Brier score (lower is better)"), ("ece", "Expected calibration error")]
    for ax, (metric, title) in zip(axes, pairs):
        base = [_paired(test, t, "baseline_logistic", metric) for t in order]
        lgbm = [_paired(test, t, "lgbm_calibrated", metric) for t in order]
        x = range(len(order))
        ax.bar([i - 0.2 for i in x], base, 0.4, label="Logistic baseline", color=WARN)
        ax.bar([i + 0.2 for i in x], lgbm, 0.4, label="LightGBM calibrated", color=ACCENT)
        ax.set_xticks(list(x))
        ax.set_xticklabels([TARGET_LABELS[t] for t in order], rotation=25, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Where the gradient boosting wins: calibration, not ranking", fontweight="bold")
    _save(fig, "calibration_comparison.png")


def figure_survival() -> None:
    cif = pd.read_csv(REPORTS / "cumulative_incidence.csv")
    km_def = pd.read_csv(REPORTS / "km_default_curve.csv")
    km_pre = pd.read_csv(REPORTS / "km_prepay_curve.csv")
    km_def = km_def[km_def["group"] == "all"]
    km_pre = km_pre[km_pre["group"] == "all"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    axes[0].plot(km_pre["loan_age_months"], km_pre["cumulative_event_prob"],
                 color=ACCENT, label="Prepayment")
    axes[0].plot(km_def["loan_age_months"], km_def["cumulative_event_prob"],
                 color=WARN, label="90+ DPD proxy")
    axes[0].set_xlabel("Loan age (months)")
    axes[0].set_ylabel("Cumulative event probability")
    axes[0].set_title("Kaplan-Meier time to event")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(linewidth=0.6)
    axes[0].set_axisbelow(True)
    axes[1].plot(cif["loan_age_months"], cif["naive_1_minus_km_default"],
                 color=MUTED, linestyle="--", label="Naive 1 - KM")
    axes[1].plot(cif["loan_age_months"], cif["cif_default"],
                 color=WARN, label="CIF default (competing risks)")
    axes[1].fill_between(
        cif["loan_age_months"],
        cif["cif_default"],
        cif["naive_1_minus_km_default"],
        color=WARN,
        alpha=0.12,
        label="Overstatement from ignoring prepayment",
    )
    axes[1].set_xlabel("Loan age (months)")
    axes[1].set_title("Competing risks correction")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(linewidth=0.6)
    axes[1].set_axisbelow(True)
    _save(fig, "survival_curves.png")


def figure_transition_matrix() -> None:
    tm = pd.read_csv(REPORTS / "markov_transition_matrix.csv").set_index("current_status")
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    im = ax.imshow(tm.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(tm.columns)))
    ax.set_xticklabels(tm.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(tm.index)))
    ax.set_yticklabels(tm.index, fontsize=8)
    for i in range(tm.shape[0]):
        for j in range(tm.shape[1]):
            v = tm.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.5 else INK)
    ax.set_title("Monthly transition matrix (from row, to column)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, "transition_matrix.png")


def figure_anomaly_distribution() -> None:
    sub = pd.read_csv(SUBMISSION / "submission.csv", usecols=["anomaly_score"])
    queue = pd.read_csv(REPORTS / "anomaly_review_queue.csv")
    cutoff = float(queue["anomaly_score"].min())
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.hist(sub["anomaly_score"], bins=70, color=ACCENT, alpha=0.85)
    ax.axvline(cutoff, color=WARN, linestyle="--", linewidth=1.1)
    ax.set_yscale("log")
    ax.annotate(
        f"lowest score in the 40-record\nreview queue ({cutoff:.3f})",
        xy=(cutoff, ax.get_ylim()[1] * 0.25),
        xytext=(10, 0),
        textcoords="offset points",
        fontsize=8,
        color=WARN,
    )
    ax.set_xlabel("Isolation forest anomaly score")
    ax.set_ylabel("Records (log scale)")
    ax.set_title("Anomaly score distribution across the scored panel")
    ax.grid(axis="y", linewidth=0.6)
    ax.set_axisbelow(True)
    _save(fig, "anomaly_distribution.png")


def figure_scenarios() -> None:
    paths = pd.read_csv(REPORTS / "scenario_markov_paths.csv")
    headline = pd.read_csv(REPORTS / "scenario_headline.csv")
    colours = {"base": MUTED, "adverse_credit": WARN, "high_prepayment": ACCENT}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    for name, group in paths.groupby("scenario_name"):
        group = group.sort_values("horizon_month")
        axes[0].plot(group["horizon_month"], group["Default"] * 100,
                     label=name, color=colours.get(name, INK), linewidth=1.8)
    axes[0].set_xlabel("Horizon (months)")
    axes[0].set_ylabel("Cumulative default share (%)")
    axes[0].set_title("Engine B: macro-conditioned Markov paths")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(linewidth=0.6)
    axes[0].set_axisbelow(True)
    order = ["base", "adverse_credit", "high_prepayment"]
    headline = headline.set_index("scenario_name").loc[order].reset_index()
    x = range(len(order))
    axes[1].bar([i - 0.2 for i in x], headline["projected_next_12m_default_flag"] * 100, 0.4,
                label="Default 12m", color=WARN)
    axes[1].bar([i + 0.2 for i in x], headline["projected_next_12m_prepayment_flag"] * 100, 0.4,
                label="Prepayment 12m", color=ACCENT)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(order, rotation=15, ha="right", fontsize=8)
    axes[1].set_ylabel("Projected rate (%)")
    axes[1].set_title("Engine A: loan-level repricing projections")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", linewidth=0.6)
    axes[1].set_axisbelow(True)
    _save(fig, "scenario_projections.png")


def figure_prepay_non_monotone() -> None:
    seg = pd.read_csv(REPORTS / "scenario_segment_prepay_by_rate_incentive.csv")
    order = ["<-1.0", "-1.0 to -0.5", "-0.5 to 0", "0 to 0.5", "0.5 to 1.0", ">1.0"]
    seg = seg.set_index("incentive_bucket").loc[order].reset_index()
    values = seg["delta_high_prepayment"] * 100
    peak = int(values.idxmax())
    colours = [WARN if i == peak else ACCENT for i in range(len(seg))]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.bar(seg["incentive_bucket"], values, color=colours)
    for i, v in enumerate(values):
        ax.annotate(f"{v:.1f}pp", (i, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8)
    ax.set_xlabel("Refinance incentive bucket (out of the money to deep in the money)")
    ax.set_ylabel("Change in projected 12m prepayment")
    ax.set_title("Not monotone in incentive: the deep-in-the-money loans are already saturated")
    ax.grid(axis="y", linewidth=0.6)
    ax.set_axisbelow(True)
    _save(fig, "prepay_non_monotone.png")


def figure_shap() -> None:
    shap_df = pd.read_csv(REPORTS / "shap_global_next_12m_default_flag.csv").head(14)
    colours = [
        WARN if d == "higher value raises risk" else GOOD if d == "higher value lowers risk" else MUTED
        for d in shap_df["direction"]
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.barh(shap_df["plain_english"][::-1], shap_df["mean_abs_shap"][::-1], color=colours[::-1])
    ax.set_xlabel("Mean absolute SHAP contribution (log-odds)")
    ax.set_title("Global drivers of the 12-month default proxy")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=WARN),
        plt.Rectangle((0, 0), 1, 1, color=GOOD),
        plt.Rectangle((0, 0), 1, 1, color=MUTED),
    ]
    ax.legend(handles, ["raises risk", "lowers risk", "non-monotone / categorical"],
              frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="x", linewidth=0.6)
    ax.set_axisbelow(True)
    _save(fig, "shap_global_default.png")


def main() -> None:
    _style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure_drift()
    figure_leakage()
    figure_model_comparison()
    figure_calibration()
    figure_survival()
    figure_transition_matrix()
    figure_anomaly_distribution()
    figure_scenarios()
    figure_prepay_non_monotone()
    figure_shap()


if __name__ == "__main__":
    main()
