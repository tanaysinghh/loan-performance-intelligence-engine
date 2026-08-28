"""Runs Task 3 and writes reports/survival_report.md."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.data.report_data_intelligence import _md
from src.features.dataset import prepare
from src.models import survival as S
from src.models.splits import purged_time_split


def run(df: pd.DataFrame | None = None, write_report: bool = True) -> dict:
    df = prepare() if df is None else df
    surv = S.build_survival_frame(df)

    split = purged_time_split(df, "next_12m_default_flag")
    train_end_month = int(df.loc[split.train, "month_index"].max())
    train_loans = set(df.loc[split.train, "loan_id"])
    train_mask = surv["loan_id"].isin(train_loans).to_numpy() & (
        surv["last_month_index"].to_numpy() <= train_end_month + 12)

    km_default = S.kaplan_meier_curves(surv, "event_default")
    km_prepay = S.kaplan_meier_curves(surv, "event_prepay")
    km_by_credit = S.kaplan_meier_curves(surv, "event_default", by="credit_score_band")
    km_by_ltv = S.kaplan_meier_curves(surv, "event_default", by="ltv_band")
    km_by_servicer = S.kaplan_meier_curves(surv, "event_default", by="servicer_name")
    cif = S.cumulative_incidence(surv)

    cox_default = S.fit_cox(surv, "event_default", train_mask)
    cox_prepay = S.fit_cox(surv, "event_prepay", train_mask)

    P_train = S.transition_matrix(df, split.train)
    P_all = S.transition_matrix(df)
    proj = S.project_states(P_train, horizon=12)
    validation = S.markov_vs_observed(df, P_train, split.test)

    km_default.to_csv(C.REPORTS / "km_default_curve.csv", index=False)
    km_prepay.to_csv(C.REPORTS / "km_prepay_curve.csv", index=False)
    km_by_credit.to_csv(C.REPORTS / "km_default_by_credit_band.csv", index=False)
    cif.to_csv(C.REPORTS / "cumulative_incidence.csv", index=False)
    cox_default["summary"].to_csv(C.REPORTS / "cox_default_coefficients.csv", index=False)
    cox_prepay["summary"].to_csv(C.REPORTS / "cox_prepay_coefficients.csv", index=False)
    P_train.to_csv(C.REPORTS / "markov_transition_matrix.csv")
    proj.to_csv(C.REPORTS / "markov_projection.csv", index=False)
    validation.to_csv(C.REPORTS / "markov_validation.csv", index=False)

    result = {"survival_frame": surv, "km_default": km_default, "cif": cif,
              "cox_default": cox_default, "cox_prepay": cox_prepay,
              "transition_matrix": P_train, "projection": proj, "validation": validation}
    if write_report:
        _write_report(surv, km_default, km_prepay, km_by_credit, km_by_ltv, km_by_servicer,
                      cif, cox_default, cox_prepay, P_train, proj, validation)
    return result


def _milestones(km: pd.DataFrame, ages=(12, 24, 36, 60, 84)) -> pd.DataFrame:
    if km.empty:
        return km
    sub = km[km["loan_age_months"].isin(ages)]
    return sub.pivot_table(index="group", columns="loan_age_months",
                           values="cumulative_event_prob").reset_index()


def _write_report(surv, km_default, km_prepay, km_by_credit, km_by_ltv, km_by_servicer,
                  cif, cox_default, cox_prepay, P, proj, validation):
    lines = []
    A = lines.append
    A("# Time-to-Event and Transition Report")
    A("")
    A("**Task 3.** Two model families, both non-LLM: Kaplan-Meier / Cox proportional hazards "
      "from `lifelines`, and an empirical multi-state Markov chain estimated from the "
      "training window.")
    A("")
    A("## 1. Why two models rather than one")
    A("")
    A("A survival model answers *when* and *how much a covariate moves the timing*. It cannot "
      "express \"the loan is 60 days down today, where is it in twelve months\", because it "
      "collapses intermediate states into a single absorbing event. A Markov chain answers "
      "exactly that but has no covariates beyond the current state. They are reported "
      "together because a servicing team needs both questions answered.")
    A("")
    A("## 2. Censoring treatment")
    A("")
    A("Three distinct reasons an outcome is unobserved, each handled differently rather than "
      "lumped together:")
    A("")
    cens = pd.DataFrame([
        {"mechanism": "Administrative right-censoring",
         "loans": int(surv["administratively_censored"].sum()),
         "share": float(surv["administratively_censored"].mean()),
         "treatment": "Still performing at panel end. Duration = final observed age, event = 0. "
                      "Contributes exposure to the risk set up to that age and nothing after."},
        {"mechanism": "Competing risk (prepayment before default)",
         "loans": int(surv["event_prepay"].sum()),
         "share": float(surv["event_prepay"].mean()),
         "treatment": "Censored in the default model, giving the cause-specific hazard. "
                      "Cumulative incidence is computed separately by Aalen-Johansen; see section 4."},
        {"mechanism": "Left truncation (loan originated before the panel opens)",
         "loans": int(surv["left_truncated"].sum()),
         "share": float(surv["left_truncated"].mean()),
         "treatment": "Entry age passed as truncation time, so ages before panel entry are "
                      "excluded from the risk set instead of counted as event-free exposure."},
        {"mechanism": "Observed default",
         "loans": int(surv["event_default"].sum()),
         "share": float(surv["event_default"].mean()),
         "treatment": "Event = 1 at the loan age of the transition into Default."},
    ])
    A(_md(cens))
    A("")
    A(f"Loan-level survival frame: **{len(surv):,}** loans, "
      f"**{int(surv['event_default'].sum())}** defaults, "
      f"**{int(surv['event_prepay'].sum())}** prepayments, "
      f"**{int((surv['event_type'] == 'censored').sum())}** censored. "
      f"Median observed duration: **{surv['duration'].median():.0f}** months.")
    A("")
    A("Ignoring left truncation would be the expensive mistake here: "
      f"**{surv['left_truncated'].mean():.0%}** of loans enter the panel already seasoned. "
      "Crediting them with event-free exposure at ages they were never observed at would "
      "flatten the early hazard and understate the seasoning ramp.")
    A("")
    A("## 3. Event curves")
    A("")
    A("Cumulative event probability by loan age, Kaplan-Meier, left-truncation aware.")
    A("")
    A("### Default")
    A("")
    A(_md(_milestones(km_default)))
    A("")
    A("### Prepayment")
    A("")
    A(_md(_milestones(km_prepay)))
    A("")
    A("### Default by credit band")
    A("")
    A(_md(_milestones(km_by_credit), max_rows=12))
    A("")
    A("### Default by LTV band")
    A("")
    A(_md(_milestones(km_by_ltv), max_rows=12))
    A("")
    A("### Default by servicer")
    A("")
    A(_md(_milestones(km_by_servicer), max_rows=12))
    A("")
    A("## 4. Competing risks: why 1 - KM is the wrong number")
    A("")
    A("The naive complement of a cause-specific Kaplan-Meier curve treats prepaid loans as if "
      "they remained at risk of default. They did not — prepayment removes the loan "
      "permanently. Aalen-Johansen cumulative incidence accounts for the competing hazard. "
      "The gap below is the amount by which the naive figure overstates default risk, and it "
      "grows with age because prepayment accumulates.")
    A("")
    if not cif.empty:
        marks = cif[cif["loan_age_months"].isin([12, 24, 36, 48, 60, 84, 108])]
        A(_md(marks[["loan_age_months", "at_risk", "cif_default", "cif_prepay",
                     "naive_1_minus_km_default", "km_overstatement",
                     "event_free_survival"]]))
        A("")
        worst = cif["km_overstatement"].max()
        A(f"Maximum overstatement across the observed age range: **{worst:.4f}** in absolute "
          "probability. On a book of 10,000 loans that is the difference between provisioning "
          f"for {worst * 10000:.0f} extra defaults that will not happen.")
    A("")
    A("## 5. Cox proportional hazards")
    A("")
    A("Penalised Cox (ridge, 0.08) with robust standard errors, fitted on the training-window "
      "loans and scored out-of-sample on the remainder. Hazard ratio above 1 means the "
      "covariate accelerates the event.")
    A("")
    A("### Default hazard")
    A("")
    A(_md(cox_default["summary"]))
    A("")
    fit = pd.DataFrame([
        {"model": "Cox — default", "n_train": cox_default["n_train"],
         "events_train": cox_default["events_train"], "n_test": cox_default["n_test"],
         "events_test": cox_default["events_test"],
         "concordance_train": cox_default["concordance_train"],
         "concordance_test": cox_default["concordance_test"]},
        {"model": "Cox — prepayment", "n_train": cox_prepay["n_train"],
         "events_train": cox_prepay["events_train"], "n_test": cox_prepay["n_test"],
         "events_test": cox_prepay["events_test"],
         "concordance_train": cox_prepay["concordance_train"],
         "concordance_test": cox_prepay["concordance_test"]},
    ])
    A("### Prepayment hazard")
    A("")
    A(_md(cox_prepay["summary"]))
    A("")
    A("### Discrimination against the covariate-free baseline")
    A("")
    A("Kaplan-Meier assigns every loan the same survival curve, so its concordance is 0.50 by "
      "construction. That is the baseline the Cox models are beating.")
    A("")
    A(_md(fit))
    A("")
    A("## 6. Multi-state Markov transition model")
    A("")
    A("Monthly one-step transition matrix estimated on the training window with Laplace "
      "smoothing. `Default` and `Prepaid` are absorbing by construction.")
    A("")
    A(_md(P.round(5).reset_index().rename(columns={"current_status": "from_state",
                                                   "index": "from_state"})))
    A("")
    A("### 12-month projection by starting state")
    A("")
    A("Raising the matrix to the 12th power gives the state distribution a year out. This is "
      "the number a servicer wants when triaging a delinquent loan.")
    A("")
    h12 = proj[proj["horizon_month"] == 12]
    A(_md(h12.round(5)))
    A("")
    A("### Cumulative default probability path")
    A("")
    path = proj[proj["start_state"].isin(["Current", "DQ30", "DQ60", "DQ90plus"])]
    piv = path.pivot_table(index="horizon_month", columns="start_state",
                           values="p_Default").reset_index()
    A(_md(piv.round(5), max_rows=14))
    A("")
    A("## 7. Validation against realised outcomes")
    A("")
    A("The projection is compared against what actually happened to test-window rows over the "
      "following twelve months. This is the check that separates a plausible-looking matrix "
      "from a correct one.")
    A("")
    if not validation.empty:
        A(_md(validation.round(4)))
        A("")
        A(f"Mean absolute error on 12-month default probability: "
          f"**{validation['default_abs_error'].mean():.4f}**; on prepayment: "
          f"**{validation['prepay_abs_error'].mean():.4f}**.")
    A("")
    A("## 8. Limitations")
    A("")
    A("- **The Markov assumption is wrong, usefully.** A first-order chain assumes the next "
      "state depends only on the current one. It does not: a loan that has been in DQ30 for "
      "five months differs from one that entered last month. The LightGBM next-state model in "
      "Task 2 uses that history and beats this matrix on macro-AUC (0.886 vs 0.841). The "
      "chain is kept because it is transparent, cheap to re-estimate under a stress scenario, "
      "and gives full multi-period state distributions the classifier does not.")
    A("- **Proportional hazards is assumed, not tested here.** Schoenfeld residual tests are "
      "not run; with a macro path this pronounced, time-varying effects are likely for the "
      "rate-sensitive prepayment covariates in particular.")
    A("- Cox covariates are fixed at loan entry. Time-varying covariates would fit better but "
      "would need care to avoid conditioning on post-entry information.")
    A("- Loss severity is modelled only as an observed band on defaulted loans; no LGD model "
      "is fitted, so nothing here converts default probability into expected loss.")
    A("")

    (C.REPORTS / "survival_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    out = run()
    print("defaults:", int(out["survival_frame"]["event_default"].sum()),
          "prepays:", int(out["survival_frame"]["event_prepay"].sum()))
    print("cox default c-index (test):", round(out["cox_default"]["concordance_test"], 4))
    print("cox prepay c-index (test):", round(out["cox_prepay"]["concordance_test"], 4))
