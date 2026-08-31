from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.data import generate_synthetic as G
from src.data import messiness as M
from src.data.exceptions_label import build_exception_labels, reconcile_servicer_feed

N_LOANS = 1500


def build_macro_scenarios(macro: pd.DataFrame) -> pd.DataFrame:
    last = macro.iloc[-1]
    horizon = 12
    rows = []
    paths = {
        "base": dict(rate=0.00, unemp=0.02, hpi=0.0005),
        "adverse_credit": dict(rate=0.03, unemp=0.19, hpi=-0.0060),
        "high_prepayment": dict(rate=-0.14, unemp=0.01, hpi=0.0028),
    }
    notes = {
        "base": "Macro path continues its current trajectory; no shock applied.",
        "adverse_credit": "Unemployment rises ~2.3pp over 12 months and HPI turns negative; "
                          "rates broadly flat. Stresses default and delinquency transitions.",
        "high_prepayment": "Market mortgage rate falls ~1.7pp over 12 months, opening refinance "
                           "incentive across seasoned high-coupon loans.",
    }
    for name, d in paths.items():
        for h in range(1, horizon + 1):
            rows.append({
                "scenario_name": name,
                "horizon_month": h,
                "market_mortgage_rate": round(last["market_mortgage_rate"] + d["rate"] * h, 4),
                "unemployment_rate": round(last["unemployment_rate"] + d["unemp"] * h, 4),
                "hpi_yoy_growth": round(last["hpi_yoy_growth"] + d["hpi"] * h, 5),
                "assumption_note": notes[name],
            })
    return pd.DataFrame(rows)


DICTIONARY = [
    ("loan_id", "string", "Unique loan identifier, stable across reporting months.", "LNxxxxxx", "core_servicing"),
    ("month_index", "int", "Zero-based index of the reporting month within the panel window.", "0..N-1 where N is the panel length in months", "derived"),
    ("reporting_month", "period[M]", "Calendar month the record describes.", "YYYY-MM", "core_servicing"),
    ("origination_month", "period[M]", "Month the loan was originated.", "YYYY-MM", "core_servicing"),
    ("loan_age_months", "int", "Months elapsed since origination as reported by the servicer.", ">=0", "core_servicing"),
    ("remaining_term_months", "int", "Contractual months remaining to maturity.", ">=0", "core_servicing"),
    ("original_balance", "float", "Balance at origination in USD.", "45000..1250000", "core_servicing"),
    ("current_balance", "float", "Unpaid principal balance at month end in USD.", ">=0", "core_servicing"),
    ("interest_rate", "float", "Note rate, annual percent.", "2.25..11.5", "core_servicing"),
    ("credit_score_band", "category", "Borrower FICO band at origination.", "|".join(C.CREDIT_BANDS), "core_servicing"),
    ("ltv_band", "category", "Loan-to-value band at origination.", "|".join(C.LTV_BANDS), "core_servicing"),
    ("dti_band", "category", "Debt-to-income band at origination.", "|".join(C.DTI_BANDS), "core_servicing"),
    ("state", "category", "US state of the collateral property.", "2-letter code", "core_servicing"),
    ("loan_purpose", "category", "Purpose at origination.", "|".join(C.LOAN_PURPOSES), "core_servicing"),
    ("occupancy_type", "category", "Occupancy classification.", "|".join(C.OCCUPANCY_TYPES), "core_servicing"),
    ("property_type", "category", "Collateral property type.", "|".join(C.PROPERTY_TYPES), "core_servicing"),
    ("servicer_name", "category", "Servicer responsible for the loan in this month.", "|".join(C.SERVICERS), "core_servicing"),
    ("current_status", "category", "Performance status at month end.", "|".join(C.STATES), "core_servicing"),
    ("days_past_due", "float", "Days past due at month end. 9999 and -1 appear as sentinel values.", ">=0 expected", "core_servicing"),
    ("modification_flag", "int", "1 once a loss-mitigation modification has been applied.", "0|1", "core_servicing"),
    ("prepayment_flag", "int", "1 if the loan prepaid in full effective next month.", "0|1", "derived"),
    ("default_flag", "int", "1 if the loan entered default effective next month.", "0|1", "derived"),
    ("loss_severity_band", "category", "Realised loss severity band, populated only on default.", "|".join(C.LOSS_SEVERITY_BANDS), "investor_feed"),
    ("last_updated_at", "datetime", "Timestamp the servicing record was last written.", "ISO datetime", "core_servicing"),
    ("source_system", "category", "System of record the row arrived from.", "|".join(C.SOURCE_SYSTEMS), "metadata"),
    ("document_status", "category", "Collateral/document file completeness.", "|".join(C.DOC_STATUSES), "doc_custody"),
    ("next_3m_delinquency_flag", "float", "1 if 30+ DPD occurs in months t+1..t+3. NaN when right-censored.", "0|1|NaN", "target"),
    ("next_6m_delinquency_flag", "float", "1 if 30+ DPD occurs in months t+1..t+6. NaN when right-censored.", "0|1|NaN", "target"),
    ("next_12m_default_flag", "float", "1 if default occurs in months t+1..t+12. NaN when right-censored.", "0|1|NaN", "target"),
    ("next_12m_prepayment_flag", "float", "1 if voluntary prepayment occurs in months t+1..t+12. NaN when right-censored.", "0|1|NaN", "target"),
    ("next_state", "category", "Performance status at month t+1.", "|".join(C.STATES), "target"),
    ("exception_required", "int", "1 if a servicing-oversight exception should be raised on this record.", "0|1", "target"),
    ("exception_type", "category", "Category of the exception raised.", "|".join(C.EXCEPTION_TYPES), "target"),
]


def main(n_loans: int = N_LOANS, seed: int = C.RANDOM_SEED) -> dict:
    rng = np.random.default_rng(seed)
    months = G.month_range(C.PANEL_START, C.PANEL_END)

    macro = G.build_macro_history(months, rng)
    loans = G.build_loan_book(n_loans, months, rng)
    panel = G.simulate_panel(loans, macro, months, rng)
    panel = G.attach_forward_targets(panel, last_month_index=int(panel["month_index"].max()))

    messy, defect_log = M.inject(panel, loans, rng)
    updates = M.build_servicer_updates(messy, rng)
    reconciled = reconcile_servicer_feed(messy, updates)
    labelled = build_exception_labels(reconciled, updates, rng)

    keep = (C.RAW_COLUMNS + C.BINARY_TARGETS
            + ["next_state", "exception_required", "exception_type"])
    out = labelled[[c for c in keep if c in labelled.columns]].copy()
    out = out.sample(frac=1.0, random_state=17).reset_index(drop=True)

    out.to_csv(C.LOAN_PANEL, index=False)
    updates.to_csv(C.SERVICER_UPDATES, index=False)
    macro.to_csv(C.MACRO_HISTORY, index=False)
    build_macro_scenarios(macro).to_csv(C.MACRO_SCENARIOS, index=False)
    pd.DataFrame(DICTIONARY, columns=["field", "dtype", "description", "allowed_values",
                                      "source_system"]).to_csv(C.DATA_DICTIONARY, index=False)
    defect_log.to_csv(C.DATA_RAW / "ground_truth_defect_log.csv", index=False)
    from src.data.validate import export_rules_json
    export_rules_json()

    out.head(500).to_csv(C.DATA_SAMPLES / "loan_panel_sample.csv", index=False)
    updates.head(300).to_csv(C.DATA_SAMPLES / "servicer_updates_sample.csv", index=False)

    summary = {
        "loans": int(out["loan_id"].nunique()),
        "rows": int(len(out)),
        "months": len(months),
        "servicer_update_rows": int(len(updates)),
        "status_mix": out["current_status"].value_counts(normalize=True).round(4).to_dict(),
        "target_rates": {t: float(out[t].mean(skipna=True)) for t in C.BINARY_TARGETS},
        "censored_rows": {t: int(out[t].isna().sum()) for t in C.BINARY_TARGETS},
        "exception_rate": float(out["exception_required"].mean()),
        "exception_mix": out["exception_type"].value_counts(normalize=True).round(4).to_dict(),
    }
    return summary


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2, default=str))
