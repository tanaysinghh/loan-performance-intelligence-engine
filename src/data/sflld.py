from __future__ import annotations

from pathlib import Path


ORIG = {
    "credit_score": 0,
    "first_payment_date": 1,
    "first_time_homebuyer": 2,
    "maturity_date": 3,
    "msa": 4,
    "mi_pct": 5,
    "num_units": 6,
    "occupancy": 7,
    "cltv": 8,
    "dti": 9,
    "original_upb": 10,
    "ltv": 11,
    "original_rate": 12,
    "channel": 13,
    "ppm_flag": 14,
    "amortization_type": 15,
    "property_state": 16,
    "property_type": 17,
    "postal_code": 18,
    "loan_seq": 19,
    "loan_purpose": 20,
    "original_term": 21,
    "num_borrowers": 22,
    "seller_name": 23,
    "super_conforming": 24,
    "pre_harp_loan_seq": 25,
    "special_eligibility": 26,
    "relief_refinance": 27,
    "property_valuation_method": 28,
    "interest_only": 29,
    "_constant_9999": 30,
}
ORIG_NCOLS = 31

PERF = {
    "loan_seq": 0,
    "reporting_month": 1,
    "current_upb": 2,
    "delinquency_status": 3,
    "loan_age": 4,
    "remaining_months": 5,
    "defect_settlement_date": 6,
    "modification_flag": 7,
    "zero_balance_code": 8,
    "zero_balance_date": 9,
    "current_rate": 10,
    "deferred_upb": 11,
    "ddlpi": 12,
    "mi_recoveries": 13,
    "net_sale_proceeds": 14,
    "non_mi_recoveries": 15,
    "expenses": 16,
    "legal_costs": 17,
    "maintenance_costs": 18,
    "taxes_insurance": 19,
    "misc_expenses": 20,
    "actual_loss": 21,
    "cumulative_modification_cost": 22,
    "step_modification_flag": 23,
    "payment_deferral": 24,
    "eltv": 25,
    "zero_balance_removal_upb": 26,
    "delinquent_accrued_interest": 27,
    "delinquency_due_to_disaster": 28,
    "borrower_assistance_code": 29,
    "current_month_modification_cost": 30,
    "interest_bearing_upb": 31,
    "mi_cancellation_indicator": 32,
    "servicer_name": 33,
    "_filler": 34,
}
PERF_NCOLS = 35

VINTAGES = ["2019", "2020", "2021", "2022", "2023"]

PANEL_FIRST_MONTH = "2019-01"
PANEL_LAST_MONTH = "2026-03"


OCCUPANCY_MAP = {"P": "primary", "S": "second_home", "I": "investment"}
PURPOSE_MAP = {"P": "purchase", "N": "rate_term_refi", "C": "cash_out_refi"}
PROPERTY_MAP = {"SF": "single_family", "PU": "pud", "CO": "condo",
                "MH": "manufactured", "CP": "condo"}

ZBC_PREPAID = {"01"}
ZBC_CREDIT_EVENT = {"02", "03", "09", "15"}
ZBC_OTHER_EXIT = {"96", "16"}

SENTINEL_CREDIT_SCORE = {"9999", ""}
SENTINEL_DTI = {"999", ""}
SENTINEL_LTV = {"999", ""}


def vintage_paths(dataset_dir: Path, vintage: str) -> tuple[Path, Path]:
    folder = dataset_dir / f"sample_{vintage}"
    return (folder / f"sample_orig_{vintage}.txt",
            folder / f"sample_perf_{vintage}.txt")


def verify_layout(dataset_dir: Path) -> dict:
    found = {}
    for v in VINTAGES:
        o_path, p_path = vintage_paths(dataset_dir, v)
        for path, expected, kind in ((o_path, ORIG_NCOLS, "origination"),
                                     (p_path, PERF_NCOLS, "performance")):
            if not path.exists():
                raise FileNotFoundError(f"missing {kind} file for {v}: {path}")
            with open(path, encoding="utf-8", errors="replace") as fh:
                n = len(fh.readline().rstrip("\n").split("|"))
            if n != expected:
                raise ValueError(
                    f"{path.name}: expected {expected} columns for the sample-file "
                    f"layout, found {n}. The layout has changed - re-verify the mapping "
                    f"in src/data/sflld.py before loading.")
            found[f"{v}_{kind}"] = n
    return found
