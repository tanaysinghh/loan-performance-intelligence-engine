"""Reader for the Freddie Mac Single-Family Loan-Level Dataset (SFLLD) sample files.

The five vintage folders under ``dataset/`` hold pipe-delimited, header-less files: one
origination file and one monthly performance file per vintage.

Layout note (important): these sample files do NOT match the stock 32/32-field layout that
Freddie Mac publishes in ``file_layout.xlsx`` and the January 2026 General User Guide. The
files on disk carry **31 origination** and **35 performance** columns. The difference is
systematic, and was verified empirically against value distributions in all five vintages
rather than assumed:

* ``Servicer Name`` (official origination position 25) has been **removed** from the
  origination file, so origination positions 25..31 map to official 26..32. Confirmed by
  unambiguous value signatures - our column 26 holds loan-sequence-formatted strings
  (Pre-HARP Loan Sequence Number), 27 is ``H``/``F`` (Special Eligibility Program), 28 is
  ``Y`` only in 2019 and never in 2023 (Relief Refinance - HARP ended), and 29 shows code
  ``4`` only from 2022 onward (ACE+PDR, effective July 2022 per the guide).
* The performance file keeps official positions 1..32 and appends three columns:
  33 = ``MI Cancellation Indicator`` (values ``7``/``N``/``Y``, matching the guide's valid
  values exactly), 34 = ``Servicer Name``, 35 = an empty filler column.

Both moved fields are time-varying, which is why they now live in the monthly file.

Two columns carry no information and are excluded from any feature use: origination column
31 is the constant ``9999`` across all 250,000 rows in all five vintages, and performance
column 35 is blank or ``0.00`` only.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Column positions (0-based) as they appear in the sample files on disk.
# ---------------------------------------------------------------------------

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
    # official 26..32 shifted down one - Servicer Name is absent from this file
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
    # appended in the sample files, absent from the published layout
    "mi_cancellation_indicator": 32,
    "servicer_name": 33,
    "_filler": 34,
}
PERF_NCOLS = 35

VINTAGES = ["2019", "2020", "2021", "2022", "2023"]

# The observed panel window across all five vintages.
PANEL_FIRST_MONTH = "2019-01"
PANEL_LAST_MONTH = "2026-03"

# ---------------------------------------------------------------------------
# Value vocabularies -> the project's canonical vocabulary in src/config.py
# ---------------------------------------------------------------------------

OCCUPANCY_MAP = {"P": "primary", "S": "second_home", "I": "investment"}
PURPOSE_MAP = {"P": "purchase", "N": "rate_term_refi", "C": "cash_out_refi"}
# CP is a co-operative share loan; it is grouped with condo. Loans on 2-4 unit
# properties are identified from Number of Units, not from Property Type.
PROPERTY_MAP = {"SF": "single_family", "PU": "pud", "CO": "condo",
                "MH": "manufactured", "CP": "condo"}

# Zero Balance Code -> terminal state.
#   01 prepaid or matured                      -> voluntary exit
#   02 third party sale, 03 short sale/charge-off,
#   09 REO disposition, 15 note sale           -> realised credit event
#   96 removal (repurchase), 16 reperforming loan sale -> neither
ZBC_PREPAID = {"01"}
ZBC_CREDIT_EVENT = {"02", "03", "09", "15"}
ZBC_OTHER_EXIT = {"96", "16"}

SENTINEL_CREDIT_SCORE = {"9999", ""}
SENTINEL_DTI = {"999", ""}
SENTINEL_LTV = {"999", ""}


def vintage_paths(dataset_dir: Path, vintage: str) -> tuple[Path, Path]:
    """Returns (origination file, performance file) for one vintage."""
    folder = dataset_dir / f"sample_{vintage}"
    return (folder / f"sample_orig_{vintage}.txt",
            folder / f"sample_perf_{vintage}.txt")


def verify_layout(dataset_dir: Path) -> dict:
    """Checks the on-disk files actually carry the 31/35 layout this module assumes.

    Raises if any vintage deviates, so a re-download with a different layout fails loudly
    at load time instead of silently producing mis-mapped columns.
    """
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
