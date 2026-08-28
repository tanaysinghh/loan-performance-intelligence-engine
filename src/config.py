"""Central paths, schema constants and shared vocabulary for the engine."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SAMPLES = ROOT / "data" / "samples"
REPORTS = ROOT / "reports"
SUBMISSION = ROOT / "submission"
ARTIFACTS = ROOT / "artifacts"

for _p in (DATA_RAW, DATA_PROCESSED, DATA_SAMPLES, REPORTS, SUBMISSION, ARTIFACTS):
    _p.mkdir(parents=True, exist_ok=True)

LOAN_PANEL = DATA_RAW / "loan_panel.csv"
SERVICER_UPDATES = DATA_RAW / "servicer_updates.csv"
MACRO_HISTORY = DATA_RAW / "macro_history.csv"
MACRO_SCENARIOS = DATA_RAW / "macro_scenarios.csv"
DATA_DICTIONARY = DATA_RAW / "data_dictionary.csv"

RANDOM_SEED = 20260828

PANEL_START = "2022-01"
PANEL_END = "2026-06"

STATES = ["Current", "DQ30", "DQ60", "DQ90plus", "Default", "Prepaid", "PaidOff"]
DELINQUENT_STATES = {"DQ30", "DQ60", "DQ90plus", "Default"}
TERMINAL_STATES = {"Default", "Prepaid", "PaidOff"}

CREDIT_BANDS = ["<580", "580-619", "620-659", "660-699", "700-739", "740-779", "780+"]
LTV_BANDS = ["<=60", "60-70", "70-80", "80-90", "90-95", ">95"]
DTI_BANDS = ["<=20", "20-30", "30-36", "36-43", ">43"]
LOSS_SEVERITY_BANDS = ["0-10", "10-25", "25-40", "40-60", "60+"]
DOC_STATUSES = ["complete", "pending", "missing", "exception"]
SOURCE_SYSTEMS = ["core_servicing", "investor_feed", "manual_upload"]
OCCUPANCY_TYPES = ["primary", "second_home", "investment"]
PROPERTY_TYPES = ["single_family", "condo", "pud", "2-4_unit", "manufactured"]
LOAN_PURPOSES = ["purchase", "rate_term_refi", "cash_out_refi"]
SERVICERS = ["Northgate Servicing", "Belmont Loan Services", "Arcadia Capital Servicing",
             "Pioneer Mortgage Ops", "Kestrel Financial"]
STATES_US = ["CA", "TX", "FL", "NY", "AZ", "NV", "GA", "OH", "IL", "NC", "WA", "MI", "PA", "NJ", "CO"]

EXCEPTION_TYPES = [
    "none",
    "missing_documentation",
    "balance_reconciliation_break",
    "stale_servicer_reporting",
    "invalid_date_relationship",
    "status_dpd_mismatch",
    "unexpected_balance_movement",
]

BINARY_TARGETS = [
    "next_3m_delinquency_flag",
    "next_6m_delinquency_flag",
    "next_12m_default_flag",
    "next_12m_prepayment_flag",
]
MULTICLASS_TARGETS = ["next_state", "exception_type"]
ALL_TARGETS = BINARY_TARGETS + MULTICLASS_TARGETS + ["exception_required"]

TRAIN_END = "2025-03"
VALID_END = "2025-09"

RAW_COLUMNS = [
    "loan_id", "month_index", "reporting_month", "origination_month", "loan_age_months",
    "remaining_term_months", "original_balance", "current_balance", "interest_rate",
    "credit_score_band", "ltv_band", "dti_band", "state", "loan_purpose", "occupancy_type",
    "property_type", "servicer_name", "current_status", "days_past_due", "modification_flag",
    "prepayment_flag", "default_flag", "loss_severity_band", "last_updated_at",
    "source_system", "document_status",
]
