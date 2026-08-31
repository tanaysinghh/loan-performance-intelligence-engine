from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from src import config as C
from src.data import messiness as M
from src.data import sflld as S
from src.data.exceptions_label import build_exception_labels, reconcile_servicer_feed
from src.data.generate_synthetic import SERVICER_OPS_NOISE, build_macro_history, month_range
from src.data.build_dataset import DICTIONARY, build_macro_scenarios

DATASET_DIR = C.ROOT / "dataset"

LOANS_PER_VINTAGE = 3_200


def _credit_band(v: str):
    if v in S.SENTINEL_CREDIT_SCORE:
        return np.nan
    s = int(v)
    for hi, name in ((580, "<580"), (620, "580-619"), (660, "620-659"),
                     (700, "660-699"), (740, "700-739"), (780, "740-779")):
        if s < hi:
            return name
    return "780+"


def _ltv_band(v: str):
    if v in S.SENTINEL_LTV:
        return np.nan
    x = float(v)
    for hi, name in ((60.001, "<=60"), (70, "60-70"), (80, "70-80"),
                     (90, "80-90"), (95, "90-95")):
        if x <= hi:
            return name
    return ">95"


def _dti_band(v: str):
    if v in S.SENTINEL_DTI:
        return np.nan
    x = float(v)
    for hi, name in ((20.001, "<=20"), (30, "20-30"), (36, "30-36"), (43, "36-43")):
        if x <= hi:
            return name
    return ">43"


def _loss_band(sev):
    if not np.isfinite(sev):
        return np.nan
    for hi, name in ((10, "0-10"), (25, "10-25"), (40, "25-40"), (60, "40-60")):
        if sev < hi:
            return name
    return "60+"


def _map_status(dq: str) -> str:
    if dq == "00":
        return "Current"
    if dq == "01":
        return "DQ30"
    if dq == "02":
        return "DQ60"
    if dq == "RA":
        return "Default"
    try:
        k = int(dq)
    except ValueError:
        return "Current"
    return "Default" if k >= 6 else "DQ90plus"


_DPD_OF_STATE = {"Current": 0.0, "DQ30": 30.0, "DQ60": 60.0, "DQ90plus": 90.0,
                 "Default": 180.0}


def _sample_loan_ids(vintage: str, n: int, seed: int) -> set[str]:
    o_path, _ = S.vintage_paths(DATASET_DIR, vintage)
    ids = []
    with open(o_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            ids.append(line.split("|")[S.ORIG["loan_seq"]])
    rng = np.random.default_rng(seed + int(vintage))
    return set(rng.choice(np.array(ids), size=min(n, len(ids)), replace=False).tolist())


def _read_origination(vintage: str, keep: set[str]) -> list[dict]:
    o_path, _ = S.vintage_paths(DATASET_DIR, vintage)
    rows = []
    with open(o_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            p = line.rstrip("\n").split("|")
            lid = p[S.ORIG["loan_seq"]]
            if lid not in keep:
                continue
            fpd = p[S.ORIG["first_payment_date"]]
            orig_month = (pd.Period(f"{fpd[:4]}-{fpd[4:]}", freq="M") - 1)
            units = int(p[S.ORIG["num_units"]] or 1)
            ptype = S.PROPERTY_MAP.get(p[S.ORIG["property_type"]], "single_family")
            rows.append({
                "loan_id": lid,
                "vintage": vintage,
                "origination_month": str(orig_month),
                "original_balance": float(p[S.ORIG["original_upb"]]),
                "interest_rate": float(p[S.ORIG["original_rate"]]),
                "credit_score_band": _credit_band(p[S.ORIG["credit_score"]]),
                "ltv_band": _ltv_band(p[S.ORIG["ltv"]]),
                "dti_band": _dti_band(p[S.ORIG["dti"]]),
                "state": p[S.ORIG["property_state"]],
                "loan_purpose": S.PURPOSE_MAP.get(p[S.ORIG["loan_purpose"]], "purchase"),
                "occupancy_type": S.OCCUPANCY_MAP.get(p[S.ORIG["occupancy"]], "primary"),
                "property_type": "2-4_unit" if units >= 2 else ptype,
                "original_term": int(p[S.ORIG["original_term"]] or 360),
            })
    return rows


def _read_performance(vintage: str, keep: set[str]) -> list[dict]:
    _, p_path = S.vintage_paths(DATASET_DIR, vintage)
    rows = []
    with open(p_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            p = line.rstrip("\n").split("|")
            lid = p[S.PERF["loan_seq"]]
            if lid not in keep:
                continue
            rm = p[S.PERF["reporting_month"]]
            upb = p[S.PERF["current_upb"]]
            loss = p[S.PERF["actual_loss"]]
            rows.append({
                "loan_id": lid,
                "reporting_month": f"{rm[:4]}-{rm[4:]}",
                "current_balance": float(upb) if upb else np.nan,
                "delinquency_status": p[S.PERF["delinquency_status"]],
                "loan_age_months": int(p[S.PERF["loan_age"]] or 0),
                "remaining_term_months": int(p[S.PERF["remaining_months"]] or 0),
                "raw_modification_flag": p[S.PERF["modification_flag"]],
                "zero_balance_code": p[S.PERF["zero_balance_code"]],
                "servicer_monthly": p[S.PERF["servicer_name"]] or "OTHER",
                "actual_loss": float(loss) if loss else np.nan,
                "borrower_assistance": p[S.PERF["borrower_assistance_code"]],
            })
    return rows


def _forward_any(panel: pd.DataFrame, flag: pd.Series, horizon: int,
                 closed: np.ndarray, last_month_index: int) -> pd.Series:
    frame = pd.DataFrame({"loan_id": panel["loan_id"].to_numpy(),
                          "f": flag.to_numpy(),
                          "mi": panel["month_index"].to_numpy()})
    gg = frame.groupby("loan_id", sort=False)
    hit = np.zeros(len(frame))
    avail = np.zeros(len(frame))
    for k in range(horizon):
        hit = np.maximum(hit, gg["f"].shift(-k).fillna(0.0).to_numpy())
        avail = avail + gg["f"].shift(-k).notna().to_numpy().astype(float)
    gap = last_month_index - frame["mi"].to_numpy()
    window_complete = (avail >= horizon) | (closed == 1)
    out = np.where(hit > 0, 1.0, np.where(window_complete, 0.0, np.nan))
    out = np.where((hit == 0) & (gap < horizon) & (closed == 0), np.nan, out)
    return pd.Series(out, index=panel.index)


def build_panel(loans_per_vintage: int = LOANS_PER_VINTAGE,
                seed: int = C.RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    S.verify_layout(DATASET_DIR)

    orig_rows, perf_rows, per_vintage = [], [], {}
    for v in S.VINTAGES:
        keep = _sample_loan_ids(v, loans_per_vintage, seed)
        o = _read_origination(v, keep)
        p = _read_performance(v, keep)
        orig_rows.extend(o)
        perf_rows.extend(p)
        per_vintage[v] = {"loans": len(o), "rows": len(p)}
        print(f"  [{v}] {len(o):,} loans, {len(p):,} monthly rows", flush=True)

    loans = pd.DataFrame(orig_rows)
    panel = pd.DataFrame(perf_rows)

    months = pd.period_range(S.PANEL_FIRST_MONTH, S.PANEL_LAST_MONTH, freq="M")
    midx = {str(m): i for i, m in enumerate(months)}
    panel["month_index"] = panel["reporting_month"].map(midx)
    panel = panel.dropna(subset=["month_index"])
    panel["month_index"] = panel["month_index"].astype(int)

    panel = panel.sort_values(["loan_id", "month_index"], kind="mergesort").reset_index(drop=True)

    uniq = {d: _map_status(d) for d in panel["delinquency_status"].unique()}
    panel["current_status"] = panel["delinquency_status"].map(uniq)
    panel["days_past_due"] = panel["current_status"].map(_DPD_OF_STATE)

    g = panel.groupby("loan_id", sort=False)
    panel["modification_flag"] = (
        g["raw_modification_flag"].transform(lambda s: s.isin(["Y", "P"]).cummax()).astype(int))

    sev = np.where(panel["actual_loss"].notna(),
                   panel["actual_loss"].abs() / panel["loan_id"].map(
                       loans.set_index("loan_id")["original_balance"]).to_numpy() * 100.0,
                   np.nan)
    panel["loss_severity_band"] = [_loss_band(x) for x in sev]

    zbc = panel["zero_balance_code"].fillna("")
    terminal = np.where(zbc.isin(list(S.ZBC_PREPAID)), "Prepaid",
                np.where(zbc.isin(list(S.ZBC_CREDIT_EVENT)), "Default",
                np.where(zbc.isin(list(S.ZBC_OTHER_EXIT)), "PaidOff", "")))
    panel["terminal_state"] = terminal

    is_last = panel["month_index"].to_numpy() == g["month_index"].transform("max").to_numpy()
    nxt = g["current_status"].shift(-1)
    panel["status_next"] = np.where(
        is_last, np.where(terminal != "", terminal, None), nxt)
    panel["status_next"] = panel["status_next"].replace({None: np.nan, "": np.nan})

    diagnostics = {
        "per_vintage": per_vintage,
        "panel_months": [S.PANEL_FIRST_MONTH, S.PANEL_LAST_MONTH],
        "last_month_index": int(len(months) - 1),
        "true_credit_event_loans": int(
            panel.loc[panel["terminal_state"] == "Default", "loan_id"].nunique()),
        "prepaid_loans": int(panel.loc[panel["terminal_state"] == "Prepaid", "loan_id"].nunique()),
        "ever_d90plus_loans": int(
            panel.loc[panel["current_status"].isin(["DQ90plus", "Default"]), "loan_id"].nunique()),
        "servicer_transfer_loans": int((g["servicer_monthly"].nunique() > 1).sum()),
    }
    return panel, loans, diagnostics


def attach_targets(panel: pd.DataFrame, last_month_index: int) -> pd.DataFrame:
    panel = panel.sort_values(["loan_id", "month_index"], kind="mergesort").reset_index(drop=True)
    g = panel.groupby("loan_id", sort=False)

    panel["next_state"] = panel["status_next"]
    dq = panel["status_next"].isin(list(C.DELINQUENT_STATES)).astype(float)
    serious = panel["status_next"].isin(["DQ90plus", "Default"]).astype(float)
    prepay = (panel["status_next"] == "Prepaid").astype(float)
    default_true = (panel["status_next"] == "Default").astype(float)

    closed = g["status_next"].transform(
        lambda s: float(s.isin(list(C.TERMINAL_STATES)).any())).to_numpy()

    panel["next_3m_delinquency_flag"] = _forward_any(panel, dq, 3, closed, last_month_index)
    panel["next_6m_delinquency_flag"] = _forward_any(panel, dq, 6, closed, last_month_index)
    panel["next_12m_default_flag"] = _forward_any(panel, serious, 12, closed, last_month_index)
    panel["next_12m_prepayment_flag"] = _forward_any(panel, prepay, 12, closed, last_month_index)

    panel["prepayment_flag"] = prepay.astype(int)
    panel["default_flag"] = default_true.astype(int)
    return panel


def main(loans_per_vintage: int = LOANS_PER_VINTAGE, seed: int = C.RANDOM_SEED) -> dict:
    rng = np.random.default_rng(seed)

    print("[sflld] reading vintage files...", flush=True)
    panel, loans, diag = build_panel(loans_per_vintage, seed)
    panel = attach_targets(panel, diag["last_month_index"])

    real_servicers = sorted(set(panel["servicer_monthly"].unique()))
    for name in real_servicers:
        if name not in SERVICER_OPS_NOISE:
            h = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
            SERVICER_OPS_NOISE[name] = 0.04 + (h % 1000) / 1000.0 * 0.24

    first_servicer = panel.groupby("loan_id", sort=False)["servicer_monthly"].first()
    loans["servicer_name"] = loans["loan_id"].map(first_servicer).fillna("OTHER")

    panel_cols = ["loan_id", "month_index", "reporting_month", "loan_age_months",
                  "remaining_term_months", "current_balance", "current_status",
                  "days_past_due", "modification_flag", "loss_severity_band",
                  "servicer_monthly", "status_next", "next_state",
                  "prepayment_flag", "default_flag"] + C.BINARY_TARGETS
    panel = panel[[c for c in panel_cols if c in panel.columns]]

    print("[sflld] injecting documented data-quality defects...", flush=True)
    messy, defect_log = M.inject(panel, loans, rng)
    messy["servicer_name"] = messy["servicer_monthly"]
    messy = messy.drop(columns=["servicer_monthly"])

    print("[sflld] building second-source servicer feed...", flush=True)
    updates = M.build_servicer_updates(messy, rng)
    reconciled = reconcile_servicer_feed(messy, updates)
    labelled = build_exception_labels(reconciled, updates, rng)

    keep = (C.RAW_COLUMNS + C.BINARY_TARGETS
            + ["next_state", "exception_required", "exception_type"])
    out = labelled[[c for c in keep if c in labelled.columns]].copy()
    out = out.sample(frac=1.0, random_state=17).reset_index(drop=True)

    out.to_csv(C.LOAN_PANEL, index=False)
    updates.to_csv(C.SERVICER_UPDATES, index=False)
    defect_log.to_csv(C.DATA_RAW / "ground_truth_defect_log.csv", index=False)
    from src.data.validate import export_rules_json
    export_rules_json()

    dictionary = pd.DataFrame(DICTIONARY, columns=["field", "dtype", "description",
                                                   "allowed_values", "source_system"])
    real_values = {
        "loan_id": ("Freddie Mac Loan Sequence Number, format PYYQnXXXXXXX "
                    "(e.g. F19Q10000056).", "PYYQnXXXXXXX"),
        "servicer_name": ("Servicer reported for the loan in this month. Changes on a real "
                          "servicing transfer.",
                          f"{out['servicer_name'].nunique()} distinct real servicer names"),
        "state": ("US state or territory of the collateral property.",
                  f"{out['state'].nunique()} USPS codes incl. DC/PR/GU/VI"),
        "next_12m_default_flag": (
            "1 if the loan reaches 90+ days past due, or a realised credit event, in months "
            "t+1..t+12. NaN when right-censored. NOTE: this is a 90+ DPD PROXY, not a "
            "realised-default rate - realised credit events occur on ~0.1% of loans.",
            "0|1|NaN"),
    }
    for field, (desc, allowed) in real_values.items():
        hit = dictionary["field"] == field
        dictionary.loc[hit, "description"] = desc
        dictionary.loc[hit, "allowed_values"] = allowed
    dictionary.to_csv(C.DATA_DICTIONARY, index=False)

    out.head(500).to_csv(C.DATA_SAMPLES / "loan_panel_sample.csv", index=False)
    updates.head(300).to_csv(C.DATA_SAMPLES / "servicer_updates_sample.csv", index=False)

    summary = {
        "source": "freddie_mac_sflld_real",
        "loans": int(out["loan_id"].nunique()),
        "rows": int(len(out)),
        "months": diag["last_month_index"] + 1,
        "panel_window": diag["panel_months"],
        "servicer_update_rows": int(len(updates)),
        "status_mix": out["current_status"].value_counts(normalize=True).round(4).to_dict(),
        "target_rates": {t: float(out[t].mean(skipna=True)) for t in C.BINARY_TARGETS},
        "target_positives": {t: int(out[t].sum(skipna=True)) for t in C.BINARY_TARGETS},
        "censored_rows": {t: int(out[t].isna().sum()) for t in C.BINARY_TARGETS},
        "exception_rate": float(out["exception_required"].mean()),
        "exception_mix": out["exception_type"].value_counts(normalize=True).round(4).to_dict(),
        "real_data_diagnostics": diag,
    }

    import json
    (C.ARTIFACTS / "sflld_build_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2, default=str))
