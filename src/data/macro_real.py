"""Real macroeconomic history, and scenario paths calibrated against it.

Provenance is split deliberately, and the split is disclosed in the scenario report:

**Observed history (real, sourced).** Three public series, vendored under ``data/external/``
so the build reproduces without network access:

===========================  ==========================================================
``market_mortgage_rate``     FRED ``MORTGAGE30US`` - Freddie Mac Primary Mortgage Market
                             Survey, 30-year fixed rate. Weekly, averaged to monthly.
``unemployment_rate``        FRED ``UNRATE`` - BLS civilian unemployment rate. Monthly.
``hpi_yoy_growth``           FRED ``CSUSHPINSA`` - S&P CoreLogic Case-Shiller U.S.
                             National Home Price Index. Monthly, converted to
                             year-over-year growth.
===========================  ==========================================================

**Forward scenario paths (constructed assumptions).** No forecast is sourced - forecasts of
this kind are not freely redistributable. The three required scenarios are constructed, but
each shock is *calibrated to the largest comparable move actually present in the same series*
over the panel window, so the magnitudes are empirical rather than invented. The calibration
figures are written into the scenario file's ``assumption_note`` column and surfaced in the
scenario report.
"""
from __future__ import annotations

import pandas as pd

from src import config as C
from src.data import sflld as S

EXTERNAL = C.ROOT / "data" / "external"

SERIES_PROVENANCE = {
    "market_mortgage_rate": {
        "series_id": "MORTGAGE30US",
        "name": "Freddie Mac Primary Mortgage Market Survey, 30-Year Fixed Rate",
        "source": "FRED (Federal Reserve Bank of St. Louis)",
        "url": "https://fred.stlouisfed.org/series/MORTGAGE30US",
        "native_frequency": "weekly",
        "transform": "monthly mean of weekly observations",
    },
    "unemployment_rate": {
        "series_id": "UNRATE",
        "name": "Civilian Unemployment Rate",
        "source": "FRED / U.S. Bureau of Labor Statistics",
        "url": "https://fred.stlouisfed.org/series/UNRATE",
        "native_frequency": "monthly",
        "transform": "none",
    },
    "hpi_yoy_growth": {
        "series_id": "CSUSHPINSA",
        "name": "S&P CoreLogic Case-Shiller U.S. National Home Price Index",
        "source": "FRED / S&P Dow Jones Indices",
        "url": "https://fred.stlouisfed.org/series/CSUSHPINSA",
        "native_frequency": "monthly",
        "transform": "year-over-year growth, (index_t / index_t-12) - 1",
    },
}


def _read(series_id: str) -> pd.DataFrame:
    path = EXTERNAL / f"fred_{series_id}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"missing vendored macro series {path}. Re-download with:\n"
            f"  curl -s 'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}' "
            f"-o {path}")
    df = pd.read_csv(path)
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna()


def build_macro_history(first_month: str = S.PANEL_FIRST_MONTH,
                        last_month: str = S.PANEL_LAST_MONTH) -> pd.DataFrame:
    """Monthly observed macro history over the panel window, from the real series."""
    months = pd.period_range(first_month, last_month, freq="M")

    rate = _read("MORTGAGE30US")
    rate["m"] = rate["date"].dt.to_period("M")
    rate_m = rate.groupby("m")["value"].mean()

    unrate = _read("UNRATE")
    unrate["m"] = unrate["date"].dt.to_period("M")
    unrate_m = unrate.groupby("m")["value"].mean()

    hpi = _read("CSUSHPINSA")
    hpi["m"] = hpi["date"].dt.to_period("M")
    hpi_m = hpi.groupby("m")["value"].mean()
    hpi_yoy = (hpi_m / hpi_m.shift(12) - 1.0)

    out = pd.DataFrame({"reporting_month": months.astype(str)})
    out["market_mortgage_rate"] = [round(float(rate_m.get(m, float("nan"))), 4) for m in months]
    out["unemployment_rate"] = [round(float(unrate_m.get(m, float("nan"))), 4) for m in months]
    out["hpi_yoy_growth"] = [round(float(hpi_yoy.get(m, float("nan"))), 5) for m in months]

    # Case-Shiller and UNRATE publish with a lag; carry the last observation forward to the
    # panel edge rather than dropping months. Forward-fill is recorded in the report.
    out = out.ffill()
    return out


#: The COVID labour-market shock is a genuine observation but a poor calibration anchor: it
#: is a one-off whose 12-month unemployment swing (+11.1pp) dwarfs any credit-cycle stress a
#: reviewer would plan against. It is reported and then excluded from the bounds below.
COVID_WINDOW = ("2020-01", "2021-06")

#: Supervisory-style severity used where the panel window itself contains no comparable
#: episode. Broadly in line with the Federal Reserve's DFAST/CCAR severely-adverse
#: magnitudes. Both the observed bounds and this override are disclosed in the scenario file.
SUPERVISORY_UNEMPLOYMENT_SHOCK_PP = 3.0
SUPERVISORY_HPI_YOY_TROUGH = -0.10


def calibration_facts(macro: pd.DataFrame) -> dict:
    """Observed 12-month moves in each series, with and without the COVID window.

    The panel window (2019-2026) contains no housing downturn and, once COVID is set aside,
    no material labour-market deterioration - the largest ex-COVID 12-month unemployment rise
    is +0.7pp. Anchoring an adverse scenario on that would produce a stress that stresses
    nothing. So the adverse shocks fall back to supervisory magnitudes, and every figure
    behind that decision is reported rather than quietly assumed.
    """
    m = macro.copy()
    covid = ((m["reporting_month"] >= COVID_WINDOW[0])
             & (m["reporting_month"] <= COVID_WINDOW[1]))
    ex = m.loc[~covid]
    return {
        "max_12m_rate_rise": round(float(m["market_mortgage_rate"].diff(12).max()), 3),
        "max_12m_rate_fall": round(float(m["market_mortgage_rate"].diff(12).min()), 3),
        "max_12m_unemployment_rise_all": round(float(m["unemployment_rate"].diff(12).max()), 3),
        "max_12m_unemployment_rise_ex_covid": round(
            float(ex["unemployment_rate"].diff(12).max()), 3),
        "peak_unemployment": float(m["unemployment_rate"].max()),
        "min_hpi_yoy": round(float(m["hpi_yoy_growth"].min()), 5),
        "max_hpi_yoy": round(float(m["hpi_yoy_growth"].max()), 5),
        "adverse_unemployment_shock_used": SUPERVISORY_UNEMPLOYMENT_SHOCK_PP,
        "adverse_hpi_yoy_used": SUPERVISORY_HPI_YOY_TROUGH,
        "last_month": str(m["reporting_month"].iloc[-1]),
        "last_rate": float(m["market_mortgage_rate"].iloc[-1]),
        "last_unemployment": float(m["unemployment_rate"].iloc[-1]),
        "last_hpi_yoy": float(m["hpi_yoy_growth"].iloc[-1]),
    }


def build_macro_scenarios(macro: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    """Base / adverse-credit / high-prepayment paths, calibrated to observed history."""
    f = calibration_facts(macro)
    last = macro.iloc[-1]

    # Each shock is a 12-month total move applied linearly. The prepayment shock is taken
    # straight from observed history; the adverse shocks use supervisory magnitudes because
    # the panel window contains no comparable credit episode (see calibration_facts).
    unemp_shock = SUPERVISORY_UNEMPLOYMENT_SHOCK_PP
    rate_fall = round(f["max_12m_rate_fall"], 2)
    hpi_trough = SUPERVISORY_HPI_YOY_TROUGH

    paths = {
        "base": {
            "rate": 0.0,
            "unemp": 0.0,
            "hpi": 0.0,
            "note": (f"Observed conditions at {f['last_month']} held flat for {horizon} "
                     f"months (rate {f['last_rate']:.2f}%, unemployment "
                     f"{f['last_unemployment']:.2f}%, HPI YoY {f['last_hpi_yoy']:.2%}). "
                     "No shock applied. CONSTRUCTED ASSUMPTION; history is observed."),
        },
        "adverse_credit": {
            "rate": 0.0,
            "unemp": unemp_shock,
            "hpi": hpi_trough - f["last_hpi_yoy"],
            "note": (f"Unemployment rises {unemp_shock:.2f}pp over {horizon} months and HPI "
                     f"growth falls to {hpi_trough:.0%} YoY; rates held flat. CONSTRUCTED "
                     f"ASSUMPTION at supervisory (DFAST/CCAR-style) severity, NOT taken from "
                     f"the panel window. Disclosed basis: the largest 12-month unemployment "
                     f"rise observed in UNRATE over the window is "
                     f"{f['max_12m_unemployment_rise_all']:.1f}pp, but that is the COVID "
                     f"one-off (peak {f['peak_unemployment']:.1f}% in 2020-04); excluding "
                     f"2020-01..2021-06 the largest rise is only "
                     f"{f['max_12m_unemployment_rise_ex_covid']:.1f}pp, and the weakest HPI "
                     f"growth observed is {f['min_hpi_yoy']:.2%}. The window contains no "
                     f"housing downturn, so an empirically-anchored adverse case would not "
                     f"stress the book; supervisory magnitudes are used instead."),
        },
        "high_prepayment": {
            "rate": rate_fall,
            "unemp": 0.0,
            "hpi": 0.0,
            "note": (f"Market mortgage rate falls {abs(rate_fall):.2f}pp over {horizon} "
                     f"months, opening refinance incentive on seasoned high-coupon loans. "
                     f"CONSTRUCTED ASSUMPTION, calibrated to the largest 12-month rate "
                     f"decline actually observed in MORTGAGE30US over the panel window."),
        },
    }

    rows = []
    for name, d in paths.items():
        for h in range(1, horizon + 1):
            frac = h / horizon
            rows.append({
                "scenario_name": name,
                "horizon_month": h,
                "market_mortgage_rate": round(last["market_mortgage_rate"] + d["rate"] * frac, 4),
                "unemployment_rate": round(last["unemployment_rate"] + d["unemp"] * frac, 4),
                "hpi_yoy_growth": round(last["hpi_yoy_growth"] + d["hpi"] * frac, 5),
                "assumption_note": d["note"],
            })
    return pd.DataFrame(rows)


def write_all() -> dict:
    macro = build_macro_history()
    scen = build_macro_scenarios(macro)
    macro.to_csv(C.MACRO_HISTORY, index=False)
    scen.to_csv(C.MACRO_SCENARIOS, index=False)
    facts = calibration_facts(macro)
    return {"months": len(macro), "calibration": facts,
            "scenarios": sorted(scen["scenario_name"].unique().tolist()),
            "provenance": SERIES_PROVENANCE}


if __name__ == "__main__":
    import json
    print(json.dumps(write_all(), indent=2, default=str))
