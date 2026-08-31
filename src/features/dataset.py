from __future__ import annotations

import pandas as pd

from src import config as C
from src.data import loaders, validate
from src.features import build_features as F

CACHE = C.DATA_PROCESSED / "model_frame.parquet"
CACHE_CSV = C.DATA_PROCESSED / "model_frame.csv"


def _cache_is_stale() -> bool:
    caches = [c for c in (CACHE, CACHE_CSV) if c.exists()]
    if not caches:
        return False
    newest_cache = max(c.stat().st_mtime for c in caches)
    sources = [C.LOAN_PANEL, C.SERVICER_UPDATES, C.MACRO_HISTORY]
    newest_source = max((s.stat().st_mtime for s in sources if s.exists()), default=0.0)
    return newest_source > newest_cache


def prepare(use_cache: bool = True) -> pd.DataFrame:
    if use_cache and _cache_is_stale():
        print("  [prepare] raw data pack is newer than the cached feature frame; rebuilding",
              flush=True)
        use_cache = False
    if use_cache:
        cached = _read_cache()
        if cached is not None:
            return cached

    panel = loaders.load_panel()
    updates = loaders.load_servicer_updates()
    macro = loaders.load_macro()

    joined = loaders.reconcile(panel, updates)
    feed_stats = {"feed_duplicate_records": joined.attrs.get("feed_duplicate_records", 0),
                  "feed_orphan_records": joined.attrs.get("feed_orphan_records", 0)}
    cleaned = loaders.clean_panel(joined)
    flagged, _ = validate.run_rules(cleaned)
    scored = validate.score_records(flagged)
    frame = F.build(scored, macro)
    frame.attrs.update(feed_stats)

    _write_cache(frame)
    return frame


def _read_cache():
    try:
        if CACHE.exists():
            return pd.read_parquet(CACHE)
    except (ImportError, ValueError, OSError):
        pass
    if CACHE_CSV.exists():
        from src.data.loaders import _PANEL_STRING_COLS
        df = pd.read_csv(CACHE_CSV, dtype=_PANEL_STRING_COLS)
        return _restore_types(df)
    return None


def _write_cache(frame: pd.DataFrame) -> None:
    try:
        frame.to_parquet(CACHE, index=False)
        return
    except (ImportError, ValueError, OSError):
        pass
    frame.to_csv(CACHE_CSV, index=False)


def _restore_types(df: pd.DataFrame) -> pd.DataFrame:
    df["reporting_period"] = pd.PeriodIndex(df["reporting_month"], freq="M")
    df["origination_period"] = pd.to_datetime(df["origination_month"], format="%Y-%m",
                                              errors="coerce").dt.to_period("M")
    for c in ("last_updated_at", "period_end", "svc_received_at"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    if "dq_band" in df.columns:
        df["dq_band"] = pd.Categorical(df["dq_band"],
                                       categories=["critical", "poor", "watch", "clean"],
                                       ordered=True)
    return df


def modelling_features(df: pd.DataFrame) -> list[str]:
    return F.feature_columns(df)
