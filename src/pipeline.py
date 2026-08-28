"""End-to-end orchestration.

    python -m src.pipeline                 full run
    python -m src.pipeline --skip-data     reuse the existing data pack
    python -m src.pipeline --stage models  run one stage and its prerequisites

Stages run in dependency order and each one writes its own report to `reports/`. The run
finishes by writing `submission/submission.csv` and a manifest recording what was produced,
by which module, and how long it took.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import pandas as pd

from src import config as C

STAGES = ["data", "profile", "models", "survival", "anomaly", "scenarios", "explain",
          "copilot", "submission", "model_card"]


def _step(name: str, fn, state: dict, log: list):
    started = time.perf_counter()
    print(f"[{name}] running...", flush=True)
    try:
        result = fn(state)
        status, error = "ok", None
    except Exception as exc:
        result, status, error = None, "failed", f"{type(exc).__name__}: {exc}"
        print(f"[{name}] FAILED: {error}", flush=True)
    elapsed = time.perf_counter() - started
    log.append({"stage": name, "status": status, "seconds": round(elapsed, 2), "error": error})
    print(f"[{name}] {status} in {elapsed:.1f}s", flush=True)
    if status == "failed":
        raise RuntimeError(f"Stage '{name}' failed: {error}")
    return result


def run(skip_data: bool = False, only_stage: str | None = None,
        n_loans: int | None = None) -> dict:
    state, log = {}, []
    started = datetime.now(timezone.utc)
    stages = STAGES if only_stage is None else STAGES[: STAGES.index(only_stage) + 1]

    def stage_data(s):
        from src.data.build_dataset import main as build_data
        if skip_data and C.LOAN_PANEL.exists():
            print("  reusing existing data pack")
            return None
        kwargs = {"n_loans": n_loans} if n_loans else {}
        return build_data(**kwargs)

    def stage_prepare(s):
        from src.features.dataset import prepare
        for cache in (C.DATA_PROCESSED / "model_frame.parquet",
                      C.DATA_PROCESSED / "model_frame.csv"):
            if cache.exists() and not skip_data:
                cache.unlink()
        s["df"] = prepare(use_cache=skip_data)
        return {"rows": len(s["df"]), "columns": s["df"].shape[1]}

    def stage_profile(s):
        from src.data.report_data_intelligence import build as build_profile
        out = build_profile()
        return {k: v for k, v in out.items() if k != "scored"}

    def stage_models(s):
        from src.models.run_performance import run as run_models
        out = run_models(s["df"])
        s["models"] = out["models"]
        return {"targets": list(out["models"].keys())}

    def stage_survival(s):
        from src.models.run_survival import run as run_surv
        out = run_surv(s["df"])
        return {"defaults": int(out["survival_frame"]["event_default"].sum()),
                "cox_c_index_test": round(out["cox_default"]["concordance_test"], 4)}

    def stage_anomaly(s):
        from src.models import performance as P
        from src.models.run_anomaly import run as run_anom
        out = run_anom(s["df"])
        s["models"]["exception_required"] = out["models"]["binary_model"]
        P.save(s["models"])
        return out["agreement"]

    def stage_scenarios(s):
        from src.scenarios.run_scenarios import run as run_scen
        out = run_scen(s["df"], s["models"])
        return {"scenarios": out["headline"]["scenario_name"].tolist()}

    def stage_explain(s):
        from src.explain.run_explain import run as run_expl
        out = run_expl(s["df"], s["models"])
        return {t: v["global"].head(3)["plain_english"].tolist() for t, v in out.items()}

    def stage_copilot(s):
        from src.copilot.run_copilot import run as run_cop
        out = run_cop(s["df"], s["models"])
        return {"mode": out["mode"], **out["stats"]}

    def stage_submission(s):
        from src.submission import build, write
        out = write(build(s["df"], s["models"]))
        return {"rows": len(out),
                "actions": out["recommended_action"].value_counts().to_dict()}

    def stage_model_card(s):
        from src.report_model_card import write as write_card
        text = write_card()
        return {"lines": len(text.splitlines())}

    steps = {
        "data": stage_data, "model_card": stage_model_card, "profile": stage_profile, "models": stage_models,
        "survival": stage_survival, "anomaly": stage_anomaly, "scenarios": stage_scenarios,
        "explain": stage_explain, "copilot": stage_copilot, "submission": stage_submission,
    }

    results = {}
    if "data" in stages:
        results["data"] = _step("data", stage_data, state, log)
    results["prepare"] = _step("prepare", stage_prepare, state, log)
    for name in stages:
        if name == "data":
            continue
        results[name] = _step(name, steps[name], state, log)

    manifest = {
        "run_started_utc": started.isoformat(),
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "total_seconds": round(sum(s["seconds"] for s in log), 1),
        "stages": log,
        "results": results,
        "outputs": sorted(p.name for p in C.REPORTS.glob("*")) +
                   sorted(f"submission/{p.name}" for p in C.SUBMISSION.glob("*")),
    }
    (C.SUBMISSION / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"\nPipeline complete in {manifest['total_seconds']}s. "
          f"Manifest: submission/run_manifest.json")
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Loan Performance Intelligence Engine pipeline")
    ap.add_argument("--skip-data", action="store_true",
                    help="reuse the existing data pack and cached model frame")
    ap.add_argument("--stage", choices=STAGES, default=None,
                    help="run up to and including this stage")
    ap.add_argument("--n-loans", type=int, default=None,
                    help="override the number of synthetic loans generated")
    args = ap.parse_args()
    run(skip_data=args.skip_data, only_stage=args.stage, n_loans=args.n_loans)


if __name__ == "__main__":
    main()
