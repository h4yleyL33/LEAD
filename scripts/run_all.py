"""Run the whole pipeline end to end, in dependency order, and record what happened.

This is the reproducibility entry point: from a fresh clone (plus the dataset) it regenerates every
artefact the manuscript cites. Each stage is a module that can also be run on its own.

Usage
-----
    python scripts/run_all.py                # everything (~25 min on 12 cores)
    python scripts/run_all.py --fast         # skip the tuning grid and the weighted ablation
    python scripts/run_all.py --from train   # resume at a stage
    python scripts/run_all.py --check        # only run the test suite

Stages that need the dataset fail with a clear message rather than a traceback, so a missing
`data/raw/lead/lead1.0-small.csv` is diagnosed in one line.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


@dataclass
class Stage:
    key: str
    label: str
    cmd: list[str]
    needs: list[str] = field(default_factory=list)   # files that must exist beforehand
    heavy: bool = False
    timeout: int = 1800


STAGES: list[Stage] = [
    Stage("audit", "Data audit", ["-m", "src.data.audit"],
          needs=["data/raw/lead/lead1.0-small.csv"]),
    Stage("eda", "EDA figures", ["-m", "src.data.eda"],
          needs=["data/raw/lead/lead1.0-small.csv"]),
    Stage("features", "Causal features", ["-m", "src.features.build_features"],
          needs=["data/raw/lead/lead1.0-small.csv"], heavy=True),
    Stage("splits", "Building-disjoint splits", ["-m", "src.features.splits"],
          needs=["data/raw/lead/lead1.0-small.csv"]),
    Stage("leakage", "Leakage tests", ["-m", "pytest", "tests/test_features_no_leakage.py", "-q"],
          needs=["data/processed/features.parquet"]),
    Stage("baselines", "Baselines", ["-m", "src.models.baselines"],
          needs=["data/processed/features.parquet", "data/processed/splits.json"]),
    Stage("tune", "Hyper-parameter grid", ["-m", "src.models.tune", "--grid"],
          needs=["data/processed/features.parquet"], heavy=True),
    Stage("train", "Train default model", ["-m", "src.models.train_xgboost"],
          needs=["data/processed/features.parquet"], heavy=True),
    Stage("train_tuned", "Train tuned model",
          ["-m", "src.models.train_xgboost", "--params-from",
           "results/metrics/tuning_best.json", "--tag", "tuned"],
          needs=["results/metrics/tuning_best.json"], heavy=True),
    Stage("train_weighted", "Train weighted ablation",
          ["-m", "src.models.train_xgboost", "--weighted"],
          needs=["data/processed/features.parquet"], heavy=True),
    Stage("select", "Select final model", ["-m", "src.models.select_final"],
          needs=["results/metrics/xgb_tuned_val.json"]),
    Stage("holdout", "Held-out evaluation", ["-m", "src.evaluation.holdout", "--variant", "final"],
          needs=["models/xgb_final.json"], heavy=True),
    Stage("compare", "Comparison tables", ["-m", "src.evaluation.compare"],
          needs=["results/metrics/holdout_test_final.json"]),
    Stage("shap", "SHAP explanations", ["-m", "src.explainability.shap_explain"],
          needs=["models/xgb_final.json"], heavy=True),
    Stage("simulate", "Geothermal simulation", ["-m", "src.simulation.geothermal"]),
    Stage("simulate_score", "Simulation transfer test", ["-m", "src.simulation.score_simulated"],
          needs=["data/synthetic/geothermal_signals_SIMULATED.csv"]),
    Stage("paper", "Manuscript build/validation", ["scripts/build_paper.py"],
          needs=["paper/main.tex"]),
    Stage("tests", "Full test suite", ["-m", "pytest", "tests", "backend/tests", "-q"],
          needs=["models/xgb_final.json"]),
]

HEAVY = {s.key for s in STAGES if s.heavy}


def run(stage: Stage) -> dict:
    missing = [p for p in stage.needs if not (ROOT / p).exists()]
    if missing:
        return {"stage": stage.key, "status": "blocked", "missing": missing, "seconds": 0.0}

    cmd = [PY, *stage.cmd]
    t0 = time.time()
    print(f"\n=== {stage.label} ===\n$ {' '.join(cmd[1:])}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=stage.timeout)
    dt = time.time() - t0
    tail = (proc.stdout or "").strip().splitlines()[-12:]
    for line in tail:
        print(f"    {line}")
    status = "ok" if proc.returncode == 0 else "failed"
    if status == "failed":
        print((proc.stderr or "")[-2000:])
    print(f"    [{status} in {dt:.1f}s]")
    return {"stage": stage.key, "status": status, "seconds": round(dt, 1),
            "returncode": proc.returncode}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip tuning and the weighted ablation")
    ap.add_argument("--from", dest="start", default=None, help="resume at this stage key")
    ap.add_argument("--only", default=None, help="comma-separated stage keys")
    ap.add_argument("--check", action="store_true", help="run only the test suite")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failing stage")
    args = ap.parse_args()

    if args.check:
        stages = [s for s in STAGES if s.key == "tests"]
    elif args.only:
        keys = {k.strip() for k in args.only.split(",")}
        stages = [s for s in STAGES if s.key in keys]
    else:
        stages = list(STAGES)
        if args.start:
            keys = [s.key for s in STAGES]
            if args.start not in keys:
                print(f"unknown stage '{args.start}'; valid: {keys}")
                return 2
            stages = stages[keys.index(args.start):]
        if args.fast:
            stages = [s for s in stages if s.key not in {"tune", "train_weighted"}]

    print(f"running {len(stages)} stage(s): {[s.key for s in stages]}")
    t0 = time.time()
    results = []
    for s in stages:
        r = run(s)
        results.append(r)
        if r["status"] == "failed" and not args.keep_going:
            print("\nstopping on failure (use --keep-going to continue)")
            break

    total = time.time() - t0
    summary = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_seconds": round(total, 1),
        "python": sys.version,
        "fast": args.fast,
        "stages": results,
        "counts": {k: sum(1 for r in results if r["status"] == k)
                   for k in ("ok", "failed", "blocked")},
    }
    out = ROOT / "results" / "metrics" / "pipeline_run.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    for r in results:
        print(f"{r['status']:8s} {r['stage']:16s} {r['seconds']:8.1f}s")
    print(f"{'total':8s} {'':16s} {total:8.1f}s")
    print(f"wrote {out}")
    return 0 if summary["counts"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
