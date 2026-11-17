"""Prepare every artefact a deliverable needs, from a clean checkout, and verify it.

A clean checkout of this repository contains code, documentation and the experiment records that are
small enough to commit (`results/metrics/*.json`, `results/tables/*`), but not the large artefacts:
the dataset, the feature matrix, the trained models and the synthetic geothermal signals. This script
is the documented way to rebuild those, and the only way `scripts/setup_web.ps1` prepares them.

Stages are ordered by dependency and are skipped when their outputs already exist, so running this
twice costs nothing. Each stage is a module of the repository, run exactly as `scripts/run_all.py`
runs it; there is no second implementation of the pipeline.

Usage
-----
    python scripts/prepare_artifacts.py                     # build what is missing (~10 min cold)
    python scripts/prepare_artifacts.py --check             # verify only, no heavy work
    python scripts/prepare_artifacts.py --force             # rebuild everything, ignore existing
    python scripts/prepare_artifacts.py --only simulate     # single stage (comma-separated)
    python scripts/prepare_artifacts.py --fast              # no smoke-test drive (see --help)

`--check` loads the final model, rebuilds the feature column list from the feature matrix and the
validation record, and refuses to pass if the model file, the feature order, the decision threshold
and the recorded metrics do not all describe the same experiment. It writes
`results/metrics/artifact_check.json` and exits non-zero on any mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
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
    outputs: list[str]
    needs: list[str] = field(default_factory=list)
    heavy: bool = False

    def missing_outputs(self) -> list[str]:
        return [o for o in self.outputs if not (ROOT / o).exists()]

    def missing_needs(self) -> list[str]:
        return [n for n in self.needs if not (ROOT / n).exists()]


# Order matters: later stages consume earlier outputs. Same commands as scripts/run_all.py, restricted
# to the subset the three deliverables need (the EDA/audit stages are documented there and run in the
# full pipeline, but nothing downstream reads their outputs).
STAGES: list[Stage] = [
    Stage("data", "Dataset (pinned by SHA-256)", ["scripts/fetch_data.py"],
          ["data/raw/lead/lead1.0-small.csv"]),
    Stage("features", "Causal feature matrix", ["-m", "src.features.build_features"],
          ["data/processed/features.parquet"], needs=["data/raw/lead/lead1.0-small.csv"],
          heavy=True),
    Stage("splits", "Building-disjoint splits", ["-m", "src.features.splits"],
          ["data/processed/splits.json"], needs=["data/raw/lead/lead1.0-small.csv"]),
    Stage("baselines", "Baselines (comparison table)", ["-m", "src.models.baselines"],
          ["results/metrics/baselines_val.json"],
          needs=["data/processed/features.parquet", "data/processed/splits.json"], heavy=True),
    Stage("tune", "Hyper-parameter grid (selection evidence)", ["-m", "src.models.tune", "--grid"],
          ["results/metrics/tuning_best.json"], needs=["data/processed/features.parquet"],
          heavy=True),
    Stage("train_primary", "Train the default model", ["-m", "src.models.train_xgboost"],
          ["models/xgb_primary.json", "results/metrics/xgb_primary_val.json"],
          needs=["data/processed/features.parquet"], heavy=True),
    Stage("train_tuned", "Train the tuned model", ["-m", "src.models.train_xgboost",
          "--params-from", "results/metrics/tuning_best.json", "--tag", "tuned"],
          ["models/xgb_tuned.json", "results/metrics/xgb_tuned_val.json"],
          needs=["data/processed/features.parquet", "results/metrics/tuning_best.json"],
          heavy=True),
    Stage("train_weighted", "Train the class-weighted ablation", ["-m", "src.models.train_xgboost",
          "--weighted"],
          ["models/xgb_weighted.json", "results/metrics/xgb_weighted_val.json"],
          needs=["data/processed/features.parquet"], heavy=True),
    Stage("select", "Freeze the final model (validation ROC-AUC rule)",
          ["-m", "src.models.select_final"],
          ["models/xgb_final.json", "results/metrics/xgb_final_val.json",
           "results/metrics/model_selection.json"],
          needs=["results/metrics/xgb_primary_val.json", "results/metrics/xgb_tuned_val.json"]),
    Stage("holdout", "Held-out test evaluation (single read of the test split)",
          ["-m", "src.evaluation.holdout", "--variant", "final"],
          ["results/metrics/holdout_test_final.json", "results/tables/holdout_per_building_final.csv",
           "results/figures/holdout_roc_final.png", "results/figures/holdout_pr_final.png",
           "results/figures/holdout_calibration_final.png"],
          needs=["models/xgb_final.json"], heavy=True),
    Stage("compare", "Comparison tables", ["-m", "src.evaluation.compare"],
          ["results/tables/model_comparison_val.md", "results/tables/model_comparison_test.md"],
          needs=["results/metrics/holdout_test_final.json", "results/metrics/baselines_val.json"]),
    Stage("shap", "SHAP explanations", ["-m", "src.explainability.shap_explain"],
          ["results/metrics/shap_summary_final.json", "results/figures/shap_global_final.png",
           "results/figures/shap_dependence_final.png", "results/tables/shap_global_final.csv"],
          needs=["models/xgb_final.json"], heavy=True),
    Stage("simulate", "SIMULATED geothermal signals", ["-m", "src.simulation.geothermal"],
          ["data/synthetic/geothermal_signals_SIMULATED.csv",
           "data/synthetic/geothermal_faults_SIMULATED.json",
           "results/metrics/simulation_summary_SIMULATED.json"]),
    Stage("simulate_score", "Simulation transfer test", ["-m", "src.simulation.score_simulated"],
          ["results/metrics/simulation_detection_SIMULATED.json",
           "results/tables/simulation_detection_SIMULATED.csv",
           "results/figures/simulation_SIMULATED.png"],
          needs=["data/synthetic/geothermal_signals_SIMULATED.csv", "models/xgb_final.json"]),
    Stage("paper", "Manuscript figures and PDF", ["scripts/build_paper.py"],
          ["paper/figures", "paper/main.pdf"], needs=["paper/main.tex"]),
]

BY_KEY = {s.key: s for s in STAGES}

# What the web service needs before it may bind a port (`scripts/start_web.ps1` checks the same list).
SERVICE_REQUIRED = [
    "models/xgb_final.json",
    "results/metrics/xgb_final_val.json",
    "results/metrics/model_selection.json",
    "results/metrics/holdout_test_final.json",
    "results/tables/holdout_per_building_final.csv",
    "data/processed/features.parquet",
    "data/processed/splits.json",
    "data/synthetic/geothermal_signals_SIMULATED.csv",
    "data/synthetic/geothermal_faults_SIMULATED.json",
    "results/metrics/simulation_summary_SIMULATED.json",
    "backend/app/static/index.html",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_stage(stage: Stage, force: bool) -> dict:
    missing = stage.missing_needs()
    if missing:
        return {"stage": stage.key, "status": "blocked", "missing": missing, "seconds": 0.0}

    if not force and not stage.missing_outputs():
        return {"stage": stage.key, "status": "have", "seconds": 0.0}

    cmd = [PY, *stage.cmd]
    t0 = time.time()
    print(f"\n=== {stage.label} ===\n$ {' '.join(cmd[1:])}", flush=True)
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        return {"stage": stage.key, "status": "failed", "seconds": round(time.time() - t0, 1),
                "returncode": None, "problem": "timed out after 3600s"}
    dt = time.time() - t0
    for line in (proc.stdout or "").strip().splitlines()[-8:]:
        print(f"    {line}")
    if proc.returncode != 0:
        print((proc.stderr or "")[-2000:])
        return {"stage": stage.key, "status": "failed", "seconds": round(dt, 1),
                "returncode": proc.returncode}

    still = stage.missing_outputs()
    if still:
        print(f"    command succeeded but did not produce: {still}")
        return {"stage": stage.key, "status": "failed", "seconds": round(dt, 1),
                "returncode": 0, "missing": still}
    print(f"    [ok in {dt:.1f}s]")
    return {"stage": stage.key, "status": "ok", "seconds": round(dt, 1), "returncode": 0}


def check() -> tuple[int, dict]:
    """Verify that model, feature order, threshold and metrics describe one experiment."""
    problems: list[str] = []
    facts: dict = {}

    for rel in SERVICE_REQUIRED:
        if not (ROOT / rel).exists():
            problems.append(f"missing required artefact: {rel}")

    if problems:  # nothing else can be checked meaningfully
        facts = {"ok": False, "problems": problems}
        return 1, facts

    import pandas as pd

    sys.path.insert(0, str(ROOT))
    from src.data.load import feature_columns  # noqa: E402
    from src.inference.service import AnomalyScorer  # noqa: E402

    val = json.loads((ROOT / "results/metrics/xgb_final_val.json").read_text(encoding="utf-8"))
    selection = json.loads((ROOT / "results/metrics/model_selection.json").read_text(encoding="utf-8"))
    holdout = json.loads((ROOT / "results/metrics/holdout_test_final.json").read_text(encoding="utf-8"))

    # 1. the model loads through the same class the API uses
    try:
        scorer = AnomalyScorer("final")
    except Exception as exc:  # pragma: no cover - reported, not raised
        problems.append(f"AnomalyScorer('final') failed to load the model: {exc!r}")
        return 1, {"ok": False, "problems": problems}

    # 2. feature order: model booster vs validation record vs the feature matrix on disk
    booster_order = [str(f) for f in (scorer.model.get_booster().feature_names or [])]
    record_order = list(scorer.feature_order)
    matrix_cols = feature_columns(pd.read_parquet(ROOT / "data/processed/features.parquet"))
    if not booster_order:
        problems.append("model carries no feature names (stored as a bare booster)")
    if len(set(map(len, [booster_order, record_order, matrix_cols]))) != 1:
        problems.append(
            f"feature counts differ: booster {len(booster_order)}, record {len(record_order)}, "
            f"matrix {len(matrix_cols)}")
    if booster_order and booster_order != record_order:
        diff = [c for c in booster_order if c not in record_order] + \
               [c for c in record_order if c not in booster_order]
        problems.append(f"model feature order != validation record order (symmetric difference {diff})")
    if record_order != matrix_cols:
        problems.append("validation record feature order != feature matrix column order")

    # 3. threshold: one number, one source
    selected = selection.get("selected")
    selected_cand = next((c for c in selection.get("candidates", []) if c.get("variant") == selected), {})
    thresholds = {
        "validation record": float(val["threshold"]),
        "selection record": float(selected_cand["threshold"]) if "threshold" in selected_cand else None,
        "scorer": float(scorer.threshold),
    }
    distinct = {v for v in thresholds.values() if v is not None}
    if len(distinct) > 1:
        problems.append(f"threshold disagrees across records: {thresholds}")

    # 4. the frozen model is the selected one, and the split is the documented one
    if val.get("selected_from") and val["selected_from"] != selected:
        problems.append(
            f"xgb_final_val.json selected_from={val.get('selected_from')!r} but "
            f"model_selection.json selected={selected!r}")
    splits = json.loads((ROOT / "data/processed/splits.json").read_text(encoding="utf-8"))
    n_train, n_val, n_test = (len(splits.get(k, [])) for k in ("train", "val", "test"))
    if (n_train, n_val, n_test) != (140, 30, 30):
        problems.append(f"split sizes are {(n_train, n_val, n_test)}, documented as (140, 30, 30)")

    # 5. metrics cited by the manuscript exist in the holdout record
    for key in ("roc_auc", "pr_auc", "brier", "event_recall"):
        if key not in holdout:
            problems.append(f"holdout record lacks '{key}'")

    facts = {
        "ok": not problems,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_file": "models/xgb_final.json",
        "model_sha256": sha256(ROOT / "models/xgb_final.json"),
        "model_tag": scorer.model_tag,
        "n_features": len(record_order),
        "feature_source": "features.parquet columns == model booster names == xgb_final_val.json['features']",
        "threshold": float(scorer.threshold),
        "thresholds_seen": thresholds,
        "validation_metrics": {k: val.get(k) for k in ("roc_auc", "pr_auc", "event_recall")},
        "holdout_metrics": {k: holdout.get(k) for k in ("roc_auc", "pr_auc", "brier", "event_recall")},
        "split_sizes": {"train": n_train, "validation": n_val, "test": n_test},
        "selected_from": val.get("selected_from"),
        "selection_rule": selection.get("rule"),
        "problems": problems,
    }
    out = ROOT / "results/metrics/artifact_check.json"
    out.write_text(json.dumps(facts, indent=2), encoding="utf-8")
    return (0 if not problems else 1), facts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="verify existing artefacts only; build nothing")
    ap.add_argument("--missing", action="store_true",
                    help="print the service artefacts that are absent, one per line (exit 1 if any)")
    ap.add_argument("--force", action="store_true", help="rebuild even when outputs exist")
    ap.add_argument("--only", default=None, help="comma-separated stage keys")
    ap.add_argument("--list", action="store_true", help="list stages and their outputs")
    args = ap.parse_args()

    if args.missing:
        absent = [rel for rel in SERVICE_REQUIRED if not (ROOT / rel).exists()]
        for rel in absent:
            print(rel)
        return 1 if absent else 0

    if args.list:
        for s in STAGES:
            mark = "have" if not s.missing_outputs() else "MISSING"
            print(f"{s.key:16s} [{mark:7s}] {s.label}")
            for o in s.outputs:
                print(f"{'':16s}   -> {o}")
        return 0

    if args.check:
        code, facts = check()
        if facts.get("ok"):
            print(f"artifact check PASSED ({facts['n_features']} features, threshold "
                  f"{facts['threshold']}, val ROC-AUC {facts['validation_metrics']['roc_auc']}, "
                  f"test ROC-AUC {facts['holdout_metrics']['roc_auc']})")
            print(f"wrote {ROOT / 'results/metrics/artifact_check.json'}")
        else:
            print("artifact check FAILED")
            for p in facts.get("problems", []):
                print(f"  - {p}")
            missing = [p.split(": ", 1)[1] for p in facts.get("problems", [])
                       if p.startswith("missing required artefact")]
            if missing:
                print("\nrecover with:")
                print("    python scripts/prepare_artifacts.py            # or scripts/setup_web.ps1")
        return code

    stages = list(STAGES)
    if args.only:
        keys = [k.strip() for k in args.only.split(",")]
        unknown = [k for k in keys if k not in BY_KEY]
        if unknown:
            print(f"unknown stage(s) {unknown}; valid: {list(BY_KEY)}")
            return 2
        stages = [BY_KEY[k] for k in keys]

    todo = [s for s in stages if args.force or s.missing_outputs()]
    if not todo:
        print("all artefacts present; nothing to build (use --force to rebuild)")
    for s in todo:
        r = run_stage(s, args.force)
        if r["status"] == "blocked":
            print(f"\nBLOCKED {s.key}: needs {r['missing']}")
            print("  fetch the dataset first:  python scripts/fetch_data.py")
            return 1
        if r["status"] == "failed":
            print(f"\nFAILED {s.key}: {s.label}")
            return 1

    code, facts = check()
    if facts.get("ok"):
        print("\nall artefacts present and consistent")
    else:
        print("\nbuilt, but the consistency check did not pass:")
        for p in facts.get("problems", []):
            print(f"  - {p}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
