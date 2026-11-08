"""Central project configuration: paths, seeds, dataset constants.

Every experiment reads from here so that a single edit changes the whole pipeline.
"""
from __future__ import annotations

import os
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"

LEAD_RAW_CSV = RAW_DIR / "lead" / "lead1.0-small.csv"
LEAD_RAW_ZIP = RAW_DIR / "lead" / "lead1.0-small.zip"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

for _p in (PROCESSED_DIR, SYNTHETIC_DIR, MODELS_DIR, FIGURES_DIR, TABLES_DIR,
           METRICS_DIR, PREDICTIONS_DIR, EXPERIMENTS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
RANDOM_SEED = 42
N_FOLDS = 5

# ----------------------------------------------------------------------------
# Dataset constants (verified values are in results/metrics/data_audit.json;
# these are only schema names, not measured quantities)
# ----------------------------------------------------------------------------
TARGET_COL = "anomaly"
TIME_COL = "timestamp"
GROUP_COL = "building_id"
VALUE_COL = "meter_reading"

# ----------------------------------------------------------------------------
# Environment capture
# ----------------------------------------------------------------------------

def environment_info() -> dict:
    """Return a dict describing the execution environment for reproducibility."""
    import platform
    import sys

    info = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cwd": str(Path.cwd()),
    }
    for mod in ("numpy", "pandas", "sklearn", "xgboost", "scipy", "shap", "optuna"):
        try:
            m = __import__(mod)
            info[f"{mod}_version"] = getattr(m, "__version__", "unknown")
        except Exception:  # pragma: no cover - optional dependency
            info[f"{mod}_version"] = None
    return info


if __name__ == "__main__":
    import json

    print(json.dumps(environment_info(), indent=2))
