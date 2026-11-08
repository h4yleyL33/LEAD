"""Feature-family ablation: how much does each family of causal features buy on its own?

Seven "family-only" models are trained with identical hyper-parameters and early stopping on the
validation buildings. Comparing them to the full model answers the research question "which
physical signal carries the anomaly information?" without the confounding of drop-one-family
retraining (which is also reported separately by `--drop` mode).

Usage
-----
    python -m src.models.ablation                  # all families, family-only models
    python -m src.models.ablation --families level,dynamics
    python -m src.models.ablation --drop           # drop-one-family-out instead
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.load import (  # noqa: E402
    feature_columns, load_features, load_splits, observed_only, split_frame,
)
from src.evaluation.metrics import event_metrics, point_metrics, select_threshold  # noqa: E402
from src.utils.config import (  # noqa: E402
    GROUP_COL, METRICS_DIR, RANDOM_SEED, TABLES_DIR, TARGET_COL, TIME_COL,
)

PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5.0,
    "reg_lambda": 1.0,
    "n_estimators": 800,
    "early_stopping_rounds": 50,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}


def run(name: str, cols: list[str], train, val, ytr, yva) -> dict:
    t = time.time()
    model = xgb.XGBClassifier(**PARAMS)
    model.fit(train[cols], ytr, eval_set=[(val[cols], yva)], verbose=False)
    p = model.predict_proba(val[cols])[:, 1]
    thr = select_threshold(yva, p)
    m = point_metrics(yva, p, threshold=thr["threshold"])
    m.update(event_metrics(yva, p, val[GROUP_COL].to_numpy(), val[TIME_COL].to_numpy(),
                           thr["threshold"]))
    m.update({"subset": name, "n_features": len(cols), "best_iteration":
              int(getattr(model, "best_iteration", 0) or 0), "fit_seconds": round(time.time() - t, 1)})
    print(f"{name:16s} n_feat={len(cols):2d}  roc_auc={m['roc_auc']:.4f}  "
          f"pr_auc={m['pr_auc']:.4f}  event_recall={m['event_recall']:.4f}  ({m['fit_seconds']}s)")
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=None, help="comma-separated subset of families")
    ap.add_argument("--drop", action="store_true", help="drop-one-family-out instead of family-only")
    args = ap.parse_args()

    meta = json.loads((METRICS_DIR / "feature_metadata.json").read_text(encoding="utf-8"))
    families: dict[str, str] = meta["families"]

    df = load_features()
    splits = load_splits()
    train = observed_only(split_frame(df, splits, "train"))
    val = observed_only(split_frame(df, splits, "val"))
    feats = feature_columns(df)
    ytr = train[TARGET_COL].astype(int).to_numpy()
    yva = val[TARGET_COL].astype(int).to_numpy()

    fam_names = sorted(set(families.values()))
    if args.families:
        fam_names = [f.strip() for f in args.families.split(",")]

    rows = []
    for fam in fam_names:
        members = [f for f in feats if families.get(f) == fam]
        if not members:
            print(f"{fam}: no features, skipped")
            continue
        if args.drop:
            cols = [f for f in feats if families.get(f) != fam]
            label = f"drop_{fam}"
        else:
            cols = members
            label = fam
        rows.append(run(label, cols, train, val, ytr, yva))

    out = pd.DataFrame(rows)
    suffix = "drop" if args.drop else "only"
    path = METRICS_DIR / f"ablation_families_{suffix}.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"results": []}
    existing["results"] = [r for r in existing["results"] if r["subset"] not in set(out["subset"])]
    existing["results"].extend(rows)
    existing["mode"] = suffix
    existing["params"] = PARAMS
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(existing["results"]).sort_values("roc_auc", ascending=False).to_csv(
        TABLES_DIR / f"ablation_families_{suffix}.csv", index=False)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
