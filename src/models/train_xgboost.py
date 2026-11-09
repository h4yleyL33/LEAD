"""Train the XGBoost anomaly detector on the causal feature matrix.

Protocol
--------
* fit on the 140 training buildings, early-stop on the 30 validation buildings (aucpr);
* the test split is never loaded here -- it is scored once, in `src/evaluation/holdout.py`;
* two variants are supported so the class-weighting question is answered empirically rather
  than by opinion:
    --weighted   sets `scale_pos_weight = neg/pos` (the usual "fix the imbalance" reflex)
    (default)    leaves the prior intact, so predicted probabilities stay calibrated

Usage
-----
    python -m src.models.train_xgboost                 # primary, unweighted
    python -m src.models.train_xgboost --weighted      # ablation
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
    GROUP_COL, METRICS_DIR, MODELS_DIR, PREDICTIONS_DIR, RANDOM_SEED, TABLES_DIR, TARGET_COL,
    TIME_COL, environment_info,
)

BASE_PARAMS: dict = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "gamma": 0.0,
    "n_estimators": 3000,
    "early_stopping_rounds": 75,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weighted", action="store_true",
                    help="set scale_pos_weight = neg/pos instead of leaving the prior intact")
    ap.add_argument("--tag", default=None, help="suffix for output filenames")
    ap.add_argument("--max-rounds", type=int, default=None)
    ap.add_argument("--params-from", default=None,
                    help="JSON file with a 'best_params' dict (e.g. results/metrics/tuning_best.json)")
    args = ap.parse_args()

    tag = args.tag or ("weighted" if args.weighted else "primary")
    t0 = time.time()

    df = load_features()
    splits = load_splits()
    train = observed_only(split_frame(df, splits, "train"))
    val = observed_only(split_frame(df, splits, "val"))
    feats = feature_columns(df)

    Xtr, ytr = train[feats], train[TARGET_COL].astype(int).to_numpy()
    Xva, yva = val[feats], val[TARGET_COL].astype(int).to_numpy()

    params = dict(BASE_PARAMS)
    if args.max_rounds:
        params["n_estimators"] = args.max_rounds
    if args.params_from:
        tuned = json.loads(Path(args.params_from).read_text(encoding="utf-8"))["best_params"]
        # Only the searched keys are overridden; the training protocol keys stay under our control.
        allowed = {"max_depth", "learning_rate", "min_child_weight", "subsample",
                   "colsample_bytree", "reg_lambda", "reg_alpha", "gamma"}
        params.update({k: v for k, v in tuned.items() if k in allowed})
        print(f"loaded tuned params from {args.params_from}")

    pos = int(ytr.sum())
    neg = int(len(ytr) - pos)
    if args.weighted:
        params["scale_pos_weight"] = neg / max(pos, 1)
    print(f"variant={tag}  train={Xtr.shape}  val={Xva.shape}  pos={pos}  neg={neg}  "
          f"scale_pos_weight={params.get('scale_pos_weight', 1.0):.2f}")

    model = xgb.XGBClassifier(**params)
    t = time.time()
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    fit_s = time.time() - t
    best_iter = int(getattr(model, "best_iteration", params["n_estimators"] - 1) or 0)
    print(f"fit {fit_s:.1f}s  best_iteration={best_iter}  best_score={model.best_score:.5f}")

    p_va = model.predict_proba(Xva)[:, 1]
    thr = select_threshold(yva, p_va)
    m = point_metrics(yva, p_va, threshold=thr["threshold"])
    m.update(event_metrics(yva, p_va, val[GROUP_COL].to_numpy(), val[TIME_COL].to_numpy(),
                           thr["threshold"]))
    m.update({
        "model": f"xgboost_{tag}",
        "variant": tag,
        "split_evaluated": "val",
        "best_iteration": best_iter,
        "fit_seconds": round(fit_s, 1),
        "params": {k: v for k, v in params.items()},
        "fbeta_optimal": thr["fbeta"],
        "n_train": int(len(Xtr)),
        "n_val": int(len(Xva)),
        "features": feats,
        "environment": environment_info(),
    })
    print(f"roc_auc={m['roc_auc']:.4f}  pr_auc={m['pr_auc']:.4f}  brier={m['brier']:.5f}  "
          f"recall@fpr1%={m['recall_at_fpr_1pct']:.4f}  event_recall={m['event_recall']:.4f}  "
          f"thr={m['threshold']:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"xgb_{tag}.json"
    model.save_model(model_path)

    pred = val[[GROUP_COL, TIME_COL, TARGET_COL]].copy()
    pred["score"] = p_va
    pred["predicted"] = (p_va >= thr["threshold"]).astype(int)
    pred.to_parquet(PREDICTIONS_DIR / f"val_predictions_{tag}.parquet", index=False)

    (METRICS_DIR / f"xgb_{tag}_val.json").write_text(json.dumps(m, indent=2), encoding="utf-8")

    # Gain-based importance, normalised, for the paper's feature discussion.
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    total = sum(gain.values()) or 1.0
    imp = pd.DataFrame({
        "feature": list(gain.keys()),
        "gain": list(gain.values()),
    })
    imp["gain_pct"] = imp["gain"] / total * 100
    imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)
    imp.to_csv(TABLES_DIR / f"feature_importance_{tag}.csv", index=False)
    print("\ntop 10 by gain:")
    print(imp.head(10).to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    print(f"\nwrote {model_path}, {METRICS_DIR / f'xgb_{tag}_val.json'}  "
          f"total {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
