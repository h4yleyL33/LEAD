"""Single-shot evaluation of a trained model on the held-out test buildings.

Run this **once** per final model. Every number reported in the paper for the test split comes
from the JSON this script writes; nothing is transcribed by hand.

What is produced
----------------
* point metrics (ROC-AUC, PR-AUC, Brier, recall at 1% / 0.1% FPR) with a building-level
  bootstrap CI;
* event metrics at the validation-selected threshold (episode recall, detection latency,
  false-alarm episodes per building-day);
* per-building breakdown, so a single catastrophic building cannot hide inside an average;
* four figures: ROC, precision-recall, calibration, score distribution.

Usage
-----
    python -m src.evaluation.holdout --variant primary
    python -m src.evaluation.holdout --variant tuned
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.load import (  # noqa: E402
    feature_columns, load_features, load_splits, observed_only, split_frame,
)
from src.evaluation.metrics import (  # noqa: E402
    bootstrap_auc_ci, event_metrics, point_metrics,
)
from src.utils.config import (  # noqa: E402
    FIGURES_DIR, GROUP_COL, METRICS_DIR, MODELS_DIR, PREDICTIONS_DIR, TABLES_DIR, TARGET_COL,
    TIME_COL,
)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9, "axes.grid": True,
    "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
    "figure.autolayout": True,
})


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.mean()) * abs(y[m].mean() - p[m].mean())
    return float(ece)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="primary", help="model tag: primary | tuned | weighted")
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args()

    model_path = MODELS_DIR / f"xgb_{args.variant}.json"
    if not model_path.exists():
        print(f"ERROR: {model_path} not found")
        return 1
    val_metrics_path = METRICS_DIR / f"xgb_{args.variant}_val.json"
    if not val_metrics_path.exists():
        print(f"ERROR: {val_metrics_path} not found (threshold must be chosen on validation)")
        return 1
    val_metrics = json.loads(val_metrics_path.read_text(encoding="utf-8"))
    threshold = float(val_metrics["threshold"])

    df = load_features()
    splits = load_splits()
    test = observed_only(split_frame(df, splits, "test"))
    feats = feature_columns(df)

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    p = model.predict_proba(test[feats])[:, 1]
    y = test[TARGET_COL].astype(int).to_numpy()
    groups = test[GROUP_COL].to_numpy()
    times = test[TIME_COL].to_numpy()

    m = point_metrics(y, p, threshold=threshold)
    m.update(event_metrics(y, p, groups, times, threshold))
    m.update(bootstrap_auc_ci(y, p, groups, n_boot=args.n_boot))
    m.update({
        "model": f"xgboost_{args.variant}",
        "split_evaluated": "test",
        "threshold_source": f"val ({val_metrics_path.name})",
        "ece_15bin": expected_calibration_error(y, p),
        "n_buildings": int(len(np.unique(groups))),
        "val_reference": {
            "roc_auc": val_metrics["roc_auc"],
            "pr_auc": val_metrics["pr_auc"],
            "event_recall": val_metrics["event_recall"],
        },
    })

    print(f"test buildings: {m['n_buildings']}  rows: {m['n']:,}  anomalies: {m['n_pos']:,}")
    print(f"ROC-AUC {m['roc_auc']:.4f}  (95% CI {m['auc_ci95_low']:.4f}-{m['auc_ci95_high']:.4f})")
    print(f"PR-AUC  {m['pr_auc']:.4f}   base rate {m['base_rate']:.4f}")
    print(f"Brier {m['brier']:.5f}   ECE {m['ece_15bin']:.5f}")
    print(f"recall@FPR1% {m['recall_at_fpr_1pct']:.4f}   @FPR0.1% {m['recall_at_fpr_0p1pct']:.4f}")
    print(f"@thr {threshold:.4f}: precision {m['precision']:.4f}  recall {m['recall']:.4f}  "
          f"F1 {m['f1']:.4f}  alert rate {m['alert_rate']:.4f}")
    print(f"events {m['n_events']}  detected {m['events_detected']}  "
          f"event recall {m['event_recall']:.4f}  latency {m['median_detection_latency_h']}h  "
          f"false-alarm episodes/building-day {m['false_alarms_per_building_day']:.3f}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / f"holdout_test_{args.variant}.json").write_text(
        json.dumps(m, indent=2), encoding="utf-8")

    pred = test[[GROUP_COL, TIME_COL, TARGET_COL]].copy()
    pred["score"] = p
    pred["predicted"] = (p >= threshold).astype(int)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(PREDICTIONS_DIR / f"test_predictions_{args.variant}.parquet", index=False)

    per_b = (pred.assign(correct=((pred["predicted"] == 1) & (pred[TARGET_COL] == 1)).astype(int))
             .groupby(GROUP_COL)
             .apply(lambda d: pd.Series({
                 "n": len(d),
                 "n_anomaly": int(d[TARGET_COL].sum()),
                 "n_flagged": int(d["predicted"].sum()),
                 "n_anomaly_flagged": int(d["correct"].sum()),
                 "recall": d["correct"].sum() / max(int(d[TARGET_COL].sum()), 1),
                 "mean_score": float(d["score"].mean()),
             }), include_groups=False)
             .reset_index())
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    per_b.to_csv(TABLES_DIR / f"holdout_per_building_{args.variant}.csv", index=False)
    print(f"\nper-building recall: median {per_b['recall'].median():.3f}  "
          f"min {per_b['recall'].min():.3f}  max {per_b['recall'].max():.3f}")

    # ---------------------------------------------------------------- figures
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.plot(fpr, tpr, color="#4C72B0", lw=1.8, label=f"XGBoost (AUC {m['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="k", lw=0.6, ls="--", label="chance")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("ROC — held-out buildings")
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    fig.savefig(FIGURES_DIR / f"holdout_roc_{args.variant}.png")
    plt.close(fig)

    prec, rec, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.plot(rec, prec, color="#C44E52", lw=1.8, label=f"PR-AUC {m['pr_auc']:.3f}")
    ax.axhline(m["base_rate"], color="k", lw=0.6, ls="--", label=f"base rate {m['base_rate']:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall — held-out buildings")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(FIGURES_DIR / f"holdout_pr_{args.variant}.png")
    plt.close(fig)

    frac_pos, mean_pred = calibration_curve(y, p, n_bins=15, strategy="quantile")
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.plot([0, 1], [0, 1], color="k", lw=0.6, ls="--")
    ax.plot(mean_pred, frac_pos, "o-", color="#55A868", lw=1.4, ms=3,
            label=f"ECE {m['ece_15bin']:.4f}")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed anomaly rate")
    ax.set_title("Calibration — held-out buildings")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(FIGURES_DIR / f"holdout_calibration_{args.variant}.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    bins = np.linspace(0, max(np.percentile(p, 99.9), threshold * 1.5), 60)
    ax.hist(p[y == 0], bins=bins, color="#4C72B0", alpha=0.75, label="normal", density=True)
    ax.hist(p[y == 1], bins=bins, color="#C44E52", alpha=0.65, label="anomaly", density=True)
    ax.axvline(threshold, color="k", ls="--", lw=1.0, label=f"threshold {threshold:.3f}")
    ax.set_yscale("log")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Density (log)")
    ax.set_title("Score separation on held-out buildings")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(FIGURES_DIR / f"holdout_scores_{args.variant}.png")
    plt.close(fig)

    print(f"\nwrote metrics, predictions, per-building table and 4 figures for '{args.variant}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
