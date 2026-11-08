"""Reference baselines.

Three deliberately weak/strong reference points, so that the XGBoost number means something:

1. `prior`      -- constant prediction at the training prevalence. ROC-AUC 0.5 by construction;
                   this is the floor and it also fixes the Brier-score reference.
2. `logistic`   -- L2 logistic regression on standardised features with train-fitted median
                   imputation. A linear model cannot represent "ratio to own median is 3x" as
                   sharply as a tree can, but it is a fair, well-tuned linear reference.
3. `iforest`    -- Isolation Forest, unsupervised, fitted on training rows only. This is the
                   standard "no labels available" industrial practice, so it is the honest
                   comparison for the claim "labels buy you something".

Everything is scored on the *validation* buildings. The test split stays untouched here.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.load import (  # noqa: E402
    feature_columns, load_features, load_splits, observed_only, split_frame,
)
from src.evaluation.metrics import event_metrics, point_metrics, select_threshold  # noqa: E402
from src.utils.config import (  # noqa: E402
    GROUP_COL, METRICS_DIR, RANDOM_SEED, TARGET_COL, TIME_COL, environment_info,
)


def evaluate(name: str, y, p, val: pd.DataFrame, out: dict) -> dict:
    thr = select_threshold(y, p)
    m = point_metrics(y, p, threshold=thr["threshold"])
    m.update(event_metrics(y, p, val[GROUP_COL].to_numpy(), val[TIME_COL].to_numpy(),
                           thr["threshold"]))
    m["model"] = name
    m["fbeta_optimal"] = thr["fbeta"]
    out[name] = m
    print(f"{name:10s} roc_auc={m['roc_auc']:.4f}  pr_auc={m['pr_auc']:.4f}  "
          f"brier={m['brier']:.5f}  recall@fpr1%={m['recall_at_fpr_1pct']:.4f}  "
          f"event_recall={m['event_recall']:.4f}")
    return m


def main() -> int:
    t0 = time.time()
    df = load_features()
    splits = load_splits()
    train = observed_only(split_frame(df, splits, "train"))
    val = observed_only(split_frame(df, splits, "val"))

    feats = feature_columns(df)
    Xtr, ytr = train[feats], train[TARGET_COL].astype(int).to_numpy()
    Xva, yva = val[feats], val[TARGET_COL].astype(int).to_numpy()
    print(f"train {Xtr.shape}  val {Xva.shape}  features {len(feats)}")

    results: dict[str, dict] = {}

    # ---- 1. prior ---------------------------------------------------------
    prior = float(ytr.mean())
    p = np.full(len(yva), prior)
    results["prior"] = point_metrics(yva, p)
    results["prior"]["model"] = "prior"
    print(f"{'prior':10s} roc_auc={results['prior']['roc_auc']:.4f}  "
          f"brier={results['prior']['brier']:.5f}  (constant predictor, AUC is 0.5 by definition)")

    # ---- 2. logistic regression ------------------------------------------
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=300, solver="lbfgs", n_jobs=-1,
                                   random_state=RANDOM_SEED)),
    ])
    t = time.time()
    pipe.fit(Xtr, ytr)
    p = pipe.predict_proba(Xva)[:, 1]
    print(f"logistic fit {time.time() - t:.1f}s")
    evaluate("logistic", yva, p, val, results)
    results["logistic"]["fit_seconds"] = round(time.time() - t, 1)

    # ---- 3. isolation forest ---------------------------------------------
    # Unsupervised: fitted on the training feature matrix only, labels never used.
    prep = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    Xtr_s = prep.fit_transform(Xtr)
    Xva_s = prep.transform(Xva)
    t = time.time()
    iso = IsolationForest(n_estimators=200, max_samples=4096, contamination="auto",
                          random_state=RANDOM_SEED, n_jobs=-1)
    iso.fit(Xtr_s)
    # Higher score = more anomalous, so negate the (lower-is-worse) decision function.
    p = -iso.score_samples(Xva_s)
    print(f"iforest fit {time.time() - t:.1f}s")
    evaluate("iforest", yva, p, val, results)
    results["iforest"]["fit_seconds"] = round(time.time() - t, 1)

    payload = {
        "split_evaluated": "val",
        "n_train": int(len(Xtr)),
        "n_val": int(len(Xva)),
        "features": feats,
        "environment": environment_info(),
        "results": results,
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / "baselines_val.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}  total {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
