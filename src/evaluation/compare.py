"""Aggregate every measured model result into one comparison table.

Sources (all written by the scripts, none typed by hand):
  results/metrics/baselines_val.json      prior / logistic / isolation forest, validation
  results/metrics/xgb_{tag}_val.json      XGBoost variants, validation
  results/metrics/holdout_test_{tag}.json final model, test

Outputs:
  results/tables/model_comparison_val.csv
  results/tables/model_comparison_val.md
  results/tables/model_comparison_test.csv / .md

Usage:
    python -m src.evaluation.compare
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import METRICS_DIR, TABLES_DIR  # noqa: E402

VAL_COLS = ["model", "roc_auc", "pr_auc", "brier", "recall_at_fpr_1pct",
            "precision", "recall", "f1", "event_recall", "median_detection_latency_h"]


def _row(name: str, m: dict) -> dict:
    return {
        "model": name,
        "roc_auc": m.get("roc_auc"),
        "pr_auc": m.get("pr_auc"),
        "brier": m.get("brier"),
        "recall_at_fpr_1pct": m.get("recall_at_fpr_1pct"),
        "precision": m.get("precision"),
        "recall": m.get("recall"),
        "f1": m.get("f1"),
        "event_recall": m.get("event_recall"),
        "median_detection_latency_h": m.get("median_detection_latency_h"),
    }


def _markdown(df: pd.DataFrame) -> str:
    head = "| model | ROC-AUC | PR-AUC | Brier | recall@FPR1% | precision | recall | F1 | event recall |\n"
    head += "|---|---|---|---|---|---|---|---|---|\n"
    lines = []
    for _, r in df.iterrows():
        def f(v, nd=4):
            return "—" if v is None or pd.isna(v) else f"{v:.{nd}f}"
        lines.append(f"| {r['model']} | {f(r['roc_auc'])} | {f(r['pr_auc'])} | "
                     f"{f(r['brier'], 5)} | {f(r['recall_at_fpr_1pct'])} | {f(r['precision'])} | "
                     f"{f(r['recall'])} | {f(r['f1'])} | {f(r['event_recall'])} |")
    return head + "\n".join(lines) + "\n"


def main() -> int:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    b = METRICS_DIR / "baselines_val.json"
    if b.exists():
        data = json.loads(b.read_text(encoding="utf-8"))["results"]
        for name in ("prior", "logistic", "iforest"):
            if name in data:
                rows.append(_row(name, data[name]))

    for tag in ("primary", "tuned", "weighted"):
        p = METRICS_DIR / f"xgb_{tag}_val.json"
        if p.exists():
            rows.append(_row(f"xgboost_{tag}", json.loads(p.read_text(encoding="utf-8"))))

    val = pd.DataFrame(rows)
    if not val.empty:
        val = val.sort_values("roc_auc", ascending=False)
        val.to_csv(TABLES_DIR / "model_comparison_val.csv", index=False)
        (TABLES_DIR / "model_comparison_val.md").write_text(_markdown(val), encoding="utf-8")
        print("validation split")
        print(val[VAL_COLS].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    test_rows = []
    for tag in ("final", "tuned", "primary"):
        p = METRICS_DIR / f"holdout_test_{tag}.json"
        if p.exists():
            test_rows.append(_row(f"xgboost_{tag} (test)", json.loads(p.read_text(encoding="utf-8"))))
    if test_rows:
        t = pd.DataFrame(test_rows)
        t.to_csv(TABLES_DIR / "model_comparison_test.csv", index=False)
        (TABLES_DIR / "model_comparison_test.md").write_text(_markdown(t), encoding="utf-8")
        print("\nheld-out test split")
        print(t[VAL_COLS].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\nwrote tables to {TABLES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
