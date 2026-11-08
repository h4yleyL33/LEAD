"""Select the final model on the validation buildings and freeze it as `xgb_final`.

Selection rule, fixed before looking at any test data (D-004): highest ROC-AUC on the validation
buildings; ties broken by PR-AUC. The test split is not read here.

The winner is copied to `models/xgb_final.json` together with its validation metrics, so the API
and the paper both refer to one immutable artefact rather than to "whichever run was last".

Usage:
    python -m src.models.select_final
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import METRICS_DIR, MODELS_DIR  # noqa: E402

CANDIDATES = ["primary", "tuned", "weighted"]


def main() -> int:
    rows = []
    for tag in CANDIDATES:
        mp = METRICS_DIR / f"xgb_{tag}_val.json"
        model = MODELS_DIR / f"xgb_{tag}.json"
        if not mp.exists() or not model.exists():
            print(f"skip {tag}: artefacts missing")
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        rows.append({
            "variant": tag,
            "roc_auc": m["roc_auc"],
            "pr_auc": m["pr_auc"],
            "brier": m["brier"],
            "event_recall": m["event_recall"],
            "threshold": m["threshold"],
            "best_iteration": m.get("best_iteration"),
            "model_path": str(model.relative_to(MODELS_DIR.parent)),
        })

    if not rows:
        print("ERROR: no candidate models found; run src.models.train_xgboost first")
        return 1

    rows.sort(key=lambda r: (r["roc_auc"], r["pr_auc"]), reverse=True)
    winner = rows[0]
    print(f"{'variant':10s} {'ROC-AUC':>8s} {'PR-AUC':>8s} {'Brier':>9s} {'ev_recall':>10s}")
    for r in rows:
        print(f"{r['variant']:10s} {r['roc_auc']:8.4f} {r['pr_auc']:8.4f} "
              f"{r['brier']:9.5f} {r['event_recall']:10.4f}")
    print(f"\nselected: {winner['variant']} (rule: max validation ROC-AUC, tie-break PR-AUC)")

    shutil.copyfile(MODELS_DIR / f"xgb_{winner['variant']}.json", MODELS_DIR / "xgb_final.json")
    src = json.loads((METRICS_DIR / f"xgb_{winner['variant']}_val.json").read_text(encoding="utf-8"))
    src["selected_from"] = winner["variant"]
    src["selection_rule"] = "max validation ROC-AUC, tie-break PR-AUC (D-004)"
    (METRICS_DIR / "xgb_final_val.json").write_text(json.dumps(src, indent=2), encoding="utf-8")

    payload = {"rule": "max validation ROC-AUC, tie-break PR-AUC", "candidates": rows,
               "selected": winner["variant"],
               "spread_roc_auc": max(r["roc_auc"] for r in rows) - min(r["roc_auc"] for r in rows)}
    (METRICS_DIR / "model_selection.json").write_text(json.dumps(payload, indent=2),
                                                      encoding="utf-8")
    print(f"wrote models/xgb_final.json and results/metrics/model_selection.json "
          f"(ROC-AUC spread across candidates: {payload['spread_roc_auc']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
