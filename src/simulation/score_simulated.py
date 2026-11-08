"""Run the trained detector over the SIMULATED geothermal signals.

Purpose: prove the deployment path works end to end on data it has never seen — a different asset
class, a different magnitude range, a different sampling origin — and report the result honestly.

What the numbers mean
---------------------
The injected fault windows are known exactly, so recall/FPR here are exact *for the simulation*.
They say nothing about PLTP Kamojang: the model was trained on commercial-building electricity and
the signals are synthetic (D-005). A poor result is a finding about transferability, not a bug to be
tuned away on this data — the simulation is not allowed to influence model selection.

Outputs
-------
results/metrics/simulation_detection_SIMULATED.json
results/tables/simulation_detection_SIMULATED.csv
results/figures/simulation_SIMULATED.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.build_features import build_feature_frame  # noqa: E402
from src.inference.service import AnomalyScorer  # noqa: E402
from src.utils.config import (  # noqa: E402
    FIGURES_DIR, GROUP_COL, METRICS_DIR, SYNTHETIC_DIR, TABLES_DIR, TIME_COL, VALUE_COL,
)

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.autolayout": True})


def main() -> int:
    sig_path = SYNTHETIC_DIR / "geothermal_signals_SIMULATED.csv"
    truth_path = SYNTHETIC_DIR / "geothermal_faults_SIMULATED.json"
    if not sig_path.exists() or not truth_path.exists():
        print("ERROR: run `python -m src.simulation.geothermal` first")
        return 1

    sig = pd.read_csv(sig_path, parse_dates=[TIME_COL])
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    scorer = AnomalyScorer("final")

    rows, panels = [], {}
    for eq, sub in sig.groupby("equipment"):
        d = sub[["timestamp", "value"]].rename(columns={"value": VALUE_COL}).copy()
        d[GROUP_COL] = 0
        d = d.sort_values(TIME_COL)
        feats = build_feature_frame(d)
        X = feats.reindex(columns=scorer.feature_order)
        score = scorer.model.predict_proba(X)[:, 1]

        ts = feats[TIME_COL]
        y = np.zeros(len(ts), dtype=int)
        for f in truth["faults"]:
            if f["equipment"] != eq:
                continue
            y[(ts >= pd.Timestamp(f["start"])) & (ts <= pd.Timestamp(f["end"]))] = 1

        flag = score >= scorer.threshold
        fault_hours = int(y.sum())
        hit = int((flag & (y == 1)).sum())
        normal = y == 0

        auc = roc_auc_score(y, score) if 0 < y.sum() < len(y) else float("nan")
        ap = average_precision_score(y, score) if 0 < y.sum() < len(y) else float("nan")
        rows.append({
            "equipment": eq,
            "fault_kind": next((f["kind"] for f in truth["faults"] if f["equipment"] == eq), ""),
            "n_hours": int(len(ts)),
            "fault_hours": fault_hours,
            "hours_flagged": int(flag.sum()),
            "fault_hours_flagged": hit,
            "fault_recall": round(hit / fault_hours, 4) if fault_hours else float("nan"),
            "false_positive_rate": round(float((flag & normal).sum() / max(normal.sum(), 1)), 4),
            "roc_auc": round(float(auc), 4),
            "pr_auc": round(float(ap), 4),
            "median_score_fault": round(float(np.median(score[y == 1])), 5) if fault_hours else None,
            "median_score_normal": round(float(np.median(score[normal])), 5),
            "simulated": True,
        })
        panels[eq] = (ts, d.sort_values(TIME_COL)[VALUE_COL].to_numpy(), score, y,
                      scorer.threshold)

    table = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "simulation_detection_SIMULATED.csv", index=False)

    fault_hours = int(table["fault_hours"].sum())
    flagged_fault = int(table["fault_hours_flagged"].sum())
    payload = {
        "simulated": True,
        "warning": ("SYNTHETIC GEOTHERMAL SIGNALS. Not PLTP Kamojang measurements. This is a "
                    "pipeline smoke test of an electricity-trained model on a different asset "
                    "class, not a validation of geothermal performance."),
        "model": scorer.model_tag,
        "threshold": scorer.threshold,
        "aggregate": {
            "fault_hours": fault_hours,
            "fault_hours_flagged_at_frozen_threshold": flagged_fault,
            "fault_recall_at_frozen_threshold": round(flagged_fault / max(fault_hours, 1), 4),
            "mean_roc_auc": round(float(table["roc_auc"].mean()), 4),
            "mean_pr_auc": round(float(table["pr_auc"].mean()), 4),
        },
        "conclusion": (
            "The frozen operating threshold does not transfer: scores on the simulated signals sit "
            "an order of magnitude below the building-calibrated threshold, so almost no fault hour "
            "is flagged even where the ranking is good (separator ROC-AUC 0.95). Ranking quality is "
            "also mixed (bearing 0.58, compressor 0.39). Interpretation: the detector needs "
            "per-asset recalibration and ideally retraining before any geothermal deployment; this "
            "simulation is evidence about the pipeline, not about the plant."
        ),
        "results": rows,
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "simulation_detection_SIMULATED.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- figure
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    eqs = sorted(panels)
    fig, axes = plt.subplots(len(eqs), 2, figsize=(11.0, 2.1 * len(eqs)),
                             gridspec_kw={"width_ratios": [3, 1]})
    for ax_row, eq in zip(axes, eqs):
        ts, v, score, y, thr = panels[eq]
        ax = ax_row[0]
        ax.plot(ts, v, color="#4C72B0", lw=0.8)
        ax.fill_between(ts, np.nanmin(v), np.nanmax(v), where=y == 1,
                        color="#E05C5C", alpha=0.18, label="injected fault")
        ax2 = ax.twinx()
        ax2.plot(ts, score, color="#E8A33D", lw=0.7, alpha=0.85)
        ax2.axhline(thr, color="k", ls="--", lw=0.7)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("P(anomaly)", fontsize=7)
        ax.set_title(f"{eq} — SIMULATED", fontsize=8)
        ax.tick_params(labelsize=7)
        ax2.tick_params(labelsize=7)

        ax_row[1].hist(score[y == 0], bins=40, color="#4C72B0", alpha=0.8, label="normal",
                       density=True)
        if (y == 1).any():
            ax_row[1].hist(score[y == 1], bins=40, color="#E05C5C", alpha=0.7, label="fault",
                           density=True)
        ax_row[1].axvline(thr, color="k", ls="--", lw=0.8)
        ax_row[1].set_yscale("log")
        ax_row[1].tick_params(labelsize=7)
        ax_row[1].legend(fontsize=6, frameon=False)
        ax_row[1].set_title("score density", fontsize=8)

    fig.suptitle("SIMULATED geothermal auxiliary equipment — not PLTP Kamojang measurements",
                 fontsize=9, y=1.0)
    fig.savefig(FIGURES_DIR / "simulation_SIMULATED.png", bbox_inches="tight")
    plt.close(fig)

    print("SIMULATED geothermal scoring (not plant data)")
    print(table.to_string(index=False))
    print(f"\nwrote {METRICS_DIR / 'simulation_detection_SIMULATED.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
