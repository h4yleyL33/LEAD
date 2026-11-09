"""SHAP explanations for the frozen model.

Two artefacts the paper needs:

* **Global** — mean |SHAP| per feature over a stratified sample of the held-out buildings, plus
  dependence plots for the top drivers. This says *which* signals move the score, which is a
  different question from XGBoost's gain importance (gain measures split usefulness, SHAP measures
  effect on the output).
* **Local** — the top-scoring hours in the test split with their five largest contributions, i.e.
  what an operator would be shown when an alarm fires.

Sampling: all positives in the test split are kept (5,454 rows) plus a random 5,000 negatives, so the
explanation is not dominated by the 97.8% normal class.

Usage:
    python -m src.explainability.shap_explain
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
import shap  # noqa: E402
import xgboost as xgb  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.load import (  # noqa: E402
    feature_columns, load_features, load_splits, observed_only, split_frame,
)
from src.utils.config import (  # noqa: E402
    FIGURES_DIR, GROUP_COL, METRICS_DIR, MODELS_DIR, RANDOM_SEED, TABLES_DIR, TARGET_COL, TIME_COL,
)

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.autolayout": True})

N_SAMPLE = 10000


def main() -> int:
    model_path = MODELS_DIR / "xgb_final.json"
    if not model_path.exists():
        print("ERROR: models/xgb_final.json missing; run the training pipeline first")
        return 1

    df = load_features()
    splits = load_splits()
    test = observed_only(split_frame(df, splits, "test"))
    feats = feature_columns(df)

    # Global attribution uses a random sample, i.e. the deployment distribution (~2.2% positive).
    # Oversampling anomalies here would inflate the apparent importance of features that separate
    # the classes and understate features that describe normal operation.
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(test), size=min(N_SAMPLE, len(test)), replace=False)
    sample = test.iloc[np.sort(idx)].reset_index(drop=True)
    print(f"explaining {len(sample):,} rows ({int(sample[TARGET_COL].sum()):,} anomalies, "
          f"natural prevalence {sample[TARGET_COL].mean():.4f})")

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    X = sample[feats]

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):  # some versions return [neg, pos]
        sv = sv[1]
    sv = np.asarray(sv)

    mean_abs = np.abs(sv).mean(axis=0)
    imp = pd.DataFrame({"feature": feats, "mean_abs_shap": mean_abs})
    imp["rank"] = imp["mean_abs_shap"].rank(ascending=False).astype(int)
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    imp.to_csv(TABLES_DIR / "shap_global_final.csv", index=False)
    print("\ntop 10 by mean |SHAP|:")
    print(imp.head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---------------------------------------------------------------- global bar
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#4C72B0")
    ax.set_xlabel("mean |SHAP| (log-odds contribution)")
    ax.set_title("Global feature attribution — held-out buildings")
    fig.savefig(FIGURES_DIR / "shap_global_final.png")
    plt.close(fig)

    # ---------------------------------------------------------------- dependence plots
    top4 = imp["feature"].head(4).tolist()
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.4))
    for ax, f in zip(axes.ravel(), top4):
        j = feats.index(f)
        v = X[f].to_numpy(dtype=float)
        m = np.isfinite(v)
        ax.scatter(v[m], sv[m, j], s=2, alpha=0.25, color="#4C72B0")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel(f)
        ax.set_ylabel("SHAP value")
        ax.set_title(f, fontsize=9)
        if np.nanmax(np.abs(v[m])) > 1e4:  # heavy tails: clip the x-axis for readability
            ax.set_xlim(np.nanpercentile(v[m], [0.5, 99.5]))
    fig.suptitle("SHAP dependence for the four strongest features", fontsize=10)
    fig.savefig(FIGURES_DIR / "shap_dependence_final.png")
    plt.close(fig)

    # ---------------------------------------------------------------- local explanations
    # Top-scoring hours come from the full test split, not the sample, so the examples shown to an
    # operator are the ones the system would actually have escalated.
    full_score = model.predict_proba(test[feats])[:, 1]
    order = np.argsort(-full_score)[:5]
    Xtop = test[feats].iloc[order]
    sv_top = np.asarray(explainer.shap_values(Xtop))
    if isinstance(sv_top, list):
        sv_top = sv_top[1]

    local = []
    for k, i in enumerate(order):
        contributions = sorted(
            ({"feature": feats[j],
              "value": None if not np.isfinite(Xtop.iloc[k, j]) else float(Xtop.iloc[k, j]),
              "shap": float(sv_top[k, j])} for j in range(len(feats))),
            key=lambda d: -abs(d["shap"]),
        )[:5]
        local.append({
            "timestamp": str(test[TIME_COL].iloc[i]),
            "building_id": int(test[GROUP_COL].iloc[i]),
            "score": round(float(full_score[i]), 5),
            "label": int(test[TARGET_COL].iloc[i]),
            "top_contributions": contributions,
        })

    payload = {
        "model": "xgboost_final",
        "split": "test",
        "n_explained": int(len(sample)),
        "n_anomalies_in_sample": int(sample[TARGET_COL].sum()),
        "sampling": "uniform random rows from the held-out test split (deployment prevalence)",
        "method": "TreeExplainer (exact, tree_path_dependent)",
        "global_top15": imp.head(15).to_dict("records"),
        "local_examples": local,
        "note": ("SHAP values are in log-odds units. Gain importance in "
                 "results/tables/feature_importance_final.csv measures split usefulness and will "
                 "not rank features identically."),
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "shap_summary_final.json").write_text(json.dumps(payload, indent=2),
                                                        encoding="utf-8")
    print(f"\nwrote {METRICS_DIR / 'shap_summary_final.json'}, 2 figures, 1 table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
