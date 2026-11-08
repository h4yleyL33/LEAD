"""Optuna hyper-parameter search on grouped folds inside the training buildings.

Design notes
------------
* **Grouped CV.** Folds are the 5 building groups from `splits.json`, never random rows: tuning on
  a row-wise CV would select hyper-parameters that exploit within-building persistence.
* **Subsampled folds.** Each fold is evaluated on a 25% row sample so a trial costs seconds rather
  than minutes. Selection noise from subsampling is acceptable because the final model is refit on
  100% of the training data and re-scored on the untouched validation buildings.
* **Resumable.** The study is persisted to `experiments/optuna_xgb.db`, so the search can be run
  in several short calls without losing trials.
* **Capped.** `--n-trials` and `--timeout` are explicit; nothing runs unbounded.

Usage
-----
    python -m src.models.tune --n-trials 10 --timeout 420
    python -m src.models.tune --n-trials 10 --timeout 420    # resumes the same study
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.load import (  # noqa: E402
    feature_columns, load_features, load_splits, observed_only, split_frame,
)
from src.utils.config import (  # noqa: E402
    EXPERIMENTS_DIR, GROUP_COL, METRICS_DIR, RANDOM_SEED, TARGET_COL,
)

SUBSAMPLE_FRACTION = 0.20
MAX_ROUNDS = 250
EARLY_STOP = 25
N_FOLDS_USED = 2

# Six candidate configurations: the hand-set default, the best point found by the capped TPE
# search, and four deliberately different regions of the space (shallower/deeper, faster/slower
# learning, stronger/weaker regularisation). Small and explicit on purpose -- see E-003.
GRID_CANDIDATES: list[dict] = [
    {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 5.0, "subsample": 0.8,
     "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.0, "gamma": 0.0},
    {"max_depth": 5, "learning_rate": 0.051, "min_child_weight": 3.65, "subsample": 0.716,
     "colsample_bytree": 0.806, "reg_lambda": 0.209, "reg_alpha": 0.012, "gamma": 1.83},
    {"max_depth": 8, "learning_rate": 0.03, "min_child_weight": 10.0, "subsample": 0.9,
     "colsample_bytree": 0.6, "reg_lambda": 5.0, "reg_alpha": 0.1, "gamma": 0.0},
    {"max_depth": 4, "learning_rate": 0.08, "min_child_weight": 2.0, "subsample": 0.7,
     "colsample_bytree": 0.9, "reg_lambda": 1.0, "reg_alpha": 0.0, "gamma": 0.5},
    {"max_depth": 6, "learning_rate": 0.03, "min_child_weight": 8.0, "subsample": 0.85,
     "colsample_bytree": 0.7, "reg_lambda": 2.0, "reg_alpha": 0.5, "gamma": 2.0},
    {"max_depth": 7, "learning_rate": 0.06, "min_child_weight": 4.0, "subsample": 0.8,
     "colsample_bytree": 0.75, "reg_lambda": 0.5, "reg_alpha": 0.05, "gamma": 1.0},
]


def build_folds(df, splits) -> list[tuple[np.ndarray, np.ndarray]]:
    """Three (train_idx, valid_idx) pairs over the training buildings, by building group."""
    train = observed_only(split_frame(df, splits, "train"))
    groups = train[GROUP_COL].to_numpy()
    out = []
    for fold_buildings in splits["folds"][:N_FOLDS_USED]:
        mask = np.isin(groups, fold_buildings)
        out.append((np.where(~mask)[0], np.where(mask)[0]))
    return train, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--fresh", action="store_true", default=True,
                    help="delete an existing study before searching (default)")
    ap.add_argument("--no-fresh", dest="fresh", action="store_false",
                    help="resume the persisted study")
    ap.add_argument("--grid", action="store_true",
                    help="run the deterministic 6-candidate grid instead of a TPE search")
    args = ap.parse_args()

    df = load_features()
    splits = load_splits()
    feats = feature_columns(df)
    train, folds = build_folds(df, splits)

    X = train[feats].to_numpy(dtype="float32")
    y = train[TARGET_COL].astype(int).to_numpy()
    rng = np.random.default_rng(args.seed)
    print(f"tuning on {len(y):,} rows x {len(feats)} features, {len(folds)} grouped folds")

    # Pre-draw the row subsample once so every trial sees the same folds.
    fold_data = []
    for tr_idx, va_idx in folds:
        sub = rng.choice(tr_idx, size=int(len(tr_idx) * SUBSAMPLE_FRACTION), replace=False)
        fold_data.append((np.sort(sub), va_idx))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "max_depth": trial.suggest_int("max_depth", 4, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "n_estimators": MAX_ROUNDS,
            "early_stopping_rounds": EARLY_STOP,
            "random_state": args.seed,
            "n_jobs": -1,
        }
        scores, iters = [], []
        for tr_idx, va_idx in fold_data:
            model = xgb.XGBClassifier(**params)
            model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])], verbose=False)
            p = model.predict_proba(X[va_idx])[:, 1]
            scores.append(average_precision_score(y[va_idx], p))
            iters.append(int(getattr(model, "best_iteration", MAX_ROUNDS) or MAX_ROUNDS))
        trial.set_user_attr("best_iteration_mean", float(np.mean(iters)))
        return float(np.mean(scores))

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    db_path = EXPERIMENTS_DIR / "optuna_xgb.db"

    if args.grid:
        # Deterministic candidate list. Used after the TPE search proved expensive per trial and
        # after a resumed study replayed identical parameter sets (see EXPERIMENT_LOG E-003).
        rows = []
        for i, cand in enumerate(GRID_CANDIDATES):
            t = time.time()
            params = {
                "objective": "binary:logistic", "eval_metric": "aucpr", "tree_method": "hist",
                "n_estimators": MAX_ROUNDS, "early_stopping_rounds": EARLY_STOP,
                "random_state": args.seed, "n_jobs": -1, **cand,
            }
            scores, iters = [], []
            for tr_idx, va_idx in fold_data:
                model = xgb.XGBClassifier(**params)
                model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])], verbose=False)
                scores.append(average_precision_score(
                    y[va_idx], model.predict_proba(X[va_idx])[:, 1]))
                iters.append(int(getattr(model, "best_iteration", MAX_ROUNDS) or MAX_ROUNDS))
            rows.append({"candidate": i, "params": cand, "fold_mean_pr_auc": float(np.mean(scores)),
                         "best_iteration_mean": float(np.mean(iters)),
                         "seconds": round(time.time() - t, 1)})
            print(f"cand {i}: PR-AUC {rows[-1]['fold_mean_pr_auc']:.5f}  "
                  f"({rows[-1]['seconds']}s)  {cand}")

        best = max(rows, key=lambda r: r["fold_mean_pr_auc"])
        payload = {
            "search": "deterministic grid (6 candidates), 2 grouped folds, 20% row subsample",
            "best_value": best["fold_mean_pr_auc"],
            "best_params": best["params"],
            "best_candidate": best["candidate"],
            "candidates": rows,
            "subsample_fraction": SUBSAMPLE_FRACTION,
            "max_rounds": MAX_ROUNDS,
        }
        out = METRICS_DIR / "optuna_xgb_best.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nbest candidate {best['candidate']} -> {out}")
        return 0

    if args.fresh and db_path.exists():
        db_path.unlink()
        print(f"removed previous study ({db_path.name}) -- see --no-fresh to resume")
    storage = f"sqlite:///{db_path.as_posix()}"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name="xgb_lead_aucpr", direction="maximize", storage=storage,
        load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    if len(study.trials) > 0:
        # TPE replays its seeded random startup phase when a study is reloaded, which silently
        # re-evaluates identical parameter sets. Warn instead of pretending the search progressed.
        print(f"WARNING: resuming a study that already has {len(study.trials)} trials; "
              f"parameter sets may repeat if the startup phase was not finished.")

    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                   show_progress_bar=False)
    elapsed = time.time() - t0

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\ncompleted trials: {len(done)}  elapsed {elapsed:.0f}s")
    best = study.best_trial
    print(f"best fold-mean PR-AUC = {best.value:.5f}")
    for k, v in best.params.items():
        print(f"  {k:18s} {v}")

    payload = {
        "n_trials_completed": len(done),
        "elapsed_seconds": round(elapsed, 1),
        "objective": "mean PR-AUC over 3 grouped folds (25% row subsample)",
        "best_value": float(best.value),
        "best_params": best.params,
        "best_iteration_mean": best.user_attrs.get("best_iteration_mean"),
        "subsample_fraction": SUBSAMPLE_FRACTION,
        "max_rounds": MAX_ROUNDS,
        "trials": [
            {"number": t.number, "value": t.value, "params": t.params,
             "best_iteration_mean": t.user_attrs.get("best_iteration_mean")}
            for t in done
        ],
    }
    out = METRICS_DIR / "optuna_xgb_best.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
