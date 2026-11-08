"""Building-disjoint train/validation/test splits.

Rationale (DECISIONS.md D-002): hourly readings from the same building are near-duplicates of
each other (lag-1 autocorrelation 0.933), so a row-wise split would leak. The deployment scenario
is "a building the model has never seen", so the split is by building.

Construction: buildings are ordered by their per-building anomaly rate and dealt round-robin into
three buckets, which balances the label distribution across splits without random luck.

Usage:
    python -m src.features.splits            # writes data/processed/splits.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import (  # noqa: E402
    GROUP_COL, LEAD_RAW_CSV, N_FOLDS, PROCESSED_DIR, RANDOM_SEED, TARGET_COL, TIME_COL,
)

SPLIT_SIZES = {"train": 140, "val": 30, "test": 30}


def make_splits(df: pd.DataFrame, seed: int = RANDOM_SEED) -> dict:
    """Return {"train": [...], "val": [...], "test": [...], "folds": [[...], ...]}."""
    g = df.groupby(GROUP_COL)[TARGET_COL].agg(["mean", "sum", "size"]).sort_values("mean")
    buildings = g.index.to_numpy()

    # Deterministic jitter within equal anomaly rates, so ties are broken reproducibly.
    rng = np.random.default_rng(seed)
    jitter = rng.permutation(len(buildings))
    order = buildings[np.lexsort((jitter, g["mean"].to_numpy()))]

    buckets: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    names = ["train", "val", "test"]
    for b in order:
        # Greedy proportional assignment: always feed the split that is furthest behind its
        # target share. Because `order` is sorted by anomaly rate, this spreads the label
        # distribution evenly across the three splits instead of concentrating it.
        k = min(names, key=lambda n: (len(buckets[n]) / SPLIT_SIZES[n], names.index(n)))
        buckets[k].append(int(b))

    for k in names:
        buckets[k] = sorted(buckets[k])

    # Grouped folds inside train, for hyper-parameter selection only.
    train_sorted = sorted(buckets["train"], key=lambda b: g.loc[b, "mean"])
    folds = [[] for _ in range(N_FOLDS)]
    for i, b in enumerate(train_sorted):
        folds[i % N_FOLDS].append(int(b))

    assert not (set(buckets["train"]) & set(buckets["val"]))
    assert not (set(buckets["train"]) & set(buckets["test"]))
    assert not (set(buckets["val"]) & set(buckets["test"]))
    assert sum(len(v) for v in buckets.values()) == df[GROUP_COL].nunique()

    return {**buckets, "folds": folds, "seed": seed}


def summarise(df: pd.DataFrame, splits: dict) -> pd.DataFrame:
    rows = []
    for name in ("train", "val", "test"):
        sub = df[df[GROUP_COL].isin(splits[name])]
        rows.append({
            "split": name,
            "n_buildings": len(splits[name]),
            "n_rows": len(sub),
            "n_anomaly": int(sub[TARGET_COL].sum()),
            "anomaly_rate": float(sub[TARGET_COL].mean()),
            "missing_rate": float(sub["meter_reading"].isna().mean())
            if "meter_reading" in sub else float("nan"),
        })
    return pd.DataFrame(rows)


def main() -> int:
    df = pd.read_csv(LEAD_RAW_CSV, parse_dates=[TIME_COL])
    splits = make_splits(df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "splits.json"
    out.write_text(json.dumps(splits, indent=2), encoding="utf-8")

    summary = summarise(df, splits)
    summary.to_csv(PROCESSED_DIR / "splits_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nfold sizes: {[len(f) for f in splits['folds']]}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
