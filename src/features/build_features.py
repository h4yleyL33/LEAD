"""Causal feature engineering for hourly energy telemetry.

Design contract (enforced by tests/test_features_no_leakage.py):

    Every feature at time t is a deterministic function of
    {(building b, reading at time s) : s <= t}, and of nothing else.

Consequences:
  * no centred windows, no forward fills from the future, no target-derived encodings;
  * `building_id` is never a feature (the evaluation protocol is building-disjoint, so an
    identifier cannot generalise);
  * peer features use only *contemporaneous* values of other buildings (observable at t).

Implementation notes:
  * Each building is reindexed onto the full hourly grid of the dataset so that a rolling
    window of k rows means exactly k hours, even where source rows are absent.
  * Past-only statistics are computed on the series shifted by one hour, so the current
    reading never enters its own baseline.
  * Absent (building, hour) rows are not synthesised back into the output: features are
    emitted only for rows that exist in the source data.

Usage:
    python -m src.features.build_features
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal, overload

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import (  # noqa: E402
    GROUP_COL, LEAD_RAW_CSV, METRICS_DIR, PROCESSED_DIR, TARGET_COL, TIME_COL, VALUE_COL,
)

EPS = 1e-6

# Window (hours) -> minimum non-NaN observations required.
WINDOWS = {6: 3, 24: 12, 168: 24, 720: 48}

# Feature family tags, used by the ablation study in Phase 5.
FAMILY: dict[str, str] = {}


def _tag(names: list[str], family: str) -> list[str]:
    for n in names:
        FAMILY[n] = family
    return names


@overload
def build_feature_frame(
    df_long: pd.DataFrame,
    grid_start: pd.Timestamp | None = None,
    grid_end: pd.Timestamp | None = None,
    return_wide: Literal[False] = False,
) -> pd.DataFrame: ...


@overload
def build_feature_frame(
    df_long: pd.DataFrame,
    grid_start: pd.Timestamp | None = None,
    grid_end: pd.Timestamp | None = None,
    *,
    return_wide: Literal[True],
) -> tuple[pd.DataFrame, pd.DataFrame]: ...


def build_feature_frame(
    df_long: pd.DataFrame,
    grid_start: pd.Timestamp | None = None,
    grid_end: pd.Timestamp | None = None,
    return_wide: bool = False,
):
    """Build the causal feature matrix.

    Parameters
    ----------
    df_long : columns [building_id, timestamp, meter_reading]
    grid_start, grid_end : optional explicit grid bounds (used by the leakage test to
        truncate history without changing the feature definition).

    Returns
    -------
    DataFrame with [building_id, timestamp] + feature columns, one row per input row.
    """
    need = {GROUP_COL, TIME_COL, VALUE_COL}
    missing = need - set(df_long.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    d = df_long[[GROUP_COL, TIME_COL, VALUE_COL]].copy()
    d[TIME_COL] = pd.to_datetime(d[TIME_COL])
    d[GROUP_COL] = d[GROUP_COL].astype("int64")

    start = grid_start if grid_start is not None else d[TIME_COL].min()
    end = grid_end if grid_end is not None else d[TIME_COL].max()
    grid = pd.date_range(start, end, freq="h")
    buildings = np.sort(d[GROUP_COL].unique())

    wide = (
        d.pivot_table(index=TIME_COL, columns=GROUP_COL, values=VALUE_COL, aggfunc="first")
        .reindex(index=grid, columns=buildings)
        .astype("float64")
    )
    v = wide  # shape (T, B)

    # ---------------------------------------------------------------- past-only baselines
    past = v.shift(1)  # strictly before t
    stats: dict[int, dict[str, pd.DataFrame]] = {}
    for w, mp in WINDOWS.items():
        r = past.rolling(window=w, min_periods=mp)
        stats[w] = {
            "mean": r.mean(),
            "std": r.std(ddof=0),
            "median": r.median(),
            "max": r.max(),
            "min": r.min(),
        }

    feats: dict[str, pd.DataFrame] = {}

    # ---- family: level (scale-free transforms only; raw level does not transfer across buildings)
    feats["f_log_value"] = np.log1p(v.clip(lower=0))
    for w in WINDOWS:
        med = stats[w]["median"]
        feats[f"f_ratio_med_{w}"] = v / (med + EPS)
    _tag(["f_log_value"], "level")
    _tag([f"f_ratio_med_{w}" for w in WINDOWS], "level")

    # ---- family: deviation (z-score / relative deviation from own recent history)
    for w in WINDOWS:
        mean, std = stats[w]["mean"], stats[w]["std"]
        feats[f"f_z_{w}"] = (v - mean) / (std + EPS)
        feats[f"f_dev_rel_{w}"] = (v - mean) / (mean.abs() + EPS)
    _tag([f"f_z_{w}" for w in WINDOWS] + [f"f_dev_rel_{w}" for w in WINDOWS], "deviation")

    # ---- family: dynamics (first differences, same-hour comparisons, momentum)
    diff1 = v.diff()
    feats["f_diff1"] = diff1
    feats["f_abs_diff1"] = diff1.abs()
    diff_std24 = diff1.shift(1).rolling(24, min_periods=12).std(ddof=0)
    feats["f_diff1_z24"] = diff1 / (diff_std24 + EPS)
    feats["f_delta_24"] = v - v.shift(24)
    feats["f_delta_168"] = v - v.shift(168)
    feats["f_ratio_24"] = v / (v.shift(24) + EPS)
    feats["f_ratio_168"] = v / (v.shift(168) + EPS)
    feats["f_momentum_6_24"] = stats[6]["mean"] / (stats[24]["mean"] + EPS)
    feats["f_momentum_24_168"] = stats[24]["mean"] / (stats[168]["mean"] + EPS)
    _tag(["f_diff1", "f_abs_diff1", "f_diff1_z24", "f_delta_24", "f_delta_168",
          "f_ratio_24", "f_ratio_168", "f_momentum_6_24", "f_momentum_24_168"], "dynamics")

    # ---- family: volatility (dispersion of recent history, not of the current point)
    for w in (24, 168):
        mean, std = stats[w]["mean"], stats[w]["std"]
        feats[f"f_cv_{w}"] = std / (mean.abs() + EPS)
        feats[f"f_range_{w}"] = (stats[w]["max"] - stats[w]["min"]) / (mean.abs() + EPS)
    _tag(["f_cv_24", "f_cv_168", "f_range_24", "f_range_168"], "volatility")

    # ---- family: data availability (missingness is itself a signal of sensor/asset trouble)
    valid = v.notna()
    feats["f_is_missing"] = (~valid).astype("float64")
    feats["f_missing_24h"] = (~valid).astype("float64").rolling(24, min_periods=1).sum()
    feats["f_hours_since_valid"] = _hours_since_last_valid(v)
    _tag(["f_is_missing", "f_missing_24h", "f_hours_since_valid"], "availability")

    # ---- family: calendar
    idx = v.index
    hour = idx.hour.to_numpy()
    dow = idx.day_of_week.to_numpy()
    month = idx.month.to_numpy()
    cal_specs = (
        ("f_hour_sin", np.sin(2 * np.pi * hour / 24)),
        ("f_hour_cos", np.cos(2 * np.pi * hour / 24)),
        ("f_dow_sin", np.sin(2 * np.pi * dow / 7)),
        ("f_dow_cos", np.cos(2 * np.pi * dow / 7)),
        ("f_month", month.astype(float)),
        ("f_is_weekend", (dow >= 5).astype(float)),
    )
    for name, arr in cal_specs:
        feats[name] = pd.DataFrame(
            np.repeat(np.asarray(arr, dtype="float64")[:, None], v.shape[1], axis=1),
            index=idx, columns=v.columns,
        )
    _tag(["f_hour_sin", "f_hour_cos", "f_dow_sin", "f_dow_cos", "f_month", "f_is_weekend"],
         "calendar")

    # ---- family: peer context (contemporaneous cross-section, scale-free)
    ratio24 = feats["f_ratio_med_24"]
    peer_med = ratio24.median(axis=1, skipna=True)
    feats["f_peer_dev"] = ratio24.sub(peer_med, axis=0)
    _tag(["f_peer_dev"], "peer")

    # ---------------------------------------------------------------- wide -> long
    names = list(feats.keys())
    stacked = {n: feats[n].to_numpy(dtype="float32").ravel(order="C") for n in names}
    index = pd.MultiIndex.from_product([grid, buildings], names=[TIME_COL, GROUP_COL])
    out = pd.DataFrame(stacked, index=index)

    if return_wide:
        return out, wide

    # Emit only rows that exist in the source data.
    present = d[[TIME_COL, GROUP_COL]].drop_duplicates()
    present = present.set_index([TIME_COL, GROUP_COL]).index
    out = out.loc[out.index.intersection(present)]
    out = out.reset_index()
    return out


def _hours_since_last_valid(v: pd.DataFrame) -> pd.DataFrame:
    """Hours since the most recent non-NaN reading, per building (0 where current is valid)."""
    n = len(v)
    pos = pd.Series(np.arange(n, dtype="float64"), index=v.index)
    out = {}
    for c in v.columns:
        valid = v[c].notna()
        last = pos.where(valid).ffill()
        out[c] = (pos - last).to_numpy()
    return pd.DataFrame(out, index=v.index)


def feature_names(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f_")]


def main() -> int:
    if not LEAD_RAW_CSV.exists():
        print(f"ERROR: {LEAD_RAW_CSV} not found; run scripts/fetch_data.py")
        return 1

    raw = pd.read_csv(LEAD_RAW_CSV, parse_dates=[TIME_COL])
    print(f"raw rows: {len(raw):,}")

    X = build_feature_frame(raw)
    print(f"feature rows: {len(X):,}  feature cols: {len(feature_names(X))}")

    # Attach the label by exact (building, timestamp) join -- never by row order.
    y = raw[[GROUP_COL, TIME_COL, TARGET_COL]]
    X = X.merge(y, on=[GROUP_COL, TIME_COL], how="inner", validate="one_to_one")
    print(f"joined rows: {len(X):,}  (raw {len(raw):,})")
    if len(X) != len(raw):
        raise SystemExit("row count mismatch after label join -- investigate before proceeding")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "features.parquet"
    X.to_parquet(out_path, index=False, compression="zstd")
    print(f"wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")

    meta = {
        "n_rows": int(len(X)),
        "n_features": len(feature_names(X)),
        "features": feature_names(X),
        "families": {f: FAMILY.get(f, "unknown") for f in feature_names(X)},
        "windows_hours": {str(k): v for k, v in WINDOWS.items()},
        "nan_rates": {f: float(X[f].isna().mean()) for f in feature_names(X)},
        "grid_start": str(X[TIME_COL].min()),
        "grid_end": str(X[TIME_COL].max()),
        "source_sha_note": "see data/README.md for dataset checksums",
    }
    (METRICS_DIR / "feature_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"families: {sorted(set(FAMILY.values()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
