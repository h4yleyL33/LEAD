"""Leakage tests for the causal feature contract.

The central test is an *empirical truncation test*: build features on the full history, then
rebuild them on history truncated at time T. Every feature value at time <= T must be bit-identical
between the two builds. If any feature peeks at the future, the truncated build will differ.

This catches mistakes that code review misses: centred windows, negative shifts, `bfill`,
whole-series statistics, and accidental use of the full-year median as a normaliser.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import build_feature_frame, feature_names

RNG = np.random.default_rng(7)
N_BUILDINGS = 4
N_HOURS = 24 * 21  # three weeks: long enough to exercise the 168 h window
SEED = 7

# Features that are pure functions of history *strictly before* t (no current value at all).
PAST_ONLY_FEATURES = [
    "f_cv_24", "f_cv_168", "f_range_24", "f_range_168",
    "f_momentum_6_24", "f_momentum_24_168",
]
# Ratio features: value = current / baseline. The baseline must be past-only.
RATIO_FEATURES = ["f_ratio_med_6", "f_ratio_med_24", "f_ratio_med_168", "f_ratio_med_720"]


def _synthetic_long() -> pd.DataFrame:
    """Deterministic synthetic telemetry with gaps and NaNs (shape only, no real data needed).

    A fresh RNG is created on every call so that repeated calls return identical frames -- a
    shared module-level RNG would silently make the truncation test compare different data.
    """
    rng = np.random.default_rng(SEED)
    start = pd.Timestamp("2020-01-01")
    grid = pd.date_range(start, periods=N_HOURS, freq="h")
    frames = []
    for b in range(N_BUILDINGS):
        base = 50.0 + 20.0 * b
        series = base + 10 * np.sin(np.arange(N_HOURS) / 24 * 2 * np.pi) + rng.normal(0, 2, N_HOURS)
        series[rng.random(N_HOURS) < 0.03] = np.nan  # sensor dropouts
        d = pd.DataFrame({"building_id": b, "timestamp": grid, "meter_reading": series})
        d = d.drop(index=d.index[100 + b * 5: 103 + b * 5])  # absent rows, like the real data
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def full_features() -> pd.DataFrame:
    return build_feature_frame(_synthetic_long())


def test_no_feature_uses_future_information(full_features: pd.DataFrame) -> None:
    """Truncating history at T must not change any feature value at time <= T."""
    df = _synthetic_long()
    cut = pd.Timestamp("2020-01-01") + pd.Timedelta(hours=N_HOURS - 24)  # keep 24 h of headroom

    truncated = build_feature_frame(
        df[df["timestamp"] <= cut],
        grid_start=df["timestamp"].min(),
        grid_end=cut,
    )

    cols = feature_names(full_features)
    left = full_features[full_features["timestamp"] <= cut].set_index(["building_id", "timestamp"])
    right = truncated.set_index(["building_id", "timestamp"])
    common = left.index.intersection(right.index)
    assert len(common) == len(left) == len(right)

    a = left.loc[common, cols].to_numpy(dtype="float64")
    b = right.loc[common, cols].to_numpy(dtype="float64")
    same_nan = np.isnan(a) == np.isnan(b)
    close = np.isclose(a, b, rtol=0, atol=1e-5, equal_nan=True)

    bad = ~(same_nan & close)
    if bad.any():
        offenders = sorted({cols[j] for j in np.where(bad)[1]})
        raise AssertionError(f"features changed when future rows were removed: {offenders}")


def test_building_id_is_not_a_feature(full_features: pd.DataFrame) -> None:
    assert "building_id" not in feature_names(full_features)
    assert "anomaly" not in feature_names(full_features)


def _pick_valid_time(df: pd.DataFrame, building: int, after_hour: int) -> pd.Timestamp:
    """First timestamp after `after_hour` where `building` has a real (non-NaN) reading."""
    sub = df[(df["building_id"] == building) & (df["timestamp"] >= df["timestamp"].min()
                                                 + pd.Timedelta(hours=after_hour))]
    sub = sub[sub["meter_reading"].notna()]
    assert len(sub) > 0
    return sub["timestamp"].iloc[0]


def test_baseline_excludes_current_reading() -> None:
    """Altering only the reading at t must leave every past-only baseline untouched."""
    df = _synthetic_long()
    b = 0
    t = _pick_valid_time(df, b, 200)

    base = build_feature_frame(df)
    mutated = df.copy()
    mask = (mutated["building_id"] == b) & (mutated["timestamp"] == t)
    assert mask.sum() == 1
    mutated.loc[mask, "meter_reading"] = 1e6  # absurd spike at exactly t

    alt = build_feature_frame(mutated)
    idx = pd.MultiIndex.from_tuples([(b, t)], names=["building_id", "timestamp"])
    before = base.set_index(["building_id", "timestamp"]).loc[idx]
    after = alt.set_index(["building_id", "timestamp"]).loc[idx]

    for col in PAST_ONLY_FEATURES:
        assert np.isclose(before[col].iloc[0], after[col].iloc[0], equal_nan=True, atol=1e-5), (
            f"{col} changed when only the current reading changed -> its baseline is not past-only"
        )

    # The current point itself must still react (otherwise the feature set is blind to the reading).
    assert not np.isclose(before["f_log_value"].iloc[0], after["f_log_value"].iloc[0])


def test_ratio_denominator_is_past_only() -> None:
    """For value/baseline features, baseline = value/ratio must not move with the current point."""
    df = _synthetic_long()
    b = 1
    t = _pick_valid_time(df, b, 300)

    base = build_feature_frame(df)
    mutated = df.copy()
    mask = (mutated["building_id"] == b) & (mutated["timestamp"] == t)
    mutated.loc[mask, "meter_reading"] = 1e6
    alt = build_feature_frame(mutated)

    idx = pd.MultiIndex.from_tuples([(b, t)], names=["building_id", "timestamp"])
    before = base.set_index(["building_id", "timestamp"]).loc[idx]
    after = alt.set_index(["building_id", "timestamp"]).loc[idx]

    v_before = float(np.expm1(before["f_log_value"].iloc[0]))
    v_after = float(np.expm1(after["f_log_value"].iloc[0]))
    for col in RATIO_FEATURES:
        d0 = v_before / float(before[col].iloc[0])
        d1 = v_after / float(after[col].iloc[0])
        assert np.isclose(d0, d1, rtol=1e-4), f"{col}: denominator moved with the current reading"


def test_split_is_building_disjoint() -> None:
    from src.features.splits import make_splits

    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "building_id": np.repeat(np.arange(200), 50),
        "anomaly": (rng.random(200 * 50) < 0.02).astype(int),
    })
    s = make_splits(df)
    assert len(s["train"]) == 140 and len(s["val"]) == 30 and len(s["test"]) == 30
    assert not set(s["train"]) & set(s["val"])
    assert not set(s["train"]) & set(s["test"])
    assert not set(s["val"]) & set(s["test"])
    assert len(set(s["train"]) | set(s["val"]) | set(s["test"])) == 200
    assert all(len(f) > 0 for f in s["folds"])
    assert len({b for f in s["folds"] for b in f}) == 140


def test_splits_are_deterministic() -> None:
    from src.features.splits import make_splits

    df = pd.DataFrame({
        "building_id": np.repeat(np.arange(200), 10),
        "anomaly": (np.arange(2000) % 37 == 0).astype(int),
    })
    a, b = make_splits(df), make_splits(df)
    assert a["train"] == b["train"] and a["test"] == b["test"] and a["val"] == b["val"]
