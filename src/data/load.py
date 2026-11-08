"""Loading helpers shared by every modelling script.

Keeping the load/split logic in one place is what makes the numbers comparable across baselines,
XGBoost runs and the API: they all slice the matrix with the same function and the same frozen
`data/processed/splits.json`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import (  # noqa: E402
    FEATURES_PARQUET, GROUP_COL, SPLITS_JSON, TARGET_COL, TIME_COL,
)


# Features that become constant (all zeros) once `observed_only()` is applied: the flag "this hour
# has no reading" cannot be true in a frame that contains only hours with readings.
DEGENERATE_AFTER_OBSERVED = {"f_is_missing"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model inputs: everything prefixed `f_`, minus keys, label and degenerate flags."""
    cols = [
        c for c in df.columns
        if c.startswith("f_") and c not in DEGENERATE_AFTER_OBSERVED
    ]
    if not cols:
        raise ValueError("no feature columns found (expected names starting with 'f_')")
    return cols


def load_features(path: Path | None = None) -> pd.DataFrame:
    p = path or FEATURES_PARQUET
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing -- run `python -m src.features.build_features` first"
        )
    df = pd.read_parquet(p)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    return df.sort_values([GROUP_COL, TIME_COL], kind="mergesort").reset_index(drop=True)


def load_splits(path: Path | None = None) -> dict:
    p = path or SPLITS_JSON
    if not p.exists():
        raise FileNotFoundError(f"{p} missing -- run `python -m src.features.splits` first")
    return json.loads(p.read_text(encoding="utf-8"))


def split_frame(df: pd.DataFrame, splits: dict, name: str) -> pd.DataFrame:
    ids = splits[name]
    out = df[df[GROUP_COL].isin(ids)]
    if out.empty:
        raise ValueError(f"split '{name}' selected no rows")
    return out.reset_index(drop=True)


def observed_only(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose `meter_reading` is absent.

    Rationale (DECISIONS.md D-011): in LEAD1.0 an hour with no reading is *never* annotated as an
    anomaly -- 0 of 107,653 missing rows carry a positive label -- because there is nothing to
    label. Keeping those rows would (a) hand the model a deterministic "missing => normal" rule
    worth 6.15% of the data, and (b) teach exactly the wrong operational lesson, since a dead
    sensor is a fault, not a healthy asset. Outages are therefore handled as a separate alarm
    class in the API, and the model is trained and scored on observed readings only.
    """
    mask = df["f_is_missing"] == 0
    return df[mask].reset_index(drop=True)


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, "pd.Series"]:
    return df[feature_columns(df)], df[TARGET_COL].astype(int)
