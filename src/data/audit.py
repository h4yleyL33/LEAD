"""Data audit for the LEAD1.0-small dataset.

Produces a machine-readable audit report (results/metrics/data_audit.json) and a
human-readable summary (results/tables/data_audit_summary.md).

Every number quoted in PROJECT_PLAN.md, PROGRESS.md and the IEEE manuscript must
come from this report or from a later experiment artifact -- never from memory.

Usage:
    python -m src.data.audit
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

from src.utils.config import (  # noqa: E402
    GROUP_COL, LEAD_RAW_CSV, METRICS_DIR, TABLES_DIR, TARGET_COL, TIME_COL,
    VALUE_COL, environment_info,
)


def _runs(mask: np.ndarray) -> np.ndarray:
    """Lengths of consecutive True runs in a boolean array."""
    if not mask.any():
        return np.array([], dtype=int)
    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return ends - starts


def audit(csv_path=LEAD_RAW_CSV) -> dict:
    df = pd.read_csv(csv_path, parse_dates=[TIME_COL])
    df = df.sort_values([GROUP_COL, TIME_COL], kind="mergesort").reset_index(drop=True)

    n_rows = int(len(df))
    n_buildings = int(df[GROUP_COL].nunique())
    ts_min, ts_max = df[TIME_COL].min(), df[TIME_COL].max()

    per_building = df.groupby(GROUP_COL).agg(
        n_readings=(VALUE_COL, "size"),
        n_missing=(VALUE_COL, lambda s: int(s.isna().sum())),
        n_anomaly=(TARGET_COL, "sum"),
        n_unique_ts=(TIME_COL, "nunique"),
        ts_min=(TIME_COL, "min"),
        ts_max=(TIME_COL, "max"),
    )
    per_building["anomaly_rate"] = per_building["n_anomaly"] / per_building["n_readings"]

    missing = int(df[VALUE_COL].isna().sum())
    non_null = df[VALUE_COL].dropna()
    dup_mask = df.duplicated(subset=[GROUP_COL, TIME_COL], keep=False)

    # Expected hourly grid per building (only for buildings fully covering the window)
    grid_min, grid_max = ts_min, ts_max
    expected_per_building = int((grid_max - grid_min) / pd.Timedelta(hours=1)) + 1

    # anomaly run-length structure (per building, contiguous hourly blocks)
    run_lengths = []
    for _, g in df.groupby(GROUP_COL, sort=False):
        run_lengths.append(_runs(g[TARGET_COL].to_numpy().astype(bool)))
    run_lengths = np.concatenate(run_lengths) if run_lengths else np.array([], dtype=int)

    hourly = df[TIME_COL].dt.hour
    dow = df[TIME_COL].dt.day_of_week
    monthly = df[TIME_COL].dt.to_period("M").astype(str)

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_file": str(csv_path),
        "environment": environment_info(),
        "schema": {
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        },
        "volume": {
            "n_rows": n_rows,
            "n_buildings": n_buildings,
            "n_unique_timestamps": int(df[TIME_COL].nunique()),
            "timestamp_min": str(ts_min),
            "timestamp_max": str(ts_max),
            "span_days": float((ts_max - ts_min) / pd.Timedelta(days=1)),
            "expected_hourly_rows_per_full_building": expected_per_building,
            "rows_per_building": {
                "min": int(per_building["n_readings"].min()),
                "median": float(per_building["n_readings"].median()),
                "max": int(per_building["n_readings"].max()),
                "mean": float(per_building["n_readings"].mean()),
            },
            "buildings_with_full_grid": int((per_building["n_readings"] == expected_per_building).sum()),
        },
        "quality": {
            "duplicate_building_timestamp_rows": int(dup_mask.sum()),
            "missing_meter_reading": missing,
            "missing_meter_reading_rate": float(missing / n_rows),
            "missing_meter_reading_by_building_max_rate": float(
                (per_building["n_missing"] / per_building["n_readings"]).max()
            ),
            "zero_readings": int((non_null == 0).sum()),
            "negative_readings": int((non_null < 0).sum()),
            "meter_reading_stats": {
                "min": float(non_null.min()),
                "p01": float(non_null.quantile(0.01)),
                "p25": float(non_null.quantile(0.25)),
                "median": float(non_null.median()),
                "mean": float(non_null.mean()),
                "p75": float(non_null.quantile(0.75)),
                "p99": float(non_null.quantile(0.99)),
                "max": float(non_null.max()),
                "skew": float(non_null.skew()),
            },
        },
        "target": {
            "n_anomaly": int(df[TARGET_COL].sum()),
            "anomaly_rate": float(df[TARGET_COL].mean()),
            "n_normal": int((df[TARGET_COL] == 0).sum()),
            "class_imbalance_ratio": float((df[TARGET_COL] == 0).sum() / max(df[TARGET_COL].sum(), 1)),
            "buildings_with_zero_anomalies": int((per_building["n_anomaly"] == 0).sum()),
            "buildings_with_all_anomalies": int((per_building["anomaly_rate"] == 1.0).sum()),
            "per_building_anomaly_rate": {
                "min": float(per_building["anomaly_rate"].min()),
                "p25": float(per_building["anomaly_rate"].quantile(0.25)),
                "median": float(per_building["anomaly_rate"].median()),
                "mean": float(per_building["anomaly_rate"].mean()),
                "p75": float(per_building["anomaly_rate"].quantile(0.75)),
                "max": float(per_building["anomaly_rate"].max()),
            },
            "anomaly_rate_by_hour": {
                int(h): float(v) for h, v in df.groupby(hourly)[TARGET_COL].mean().items()
            },
            "anomaly_rate_by_dayofweek": {
                int(d): float(v) for d, v in df.groupby(dow)[TARGET_COL].mean().items()
            },
            "anomaly_rate_by_month": {
                str(m): float(v) for m, v in df.groupby(monthly)[TARGET_COL].mean().items()
            },
            "anomaly_event_runs": {
                "n_events": int(len(run_lengths)),
                "mean_length_h": float(run_lengths.mean()) if len(run_lengths) else 0.0,
                "median_length_h": float(np.median(run_lengths)) if len(run_lengths) else 0.0,
                "p95_length_h": float(np.percentile(run_lengths, 95)) if len(run_lengths) else 0.0,
                "max_length_h": int(run_lengths.max()) if len(run_lengths) else 0,
                "share_singleton_events": float((run_lengths == 1).mean()) if len(run_lengths) else 0.0,
            },
        },
        "per_building": {
            str(int(b)): {
                "n_readings": int(r.n_readings),
                "n_missing": int(r.n_missing),
                "n_anomaly": int(r.n_anomaly),
                "anomaly_rate": float(r.anomaly_rate),
                "ts_min": str(r.ts_min),
                "ts_max": str(r.ts_max),
            }
            for b, r in per_building.iterrows()
        },
    }
    return report, per_building


def main() -> int:
    if not LEAD_RAW_CSV.exists():
        print(f"ERROR: dataset not found at {LEAD_RAW_CSV}")
        return 1
    report, per_building = audit()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    out_json = METRICS_DIR / "data_audit.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    per_building.sort_values("anomaly_rate", ascending=False).to_csv(
        TABLES_DIR / "per_building_audit.csv"
    )

    v, q, t = report["volume"], report["quality"], report["target"]
    md = f"""# Data Audit — LEAD1.0-small

Generated: {report['generated_utc']}
Source: `{report['source_file']}`

## Volume
| Metric | Value |
|---|---|
| Rows | {v['n_rows']:,} |
| Buildings | {v['n_buildings']} |
| Unique timestamps | {v['n_unique_timestamps']:,} |
| Time range | {v['timestamp_min']} → {v['timestamp_max']} |
| Span (days) | {v['span_days']:.1f} |
| Readings/building (min/median/max) | {v['rows_per_building']['min']:,} / {v['rows_per_building']['median']:.0f} / {v['rows_per_building']['max']:,} |
| Buildings with complete hourly grid | {v['buildings_with_full_grid']} |

## Quality
| Metric | Value |
|---|---|
| Duplicate (building, timestamp) rows | {q['duplicate_building_timestamp_rows']:,} |
| Missing `meter_reading` | {q['missing_meter_reading']:,} ({q['missing_meter_reading_rate']*100:.2f}%) |
| Max missing rate in one building | {q['missing_meter_reading_by_building_max_rate']*100:.2f}% |
| Zero readings | {q['zero_readings']:,} |
| Negative readings | {q['negative_readings']:,} |
| Median / mean kWh | {q['meter_reading_stats']['median']:.2f} / {q['meter_reading_stats']['mean']:.2f} |
| p99 / max kWh | {q['meter_reading_stats']['p99']:.2f} / {q['meter_reading_stats']['max']:.2f} |
| Skewness | {q['meter_reading_stats']['skew']:.2f} |

## Target
| Metric | Value |
|---|---|
| Anomalies | {t['n_anomaly']:,} |
| Anomaly rate | {t['anomaly_rate']*100:.2f}% |
| Imbalance ratio (normal:anomaly) | {t['class_imbalance_ratio']:.2f}:1 |
| Buildings with no anomalies | {t['buildings_with_zero_anomalies']} |
| Buildings that are 100% anomalous | {t['buildings_with_all_anomalies']} |
| Per-building anomaly rate (min/median/max) | {t['per_building_anomaly_rate']['min']*100:.2f}% / {t['per_building_anomaly_rate']['median']*100:.2f}% / {t['per_building_anomaly_rate']['max']*100:.2f}% |
| Anomaly events (contiguous runs) | {t['anomaly_event_runs']['n_events']:,} |
| Median event length | {t['anomaly_event_runs']['median_length_h']:.1f} h |
| Max event length | {t['anomaly_event_runs']['max_length_h']} h |
| Singleton events | {t['anomaly_event_runs']['share_singleton_events']*100:.1f}% |
"""
    (TABLES_DIR / "data_audit_summary.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
