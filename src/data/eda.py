"""Exploratory data analysis for LEAD1.0-small.

Writes figures to results/figures/eda_*.png and a machine-readable summary to
results/metrics/eda_summary.json. All figures are regenerated from the raw CSV;
none are hand-edited.

Usage:
    python -m src.data.eda
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import (  # noqa: E402
    FIGURES_DIR, GROUP_COL, LEAD_RAW_CSV, METRICS_DIR, TARGET_COL, TIME_COL, VALUE_COL,
)

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": True,
})

C_NORMAL = "#4C72B0"
C_ANOM = "#C44E52"


def load() -> pd.DataFrame:
    df = pd.read_csv(LEAD_RAW_CSV, parse_dates=[TIME_COL])
    return df.sort_values([GROUP_COL, TIME_COL], kind="mergesort").reset_index(drop=True)


def fig_class_balance(df: pd.DataFrame, out: Path) -> dict:
    counts = df[TARGET_COL].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    bars = ax.bar(["Normal", "Anomaly"], [counts.get(0, 0), counts.get(1, 0)],
                  color=[C_NORMAL, C_ANOM])
    ax.set_yscale("log")
    ax.set_ylabel("Hourly readings (log scale)")
    ax.set_title("Class balance: 2.13% anomalous")
    for b, v in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    fig.savefig(out)
    plt.close(fig)
    return {"normal": int(counts.get(0, 0)), "anomaly": int(counts.get(1, 0))}


def fig_load_profile(df: pd.DataFrame, out: Path) -> dict:
    d = df.dropna(subset=[VALUE_COL]).copy()
    d["hour"] = d[TIME_COL].dt.hour
    d["is_weekend"] = d[TIME_COL].dt.day_of_week >= 5

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    for flag, label, color in ((False, "Weekday", C_NORMAL), (True, "Weekend", "#DD8452")):
        g = d[d["is_weekend"] == flag].groupby("hour")[VALUE_COL]
        med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
        axes[0].plot(med.index, med.values, label=label, color=color, lw=1.8)
        axes[0].fill_between(med.index, q1.values, q3.values, color=color, alpha=0.18)
    axes[0].set_xlabel("Hour of day")
    axes[0].set_ylabel("meter_reading (kWh)")
    axes[0].set_title("Median load profile with IQR")
    axes[0].legend(frameon=False)
    axes[0].set_xticks(range(0, 24, 3))

    rate = df.groupby(df[TIME_COL].dt.hour)[TARGET_COL].mean() * 100
    axes[1].bar(rate.index, rate.values, color=C_ANOM, alpha=0.85)
    axes[1].set_xlabel("Hour of day")
    axes[1].set_ylabel("Anomaly rate (%)")
    axes[1].set_title("Anomaly rate by hour of day")
    axes[1].set_xticks(range(0, 24, 3))
    fig.savefig(out)
    plt.close(fig)
    return {
        "anomaly_rate_by_hour_pct": {int(k): round(float(v), 4) for k, v in rate.items()},
        "peak_anomaly_hour": int(rate.idxmax()),
        "min_anomaly_hour": int(rate.idxmin()),
    }


def fig_week_heatmap(df: pd.DataFrame, out: Path) -> dict:
    d = df.copy()
    d["dow"] = d[TIME_COL].dt.day_of_week
    d["hour"] = d[TIME_COL].dt.hour
    pivot = d.pivot_table(index="dow", columns="hour", values=TARGET_COL, aggfunc="mean") * 100

    fig, ax = plt.subplots(figsize=(9.0, 2.6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="magma_r")
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day")
    ax.set_title("Anomaly rate (%) by day of week and hour")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="%")
    fig.savefig(out)
    plt.close(fig)
    return {"max_cell_pct": round(float(np.nanmax(pivot.values)), 3),
            "min_cell_pct": round(float(np.nanmin(pivot.values)), 3)}


def fig_building_distribution(df: pd.DataFrame, out: Path) -> dict:
    g = df.groupby(GROUP_COL).agg(n=(TARGET_COL, "size"), a=(TARGET_COL, "sum"))
    g["rate"] = g["a"] / g["n"] * 100

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    axes[0].hist(g["rate"], bins=30, color=C_NORMAL, edgecolor="white")
    axes[0].set_xlabel("Per-building anomaly rate (%)")
    axes[0].set_ylabel("Number of buildings")
    axes[0].set_title(f"Anomaly-rate heterogeneity across {len(g)} buildings")
    axes[0].axvline(g["rate"].median(), color=C_ANOM, ls="--", lw=1.2,
                    label=f"median {g['rate'].median():.2f}%")
    axes[0].legend(frameon=False)

    axes[1].scatter(g["n"], g["rate"], s=14, alpha=0.65, color=C_NORMAL)
    axes[1].set_xlabel("Readings in building")
    axes[1].set_ylabel("Anomaly rate (%)")
    axes[1].set_title("Coverage vs anomaly rate")
    fig.savefig(out)
    plt.close(fig)
    return {"median_rate_pct": round(float(g["rate"].median()), 4),
            "iqr_pct": [round(float(g["rate"].quantile(.25)), 4),
                        round(float(g["rate"].quantile(.75)), 4)]}


def fig_timeseries_examples(df: pd.DataFrame, out: Path) -> dict:
    g = df.groupby(GROUP_COL).agg(n=(TARGET_COL, "size"), a=(TARGET_COL, "sum"))
    g["rate"] = g["a"] / g["n"]
    # pick buildings with complete coverage and anomaly rate near the median
    complete = g[g["n"] == g["n"].max()]
    pick = complete.iloc[(complete["rate"] - complete["rate"].median()).abs().argsort()[:2]].index
    pick = [int(b) for b in pick]

    fig, axes = plt.subplots(len(pick), 1, figsize=(9.0, 2.6 * len(pick)), squeeze=False)
    for ax, b in zip(axes[:, 0], pick):
        d = df[df[GROUP_COL] == b]
        ax.plot(d[TIME_COL], d[VALUE_COL], color=C_NORMAL, lw=0.5, label="meter_reading")
        an = d[d[TARGET_COL] == 1]
        ax.scatter(an[TIME_COL], an[VALUE_COL], s=6, color=C_ANOM, zorder=3, label="annotated anomaly")
        ax.set_title(f"Building {b} — full year, hourly")
        ax.set_ylabel("kWh")
        ax.legend(frameon=False, loc="upper right", fontsize=7)
    fig.savefig(out)
    plt.close(fig)
    return {"example_buildings": pick}


def fig_missingness(df: pd.DataFrame, out: Path) -> dict:
    g = df.groupby(GROUP_COL).apply(
        lambda s: s[VALUE_COL].isna().mean() * 100, include_groups=False
    )
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.hist(g.values, bins=40, color="#55A868", edgecolor="white")
    ax.set_xlabel("Missing meter_reading (%)")
    ax.set_ylabel("Number of buildings")
    ax.set_title("Missingness is concentrated in a few buildings")
    fig.savefig(out)
    plt.close(fig)
    return {"buildings_over_10pct_missing": int((g > 10).sum()),
            "buildings_over_50pct_missing": int((g > 50).sum())}


def fig_autocorrelation(df: pd.DataFrame, out: Path, max_lag: int = 48) -> dict:
    lags = list(range(1, max_lag + 1))
    curves = []
    for _, d in df.groupby(GROUP_COL):
        s = d[VALUE_COL].to_numpy(dtype=float)
        s = s[~np.isnan(s)]
        if len(s) < max_lag * 4:
            continue
        s = s - s.mean()
        denom = np.dot(s, s)
        if denom <= 0:
            continue
        curves.append([np.dot(s[:-k], s[k:]) / denom for k in lags])
    curves = np.array(curves)
    med = np.median(curves, axis=0)
    lo, hi = np.percentile(curves, [10, 90], axis=0)

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot(lags, med, color=C_NORMAL, lw=1.8, label="median over buildings")
    ax.fill_between(lags, lo, hi, color=C_NORMAL, alpha=0.18, label="10–90th pct")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Autocorrelation of meter_reading")
    ax.set_title("Strong short-range autocorrelation motivates causal windows")
    ax.legend(frameon=False)
    fig.savefig(out)
    plt.close(fig)
    return {"n_buildings": int(curves.shape[0]),
            "acf_lag1_median": round(float(med[0]), 4),
            "acf_lag24_median": round(float(med[23]), 4)}


def fig_event_lengths(df: pd.DataFrame, out: Path) -> dict:
    lengths = []
    for _, d in df.groupby(GROUP_COL, sort=False):
        m = d[TARGET_COL].to_numpy().astype(bool)
        if not m.any():
            continue
        padded = np.concatenate(([False], m, [False]))
        diff = np.diff(padded.astype(np.int8))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        lengths.extend((ends - starts).tolist())
    lengths = np.array(lengths)

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    bins = np.logspace(0, np.log10(max(lengths.max(), 2)), 30)
    ax.hist(lengths, bins=bins, color=C_ANOM, alpha=0.85, edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("Event length (consecutive hours)")
    ax.set_ylabel("Number of events")
    ax.set_title(f"{len(lengths):,} anomaly events — median {np.median(lengths):.0f} h")
    fig.savefig(out)
    plt.close(fig)
    return {"n_events": int(len(lengths)), "median_h": float(np.median(lengths)),
            "p95_h": float(np.percentile(lengths, 95)), "max_h": int(lengths.max())}


def main() -> int:
    if not LEAD_RAW_CSV.exists():
        print(f"ERROR: {LEAD_RAW_CSV} not found; run scripts/fetch_data.py")
        return 1
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = load()
    print(f"loaded {len(df):,} rows, {df[GROUP_COL].nunique()} buildings")

    summary = {
        "class_balance": fig_class_balance(df, FIGURES_DIR / "eda_class_balance.png"),
        "load_profile": fig_load_profile(df, FIGURES_DIR / "eda_load_profile.png"),
        "week_heatmap": fig_week_heatmap(df, FIGURES_DIR / "eda_week_heatmap.png"),
        "building_distribution": fig_building_distribution(
            df, FIGURES_DIR / "eda_building_distribution.png"),
        "timeseries_examples": fig_timeseries_examples(
            df, FIGURES_DIR / "eda_timeseries_examples.png"),
        "missingness": fig_missingness(df, FIGURES_DIR / "eda_missingness.png"),
        "autocorrelation": fig_autocorrelation(df, FIGURES_DIR / "eda_autocorrelation.png"),
        "event_lengths": fig_event_lengths(df, FIGURES_DIR / "eda_event_lengths.png"),
    }
    (METRICS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
