"""Geothermal auxiliary-equipment simulator — **all output is SIMULATED**.

Why this exists
---------------
No PLTP Kamojang SCADA export is available (and real plant data would be confidential), so the
deployment path is demonstrated on synthetic signals with documented, physically motivated
generators and *known* fault windows. This lets the system show what an operator would see, and
lets us measure whether the detector's alarms land inside the injected faults.

What this is NOT
----------------
It is not a validated model of Kamojang, of any other geothermal plant, or of the equipment
involved. Parameters are plausible order-of-magnitude values chosen for demonstration, not fitted
to measurements. Every output row carries `simulated = True`, every file is suffixed `_SIMULATED`,
and no result derived from it may be reported as a plant measurement (D-005).

Signals (hourly, 90 days by default)
------------------------------------
pump_power_kw          cooling-water circulation pump electrical power
compressor_current_a   NCG extraction compressor motor current
bearing_temp_c         turbine bearing metal temperature
separator_level_pct    separator water level

Injected faults (ground truth written to a companion JSON)
---------------------------------------------------------
cavitation             pump: flow restriction -> power oscillation + elevated noise, 36 h
bearing_overheat       bearing: slow ramp, then a trip event, 72 h
level_drift            separator: step drift of the level controller, 24 h
compressor_fouling     compressor: gradual current rise at constant load, 120 h
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import METRICS_DIR, RANDOM_SEED, SYNTHETIC_DIR  # noqa: E402

SIM_SUFFIX = "_SIMULATED"


@dataclass
class Fault:
    equipment: str
    kind: str
    start_hour: int
    duration_h: int
    magnitude: float


@dataclass
class SimulationConfig:
    days: int = 90
    seed: int = RANDOM_SEED
    start: str = "2026-01-01"
    faults: list[Fault] = field(default_factory=lambda: [
        Fault("pump", "cavitation", 24 * 20 + 6, 36, 0.10),
        Fault("bearing", "bearing_overheat", 24 * 38, 72, 5.0),
        Fault("separator", "level_drift", 24 * 55 + 12, 24, 6.0),
        Fault("compressor", "compressor_fouling", 24 * 70, 120, 0.08),
    ])


def _load_profile(hours: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Diurnal demand shape with a mild weekday effect and smooth weather drift."""
    h = hours % 24
    daily = 1.0 + 0.18 * np.sin(2 * np.pi * (h - 9) / 24.0)
    dow = ((hours // 24) + 3) % 7
    weekly = np.where(dow >= 5, 0.94, 1.0)
    drift = 1.0 + 0.05 * np.sin(2 * np.pi * hours / (24 * 30.0))
    return daily * weekly * drift


def generate(cfg: SimulationConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (signals_long, faults) where signals_long has one row per (equipment, hour)."""
    cfg = cfg or SimulationConfig()
    rng = np.random.default_rng(cfg.seed)

    n = cfg.days * 24
    hours = np.arange(n, dtype=float)
    ts = pd.date_range(cfg.start, periods=n, freq="h")
    profile = _load_profile(hours, rng)

    # ---------------------------------------------------------------- base signals
    pump = 420.0 * profile * (1.0 + rng.normal(0, 0.012, n))
    comp = 118.0 * profile * (1.0 + rng.normal(0, 0.015, n))
    temp = 61.0 + 5.5 * (profile - 0.95) + rng.normal(0, 0.35, n)
    level = 50.0 + 1.8 * np.sin(2 * np.pi * hours / 24.0 + 0.7) + rng.normal(0, 0.5, n)

    truth = pd.DataFrame({
        "timestamp": ts,
        "pump_power_kw": pump,
        "compressor_current_a": comp,
        "bearing_temp_c": temp,
        "separator_level_pct": level,
    })

    # ---------------------------------------------------------------- inject faults
    fault_rows = []
    for f in cfg.faults:
        s, e = f.start_hour, min(f.start_hour + f.duration_h, n)
        span = np.arange(s, e)
        frac = (span - s) / max(len(span) - 1, 1)
        col = {"pump": "pump_power_kw", "bearing": "bearing_temp_c",
               "separator": "separator_level_pct", "compressor": "compressor_current_a"}[f.equipment]
        base = truth[col].to_numpy().copy()

        if f.kind == "cavitation":
            osc = 1.0 + f.magnitude * np.sin(2 * np.pi * span / 3.0) * (0.4 + 0.6 * frac)
            base[s:e] *= osc
            base[s:e] += rng.normal(0, 420.0 * 0.035, len(span))  # extra turbulence
        elif f.kind == "bearing_overheat":
            base[s:e] += f.magnitude * frac ** 1.6
            base[e - 1] += 1.5  # trip spike
        elif f.kind == "level_drift":
            base[s:e] += f.magnitude * np.minimum(frac * 2.0, 1.0)
        elif f.kind == "compressor_fouling":
            base[s:e] *= 1.0 + f.magnitude * frac
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown fault kind: {f.kind}")

        truth[col] = base
        fault_rows.append({
            "equipment": f.equipment, "kind": f.kind,
            "start": str(ts[s]), "end": str(ts[e - 1]),
            "start_hour": int(s), "duration_h": int(len(span)),
            "magnitude": f.magnitude, "column": col, "simulated": True,
        })

    # ---------------------------------------------------------------- long format
    long = truth.melt(id_vars="timestamp", var_name="signal", value_name="value")
    long["equipment"] = long["signal"].map({
        "pump_power_kw": "pump",
        "compressor_current_a": "compressor",
        "bearing_temp_c": "bearing",
        "separator_level_pct": "separator",
    })
    long["unit"] = long["signal"].map({
        "pump_power_kw": "kW",
        "compressor_current_a": "A",
        "bearing_temp_c": "degC",
        "separator_level_pct": "%",
    })
    long["simulated"] = True
    long = long[["timestamp", "equipment", "signal", "value", "unit", "simulated"]]

    faults = pd.DataFrame(fault_rows)
    return long, faults


def write(cfg: SimulationConfig | None = None) -> dict:
    long, faults = generate(cfg)
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    sig_path = SYNTHETIC_DIR / f"geothermal_signals{SIM_SUFFIX}.csv"
    long.to_csv(sig_path, index=False)

    truth_path = SYNTHETIC_DIR / f"geothermal_faults{SIM_SUFFIX}.json"
    payload = {
        "simulated": True,
        "warning": ("SYNTHETIC DATA. Not PLTP Kamojang measurements. Generators use plausible "
                    "parameter values for demonstration only; nothing here is fitted to a real "
                    "plant."),
        "config": {
            "days": (cfg or SimulationConfig()).days,
            "seed": (cfg or SimulationConfig()).seed,
            "start": (cfg or SimulationConfig()).start,
        },
        "faults": faults.to_dict("records"),
    }
    truth_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = {
        "rows": int(len(long)),
        "equipment": sorted(long["equipment"].unique().tolist()),
        "signals": sorted(long["signal"].unique().tolist()),
        "faults": int(len(faults)),
        "hours": int(long["timestamp"].nunique()),
        "signals_path": str(sig_path.relative_to(sig_path.parents[2])),
        "truth_path": str(truth_path.relative_to(truth_path.parents[2])),
        "simulated": True,
    }
    (METRICS_DIR / f"simulation_summary{SIM_SUFFIX}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    s = write()
    print("SIMULATED geothermal data generated (not plant measurements)")
    print(json.dumps(s, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
