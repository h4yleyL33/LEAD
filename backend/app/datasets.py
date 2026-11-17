"""Storage for imported timeline datasets.

An imported dataset is kept as the original uploaded CSV plus a scored parquet next to it, so the
dashboard can list, open, score and delete datasets without touching the training artefacts under
`data/processed/` or the frozen splits.

Nothing here is model state. Deleting a dataset never invalidates the scorer.
"""
from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.utils.config import GROUP_COL, PROJECT_ROOT, TIME_COL, VALUE_COL

DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
META_FILE = DATASETS_DIR / "index.json"

# Column aliases accepted on import, mapped to the canonical schema.
ALIASES: dict[str, str] = {
    "building_id": GROUP_COL,
    "building": GROUP_COL,
    "asset_id": GROUP_COL,
    "asset": GROUP_COL,
    "meter_id": GROUP_COL,
    "timestamp": TIME_COL,
    "time": TIME_COL,
    "datetime": TIME_COL,
    "date": TIME_COL,
    "ts": TIME_COL,
    "meter_reading": VALUE_COL,
    "reading": VALUE_COL,
    "value": VALUE_COL,
    "kwh": VALUE_COL,
    "energy": VALUE_COL,
    "consumption": VALUE_COL,
}

REQUIRED = (GROUP_COL, TIME_COL, VALUE_COL)


def _norm(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


@dataclass
class Dataset:
    dataset_id: str
    name: str
    created_at: float
    n_rows: int
    n_buildings: int
    start: str | None = None
    end: str | None = None
    score: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "created_at": self.created_at,
            "n_rows": self.n_rows,
            "n_buildings": self.n_buildings,
            "start": self.start,
            "end": self.end,
            "score": self.score,
        }


def _load_index() -> dict[str, dict]:
    if not META_FILE.exists():
        return {}
    try:
        return json.loads(META_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_index(idx: dict[str, dict]) -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    META_FILE.write_text(json.dumps(idx, indent=2), encoding="utf-8")


def list_datasets() -> list[dict]:
    """Newest first."""
    idx = _load_index()
    return sorted(idx.values(), key=lambda d: d.get("created_at", 0), reverse=True)


def get_meta(dataset_id: str) -> dict | None:
    return _load_index().get(dataset_id)


def dataset_dir(dataset_id: str) -> Path:
    return DATASETS_DIR / dataset_id


def raw_path(dataset_id: str) -> Path:
    return dataset_dir(dataset_id) / "raw.csv"


def scored_path(dataset_id: str) -> Path:
    return dataset_dir(dataset_id) / "scored.parquet"


def read_raw(dataset_id: str) -> pd.DataFrame:
    p = raw_path(dataset_id)
    if not p.exists():
        raise FileNotFoundError(f"dataset {dataset_id} has no stored file")
    df = pd.read_csv(p)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    return df


def read_scored(dataset_id: str) -> pd.DataFrame | None:
    p = scored_path(dataset_id)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    return df.sort_values([GROUP_COL, TIME_COL]).reset_index(drop=True)


def canonicalize(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rename alias columns onto [building_id, timestamp, meter_reading].

    Returns the canonical frame plus a report of what was mapped. Raises ValueError when a required
    column cannot be resolved -- better to say so than to guess a column silently.
    """
    rename: dict[str, str] = {}
    for col in df.columns:
        target = ALIASES.get(_norm(col))
        if target and target not in rename.values():
            rename[col] = target
    out = df.rename(columns=rename)

    missing = [c for c in REQUIRED if c not in out.columns]
    if missing:
        raise ValueError(
            f"could not find required column(s) {missing}; "
            f"accepted aliases for time/value: "
            f"{sorted(a for a, t in ALIASES.items() if t in (TIME_COL, VALUE_COL))}"
        )

    out[TIME_COL] = pd.to_datetime(out[TIME_COL], errors="coerce")
    out[GROUP_COL] = pd.to_numeric(out[GROUP_COL], errors="coerce")
    out[VALUE_COL] = pd.to_numeric(out[VALUE_COL], errors="coerce")

    bad_ts = int(out[TIME_COL].isna().sum())
    out = out.dropna(subset=[TIME_COL])
    out[GROUP_COL] = out[GROUP_COL].fillna(0).astype("int64")
    out = out.sort_values([GROUP_COL, TIME_COL]).drop_duplicates(
        subset=[GROUP_COL, TIME_COL], keep="last"
    )

    report = {
        "mapped_columns": rename,
        "dropped_unparseable_timestamps": bad_ts,
        "duplicates_removed": int(len(df) - len(out) - bad_ts) if bad_ts else None,
        "n_rows": int(len(out)),
        "n_buildings": int(out[GROUP_COL].nunique()),
    }
    if out.empty:
        raise ValueError("no usable rows after parsing timestamp/building_id/meter_reading")
    return out.reset_index(drop=True), report


def create_dataset(name: str, df: pd.DataFrame, report: dict) -> Dataset:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    ds_id = uuid.uuid4().hex[:12]
    d = dataset_dir(ds_id)
    d.mkdir(parents=True, exist_ok=True)

    df.to_csv(raw_path(ds_id), index=False)
    ds = Dataset(
        dataset_id=ds_id,
        name=str(name)[:120],
        created_at=time.time(),
        n_rows=int(len(df)),
        n_buildings=int(df[GROUP_COL].nunique()),
        start=df[TIME_COL].min().strftime("%Y-%m-%dT%H:%M"),
        end=df[TIME_COL].max().strftime("%Y-%m-%dT%H:%M"),
    )
    idx = _load_index()
    idx[ds_id] = {**ds.to_dict(), "import": report}
    _save_index(idx)
    return ds


def save_scored(dataset_id: str, scored: pd.DataFrame, score_summary: dict) -> None:
    d = dataset_dir(dataset_id)
    if not d.exists():
        raise FileNotFoundError(f"unknown dataset {dataset_id}")
    scored.to_parquet(scored_path(dataset_id), index=False, compression="zstd")
    idx = _load_index()
    if dataset_id in idx:
        idx[dataset_id]["score"] = score_summary
        _save_index(idx)


def delete_dataset(dataset_id: str) -> bool:
    idx = _load_index()
    if dataset_id not in idx:
        return False
    d = dataset_dir(dataset_id)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    idx.pop(dataset_id, None)
    _save_index(idx)
    return True


def rename_dataset(dataset_id: str, name: str) -> dict | None:
    idx = _load_index()
    if dataset_id not in idx:
        return None
    idx[dataset_id]["name"] = str(name)[:120]
    _save_index(idx)
    return idx[dataset_id]


# --------------------------------------------------------------------------- scoring
def score_frame_long(scorer, df: pd.DataFrame) -> pd.DataFrame:
    """Score every row of a long frame, one building at a time.

    Each building is scored from its own history through `build_feature_frame`, which is the same
    causal builder used in training and by the API, so an imported timeline is scored exactly like a
    live one. Rows are emitted only where the source has a row.
    """
    import numpy as np

    from src.data.load import feature_columns
    from src.features.build_features import build_feature_frame

    cols = scorer.feature_order or None
    out = []
    for bid, sub in df.groupby(GROUP_COL):
        sub = sub[[GROUP_COL, TIME_COL, VALUE_COL]].sort_values(TIME_COL)
        feats = build_feature_frame(sub)
        if feats.empty:
            continue
        X = feats.reindex(columns=cols) if cols else feats[feature_columns(feats)]
        proba = np.asarray(scorer.model.predict_proba(X)[:, 1], dtype="float64")
        raw = (
            sub.set_index(TIME_COL)[VALUE_COL].reindex(feats[TIME_COL]).to_numpy()
        )
        missing = pd.isna(raw)
        thr = scorer.threshold
        out.append(
            pd.DataFrame(
                {
                    GROUP_COL: int(bid),
                    TIME_COL: feats[TIME_COL].to_numpy(),
                    VALUE_COL: raw,
                    "score": np.where(missing, np.nan, proba),
                    "flag": (~missing) & (proba >= thr),
                    "alarm": np.where(
                        missing, "OUTAGE", np.where(proba >= thr, "ANOMALY", "NORMAL")
                    ),
                }
            )
        )
    if not out:
        return pd.DataFrame(
            {GROUP_COL: [], TIME_COL: [], VALUE_COL: [], "score": [], "flag": [], "alarm": []}
        )
    return pd.concat(out, ignore_index=True).sort_values([GROUP_COL, TIME_COL]).reset_index(drop=True)


def summarize_scored(scored: pd.DataFrame, threshold: float) -> dict:
    """Aggregate a scored long frame into the counters the dashboard shows."""
    if scored.empty:
        return {
            "n_rows": 0,
            "n_buildings": 0,
            "n_anomaly": 0,
            "n_outage": 0,
            "n_normal": 0,
            "alert_rate": 0.0,
            "buildings": [],
        }
    g = scored.groupby(GROUP_COL)
    per_building = [
        {
            "building_id": int(b),
            "n_rows": int(len(sub)),
            "n_anomaly": int((sub["alarm"] == "ANOMALY").sum()),
            "n_outage": int((sub["alarm"] == "OUTAGE").sum()),
            "max_score": None
            if sub["score"].dropna().empty
            else round(float(sub["score"].max()), 5),
        }
        for b, sub in g
    ]
    n = len(scored)
    n_anom = int((scored["alarm"] == "ANOMALY").sum())
    return {
        "n_rows": int(n),
        "n_buildings": int(scored[GROUP_COL].nunique()),
        "n_anomaly": n_anom,
        "n_outage": int((scored["alarm"] == "OUTAGE").sum()),
        "n_normal": int((scored["alarm"] == "NORMAL").sum()),
        "alert_rate": round(n_anom / n, 6) if n else 0.0,
        "threshold": round(float(threshold), 6),
        "buildings": sorted(per_building, key=lambda x: -x["n_anomaly"]),
    }