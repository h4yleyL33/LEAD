"""In-memory live buffer: real-time routing of incoming readings to the scorer.

The model needs history to score (24 h minimum, 168 h for the long-window features), so "realtime"
here means: keep a bounded ring buffer of recent hours per building, append new readings to it, and
score the buffer through the *same* `AnomalyScorer` path used by `/score` and `/batch`.

This is deliberately process-local and ephemeral: nothing written here is training data, nothing here
survives a restart, and a restart is the correct behaviour for a live tail. Historical timelines are
imported through `backend.app.datasets` instead.

Causality is preserved because the feature builder only ever looks backwards, so appending a new hour
cannot change the score of an earlier hour.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.load import feature_columns
from src.features.build_features import build_feature_frame
from src.inference.service import AnomalyScorer
from src.utils.config import GROUP_COL, TIME_COL, VALUE_COL

# Hours of history kept per building. 2200 covers the 720 h feature window with headroom.
BUFFER_HOURS = 2200
# Rows returned per poll at most, so a big catch-up cannot flood the dashboard.
DEFAULT_LIMIT = 500


@dataclass
class BuildingBuffer:
    points: deque[tuple[pd.Timestamp, float]]
    updated_at: float

    def frame(self) -> pd.DataFrame:
        if not self.points:
            return pd.DataFrame({TIME_COL: [], VALUE_COL: []})
        df = pd.DataFrame(list(self.points), columns=[TIME_COL, VALUE_COL])
        df = df.sort_values(TIME_COL).drop_duplicates(subset=[TIME_COL], keep="last")
        if len(df) > BUFFER_HOURS:
            df = df.tail(BUFFER_HOURS)
        return df.reset_index(drop=True)


class LiveHub:
    """Thread-safe ring buffers keyed by building_id."""

    def __init__(self, scorer: AnomalyScorer):
        self.scorer = scorer
        self._bufs: dict[int, BuildingBuffer] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ ingest
    def ingest(self, records: list[dict]) -> dict:
        accepted = 0
        rejected = 0
        touched: set[int] = set()
        now = time.time()
        with self._lock:
            for r in records:
                bid = r.get(GROUP_COL)
                ts = pd.to_datetime(r.get(TIME_COL), errors="coerce")
                val = pd.to_numeric(pd.Series([r.get(VALUE_COL)]), errors="coerce").iloc[0]
                if bid is None or pd.isna(ts):
                    rejected += 1
                    continue
                bid = int(bid)
                buf = self._bufs.get(bid)
                if buf is None:
                    buf = BuildingBuffer(points=deque(), updated_at=now)
                    self._bufs[bid] = buf
                # A null value is a legitimate reading: it is the OUTAGE case.
                buf.points.append((ts, float("nan") if pd.isna(val) else float(val)))
                buf.updated_at = now
                touched.add(bid)
                accepted += 1
        return {
            "accepted": accepted,
            "rejected": rejected,
            "buildings": len(touched),
            "n_buildings_tracked": len(self._bufs),
        }

    def seed(self, history: pd.DataFrame) -> int:
        """Preload history (building_id, timestamp, meter_reading); used by /live/{id}/seed."""
        n = 0
        with self._lock:
            for bid, sub in history.groupby(GROUP_COL):
                buf = self._bufs.setdefault(
                    int(bid), BuildingBuffer(points=deque(), updated_at=time.time())
                )
                for ts, val in zip(sub[TIME_COL], sub[VALUE_COL]):
                    buf.points.append((pd.Timestamp(ts), float("nan") if pd.isna(val) else float(val)))
                    n += 1
        return n

    # ------------------------------------------------------------------ read
    def status(self) -> dict:
        with self._lock:
            bufs = {b: buf for b, buf in self._bufs.items()}
        return {
            "n_buildings_tracked": len(bufs),
            "n_points": int(sum(len(b.points) for b in bufs.values())),
            "buildings": [
                {
                    "building_id": b,
                    "n_points": len(buf.points),
                    "last": str(buf.points[-1][0]) if buf.points else None,
                    "updated_at": buf.updated_at,
                }
                for b, buf in sorted(bufs.items())
            ],
        }

    def snapshot(
        self, building_id: int, since: str | None = None, limit: int = DEFAULT_LIMIT
    ) -> dict:
        """Return scored points for one building, newest last.

        `since` is an ISO timestamp; only points at or after it are returned, which is what makes
        polling cheap -- the client remembers the last timestamp it saw and asks for the delta.
        """
        with self._lock:
            buf = self._bufs.get(building_id)
            if buf is None:
                return {
                    "building_id": building_id,
                    "threshold": self.scorer.threshold,
                    "n_points": 0,
                    "latest": None,
                    "points": [],
                }
            frame = buf.frame()
            updated_at = buf.updated_at

        if frame.empty:
            return {
                "building_id": building_id,
                "threshold": self.scorer.threshold,
                "n_points": 0,
                "latest": None,
                "points": [],
            }

        since_ts = pd.to_datetime(since, errors="coerce") if since else None
        scored = self._score_frame(frame, building_id)
        latest = scored[TIME_COL].max()

        out = scored
        if since_ts is not None and not pd.isna(since_ts):
            out = out[out[TIME_COL] >= since_ts]
        if limit and len(out) > limit:
            out = out.tail(limit)

        return {
            "building_id": building_id,
            "threshold": self.scorer.threshold,
            "model": self.scorer.model_tag,
            "n_points": int(len(scored)),
            "n_returned": int(len(out)),
            "latest": latest.strftime("%Y-%m-%dT%H:%M") if pd.notna(latest) else None,
            "updated_at": updated_at,
            "points": [
                {
                    "t": r[TIME_COL].strftime("%Y-%m-%dT%H:%M"),
                    "v": None if pd.isna(r[VALUE_COL]) else round(float(r[VALUE_COL]), 3),
                    "score": None if pd.isna(r["score"]) else round(float(r["score"]), 5),
                    "alarm": r["alarm"],
                    "flag": int(r["flag"]),
                }
                for r in out.to_dict("records")
            ],
        }

    def _score_frame(self, frame: pd.DataFrame, building_id: int) -> pd.DataFrame:
        d = frame.copy()
        d[GROUP_COL] = building_id
        feats = build_feature_frame(d)
        if feats.empty:
            return pd.DataFrame(
                {TIME_COL: [], VALUE_COL: [], "score": [], "flag": [], "alarm": []}
            )
        cols = self.scorer.feature_order or feature_columns(feats)
        X = feats.reindex(columns=cols)
        proba = np.asarray(self.scorer.model.predict_proba(X)[:, 1], dtype="float64")
        thr = self.scorer.threshold

        # OUTAGE hours have no reading and therefore no score; keep them in the tail as OUTAGE.
        present = d.set_index(TIME_COL)[VALUE_COL]
        raw = present.reindex(feats[TIME_COL]).to_numpy()
        missing_mask = pd.isna(raw)

        alarm = np.where(missing_mask, "OUTAGE", np.where(proba >= thr, "ANOMALY", "NORMAL"))
        out = pd.DataFrame(
            {
                TIME_COL: feats[TIME_COL].to_numpy(),
                VALUE_COL: raw,
                "score": np.where(missing_mask, np.nan, proba),
                "flag": (~missing_mask) & (proba >= thr),
                "alarm": alarm,
            }
        )
        return out.sort_values(TIME_COL).reset_index(drop=True)


_HUB: LiveHub | None = None


def get_hub(scorer: AnomalyScorer) -> LiveHub:
    global _HUB
    if _HUB is None:
        _HUB = LiveHub(scorer)
    return _HUB