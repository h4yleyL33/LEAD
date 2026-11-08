"""Inference service: the only place where a score is produced.

Both the FastAPI backend and the dashboard call `AnomalyScorer`, so training and serving cannot
drift apart: features are computed by the *same* `build_feature_frame` used to build the training
matrix, on the caller's history, with the same past-only rules.

Two alarm types are distinguished, which matters operationally:

* `ANOMALY`   -- the model's probability exceeds the validation-selected threshold;
* `OUTAGE`    -- the hour has no reading at all. LEAD1.0 never labels a missing hour as an anomaly
                 (0 of 107,653), so the model cannot express this case; a dead sensor is still a
                 fault and must be surfaced, just under its own label.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.load import feature_columns  # noqa: E402
from src.features.build_features import build_feature_frame  # noqa: E402
from src.utils.config import GROUP_COL, METRICS_DIR, MODELS_DIR, TARGET_COL, TIME_COL, VALUE_COL

MIN_HISTORY_HOURS = 24 * 7  # below one week the long-window features are mostly NaN


@dataclass
class ScoreResult:
    building_id: int
    timestamp: str
    score: float
    threshold: float
    alarm: str  # "ANOMALY" | "OUTAGE" | "NORMAL"
    reading: float | None
    features: dict = field(default_factory=dict)
    model_tag: str = ""
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "building_id": self.building_id,
            "timestamp": self.timestamp,
            "score": round(float(self.score), 6),
            "threshold": round(float(self.threshold), 6),
            "alarm": self.alarm,
            "reading": None if self.reading is None or np.isnan(self.reading)
            else float(self.reading),
            "model": self.model_tag,
            "notes": self.notes,
        }


class AnomalyScorer:
    """Loads one XGBoost model plus its validation-selected threshold."""

    def __init__(self, variant: str = "final", models_dir: Path | None = None):
        self.variant = variant
        self.model_tag = f"xgboost_{variant}"
        mdir = models_dir or MODELS_DIR
        model_path = mdir / f"xgb_{variant}.json"
        metrics_path = METRICS_DIR / f"xgb_{variant}_val.json"
        if not model_path.exists():
            raise FileNotFoundError(f"model not found: {model_path}")
        if not metrics_path.exists():
            raise FileNotFoundError(f"validation metrics not found: {metrics_path}")

        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        meta = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.threshold = float(meta["threshold"])
        self.val_metrics = {k: meta[k] for k in ("roc_auc", "pr_auc", "event_recall")
                            if k in meta}
        self.feature_order = list(meta.get("features") or [])

    # ------------------------------------------------------------------ helpers
    def _frame(self, history: pd.DataFrame, building_id: int) -> pd.DataFrame:
        d = history[[TIME_COL, VALUE_COL]].copy()
        d[TIME_COL] = pd.to_datetime(d[TIME_COL])
        d[GROUP_COL] = building_id
        d = d.sort_values(TIME_COL).drop_duplicates(subset=[TIME_COL], keep="last")
        return d

    def score_history(self, history: pd.DataFrame, building_id: int = 0) -> ScoreResult:
        """Score the last hour of `history` (a DataFrame with timestamp + meter_reading)."""
        d = self._frame(history, building_id)
        if d.empty:
            raise ValueError("empty history")

        last_ts = d[TIME_COL].max()
        last_reading = d.loc[d[TIME_COL] == last_ts, VALUE_COL].iloc[0]

        feats = build_feature_frame(d)
        if feats.empty:
            raise ValueError("could not build features from the supplied history")
        row = feats.sort_values(TIME_COL).iloc[[-1]]

        notes: list[str] = []
        if len(d) < MIN_HISTORY_HOURS:
            notes.append(
                f"history shorter than {MIN_HISTORY_HOURS} h; long-window features are unreliable"
            )

        if pd.isna(last_reading):
            return ScoreResult(
                building_id=int(building_id), timestamp=str(last_ts), score=float("nan"),
                threshold=self.threshold, alarm="OUTAGE", reading=None,
                model_tag=self.model_tag,
                notes=notes + ["no reading at this hour: reported as OUTAGE, not scored"],
            )

        cols = self.feature_order or feature_columns(feats)
        X = row.reindex(columns=cols)
        score = float(self.model.predict_proba(X)[:, 1][0])
        alarm = "ANOMALY" if score >= self.threshold else "NORMAL"

        return ScoreResult(
            building_id=int(building_id), timestamp=str(last_ts), score=score,
            threshold=self.threshold, alarm=alarm, reading=float(last_reading),
            features={c: (None if pd.isna(X.iloc[0][c]) else float(X.iloc[0][c])) for c in cols},
            model_tag=self.model_tag, notes=notes,
        )

    def score_frame(self, history: pd.DataFrame, building_col: str = GROUP_COL) -> list[ScoreResult]:
        """Score every building present in a long history frame."""
        out = []
        for b, sub in history.groupby(building_col):
            try:
                out.append(self.score_history(sub, building_id=int(b)))
            except ValueError as exc:  # pragma: no cover - defensive
                out.append(ScoreResult(
                    building_id=int(b), timestamp="", score=float("nan"),
                    threshold=self.threshold, alarm="ERROR", reading=None,
                    model_tag=self.model_tag, notes=[str(exc)],
                ))
        return out


def _demo() -> int:
    """Score the last hour of a real building from the training data, as a smoke test."""
    from src.data.load import load_features

    df = load_features()
    b = int(df[GROUP_COL].iloc[0])
    hist = df[df[GROUP_COL] == b][[TIME_COL, VALUE_COL]]
    scorer = AnomalyScorer()
    res = scorer.score_history(hist, building_id=b)
    print(json.dumps(res.to_dict(), indent=2))
    print(f"label at that hour: {int(df[(df[GROUP_COL] == b)].sort_values(TIME_COL)[TARGET_COL].iloc[-1])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())
