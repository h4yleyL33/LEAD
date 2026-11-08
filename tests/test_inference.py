"""Inference-service tests.

The important one is `test_serving_matches_batch_features`: it rebuilds features for one building
from its raw history through the *serving* path and checks the resulting score equals the score the
model gives on the stored training-time feature row. That is the train/serve consistency guarantee
the paper claims, and it fails loudly if the two paths ever diverge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.load import load_features
from src.inference.service import AnomalyScorer, MIN_HISTORY_HOURS
from src.utils.config import GROUP_COL, MODELS_DIR, TIME_COL, VALUE_COL

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "xgb_primary.json").exists(),
    reason="trained model not present; run `python -m src.models.train_xgboost`",
)


@pytest.fixture(scope="module")
def scorer() -> AnomalyScorer:
    return AnomalyScorer("primary")


@pytest.fixture(scope="module")
def one_building() -> pd.DataFrame:
    df = load_features()
    b = int(df[GROUP_COL].iloc[0])
    return df[df[GROUP_COL] == b].sort_values(TIME_COL).reset_index(drop=True)


def test_serving_matches_batch_features(scorer: AnomalyScorer, one_building: pd.DataFrame) -> None:
    hist = one_building[[TIME_COL, VALUE_COL]]
    res = scorer.score_history(hist, building_id=int(one_building[GROUP_COL].iloc[0]))
    assert res.alarm in {"NORMAL", "ANOMALY", "OUTAGE"}
    assert 0.0 <= res.score <= 1.0

    # Same row scored straight from the stored feature matrix must give the same probability.
    row = one_building[one_building[TIME_COL] == pd.Timestamp(res.timestamp)]
    assert len(row) == 1
    cols = scorer.feature_order
    direct = float(scorer.model.predict_proba(row[cols])[:, 1][0])
    assert direct == pytest.approx(res.score, abs=1e-6)


def test_outage_is_reported_as_outage_not_anomaly(scorer: AnomalyScorer,
                                                 one_building: pd.DataFrame) -> None:
    hist = one_building[[TIME_COL, VALUE_COL]].copy()
    hist.loc[hist.index[-1], VALUE_COL] = np.nan
    res = scorer.score_history(hist, building_id=int(one_building[GROUP_COL].iloc[0]))
    assert res.alarm == "OUTAGE"
    assert np.isnan(res.score)
    assert any("OUTAGE" in n for n in res.notes)


def test_short_history_is_flagged(scorer: AnomalyScorer, one_building: pd.DataFrame) -> None:
    hist = one_building[[TIME_COL, VALUE_COL]].head(MIN_HISTORY_HOURS - 24)
    res = scorer.score_history(hist, building_id=0)
    assert any("shorter than" in n for n in res.notes)


def test_score_frame_covers_every_building(scorer: AnomalyScorer) -> None:
    df = load_features()
    buildings = sorted(df[GROUP_COL].unique())[:3]
    hist = df[df[GROUP_COL].isin(buildings)][[GROUP_COL, TIME_COL, VALUE_COL]]
    results = scorer.score_frame(hist)
    assert len(results) == len(buildings)
    assert {r.building_id for r in results} == set(buildings)
    assert all(r.alarm != "ERROR" for r in results)


def test_alert_is_deterministic(scorer: AnomalyScorer, one_building: pd.DataFrame) -> None:
    hist = one_building[[TIME_COL, VALUE_COL]]
    a = scorer.score_history(hist, building_id=0)
    b = scorer.score_history(hist, building_id=0)
    assert a.score == b.score
    assert a.alarm == b.alarm
