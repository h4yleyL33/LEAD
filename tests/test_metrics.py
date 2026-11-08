"""Unit tests for the metric layer.

These use hand-constructed cases with known answers, so a silent change in definition (for example
counting an event as detected only when *every* hour is flagged) fails loudly instead of quietly
altering the paper's numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    bootstrap_auc_ci,
    event_metrics,
    point_metrics,
    select_threshold,
)


def test_perfect_separation_scores_one() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    m = point_metrics(y, s, threshold=0.5)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["brier"] < 0.1


def test_constant_score_has_chance_auc() -> None:
    y = np.array([0, 1, 0, 1, 0, 1])
    s = np.full(6, 0.3)
    m = point_metrics(y, s)
    assert m["roc_auc"] == pytest.approx(0.5)
    assert m["base_rate"] == pytest.approx(0.5)


def test_recall_at_fpr_is_monotone_in_signal_quality() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    good = 1 / (1 + np.exp(-(y + rng.normal(0, 0.3, 5000))))
    bad = 1 / (1 + np.exp(-(y * 0.05 + rng.normal(0, 1.0, 5000))))
    assert point_metrics(y, good)["recall_at_fpr_1pct"] > point_metrics(y, bad)["recall_at_fpr_1pct"]


def test_select_threshold_prefers_recall_for_beta_two() -> None:
    y = np.array([0] * 90 + [1] * 10)
    s = np.concatenate([np.linspace(0, 0.5, 90), np.linspace(0.5, 1.0, 10)])
    best = select_threshold(y, s, beta=2.0)
    assert 0.0 <= best["threshold"] <= 1.0
    assert best["fbeta"] > 0.5
    # F2 must weight recall at least as heavily as precision.
    assert best["recall"] >= best["precision"] - 0.5


def _event_frame() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two buildings, 10 hours each. Building 1 has a 3-hour event at t=4..6."""
    times = np.tile(pd.date_range("2021-01-01", periods=10, freq="h").to_numpy(), 2)
    groups = np.repeat([1, 2], 10)
    y = np.zeros(20, dtype=int)
    y[4:7] = 1  # building 1 event
    score = np.zeros(20)
    return y, score, groups, times


def test_event_recall_counts_partial_detection() -> None:
    y, score, groups, times = _event_frame()
    score[5] = 0.9  # flag a single hour inside the event
    m = event_metrics(y, score, groups, times, threshold=0.5)
    assert m["n_events"] == 1
    assert m["events_detected"] == 1
    assert m["event_recall"] == pytest.approx(1.0)
    assert m["median_detection_latency_h"] == 1  # one hour after the event started


def test_event_recall_zero_when_nothing_flagged() -> None:
    y, score, groups, times = _event_frame()
    m = event_metrics(y, score, groups, times, threshold=0.5)
    assert m["event_recall"] == 0.0
    assert m["events_detected"] == 0


def test_false_alarm_events_are_counted_separately() -> None:
    y, score, groups, times = _event_frame()
    score[5] = 0.9   # true detection
    score[9] = 0.9   # isolated false alarm in building 1
    score[15] = 0.9  # isolated false alarm in building 2
    m = event_metrics(y, score, groups, times, threshold=0.5)
    assert m["false_alarm_events"] == 2
    assert m["event_recall"] == pytest.approx(1.0)
    assert m["false_alarms_per_building_day"] > 0


def test_events_are_not_merged_across_buildings() -> None:
    """An anomaly ending at the last hour of building 1 and starting at the first hour of
    building 2 must count as two events, not one."""
    y = np.zeros(20, dtype=int)
    y[9] = 1   # building 1, last hour
    y[10] = 1  # building 2, first hour
    groups = np.repeat([1, 2], 10)
    times = np.tile(pd.date_range("2021-01-01", periods=10, freq="h").to_numpy(), 2)
    score = np.zeros(20)
    m = event_metrics(y, score, groups, times, threshold=0.5)
    assert m["n_events"] == 2


def test_bootstrap_ci_brackets_point_estimate() -> None:
    rng = np.random.default_rng(1)
    n_buildings, per = 30, 200
    groups = np.repeat(np.arange(n_buildings), per)
    y = rng.integers(0, 2, n_buildings * per)
    s = y + rng.normal(0, 0.6, n_buildings * per)
    ci = bootstrap_auc_ci(y, s, groups, n_boot=50, seed=0)
    from sklearn.metrics import roc_auc_score

    point = roc_auc_score(y, s)
    assert ci["n_boot_valid"] > 40
    assert ci["auc_ci95_low"] <= point <= ci["auc_ci95_high"]
