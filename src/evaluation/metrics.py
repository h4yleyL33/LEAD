"""Shared metric definitions for anomaly detection on imbalanced hourly telemetry.

Two families of numbers are reported, because they answer different questions:

*Point-wise* — "if I look at one hour, how well is it ranked?" (ROC-AUC, PR-AUC, Brier).
*Event-wise* — "if a fault lasts six hours, did we notice it at all?" (event recall, detection
latency). A model can have a respectable point AUC while still missing most episodes, so both are
required; the paper reports both.

Threshold selection is always done on validation data, never on test.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def point_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float | None = None) -> dict:
    """Threshold-free ranking metrics plus (optionally) metrics at a fixed operating point."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    out: dict[str, float] = {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        # Brier is only defined for probabilities; unsupervised scores (e.g. Isolation Forest
        # decision values) are unbounded, so report NaN rather than a meaningless number.
        "brier": float(brier_score_loss(y_true, y_score))
        if (y_score.min() >= 0.0 and y_score.max() <= 1.0) else float("nan"),
        "base_rate": float(y_true.mean()),
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
    }
    # Recall achievable while holding the false-positive rate at 1% -- a maintenance team can
    # tolerate roughly one false alarm per 100 normal hours.
    fpr, tpr, _ = roc_curve(y_true, y_score)
    out["recall_at_fpr_1pct"] = float(np.interp(0.01, fpr, tpr))
    out["recall_at_fpr_0p1pct"] = float(np.interp(0.001, fpr, tpr))

    if threshold is not None:
        pred = (y_score >= threshold).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )
        out.update({
            "threshold": float(threshold),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "alert_rate": float(pred.mean()),
        })
    return out


def select_threshold(
    y_true: np.ndarray, y_score: np.ndarray, beta: float = 2.0, grid: int = 500
) -> dict:
    """Pick the operating point maximising F-beta on validation data.

    beta=2 weights recall twice as heavily as precision, matching the cost asymmetry in
    maintenance: a missed incipient fault costs far more than an extra inspection.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    lo, hi = float(np.min(y_score)), float(np.max(y_score))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return {"threshold": float(lo), "fbeta": 0.0, "precision": 0.0, "recall": 0.0}

    best = {"threshold": float(hi), "fbeta": -1.0, "precision": 0.0, "recall": 0.0}
    for t in np.linspace(lo, hi, grid):
        pred = y_score >= t
        tp = int(np.sum(pred & (y_true == 1)))
        if tp == 0:
            continue
        fp = int(np.sum(pred & (y_true == 0)))
        fn = int(np.sum(~pred & (y_true == 1)))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        b2 = beta * beta
        denom = b2 * precision + recall
        fbeta = (1 + b2) * precision * recall / denom if denom > 0 else 0.0
        if fbeta > best["fbeta"]:
            best = {"threshold": float(t), "fbeta": float(fbeta),
                    "precision": float(precision), "recall": float(recall)}
    return best


def _events(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as (start, end_exclusive)."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    padded = np.concatenate(([False], m, [False]))
    d = np.diff(padded.astype(np.int8))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def event_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    times: np.ndarray,
    threshold: float,
) -> dict:
    """Episode-level detection: did we flag at least one hour of each true anomaly event?

    `groups` must be the building id and `times` the timestamp of each row, so that event
    boundaries are computed inside a building and never across buildings.
    """
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(y_score) >= threshold).astype(int)
    groups = np.asarray(groups)
    times = np.asarray(times)

    n_events = 0
    detected = 0
    latencies: list[int] = []
    false_alarm_events = 0

    for g in np.unique(groups):
        sel = groups == g
        order = np.argsort(times[sel], kind="mergesort")
        yt = y_true[sel][order]
        pr = pred[sel][order]

        for start, end in _events(yt == 1):
            n_events += 1
            hit = np.where(pr[start:end] == 1)[0]
            if len(hit) > 0:
                detected += 1
                latencies.append(int(hit[0]))

        for start, end in _events(pr == 1):
            if yt[start:end].sum() == 0:
                false_alarm_events += 1

    return {
        "n_events": int(n_events),
        "events_detected": int(detected),
        "event_recall": float(detected / n_events) if n_events else 0.0,
        "median_detection_latency_h": float(np.median(latencies)) if latencies else float("nan"),
        "false_alarm_events": int(false_alarm_events),
        "false_alarms_per_building_day": float(
            false_alarm_events / (len(np.unique(groups)) * 365.0)
        ),
    }


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 200,
    seed: int = 0,
) -> dict:
    """Building-level bootstrap CI for ROC-AUC.

    Resampling *buildings* (not rows) respects the dependence structure introduced by the
    building-disjoint protocol; a row bootstrap would produce a falsely narrow interval.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    groups = np.asarray(groups)

    uniq = np.unique(groups)
    index = {g: np.where(groups == g)[0] for g in uniq}
    aucs = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index[g] for g in pick])
        yt = y_true[idx]
        if yt.min() == yt.max():
            continue
        aucs.append(roc_auc_score(yt, y_score[idx]))

    aucs = np.array(aucs)
    return {
        "n_boot_valid": int(len(aucs)),
        "auc_mean": float(aucs.mean()) if len(aucs) else float("nan"),
        "auc_ci95_low": float(np.percentile(aucs, 2.5)) if len(aucs) else float("nan"),
        "auc_ci95_high": float(np.percentile(aucs, 97.5)) if len(aucs) else float("nan"),
    }
