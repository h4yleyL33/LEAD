# Phase 2 — Data Understanding

**CRISP-DM phase:** 2 of 6
**Status:** complete
**Regenerate everything in this document with:**
```bash
python -m src.data.audit     # -> results/metrics/data_audit.json
python -m src.data.eda       # -> results/figures/eda_*.png, results/metrics/eda_summary.json
```

---

## 2.1 Data acquisition and provenance

| Item | Value |
|---|---|
| Dataset | LEAD1.0-small (LEAD1.0 reference subset) |
| Upstream origin | ASHRAE Great Energy Predictor III hourly electricity readings, re-annotated |
| Source | `https://github.com/samy101/lead-dataset` → `data/lead1.0-small.zip` |
| Citation | Gulati & Arjunan, *LEAD1.0: A Large-scale Annotated Dataset for Energy Anomaly Detection in Commercial Buildings*, BuildSys '22 |
| Archive SHA-256 | `b0e34c66…e39d8f` |
| CSV SHA-256 | `a161537f…62b32c` |
| Rows | 1,749,494 |
| Buildings | 200 |
| Period | 2016-01-01 00:00 → 2016-12-31 23:00 (8,784 distinct hours) |

The Kaggle competition split is unavailable in this environment (HTTP 401, no credentials).
That is recorded as blocker B-001 and does not compromise the scientific pipeline: the data used
here is real, human-annotated, and of the same schema.

## 2.2 Structure

- One row = one (building, hour). 200 × 8,784 = 1,756,800 possible cells; 1,749,494 present,
  i.e. **7,306 cells are absent entirely** (not NaN — no row).
- Readings per building: min 7,471 / median 8,760 / max 8,784. **96 of 200 buildings** cover the
  complete hourly grid; the rest have gaps.
- `meter_reading` is a positive continuous quantity: median 57.10 kWh, mean 137.03 kWh,
  p99 968.92 kWh, max 3,758.00 kWh, skewness 2.96 — heavy right tail, orders of magnitude between
  buildings.

## 2.3 Data quality

| Issue | Magnitude | Decision |
|---|---|---|
| Missing `meter_reading` | 107,653 rows (6.15%); worst building 61.59% | Keep rows; NaN is informative (sensor outage). XGBoost handles NaN natively; linear baselines use train-fitted median + `is_missing` flag. |
| Duplicate (building, hour) | 0 | Nothing to deduplicate. |
| Negative readings | 0 | No sign errors. |
| Zero readings | 25 | Physically implausible for a metered building but rare; retained, they are plausible outage markers. |
| Missing rows (absent cells) | 7,306 | Do **not** forward-fill across gaps; rolling features become NaN across gaps, which is honest. |
| Buildings with >10% missing | 31 (2 with >50%) | Kept; split stratification must not put a near-empty building alone in test. |

## 2.4 Target characterisation

- 37,296 anomalies → **2.13%** positive rate; **45.91 : 1** imbalance.
- Per-building anomaly rate: min 0.58%, p25 1.27%, median 1.94%, p75 2.73%, max 8.82%.
  Every building contains at least one anomaly; none is entirely anomalous.
- 5,503 contiguous anomaly events; median 2 h, p95 24 h, max 167 h; **47.2% are single-hour**
  events, i.e. roughly half the label mass is isolated points and half is sustained episodes.

**Interpretation.** The annotation is *event-like*, not i.i.d. point noise. A model that only looks
at the instantaneous value will miss sustained episodes whose individual points look unremarkable;
conversely, sustained episodes will dominate any per-point metric. This motivates (a) rolling
level/volatility features and (b) reporting both point-wise AUC and event-level recall.

## 2.5 Temporal structure

- Anomaly rate by hour of day is nearly flat: min 1.83% (18:00), max 2.14% (22:00).
- Weekday × hour heat-map ranges only 1.10%–3.61% — no strong calendar confound, but the
  variation is large enough to be worth giving the model calendar features rather than assuming
  stationarity.
- Median autocorrelation of `meter_reading`: **0.933 at lag 1 h**, **0.681 at lag 24 h**.
  Consumption is dominated by a strong daily cycle plus persistence.

**Consequence for Phase 3.** Autocorrelation at lag 1 of 0.93 means a *random* row split would let
the model read the answer off neighbouring hours. This is precisely why the evaluation protocol is
building-disjoint (D-002) and why every rolling feature must be strictly backward-looking (D-003).

## 2.6 Figures

| File | What it shows |
|---|---|
| `eda_class_balance.png` | 45.9:1 imbalance on a log axis |
| `eda_load_profile.png` | Weekday vs weekend median load with IQR; flat hourly anomaly rate |
| `eda_week_heatmap.png` | Anomaly rate by day-of-week × hour (1.10%–3.61%) |
| `eda_building_distribution.png` | Heterogeneity of per-building anomaly rate; coverage vs rate |
| `eda_timeseries_examples.png` | Two full-year buildings with anomalies overlaid |
| `eda_missingness.png` | Missingness concentrated in a few buildings (31 > 10%, 2 > 50%) |
| `eda_autocorrelation.png` | ACF decay to 48 h — the empirical case for causal windows |
| `eda_event_lengths.png` | Event-length distribution, median 2 h, max 167 h |

## 2.7 Data-quality risks carried into modelling

1. **Building-level scale differences** (median 57 kWh vs p99 969 kWh across the pool) mean raw
   level features are not comparable across buildings. Features must be *relative* (ratios,
   z-scores within building, log transforms) for a building-disjoint model to generalise.
2. **Label noise.** Annotations were produced by humans/rule-assisted review; some "anomalies" are
   benign operating changes. This caps achievable AUC and must be stated in the paper's limitations
   rather than explained away.
3. **Gaps.** 7,306 absent hours plus 6.15% NaN readings mean rolling windows must tolerate missing
   data without silently inventing it (`min_periods` chosen explicitly).

## 2.8 Exit criteria (met)

- [x] Volume, structure, quality and target distribution verified by a re-runnable script.
- [x] Class imbalance, event structure and autocorrelation quantified.
- [x] Consequences for split design and feature causality written down before modelling.
- [x] Figures regenerated from raw data with no manual editing.
