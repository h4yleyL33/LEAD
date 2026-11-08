# Data Audit — LEAD1.0-small

Generated: 2026-11-08T22:24:26+00:00
Source: `E:\project\LAB_AI\AIoT\data\raw\lead\lead1.0-small.csv`

## Volume
| Metric | Value |
|---|---|
| Rows | 1,749,494 |
| Buildings | 200 |
| Unique timestamps | 8,784 |
| Time range | 2016-01-01 00:00:00 → 2016-12-31 23:00:00 |
| Span (days) | 366.0 |
| Readings/building (min/median/max) | 7,471 / 8783 / 8,784 |
| Buildings with complete hourly grid | 96 |

## Quality
| Metric | Value |
|---|---|
| Duplicate (building, timestamp) rows | 0 |
| Missing `meter_reading` | 107,653 (6.15%) |
| Max missing rate in one building | 61.59% |
| Zero readings | 25 |
| Negative readings | 0 |
| Median / mean kWh | 84.10 / 179.90 |
| p99 / max kWh | 1461.93 / 6596.89 |
| Skewness | 9.15 |

## Target
| Metric | Value |
|---|---|
| Anomalies | 37,296 |
| Anomaly rate | 2.13% |
| Imbalance ratio (normal:anomaly) | 45.91:1 |
| Buildings with no anomalies | 0 |
| Buildings that are 100% anomalous | 0 |
| Per-building anomaly rate (min/median/max) | 0.58% / 1.94% / 8.82% |
| Anomaly events (contiguous runs) | 5,503 |
| Median event length | 2.0 h |
| Max event length | 167 h |
| Singleton events | 47.2% |
