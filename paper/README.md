# Manuscript

`main.tex` is in IEEE conference format (`IEEEtran`, `conference` option), two columns.
Every number in it comes from `results/metrics/*.json`; `tests/test_paper_numbers.py` fails if the
two ever disagree.

## Build

```bash
python scripts/build_paper.py          # copies figures, validates, then runs pdflatex twice
```

Requires a LaTeX distribution with the `IEEEtran` class (MiKTeX or TeX Live). This environment has
none, so no PDF is committed; see blocker B-003 in `../BLOCKERS.md`. The source validates
structurally (citations, labels, figures, brace balance), but it has not been compiled here, and a
manuscript that has never been compiled should not be claimed as camera-ready.

### Overleaf (no local install)

1. Zip `paper/main.tex` and `paper/figures/`.
2. Upload, set `main.tex` as the main document.
3. The `IEEEtran` class ships with Overleaf; no extra package installation is needed.

## Structure

| Section | Contents |
|---|---|
| I Introduction | geothermal condition-monitoring problem, label scarcity, temporal leakage |
| II Related Work | anomaly detection, boosting, SHAP, dependence-aware validation, datasets |
| III Data | audit table, problem statement, splits, missing-reading policy |
| IV Method | causal feature contract, 34 features, model, metrics, capped tuning |
| V Results | baselines, family ablation, class weighting, held-out table, per-asset spread, SHAP |
| VI Deployment | one scoring path, dashboard, transfer test on **SIMULATED** geothermal signals |
| VII Discussion | why dynamics dominate, why thresholds are the fragile part |
| VIII Threats to validity | label noise, domain, single split, threshold transfer, incomparable benchmarks |
| IX Reproducibility | pipeline stages, pinned versions, artefact-to-claim test |
| X Conclusion | |

## Figures

| File | Source script |
|---|---|
| `holdout_roc_final.png`, `holdout_pr_final.png` | `src/evaluation/holdout.py` |
| `holdout_calibration_final.png` | `src/evaluation/holdout.py` |
| `shap_global_final.png` | `src/explainability/shap_explain.py` |

`scripts/build_paper.py` copies these from `results/figures/` so the manuscript cannot show a stale
plot that no script produced.

## Claims that are deliberately absent

- No comparison with published LEAD1.0 leaderboard numbers (different split, metrics not
  reproducible here; B-001).
- No claim about PLTP Kamojang performance. The geothermal experiment is **SIMULATED** and is
  described as a pipeline test with a negative transfer result.
- No claim that the model is production-ready: there is no authentication, no drift detection and no
  live SCADA connector.
