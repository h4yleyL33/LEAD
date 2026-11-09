"""Guard against the manuscript drifting away from the measured artefacts.

Three checks:

1. **Numbers.** Every headline value quoted in `paper/main.tex` must equal the value stored in the
   JSON produced by the script that computed it. If a model is retrained and the numbers move, this
   test fails until the paper is updated — which is the intended behaviour.
2. **References.** Every `\\cite{...}` key must exist in the bibliography, and every `\\ref{...}`
   must have a matching `\\label{...}`.
3. **Figures.** Every `\\includegraphics{...}` file must exist (in `paper/figures/` after the build
   script has copied them, or in `results/figures/`).

No LaTeX installation is required for these checks.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper" / "main.tex"
METRICS = ROOT / "results" / "metrics"
FIGURES = ROOT / "results" / "figures"


def _metrics(name: str) -> dict:
    return json.loads((METRICS / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tex() -> str:
    if not TEX.exists():
        pytest.skip("manuscript not present")
    return TEX.read_text(encoding="utf-8")


def _num(v: float, nd: int = 4) -> str:
    return f"{v:.{nd}f}"


# (json file, key path, decimals, human label, scale)
CASES: list[tuple[str, tuple[str, ...], int, str, float]] = [
    ("holdout_test_final.json", ("roc_auc",), 4, "test ROC-AUC", 1.0),
    ("holdout_test_final.json", ("pr_auc",), 4, "test PR-AUC", 1.0),
    ("holdout_test_final.json", ("brier",), 5, "test Brier", 1.0),
    ("holdout_test_final.json", ("ece_quantile_12bin",), 4, "test ECE", 1.0),
    ("holdout_test_final.json", ("recall_at_fpr_1pct",), 4, "recall at 1% FPR", 1.0),
    ("holdout_test_final.json", ("recall_at_fpr_0p1pct",), 4, "recall at 0.1% FPR", 1.0),
    ("holdout_test_final.json", ("precision",), 4, "precision at threshold", 1.0),
    ("holdout_test_final.json", ("recall",), 4, "recall at threshold", 1.0),
    ("holdout_test_final.json", ("event_recall",), 4, "episode recall", 1.0),
    ("holdout_test_final.json", ("auc_ci95_low",), 4, "CI low", 1.0),
    ("holdout_test_final.json", ("auc_ci95_high",), 4, "CI high", 1.0),
    ("xgb_final_val.json", ("roc_auc",), 4, "validation ROC-AUC", 1.0),
    ("xgb_final_val.json", ("pr_auc",), 4, "validation PR-AUC", 1.0),
    ("xgb_final_val.json", ("brier",), 4, "validation Brier", 1.0),
    ("xgb_final_val.json", ("event_recall",), 3, "validation episode recall", 1.0),
    ("baselines_val.json", ("results", "logistic", "roc_auc"), 4, "logistic ROC-AUC", 1.0),
    ("baselines_val.json", ("results", "iforest", "roc_auc"), 4, "iforest ROC-AUC", 1.0),
    ("baselines_val.json", ("results", "iforest", "pr_auc"), 4, "iforest PR-AUC", 1.0),
    ("data_audit.json", ("target", "anomaly_rate"), 2, "dataset anomaly rate %", 100.0),
    ("data_audit.json", ("target", "class_imbalance_ratio"), 2, "class imbalance ratio", 1.0),
    ("data_audit.json", ("volume", "n_rows"), 0, "row count", 1.0),
    ("data_audit.json", ("quality", "missing_meter_reading_rate"), 2, "missing rate %", 100.0),
]


def _normalise(tex: str) -> str:
    """Strip LaTeX digit-grouping and escape characters so numeric comparison is literal."""
    return tex.replace("{,}", "").replace("\\%", "%").replace("~", " ")


def _dig(d: dict, path: tuple[str, ...]):
    for k in path:
        if isinstance(d, list):
            d = d[k]  # type: ignore[index]
        else:
            d = d[k]
    return d


def test_headline_numbers_match_artifacts(tex: str) -> None:
    normalised = _normalise(tex)
    missing, absent = [], []
    for fname, path, nd, label, scale in CASES:
        try:
            value = _dig(_metrics(fname), path)
        except (KeyError, IndexError, TypeError):
            absent.append(f"{fname}:{'.'.join(path)}")
            continue
        if not isinstance(value, (int, float)):
            continue
        needle = _num(float(value) * scale, nd)
        if needle not in normalised:
            missing.append(f"{label}: expected '{needle}' from {fname}:{'.'.join(path)}")
    assert not missing, "manuscript out of sync with results:\n  " + "\n  ".join(missing)
    assert len(absent) < len(CASES) / 2, f"too many metrics files unavailable: {absent}"


def test_ablation_numbers_match(tex: str) -> None:
    rows = _metrics("ablation_families_only.json")["results"]
    by_name = {r["subset"]: r for r in rows}
    missing = []
    for fam in ("dynamics", "deviation", "level", "volatility", "calendar", "availability"):
        if fam not in by_name:
            continue
        for key, nd in (("roc_auc", 4), ("pr_auc", 4)):
            needle = _num(float(by_name[fam][key]), nd)
            if needle not in tex:
                missing.append(f"{fam} {key}: expected '{needle}'")
    assert not missing, "ablation table out of sync:\n  " + "\n  ".join(missing)


def test_simulation_numbers_match(tex: str) -> None:
    data = _metrics("simulation_detection_SIMULATED.json")
    rows = {r["equipment"]: r for r in data["results"]}
    missing = []
    for eq in ("separator", "pump", "bearing", "compressor"):
        if eq not in rows:
            continue
        needle = f"{rows[eq]['roc_auc']:.3f}"
        if needle not in tex:
            missing.append(f"{eq} ROC-AUC: expected '{needle}'")
    total = data["aggregate"]["fault_hours"]
    if f"of {total} fault hours" not in tex:
        missing.append(f"aggregate fault-hour count: expected 'of {total} fault hours'")
    assert not missing, "simulation table out of sync:\n  " + "\n  ".join(missing)


def test_citations_are_defined(tex: str) -> None:
    cited = set()
    for group in re.findall(r"\\cite\{([^}]*)\}", tex):
        cited.update(k.strip() for k in group.split(","))
    defined = set(re.findall(r"\\bibitem\{([^}]*)\}", tex))
    undefined = sorted(cited - defined)
    unused = sorted(defined - cited)
    assert not undefined, f"cited but not defined: {undefined}"
    assert not unused, f"defined but never cited: {unused}"


def test_refs_have_labels(tex: str) -> None:
    refs = set(re.findall(r"\\ref\{([^}]*)\}", tex))
    labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
    missing = sorted(refs - labels)
    assert not missing, f"\\ref without \\label: {missing}"


def test_figures_exist(tex: str) -> None:
    names = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex)
    assert names, "no figures referenced"
    missing = [
        n for n in names
        if not (ROOT / "paper" / "figures" / n).exists() and not (FIGURES / n).exists()
    ]
    assert not missing, f"missing figure files: {missing}"


def test_braces_balanced(tex: str) -> None:
    # Crude but effective: an unbalanced brace breaks any LaTeX build.
    assert tex.count("{") == tex.count("}"), "unbalanced braces in manuscript"
    assert tex.count("\\begin{") == tex.count("\\end{"), "unbalanced environments"


def test_simulated_disclaimer_present(tex: str) -> None:
    """The manuscript must state, in the abstract and the transfer section, that the geothermal
    signals are simulated. This is a hard requirement, not a style preference."""
    abstract = tex.split("\\begin{abstract}")[1].split("\\end{abstract}")[0]
    assert "SIMULATED" in abstract or "\\simnote" in abstract
    assert "not PLTP Kamojang" in tex or "not a geothermal plant" in tex.lower() or \
           "not on a geothermal plant" in tex
