"""Build the IEEE manuscript.

Steps
-----
1. Copy the figures the manuscript references from `results/figures/` into `paper/figures/`.
2. Run `pdflatex` twice (twice for cross-references) if a LaTeX installation is available.
3. If no LaTeX is installed, validate the source instead and report clearly what is missing.

The validation is the same check that `tests/test_paper_numbers.py` performs: citations defined,
references labelled, figures present, braces balanced. It is not a substitute for compiling, and the
script says so rather than pretending the PDF exists.

Usage:
    python scripts/build_paper.py
    python scripts/build_paper.py --no-copy
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
FIG_SRC = ROOT / "results" / "figures"


def referenced_figures(tex: str) -> list[str]:
    return re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex)


def validate(tex: str) -> list[str]:
    problems = []
    cited = set()
    for group in re.findall(r"\\cite\{([^}]*)\}", tex):
        cited.update(k.strip() for k in group.split(","))
    defined = set(re.findall(r"\\bibitem\{([^}]*)\}", tex))
    if cited - defined:
        problems.append(f"cited but not defined: {sorted(cited - defined)}")
    if defined - cited:
        problems.append(f"defined but never cited: {sorted(defined - cited)}")

    refs = set(re.findall(r"\\ref\{([^}]*)\}", tex))
    labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
    if refs - labels:
        problems.append(f"\\ref without \\label: {sorted(refs - labels)}")

    if tex.count("{") != tex.count("}"):
        problems.append("unbalanced braces")
    if tex.count("\\begin{") != tex.count("\\end{"):
        problems.append("unbalanced environments")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-copy", action="store_true", help="skip copying figures")
    args = ap.parse_args()

    tex_path = PAPER / "main.tex"
    if not tex_path.exists():
        print(f"ERROR: {tex_path} not found")
        return 1
    tex = tex_path.read_text(encoding="utf-8")

    figures = referenced_figures(tex)
    print(f"manuscript references {len(figures)} figure file(s)")

    dest = PAPER / "figures"
    if not args.no_copy:
        dest.mkdir(parents=True, exist_ok=True)
        missing = []
        for f in figures:
            src = FIG_SRC / f
            if src.exists():
                shutil.copyfile(src, dest / f)
            else:
                missing.append(f)
        print(f"copied {len(figures) - len(missing)} figure(s) to {dest}")
        if missing:
            print(f"ERROR: missing figures: {missing}")
            return 1

    problems = validate(tex)
    if problems:
        print("source validation FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("source validation passed (citations, labels, figures, braces)")

    latex = shutil.which("pdflatex")
    if not latex:
        print("\nNo pdflatex found: the PDF was NOT built.")
        print("Install a TeX distribution (MiKTeX or TeX Live) and re-run this script, or")
        print("upload paper/main.tex plus paper/figures/ to Overleaf. See paper/README.md.")
        print("Recorded as blocker B-003 in BLOCKERS.md.")
        return 0

    for attempt in (1, 2):
        print(f"\npdflatex pass {attempt}...")
        proc = subprocess.run(
            [latex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=PAPER, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(proc.stdout[-4000:])
            print(f"pdflatex failed on pass {attempt}")
            return 1

    pdf = PAPER / "main.pdf"
    if pdf.exists():
        print(f"\nbuilt {pdf} ({pdf.stat().st_size/1024:.0f} KB)")
        return 0
    print("pdflatex reported success but no PDF was produced")
    return 1


if __name__ == "__main__":
    sys.exit(main())
