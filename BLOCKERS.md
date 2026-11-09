# BLOCKERS

External dependencies that this agent cannot resolve without the user. Each entry states what is
blocked, what was tried, and the exact user action that lifts it. Independent work continues
around every blocker.

---

## B-001 — Kaggle competition dataset requires credentials

**Status:** OPEN (non-blocking for the research pipeline)
**Opened:** 2026-11-09

**What is blocked**
The exact *Large-scale Energy Anomaly Detection* (LEAD) Kaggle competition split:
`train.csv` (200 buildings), `test.csv` (206 buildings), `train_features.csv`,
`test_features.csv`, `submission_format.csv` — ~1.34 GB. Also blocked: any leaderboard-comparable
score.

**What was tried**
1. `~/.kaggle/kaggle.json` — absent.
2. `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables — unset.
3. `kaggle` CLI — not installed; `kaggle` API `GET /api/v1/competitions/list` → **HTTP 401**.
4. Public mirrors searched (Hugging Face datasets API, GitHub code search, web search) — no
   legitimate full mirror of the competition split found.
5. Fallback obtained: `lead1.0-small.csv` from the authors' public repository — **real** annotated
   data, same schema, 200 buildings, 1.75 M rows. In use.

**Impact**
No impact on scientific validity of the pipeline, model, backend, dashboard, simulation, or paper.
Impact is limited to: no leaderboard ranking claim, no `row_id` submission artifact.

**User action to lift (optional)**
Create `C:\Users\<you>\.kaggle\kaggle.json` with a Kaggle API token, accept the competition rules
on kaggle.com, then tell the agent to re-run `scripts/fetch_data.py --kaggle`. The pipeline will
switch to the competition split automatically if `data/raw/lead/train.csv` exists.

---

## B-002 — No real PLTP Kamojang SCADA/telemetry export

**Status:** OPEN (by design; simulation is the specified deliverable)
**Opened:** 2026-11-09

**What is blocked**
Any empirical claim about the actual Kamojang plant (real anomaly prevalence, real equipment
failure rates, real sensor statistics).

**What was done instead**
`src/simulation/` generates physically motivated auxiliary-equipment signals with documented
parameters, explicitly labelled SIMULATED. The manuscript describes the geothermal section as a
*transferability demonstration on synthetic signals*, not as plant evidence.

**User action to lift (optional)**
Provide a de-identified SCADA export (timestamp, tag, value, unit, equipment id) with written
permission to use it. Until then, no real-plant claim will be made.

---

## B-003 — No LaTeX distribution, so the manuscript PDF cannot be compiled here

**Status:** OPEN (blocks the PDF artefact only; the source is complete and validated)
**Opened:** 2026-11-09

**What is blocked**
`paper/main.pdf`. `pdflatex`, `latexmk`, `tectonic` and `pandoc` are all absent from this
environment, and `IEEEtran.cls` is not installed.

**What was done instead**
1. `paper/main.tex` written in IEEE conference format with the full content.
2. `scripts/build_paper.py` copies the referenced figures from `results/figures/` and validates the
   source: every `\cite` key defined, every `\ref` labelled, every figure file present, braces and
   environments balanced.
3. `tests/test_paper_numbers.py` (8 tests) verifies that every headline number in the manuscript
   equals the value in `results/metrics/*.json`, so the text cannot drift from the artefacts.
4. `paper/README.md` documents the build and the Overleaf route.

**Honest limitation**
The source has never been compiled, so page count, float placement and any LaTeX typo are unverified.
It must not be described as camera-ready until it compiles cleanly.

**User action to lift (optional)**
Install MiKTeX or TeX Live, then run `python scripts/build_paper.py`. Or upload `paper/main.tex` plus
`paper/figures/` to Overleaf (the `IEEEtran` class is preinstalled there).
