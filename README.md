# Embryo Lab Data Analytics

**English** · [Русский](README.ru.md)

[![tests](https://github.com/embryossa/embryo-lab-data/actions/workflows/tests.yml/badge.svg)](https://github.com/embryossa/embryo-lab-data/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A local, end-to-end analytics pipeline for an IVF/ICSI embryology laboratory. It takes a
laboratory export (Excel/CSV) and produces:

- laboratory KPIs against the Vienna Consensus, with numerators, denominators, 95% CIs and
  competency benchmarks, split by period, method, embryologist and physician;
- predictions from external models (IVF Digital Twin, TabPFN, a TabPFN-3 regression of the
  embryo funnel, and an optional independent DNN), reconciled against observed outcomes;
- forensics of the culture window of flagged cycles: incubator logs (MIRI), consumable lot
  changes, PGT-A reports and oocyte-to-embryo arrest patterns (OZEMA);
- a pseudonymised review by Claude Code agents and a single HTML/XLSX report.

This is analysis support for a laboratory. **It is not a medical device and does not
recommend treatment.** Causal explanations in the report are hypotheses, not diagnoses.

> Most in-code comments, configuration notes and the agent instructions in `.claude/` are in
> Russian, the working language of the laboratories that use the pipeline. This README and
> `reports/README.md` are the English entry points.

## How it works

```
export (xlsx/csv)
  → column resolution (config/columns.yaml, no guessing)          src/embryo/ingest.py
  → normalisation: outcome, AMH/BMI imputation, zero ≠ missing      src/embryo/normalize.py
  → keyed-hash (HMAC) cycle_uid; IDs and names stay local           src/embryo/privacy.py
  → laboratory KPIs (Σ numerator / Σ denominator, Wilson CI)        src/embryo/kpi.py
  → external models: Digital Twin 7.1 REST API, TabPFN, TabPFN-3     src/embryo/*_runner.py, dt_api.py
  → join + versioned contract checks                                src/embryo/reconcile.py, contracts.py
  → calibration, funnel, clusters, review list                       src/embryo/analytics.py
  → culture-window forensics: MIRI, lots, PGT-A, OZEMA               incubator.py, lots.py, pgt.py, ozema.py
  → agent packets (cycle_uid + allow-listed features only)
  → Claude Code agents interpret; Python re-joins identifiers locally
  → atomic CSV/XLSX/HTML/JSON + run manifest with sha256             src/embryo/_io.py, report.py
```

Python computes and classifies; the agents only explain. The entry point is the Claude Code
skill `/run-analysis` (`.claude/skills/run-analysis/SKILL.md`), which runs
`scripts/run_analysis.py`, orchestrates the agents in `.claude/agents/` and finishes with
`scripts/finalize_report.py`. The owner's fixed methodological decisions are listed in
`CLAUDE.md`; methodology and deviations from the Vienna Consensus source are in
`docs/METHODOLOGY.md`; artefact contracts are in `docs/CONTRACTS.md`.

## Privacy model

- Raw exports, incubator logs, PGT-A PDFs, processed outputs and the `cycle_uid` crosswalk
  live under `data/` and are git-ignored. **This repository contains no patient data**: the
  only data files are synthetic fixtures (`tests/fixtures/`, `data/sample/`) and aggregated
  result tables in `reports/`.
- Agents never see IDs, names, dates of birth or partner data. They read only the
  `agent_packets` folder of the current run: `cycle_uid` plus an allow-list of analytic
  features. Identifiers are re-attached by local Python after interpretation.
- Only `cycle_uid` and numeric features are sent to the Digital Twin service.
- Tokens, machine paths and internal resource IDs live in git-ignored `*.local.*` files;
  every one has a committed `*.example` template.

## Repository layout

```
src/embryo/          pipeline core (ingest, normalize, kpi, analytics, reconcile, contracts,
                     privacy, incubator, lots, pgt, ozema, report, adapters to external tools)
  clinics/           per-clinic export adapters
scripts/             run_analysis.py (orchestrator), finalize_report.py, preflight.py,
                     check_setup.py, run_off_cohort.py
config/              settings.yaml, columns.yaml, clinics/<name>/, *.example templates
.claude/             Claude Code skill /run-analysis, agent roles, domain skills
                     (vienna-kpi, dt-ensemble, miri-forensics, lot-forensics, ozema, ...)
tests/               pytest suite on synthetic fixtures (e2e replay, contracts, privacy)
docs/                methodology, contracts, per-clinic notes, TabPFN install
external/            batch adapters copied into the external TabPFN / DNN projects
notebooks/reference/ original DataLore notebook the analytics module was ported from
teable/              optional sync of results into a self-hosted Teable instance
reports/             analysis code, aggregated tables and figures for manuscripts
data/                git-ignored, except data/sample/ (synthetic)
```

## Installation

Python 3.11, Windows (the external model services and batch files are Windows-based; the
core and the test suite are plain Python).

```powershell
git clone https://github.com/embryossa/embryo-lab-data.git
cd embryo-lab-data
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
python -m ruff check .
```

`requirements.txt` covers the deterministic core only. The prediction models are separate
projects with their own environments and are not part of this repository:

| Component | Role | How it is reached |
|---|---|---|
| IVF Digital Twin 7.1 | ensemble prediction of pregnancy and funnel | local REST API service (`src/embryo/dt_api.py`) |
| TabPFN | pregnancy probability per transfer with CI | batch adapter, `external/tabpfn/`, `docs/TABPFN_INSTALL.md` |
| TabPFN-3 regression | P10–P90 forecast of MII/2PN/Bl/Good Bl/Frozen | own venv, soft-fail step |
| DNN verifier (optional) | independent cross-check | `external/dnn/Evo_calculations.py`, requires the training scaler |

Machine-specific paths go to `config/machine.local.yaml` (template:
`config/machine.local.yaml.example`); per-clinic secrets go to `config/settings.local.yaml`
and `config/clinics/*/settings.local.yaml`. `python scripts/check_setup.py --all-clinics`
verifies the setup. The full walkthrough, including the Digital Twin API key setup, is in
[README.ru.md](README.ru.md).

## Running

Open the repository in Claude Code and run `/run-analysis`. Without the external models the
pipeline can be exercised through the test suite, which replays the whole run on synthetic
fixtures (`tests/test_run_analysis_e2e.py`).

Main outputs (`data/processed/`, git-ignored): `DT_Analytics_Report.html/.xlsx` (13-tab
report), `dt_predictions.csv`, `review_list.csv`, `miri_events.csv`, `lot_events.csv`,
`pgt_summary.csv`, `ozema_cycles.csv`, `regression_qc.csv`, `run_manifest.json`.

## Manuscript analyses

`reports/` holds the scripts, aggregated tables and figures behind the manuscripts. See
[reports/README.md](reports/README.md) for what each folder supports and what is needed to
re-run it. Cycle-level data are not published.

## License and citation

MIT, see [LICENSE](LICENSE). Citation metadata: [CITATION.cff](CITATION.cff).
