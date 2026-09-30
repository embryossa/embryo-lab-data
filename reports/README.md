# Manuscript analyses

Scripts, aggregated result tables and figures behind the manuscripts that use this pipeline.
Each folder is self-contained: its scripts regenerate the tables and figures next to them.

## What is and is not published

Published: analysis code, aggregated tables (metrics, CIs, tests, distributions), figures
(300 dpi PNG + vector PDF), figure legends and written reports.

Not published: cycle-level data. Per-cycle predictions, API payloads, outcomes, per-cycle
figure inputs, run logs and intermediate pickles stay on the laboratory machine and are
listed in `.gitignore`. Re-running a folder therefore needs the local inputs named in its
scripts:

- `data/processed/...` and `data/inbox/...` — pipeline outputs and exports (git-ignored);
- external projects (IVF Digital Twin, the TabPFN training context, the TabFM work folder) —
  located through `config/research_paths.local.json` (template
  `config/research_paths.local.json.example`) or `EMBRYO_<KEY>` environment variables, see
  `_local_paths.py`.

On Windows set `PYTHONIOENCODING=utf-8` before running scripts that print non-ASCII tables.

## Folders

| Folder | Content | Cohort |
|---|---|---|
| `ivfworldwide_figures/` | CSDI Digital Twin vs TabPFN-3: funnel-stage accuracy, interval calibration, recalibration, subgroups (Figures 1–4) | prospective, n = 250 |
| `ivfworldwide_benchmark/` | same comparison on a public benchmark, plus pregnancy classification (Figures B, B2) | FRESH + FET_FC, n = 489 |
| `rbmo_prospective/` | four-centre prospective cohort: Bland–Altman agreement DT vs TabPFN, prognosis by centre; `build_cohort.py` freezes the cohort | 4 centres, n = 1,026 |
| `rbmo_prospective/guo_three_stage/` | three-stage model of Guo et al. (2026), leave-one-centre-out, against Digital Twin; training-set overlap check | same |
| `rbmo_prospective/kat_components/` | KAN and FT-Transformer components of the KAT layer scored separately | same |
| `cohort_a_444/` | full Digital Twin re-run on centre A validation cohort | n = 444 transfers |
| `publication_60_dt71/` | Digital Twin 7.1 (REST API) run on the centre D publication table | n = 59 transfers |
| `centerD_60cycles_dt71/` | earlier payload build for the same table; superseded by `publication_60_dt71/`, kept for provenance | — |
| `csbj_data_quality/` | data quality, anomaly and distribution analyses (aggregates only) | 4 centres + centre D |
| `tabfm_vs_tabpfn/` | TabFM 1.0.0 vs TabPFN-3 on laboratory data (report in Russian) | centres A, D |

Centre codes used in the manuscripts: A = Fertimed, B = GGRC, C = SkyFert, D = Two-Line.
