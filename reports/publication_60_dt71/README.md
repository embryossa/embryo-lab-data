# Publication run — IVF Digital Twin 7.1, "OPU table for publication.xlsx"

Run 2026-09-29, DT 7.1 REST API (service IVFClinicIntegration71), n_sim 2000,
model_fingerprint 91704e29…. Only DT: no TabPFN / TabPFN-3 / DNN / PGT-A / MIRI / lots / agents.
Source sha256 in `run_manifest.json`; supersedes the stale `reports/centerD_60cycles_dt71`
(built from an older version of the same file, never scored).

Cohort: 126 source rows -> 60 with known outcome -> 59 eligible (ET>0, 2PN≠0), 26 pregnancies.
DT coverage: 57 transfer view, 2 retrieval_fallback (`cascade:known_pn2` at transfer view).
Imputed inputs: AMH 53/59, BMI 22/59, sperm source inferred 6/59; AFC taken from follicles 11/59.
One cycle (Bl=0 observed, transfer day missing) gets headline probability 0.

Files (cycle_uid only; crosswalk and replay file with _key are in
`data/processed/local_private/publication_60_dt71/`):
- `dt71_predictions.csv` — per cycle: observed inputs/outcome + all DT 7.1 outputs
- `pregnancy_metrics.csv` — AUC (2,000 outcome-stratified bootstrap 95% CI), Brier, calibration gap
- `calibration_bins.csv` — headline probability quintiles, observed rate with Wilson 95% CI
- `stage_agreement.csv` — retrieval-view funnel forecast vs observed (Spearman, 95% PI coverage)
- `cohort_characteristics.csv` — median [IQR] by outcome
- `run_dt.py` (`--check` = dry validation), `evaluate.py`, `run_log.txt`, `run_manifest.json`
