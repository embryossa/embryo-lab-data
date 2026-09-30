#!/usr/bin/env python3
"""Step 3: DT 7.1 output of the separate 60-cycle center D run -> tables (no agents, no interpretation).

Reads dt71_results.json (step 2), imputation_flags.csv and outcomes.csv (step 1) from this
folder, maps the rows with embryo-lab-data dt_api.to_dt_output (transfer view, retrieval
fallback, funnel from the retrieval view), and writes:
  dt71_predictions.csv   one row per cycle: cycle_uid, outcome, observed counts, DT 7.1 outputs
  pregnancy_metrics.csv  AUC (2,000 outcome-stratified bootstrap CI), Brier, mean prediction, gap
  stage_agreement.csv    retrieval-view forecasts vs observed: Spearman rho (bootstrap CI), 95% coverage
  run_summary.json       coverage, timepoints used, fingerprint, imputation counts
No identifiers are read or written; results are NOT merged with any earlier run.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from embryo import dt_api  # noqa: E402

results = json.loads((HERE / "dt71_results.json").read_text(encoding="utf-8"))
flags = pd.read_csv(HERE / "imputation_flags.csv")
obs = pd.read_csv(HERE / "outcomes.csv")
src = obs[["cycle_uid"]].copy(); src["_key"] = src["cycle_uid"]
dt = dt_api.to_dt_output(src, results, flags, ("transfer", "retrieval"), fallback=True)
keep = ["cycle_uid", "dt_api_status", "dt_timepoint_used", "dt_api_model_fingerprint", "dt_api_seed",
        "clinical_probability", "clinical_probability_kind", "befe_per_transfer", "befe_ci_low", "befe_ci_high",
        "befe_reliability", "befe_n_eff", "p_per_transfer", "l1_l4_per_transfer", "p_kat_raw", "p_csdi", "gnn_prob",
        "gnn_ensemble_prob", "bayes_mean", "cycle_probability", "p_overall_cycle", "ood_final",
        "retrieval_clinical_probability", "retrieval_befe_per_transfer", "retrieval_p_per_transfer",
        "retrieval_p_kat_raw", "retrieval_cycle_probability",
        "med_pn2", "p025_pn2", "p975_pn2", "med_blasts", "p025_blasts", "p975_blasts", "med_good", "p025_good",
        "p975_good", "p_no_blast", "p_no_good_blast",
        "amh_imputed", "bmi_imputed", "afc_source", "attempt_imputed", "sperm_source_inferred", "okk_raised_for_thawed"]
dt = dt[[c for c in keep if c in dt.columns]]
full = obs.merge(dt, on="cycle_uid", how="left", validate="one_to_one")
full["Preg"] = pd.to_numeric(full["Preg"], errors="coerce")
full.to_csv(HERE / "dt71_predictions.csv", index=False)

rng = np.random.default_rng(0)
def boot_auc(y, p, B=2000):
    y, p = np.asarray(y), np.asarray(p); pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(B):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        vals.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(vals, [2.5, 97.5])

ok = full[full["dt_api_status"] == "complete"]
rows = []
for col, label in [("clinical_probability", "L7 headline per-transfer (transfer view)"),
                   ("befe_per_transfer", "L7 BEFE per-transfer (transfer view)"),
                   ("p_per_transfer", "L2 per-transfer (transfer view)"),
                   ("p_kat_raw", "L3 KAT (transfer view)"),
                   ("p_csdi", "L5 CSDI (transfer view)"),
                   ("gnn_prob", "L6 graph (transfer view)"),
                   ("cycle_probability", "Cycle probability (transfer view)"),
                   ("retrieval_clinical_probability", "L7 headline per-transfer (retrieval view)"),
                   ("retrieval_p_kat_raw", "L3 KAT (retrieval view)")]:
    if col not in ok.columns:
        continue
    d = ok[["Preg", col]].apply(pd.to_numeric, errors="coerce").dropna()
    if d["Preg"].nunique() < 2 or len(d) < 10:
        continue
    y, p = d["Preg"].to_numpy().astype(int), d[col].to_numpy()
    lo, hi = boot_auc(y, p)
    rows.append({"output": label, "column": col, "n": len(d), "events": int(y.sum()),
                 "auc": roc_auc_score(y, p), "auc_lo": lo, "auc_hi": hi,
                 "brier": float(np.mean((p - y) ** 2)), "mean_pred": float(p.mean()),
                 "observed": float(y.mean()), "gap_pp": 100 * float(p.mean() - y.mean())})
pd.DataFrame(rows).to_csv(HERE / "pregnancy_metrics.csv", index=False)

srows = []
for stage, med, lo, hi, real in [("2PN zygotes", "med_pn2", "p025_pn2", "p975_pn2", "two_pn"),
                                 ("Blastocysts", "med_blasts", "p025_blasts", "p975_blasts", "bl_total"),
                                 ("Good-quality blastocysts", "med_good", "p025_good", "p975_good", "bl_good")]:
    if med not in ok.columns or real not in ok.columns:
        continue
    d = ok[[med, lo, hi, real]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 10:
        continue
    rho = spearmanr(d[med], d[real]).correlation
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(d), len(d))
        bs.append(spearmanr(d[med].to_numpy()[i], d[real].to_numpy()[i]).correlation)
    cov = float(((d[real] >= d[lo]) & (d[real] <= d[hi])).mean())
    srows.append({"stage": stage, "known_to_model": "OCC + MII (retrieval view)", "n": len(d), "rho": rho,
                  "rho_lo": np.nanpercentile(bs, 2.5), "rho_hi": np.nanpercentile(bs, 97.5),
                  "coverage_95": cov, "median_abs_error": float((d[med] - d[real]).abs().median())})
pd.DataFrame(srows).to_csv(HERE / "stage_agreement.csv", index=False)

fps = sorted(set(full["dt_api_model_fingerprint"].dropna().astype(str)))
summary = {"cycles": int(len(full)), "complete": int((full["dt_api_status"] == "complete").sum()),
           "timepoint_used": full["dt_timepoint_used"].value_counts(dropna=False).astype(int).to_dict(),
           "model_fingerprints": fps, "pregnancies": int(full["Preg"].sum()),
           "imputed": {c: int(pd.Series(full[c]).fillna(False).astype(bool).sum())
                       for c in ("amh_imputed", "bmi_imputed", "attempt_imputed", "sperm_source_inferred",
                                 "okk_raised_for_thawed") if c in full.columns},
           "afc_source": full["afc_source"].value_counts(dropna=False).astype(int).to_dict() if "afc_source" in full else {}}
(HERE / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
print(pd.DataFrame(rows).round(3).to_string(index=False))
print(pd.DataFrame(srows).round(3).to_string(index=False))
