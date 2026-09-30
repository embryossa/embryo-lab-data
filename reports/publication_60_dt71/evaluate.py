#!/usr/bin/env python3
"""Tables for the publication from dt71_predictions.csv (run_dt.py). No agents, no interpretation.

Writes:
  pregnancy_metrics.csv  per DT output: n, events, ROC-AUC (2,000 outcome-stratified bootstrap
                         95% CI), Brier, mean prediction vs observed rate
  calibration_bins.csv   headline probability by quintile: mean predicted vs observed (Wilson CI)
  stage_agreement.csv    retrieval-view funnel forecast vs observed: Spearman rho (bootstrap CI),
                         share of observed counts inside the model's 95% interval
  cohort_characteristics.csv  median [IQR] of inputs, by outcome
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def proportion_confint(k, n, z=1.959964):
    """Wilson score 95% interval."""
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den; h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return c - h, c + h

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / "dt71_predictions.csv")
df["Preg"] = pd.to_numeric(df["Preg"], errors="coerce")
ok = df[df["dt_api_status"].eq("complete") | df["dt_timepoint_used"].eq("retrieval_fallback")]
rng = np.random.default_rng(0)

def boot_auc(y, p, B=2000):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    return np.percentile([roc_auc_score(y[i], p[i]) for i in
                          (np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(B))], [2.5, 97.5])

rows = []
for col, label in [("clinical_probability", "L7 headline per-transfer"),
                   ("befe_per_transfer", "L7 BEFE per-transfer"),
                   ("p_per_transfer", "L2 per-transfer"),
                   ("l1_l4_per_transfer", "L1-L4 per-transfer"),
                   ("p_kat_raw", "L3 KAT"), ("p_csdi", "L5 CSDI"), ("gnn_prob", "L6 graph"),
                   ("bayes_mean", "Bayes mean"), ("cycle_probability", "Cycle probability"),
                   ("retrieval_clinical_probability", "L7 headline, retrieval view (OCC/MII only)")]:
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
pm = pd.DataFrame(rows); pm.to_csv(HERE / "pregnancy_metrics.csv", index=False)

d = ok[["Preg", "clinical_probability"]].apply(pd.to_numeric, errors="coerce").dropna()
d["bin"] = pd.qcut(d["clinical_probability"].rank(method="first"), 5, labels=False) + 1
cb = []
for b, g in d.groupby("bin"):
    k, n = int(g["Preg"].sum()), len(g)
    lo, hi = proportion_confint(k, n)
    cb.append({"quintile": b, "n": n, "pred_min": g["clinical_probability"].min(), "pred_max": g["clinical_probability"].max(),
               "mean_pred": g["clinical_probability"].mean(), "observed": k / n, "obs_lo": lo, "obs_hi": hi})
pd.DataFrame(cb).to_csv(HERE / "calibration_bins.csv", index=False)

srows = []
for stage, med, lo, hi, real in [("2PN zygotes", "med_pn2", "p025_pn2", "p975_pn2", "two_pn"),
                                 ("Blastocysts", "med_blasts", "p025_blasts", "p975_blasts", "bl_total"),
                                 ("Good-quality blastocysts", "med_good", "p025_good", "p975_good", "bl_good")]:
    if not {med, lo, hi, real} <= set(ok.columns):
        continue
    s = ok[[med, lo, hi, real]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(s) < 10:
        continue
    bs = [spearmanr(s[med].to_numpy()[i], s[real].to_numpy()[i]).correlation
          for i in (rng.integers(0, len(s), len(s)) for _ in range(2000))]
    srows.append({"stage": stage, "model_knows": "OCC + MII (retrieval view)", "n": len(s),
                  "rho": spearmanr(s[med], s[real]).correlation,
                  "rho_lo": np.nanpercentile(bs, 2.5), "rho_hi": np.nanpercentile(bs, 97.5),
                  "coverage_95": float(((s[real] >= s[lo]) & (s[real] <= s[hi])).mean()),
                  "median_abs_error": float((s[med] - s[real]).abs().median())})
pd.DataFrame(srows).to_csv(HERE / "stage_agreement.csv", index=False)

cc = []
for c in ["age", "amh", "afc", "bmi", "attempt_no", "follicles", "occ", "mii", "two_pn", "bl_total", "bl_good", "transferred"]:
    if c not in df.columns:
        continue
    r = {"variable": c}
    for lab, g in [("all", df), ("pregnant", df[df.Preg == 1]), ("not_pregnant", df[df.Preg == 0])]:
        v = pd.to_numeric(g[c], errors="coerce").dropna()
        r[lab] = f"{v.median():.1f} [{v.quantile(.25):.1f}–{v.quantile(.75):.1f}] (n={len(v)})"
    cc.append(r)
pd.DataFrame(cc).to_csv(HERE / "cohort_characteristics.csv", index=False)

print(pm.round(3).to_string(index=False))
print(pd.DataFrame(cb).round(3).to_string(index=False))
print(pd.DataFrame(srows).round(3).to_string(index=False))
