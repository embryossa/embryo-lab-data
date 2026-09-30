#!/usr/bin/env python3
"""Data-quality, anomaly and distribution analyses for the CSBJ manuscript (aggregates only).

Inputs (no identifiers are read or written):
  A. data/processed/reference/rbmo_prospective_2026/rbmo_prospective_cohort.csv  (1,026 cycles, centers A-D;
     DT 7.0-era outputs 'DIGITAL TWIN' and TabPFN 'PRAI', inputs, observed counts)
  B. reports/publication_60_dt71/dt71_predictions.csv (center D, 59 transfers, DT 7.1 outputs)
  C. center D laboratory population from data/inbox/publication_60/OPU table for publication.xlsx via the
     unchanged embryo-lab-data canonical frame and competency mask (only numeric stage columns used)
  D. data/processed/miri_summary.csv + miri_events.csv (culture-window incubator logs of the review-list cycles)
Methods: IQR rule (outside Q1-1.5 IQR, Q3+1.5 IQR), |z|>3, rule-based plausibility (numerator > denominator
along the cascade), Shapiro-Wilk, one- and two-sample Kolmogorov-Smirnov.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
out = {}

# ---------------- A. four-center cohort
A = pd.read_csv(ROOT / "data/processed/reference/rbmo_prospective_2026/rbmo_prospective_cohort.csv",
                encoding="utf-8-sig", low_memory=False)
A["centre_code"] = A["centre"].map({"Fertimed": "A", "GGRC": "B", "SkyFert": "C", "Two-Line": "D"})
num = {"age": "age", "follicles": "N folicules OPU", "occ": "OCC", "mii": "MII", "two_pn": "real_pn2",
       "cleavage": "real_cleav", "blastocysts": "real_bl", "good_bl": "real_goodbl", "amh": "amh", "afc": "afc", "bmi": "bmi"}
for k, c in num.items():
    A[k] = pd.to_numeric(A[c], errors="coerce")
rows = []
for cen, g in list(A.groupby("centre_code")) + [("All", A)]:
    for k in num:
        x = g[k].dropna()
        if len(x) < 5:
            rows.append({"centre": cen, "variable": k, "n": len(x), "missing": int(g[k].isna().sum())}); continue
        q1, q3 = x.quantile([.25, .75]); iqr = q3 - q1
        iqr_out = ((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum()
        z_out = (np.abs(stats.zscore(x)) > 3).sum() if x.std() > 0 else 0
        rows.append({"centre": cen, "variable": k, "n": len(x), "missing": int(g[k].isna().sum()),
                     "median": x.median(), "q1": q1, "q3": q3, "iqr_outliers": int(iqr_out), "z3_outliers": int(z_out),
                     "mode_share": float((x == x.mode().iloc[0]).mean()), "mode": float(x.mode().iloc[0])})
anom = pd.DataFrame(rows); anom.to_csv(HERE / "anomalies_four_centre.csv", index=False)
plaus = []
for cen, g in list(A.groupby("centre_code")) + [("All", A)]:
    def cnt(a, b):
        d = g[[a, b]].dropna(); return int((d[a] > d[b]).sum()), int(len(d))
    r = {"centre": cen, "n": len(g)}
    for name, (a, b) in {"MII>OCC": ("mii", "occ"), "2PN>MII": ("two_pn", "mii"), "cleavage>2PN": ("cleavage", "two_pn"),
                         "blastocysts>2PN": ("blastocysts", "two_pn"), "good>blastocysts": ("good_bl", "blastocysts")}.items():
        r[name], r[name + "_n"] = cnt(a, b)
    r["amh_default_2.0"] = int((g["amh"] == 2.0).sum()); r["afc_default_12"] = int((g["afc"] == 12).sum())
    r["bmi_default_23_or_24"] = int(g["bmi"].isin([23.0, 24.0]).sum())
    r["blastocyst_missing"] = int(g["blastocysts"].isna().sum())
    plaus.append(r)
plaus = pd.DataFrame(plaus); plaus.to_csv(HERE / "plausibility_four_centre.csv", index=False)

dist = []
A["preg"] = pd.to_numeric(A["Preg"], errors="coerce")
for model, col in {"Digital Twin (combined)": "DIGITAL TWIN", "TabPFN": "PRAI"}.items():
    A[col] = pd.to_numeric(A[col], errors="coerce")
    A.loc[A[col] > 1, col] = A.loc[A[col] > 1, col] / 100.0
    for cen, g in list(A.groupby("centre_code")) + [("All", A)]:
        x = g[col].dropna()
        sw = stats.shapiro(x) if 3 <= len(x) <= 5000 else (np.nan, np.nan)
        rest = A.loc[A["centre_code"] != cen, col].dropna() if cen != "All" else None
        ks_rest = stats.ks_2samp(x, rest) if rest is not None and len(rest) else None
        p1, p0 = g.loc[g["preg"] == 1, col].dropna(), g.loc[g["preg"] == 0, col].dropna()
        ks_out = stats.ks_2samp(p1, p0) if len(p1) > 2 and len(p0) > 2 else None
        dist.append({"model": model, "centre": cen, "n": len(x), "mean": x.mean(), "sd": x.std(), "median": x.median(),
                     "skew": stats.skew(x), "shapiro_W": sw[0], "shapiro_p": sw[1],
                     "ks_vs_other_centres_D": ks_rest.statistic if ks_rest else np.nan,
                     "ks_vs_other_centres_p": ks_rest.pvalue if ks_rest else np.nan,
                     "ks_preg_vs_not_D": ks_out.statistic if ks_out else np.nan,
                     "ks_preg_vs_not_p": ks_out.pvalue if ks_out else np.nan})
pd.DataFrame(dist).to_csv(HERE / "distributions_four_centre.csv", index=False)
# input distributions: KS of each centre vs the others for key inputs
ksin = []
for k in ["age", "follicles", "occ", "mii", "two_pn"]:
    for cen, g in A.groupby("centre_code"):
        x, y = g[k].dropna(), A.loc[A["centre_code"] != cen, k].dropna()
        if len(x) > 5:
            r = stats.ks_2samp(x, y); sw = stats.shapiro(x)
            ksin.append({"variable": k, "centre": cen, "n": len(x), "ks_D": r.statistic, "ks_p": r.pvalue,
                         "shapiro_W": sw[0], "shapiro_p": sw[1]})
pd.DataFrame(ksin).to_csv(HERE / "input_distributions_four_centre.csv", index=False)

# ---------------- B. center D, DT 7.1 run
B = pd.read_csv(ROOT / "reports/publication_60_dt71/dt71_predictions.csv", encoding="utf-8-sig", low_memory=False)
bd = []
for col in ["clinical_probability", "befe_per_transfer", "p_kat_raw", "p_csdi", "gnn_prob", "p_per_transfer"]:
    if col not in B: continue
    x = pd.to_numeric(B[col], errors="coerce").dropna(); y = pd.to_numeric(B["Preg"], errors="coerce")
    p1 = pd.to_numeric(B.loc[y == 1, col], errors="coerce").dropna(); p0 = pd.to_numeric(B.loc[y == 0, col], errors="coerce").dropna()
    sw = stats.shapiro(x); ks1 = stats.kstest((x - x.mean()) / x.std(), "norm"); ks2 = stats.ks_2samp(p1, p0)
    bd.append({"output": col, "n": len(x), "mean": x.mean(), "sd": x.std(), "skew": stats.skew(x),
               "shapiro_W": sw.statistic, "shapiro_p": sw.pvalue, "ks_normal_D": ks1.statistic, "ks_normal_p": ks1.pvalue,
               "ks_preg_vs_not_D": ks2.statistic, "ks_preg_vs_not_p": ks2.pvalue,
               "mean_preg": p1.mean(), "mean_not": p0.mean(), "mannwhitney_p": stats.mannwhitneyu(p1, p0).pvalue,
               "iqr_outliers": int(((x < x.quantile(.25) - 1.5 * (x.quantile(.75) - x.quantile(.25))) |
                                    (x > x.quantile(.75) + 1.5 * (x.quantile(.75) - x.quantile(.25)))).sum()),
               "zero_probability": int((x == 0).sum())})
pd.DataFrame(bd).to_csv(HERE / "distributions_centerD_dt71.csv", index=False)

# ---------------- C. center D laboratory population (stage columns only)
import run_analysis as ra
from embryo import ingest, kpi
cfg = ra._settings(ROOT / "config"); cc = ingest.load_config(str(ROOT / "config" / "columns.yaml"))
_, fr, _ = ra._normalise_source(ROOT / "data/inbox/publication_60/OPU table for publication.xlsx", cc, cfg)
lab = fr.loc[~kpi.competency_exclusion_mask(fr, cc)]
cols = ["age", "follicles", "occ", "mii", "inseminated", "two_pn", "cleavage_d3", "bl_total", "bl_good"]
L = lab[cols].apply(pd.to_numeric, errors="coerce")
L["FR"] = L["two_pn"] / L["mii"]; L["ClR"] = L["cleavage_d3"] / L["two_pn"]; L["BlR"] = L["bl_total"] / L["two_pn"]
L["MIIR"] = L["mii"] / L["occ"]; L["OCCR"] = L["occ"] / L["follicles"]
cd = []
for k in cols + ["FR", "ClR", "BlR", "MIIR", "OCCR"]:
    x = L[k].replace([np.inf, -np.inf], np.nan).dropna()
    q1, q3 = x.quantile([.25, .75]); iqr = q3 - q1
    sw = stats.shapiro(x) if len(x) >= 3 else (np.nan, np.nan)
    cd.append({"variable": k, "n": len(x), "median": x.median(), "q1": q1, "q3": q3,
               "iqr_outliers": int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum()),
               "z3_outliers": int((np.abs(stats.zscore(x)) > 3).sum()) if x.std() > 0 else 0,
               "shapiro_W": sw[0], "shapiro_p": sw[1], "gt_one": int((x > 1).sum()) if k in ("FR", "ClR", "BlR", "MIIR", "OCCR") else None})
pd.DataFrame(cd).to_csv(HERE / "anomalies_centerD_lab.csv", index=False)
oow = L[["cleavage_d3", "two_pn"]].dropna()
out["centerD_lab"] = {"cycles": int(len(lab)), "cleavage_gt_2pn": int((oow.cleavage_d3 > oow.two_pn).sum()),
                      "with_both": int(len(oow)),
                      "excess_cleaved_embryos": int((oow.cleavage_d3 - oow.two_pn).clip(lower=0).sum()),
                      "zero_2pn": int((L["two_pn"] == 0).sum()), "zero_bl": int((L["bl_total"] == 0).sum())}

# ---------------- D. incubator logs
ms = pd.read_csv(ROOT / "data/processed/miri_summary.csv", encoding="utf-8-sig")
me = pd.read_csv(ROOT / "data/processed/miri_events.csv", encoding="utf-8-sig")
has_logs = ms["n_points"].fillna(0) > 0
out["miri"] = {"cycles": int(len(ms)), "with_logs": int(has_logs.sum()),
               "max_severity": ms["max_severity"].value_counts(dropna=False).to_dict(),
               "with_logs_warning_or_critical": int((has_logs & ms["max_severity"].isin(["warning", "critical"])).sum()),
               "gap_events_with_logs": int(ms.loc[has_logs, "gap_count"].fillna(0).sum()),
               "western_electric_rule1_per_cycle_median": float(ms.loc[has_logs, "western_electric_rule1"].dropna().map(
                   lambda s: sum(__import__("ast").literal_eval(s).values())).median()),
               "lid_open_events_per_cycle_median": float(pd.to_numeric(ms.loc[has_logs, "lid_open_events"], errors="coerce").median())
                   if pd.to_numeric(ms.loc[has_logs, "lid_open_events"], errors="coerce").notna().any() else None,
               "event_types": me["event_type"].value_counts().to_dict(), "sensors": me["sensor"].value_counts().to_dict(),
               "severity": me["severity"].value_counts().to_dict(),
               "gap_reason": me["gap_reason"].value_counts(dropna=True).to_dict() if "gap_reason" in me else {},
               "event_duration_median_min": float(me["duration_minutes"].median()),
               "event_duration_p90_min": float(me["duration_minutes"].quantile(.9))}
(HERE / "data_quality_summary.json").write_text(json.dumps(out, indent=2, default=float, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=1, default=float, ensure_ascii=False))
print(plaus.to_string(index=False))
print(anom[anom.centre == "All"].round(2).to_string(index=False))
print(pd.DataFrame(dist).round(4).to_string(index=False))
print(pd.DataFrame(bd).round(4).to_string(index=False))
print(pd.DataFrame(cd).round(3).to_string(index=False))
print(pd.DataFrame(ksin).round(4).to_string(index=False))
