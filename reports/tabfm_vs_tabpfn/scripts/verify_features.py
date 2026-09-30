# -*- coding: utf-8 -*-
"""Sanity check: reproduce the pipeline's own TabPFN numbers for the Two-Line
run from my independently rebuilt 18-feature matrix.

If my feature construction matches mc_features._build_features, running TabPFN
under the production context recipe (max_n=5000, balanced, seed=42) must land
on essentially the same probabilities as data/processed/tabpfn_predictions.csv.
"""
from __future__ import annotations
import os
import numpy as np, pandas as pd
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "true")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _local_paths import REPO_ROOT, local_path  # noqa: E402

D = str(local_path("tabfm_workdir") / "data")
TARGET = "Исход переноса"
REF = str(REPO_ROOT / "data/processed/tabpfn_predictions.csv")
ENR = str(REPO_ROOT / "data/processed/OPU_table_enriched.xlsx")

# ── production context recipe (tabpfn_backend.load_context) ──────────────────
A = pd.read_parquet(f"{D}/panel_A_16k.parquet")
yA = A.pop(TARGET).astype(int)
max_n, seed = 5000, 42
rng = np.random.default_rng(seed)
n_full = len(yA)
idx = []
for cls in np.unique(yA):
    cls_idx = np.where(yA.values == cls)[0]
    take = max(1, min(int(round(max_n * len(cls_idx) / n_full)), len(cls_idx)))
    idx.append(rng.choice(cls_idx, take, replace=False))
sel = np.concatenate(idx)
rng.shuffle(sel)
Xc = A.iloc[sel].values.astype("float32")
yc = yA.iloc[sel].values.astype(int)
print("context", Xc.shape, "pos_rate", yc.mean().round(4), flush=True)

C = pd.read_parquet(f"{D}/panel_C_twoline_prod.parquet")
yC = C.pop(TARGET).astype(int)

from tabpfn import TabPFNClassifier
clf = TabPFNClassifier(device="cpu", fit_mode="fit_with_cache",
                       ignore_pretraining_limits=True)
clf.fit(Xc, yc)
p = clf.predict_proba(C.values.astype("float32"))[:, list(clf.classes_).index(1)]
print("mine mean", round(float(p.mean()), 4), flush=True)

# align to the pipeline output via the enriched table's _key order
enr = pd.read_excel(ENR)
ref = pd.read_csv(REF)
keep_mask = pd.read_parquet(f"{D}/panel_C_twoline_prod.parquet").index  # positional
mine = pd.DataFrame({"p_mine": p})

ref_map = dict(zip(ref.patient_id.astype(str), ref.p_tabpfn))
# rebuild the same keep mask used in prep.py
import prep
y_b = prep.binarize(enr["Preg"])
def n(c): return pd.to_numeric(enr[c], errors="coerce")
mii_eff = n("MII").where(n("MII").notna(), n("Inseminated"))
core = pd.concat([n("Age"), n("# Attempt"), n("N folicules OPU"), n("OCC"), mii_eff, n("2pN")], axis=1)
keep = y_b.notna() & core.notna().all(axis=1) & (n("ET").fillna(0) > 0)
keys = enr.loc[keep, "_key"].astype(str).values
mine["_key"] = keys
mine["p_pipeline"] = [ref_map.get(k, np.nan) for k in keys]
ok = mine.p_pipeline.notna()
print(f"matched {ok.sum()}/{len(mine)} rows against the pipeline output", flush=True)
if ok.sum():
    d = (mine.p_mine[ok] - mine.p_pipeline[ok]).abs()
    print(f"|Δp| mean={d.mean():.4f} max={d.max():.4f} corr={np.corrcoef(mine.p_mine[ok], mine.p_pipeline[ok])[0,1]:.4f}", flush=True)
    print(mine.loc[ok, ["p_mine", "p_pipeline"]].head(10).round(4).to_string(), flush=True)
mine.to_csv(local_path("tabfm_workdir") / "verify_features.csv", index=False, encoding="utf-8")
