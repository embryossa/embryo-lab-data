# -*- coding: utf-8 -*-
"""External-validity run: one context fit per (model, seed), reused for every
clinic query set. Writes raw per-row predictions.

  python run_clinics.py --ctx 2000 --seeds 42,7,13 --tabfm-n-est 4
"""
from __future__ import annotations
import argparse, os, time
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "true")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _local_paths import local_path  # noqa: E402

D = str(local_path("tabfm_workdir") / "data")
OUT = str(local_path("tabfm_workdir") / "preds")
os.makedirs(OUT, exist_ok=True)
TARGET = "Исход переноса"

QUERIES = {
    "B_fertimed_prod":   "panel_B_fertimed_prod.parquet",
    "B_fertimed_native": "panel_B_fertimed_native.parquet",
    "C_twoline_prod":    "panel_C_twoline_prod.parquet",
    "C_twoline_native":  "panel_C_twoline_native.parquet",
}

_TABFM = None


def tabfm_model():
    global _TABFM
    if _TABFM is None:
        from tabfm import tabfm_v1_0_0_pytorch as T
        _TABFM = T.load()
    return _TABFM


def load(fn):
    df = pd.read_parquet(f"{D}/{fn}")
    y = df.pop(TARGET).astype(int)
    return df.reset_index(drop=True), y.reset_index(drop=True)


def make_fitted(model, Xc, yc, seed, n_est):
    """Return (predict_fn, fit_seconds)."""
    t = time.time()
    if model == "tabfm":
        from tabfm import TabFMClassifier
        clf = TabFMClassifier(model=tabfm_model(), n_estimators=n_est, random_state=seed,
                              cache_context=True, maybe_quantize_kv_cache=False)
        clf.fit(Xc, np.asarray(yc, int))
        col = list(clf.classes_).index(1)
        fn = lambda Xq: clf.predict_proba(Xq)[:, col]
    elif model == "tabpfn":
        from tabpfn import TabPFNClassifier
        clf = TabPFNClassifier(device="cpu", fit_mode="fit_with_cache",
                               ignore_pretraining_limits=True, random_state=seed)
        clf.fit(Xc.values.astype("float32"), np.asarray(yc, int))
        col = list(clf.classes_).index(1)
        fn = lambda Xq: clf.predict_proba(Xq.values.astype("float32"))[:, col]
    elif model == "logreg":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=seed))
        clf.fit(Xc.values, np.asarray(yc, int))
        fn = lambda Xq: clf.predict_proba(Xq.values)[:, 1]
    elif model == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(random_state=seed)
        clf.fit(Xc.values, np.asarray(yc, int))
        fn = lambda Xq: clf.predict_proba(Xq.values)[:, 1]
    elif model == "prior":
        base = float(np.mean(yc))
        fn = lambda Xq: np.full(len(Xq), base)
    else:
        raise ValueError(model)
    return fn, time.time() - t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=2000)
    ap.add_argument("--seeds", type=str, default="42,7,13")
    ap.add_argument("--tabfm-n-est", type=int, default=4)
    ap.add_argument("--models", type=str, default="tabfm,tabpfn,logreg,hgb,prior")
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    models = a.models.split(",")

    XA, yA = load("panel_A_16k.parquet")
    Q = {k: load(v) for k, v in QUERIES.items()}
    print({k: v[0].shape for k, v in Q.items()}, flush=True)

    rows = []
    for seed in seeds:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=a.ctx, random_state=seed)
        ci, _ = next(sss.split(XA, yA))
        Xc, yc = XA.iloc[ci].reset_index(drop=True), yA.iloc[ci].reset_index(drop=True)
        for m in models:
            fn, t_fit = make_fitted(m, Xc, yc, seed, a.tabfm_n_est)
            print(f"[seed {seed}] {m:7s} fit {t_fit:7.1f}s", flush=True)
            for qname, (Xq, yq) in Q.items():
                t = time.time(); p = np.clip(fn(Xq), 0, 1); t_pred = time.time() - t
                rows.append(pd.DataFrame({
                    "query": qname, "ctx_seed": seed, "model": m,
                    "row": np.arange(len(yq)), "y": yq.values, "p": p,
                    "fit_seconds": t_fit, "pred_seconds": t_pred}))
                print(f"    -> {qname:20s} {t_pred:6.1f}s mean_p={p.mean():.4f}", flush=True)
            pd.concat(rows).to_parquet(f"{OUT}/clinics_ctx{a.ctx}{a.tag}.parquet")
    print("DONE", flush=True)
