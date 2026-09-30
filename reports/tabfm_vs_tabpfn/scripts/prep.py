# -*- coding: utf-8 -*-
"""Build the three comparison panels. Standalone: reads sources read-only,
writes only into <tabfm_workdir>/data/ (see reports/_local_paths.py).

Panel A  : 16k development cohort (all_df_with_KPI.xlsx) -- native columns.
Panel B  : Fertimed, first 6 months of 2026 (raw 1C export, RU schema).
Panel C  : Two-Line, last run cohort (OPU_table_enriched.xlsx, EN schema).

Panels B/C are built with the production inference convention
(mc_features._build_features), i.e. exactly the 18-vector the pipeline feeds
TabPFN today. Panel B additionally gets a "native" variant that uses the real
day-3 / day-5 / frozen / transferred columns instead of the substitutions.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _local_paths import REPO_ROOT, local_path  # noqa: E402

OUT = str(local_path("tabfm_workdir") / "data")
os.makedirs(OUT, exist_ok=True)

CTX_PATH = str(local_path("tabpfn_context"))
FERT_PATH = str(REPO_ROOT / "data/inbox/fertimed/6 месяцев 2026_txtoutcome_cleaned.xlsx")
TWOLINE_PATH = str(REPO_ROOT / "data/processed/OPU_table_enriched.xlsx")
TARGET = "Исход переноса"

NN_FEATURE_NAMES = [
    "Возраст", "№ попытки", "Количество фолликулов", "Число ОКК",
    "Число инсеминированных", "2 pN", "Число дробящихся на 3 день",
    "Число Bl", "Число Bl хор.кач-ва", "Частота оплодотворения",
    "Частота дробления", "Частота формирования бластоцист",
    "Частота формирования бластоцист хорошего качества",
    "Частота получения ОКК", "Число эмбрионов 5 дня",
    "Заморожено эмбрионов", "Перенесено эмбрионов", "KPIScore",
]


def _kpi_score(age, follicles, mii, fert_rate, good):
    age = np.asarray(age, dtype=float)
    follicles = np.asarray(follicles, dtype=float)
    a = np.where(age >= 40, 1, np.where(age <= 36, 5, 3))
    b = np.where(follicles > 15, 5, np.where(follicles >= 8, 3, 1))
    c = np.where(mii <= 3, 1, np.where(mii <= 7, 3, 5))
    d = np.where(fert_rate < 0.50, 1, np.where(fert_rate <= 0.65, 3, 5))
    e = np.where(good == 0, 1, np.where(good <= 2, 3, 5))
    return (a + b + c + d + e).astype(float)


def build_features(age, attempt, follicles, okk, mii, pn2, blasts, good,
                   day3=None, day5=None, frozen=None, transferred=None):
    """Vectorised copy of mc_features._build_features.

    day3/day5/frozen/transferred default to the production substitutions
    (pn2, pn2, good, 1). Passing them explicitly gives the "native" variant.
    """
    age = np.asarray(age, dtype=float)
    attempt = np.asarray(attempt, dtype=float)
    follicles = np.asarray(follicles, dtype=float)
    okk = np.asarray(okk, dtype=float)
    mii = np.asarray(mii, dtype=float)
    pn2 = np.asarray(pn2, dtype=float)
    blasts = np.asarray(blasts, dtype=float)
    good = np.asarray(good, dtype=float)

    fert_rate = np.where(mii > 0, pn2 / np.maximum(mii, 1), 0.0)
    cleav_rate = np.where(pn2 > 0, 1.0, 0.0)
    blast_rate = np.where(pn2 > 0, blasts / np.maximum(pn2, 1), 0.0)
    good_rate = np.where(pn2 > 0, good / np.maximum(pn2, 1), 0.0)
    okk_rate = okk / np.maximum(follicles, 1)
    kpi = _kpi_score(age, follicles, mii, fert_rate, good)

    return pd.DataFrame({
        "Возраст": age,
        "№ попытки": attempt,
        "Количество фолликулов": follicles,
        "Число ОКК": okk,
        "Число инсеминированных": mii,
        "2 pN": pn2,
        "Число дробящихся на 3 день": pn2 if day3 is None else np.asarray(day3, float),
        "Число Bl": blasts,
        "Число Bl хор.кач-ва": good,
        "Частота оплодотворения": fert_rate,
        "Частота дробления": cleav_rate,
        "Частота формирования бластоцист": blast_rate,
        "Частота формирования бластоцист хорошего качества": good_rate,
        "Частота получения ОКК": okk_rate,
        "Число эмбрионов 5 дня": pn2 if day5 is None else np.asarray(day5, float),
        "Заморожено эмбрионов": good if frozen is None else np.asarray(frozen, float),
        "Перенесено эмбрионов": np.ones_like(age) if transferred is None
                                 else np.asarray(transferred, float),
        "KPIScore": kpi,
    })[NN_FEATURE_NAMES]


NEG_SUB = ["не наступила", "биохимическая", "внематочная", "замершая", "no pregnancy"]
POS_SUB = ["клиническая беременность", "беременность клиническая", "беременность +28",
           "clinical pregnancy", "беременность наступила"]
AFF_EXACT = {"да", "yes", "y", "+"}
NEG_EXACT = {"нет", "no", "n", "-", "неизвестно"}


def binarize(series: pd.Series) -> pd.Series:
    out = []
    for v in series:
        if pd.isna(v):
            out.append(np.nan); continue
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            f = float(v)
            out.append(1.0 if f == 1 else (0.0 if f == 0 else np.nan)); continue
        s = str(v).strip().lower()
        if not s:
            out.append(np.nan); continue
        if s in AFF_EXACT: out.append(1.0); continue
        if s in NEG_EXACT: out.append(0.0); continue
        if any(t in s for t in NEG_SUB): out.append(0.0); continue
        if any(t in s for t in POS_SUB): out.append(1.0); continue
        out.append(0.0)
    return pd.Series(out, index=series.index, dtype=float)


def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)


# ── Panel A ──────────────────────────────────────────────────────────────────
def panel_a():
    df = pd.read_excel(CTX_PATH)
    miss = [c for c in NN_FEATURE_NAMES if c not in df.columns]
    assert not miss, miss
    X = df[NN_FEATURE_NAMES].replace([np.inf, -np.inf], np.nan)
    f32max = np.finfo(np.float32).max
    X = X.mask(X.abs() > f32max)
    X = X.fillna(X.median(numeric_only=True))
    y = pd.to_numeric(df[TARGET], errors="coerce")
    keep = y.notna()
    X, y = X[keep].reset_index(drop=True), y[keep].astype(int).reset_index(drop=True)
    X.assign(**{TARGET: y}).to_parquet(f"{OUT}/panel_A_16k.parquet")
    print(f"[A] 16k cohort: {X.shape}, pos_rate={y.mean():.4f}")
    return X, y


# ── Panel B ──────────────────────────────────────────────────────────────────
def panel_b():
    df = pd.read_excel(FERT_PATH)
    y = binarize(df[TARGET])
    okk = _num(df, "Число ОКК")
    mii = _num(df, "Число MII")
    ins = _num(df, "Число инсеминированных")
    # production rule: MII used when actually known, else Inseminated
    mii_eff = mii.where(mii.notna(), ins)
    pn2 = _num(df, "2 pN")
    bl = _num(df, "Число Bl")
    good = _num(df, "Число Bl хор.кач-ва")
    age = _num(df, "Возраст")
    att = _num(df, "№ попытки")
    foll = _num(df, "Количество фолликулов")
    d3 = _num(df, "Число дробящихся на 3 день")
    d5 = _num(df, "Число эмбрионов 5 дня")
    fro = _num(df, "Заморожено эмбрионов")
    tr = _num(df, "Перенесено эмбрионов")

    # zero is an observation, NaN is not: only counts that are genuinely absent
    # get filled, and rows lacking a core count or the label are dropped.
    core = pd.concat([age, att, foll, okk, mii_eff, pn2], axis=1)
    keep = y.notna() & core.notna().all(axis=1) & (tr.fillna(0) > 0)
    bl = bl.fillna(0.0); good = good.fillna(0.0)
    d3 = d3.fillna(pn2); d5 = d5.fillna(pn2); fro = fro.fillna(0.0)

    Xp = build_features(age, att, foll, okk, mii_eff, pn2, bl, good)[keep.values]
    Xn = build_features(age, att, foll, okk, mii_eff, pn2, bl, good,
                        day3=d3, day5=d5, frozen=fro, transferred=tr)[keep.values]
    yy = y[keep].astype(int)
    Xp.reset_index(drop=True).assign(**{TARGET: yy.values}).to_parquet(f"{OUT}/panel_B_fertimed_prod.parquet")
    Xn.reset_index(drop=True).assign(**{TARGET: yy.values}).to_parquet(f"{OUT}/panel_B_fertimed_native.parquet")
    print(f"[B] Fertimed 6m: {Xp.shape} of {len(df)} raw, pos_rate={yy.mean():.4f}")
    return Xp, yy


# ── Panel C ──────────────────────────────────────────────────────────────────
def panel_c():
    df = pd.read_excel(TWOLINE_PATH)
    y = binarize(df["Preg"])
    okk = _num(df, "OCC")
    mii = _num(df, "MII")
    ins = _num(df, "Inseminated")
    mii_eff = mii.where(mii.notna(), ins)
    pn2 = _num(df, "2pN")
    bl = _num(df, "Bl")
    good = _num(df, "Good Bl")
    age = _num(df, "Age")
    att = _num(df, "# Attempt")
    foll = _num(df, "N folicules OPU")
    d3 = _num(df, "Cleavage")
    fro = _num(df, "Cryo")
    tr = _num(df, "ET")

    core = pd.concat([age, att, foll, okk, mii_eff, pn2], axis=1)
    keep = y.notna() & core.notna().all(axis=1) & (tr.fillna(0) > 0)
    bl = bl.fillna(0.0); good = good.fillna(0.0)
    d3 = d3.fillna(pn2); fro = fro.fillna(0.0)

    Xp = build_features(age, att, foll, okk, mii_eff, pn2, bl, good)[keep.values]
    Xn = build_features(age, att, foll, okk, mii_eff, pn2, bl, good,
                        day3=d3, day5=pn2, frozen=fro, transferred=tr)[keep.values]
    yy = y[keep].astype(int)
    Xp.reset_index(drop=True).assign(**{TARGET: yy.values}).to_parquet(f"{OUT}/panel_C_twoline_prod.parquet")
    Xn.reset_index(drop=True).assign(**{TARGET: yy.values}).to_parquet(f"{OUT}/panel_C_twoline_native.parquet")
    print(f"[C] Two-Line last run: {Xp.shape} of {len(df)} rows, pos_rate={yy.mean():.4f}")
    return Xp, yy


if __name__ == "__main__":
    XA, yA = panel_a()
    XB, yB = panel_b()
    XC, yC = panel_c()

    # leakage probe: exact 18-vector duplicates between clinic panels and the context
    key_a = set(map(tuple, np.round(XA.values.astype(float), 6)))
    for name, X in (("B/Fertimed", XB), ("C/Two-Line", XC)):
        k = list(map(tuple, np.round(X.values.astype(float), 6)))
        dup = sum(1 for t in k if t in key_a)
        print(f"[leak] {name}: {dup}/{len(k)} rows have an exact 18-vector twin in the 16k context")
