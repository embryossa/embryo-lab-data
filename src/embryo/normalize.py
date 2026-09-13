"""
normalize.py — детерминированная нормализация (CLAUDE.md §3).
  * Исход -> бинарь. Preg=1 = клиническая беременность ИЛИ утвердительное да/yes.
    Всё прочее заполненное -> 0. Пусто -> NaN (строка отсеивается фильтром).
  * AMH: реальное значение в приоритете; иначе оценка age+AFC+OCC (флаг amh_imputed).
  * BMI: при отсутствии константа 24.0 (флаг bmi_imputed).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _norm(x) -> str:
    return str(x).strip().lower() if pd.notna(x) else ""


def parse_dates(values) -> pd.Series:
    """Разобрать даты, встречающиеся в этом пайплайне ДВУХ разных видов.

    Сырые выгрузки МИС дают "дд.мм.гггг" — для них нужен ``dayfirst=True``.
    Но всё, что пайплайн уже записал и читает обратно (dt_predictions.csv,
    regression_qc.csv, OPU_table_enriched.xlsx), приходит в ISO ``ГГГГ-ММ-ДД``,
    а ``dayfirst=True`` переставляет местами день и месяц И В НЁМ: pandas 3.0.3
    разбирает "2026-01-05" как 1 мая. ``format="mixed"`` от этого не спасает —
    проверено на pandas 3.0.3. Молча, без исключения: страдает каждая дата с
    днём ≤ 12, то есть примерно 40% строк.

    Поэтому однозначный ISO разбирается ПЕРВЫМ, и только на том, что им не
    разобралось, применяется dayfirst. Тот же приём уже описан в
    ``pgt._parse_cohort_date`` — здесь он вынесен в одно место на весь пакет.
    """
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    iso = pd.to_datetime(series, errors="coerce", format="ISO8601")
    if iso.notna().all():
        return iso
    fallback = pd.to_datetime(series, errors="coerce", dayfirst=True, format="mixed")
    return iso.fillna(fallback)


def outcome_to_binary(series: pd.Series, outcome_cfg: dict) -> pd.Series:
    aff = set(_norm(m) for m in outcome_cfg.get("affirmative_exact", []))
    neg = set(_norm(m) for m in outcome_cfg.get("negative_exact", []))
    pos_sub = [m.lower() for m in outcome_cfg.get("positive_substrings", [])]
    neg_sub = [m.lower() for m in outcome_cfg.get("negative_substrings", [])]
    num_pos = set(outcome_cfg.get("numeric_positive", [1]))
    num_neg = set(outcome_cfg.get("numeric_negative", [0]))

    def classify(v):
        if pd.isna(v) or _norm(v) == "":
            return np.nan
        try:                                   # числовой путь
            fv = float(v)
            if fv in num_pos:
                return 1
            if fv in num_neg:
                return 0
        except (ValueError, TypeError):
            pass
        s = _norm(v)
        if s in aff:
            return 1
        if s in neg:
            return 0
        if any(sub in s for sub in neg_sub):   # негативы проверяем ПЕРЕД позитивами
            return 0
        if any(sub in s for sub in pos_sub):
            return 1
        return 0

    return series.apply(classify)


def impute_amh(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = cfg["amh_impute"]
    df = df.copy()
    if "amh" not in df.columns:
        df["amh"] = np.nan
    df["amh"] = pd.to_numeric(df["amh"], errors="coerce")
    df["amh_imputed"] = df["amh"].isna()

    def _scalar(v, default):
        v = pd.to_numeric(pd.Series([v]).iloc[0], errors="coerce")
        return default if pd.isna(v) else float(v)

    def est(row):
        afc_raw = row.get("afc")
        if afc_raw is None or pd.isna(pd.Series([afc_raw]).iloc[0]):
            afc_raw = row.get("follicles")           # fallback: фолликулы, если АФК нет
        afc = _scalar(afc_raw, 0)
        occ = _scalar(row.get("occ"), 0)
        age = _scalar(row.get("age"), 35)
        val = (c["a_afc"] * afc + c["b_occ"] * occ
               - c["c_age"] * max(0.0, age - 35) + c["intercept"])
        return float(np.clip(val, c["floor"], c["ceil"]))

    mask = df["amh"].isna()
    df.loc[mask, "amh"] = df.loc[mask].apply(est, axis=1)
    return df


def impute_bmi(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    const = cfg["bmi_impute"]["constant"]
    if "bmi" not in df.columns:
        df["bmi"] = np.nan
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    df["bmi_imputed"] = df["bmi"].isna()
    df.loc[df["bmi"].isna(), "bmi"] = const
    return df


def normalize(df: pd.DataFrame, columns_cfg: dict, settings_cfg: dict) -> pd.DataFrame:
    df = df.copy()
    if "transfer_outcome" in df.columns:
        df["Preg"] = outcome_to_binary(df["transfer_outcome"], columns_cfg["outcome"])
    df = impute_amh(df, settings_cfg)
    df = impute_bmi(df, settings_cfg)
    return df
