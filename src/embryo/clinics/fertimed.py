"""
fertimed.py — правила подготовки выгрузки МИС 1С клиники Fertimed к
канонической форме, которую уже понимают src/embryo/kpi.py,
scripts/run_analysis.py и внешние ранеры DT/TabPFN/DNN.

Правила согласованы с владельцем 2026-07-18 (обоснование каждого пункта —
docs/FERTIMED.md; здесь только код, не менять правила без обновления доки).

Одна строка выгрузки = один цикл/попытка. Связывание строк одного пациента
между свежим забором и последующим криопереносом — задача МИС/её инженеров,
не этого модуля: криоперенос-строка уже содержит данные протокола
оплодотворения, перенесённые 1С автоматически.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

from .. import ingest, normalize

# Значение, которое scripts/run_analysis.py подставляет вместо канонического
# "mii" при построении входа для DT/TabPFN (и только для него — лабораторный
# KPI MIIR всегда считается по настоящему "mii", см. docs/FERTIMED.md).
MII_ML_FALLBACK_COLUMN = "mii_ml_fallback"
OCC_ML_FALLBACK_COLUMN = "occ_ml_fallback"
FOLLICLES_ML_FALLBACK_COLUMN = "follicles_ml_fallback"

# Владелец подтвердил (2026-07-18): для анализа берём только эти значения
# "Лечение"; пустые строки, "Только пункция", "Инсеминация" (ВМИ — другая
# модальность) и "ПГТ на размороженных эмбрионах" отбрасываются.
_ALLOWED_TREATMENT = {
    "криоперенос", "икси", "эко", "эко + икси", "получение як (реципиент)",
}

# Явно не используются в этом отчёте (см. docs/FERTIMED.md): второе/третье
# лицо в строке (сурмама, партнёр, доноры), ненадёжные/неверные поля (SMF,
# КАФ) и данные родов/осложнений вне области клинической беременности.
_DROP_RAW_COLUMNS = [
    "ФИО Сурмамы", "Партнёр",
    "Fresh OD", "fresh OD (код донора)", "Vitrified OD", "vitrified OD (код донора)",
    "ДС", "ДС (код донора)", "ДЭ", "ДЭ (Код донора)",
    "SMF", "FET", "PGT-TVP",
    "КАФ",
    "Внематочная", "Количество плодных яиц", "Роды Да/Нет", "Срок родов",
    "Метод родоразрешения", "Количество детей", "Комментарий по исходу беременности",
]

_AGE_RE = re.compile(r"(\d+)")
_POSITIVE_OUTCOME_MARKER = "+28"
_NEGATIVE_OUTCOME_MARKERS = ("не наступила", "биохимическая")


def _parse_age(value) -> float:
    """"34 года" / "26 лет" / "31 год" -> 34.0 / 26.0 / 31.0."""
    if pd.isna(value):
        return np.nan
    m = _AGE_RE.search(str(value))
    return float(m.group(1)) if m else np.nan


def _fillna_default(series: pd.Series, default: float) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _is_yes(value) -> bool:
    return pd.notna(value) and str(value).strip().lower() == "да"


def _fertilization_method(df: pd.DataFrame) -> pd.Series:
    """Из флагов IVF/ICSI, НЕ из "Лечение" (владелец: ненадёжен для метода)."""
    icsi = df["icsi_flag"].apply(_is_yes) if "icsi_flag" in df.columns else pd.Series(False, index=df.index)
    ivf = df["ivf_flag"].apply(_is_yes) if "ivf_flag" in df.columns else pd.Series(False, index=df.index)
    method = pd.Series(np.nan, index=df.index, dtype="object")
    method[ivf & icsi] = "IVF+ICSI"
    method[icsi & ~ivf] = "ICSI"
    method[ivf & ~icsi] = "IVF"
    return method


def _mii_ml_fallback(df: pd.DataFrame) -> pd.Series:
    """Только для входа нейросетей: ЭКО-циклы без известного MII получают
    MII=Число инсеминированных, чтобы не выпасть из прогноза. Лабораторный
    MIIR эту колонку не использует — считается по настоящему "mii"."""
    mii = pd.to_numeric(df.get("mii"), errors="coerce")
    inseminated = pd.to_numeric(df.get("inseminated"), errors="coerce")
    is_ivf = df["fertilization_method"].eq("IVF") if "fertilization_method" in df.columns else pd.Series(False, index=df.index)
    return mii.where(mii.notna() | ~is_ivf, inseminated)


def _occ_ml_fallback(df: pd.DataFrame) -> pd.Series:
    """Только для входа нейросетей: неизвестное число ОКК получает модельный MII
    (звено цепочки MII->OCC->фолликулы). Лабораторные OCCR/MIIR/3pNR эту колонку
    НЕ используют — они считаются по настоящему "occ", иначе MIIR=MII/OCC
    вырождается ровно в 1.0 на каждой подставленной строке."""
    occ = pd.to_numeric(df.get("occ"), errors="coerce")
    mii_model = pd.to_numeric(df.get(MII_ML_FALLBACK_COLUMN), errors="coerce")
    return occ.where(occ.notna(), mii_model)


def _follicles_ml_fallback(df: pd.DataFrame) -> pd.Series:
    """Только для входа нейросетей: неизвестное число фолликулов получает
    модельный ОКК. Раньше эта подстановка (_follicles_fallback) применялась к
    основному df и попадала в лабораторный KPI, где OCCR=OCC/фолликулы с
    производным знаменателем давал ровно 1.0 — тот же дефект, что с производной
    колонкой фолликулов в GGRC."""
    follicles = pd.to_numeric(df.get("follicles"), errors="coerce")
    occ_model = pd.to_numeric(df.get(OCC_ML_FALLBACK_COLUMN), errors="coerce")
    return follicles.where(follicles.notna(), occ_model)


def _frozen_sum(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in ("frozen", "frozen_blastocysts") if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    return numeric.sum(axis=1, min_count=1)


def _priority_fill(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Первое непустое значение по приоритету колонок (слева направо)."""
    result = pd.Series(np.nan, index=df.index, dtype="object")
    for col in columns:
        if col not in df.columns:
            continue
        candidate = df[col]
        empty = candidate.isna() | (candidate.astype(str).str.strip() == "")
        mask = result.isna() & ~empty
        result.loc[mask] = candidate.loc[mask]
    return result


def _cohort_mask(df: pd.DataFrame) -> pd.Series:
    """Владелец (2026-07-18): главный фильтр — наличие исхода переноса, затем
    наличие числа инсеминированных (пусто -> строка удаляется), затем
    допустимая категория "Лечение"."""
    outcome = df["transfer_outcome"]
    outcome_present = outcome.notna() & (outcome.astype(str).str.strip() != "")
    inseminated_present = pd.to_numeric(df["inseminated"], errors="coerce").notna()
    if "treatment_type" in df.columns:
        treatment_ok = df["treatment_type"].astype(str).str.strip().str.lower().isin(_ALLOWED_TREATMENT)
    else:
        treatment_ok = pd.Series(False, index=df.index)
    return outcome_present & inseminated_present & treatment_ok


def _classify_outcome(value) -> float:
    """1 = любая беременность ("+28 дней", одно-/многоплодная — факт наступления,
    без разбора формы), 0 = "не наступила"/"биохимическая". Строка с несколькими
    склеенными через ";" исходами (артефакт выгрузки) -> 1, если среди них есть
    хоть один положительный маркер (владелец подтвердил: без предупреждения)."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    if not s:
        return np.nan
    if _POSITIVE_OUTCOME_MARKER in s:
        return 1.0
    if any(marker in s for marker in _NEGATIVE_OUTCOME_MARKERS):
        return 0.0
    warnings.warn(f"Нераспознанный текст исхода переноса (Fertimed): {value!r} -> оставлено как пропуск")
    return np.nan


def prepare(raw: pd.DataFrame, columns_cfg: dict, settings_cfg: dict) -> tuple[pd.DataFrame, list]:
    """Резолв колонок + правила когорты/импутации/кодирования Fertimed.
    Возвращает (normalized_df, unknown_columns) — та же форма, что
    scripts/run_analysis.py._normalise_source ожидает от normalize.normalize
    для клиники A."""
    resolved, unknown, missing = ingest.resolve_columns(raw, columns_cfg)
    if missing:
        raise ValueError(f"нет обязательных колонок: {missing}")

    df = resolved.drop(columns=[c for c in _DROP_RAW_COLUMNS if c in resolved.columns])

    df["age"] = df["age"].apply(_parse_age)
    df["attempt_no"] = _fillna_default(df["attempt_no"], 1)
    df["transfer_day"] = _fillna_default(df["transfer_day"], 5)
    df["fertilization_method"] = _fertilization_method(df)
    df["frozen"] = _frozen_sum(df)
    df["embryologist"] = _priority_fill(df, ["embryologist_fert", "embryologist_opu", "embryologist_et"])
    df["doctor"] = _priority_fill(df, ["doctor_opu", "doctor_et"])
    # Цепочка MII->OCC->фолликулы живёт ОТДЕЛЬНЫМИ колонками и подставляется в
    # канонические поля только при сборке model_input (scripts/run_analysis.py,
    # quirks.model_fallback_columns). Сами "mii"/"occ"/"follicles" остаются
    # сырыми, чтобы Vienna KPI и OZEMA считались по фактически измеренному.
    df[MII_ML_FALLBACK_COLUMN] = _mii_ml_fallback(df)
    df[OCC_ML_FALLBACK_COLUMN] = _occ_ml_fallback(df)
    df[FOLLICLES_ML_FALLBACK_COLUMN] = _follicles_ml_fallback(df)

    df = df.loc[_cohort_mask(df)].reset_index(drop=True)
    if df.empty:
        raise ValueError("после фильтра когорты Fertimed не осталось строк")

    df["Preg"] = df["transfer_outcome"].apply(_classify_outcome)
    df = normalize.impute_amh(df, settings_cfg)
    df = normalize.impute_bmi(df, settings_cfg)
    return df, unknown
