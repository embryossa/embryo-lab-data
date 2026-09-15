"""off_cohort.py — прогноз моделей для циклов вне когорты отчёта.

Основной пайплайн (scripts/run_analysis.py) отдаёт моделям только циклы с переносом
и известным исходом: KPI, калибровка и review-list требуют исхода. Остальные циклы
OPU-таблицы (freeze-all, отмена переноса, остановка развития, перенос без внесённого
исхода) попадали в Teable только фактической воронкой, и их карточки не несли ни
одного прогноза. Этот слой считает для них модели отдельно и НИКОГДА не смешивается
с отчётной когортой: ни KPI, ни калибровка, ни review-list, ни agent packet его не
читают.

Уровень моделей определяется лабораторными данными цикла (владелец, 2026-09-15):
  * full   — 2PN > 0: Digital Twin 7.1 (transfer + retrieval), TabPFN PRAI, TabPFN-3,
             DNN. Без переноса вероятность — оценка для гипотетического переноса.
  * funnel — OCC > 0, но 2PN = 0 (или неизвестно): только воронка DT из точки
             retrieval; вероятность беременности не выдаётся — зигот нет.
  * none   — OCC = 0 (ооциты не получены) или нет лабораторных данных: модели не
             применяются, карточка показывает причину.
Исключение 2PN=0 из KPI/модельного датасета (CLAUDE.md) этим не затрагивается.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import _io
from .dt_api import FUNNEL_COLUMNS

TIER_FULL, TIER_FUNNEL, TIER_NONE = "full", "funnel", "none"

SCOPE_LABELS = {
    "no_transfer": "Без переноса: вероятность — для гипотетического переноса",
    "no_transfer_no_fertilization": "Без переноса, 2PN=0: только воронка DT, без вероятности",
    "no_oocytes": "Ооциты не получены (OCC=0): модели не применялись",
    "no_lab_data": "Нет лабораторных данных: модели не применялись",
    "transfer_pending_outcome": "Перенос, исход ещё не внесён",
    "cohort_not_in_last_run": "Когорта отчёта, ещё не в прогоне пайплайна",
}

# Столбцы DT, которые остаются у циклов уровня funnel: воронка, кластер, стимуляция,
# OHSS и служебные поля. Всё прочее (вероятности беременности, BEFE, Bayes, KAT, GNN,
# кумулятивные и banking-оценки) очищается — у цикла без зигот это не прогноз.
_FUNNEL_KEEP_PREFIXES = ("med_", "p025_", "p975_", "cluster_", "ohss_", "stim_", "dt_api_",
                         "known_", "real_")
_FUNNEL_KEEP = {
    "_key", "cycle_uid", "timestamp", "clinic_name", "app_version", "age", "amh", "afc", "bmi",
    "attempt_number", "sperm_source", "follicles_tvp", "dominant_cluster", "p_no_blast",
    "p_no_good_blast", "dt_timepoint_used", "dt_funnel_timepoint", "nn_source", "notes",
    "patient_name", "amh_imputed", "bmi_imputed", "afc_source", "attempt_imputed",
    "sperm_source_inferred", "okk_raised_for_thawed", "Age", "Attempt", "N folicules OPU",
    "OCC", "MII", "Inseminated", "2pN", "Bl", "Good Bl", *FUNNEL_COLUMNS,
}


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def classify(cycles: pd.DataFrame, reported_uids: set[str]) -> pd.DataFrame:
    """Циклы, которых нет в последнем прогоне пайплайна, с охватом и уровнем моделей.

    ``cycles`` — нормализованная OPU-таблица с ``_key``/``cycle_uid``;
    ``reported_uids`` — cycle_uid из dt_predictions.csv основного прогона.
    """
    out = cycles.loc[~cycles["cycle_uid"].isin(reported_uids)].copy()
    occ, two_pn = _num(out, "occ"), _num(out, "two_pn")
    transferred, preg = _num(out, "transferred"), _num(out, "Preg")

    tier = pd.Series(TIER_NONE, index=out.index, dtype=object)
    tier[occ.gt(0)] = TIER_FUNNEL
    tier[two_pn.gt(0)] = TIER_FULL

    scope = pd.Series("no_transfer", index=out.index, dtype=object)
    scope[tier.eq(TIER_FUNNEL)] = "no_transfer_no_fertilization"
    scope[occ.eq(0)] = "no_oocytes"
    scope[occ.isna() & tier.eq(TIER_NONE)] = "no_lab_data"
    scope[transferred.gt(0) & preg.isna()] = "transfer_pending_outcome"
    scope[transferred.gt(0) & preg.notna() & two_pn.gt(0)] = "cohort_not_in_last_run"

    out["model_tier"] = tier
    out["cohort_scope"] = scope
    out["cohort_scope_label"] = scope.map(SCOPE_LABELS)
    return out


def strip_to_funnel(dt: pd.DataFrame) -> pd.DataFrame:
    """Очищает у выхода DT всё, кроме воронки и описательных полей (уровень funnel)."""
    out = dt.copy()
    for column in out.columns:
        if column in _FUNNEL_KEEP or column.startswith(_FUNNEL_KEEP_PREFIXES):
            continue
        out[column] = np.nan
    return out


def empty_dt_rows(cycles: pd.DataFrame) -> pd.DataFrame:
    """Строки в контракте DT_OUTPUT без прогноза — для циклов уровня none."""
    return pd.DataFrame({"_key": cycles["_key"].to_numpy(), "cycle_uid": cycles["cycle_uid"].to_numpy(),
                         "DIGITAL TWIN": np.nan, "dt_timepoint_used": "none"})


# Фактические значения, которые карточка и Teable читают из dt_predictions.csv.
# Эхо DT у уровня funnel неполное, у уровня none его нет вовсе, поэтому истинные
# до-модельные значения переносятся из OPU-таблицы для всех строк (как
# analytics._restore_true_counts в основном прогоне).
FACT_COLUMNS = {
    "Age": "age", "age": "age", "amh": "amh", "bmi": "bmi", "afc": "afc",
    "attempt_number": "attempt_no", "Attempt": "attempt_no", "sperm_source": "sperm_source",
    "N folicules OPU": "follicles", "OCC": "occ", "MII": "mii", "Inseminated": "inseminated",
    "2pN": "two_pn", "Bl": "bl_total", "Good Bl": "bl_good", "real_cryo": "frozen",
    "real_cleav": "cleavage_d3", "patient_name": "Patient Full Name",
    "amh_imputed": "amh_imputed", "bmi_imputed": "bmi_imputed",
}


TRANSFERRED_FEATURE = "Перенесено эмбрионов"


def dnn_input_with_zero_transfer(model_input_path, private_dir) -> tuple[Path, int]:
    """Вход DNN, где пустое ET цикла вне когорты отчёта читается как 0 перенесённых.

    В OPU-таблице у цикла без переноса ET просто не заполнен, а DNN (Evo_calculations.py)
    отбрасывает строку с пропуском в любом признаке обучения — без подстановки она
    оценивала 4 цикла из 57. Владелец, 2026-09-15: в этом слое пустое ET = 0, только
    для входа DNN; OPU-таблица, лабораторные KPI и входы DT/TabPFN не меняются.
    Флаг строки — et_zero_for_dnn.
    """
    frame = pd.read_excel(model_input_path)
    if TRANSFERRED_FEATURE not in frame.columns:
        return Path(model_input_path), 0
    value = pd.to_numeric(frame[TRANSFERRED_FEATURE], errors="coerce")
    fill = value.isna()
    frame[TRANSFERRED_FEATURE] = value.where(~fill, 0)
    frame["et_zero_for_dnn"] = fill
    out = Path(private_dir) / "model_input_dnn_et0.xlsx"
    _io.atomic_write_excel({"model_input": frame}, out)
    return out, int(fill.sum())


def restore_facts(predictions: pd.DataFrame, cycles: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    source = cycles.set_index("cycle_uid")
    for target, column in FACT_COLUMNS.items():
        if column not in source.columns:
            continue
        values = out["cycle_uid"].map(source[column])
        out[target] = values.to_numpy()
    for column in ("model_tier", "cohort_scope", "cohort_scope_label"):
        out[column] = out["cycle_uid"].map(source[column]).to_numpy()
    return out
