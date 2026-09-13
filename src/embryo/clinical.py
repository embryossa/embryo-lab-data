"""clinical.py — описательный клинический контекст цикла для итоговой интерпретации.

Собирает поля выгрузки МИС, которые НЕ участвуют ни в фильтрации когорты, ни в
расчёте KPI, ни во входе DT/TabPFN/DNN: тип протокола стимуляции, суммарная доза
ФСГ, триггер, дни менструального цикла (старт протокола/стимуляции/пункция),
длительность стимуляции, базальный ФСГ, АМГ, КАФ и коды МКБ-10 основного и
сопутствующего диагноза.

Модуль ничего не интерпретирует и ничего не сравнивает с порогами — он считает и
классифицирует, а объясняет роль `clinical-context-analyst` (так же, как MIRI/
лоты/ПГТ: Python считает, агент объясняет).

ГРАНИЦА МОДЕЛЕЙ. Колонки этого слоя живут под префиксами ``clin_``/``stim_``.
Ни ``_add_opu_aliases``, ни ``_add_dt_aliases`` в scripts/run_analysis.py их не
перечисляют, поэтому во вход нейросетей они физически не попадают. В частности
``clin_afc`` — это НЕ канонический ``afc``: канонический уходит в μ первой стадии
ZINB и в ``normalize.impute_amh``, а КАФ по решению владельца (2026-07-31)
остаётся только описательным (см. docs/FERTIMED.md).

ЗАПОЛНЕННОСТЬ — ЧАСТЬ ДАННЫХ. Каждый блок несёт ``n_known``/``n_total``: у поля
"Триггер" в июньской выгрузке заполнено 20 строк из 105, и агент, не видящий
знаменателя, примет распределение по 20 строкам за портрет когорты. Пустое поле
здесь — "не заполнено в МИС", а не "не назначалось".
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import yaml

from .normalize import parse_dates

# Префиксы колонок описательного клинического слоя. Всё, что не начинается с них,
# этот модуль не трогает.
CLINICAL_PREFIXES = ("clin_", "stim_")

# Колонки, которые уходят в агентский пакет по-цикловым записям. Только этот
# список — не "все clin_*", чтобы добавление колонки в columns.yaml не расширяло
# агентский пакет молча.
PACKET_COLUMNS = [
    "stim_protocol_type", "stim_fsh_dose", "stim_fsh_dose_per_day", "stim_trigger",
    "stim_protocol_start_day", "stim_stimulation_start_day", "stim_duration_days",
    "clin_opu_cycle_day", "clin_afc", "clin_fsh_basal",
    "clin_days_start_to_opu", "clin_days_lmp_to_opu", "clin_termination_reason",
    "clin_dx_primary_code", "clin_dx_primary_group",
    "clin_dx_secondary_codes", "clin_dx_secondary_groups",
    "clin_dx_groups_all",
]

# Даты в выгрузке 1С приходят строками "дд.мм.гггг": без dayfirst 05.06.2026
# прочиталось бы как 6 мая. Разбор идёт через normalize.parse_dates — он держит
# оба формата, потому что тот же кадр может прийти уже переписанным в ISO.
_DATE_COLUMNS = ("clin_cycle_start_date", "clin_lmp_date")

_CODE_SPLIT_RE = re.compile(r"[;,]")
_CODE_CLEAN_RE = re.compile(r"[^A-ZА-Я0-9.]", re.I)


# --------------------------------------------------------------------------- #
# справочник диагнозов
# --------------------------------------------------------------------------- #
def load_diagnosis_groups(path: str | Path) -> dict:
    """Читает config/clinics/<клиника>/diagnosis_groups.yaml.

    Отсутствующий файл — не ошибка: тогда группировка не выполняется и агент
    получает только сырые коды (честнее, чем угаданные группы).
    """
    p = Path(path)
    if not p.is_file():
        return {}
    with p.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    prefixes = {str(k).strip().upper(): str(v) for k, v in (cfg.get("prefixes") or {}).items()}
    return {"prefixes": prefixes, "fallback": str(cfg.get("fallback") or "прочее / вне справочника")}


def _clean_code(value: object) -> str:
    return _CODE_CLEAN_RE.sub("", str(value)).strip().upper()


def split_codes(value: object) -> list[str]:
    """"N80.1; D25.2" -> ["N80.1", "D25.2"]. Пустое/NaN -> []."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return []
    codes = []
    for part in _CODE_SPLIT_RE.split(str(value)):
        code = _clean_code(part)
        if code:
            codes.append(code)
    return codes


def classify_code(code: str, groups: dict) -> str:
    """Самый длинный совпавший префикс выигрывает: "N97.1" точнее, чем "N97"."""
    if not groups:
        return ""
    prefixes = groups.get("prefixes") or {}
    best_prefix, best_label = "", ""
    code = _clean_code(code)
    for prefix, label in prefixes.items():
        if code.startswith(prefix) and len(prefix) > len(best_prefix):
            best_prefix, best_label = prefix, label
    return best_label or groups.get("fallback", "")


def classify_codes(value: object, groups: dict) -> list[str]:
    """Список групп без повторов, порядок как в исходной строке."""
    seen, out = set(), []
    for code in split_codes(value):
        label = classify_code(code, groups)
        if label and label not in seen:
            seen.add(label)
            out.append(label)
    return out


# --------------------------------------------------------------------------- #
# по-цикловый кадр
# --------------------------------------------------------------------------- #
def has_clinical_columns(frame: pd.DataFrame) -> bool:
    return any(str(c).startswith(CLINICAL_PREFIXES) for c in frame.columns)


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _dates(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    return parse_dates(frame[column])


def build_cycle_frame(frame: pd.DataFrame, groups: dict) -> pd.DataFrame:
    """cycle_uid + описательные клинические признаки одного цикла.

    Идентификаторы сюда не попадают: единственный ключ — cycle_uid, всё
    остальное — клинические признаки из allow-list PACKET_COLUMNS.
    """
    if "cycle_uid" not in frame.columns:
        raise KeyError("clinical.build_cycle_frame требует cycle_uid")
    out = pd.DataFrame({"cycle_uid": frame["cycle_uid"].to_numpy()}, index=frame.index)

    for column in ("stim_protocol_type", "stim_trigger", "clin_termination_reason"):
        if column in frame.columns:
            text = frame[column].astype("string").str.strip()
            out[column] = text.replace({"": pd.NA})
        else:
            out[column] = pd.Series(pd.NA, index=frame.index, dtype="string")

    dose = _num(frame, "stim_fsh_dose")
    duration = _num(frame, "stim_duration_days")
    out["stim_fsh_dose"] = dose
    out["stim_duration_days"] = duration
    # Средняя суточная доза — производная величина, а не поле МИС: подневного
    # дозирования в выгрузке нет (см. docs/FERTIMED.md). Определена только там,
    # где известны обе величины и длительность строго положительна.
    out["stim_fsh_dose_per_day"] = (dose / duration.where(duration > 0)).round(1)
    out["stim_protocol_start_day"] = _num(frame, "stim_protocol_start_day")
    out["stim_stimulation_start_day"] = _num(frame, "stim_stimulation_start_day")
    out["clin_opu_cycle_day"] = _num(frame, "clin_opu_cycle_day")
    out["clin_afc"] = _num(frame, "clin_afc")
    out["clin_fsh_basal"] = _num(frame, "clin_fsh_basal")

    date_opu = _dates(frame, "date_opu")
    # Отрицательный интервал "старт программы -> пункция" не бывает клинической
    # величиной: пункция не может предшествовать началу собственной программы.
    # Он означает, что две даты принадлежат РАЗНЫМ циклам — у Fertimed это
    # криоперенос: 1С переносит на такую строку исходную "Дата пункции" свежего
    # забора, а "Дата начала" описывает уже текущую программу переноса (в
    # июньской выгрузке медиана −105 дней на 33 строках из 34). Такое значение
    # гасится в пропуск, а не подаётся агенту как срок: иначе он получит
    # "стимуляция началась через 105 дней после пункции". Сколько строк погашено —
    # видно по n_known в когортной сводке.
    # "ДПМ -> пункция" остаётся: последняя менструация переносится 1С вместе с
    # датой пункции, обе относятся к одному забору (медиана 13 дней на всей
    # когорте, включая криоперенос).
    start_to_opu = (date_opu - _dates(frame, "clin_cycle_start_date")).dt.days
    out["clin_days_start_to_opu"] = start_to_opu.where(start_to_opu >= 0)
    out["clin_days_lmp_to_opu"] = (date_opu - _dates(frame, "clin_lmp_date")).dt.days

    primary = frame["clin_dx_primary"] if "clin_dx_primary" in frame.columns else pd.Series(pd.NA, index=frame.index)
    secondary = frame["clin_dx_secondary"] if "clin_dx_secondary" in frame.columns else pd.Series(pd.NA, index=frame.index)
    primary_codes = primary.map(split_codes)
    secondary_codes = secondary.map(split_codes)
    out["clin_dx_primary_code"] = primary_codes.map(lambda codes: codes[0] if codes else None)
    out["clin_dx_primary_group"] = primary.map(lambda v: (classify_codes(v, groups) or [None])[0])
    out["clin_dx_secondary_codes"] = secondary_codes
    out["clin_dx_secondary_groups"] = secondary.map(lambda v: classify_codes(v, groups))
    # Объединённый набор групп цикла — то, по чему считаются доли в когорте:
    # деление на "основной/сопутствующий" в 1С заполняется непоследовательно.
    out["clin_dx_groups_all"] = [
        sorted(set(classify_codes(p, groups)) | set(classify_codes(s, groups)))
        for p, s in zip(primary, secondary)
    ]
    return out[["cycle_uid", *[c for c in PACKET_COLUMNS if c in out.columns]]]


# --------------------------------------------------------------------------- #
# когортная сводка
# --------------------------------------------------------------------------- #
def _numeric_summary(series: pd.Series, total: int) -> dict:
    known = series.dropna()
    summary = {"n_known": int(len(known)), "n_total": int(total)}
    if known.empty:
        return summary
    summary.update({
        "median": round(float(known.median()), 2),
        "p25": round(float(known.quantile(0.25)), 2),
        "p75": round(float(known.quantile(0.75)), 2),
        "min": round(float(known.min()), 2),
        "max": round(float(known.max()), 2),
    })
    return summary


def _category_summary(series: pd.Series, total: int) -> dict:
    known = series.dropna()
    counts = known.value_counts()
    return {
        "n_known": int(len(known)),
        "n_total": int(total),
        "counts": {str(k): int(v) for k, v in counts.items()},
    }


def _group_summary(lists: pd.Series, total: int) -> dict:
    """Доли считаются от числа циклов с ХОТЬ ОДНИМ распознанным кодом.

    Один цикл может попасть в несколько групп (основной + сопутствующие),
    поэтому сумма counts превышает n_known — это не ошибка, а свойство
    множественного диагноза; знаменатель назван явно.
    """
    counts: dict[str, int] = {}
    n_known = 0
    for groups in lists:
        items = list(groups) if isinstance(groups, (list, tuple)) else []
        if not items:
            continue
        n_known += 1
        for item in items:
            counts[str(item)] = counts.get(str(item), 0) + 1
    ordered = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
    return {
        "n_cycles_with_any_code": n_known,
        "n_total": int(total),
        "counts_by_group": ordered,
        "note": "цикл может входить в несколько групп; сумма counts_by_group больше "
                "n_cycles_with_any_code — это множественный диагноз, не ошибка",
    }


def build_cohort_summary(cycles: pd.DataFrame) -> dict:
    """Когортная сводка описательного слоя (без идентификаторов)."""
    total = int(len(cycles))
    summary: dict = {
        "population": "когорта отчёта (перенос>0 и известный исход переноса)",
        "n_cycles": total,
    }
    if total == 0:
        return summary

    summary["protocol_type"] = _category_summary(cycles.get("stim_protocol_type", pd.Series(dtype="string")), total)
    summary["trigger"] = _category_summary(cycles.get("stim_trigger", pd.Series(dtype="string")), total)
    summary["termination_reason"] = _category_summary(
        cycles.get("clin_termination_reason", pd.Series(dtype="string")), total)
    for column, label in [
        ("stim_fsh_dose", "fsh_total_dose_iu"),
        ("stim_fsh_dose_per_day", "fsh_dose_per_day_iu"),
        ("stim_duration_days", "stimulation_duration_days"),
        ("stim_protocol_start_day", "protocol_start_cycle_day"),
        ("stim_stimulation_start_day", "stimulation_start_cycle_day"),
        ("clin_opu_cycle_day", "opu_cycle_day"),
        ("clin_afc", "afc"),
        ("clin_fsh_basal", "fsh_basal"),
        ("clin_days_start_to_opu", "days_from_cycle_start_to_opu"),
        ("clin_days_lmp_to_opu", "days_from_lmp_to_opu"),
    ]:
        series = cycles[column] if column in cycles.columns else pd.Series(dtype=float)
        summary[label] = _numeric_summary(pd.to_numeric(series, errors="coerce"), total)

    if "clin_dx_groups_all" in cycles.columns:
        summary["diagnosis_groups"] = _group_summary(cycles["clin_dx_groups_all"], total)
    if "clin_dx_primary_code" in cycles.columns:
        summary["primary_diagnosis_codes"] = _category_summary(
            cycles["clin_dx_primary_code"].astype("string"), total)

    # Дозу назначают по ожидаемому ответу, поэтому её распределение читается
    # только внутри типа протокола; естественный цикл дозы не имеет вовсе.
    if {"stim_protocol_type", "stim_fsh_dose"} <= set(cycles.columns):
        by_protocol = {}
        for name, group in cycles.groupby(cycles["stim_protocol_type"].astype("string"), dropna=True):
            dose = pd.to_numeric(group["stim_fsh_dose"], errors="coerce")
            by_protocol[str(name)] = {
                "n_cycles": int(len(group)),
                **_numeric_summary(dose, int(len(group))),
            }
        summary["fsh_total_dose_by_protocol"] = by_protocol
    summary["fill_rate_note"] = (
        "n_known/n_total — заполненность поля в МИС, а не клинический факт: пустое "
        "поле означает «не заполнено в выгрузке», не «не назначалось». Поля "
        "стимуляции структурно пусты в криопереносных и естественных циклах."
    )
    summary["days_from_cycle_start_to_opu_note"] = (
        "Определён только там, где дата старта программы и дата пункции относятся "
        "к одному циклу. В криопереносной строке 1С несёт исходную дату пункции "
        "свежего забора при дате старта текущей программы переноса — такой "
        "интервал погашен в пропуск, поэтому n_known здесь может быть много "
        "меньше числа циклов. Это не потеря данных, а отказ считать срок между "
        "разными циклами."
    )
    return summary


def build_lab_population_summary(frame: pd.DataFrame, cohort_n: int | None = None) -> dict:
    """Причина завершения программы по ВСЕЙ выгрузке — отдельный блок.

    Почему отдельно, а не в общей сводке: в когорте отчёта (перенос>0 и известный
    исход) причина завершения почти всегда «Перенос эмбрионов» — по определению
    когорты, а не как наблюдение. Всё содержательное («Остановка развития
    эмбрионов», «Отсутствие оплодотворения», «Овуляция», «Отсутствие пригодных к
    оплодотворению ооцитов») живёт именно в строках, до переноса не дошедших.

    Кадр сюда подаётся ДО клиничного когортного фильтра, а не «широкая
    lab-популяция» пайплайна: у Fertimed та уже отфильтрована собственным
    `_cohort_mask` (требует заполненного исхода переноса), поэтому в июньском
    прогоне давала ровно те же 34 строки, что и когорта отчёта, и блок был
    подписан честным на вид, но неверным знаменателем. Проверено на выгрузке:
    до фильтра 105 строк, среди них 8 остановок развития, 3 овуляции, 2 отсутствия
    оплодотворения — то есть блок молча терял ровно то, ради чего существует.

    `n_cycles_report_cohort` печатается рядом специально: пока два числа видны
    вместе, знаменатели невозможно перепутать.
    """
    if "clin_termination_reason" not in frame.columns:
        return {}
    total = int(len(frame))
    reason = frame["clin_termination_reason"].astype("string").str.strip().replace({"": pd.NA})
    summary = {
        "population": "ВСЯ выгрузка МИС до когортного фильтра клиники — другой "
                      "знаменатель, чем у всех остальных блоков clinical_context",
        "n_rows_export": total,
        "termination_reason": _category_summary(reason, total),
        "note": "Не сравнивай эти доли с долями по когорте отчёта и не складывай их: "
                "это разные популяции. Причины, отличные от переноса/криоконсервации, "
                "встречаются почти исключительно здесь — в строках, до переноса не "
                "дошедших, которых в когорте отчёта нет по определению. Причина "
                "завершения — запись оператора МИС, а не установленная причина неудачи.",
    }
    if cohort_n is not None:
        summary["n_cycles_report_cohort"] = int(cohort_n)
    return summary


def build_clinical_context(frame: pd.DataFrame, groups: dict,
                           export_population: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """(по-цикловый кадр, когортная сводка) для пакета агента и CSV.

    ``export_population`` — резолвнутая выгрузка ДО клиничного когортного фильтра,
    только для распределения причин завершения программы (см.
    build_lab_population_summary). Ничего другого из неё не берётся.
    """
    cycles = build_cycle_frame(frame, groups)
    summary = build_cohort_summary(cycles)
    if export_population is not None and not export_population.empty:
        export_summary = build_lab_population_summary(export_population, cohort_n=len(cycles))
        if export_summary:
            summary["export_population_termination"] = export_summary
    return cycles, summary
