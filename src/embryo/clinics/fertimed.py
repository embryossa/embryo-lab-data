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
# лицо в строке (сурмама, партнёр, доноры), ненадёжное поле SMF и данные
# родов/осложнений вне области клинической беременности.
# КАФ вышла из этого списка 2026-07-31 — см. AFC_PLAUSIBLE_MAX ниже.
#
# Сопоставление НЕ побуквенное (_normalise_header): 1С пишет одну и ту же
# колонку по-разному от выгрузки к выгрузке — "ФИОСурмамы" (шестимесячная),
# "ФИО сурмамы" (июньская), "Партнер" против "Партнёр". До 2026-07-31 список
# сравнивался строкой в строку, и обе PHI-колонки второго/третьего лица не
# удалялись ни разу: имя сурмамы и партнёр оставались в локальных processed-
# файлах. В агентский пакет они не попадали никогда (там свой allow-list и
# privacy._assert_agent_safe), но и в локальном слое им не место.
_DROP_RAW_COLUMNS = [
    "ФИО Сурмамы", "Партнёр", "Номер карты партнера",
    "Fresh OD", "fresh OD (код донора)", "Vitrified OD", "vitrified OD (код донора)",
    "ДС", "ДС (код донора)", "ДЭ", "ДЭ (Код донора)",
    "SMF", "FET", "PGT-TVP",
    "Внематочная", "Количество плодных яиц", "Роды Да/Нет", "Срок родов",
    "Метод родоразрешения", "Количество детей", "Комментарий по исходу беременности",
]

# Верхняя граница физиологичного антрального счёта. В майской выгрузке колонка
# "КАФ" содержала билинговые суммы (84 строки из 145 в диапазоне 270–250344),
# в исправленной июньской — реальный счёт (max 30, r=0.93 с числом фолликулов).
# Порог не "чинит" данные, а не даёт молча вернуться прежнему дефекту: значение
# выше границы гасится в пропуск с предупреждением, а не подставляется в
# интерпретацию как счёт фолликулов. 60 — заведомо выше любого реального АФК
# (тяжёлый СПКЯ даёт ~40-50) и заведомо ниже билинговых сумм.
AFC_PLAUSIBLE_MAX = 60.0

_AGE_RE = re.compile(r"(\d+)")
_POSITIVE_OUTCOME_MARKER = "+28"
_NEGATIVE_OUTCOME_MARKERS = ("не наступила", "биохимическая")


def _normalise_header(name: object) -> str:
    """Заголовок к сравнимому виду: регистр, пробелы и «ё»/«е» игнорируются."""
    return re.sub(r"\s+", "", str(name)).strip().lower().replace("ё", "е")


def _drop_unused_raw(df: pd.DataFrame) -> pd.DataFrame:
    targets = {_normalise_header(name) for name in _DROP_RAW_COLUMNS}
    return df.drop(columns=[c for c in df.columns if _normalise_header(c) in targets])


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


def _guard_clinical_afc(df: pd.DataFrame) -> pd.Series:
    """КАФ -> число или пропуск; нефизиологичные значения гасятся с предупреждением.

    Колонка описательная (clin_afc), во вход DT/TabPFN и в normalize.impute_amh
    не идёт — там работает канонический "afc", которого у Fertimed нет.
    """
    if "clin_afc" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    afc = pd.to_numeric(df["clin_afc"], errors="coerce")
    implausible = afc.notna() & ((afc > AFC_PLAUSIBLE_MAX) | (afc < 0))
    if bool(implausible.any()):
        warnings.warn(
            f"КАФ вне физиологичного диапазона 0..{AFC_PLAUSIBLE_MAX:.0f} "
            f"в {int(implausible.sum())} строках (пример: "
            f"{afc.loc[implausible].iloc[0]!r}) -> оставлено как пропуск. "
            "Похоже на возврат дефекта майской выгрузки (билинговые суммы под "
            "заголовком КАФ), см. docs/FERTIMED.md."
        )
    return afc.where(~implausible)


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
    допустимая категория "Лечение".

    Это когорта ОТЧЁТА (вход DT/TabPFN/DNN, review-list, реконсилиация по
    исходу) — она обязана быть гейтована исходом. Лабораторные KPI считаются
    НЕ по ней, а по широкой популяции ``prepare_wide`` ниже."""
    outcome = df["transfer_outcome"]
    outcome_present = outcome.notna() & (outcome.astype(str).str.strip() != "")
    inseminated_present = pd.to_numeric(df["inseminated"], errors="coerce").notna()
    if "treatment_type" in df.columns:
        treatment_ok = df["treatment_type"].astype(str).str.strip().str.lower().isin(_ALLOWED_TREATMENT)
    else:
        treatment_ok = pd.Series(False, index=df.index)
    return outcome_present & inseminated_present & treatment_ok


def _tokens_of(columns_cfg: dict, section: str, key: str) -> set[str]:
    cfg = (columns_cfg or {}).get(section, {}) or {}
    return {str(token).strip().lower() for token in (cfg.get(key) or [])}


def _lab_population_mask(df: pd.DataFrame, columns_cfg: dict) -> pd.Series:
    """Строки широкой lab-популяции: ВСЕ циклы, кроме перечисленных категорий
    "Лечение", независимо от того, был перенос или нет.

    Владелец (2026-07-31): лабораторные KPI — это показатели работы
    лаборатории, а не переносов. Исключаются только те категории, которые не
    являются свежим циклом этой лаборатории: "Криоперенос" (счётчики воронки
    перенесены 1С из исходного протокола, сам протокол уже присутствует в
    популяции отдельной строкой — иначе одна пункция считалась бы дважды) и
    "Инсеминация" (ВМИ — другая модальность). "Только пункция" здесь
    ОСТАЁТСЯ: пункция состоялась, фолликулы/ОКК/MII заполнены, и OCCR/MIIR
    врача обязаны её считать; из метрик эмбриолога она уходит отдельно, через
    ``kpi.embryologist_only_exclusion_mask``.

    Гейта по исходу переноса и по числу инсеминированных здесь нет намеренно:
    цикл, закончившийся отсутствием оплодотворения или остановкой развития, —
    это результат работы лаборатории, а не отсутствие данных.
    """
    excluded = _tokens_of(columns_cfg, "lab_population", "excluded_treatment_tokens")
    if not excluded or "treatment_type" not in df.columns:
        return pd.Series(True, index=df.index)
    treatment = df["treatment_type"].astype("string").str.strip().str.lower()
    return ~treatment.isin(excluded).fillna(False)


# Цепочка стадий: каждый элемент — (стадия, предшествующая стадия). Пустая
# стадия у ЗАКРЫТОГО цикла означает "до неё не дошли", если предшествующая
# стадия известна и строго положительна.
_CLOSED_CYCLE_ZERO_CHAIN = (
    ("cleavage_d3", "two_pn"),
    ("embryos_d5", "cleavage_d3"),
    ("bl_total", "two_pn"),
    ("bl_good", "bl_total"),
)


def _fill_closed_cycle_zeros(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """У закрытого цикла пустой счётчик ниже достигнутой стадии — наблюдённый 0.

    Владелец подтвердил 2026-07-31. Признак закрытия — заполненная "Причина
    завершения программы ВРТ" (``clin_termination_reason``): "Остановка
    развития эмбрионов", "Отсутствие оплодотворения", "Овуляция" и т.п. Пустая
    причина означает цикл, ещё идущий на момент выгрузки, — там пропуск
    остаётся пропуском и строку убирает ``kpi.competency_exclusion_mask``.

    Без этого правила BlR/TGBDR систематически завышены: ``_aggregate_one``
    требует непустой числитель, поэтому цикл с 2PN=5 и пустым Bl выпадал из
    расчёта целиком — то есть из знаменателя уходили ровно те циклы, где
    бластоцист не получилось (в июньской выгрузке таких 6).

    2PN сознательно НЕ заполняется нулём: строки с 2PN=0 исключаются из
    KPI/модельного набора отдельным неподвижным решением владельца, и
    производить их из пропусков здесь означало бы менять то решение молча.
    """
    filled: dict[str, int] = {}
    if "clin_termination_reason" not in df.columns:
        return df, filled
    reason = df["clin_termination_reason"]
    closed = reason.notna() & (reason.astype(str).str.strip() != "")
    if not bool(closed.any()):
        return df, filled
    out = df.copy()
    for stage, previous in _CLOSED_CYCLE_ZERO_CHAIN:
        if stage not in out.columns or previous not in out.columns:
            continue
        stage_values = pd.to_numeric(out[stage], errors="coerce")
        previous_values = pd.to_numeric(out[previous], errors="coerce")
        # Достаточно, чтобы предшествующая стадия была ИЗВЕСТНА: её ноль столь
        # же однозначно означает ноль ниже по цепочке, сколько и её положительное
        # значение — при закрытом цикле.
        target = closed & stage_values.isna() & previous_values.notna()
        if not bool(target.any()):
            continue
        out.loc[target, stage] = 0.0
        filled[stage] = int(target.sum())
    return out, filled


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


def _prepare_common(raw: pd.DataFrame, columns_cfg: dict) -> tuple[pd.DataFrame, list]:
    """Резолв колонок и построчные правила Fertimed, общие для обеих когорт
    (отчётной и широкой lab-популяции). Фильтр строк сюда не входит."""
    resolved, unknown, missing = ingest.resolve_columns(raw, columns_cfg)
    if missing:
        raise ValueError(f"нет обязательных колонок: {missing}")

    df = _drop_unused_raw(resolved)

    df["age"] = df["age"].apply(_parse_age)
    df["attempt_no"] = _fillna_default(df["attempt_no"], 1)
    df["transfer_day"] = _fillna_default(df["transfer_day"], 5)
    df["fertilization_method"] = _fertilization_method(df)
    df["frozen"] = _frozen_sum(df)
    df["embryologist"] = _priority_fill(df, ["embryologist_fert", "embryologist_opu", "embryologist_et"])
    df["doctor"] = _priority_fill(df, ["doctor_opu", "doctor_et"])
    # Описательный клинический слой (clin_*/stim_*) остаётся в кадре как есть:
    # его собирает embryo.clinical уже после присвоения cycle_uid. Здесь —
    # только защита диапазона КАФ, потому что это правило конкретной выгрузки.
    df["clin_afc"] = _guard_clinical_afc(df)
    # Цепочка MII->OCC->фолликулы живёт ОТДЕЛЬНЫМИ колонками и подставляется в
    # канонические поля только при сборке model_input (scripts/run_analysis.py,
    # quirks.model_fallback_columns). Сами "mii"/"occ"/"follicles" остаются
    # сырыми, чтобы Vienna KPI и OZEMA считались по фактически измеренному.
    df[MII_ML_FALLBACK_COLUMN] = _mii_ml_fallback(df)
    df[OCC_ML_FALLBACK_COLUMN] = _occ_ml_fallback(df)
    df[FOLLICLES_ML_FALLBACK_COLUMN] = _follicles_ml_fallback(df)
    return df, unknown


def prepare(raw: pd.DataFrame, columns_cfg: dict, settings_cfg: dict) -> tuple[pd.DataFrame, list]:
    """Когорта ОТЧЁТА: резолв колонок + правила когорты/импутации/кодирования
    Fertimed. Возвращает (normalized_df, unknown_columns) — та же форма, что
    scripts/run_analysis.py._normalise_source ожидает от normalize.normalize
    для клиники A."""
    df, unknown = _prepare_common(raw, columns_cfg)

    df = df.loc[_cohort_mask(df)].reset_index(drop=True)
    if df.empty:
        raise ValueError("после фильтра когорты Fertimed не осталось строк")

    df["Preg"] = df["transfer_outcome"].apply(_classify_outcome)
    df = normalize.impute_amh(df, settings_cfg)
    df = normalize.impute_bmi(df, settings_cfg)
    return df, unknown


def prepare_wide(raw: pd.DataFrame, columns_cfg: dict, settings_cfg: dict) -> tuple[pd.DataFrame, list]:
    """Широкая lab-популяция: та же построчная подготовка, что и в ``prepare``,
    но БЕЗ гейта по исходу переноса — источник всех лабораторных KPI и OZEMA.

    Форма возврата совпадает с ``prepare``/``ggrc.prepare_wide``, поэтому
    scripts/run_analysis.py вызывает её через тот же clinic-adapter контракт.
    Отличий от отчётной когорты ровно три: фильтр строк
    (``_lab_population_mask``), заполнение наблюдённых нулей у закрытых циклов
    (``_fill_closed_cycle_zeros``) и допустимо пустой ``Preg``.
    """
    df, unknown = _prepare_common(raw, columns_cfg)

    df = df.loc[_lab_population_mask(df, columns_cfg)].reset_index(drop=True)
    if df.empty:
        raise ValueError("после фильтра широкой lab-популяции Fertimed не осталось строк")
    df, filled = _fill_closed_cycle_zeros(df)
    if filled:
        warnings.warn(
            "Широкая lab-популяция Fertimed: пустые счётчики закрытых циклов "
            f"приняты за наблюдённый 0 — {filled} (правило владельца 2026-07-31, "
            "см. docs/FERTIMED.md)."
        )

    # Исход известен не у всех строк широкой популяции — это норма (цикл мог
    # закончиться до переноса). Preg остаётся пропуском, лабораторные KPI его
    # не читают.
    df["Preg"] = df["transfer_outcome"].apply(_classify_outcome)
    df = normalize.impute_amh(df, settings_cfg)
    df = normalize.impute_bmi(df, settings_cfg)
    return df, unknown
