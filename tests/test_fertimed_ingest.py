"""Синтетические (вымышленные ФИО/ID) фикстуры и тесты бизнес-правил Fertimed
(см. src/embryo/clinics/fertimed.py и docs/FERTIMED.md)."""
from pathlib import Path

import pandas as pd
import pytest

from embryo import ingest
from embryo.clinics import fertimed

ROOT = Path(__file__).resolve().parents[1]
COLUMNS_CFG = ingest.load_config(str(ROOT / "config/clinics/fertimed/columns.yaml"))
SETTINGS_CFG = {
    "amh_impute": {"a_afc": 0.16, "b_occ": 0.05, "c_age": 0.08, "intercept": 0.5, "floor": 0.1, "ceil": 12.0},
    "bmi_impute": {"constant": 24.0},
}


def _raw_rows(**overrides) -> pd.DataFrame:
    base = {
        "Номер карты пациента": [
            "CASE-CRYO-OK", "CASE-ONLY-PUNCTURE", "CASE-IUI", "CASE-NO-INSEMINATED",
            "CASE-NO-OUTCOME", "CASE-IVF-MII-FALLBACK", "CASE-ICSI-NO-MII-FALLBACK",
            "CASE-FOLLICLES-FALLBACK", "CASE-DEFAULTS", "CASE-FROZEN-SUM",
            "CASE-COMPOSITE-OUTCOME", "CASE-BIOCHEMICAL", "CASE-EMBRYOLOGIST-PRIORITY",
            "CASE-DOCTOR-PRIORITY", "CASE-AGE-TEXT",
        ],
        "ФИО": ["Тестовая Пациентка"] * 15,
        "Возраст": ["34 года", "30 лет", "28 лет", "31 год", "27 лет", "33 года", "29 лет",
                    "35 лет", "40 лет", "26 лет", "32 года", "38 лет", "25 лет", "36 лет", "26 лет"],
        "№ попытки": [1, 1, 1, 1, 1, 1, 1, 1, None, 1, 1, 1, 1, 1, 1],
        "Дата пункции": ["2026-05-01"] * 15,
        "Дата переноса": ["2026-05-06"] * 15,
        "Лечение": [
            "Криоперенос", "Только пункция", "Инсеминация", "ИКСИ", "ЭКО",
            "ЭКО", "ИКСИ", "ИКСИ", "Криоперенос", "ЭКО + ИКСИ",
            "ИКСИ", "ЭКО", "ИКСИ", "ИКСИ", "Получение ЯК (реципиент)",
        ],
        "IVF": [None, None, None, None, "Да", "Да", None, None, None, "Да",
                None, "Да", None, None, None],
        "ICSI": ["Да", None, None, "Да", None, None, "Да", "Да", "Да", "Да",
                 "Да", None, "Да", "Да", None],
        "Количество фолликулов": [12, 10, 8, 9, 7, 10, 8, None, 6, 11, 9, 8, 7, 9, 8],
        "Число ОКК": [10, 8, 6, 7, 5, 8, 6, 5, 5, 9, 7, 6, 6, 7, 6],
        "Число инсеминированных": [8, 6, 4, None, 5, 6, 4, 4, 4, 7, 5, 4, 4, 5, 4],
        "Число MII": [8, 6, 4, 5, 4, None, None, 4, 4, 6, 4, 4, 4, 5, 4],
        "2 pN": [6, 4, 3, 4, 3, 4, 3, 3, 3, 5, 3, 3, 3, 4, 3],
        "Число дробящихся на 3 день": [5, 3, 2, 3, 2, 3, 2, 2, 2, 4, 2, 2, 2, 3, 2],
        "Число Bl": [3, 2, 1, 2, 1, 2, 1, 1, 1, 3, 1, 1, 1, 2, 1],
        "Число Bl хор.кач-ва": [2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1],
        "Исход переноса": [
            "Беременность +28 дней (одноплодная)", "Беременность не наступила",
            "Беременность не наступила", "Беременность не наступила", None,
            "Беременность не наступила", "Беременность не наступила", "Беременность не наступила",
            "Беременность не наступила", "Беременность не наступила",
            "Беременность не наступила; Беременность +28 дней (одноплодная); "
            "Беременность +28 дней (многоплодная)",
            "Беременность биохимическая", "Беременность не наступила",
            "Беременность не наступила", "Беременность не наступила",
        ],
        "День переноса": [5, 5, 5, 5, 5, 5, 5, 5, None, 5, 5, 5, 5, 5, 5],
        "Перенесено эмбрионов": [1] * 15,
        "Заморожено эмбрионов": [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        "Заморожено бластоцист": [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
        "Эмбриолог на пункции": ["Эмбриолог О"] * 12 + ["Эмбриолог О-приоритет", None] + ["Эмбриолог О"],
        "Эмбриолог, проводивший оплодотворение": (
            ["Эмбриолог Ф"] * 12 + [None, None] + ["Эмбриолог Ф"]
        ),
        "Эмбриолог на переносе": ["Эмбриолог Т"] * 12 + [None, "Эмбриолог Т-приоритет"] + ["Эмбриолог Т"],
        "Врач на пункции": ["Врач О"] * 13 + [None, "Врач О"],
        "Врач на переносе": ["Врач Т"] * 13 + ["Врач Т-приоритет", "Врач Т"],
        "АМГ": [1.2] * 15,
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _prepared():
    raw = _raw_rows()
    df, unknown = fertimed.prepare(raw, COLUMNS_CFG, SETTINGS_CFG)
    return df.set_index(df["patient_id"], drop=False)


def test_cohort_filter_drops_only_puncture_iui_and_missing_transfer_fields():
    df = _prepared()
    kept = set(df["patient_id"])
    assert "CASE-ONLY-PUNCTURE" not in kept        # "Только пункция" не в whitelist
    assert "CASE-IUI" not in kept                  # "Инсеминация" (ВМИ) не в whitelist
    assert "CASE-NO-INSEMINATED" not in kept       # пусто "Число инсеминированных"
    assert "CASE-NO-OUTCOME" not in kept           # пусто "Исход переноса"
    assert "CASE-CRYO-OK" in kept                  # криоперенос с данными — валиден


def test_follicles_fallback_is_model_only_and_leaves_raw_follicles_missing():
    """Подстановка фолликулы<-ОКК больше НЕ идёт в канонический "follicles":
    иначе OCCR=OCC/фолликулы считался бы по производному знаменателю и давал
    ровно 1.0. Значение живёт отдельной колонкой и подставляется только в
    model_input (scripts/run_analysis.py)."""
    df = _prepared()
    row = df.loc["CASE-FOLLICLES-FALLBACK"]
    assert pd.isna(row["follicles"])                                    # сырое остаётся пропуском
    assert row["occ"] == 5
    assert row[fertimed.FOLLICLES_ML_FALLBACK_COLUMN] == 5              # вход модели -> ОКК


def test_occ_fallback_is_model_only_and_chains_into_follicles():
    """Цепочка MII->OCC->фолликулы: ЭКО-цикл без ОКК и без фолликулов получает
    для моделей ОКК=модельный MII, а фолликулы=этот же ОКК. Лабораторные
    "occ"/"follicles" при этом остаются пропусками."""
    df = _prepared()
    row = df.loc["CASE-IVF-MII-FALLBACK"]
    if pd.isna(row["occ"]):
        assert row[fertimed.OCC_ML_FALLBACK_COLUMN] == row[fertimed.MII_ML_FALLBACK_COLUMN]
        if pd.isna(row["follicles"]):
            assert row[fertimed.FOLLICLES_ML_FALLBACK_COLUMN] == row[fertimed.OCC_ML_FALLBACK_COLUMN]


def test_mii_ml_fallback_only_for_ivf_rows_not_icsi():
    df = _prepared()
    ivf_row = df.loc["CASE-IVF-MII-FALLBACK"]
    icsi_row = df.loc["CASE-ICSI-NO-MII-FALLBACK"]
    assert pd.isna(ivf_row["mii"])                                     # реальный MII неизвестен
    assert ivf_row[fertimed.MII_ML_FALLBACK_COLUMN] == ivf_row["inseminated"]  # но вход модели -> inseminated
    assert pd.isna(icsi_row["mii"])
    assert pd.isna(icsi_row[fertimed.MII_ML_FALLBACK_COLUMN])          # для ИКСИ fallback не применяется


def test_attempt_no_and_transfer_day_defaults():
    df = _prepared()
    row = df.loc["CASE-DEFAULTS"]
    assert row["attempt_no"] == 1
    assert row["transfer_day"] == 5


def test_frozen_is_sum_of_embryos_and_blastocysts():
    df = _prepared()
    row = df.loc["CASE-FROZEN-SUM"]
    assert row["frozen"] == 5


def test_outcome_encoding_positive_negative_composite_and_biochemical():
    df = _prepared()
    assert df.loc["CASE-CRYO-OK", "Preg"] == 1.0
    assert df.loc["CASE-BIOCHEMICAL", "Preg"] == 0.0
    # артефакт выгрузки (три исхода через ";") -> 1, т.к. есть хотя бы один "+28"
    assert df.loc["CASE-COMPOSITE-OUTCOME", "Preg"] == 1.0


def test_embryologist_priority_fert_then_opu_then_et():
    df = _prepared()
    normal = df.loc["CASE-CRYO-OK"]
    assert normal["embryologist"] == "Эмбриолог Ф"       # оплодотворение в приоритете
    fallback_to_opu = df.loc["CASE-EMBRYOLOGIST-PRIORITY"]
    assert fallback_to_opu["embryologist"] == "Эмбриолог О-приоритет"  # оплодотворение пусто -> пункция
    fallback_to_et = df.loc["CASE-DOCTOR-PRIORITY"]
    assert fallback_to_et["embryologist"] == "Эмбриолог Т-приоритет"   # оплодотворение и пункция пусты -> перенос


def test_doctor_priority_opu_then_et():
    df = _prepared()
    normal = df.loc["CASE-CRYO-OK"]
    assert normal["doctor"] == "Врач О"
    fallback = df.loc["CASE-EMBRYOLOGIST-PRIORITY"]
    assert fallback["doctor"] == "Врач О"                 # пункция заполнена -> берём её
    fallback_to_et = df.loc["CASE-DOCTOR-PRIORITY"]
    assert fallback_to_et["doctor"] == "Врач Т-приоритет"  # пункция пуста -> перенос


def test_age_text_parsed_to_number():
    df = _prepared()
    assert df.loc["CASE-AGE-TEXT", "age"] == 26.0
    assert df.loc["CASE-CRYO-OK", "age"] == 34.0


def test_fertilization_method_from_flags_not_from_treatment_label():
    df = _prepared()
    # "Получение ЯК (реципиент)" в графе "Лечение", но ни IVF, ни ICSI не стоит "Да"
    row = df.loc["CASE-AGE-TEXT"]
    assert pd.isna(row["fertilization_method"])


def test_missing_required_column_raises():
    raw = _raw_rows().drop(columns=["Число ОКК"])
    with pytest.raises(ValueError):
        fertimed.prepare(raw, COLUMNS_CFG, SETTINGS_CFG)


def test_unrecognized_outcome_text_becomes_missing_and_warns():
    raw = _raw_rows()
    raw.loc[raw["Номер карты пациента"] == "CASE-CRYO-OK", "Исход переноса"] = "Что-то новое"
    with pytest.warns(UserWarning, match="Нераспознанный текст исхода"):
        df, _ = fertimed.prepare(raw, COLUMNS_CFG, SETTINGS_CFG)
    # эта строка выпадает из финального датасета: Preg=NaN уже после того как
    # прошла фильтр когорты (исход был непустым), но остаётся неклассифицированной
    row = df.loc[df["patient_id"] == "CASE-CRYO-OK"].iloc[0]
    assert pd.isna(row["Preg"])


def _wide_raw() -> pd.DataFrame:
    """Та же фикстура плюс "Причина завершения программы ВРТ" и строки без исхода.

    Пустой исход у свежих циклов — не искусственный случай, а норма июньской
    выгрузки: заполненный исход там стоит почти только у криопереносов.
    """
    raw = _raw_rows()
    raw["Причина завершения программы ВРТ"] = [
        "Перенос эмбрионов", "Криоконсервация ооцитов", "Инсеминация",
        "Остановка развития эмбрионов", "Остановка развития эмбрионов",
        None, "ПГТ", "ПГТ", "Перенос эмбрионов", "ПГТ",
        "ПГТ", "ПГТ", "ПГТ", "ПГТ", "ПГТ",
    ]
    # свежие циклы без переноса: у половины строк исход пуст
    raw.loc[3:7, "Исход переноса"] = None
    # закрытый цикл, у которого бластоцист не получилось
    raw.loc[3, "Число Bl"] = None
    raw.loc[3, "Число Bl хор.кач-ва"] = None
    # цикл, ещё идущий на момент выгрузки (причина завершения пуста)
    raw.loc[5, "Число Bl"] = None
    raw.loc[5, "Число Bl хор.кач-ва"] = None
    return raw


def test_prepare_wide_keeps_cycles_without_transfer_outcome():
    """Лабораторные KPI считаются по всем циклам, а не только по переносам."""
    wide, _ = fertimed.prepare_wide(_wide_raw(), COLUMNS_CFG, SETTINGS_CFG)
    assert wide["transfer_outcome"].isna().any()          # строки без исхода сохранены
    assert wide["Preg"].isna().any()                      # и это не ошибка кодирования
    cohort, _ = fertimed.prepare(_wide_raw(), COLUMNS_CFG, SETTINGS_CFG)
    assert len(wide) > len(cohort)


def test_prepare_wide_drops_only_cryo_and_insemination():
    wide, _ = fertimed.prepare_wide(_wide_raw(), COLUMNS_CFG, SETTINGS_CFG)
    treatments = set(wide["treatment_type"])
    assert "Криоперенос" not in treatments      # пункция уже есть отдельной строкой
    assert "Инсеминация" not in treatments      # ВМИ — другая модальность
    assert "Только пункция" in treatments       # остаётся ради OCCR/MIIR врача
    assert {"ИКСИ", "ЭКО", "ЭКО + ИКСИ", "Получение ЯК (реципиент)"} <= treatments


def test_prepare_wide_reads_blank_counts_of_closed_cycle_as_observed_zero():
    wide, _ = fertimed.prepare_wide(_wide_raw(), COLUMNS_CFG, SETTINGS_CFG)
    by_id = wide.set_index("patient_id")
    closed = by_id.loc["CASE-NO-INSEMINATED"]              # причина: остановка развития
    assert float(closed["bl_total"]) == 0.0
    assert float(closed["bl_good"]) == 0.0
    still_open = by_id.loc["CASE-IVF-MII-FALLBACK"]        # причина завершения пуста
    assert bool(pd.isna(still_open["bl_total"]))


def test_prepare_wide_never_manufactures_two_pn_zero():
    """2PN=0 исключает строку по неподвижному решению владельца — производить
    такие строки из пропусков нельзя даже у закрытого цикла."""
    raw = _wide_raw()
    raw.loc[3, "2 pN"] = None
    wide, _ = fertimed.prepare_wide(raw, COLUMNS_CFG, SETTINGS_CFG)
    row = wide.set_index("patient_id").loc["CASE-NO-INSEMINATED"]
    assert bool(pd.isna(row["two_pn"]))
    assert bool(pd.isna(row["bl_total"]))   # предшествующая стадия неизвестна -> не заполняем
