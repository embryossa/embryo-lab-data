"""Синтетические тесты адаптера PERSONA (src/embryo/clinics/persona.py, docs/PERSONA.md):
лист "нов<год>" с заголовком не в первой строке, блоки кЭКО/И/DO/ЗО/∑, раскладка
смешанного цикла на две подстроки, исход по УЗИ, оценка AFC, межфайловые дубли и
регистрация клиники в scripts/run_analysis.py."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from embryo import ingest
from embryo.clinics import persona

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_analysis import (_CLINIC_NORMALIZERS, _CLINIC_WIDE_FROM_SOURCE,  # noqa: E402
                          _add_opu_aliases, _dedupe_cycle_keys)

ROOT = Path(__file__).resolve().parents[1]
COLUMNS_CFG = ingest.load_config(str(ROOT / "config/clinics/persona/columns.yaml"))
SETTINGS_CFG = {
    "persona": {"afc_per_occ": 1.16, "bmi_plausible": [12, 70]},
    "amh_impute": {"a_afc": 0.16, "b_occ": 0.05, "c_age": 0.08, "intercept": 0.5, "floor": 0.1, "ceil": 12.0},
    "bmi_impute": {"constant": 24.0},
}
STEMS = ["Получено ооцитов", "Зрелых (MII)", "Норм.опл.", "атрез.", "Аномал.опл.", "Дробление",
         "Выход б/ц", "Бласт н.луч качества  минимум 3ВВ", "ПЭ 3-4 сутки", "ПЭ 5-6 сутки", "n заморож.эмбр"]
FIELDS = ["occ", "mii", "two_pn", "degrade", "abn", "cleavage_d3", "bl_total", "bl_good", "et34", "et56", "frozen"]

# Три программы: смешанный ЭКО/ИКСИ с переносом и клинической беременностью;
# ИКСИ + донорские ооциты, перенос, ХГЧ отр; криоперенос без воронки.
PROGRAMS = [
    {"Категория программы": "ТВП", "ФИО пациента": "Тестова Анна", "Год рождения": "1990-01-01",
     "Дата Программы": "2024-03-01", "Возр пациентов": 34, "ИМТ жены": 442, "АМГ": "2,5",
     "Номер попытки": 1, "эмбр опл": "XXX", "Врач ТВП": "YYY", "ОП/ЕТ": "ЕТ",
     "Дата переноса": "2024-03-06", "День ПЕ": 5, "результат ХГЧ": "пол", "результат УЗИ 2": "пол",
     "кЭКО": [5, 4, 3, 0, 0, 3, 2, 1, 0, 1, 1], "И": [6, 5, 4, 1, 0, 4, 2, 1, 0, 0, 2]},
    {"Категория программы": "ТВП", "ФИО пациента": "Пробная Вера", "Год рождения": "1985-05-05",
     "Дата Программы": "2024-04-10", "Возр пациентов": 39, "ИМТ жены": 23.5, "АМГ": None,
     "Номер попытки": 0, "ОП/ЕТ": "ЕТ", "Дата переноса": "2024-04-15", "День ПЕ": 5,
     "результат ХГЧ": "отр", "КОД ДС/ ФИО": "ДС-1",
     "И": [4, 3, 3, 0, 0, 3, 1, 0, 0, 1, 0], "DO": [8, 7, 6, 0, 0, 6, 4, 2, 0, 0, 3]},
    {"Категория программы": "КРИО ПЕ", "ФИО пациента": "Тестова Анна", "Год рождения": "1990-01-01",
     "Дата Программы": "2024-06-01", "Возр пациентов": 34, "ОП/ЕТ": "ЕТ",
     "результат ХГЧ": "пол", "результат УЗИ 2": "отр"},
]


def _sheet_rows() -> list[list]:
    header = ["Категория программы", "ФИО пациента", "Год рождения", "Дата Программы", "Возр пациентов",
              "ИМТ жены", "АМГ", "Номер попытки", "эмбр опл", "", "Эмбриолог ТВП", "ОП/ЕТ",
              "Дата переноса", "День ПЕ", "результат ХГЧ", "результат УЗИ 2", "КОД ДС/ ФИО"]
    blocks = {"кЭКО": "кЭКО", "И": "И", "DO": "DO", "∑": "∑"}
    for prefix in blocks:
        for stem in STEMS:
            header.append("атрез." if (prefix == "кЭКО" and stem == "атрез.") else f"{prefix} {stem}")
    header += ["ЗО Получено ооцитов пац /ДО", "ЗО Зрелых (MII)"]
    rows = [["Отчёт PERSONA"] + [None] * (len(header) - 1), header]
    for p in PROGRAMS:
        row = {h: p.get(h) for h in header}
        row[""] = p.get("Врач ТВП")
        for prefix in ("кЭКО", "И", "DO"):
            for stem, value in zip(STEMS, p.get(prefix) or [None] * len(STEMS)):
                row["атрез." if (prefix == "кЭКО" and stem == "атрез.") else f"{prefix} {stem}"] = value
        for i, stem in enumerate(STEMS):
            vals = [p[b][i] for b in ("кЭКО", "И", "DO") if p.get(b)]
            row[f"∑ {stem}"] = sum(vals) if vals else 0
        rows.append([row.get(h) for h in header])
    rows.append([None] * len(header))      # протянутое форматирование
    return rows


@pytest.fixture()
def workbook(tmp_path) -> Path:
    path = tmp_path / "Нов. отчет 2024.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame([["шаблон"]]).to_excel(writer, sheet_name="исходник образец", header=False, index=False)
        pd.DataFrame(_sheet_rows()).to_excel(writer, sheet_name="нов2024", header=False, index=False)
    return path


@pytest.fixture()
def staged(workbook) -> pd.DataFrame:
    return persona.stage_export(persona.read_export(workbook, COLUMNS_CFG), COLUMNS_CFG, SETTINGS_CFG)


def test_read_export_finds_sheet_header_and_blank_doctor(workbook):
    raw = persona.read_export(workbook, COLUMNS_CFG)
    assert len(raw) == 3                              # пустая строка-паддинг отброшена
    assert "Врач ТВП" in raw.columns                  # пустой заголовок перед "Эмбриолог ТВП"
    assert set(raw["_sheet_year"]) == {2024}


def test_stage_blocks_outcome_afc_and_plausibility(staged):
    mixed, donor, fet = staged.iloc[0], staged.iloc[1], staged.iloc[2]
    assert mixed["ivf_two_pn"] == 3 and mixed["icsi_two_pn"] == 4 and mixed["ivf_degrade"] == 0
    assert mixed["Preg"] == 1.0 and donor["Preg"] == 0.0 and fet["Preg"] == 0.0
    assert np.isnan(mixed["bmi"])                     # ИМТ 442 — ошибка ввода -> пропуск
    assert mixed["amh"] == pytest.approx(2.5)         # "2,5" -> 2.5
    assert np.isnan(donor["amh"])                     # пусто остаётся пустым
    assert np.isnan(donor["attempt_no"])              # попытка 0 -> пропуск
    assert mixed["afc_from_occ"] == round((5 + 6) * 1.16)
    assert donor["afc_from_occ"] == round(4 * 1.16)   # донорские ОКК в оценку AFC не входят
    assert donor["sperm_source"] == "donor"
    assert pd.isna(fet["date_opu"]) and not fet["has_lab"]


def test_program_frame_sums_lab_blocks(staged):
    programs = persona.program_frame(staged)
    assert len(programs) == 2                         # криоперенос без воронки не входит
    mixed, donor = programs.iloc[0], programs.iloc[1]
    assert mixed["two_pn"] == 7 and mixed["fertilization_method"] == "IVF+ICSI"
    assert mixed["inseminated"] == 5 + 5              # ОКК кЭКО + MII ИКСИ
    assert mixed["transferred"] == 1
    assert donor["oocyte_source"] == "own+donor" and pd.isna(donor["inseminated"])


def test_lab_parts_split_mixed_cycle_and_keep_unique_keys(staged):
    wide, unknown = persona.prepare_wide(staged, COLUMNS_CFG, SETTINGS_CFG)
    assert list(wide["lab_part"]) == ["ivf", "icsi", "icsi", "donor"]
    assert list(wide["fertilization_method"].fillna("—")) == ["IVF", "ICSI", "ICSI", "—"]
    assert wide.loc[0, "inseminated"] == 5 and wide.loc[1, "inseminated"] == 5
    assert unknown == []
    aliased = _add_opu_aliases(wide, clinic="persona")
    deduped, _ = _dedupe_cycle_keys(aliased, "test")
    assert len(deduped) == 4                          # подстроки одной пункции не схлопнулись


def test_cross_file_duplicate_keeps_own_year(tmp_path, workbook):
    other = tmp_path / "Новый отчет 2023.xlsx"
    rows = _sheet_rows()
    with pd.ExcelWriter(other) as writer:
        pd.DataFrame(rows[:3]).to_excel(writer, sheet_name="нов2023", header=False, index=False)
    staged, report = persona.stage_exports([other, workbook], COLUMNS_CFG, SETTINGS_CFG)
    assert report["cross_file_duplicates_dropped"] == 1
    assert len(staged) == 3 and set(staged["sheet_year"]) == {2024}


def test_newer_book_of_the_same_year_supersedes_the_older(tmp_path, workbook):
    # Повторная выгрузка 2024 (обновлённые исходы) рядом со старой: старая книга пропускается
    # целиком, иначе программы года задвоились бы.
    import os
    import time
    newer = tmp_path / "Нов. отчет 2024 (обновление).xlsx"
    with pd.ExcelWriter(newer) as writer:
        pd.DataFrame(_sheet_rows()).to_excel(writer, sheet_name="нов2024", header=False, index=False)
    old_time = time.time() - 3600
    os.utime(workbook, (old_time, old_time))
    staged, report = persona.stage_exports([workbook, newer], COLUMNS_CFG, SETTINGS_CFG)
    assert len(staged) == 3
    assert len(report["books_superseded"]) == 1 and Path(workbook).name in report["books_superseded"][0]


def test_demo_export_has_no_phi(staged):
    demo = persona.demo_export(staged, b"secret", COLUMNS_CFG)
    assert len(demo) == 4 and demo["Cycle ID"].is_unique
    text = demo.to_csv(index=False)
    assert "Тестова" not in text and "1990-01-01" not in text
    assert set(demo["Oocyte source"]) == {"Own", "Donor"}


def test_registered_in_run_analysis():
    assert _CLINIC_NORMALIZERS["persona"] is persona.prepare
    assert _CLINIC_WIDE_FROM_SOURCE["persona"] is persona.prepare_wide


def test_freeze_all_stays_in_competency_tables_unless_flag_configured(staged):
    from embryo import kpi
    s = staged.copy()
    s.loc[1, "freeze_all"] = True                     # вторая программа: ОП, эмбрионы заморожены
    wide, _ = persona.prepare_wide(s, COLUMNS_CFG, SETTINGS_CFG)
    # PERSONA: таблицы компетентности — на всей лабораторной работе (решение владельца).
    assert not kpi.competency_exclusion_mask(wide, COLUMNS_CFG).any()
    # Общая настройка kpi: флаг-колонка исключает обе строки-блока программы.
    cfg = {"competency_exclusion": {**COLUMNS_CFG["competency_exclusion"],
                                    "freeze_all_flag_column": "freeze_all"}}
    assert list(kpi.competency_exclusion_mask(wide, cfg)) == [False, False, True, True]


def test_stage_sets_freeze_all_from_cancel_and_frozen(workbook):
    raw = persona.read_export(workbook, COLUMNS_CFG)
    assert list(persona.stage_export(raw, COLUMNS_CFG, SETTINGS_CFG)["freeze_all"]) == [False, False, False]
    raw.loc[1, "ОП/ЕТ"] = "ОП"                        # перенос отменён, заморожено 0 + 3
    raw.loc[2, "ОП/ЕТ"] = "ОП"                        # криоперенос: своей заморозки нет
    assert list(persona.stage_export(raw, COLUMNS_CFG, SETTINGS_CFG)["freeze_all"]) == [False, True, False]


def test_afc_from_donor_occ_only_without_own_oocytes(workbook):
    raw = persona.read_export(workbook, COLUMNS_CFG)
    for col in [c for c in raw.columns if c.startswith("И ")]:
        raw.loc[1, col] = None                        # вторая программа — только донорские ооциты
    staged = persona.stage_export(raw, COLUMNS_CFG, SETTINGS_CFG)
    assert staged.loc[1, "afc_from_occ"] == round(8 * 1.16) and bool(staged.loc[1, "afc_from_donor_occ"])
    assert not bool(staged.loc[0, "afc_from_donor_occ"])
    assert pd.isna(staged.loc[2, "afc_from_occ"])     # криоперенос: ОКК нет


def test_laya_export_is_cycle_level_with_outcome_and_no_phi(staged):
    laya = persona.laya_export(staged, b"secret")
    assert len(laya) == 2                             # обе свежие программы с переносом и исходом
    assert list(laya["2PN"]) == [7, 9] and list(laya["Inseminated"].fillna(-1)) == [10, -1]
    assert list(laya["Preg"]) == [1.0, 0.0]
    assert list(laya["Oocyte source"]) == ["Own", "Own+Donor"]
    assert "Тестова" not in laya.to_csv(index=False)


def test_persona_year_run_tags_outputs_but_keeps_secret_and_inputs():
    import yaml
    from run_analysis import _persona_year_run
    cfg = yaml.safe_load((ROOT / "config/clinics/persona/settings.yaml").read_text(encoding="utf-8"))
    secret = cfg["privacy"]["secret_file"]
    path = _persona_year_run(cfg, "2024", None)
    assert path == "data/processed/persona/local_private/staged/persona_2024.xlsx"
    assert cfg["paths"]["processed_dir"] == "data/processed/persona/2024"
    assert cfg["paths"]["local_private_dir"] == "data/processed/persona/2024/local_private"
    assert cfg["paths"]["agent_packets_dir"] == "data/processed/persona/2024/agent_packets"
    assert cfg["privacy"]["crosswalk_file"].startswith("data/processed/persona/2024/")
    assert cfg["external_tools"]["tabpfn"]["output"].startswith("data/processed/persona/2024/")
    assert cfg["privacy"]["secret_file"] == secret                      # cycle_uid общий для всех лет
    assert cfg["paths"]["staged_dir"] == "data/processed/persona/local_private/staged"
    with pytest.raises(SystemExit):
        _persona_year_run(cfg, None, None)
