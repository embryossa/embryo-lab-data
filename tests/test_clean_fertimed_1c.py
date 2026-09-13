"""Очиститель сырой выгрузки 1С Fertimed (.claude/skills/reformat-1c-report).

Скрипт самостоятельный и намеренно не импортирует src/embryo — поэтому его
правила НЕ проверяются тестами пайплайна и могут разъехаться с ним незаметно.
Отсюда две группы проверок: (1) правила очистки как таковые — фильтр когорты,
цепочка MII→ОКК→фолликулы, приоритет эмбриолога/врача, перекодировка исхода;
(2) совпадение констант импутации АМГ с config/clinics/fertimed/settings.yaml,
чтобы «одинаковая формула» оставалась одинаковой на деле.
"""
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / ".claude" / "skills" / "reformat-1c-report" / "clean_fertimed_1c.py"


def _load_module():
    spec = spec_from_file_location("clean_fertimed_1c", SCRIPT)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cleaner = _load_module()


def _raw(**overrides) -> pd.DataFrame:
    base = {
        "Лечение": ["ИКСИ", "Криоперенос", "Только пункция", "ЭКО", "Инсеминация", "ЭКО + ИКСИ"],
        "Исход переноса": ["Беременность +28 дней (одноплодная)", "Беременность не наступила",
                            "Беременность +28 дней", "", "Беременность не наступила",
                            "Биохимическая беременность"],
        "Число инсеминированных": [8, 6, 10, 5, 4, 7],
        "№ попытки": [1, None, 1, 1, 1, None],
        "День переноса": [5, None, 5, 5, 5, None],
        "Количество фолликулов": [12, None, 15, 8, 6, None],
        "Число ОКК": [10, None, 12, 6, 5, None],
        "Возраст": ["34 года", "26 лет", "31 год", "40 лет", "29 лет", "не указан"],
        "АМГ": [2.5, None, 3.0, 1.1, 2.0, None],
        "Число MII": [8, None, 10, None, 4, None],
        "Эмбриолог, проводивший оплодотворение": ["Emb A", None, "Emb C", "Emb D", "Emb E", None],
        "Эмбриолог на пункции": ["Emb A", None, "Emb C", "Emb D", "Emb E", "Emb F"],
        "Эмбриолог на переносе": ["Emb A", "Emb B", "Emb C", "Emb D", "Emb E", "Emb G"],
        "Врач на пункции": ["Dr A", None, "Dr C", "Dr D", "Dr E", None],
        "Врач на переносе": ["Dr A", "Dr B", "Dr C", "Dr D", "Dr E", "Dr F"],
        "Заморожено эмбрионов": [2, 0, 3, 1, 0, 1],
        "Заморожено бластоцист": [1, 2, 0, 0, 0, 2],
        "КАФ": [11, 9, 14, 7, 8, 10],
        "Донор ооцитов, код": ["", "", "", "", "", ""],   # не используется — должно уцелеть
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_cohort_filter_keeps_only_rows_with_outcome_and_allowed_treatment():
    kept, stats = cleaner.clean(_raw())
    # Отсеиваются: "Только пункция" (лечение вне когорты), пустой исход,
    # "Инсеминация" (лечение вне когорты). Остаются 3 строки.
    assert stats["rows_in"] == 6 and stats["rows_out"] == 3
    assert set(kept["Лечение"]) == {"ИКСИ", "Криоперенос", "ЭКО + ИКСИ"}


def test_row_without_inseminated_count_is_dropped():
    raw = _raw()
    raw.loc[0, "Число инсеминированных"] = None
    kept, stats = cleaner.clean(raw)
    assert stats["dropped_missing_inseminated"] == 1
    assert "ИКСИ" not in set(kept["Лечение"])


def test_column_names_and_order_are_preserved_exactly():
    raw = _raw()
    kept, _ = cleaner.clean(raw)
    assert list(kept.columns) == list(raw.columns)
    # Неиспользуемые колонки остаются на месте нетронутыми.
    assert "Донор ооцитов, код" in kept.columns and "КАФ" in kept.columns


def test_fill_chain_leaves_no_blanks_in_mii_occ_and_follicles():
    kept, stats = cleaner.clean(_raw())
    for col in ("Число MII", "Число ОКК", "Количество фолликулов"):
        assert pd.to_numeric(kept[col], errors="coerce").notna().all(), col
    # Криоперенос: MII пуст -> берётся Число инсеминированных (6), затем ОКК и
    # фолликулы наследуют его по цепочке.
    cryo = kept.loc[kept["Лечение"] == "Криоперенос"].iloc[0]
    assert cryo["Число MII"] == 6 and cryo["Число ОКК"] == 6 and cryo["Количество фолликулов"] == 6
    assert stats["mii_filled"] == 2


def test_afc_column_is_never_used_as_a_follicle_fallback():
    # "КАФ" заполняется в 1С ненадёжно — правило владельца: не использовать.
    raw = _raw()
    raw["КАФ"] = [99, 99, 99, 99, 99, 99]
    kept, _ = cleaner.clean(raw)
    assert 99 not in set(pd.to_numeric(kept["Количество фолликулов"], errors="coerce"))


def test_attempt_and_transfer_day_defaults():
    kept, stats = cleaner.clean(_raw())
    assert pd.to_numeric(kept["№ попытки"], errors="coerce").notna().all()
    cryo = kept.loc[kept["Лечение"] == "Криоперенос"].iloc[0]
    assert cryo["№ попытки"] == 1 and cryo["День переноса"] == 5
    assert stats["attempt_no_filled"] == 2 and stats["transfer_day_filled"] == 2


def test_embryologist_and_doctor_priority_chain():
    kept, _ = cleaner.clean(_raw())
    cryo = kept.loc[kept["Лечение"] == "Криоперенос"].iloc[0]
    # Пусто и в "на пункции" -> берётся "на переносе".
    assert cryo["Эмбриолог, проводивший оплодотворение"] == "Emb B"
    assert cryo["Врач на пункции"] == "Dr B"
    mixed = kept.loc[kept["Лечение"] == "ЭКО + ИКСИ"].iloc[0]
    # "на пункции" заполнен -> он приоритетнее "на переносе".
    assert mixed["Эмбриолог, проводивший оплодотворение"] == "Emb F"


def test_age_is_recomputed_to_a_number_and_unparsed_left_blank():
    kept, stats = cleaner.clean(_raw())
    ages = kept["Возраст"].tolist()
    assert 34 in ages and 26 in ages
    # "не указан" не парсится — остаётся пустым, а не выдумывается.
    assert pd.isna(ages[-1]) and stats["age_unparsed_left_blank"] == 1


def test_outcome_is_encoded_by_the_plus28_marker_only():
    kept, stats = cleaner.clean(_raw())
    outcomes = dict(zip(kept["Лечение"], kept["Исход переноса"]))
    assert outcomes["ИКСИ"] == 1                      # "+28 дней"
    assert outcomes["Криоперенос"] == 0
    # Биохимическая беременность — НЕ клиническая: 0.
    assert outcomes["ЭКО + ИКСИ"] == 0
    assert stats["outcome_pregnant"] == 1 and stats["outcome_not_pregnant"] == 2


def test_frozen_embryos_is_a_full_recompute_not_a_gap_fill():
    kept, _ = cleaner.clean(_raw())
    by_treatment = dict(zip(kept["Лечение"], kept["Заморожено эмбрионов"]))
    assert by_treatment["ИКСИ"] == 3          # 2 + 1, перезаписано несмотря на непустое значение
    assert by_treatment["Криоперенос"] == 2   # 0 + 2
    # Исходная колонка бластоцист остаётся нетронутой.
    assert kept.loc[kept["Лечение"] == "ИКСИ", "Заморожено бластоцист"].iloc[0] == 1


def test_amh_is_imputed_within_the_configured_bounds():
    kept, stats = cleaner.clean(_raw())
    amh = pd.to_numeric(kept["АМГ"], errors="coerce")
    assert amh.notna().all()
    assert (amh >= cleaner.AMH_FLOOR).all() and (amh <= cleaner.AMH_CEIL).all()
    assert stats["amh_imputed"] == 2
    # Реальное значение приоритетнее импутации.
    assert kept.loc[kept["Лечение"] == "ИКСИ", "АМГ"].iloc[0] == 2.5


def test_amh_constants_match_the_clinic_settings():
    """«Та же формула, что в пайплайне» — это должно быть правдой, а не намерением."""
    cfg = yaml.safe_load((REPO / "config" / "clinics" / "fertimed" / "settings.yaml")
                         .read_text(encoding="utf-8"))
    amh = cfg["amh_impute"]
    assert (cleaner.AMH_A_AFC, cleaner.AMH_B_OCC, cleaner.AMH_C_AGE) == (
        amh["a_afc"], amh["b_occ"], amh["c_age"])
    assert (cleaner.AMH_INTERCEPT, cleaner.AMH_FLOOR, cleaner.AMH_CEIL) == (
        amh["intercept"], amh["floor"], amh["ceil"])
    assert cleaner.BMI_CONSTANT == cfg["bmi_impute"]["constant"]


def test_export_in_a_foreign_format_is_rejected_with_the_missing_columns():
    with pytest.raises(ValueError, match="нет ожидаемых колонок"):
        cleaner.clean(pd.DataFrame({"Лечение": ["ИКСИ"]}))


def test_script_writes_cleaned_file_and_prints_its_path_last(tmp_path, capsys):
    source = tmp_path / "выгрузка.xlsx"
    _raw().to_excel(source, index=False)
    assert cleaner.main([str(source)]) == 0

    captured = capsys.readouterr()
    # Последняя строка stdout — ровно путь к результату: агент отдаёт её пользователю как есть.
    printed = captured.out.strip().splitlines()[-1]
    assert printed == str(tmp_path / "выгрузка_cleaned.xlsx")
    assert Path(printed).is_file()
    assert len(pd.read_excel(printed)) == 3
    assert "[ok] строк было: 6, осталось: 3" in captured.err


def test_script_reports_a_missing_input_instead_of_crashing(tmp_path, capsys):
    assert cleaner.main([str(tmp_path / "нет.xlsx")]) == 1
    assert "входной файл не найден" in capsys.readouterr().err


def test_script_honours_an_explicit_output_path(tmp_path):
    source = tmp_path / "выгрузка.xlsx"
    _raw().to_excel(source, index=False)
    target = tmp_path / "out" / "результат.xlsx"
    assert cleaner.main([str(source), "--output", str(target)]) == 0
    assert target.is_file()


def test_cleaned_suffix_is_the_one_the_pipeline_deliberately_ignores(tmp_path, capsys):
    """Основной пайплайн не должен подхватывать результат этого скилла из inbox.

    run_analysis._input_path отбрасывает файлы с суффиксом "_cleaned" — значит
    суффикс здесь обязан оставаться именно таким.
    """
    source = tmp_path / "выгрузка.xlsx"
    _raw().to_excel(source, index=False)
    cleaner.main([str(source)])
    printed = capsys.readouterr().out.strip().splitlines()[-1]
    assert Path(printed).stem.endswith("_cleaned")
