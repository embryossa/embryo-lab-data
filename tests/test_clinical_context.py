"""Описательный клинический слой (src/embryo/clinical.py) и защита КАФ Fertimed.

Проверяется ровно то, что дороже всего стоит ошибкой: слой не протекает во вход
моделей, знаменатель заполненности честен, группы МКБ считаются по длиннейшему
префиксу, а нефизиологичный КАФ прежней выгрузки гасится, а не подставляется.
"""
from pathlib import Path

import pandas as pd
import pytest

from embryo import clinical, ingest
from embryo.clinics import fertimed

ROOT = Path(__file__).resolve().parents[1]
COLUMNS_CFG = ingest.load_config(str(ROOT / "config/clinics/fertimed/columns.yaml"))
GROUPS = clinical.load_diagnosis_groups(ROOT / "config/clinics/fertimed/diagnosis_groups.yaml")


# --------------------------------------------------------------------------- #
# справочник МКБ-10
# --------------------------------------------------------------------------- #
def test_diagnosis_config_loads():
    assert GROUPS and GROUPS["prefixes"] and GROUPS["fallback"]


def test_longest_prefix_wins():
    # N97.1 (трубный) точнее, чем общий N97 — иначе весь N97 схлопнется в одну корзину
    assert clinical.classify_code("N97.1", GROUPS) == "трубно-перитонеальный фактор"
    assert clinical.classify_code("N97.4", GROUPS) == "мужской фактор"
    assert clinical.classify_code("N97", GROUPS) == "бесплодие, рубрика N97 без уточнения"


def test_unknown_code_goes_to_fallback_not_to_a_guess():
    assert clinical.classify_code("Q99.9", GROUPS) == GROUPS["fallback"]


def test_split_codes_handles_multi_value_cell_and_blanks():
    assert clinical.split_codes("N80.1; D25.2") == ["N80.1", "D25.2"]
    assert clinical.split_codes("N80.1,D25.2") == ["N80.1", "D25.2"]
    assert clinical.split_codes(None) == []
    assert clinical.split_codes(float("nan")) == []
    assert clinical.split_codes("   ") == []


def test_classify_codes_deduplicates_groups():
    # два разных кода одной корзины не должны удваивать группу
    assert clinical.classify_codes("N97.1; N73.6", GROUPS) == ["трубно-перитонеальный фактор"]


# --------------------------------------------------------------------------- #
# по-цикловый кадр и когортная сводка
# --------------------------------------------------------------------------- #
def _clinical_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "cycle_uid": ["CYC-A", "CYC-B", "CYC-C"],
        "date_opu": ["14.06.2026", "20.06.2026", "25.06.2026"],
        "stim_protocol_type": ["Короткий с антагонистами", "В естественном цикле", " Длинный "],
        "stim_fsh_dose": [2250, None, 3000],
        "stim_duration_days": [10, None, 0],          # 0 -> суточная доза не определена
        "stim_trigger": ["Овитрель", None, ""],
        "stim_protocol_start_day": [2, None, 3],
        "stim_stimulation_start_day": [2, None, 3],
        "clin_opu_cycle_day": [14, 16, 15],
        "clin_afc": [12, None, 25],
        "clin_fsh_basal": [6.0, None, 9.0],
        "clin_cycle_start_date": ["01.06.2026", None, "05.06.2026"],
        "clin_lmp_date": ["01.06.2026", None, None],
        "clin_dx_primary": ["N97.1", "N97.8", "N80.0"],
        "clin_dx_secondary": ["D25.1; N80.0", None, ""],
        "clin_termination_reason": ["Перенос эмбрионов", "Перенос эмбрионов", None],
    })


def _lab_population() -> pd.DataFrame:
    return pd.DataFrame({"clin_termination_reason": [
        "Перенос эмбрионов", "Перенос эмбрионов", "Остановка развития эмбрионов",
        "Отсутствие оплодотворения", "Овуляция", None,
    ]})


def test_cycle_frame_exposes_only_allowlisted_columns():
    cycles = clinical.build_cycle_frame(_clinical_frame(), GROUPS)
    assert list(cycles.columns)[0] == "cycle_uid"
    assert set(cycles.columns) <= {"cycle_uid", *clinical.PACKET_COLUMNS}
    # сырые даты и сырые строки диагноза наружу не выходят
    for leaked in ("clin_cycle_start_date", "clin_lmp_date", "clin_dx_primary", "clin_dx_secondary"):
        assert leaked not in cycles.columns


def test_dates_are_parsed_day_first_not_month_first():
    cycles = clinical.build_cycle_frame(_clinical_frame(), GROUPS)
    row = cycles.set_index("cycle_uid").loc["CYC-A"]
    # 01.06.2026 -> 14.06.2026 = 13 дней; при month-first разборе вышло бы иначе
    assert row["clin_days_start_to_opu"] == 13
    assert row["clin_days_lmp_to_opu"] == 13


def test_negative_start_to_opu_is_dropped_not_reported_as_a_duration():
    """Криоперенос: 1С несёт исходную дату пункции при дате старта новой программы.

    Интервал между разными циклами не должен доехать до агента как срок.
    """
    frame = _clinical_frame()
    frame.loc[frame["cycle_uid"] == "CYC-C", "clin_cycle_start_date"] = "20.09.2026"  # позже пункции
    cycles = clinical.build_cycle_frame(frame, GROUPS).set_index("cycle_uid")
    assert pd.isna(cycles.loc["CYC-C", "clin_days_start_to_opu"])
    assert cycles.loc["CYC-A", "clin_days_start_to_opu"] == 13   # корректный остался
    summary = clinical.build_cohort_summary(clinical.build_cycle_frame(frame, GROUPS))
    assert summary["days_from_cycle_start_to_opu"]["n_known"] == 1
    assert "days_from_cycle_start_to_opu_note" in summary


def test_dose_per_day_undefined_when_duration_missing_or_zero():
    cycles = clinical.build_cycle_frame(_clinical_frame(), GROUPS).set_index("cycle_uid")
    assert cycles.loc["CYC-A", "stim_fsh_dose_per_day"] == pytest.approx(225.0)
    assert pd.isna(cycles.loc["CYC-B", "stim_fsh_dose_per_day"])   # длительности нет
    assert pd.isna(cycles.loc["CYC-C", "stim_fsh_dose_per_day"])   # деление на ноль


def test_blank_trigger_is_missing_not_a_category():
    cycles = clinical.build_cycle_frame(_clinical_frame(), GROUPS).set_index("cycle_uid")
    assert pd.isna(cycles.loc["CYC-C", "stim_trigger"])            # "" -> пропуск
    assert cycles.loc["CYC-C", "stim_protocol_type"] == "Длинный"  # но пробелы обрезаны


def test_all_groups_merge_primary_and_secondary():
    cycles = clinical.build_cycle_frame(_clinical_frame(), GROUPS).set_index("cycle_uid")
    assert cycles.loc["CYC-A", "clin_dx_primary_code"] == "N97.1"
    assert set(cycles.loc["CYC-A", "clin_dx_groups_all"]) == {
        "трубно-перитонеальный фактор", "миома матки", "эндометриоз"}


def test_cohort_summary_reports_denominator_for_every_field():
    cycles = clinical.build_cycle_frame(_clinical_frame(), GROUPS)
    summary = clinical.build_cohort_summary(cycles)
    assert summary["n_cycles"] == 3
    # триггер известен у одной строки из трёх — знаменатель обязан быть виден
    assert summary["trigger"]["n_known"] == 1
    assert summary["trigger"]["n_total"] == 3
    assert summary["fsh_total_dose_iu"]["n_known"] == 2
    assert summary["afc"]["n_known"] == 2
    assert "fill_rate_note" in summary


def test_cohort_diagnosis_counts_allow_multiple_groups_per_cycle():
    cycles = clinical.build_cycle_frame(_clinical_frame(), GROUPS)
    groups = clinical.build_cohort_summary(cycles)["diagnosis_groups"]
    assert groups["n_cycles_with_any_code"] == 3
    # CYC-A даёт три группы, поэтому сумма счётчиков больше числа циклов
    assert sum(groups["counts_by_group"].values()) > groups["n_cycles_with_any_code"]


def test_dose_summary_is_split_by_protocol():
    cycles = clinical.build_cycle_frame(_clinical_frame(), GROUPS)
    by_protocol = clinical.build_cohort_summary(cycles)["fsh_total_dose_by_protocol"]
    assert by_protocol["В естественном цикле"]["n_known"] == 0   # дозы у естественного цикла нет
    assert by_protocol["Короткий с антагонистами"]["median"] == 2250.0


def test_empty_cohort_does_not_crash():
    empty = clinical.build_cycle_frame(
        pd.DataFrame({"cycle_uid": pd.Series(dtype="object")}), GROUPS)
    assert clinical.build_cohort_summary(empty)["n_cycles"] == 0


def test_missing_diagnosis_config_yields_raw_codes_not_guessed_groups():
    empty_groups = clinical.load_diagnosis_groups(ROOT / "config/clinics/fertimed/nope.yaml")
    assert empty_groups == {}
    cycles = clinical.build_cycle_frame(_clinical_frame(), empty_groups)
    assert cycles.set_index("cycle_uid").loc["CYC-A", "clin_dx_primary_code"] == "N97.1"
    assert cycles.set_index("cycle_uid").loc["CYC-A", "clin_dx_groups_all"] == []


def test_termination_reason_kept_per_cycle_and_summarised():
    cycles = clinical.build_cycle_frame(_clinical_frame(), GROUPS).set_index("cycle_uid")
    assert cycles.loc["CYC-A", "clin_termination_reason"] == "Перенос эмбрионов"
    assert pd.isna(cycles.loc["CYC-C", "clin_termination_reason"])


def test_export_termination_block_carries_both_denominators_side_by_side():
    """Содержательные причины живут ВНЕ когорты отчёта — до когортного фильтра.

    Первая версия блока получала «широкую lab-популяцию» пайплайна, которая у
    Fertimed уже отфильтрована клиничным _cohort_mask; блок выходил с подписью
    «без гейта» и ровно тем же составом, что когорта отчёта, то есть терял ровно
    те причины, ради которых существует. Оба знаменателя обязаны быть рядом.
    """
    _, summary = clinical.build_clinical_context(
        _clinical_frame(), GROUPS, export_population=_lab_population())
    block = summary["export_population_termination"]
    assert block["n_rows_export"] == 6                  # вся выгрузка
    assert block["n_cycles_report_cohort"] == 3         # когорта отчёта, рядом
    assert summary["n_cycles"] == 3
    counts = block["termination_reason"]["counts"]
    assert counts["Остановка развития эмбрионов"] == 1
    assert counts["Отсутствие оплодотворения"] == 1
    assert "до когортного фильтра" in block["population"]


def test_export_termination_block_absent_when_not_supplied():
    _, summary = clinical.build_clinical_context(_clinical_frame(), GROUPS)
    assert "export_population_termination" not in summary


def test_has_clinical_columns_detects_layer():
    assert clinical.has_clinical_columns(_clinical_frame())
    assert not clinical.has_clinical_columns(pd.DataFrame({"occ": [1], "cycle_uid": ["X"]}))


# --------------------------------------------------------------------------- #
# защита КАФ на стороне адаптера Fertimed
# --------------------------------------------------------------------------- #
def test_afc_guard_keeps_physiological_and_drops_billing_sums():
    frame = pd.DataFrame({"clin_afc": [12, 30, 250344, 270, None, -3]})
    with pytest.warns(UserWarning, match="КАФ вне физиологичного диапазона"):
        guarded = fertimed._guard_clinical_afc(frame)
    assert list(guarded[:2]) == [12.0, 30.0]
    assert guarded[2:].isna().all()


def test_afc_guard_is_silent_on_clean_data():
    import warnings as _warnings
    frame = pd.DataFrame({"clin_afc": [5, 14, None]})
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")          # любое предупреждение -> провал
        guarded = fertimed._guard_clinical_afc(frame)
    assert guarded.tolist()[:2] == [5.0, 14.0]


def test_kaf_maps_to_clin_afc_and_never_to_canonical_afc():
    """Владелец (2026-07-31): КАФ идёт только в интерпретацию, не в модели.

    Канонический "afc" уходит во вход DT (μ первой стадии ZINB) и в
    normalize.impute_amh — попадание КАФ туда молча изменило бы прогнозы.
    """
    aliases = {name: cfg.get("aliases", []) for name, cfg in COLUMNS_CFG["canonical"].items()}
    assert "КАФ" in aliases["clin_afc"]
    assert "afc" not in aliases                       # канонического afc у Fertimed нет вовсе
    for name, alias_list in aliases.items():
        if name != "clin_afc":
            assert "КАФ" not in alias_list, f"КАФ протёк в {name}"


def test_clinical_columns_are_absent_from_model_input_aliases():
    """Слой не должен иметь пути во вход нейросетей."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_ra", ROOT / "scripts/run_analysis.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for canon in module._OPU_ALIASES:
        assert not str(canon).startswith(clinical.CLINICAL_PREFIXES)
    frame = pd.DataFrame({"cycle_uid": ["X"], "clin_afc": [12], "stim_fsh_dose": [2250]})
    aliased = module._add_opu_aliases(frame, clinic="fertimed")
    assert "Antral follicule count" not in aliased.columns
