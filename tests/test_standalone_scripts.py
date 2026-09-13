"""Отдельные скрипты, живущие рядом с пайплайном, но не внутри /run-analysis.

  * scripts/skyfert_kpi_test.py       — KPI/данные Skyfert без DT/TabPFN и отчёта
  * scripts/adhoc_ggrc_2025_reformat.py — разовый переформаттер выгрузок GGRC 2025
  * scripts/build_model_comparison_table.py — таблица TabPFN-3 против Digital Twin

Все три запускаются руками и редко, поэтому ломаются незаметно: ошибка вылезает
в тот момент, когда скрипт понадобился. Каждый прогоняется здесь целиком на
синтетическом входе, с путями, уведёнными в tmp — ни один не пишет в репозиторий.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import adhoc_ggrc_2025_reformat as ggrc_adapter  # noqa: E402
import build_model_comparison_table as comparison  # noqa: E402
import skyfert_kpi_test  # noqa: E402


# ------------------------------------------------------ skyfert_kpi_test.py

def _skyfert_raw() -> pd.DataFrame:
    # Форма реальной выгрузки Skyfert (см. docs/SKYFERT.md): нет дат, нет MII,
    # нет эмбриолога/врача, ЭМК повторяется у физически разных пациенток.
    return pd.DataFrame({
        "ЭМК": ["5822", "5822", "9001", "9002", "9003"],
        "Возраст": [34, 42, 28, 31, 27],
        "№ попытки": [1, 1, 1, None, 1],
        "ЭКО": ["ИКСИ", "ЭКО", "ИКСИ", "ИКСИ", None],
        "Количество фолликулов": [12, 10, 8, None, 7],
        "Число ОКК": [10, 8, 6, 5, 5],
        "Число инсеминированных": [8, 6, 4, 4, 4],
        "2 pN": [6, 4, 3, 3, 3],
        "Число дробящихся на 3 день": [5, 3, 2, 2, 2],
        "Число эмбрионов 5 дня": [4, 2, 2, 2, 1],
        "Число Bl": [3, 2, 1, 1, 1],
        "Число Bl хор.кач-ва": [2, 1, 1, 1, 1],
        "День переноса": [5, 5, 5, None, 5],
        "Исход переноса": ["Беременность клиническая", "Беременность не наступила",
                            "Беременность не наступила", "Беременность не наступила", None],
        "Перенесено эмбрионов": [1, 1, 1, 1, 0],
        "Заморожено эмбрионов": [0, 0, 0, 0, 2],
    })


@pytest.fixture
def skyfert_run(tmp_path, monkeypatch):
    """Настоящий конфиг клиники, но все выходные пути — в tmp."""
    monkeypatch.setattr(skyfert_kpi_test, "ROOT", tmp_path)
    source = tmp_path / "SkyFert.xlsx"
    _skyfert_raw().to_excel(source, index=False)
    return tmp_path, source


def test_skyfert_script_runs_and_writes_its_declared_outputs(skyfert_run, capsys):
    tmp_path, source = skyfert_run
    assert skyfert_kpi_test.main([str(source)]) == 0

    processed = tmp_path / "data" / "processed" / "skyfert"
    private = processed / "local_private"
    for name in ("kpi_overall.csv", "ozema_cycles.csv",
                 "ozema_genetic_counseling_candidates.csv"):
        assert (processed / name).is_file(), name
    for name in ("cycle_uid_crosswalk.csv", "skyfert_lab_population.csv",
                 "skyfert_kpi_cycle.csv"):
        assert (private / name).is_file(), name

    out = capsys.readouterr().out
    assert "[eligible]" in out and "[ozema]" in out


def test_skyfert_script_keeps_colliding_emk_as_separate_cycles(skyfert_run):
    tmp_path, source = skyfert_run
    skyfert_kpi_test.main([str(source)])
    crosswalk = pd.read_csv(tmp_path / "data" / "processed" / "skyfert" /
                            "local_private" / "cycle_uid_crosswalk.csv")
    # ЭМК 5822 повторяется у двух РАЗНЫХ пациенток: cycle_uid обязан их развести,
    # иначе строки схлопнутся в один цикл (docs/SKYFERT.md).
    assert crosswalk["cycle_uid"].nunique() == 5
    assert (crosswalk["patient_id"].astype(str) == "5822").sum() == 2


def test_skyfert_ozema_output_carries_no_patient_identifiers(skyfert_run):
    tmp_path, source = skyfert_run
    skyfert_kpi_test.main([str(source)])
    ozema = pd.read_csv(tmp_path / "data" / "processed" / "skyfert" / "ozema_cycles.csv")
    assert "patient_id" not in ozema.columns and "patient_name" not in ozema.columns
    assert "cycle_uid" in ozema.columns


def test_skyfert_script_rejects_a_cohort_without_outcomes(tmp_path, monkeypatch):
    monkeypatch.setattr(skyfert_kpi_test, "ROOT", tmp_path)
    raw = _skyfert_raw()
    raw["Перенесено эмбрионов"] = 0
    source = tmp_path / "SkyFert.xlsx"
    raw.to_excel(source, index=False)
    with pytest.raises(ValueError, match="переносом"):
        skyfert_kpi_test.main([str(source)])


# --------------------------------------------- adhoc_ggrc_2025_reformat.py

def _ggrc_transfer_source() -> pd.DataFrame:
    rows = {col: [None, None, None] for col in ggrc_adapter.TRANSFER_TARGET_COLUMNS}
    rows.pop("OPU Code")
    rows.pop("Fertilization Method")
    frame = pd.DataFrame(rows)
    frame["OPU Number"] = [207395, 207396, 200001]
    frame["OPU Number.1"] = [207395, 207396, 200001]
    frame["fertilization"] = ["ICSI", "ICSI", "IVF"]
    # 2024-я строка должна отсеяться: скрипт оставляет только пункции 2025 года.
    frame["OPU Date"] = pd.to_datetime(["2025-03-01", "2025-04-02", "2024-11-11"])
    frame["Patients name"] = ["A", "B", "C"]
    frame["Age"] = [33, 39, 41]
    return frame


def _ggrc_kpi_wide_source() -> pd.DataFrame:
    rows = {col: [None, None, None] for col in ggrc_adapter.KPI_WIDE_TARGET_COLUMNS}
    rows.pop("OPU Code")
    rows.pop("Fertilization Method")
    rows.pop("OPU Date")                 # ключевая особенность: даты в этом файле НЕТ
    frame = pd.DataFrame(rows)
    frame["OPU Number"] = [207395, 209999, None]   # последняя — пустая строка-заполнитель
    frame["fertilization"] = ["ICSI", "IVF", None]
    frame["Oocytes N"] = [10, 8, None]
    return frame


@pytest.fixture
def ggrc_sources(tmp_path, monkeypatch):
    transfer = tmp_path / "предсказания 2025.xlsx"
    kpi_wide = tmp_path / "KPI 2025.xlsx"
    _ggrc_transfer_source().to_excel(transfer, sheet_name="Sheet1", index=False)
    _ggrc_kpi_wide_source().to_excel(kpi_wide, sheet_name="Sheet1", index=False)
    monkeypatch.setattr(ggrc_adapter, "IN_TRANSFER", transfer)
    monkeypatch.setattr(ggrc_adapter, "IN_KPI_WIDE", kpi_wide)
    monkeypatch.setattr(ggrc_adapter, "OUT_DIR", tmp_path / "_2025_adapted")
    return tmp_path


def test_ggrc_transfer_keeps_only_2025_and_normalises_headers(ggrc_sources, capsys):
    out = ggrc_adapter.build_transfer_2025()
    assert len(out) == 2, "строка 2024 года должна отсеяться"
    assert list(out.columns) == ggrc_adapter.TRANSFER_TARGET_COLUMNS
    assert "OPU Number" not in out.columns and "OPU Number.1" not in out.columns
    assert out["Fertilization Method"].tolist() == ["ICSI", "ICSI"]
    assert "561" not in capsys.readouterr().out  # печатает реальные, а не зашитые числа


def test_ggrc_transfer_guards_the_duplicate_column_assumption(ggrc_sources):
    source = _ggrc_transfer_source()
    source.loc[0, "OPU Number.1"] = 999999
    source.to_excel(ggrc_adapter.IN_TRANSFER, sheet_name="Sheet1", index=False)
    # Молча дропнуть колонку, переставшую быть дубликатом, значит потерять данные.
    with pytest.raises(AssertionError, match="no longer an exact duplicate"):
        ggrc_adapter.build_transfer_2025()


def test_ggrc_kpi_wide_backfills_opu_date_and_drops_padding(ggrc_sources, capsys):
    out = ggrc_adapter.build_kpi_wide_2025()
    assert len(out) == 2, "пустая строка-заполнитель должна отсеяться"
    assert list(out.columns) == ggrc_adapter.KPI_WIDE_TARGET_COLUMNS
    # Без даты ключ "_key" = "{OPU Code}#NA" не совпал бы ни с одной строкой файла
    # переносов и обнулил бы истинные счётчики почти по всей когорте.
    dated = out.set_index("OPU Code")["OPU Date"]
    assert pd.notna(dated.loc[207395])
    assert pd.isna(dated.loc[209999]), "цикла нет в файле переносов — настоящий пропуск"
    assert "backfilled OPU Date for 1/2" in capsys.readouterr().out


def test_ggrc_adapter_writes_both_files_with_expected_sheets(ggrc_sources):
    ggrc_adapter.main()
    out_dir = ggrc_adapter.OUT_DIR
    transfer = pd.read_excel(out_dir / "transfer_2025.xlsx", sheet_name="Sheet1")
    kpi_wide = pd.read_excel(out_dir / "kpi_wide_2025.xlsx", sheet_name="IVF Master DB")
    assert len(transfer) == 2 and len(kpi_wide) == 2
    assert list(transfer.columns) == ggrc_adapter.TRANSFER_TARGET_COLUMNS


# ------------------------------------- build_model_comparison_table.py

def _dt_predictions() -> pd.DataFrame:
    return pd.DataFrame({
        "cycle_uid": ["CYC-1", "CYC-2"],
        "patient_id": ["P1#20260101", "P2#20260102"],   # локальный ключ, в вывод не идёт
        "Patient Full Name": ["Пациент Один", "Пациент Два"],
        "age": [33, 39], "amh": [2.1, 0.9], "afc": [12, 6], "bmi": [23.0, 27.5],
        "attempt_number": [1, 2], "sperm_source": ["ejaculate", "ejaculate"],
        "IVF/ICSI": ["ICSI", "ICSI"], "Embryologist": ["Emb A", "Emb B"],
        "Doctor": ["Dr A", "Dr B"],
        "OCC": [10, 6], "MII": [8, 5], "2pN": [6, 3], "Bl": [3, 1], "Good Bl": [2, 0],
        "med_okk": [9.5, 6.2], "p025_okk": [7, 4], "p975_okk": [12, 8],
        "med_mii": [8, 5], "med_pn2": [6, 3], "med_blasts": [3.2, 1.4], "med_good": [2.1, 0.6],
        "known_okk": [10, None],
    })


def _regression_qc() -> pd.DataFrame:
    frame = pd.DataFrame({"cycle_uid": ["CYC-1", "CYC-2"]})
    for stage in comparison.TABPFN3_QUANTILE_STAGES:
        for col in comparison._quantile_cols(f"TPFN3_{stage}"):
            frame[col] = [1.0, 2.0]
        frame[f"{stage}_actual"] = [3, 1]
        frame[f"{stage}_in_interval80"] = [True, False]
    return frame


def test_quantile_columns_cover_the_full_p10_p90_set():
    assert comparison._quantile_cols("TPFN3_mii") == [
        "TPFN3_mii_p10", "TPFN3_mii_p25", "TPFN3_mii_p50",
        "TPFN3_mii_p75", "TPFN3_mii_p90", "TPFN3_mii_interval80_width",
    ]


def test_comparison_table_is_one_row_per_cycle_and_carries_no_identifiers():
    out = comparison.build(_dt_predictions(), _regression_qc())
    assert out["cycle_uid"].tolist() == ["CYC-1", "CYC-2"]
    assert out["cycle_uid"].is_unique
    for forbidden in ("patient_id", "Patient Full Name", "ID", "_key"):
        assert forbidden not in out.columns
    # Обе модели должны стоять рядом на одной стадии — ради этого таблица и нужна.
    assert out["dt_occ_pred"].tolist() == [9.5, 6.2]
    assert "tabpfn3_mii_p50" in out.columns


def test_comparison_table_survives_a_cycle_present_in_only_one_source():
    dt = _dt_predictions()
    rq = _regression_qc().iloc[[0]]        # для CYC-2 регрессии нет
    out = comparison.build(dt, rq)
    assert len(out) == 2
    assert pd.isna(out.set_index("cycle_uid").loc["CYC-2", "tabpfn3_mii_p50"])


def test_comparison_main_writes_the_csv_next_to_the_run(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _dt_predictions().to_csv(run_dir / "dt_predictions.csv", index=False)
    _regression_qc().to_csv(run_dir / "regression_qc.csv", index=False)
    monkeypatch.setattr(sys, "argv", ["build_model_comparison_table.py", "--run-dir", str(run_dir)])

    comparison.main()

    target = run_dir / "model_comparison_tabpfn3_vs_digitaltwin.csv"
    assert target.is_file()
    assert len(pd.read_csv(target)) == 2
    out = capsys.readouterr().out
    assert "Wrote 2 rows" in out
    assert "Data-quality flags" in out
