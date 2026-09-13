"""Разбор дат: ISO из файлов пайплайна против "дд.мм.гггг" из выгрузок МИС.

Пайплайн читает даты ДВУХ видов. Сырая выгрузка 1С/МИС даёт "05.01.2026"
(5 января) — без ``dayfirst=True`` это прочтётся как 1 мая. Но всё, что пайплайн
сам записал и читает обратно — dt_predictions.csv, regression_qc.csv,
OPU_table_enriched.xlsx — приходит в ISO "2026-01-05", и ``dayfirst=True``
переставляет день с месяцем УЖЕ В НЁМ: pandas 3.0.3 читает "2026-01-05" как
1 мая, а ``format="mixed"`` от этого не спасает.

Дефект тихий — исключения нет, страдает каждая дата с днём ≤ 12 (~40% строк),
и цена разная по местам: в KPI-срезе «по времени» цикл уезжает в чужой месяц,
в карточке Teable и в журнале отчёта пациентке показывается чужая дата.

Здесь проверяются оба формата во всех точках разбора сразу — и в пакете embryo,
и в самостоятельном слое teable, где та же логика продублирована намеренно.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "teable"))

import cards_build  # noqa: E402
import teable_sync  # noqa: E402
from embryo import clinical, kpi, report  # noqa: E402
from embryo.normalize import parse_dates  # noqa: E402


# Дни ≤ 12 неоднозначны и ловят подмену; 13-е — контроль, оно не переставляется.
AMBIGUOUS = [("2026-01-05", 2026, 1, 5), ("05.01.2026", 2026, 1, 5),
             ("2026-03-11", 2026, 3, 11), ("11.03.2026", 2026, 3, 11)]
UNAMBIGUOUS = [("2026-01-13", 2026, 1, 13), ("13.01.2026", 2026, 1, 13)]


@pytest.mark.parametrize("value, year, month, day", AMBIGUOUS + UNAMBIGUOUS)
def test_parse_dates_reads_both_formats_correctly(value, year, month, day):
    parsed = parse_dates([value]).iloc[0]
    assert (parsed.year, parsed.month, parsed.day) == (year, month, day)


def test_parse_dates_handles_a_frame_mixing_both_formats():
    parsed = parse_dates(pd.Series(["2026-01-05", "05.01.2026", "2026-01-13", None, "мусор"]))
    assert parsed.iloc[0] == parsed.iloc[1] == pd.Timestamp("2026-01-05")
    assert parsed.iloc[2] == pd.Timestamp("2026-01-13")
    assert parsed.iloc[3:].isna().all()


def test_parse_dates_passes_through_real_timestamps():
    source = pd.Series(pd.to_datetime(["2026-01-05", "2026-07-11"]))
    assert parse_dates(source).tolist() == source.tolist()


def test_parse_dates_returns_all_nat_for_unparseable_input():
    assert parse_dates(["не дата", ""]).isna().all()


@pytest.mark.parametrize("value, year, month, day", AMBIGUOUS + UNAMBIGUOUS)
def test_kpi_time_slice_puts_the_cycle_in_the_right_month(value, year, month, day):
    # Цикл, уехавший в чужой месяц, портит срез «KPI по времени» молча.
    cycles = pd.DataFrame({
        "date_opu": [value], "occ": [8], "follicles": [10], "mii": [7],
        "inseminated": [8], "two_pn": [6], "cleavage_d3": [6],
        "bl_total": [3], "bl_good": [2], "three_pn": [0], "degrade": [0],
    })
    computed = kpi.compute_kpi(cycles)
    slices = kpi.aggregate_kpi_slices(computed, time_col="date_opu")
    periods = set(slices["time"]["_kpi_period"].astype(str))
    assert periods == {f"{year}-{month:02d}"}


@pytest.mark.parametrize("value, year, month, day", AMBIGUOUS + UNAMBIGUOUS)
def test_report_russian_date_label(value, year, month, day):
    assert report._ru_date(value).startswith(str(day))


@pytest.mark.parametrize("value, year, month, day", AMBIGUOUS + UNAMBIGUOUS)
def test_teable_sync_iso_date(value, year, month, day):
    assert teable_sync.iso_date(value) == f"{year}-{month:02d}-{day:02d}"


@pytest.mark.parametrize("value, year, month, day", AMBIGUOUS + UNAMBIGUOUS)
def test_cards_build_date(value, year, month, day):
    assert cards_build._date(value) == f"{year}-{month:02d}-{day:02d}"


def test_teable_date_helpers_agree_with_the_package_parser():
    # Дублирование логики через границу пакета намеренное — но расходиться
    # эти три реализации не должны.
    for value, year, month, day in AMBIGUOUS + UNAMBIGUOUS:
        expected = f"{year}-{month:02d}-{day:02d}"
        assert teable_sync.iso_date(value) == expected
        assert cards_build._date(value) == expected
        assert parse_dates([value]).iloc[0].date().isoformat() == expected


def test_teable_date_helpers_return_none_on_missing_input():
    for helper in (teable_sync.iso_date, cards_build._date):
        assert helper(None) is None
        assert helper(float("nan")) is None
        assert helper("не дата") is None


def test_clinical_dates_use_the_shared_parser():
    frame = pd.DataFrame({"clin_cycle_start_date": ["05.01.2026", "2026-01-05"]})
    parsed = clinical._dates(frame, "clin_cycle_start_date")
    assert parsed.iloc[0] == parsed.iloc[1] == pd.Timestamp("2026-01-05")
    # Отсутствующая колонка — не ошибка: клиники присылают разный набор полей.
    assert clinical._dates(frame, "нет-такой").isna().all()


def test_no_bare_dayfirst_parsing_is_left_in_the_pipeline():
    """dayfirst=True без ISO-разбора перед ним — тот самый дефект.

    Проверка по исходникам намеренно: она ловит НОВОЕ место разбора дат,
    добавленное позже, а не только те, что чинились сейчас.
    """
    offenders = []
    for path in [*(REPO / "src" / "embryo").rglob("*.py"),
                 *(REPO / "scripts").glob("*.py"),
                 *(REPO / "teable").glob("*.py")]:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "dayfirst" not in line or line.lstrip().startswith("#"):
                continue
            # Разрешено внутри разборщиков, где ISO уже попробован строкой выше.
            if path.name in ("normalize.py", "teable_sync.py", "cards_build.py", "pgt.py"):
                continue
            offenders.append(f"{path.relative_to(REPO)}:{lineno}")
    assert not offenders, ("разбор дат мимо normalize.parse_dates: " + ", ".join(offenders))
