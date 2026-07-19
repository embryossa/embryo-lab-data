from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

fitz = pytest.importorskip("fitz")

from embryo.clinics import fertimed_pgt as fpgt


_CYRILLIC_FONT = r"C:\Windows\Fonts\bahnschrift.ttf"  # arial/tahoma/verdana/segoeui each
# collapse one of space/hyphen/parenthesis onto a lookalike codepoint (NBSP, soft-hyphen,
# ornate parens) via shared glyphs, so PyMuPDF's extraction silently swaps them in —
# bahnschrift is one of the few common fonts that round-trips ASCII exactly, which the
# header label-matching and karyotype-continuation parsing below both depend on.


def _write_pdf(path: Path, lines: list[str]) -> Path:
    """Synthesize a minimal PDF whose text extracts one line per line,
    matching f-genetics.com's PyMuPDF layout (no real patient data).
    Base14 fonts (the insert_text default) have no Cyrillic glyphs and
    silently drop the text on extraction, so an embedded TTF is required."""
    doc = fitz.open()
    page = doc.new_page()
    y = 50.0
    for line in lines:
        if y > 780:
            page = doc.new_page()
            y = 50.0
        page.insert_text((50, y), line, fontsize=10, fontname="F0", fontfile=_CYRILLIC_FONT)
        y += 12.0
    doc.save(path)
    doc.close()
    return path


def _header(patient: str, id_lab: str, issued_date: str, embryologist: str = "Иванов И.И.",
            patient_suffix: str | None = None) -> list[str]:
    lines = [
        "f-genetics.com",
        "ID пациента в организации", "1-Б",
        "ID пациента в лаборатории", id_lab,
        "Результат",
        "Пациент", patient,
    ]
    if patient_suffix:
        lines.append(patient_suffix)
    lines += [
        "Партнер", "Партнёр Тестов",
        "Дата биопсии", "01.01.2026",
        "Дата получения", "02.01.2026",
        "Дата выдачи", issued_date,
        "Эмбриолог", embryologist,
        "Организация, город", "ООО «Тест»",
        "Показания", "скрининг",
        "Результат ПГТ-А (преимплантационного генетического тестирования на хромосомные аномалии)",
        patient,
        "№", "образца", "Кариотип", "Согласно ISCN 2020", "Интерпретация", "Рекомендации",
        "По переносу эмбриона",
    ]
    return lines


def _footer() -> list[str]:
    return ["Тестирование выполнено при помощи технологии полупроводникового секвенирования "
            "(NGS) на платформе F-Genetics.                                            1 из 1"]


def test_parse_report_classifies_euploid_aneuploid_and_mosaic(tmp_path):
    lines = (
        _header("Тестова Анна Ивановна", "111", "27.01.2026")
        + ["T1", "seq(X,Y)x1,(1-22)x2", "Норма", "Рекомендован"]
        + ["T2", "seq(22)x3", "Трисомия по хромосоме 22", "Не рекомендован"]
        + ["T3", "seq(X,1-14,15q11.2q22.31,16-22)x2,", "(15q22.31q26.3)x1[0.45]",
           "Мозаичная форма сегментарной", "делеции хромосомы 15",
           "Рекомендован с", "информированного согласия", "пациента*"]
        + _footer()
    )
    pdf = _write_pdf(tmp_path / "r.pdf", lines)
    report = fpgt.parse_report(pdf)

    assert report["patient_name"] == "Тестова Анна Ивановна"
    assert report["id_lab"] == "111"
    assert report["issued_dates"] == ["27.01.2026"]
    assert len(report["embryos"]) == 3

    euploid, aneuploid, mosaic = report["embryos"]
    assert euploid["category"] == "euploid"
    assert euploid["recommendation_category"] == "recommended"
    assert aneuploid["category"] == "aneuploid"
    assert aneuploid["recommendation_category"] == "not_recommended"
    assert mosaic["category"] == "mosaic"
    assert mosaic["karyotype"] == "seq(X,1-14,15q11.2q22.31,16-22)x2, (15q22.31q26.3)x1[0.45]"
    assert mosaic["recommendation_category"] == "recommended_with_consent"
    assert mosaic["recommendation"] == "Рекомендован с информированного согласия пациента*"


def test_parse_report_handles_case_level_note_after_last_row(tmp_path):
    lines = (
        _header("Носитель Транслокации", "222", "01.02.2026")
        + ["T1", "seq(X,Y)x1,(1-22)x2", "Норма", "Рекомендован"]
        + ["По результатам ПГТ-А можно предположить, что один из партнеров является носителем.",
           "Рекомендовано кариотипирование обоих партнеров."]
        + _footer()
    )
    pdf = _write_pdf(tmp_path / "r.pdf", lines)
    report = fpgt.parse_report(pdf)

    assert len(report["embryos"]) == 1
    assert report["embryos"][0]["recommendation_category"] == "recommended"
    assert "кариотипирование" in report["case_note"]


def test_parse_report_handles_not_tested_sample(tmp_path):
    lines = (
        _header("Частично Выдана", "333", "01.03.2026")
        + ["T1", "seq(X,Y)x1,(1-22)x2", "Норма", "Рекомендован"]
        + ["T2", "На хранении"]
        + _footer()
    )
    pdf = _write_pdf(tmp_path / "r.pdf", lines)
    report = fpgt.parse_report(pdf)

    assert len(report["embryos"]) == 2
    assert report["embryos"][1]["category"] == "not_tested"
    assert report["embryos"][1]["recommendation_category"] is None


def test_parse_report_strips_patient_name_suffix(tmp_path):
    lines = (
        _header("Суффиксова Мария", "444", "01.04.2026", patient_suffix="(ДО № RBO-00001)")
        + ["T1", "seq(X,Y)x1,(1-22)x2", "Норма", "Рекомендован"]
        + _footer()
    )
    pdf = _write_pdf(tmp_path / "r.pdf", lines)
    report = fpgt.parse_report(pdf)

    assert report["patient_name"] == "Суффиксова Мария"
    assert "RBO-00001" in report["patient_name_raw"]


def test_summarize_embryos_computes_euploid_rate_and_recommendation_counts():
    embryos = [
        {"category": "euploid", "recommendation_category": "recommended"},
        {"category": "euploid", "recommendation_category": "recommended"},
        {"category": "aneuploid", "recommendation_category": "not_recommended"},
        {"category": "not_tested", "recommendation_category": None},
    ]
    summary = fpgt.summarize_embryos(embryos)

    assert summary["n_biopsied"] == 4
    assert summary["n_tested"] == 3
    assert summary["n_not_tested"] == 1
    assert summary["euploid_rate"] == pytest.approx(round(2 / 3, 4))
    assert summary["n_recommended"] == 2
    assert summary["n_not_recommended"] == 1
    # Not available from this report format, unlike clinic A's eMap Cycle Report.
    assert summary["mean_overall_noise_euploid"] is None
    assert summary["analysis_settings_variable"] is None


def _cohort(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_analyze_reports_matches_single_candidate_by_patient_name(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    lines = (
        _header("Единая Кандидатка", "555", "10.02.2026")
        + ["T1", "seq(X,Y)x1,(1-22)x2", "Норма", "Рекомендован"]
        + _footer()
    )
    _write_pdf(reports_dir / "any_name.pdf", lines)

    cohort = _cohort([
        {"cycle_uid": "CYC-1", "Patient Full Name": "Единая Кандидатка",
         "Date OPU": "2026-02-01", "Date ET": "2026-02-06"},
    ])
    result = fpgt.analyze_reports(cohort, reports_dir)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["cycle_uid"] == "CYC-1"
    assert row["n_biopsied"] == 1
    assert row["n_euploid"] == 1
    assert row["match_reason"] == "unique_name_match"
    assert not result.attrs["events"].empty
    assert result.attrs["unmatched"] == []


def test_analyze_reports_disambiguates_repeat_attempts_by_report_date(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    lines = (
        _header("Повторная Пациентка", "666", "05.05.2026")
        + ["T1", "seq(22)x3", "Трисомия по хромосоме 22", "Не рекомендован"]
        + _footer()
    )
    _write_pdf(reports_dir / "any_name.pdf", lines)

    cohort = _cohort([
        {"cycle_uid": "CYC-OLD", "Patient Full Name": "Повторная Пациентка",
         "Date OPU": "2026-01-01", "Date ET": "2026-01-06"},
        {"cycle_uid": "CYC-NEW", "Patient Full Name": "Повторная Пациентка",
         "Date OPU": "2026-05-01", "Date ET": "2026-05-06"},
    ])
    result = fpgt.analyze_reports(cohort, reports_dir)

    matched = result[result["n_biopsied"] > 0]
    assert len(matched) == 1
    assert matched.iloc[0]["cycle_uid"] == "CYC-NEW"
    assert matched.iloc[0]["match_reason"] == "matched_by_report_date"
    other = result[result["cycle_uid"] == "CYC-OLD"].iloc[0]
    assert other["pgt_note"] == "нет скачанного отчёта ПГТ (f-genetics.com)"


def test_analyze_reports_reports_no_name_match(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    lines = (
        _header("Никого Нет В Когорте", "777", "01.01.2026")
        + ["T1", "seq(X,Y)x1,(1-22)x2", "Норма", "Рекомендован"]
        + _footer()
    )
    _write_pdf(reports_dir / "any_name.pdf", lines)

    cohort = _cohort([
        {"cycle_uid": "CYC-1", "Patient Full Name": "Кто-То Другой",
         "Date OPU": "2026-01-01", "Date ET": "2026-01-06"},
    ])
    result = fpgt.analyze_reports(cohort, reports_dir)

    assert result.attrs["unmatched"] == [{"file": "any_name.pdf", "reason": "no_name_match"}]
    assert result.iloc[0]["pgt_note"] == "нет скачанного отчёта ПГТ (f-genetics.com)"


def test_analyze_reports_requires_cycle_uid_and_name_columns(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    with pytest.raises(KeyError):
        fpgt.analyze_reports(_cohort([{"Patient Full Name": "X"}]), reports_dir)
    with pytest.raises(KeyError):
        fpgt.analyze_reports(_cohort([{"cycle_uid": "CYC-1"}]), reports_dir)
