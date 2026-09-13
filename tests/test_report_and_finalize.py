"""HTML-отчёт (src/embryo/report.py) и его финализация (scripts/finalize_report.py).

report.py — самый крупный модуль пайплайна и последний шаг прогона: падение здесь
обесценивает весь уже посчитанный анализ. Ломается он не на «нормальных» данных,
а на вырожденных — пустой review-list, когорта из одной строки, сплошные пропуски
в вероятностях, отсутствующая колонка. Именно эти входы тут и проверяются: отчёт
обязан собраться и остаться читаемым, а не упасть или молча потерять вкладку.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import finalize_report  # noqa: E402
from embryo import analytics, report  # noqa: E402


TABS = ("Обзор", "KPI эмбриологов", "KPI врачей", "Модели и калибровка",
        "Review-list", "MIRI", "OZEMA")


def _cohort(n: int = 8) -> pd.DataFrame:
    rng = np.arange(n)
    return pd.DataFrame({
        # Один и тот же кадр играет обе роли, что и в пайплайне: dt_predictions
        # (ключ patient_id) и кадр OPU (ключ _key/ID) — analytics.build_merged
        # джойнит их именно так.
        "patient_id": [f"P{i:03d}" for i in rng],
        "ID": [f"P{i:03d}" for i in rng],
        "_key": [f"P{i:03d}" for i in rng],
        "cycle_uid": [f"CYC-{i:016X}" for i in rng],
        "Patient Full Name": [f"Пациент {i}" for i in rng],
        "Preg": [i % 2 for i in rng],
        "PRAI": np.linspace(0.2, 0.9, n),
        "DIGITAL TWIN": np.linspace(0.25, 0.85, n),
        "ci_width_pp": np.linspace(5, 30, n),
        "ci_wide": np.linspace(5, 30, n) > 20,
        "Age": 30 + rng % 10,
        "age": 30 + rng % 10,
        "AMH": np.linspace(0.8, 4.0, n),
        "BMI": np.linspace(19, 31, n),
        "N folicules OPU": 8 + rng,
        "OCC": 6 + rng,
        "MII": 5 + rng,
        "2pN": 4 + rng,
        "Cleavage": 4 + rng,
        "Bl": 2 + rng % 4,
        "Good Bl": 1 + rng % 3,
        "Cryo": rng % 3,
        "ET": 1,
        "Date OPU": pd.date_range("2026-01-01", periods=n, freq="D"),
        "Date ET": pd.date_range("2026-01-06", periods=n, freq="D"),
        "IVF/ICSI": "ICSI",
        "Embryologist": [("Emb A", "Emb B")[i % 2] for i in rng],
        "Doctor": [("Dr A", "Dr B")[i % 2] for i in rng],
        "FR": np.linspace(0.6, 0.9, n),
        "ClR": np.linspace(0.7, 0.95, n),
        "BlR": np.linspace(0.3, 0.7, n),
        "TGBDR": np.linspace(0.2, 0.5, n),
        "OCCR": np.linspace(0.6, 0.9, n),
        "DegR": np.linspace(0.0, 0.1, n),
        "3pNR": np.linspace(0.0, 0.1, n),
    })


def _analytics_result(tmp_path: Path, df: pd.DataFrame) -> dict:
    """Настоящий результат analytics.run_analytics — не самодельный словарь.

    Иначе тест зафиксировал бы форму, которой пайплайн не производит, и разошёлся
    бы с ним молча.
    """
    return analytics.run_analytics(df.copy(), df.copy(), tmp_path / "analytics.xlsx")


def _build(tmp_path: Path, df: pd.DataFrame, **kwargs) -> str:
    out = tmp_path / "report.html"
    result = report.build_report(_analytics_result(tmp_path, df), df.copy(),
                                 out_html=str(out), **kwargs)
    assert isinstance(result, dict) and "warnings" in result
    return out.read_text(encoding="utf-8")


def test_report_builds_all_tabs_on_a_normal_cohort(tmp_path):
    html = _build(tmp_path, _cohort())
    for tab in TABS:
        assert tab in html, f"нет вкладки: {tab}"


def test_report_survives_single_row_cohort(tmp_path):
    # n=1: любой доверительный интервал/корреляция вырождены, деление на (n-1) падает.
    html = _build(tmp_path, _cohort(1))
    assert "Обзор" in html


def test_report_survives_cohort_without_any_known_outcome(tmp_path):
    df = _cohort()
    df["Preg"] = np.nan
    html = _build(tmp_path, df)
    assert "Модели и калибровка" in html


def test_report_survives_all_missing_model_probabilities(tmp_path):
    df = _cohort()
    df["PRAI"] = np.nan
    df["DIGITAL TWIN"] = np.nan
    df["ci_width_pp"] = np.nan
    df["ci_wide"] = False
    html = _build(tmp_path, df)
    assert "Review-list" in html


def test_report_survives_single_outcome_class(tmp_path):
    # Все беременности: ROC-AUC не определён — отчёт обязан это пережить.
    df = _cohort()
    df["Preg"] = 1
    assert "Обзор" in _build(tmp_path, df)


def test_report_survives_missing_optional_columns(tmp_path):
    df = _cohort().drop(columns=["AMH", "BMI", "Good Bl", "Cryo", "Embryologist", "Doctor"])
    assert "Обзор" in _build(tmp_path, df)


def test_report_renders_forensic_tabs_from_real_frames(tmp_path):
    df = _cohort()
    miri = pd.DataFrame([{
        "cycle_uid": df["cycle_uid"].iloc[0], "window_start": "2026-01-01",
        "window_end": "2026-01-07", "n_points": 100, "event_count": 2,
        "warning_count": 1, "critical_count": 0, "max_severity": "warning",
        "warning": True, "gap_count": 1, "max_event_duration_minutes": 42.0, "error": None,
    }])
    miri_events = pd.DataFrame([{
        "cycle_uid": df["cycle_uid"].iloc[0], "event_type": "excursion", "sensor": "Temp1",
        "duration_minutes": 42.0, "max_deviation": 0.8, "severity": "warning",
        "recovery_time_minutes": 12.0,
    }])
    lots_summary = pd.DataFrame([{
        "cycle_uid": df["cycle_uid"].iloc[0], "window_start": "2026-01-01",
        "window_end": "2026-01-07", "n_items_tracked": 5, "n_items_changed": 1,
        "changed_items": "среда culture", "lot_change_flag": True, "error": None, "note": "",
    }])
    ozema = pd.DataFrame([{
        "patient_id": "P000", "cycle_uid": df["cycle_uid"].iloc[0], "ozema_flag": "OMA",
        "flagged": True, "flagged_cycles_n": 1, "age": 35,
        "monitoring_note": "кандидат на обсуждение", "embryologist": "Emb A", "doctor": "Dr A",
    }])
    html = _build(tmp_path, df, incubator_findings=miri, incubator_events=miri_events,
                  lot_summary=lots_summary, ozema_cycles=ozema,
                  miri_baseline={"flagged_n": 1, "baseline_n": 7,
                                 "flagged_warning_rate": 1.0, "baseline_warning_rate": 0.0})
    assert "MIRI" in html and "OZEMA" in html


def test_ozema_table_without_flag_column_renders_blank_instead_of_crashing():
    # DataFrame.get на отсутствующей колонке возвращает не Series, а само
    # значение по умолчанию — обычная ловушка, роняющая последний шаг прогона
    # уже после того, как весь анализ посчитан.
    frame = pd.DataFrame([{"patient_id": "P000", "age": 35}])
    html = report._ozema_table(frame)
    assert "P000" in html


def test_analysis_summary_is_cohort_level_and_json_safe(tmp_path):
    df = _cohort()
    summary = report.build_analysis_summary(_analytics_result(tmp_path, df)["df_u"])
    assert set(summary) == {"cluster_summary", "kat_vs_prai", "intra_inter_cluster_similarity"}
    # Пакет уходит агенту: ни одного идентификатора пациентки на этом уровне.
    blob = repr(summary)
    for leaked in ("P000", "Пациент 0", "CYC-"):
        assert leaked not in blob


def test_analysis_summary_handles_empty_frame():
    assert isinstance(report.build_analysis_summary(pd.DataFrame()), dict)


# ------------------------------------------------------- finalize_report.py

_LEAD_MD = "## Вывод\n\nКогорта **малая**: ДИ широкие.\n\n- пункт один\n- пункт два\n"


def _finalize(tmp_path: Path, html: str, text: str = _LEAD_MD, monkeypatch=None):
    report_path = tmp_path / "report.html"
    text_path = tmp_path / "lead_analysis.md"
    report_path.write_text(html, encoding="utf-8")
    text_path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["finalize_report.py", "--report", str(report_path),
                                      "--text-file", str(text_path)])
    finalize_report.main()
    return report_path.read_text(encoding="utf-8")


def test_finalize_renders_markdown_into_the_contract_slot(tmp_path, monkeypatch):
    out = _finalize(tmp_path, "<html><body><!-- LEAD_START -->старое<!-- LEAD_END --></body></html>",
                    monkeypatch=monkeypatch)
    assert "<h2>Вывод</h2>" in out
    assert "<strong>малая</strong>" in out
    assert "<li>пункт один</li>" in out
    assert "старое" not in out, "прежний текст должен заменяться, а не дублироваться"
    assert out.count("<!-- LEAD_START -->") == 1 and out.count("<!-- LEAD_END -->") == 1
    assert "id='lead-analysis'" in out


def test_finalize_is_idempotent_across_repeated_runs(tmp_path, monkeypatch):
    first = _finalize(tmp_path, "<html><!-- LEAD_START --><!-- LEAD_END --></html>", monkeypatch=monkeypatch)
    second = _finalize(tmp_path, first, monkeypatch=monkeypatch)
    assert second.count("id='lead-analysis'") == 1


def test_finalize_refuses_a_report_without_the_insertion_contract(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="insertion contract"):
        _finalize(tmp_path, "<html><body>нет якорей</body></html>", monkeypatch=monkeypatch)


def test_finalize_keeps_html_entities_from_the_agent_text(tmp_path, monkeypatch):
    # "&gt;" в тексте агента — уже готовая сущность; повторное экранирование
    # показало бы на странице "&amp;gt;".
    out = _finalize(tmp_path, "<html><!-- LEAD_START --><!-- LEAD_END --></html>",
                    text="ДИ &gt; 20 пп", monkeypatch=monkeypatch)
    assert "&gt; 20 пп" in out and "&amp;gt;" not in out


def test_real_report_accepts_the_lead_analyst_insertion(tmp_path, monkeypatch):
    # Сквозная связка: якоря, которые ищет finalize_report, обязан выдавать report.py.
    html = _build(tmp_path, _cohort())
    assert "<!-- LEAD_START -->" in html and "<!-- LEAD_END -->" in html
    out = _finalize(tmp_path, html, monkeypatch=monkeypatch)
    assert "<h2>Вывод</h2>" in out
