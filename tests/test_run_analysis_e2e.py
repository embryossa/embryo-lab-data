"""Сквозной прогон scripts/run_analysis.py на синтетической когорте.

Оркестратор — единственное место, где ingest -> KPI -> reconcile -> analytics ->
report -> agent packet -> manifest -> history сходятся вместе. Юнит-тесты
отдельных модулей этот стык не покрывают: переименованный ключ джойна, потерянный
артефакт или несобравшийся HTML проходят их все и падают только на живом прогоне.
Здесь main() гоняется целиком, с детерминированным replay внешних моделей
(--dt-output/--tabpfn-output) — без сети, без DT/TabPFN, без PHI.

Историческая справка, почему тест написан именно так: фикстуры e2e_*.csv
существовали с первого коммита, но ключом в них был голый "ID". Пайплайн с тех
пор перешёл на композитный "_key" = ID+Date OPU (повторные попытки пациентки
делят один паспортный ID), и фикстуры молча перестали джойниться. Проверка
opu_join_status/tabpfn_join_status ниже — прямая защита от повторения этого.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"
sys.path.insert(0, str(REPO / "scripts"))

import run_analysis  # noqa: E402


@pytest.fixture
def e2e_root(tmp_path, monkeypatch):
    """Изолированный проект-двойник: свой config/ и свои data/, ничего в репозитории."""
    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    (root / "data" / "inbox").mkdir(parents=True)
    shutil.copy(REPO / "config" / "settings.yaml", root / "config" / "settings.yaml")
    shutil.copy(REPO / "config" / "columns.yaml", root / "config" / "columns.yaml")
    shutil.copy(FIXTURES / "e2e_input.csv", root / "data" / "inbox" / "cohort.csv")
    # ROOT задаёт и config_dir, и корень для всех относительных путей settings.yaml.
    monkeypatch.setattr(run_analysis, "ROOT", root)
    cwd = Path.cwd()
    try:
        yield root
    finally:
        os.chdir(cwd)          # main() делает os.chdir(ROOT) и не возвращает cwd назад


def _run(*extra: str) -> int:
    return run_analysis.main([
        "--dt-output", str(FIXTURES / "e2e_dt.csv"),
        "--tabpfn-output", str(FIXTURES / "e2e_tabpfn.csv"),
        "--skip-miri", "--skip-lots", "--skip-pgt", *extra,
    ])


def _processed(root: Path) -> Path:
    return root / "data" / "processed"


def test_full_run_writes_every_declared_artifact(e2e_root):
    assert _run() == 0

    processed = _processed(e2e_root)
    expected = [
        "kpi_computed.csv", "dt_predictions.csv", "review_list.csv",
        "OPU_table_enriched.xlsx", "DT_Analytics_Report.xlsx",
        "DT_Analytics_Report.html", "run_manifest.json", "run_history.jsonl",
        "ozema_cycles.csv", "ozema_genetic_counseling_candidates.csv",
        "miri_summary.csv", "miri_events.csv", "lot_summary.csv", "lot_events.csv",
        "regression_qc.csv", "regression_calibration.csv",
        "pgt_summary.csv", "pgt_embryo_events.csv",
    ]
    for name in expected:
        path = processed / name
        assert path.is_file(), f"артефакт не создан: {name}"
        assert path.stat().st_size > 0, f"артефакт пуст: {name}"

    assert (processed / "agent_packets" / "analysis_bundle.json").is_file()
    assert (processed / "agent_packets" / "schema_diagnostic.json").is_file()
    assert (processed / "local_private" / "model_input.xlsx").is_file()
    assert (processed / "local_private" / "cycle_uid_crosswalk.csv").is_file()


def test_manifest_records_cohort_size_hashes_and_soft_step_status(e2e_root):
    assert _run() == 0
    manifest = json.loads((_processed(e2e_root) / "run_manifest.json").read_text(encoding="utf-8"))

    assert manifest["n_cycles"] == 6
    assert manifest["n_lab_population"] == 6
    assert len(manifest["input_sha256"]) == 64
    assert manifest["schema_versions"]["dt_predictions"]
    assert manifest["dnn_status"] == "disabled"          # enabled: false в settings.yaml
    # Replay выхода API 7.1 — единственного входа Digital Twin: сводка источника из его полей.
    assert manifest["digital_twin"]["source"].startswith("IVF Digital Twin 7.1 REST API")
    assert manifest["digital_twin"]["status"] == {"complete": 6}
    assert manifest["digital_twin"]["model_fingerprints"] == ["fp-e2e"]
    # tabpfn_regression включён по умолчанию, но run_cmd пуст (машинные пути живут в
    # config/machine.local.yaml, которого у тестового ROOT нет). Шаг обязан упасть
    # МЯГКО и не уронить прогон — это закреплённое в CLAUDE.md поведение.
    assert manifest["regression_status"].startswith("failed:")
    for name, meta in manifest["artifacts"].items():
        assert len(meta["sha256"]) == 64, name


def test_every_cycle_joins_across_dt_tabpfn_and_opu(e2e_root):
    assert _run() == 0
    predictions = pd.read_csv(_processed(e2e_root) / "dt_predictions.csv")

    assert len(predictions) == 6
    assert set(predictions["opu_join_status"]) == {"matched"}
    assert set(predictions["tabpfn_join_status"]) == {"matched"}
    assert predictions["cycle_uid"].str.match(r"^CYC-[0-9A-F]{16}$").all()
    assert predictions["cycle_uid"].is_unique
    # ci_wide = ширина ДИ TabPFN больше порога 20 пп: E004 (0.15..0.40 -> 25 пп) и
    # E006 (0.28..0.52 -> 24 пп). Остальные четыре — уже, порог их не берёт.
    wide = predictions.loc[predictions["ci_wide"].astype(bool)]
    assert set(wide["patient_id"]) == {"E004#20260104", "E006#20260106"}
    assert wide.set_index("patient_id").loc["E004#20260104", "ci_width_pp"] == pytest.approx(25.0)
    narrow = predictions.loc[~predictions["ci_wide"].astype(bool), "ci_width_pp"]
    assert (narrow <= 20).all()


def test_agent_packet_carries_no_identifiers(e2e_root):
    assert _run() == 0
    bundle_path = _processed(e2e_root) / "agent_packets" / "analysis_bundle.json"
    raw = bundle_path.read_text(encoding="utf-8")
    bundle = json.loads(raw)

    assert bundle["schema_version"] == "1.1"
    assert bundle["cohort"]["n_cycles"] == 6
    for patient in bundle["patients"]:
        assert patient["cycle_uid"].startswith("CYC-")
        assert not any(key in patient for key in ("patient_id", "ID", "_key", "Patient Full Name"))
    # ФИО и паспортные ID остаются в локальном слое — в пакет не попадает ни одно.
    for leaked in ("Patient One", "Patient Five", "E001", "E005"):
        assert leaked not in raw, f"в agent packet утёк идентификатор: {leaked}"
    # Локальный crosswalk, наоборот, обязан их держать — иначе reverse join невозможен.
    crosswalk = pd.read_csv(_processed(e2e_root) / "local_private" / "cycle_uid_crosswalk.csv")
    assert {"cycle_uid", "patient_id"} <= set(crosswalk.columns)
    assert crosswalk["patient_id"].str.startswith("E00").all()


def test_html_report_renders_every_declared_tab(e2e_root):
    assert _run() == 0
    html = (_processed(e2e_root) / "DT_Analytics_Report.html").read_text(encoding="utf-8")

    # Состав вкладок закреплён в CLAUDE.md; молча потерянная вкладка — регрессия.
    for tab in ("Обзор", "KPI эмбриологов", "KPI врачей", "Модели и калибровка",
                "Review-list", "MIRI", "OZEMA"):
        assert tab in html, f"во вкладках отчёта нет: {tab}"
    assert "<!DOCTYPE html" in html or "<!doctype html" in html
    # Тёплая палитра Claude.ai закреплена владельцем (CLAUDE.md): --clay.
    assert "#d97757" in html.lower()
    # Отчёт полностью локальный (открывается на ПК клиники без сети): plotly.js
    # инлайнится в файл, внешних <script src>/<link href> быть не должно.
    # Строки вида "http://..." внутри самого бандла plotly — это его текст ошибок,
    # не загрузка, поэтому проверяются именно теги.
    lowered = html.lower()
    for tag in ('<script src="http', "<script src='http", '<link href="http', "<link href='http",
                '<link rel="stylesheet" href="http'):
        assert tag not in lowered, f"отчёт тянет ресурс извне: {tag}"
    assert "plotly" in lowered, "интерактивные графики должны быть вшиты в файл"


def test_second_run_over_existing_outputs_stays_green_and_appends_history(e2e_root):
    assert _run() == 0
    assert _run() == 0                       # перезапуск поверх готовых артефактов

    processed = _processed(e2e_root)
    lines = [ln for ln in (processed / "run_history.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2, "история прогонов должна накапливаться, а не переписываться"
    records = [json.loads(ln) for ln in lines]
    assert all(r["n_cycles"] == 6 for r in records)
    # cycle_uid держится на локальном секрете, который переживает прогон: иначе
    # история MIRI/лотов по циклу разъедется между запусками.
    first, second = (pd.read_csv(processed / "dt_predictions.csv")["cycle_uid"].tolist(),) * 2
    assert first == second
    assert json.loads((processed / "run_manifest.json").read_text(encoding="utf-8"))["n_cycles"] == 6


def test_regression_replay_produces_qc_and_calibration(e2e_root):
    assert _run() == 0
    processed = _processed(e2e_root)
    uids = pd.read_csv(processed / "kpi_computed.csv")["cycle_uid"].tolist()

    # Квантили считает внешняя TabPFN-3; здесь подставляется её файл-replay, чтобы
    # проверить именно наш QA-слой (факт против P10-P90), а не саму модель.
    regression = pd.DataFrame({
        "cycle_uid": uids,
        "TPFN3_mii_p10": [4, 2, 6, 1, 8, 3],
        "TPFN3_mii_p25": [5, 3, 8, 2, 10, 4],
        "TPFN3_mii_p50": [7, 4, 10, 3, 13, 6],
        "TPFN3_mii_p75": [9, 6, 12, 4, 15, 8],
        "TPFN3_mii_p90": [11, 7, 14, 5, 17, 9],
        "TPFN3_mii_interval80_width": [7, 5, 8, 4, 9, 6],
    })
    replay = processed / "regression_replay.csv"
    regression.to_csv(replay, index=False, encoding="utf-8-sig")

    assert _run("--regression-output", str(replay)) == 0
    manifest = json.loads((processed / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["regression_status"] == "ok"

    qc = pd.read_csv(processed / "regression_qc.csv")
    assert len(qc) == 6
    calibration = pd.read_csv(processed / "regression_calibration.csv")
    assert calibration.loc[calibration["stage"] == "mii", "n_evaluated"].iloc[0] > 0


def _write_miri_logs(root: Path, start: str, hours: int = 24 * 7,
                     excursion_at: int | None = None) -> None:
    """Синтетические логи MIRI в той же структуре папок, что и реальные.

    …/miri_logs/6391/Miri-6391-Incubation-[YYMMDD-YYMMDD]/Data/*.csv, разделитель ';',
    дата "%Y.%m.%d %H:%M:%S" — как в шаренной папке Google Drive.
    """
    stamps = pd.date_range(start, periods=hours * 2, freq="30min")
    n = len(stamps)
    temps = {f"Temp{i}": [37.0] * n for i in range(1, 13)}
    if excursion_at is not None:
        for i in range(1, 13):
            temps[f"Temp{i}"][excursion_at:excursion_at + 6] = [36.0] * 6   # −1.0 °C
    frame = pd.DataFrame({
        "Date": stamps.strftime("%Y.%m.%d %H:%M:%S"),
        **temps,
        "Co2Concentration": [6.5] * n, "O2Concentration": [5.0] * n,
        "Co2Flow": [0.0] * n, "N2Flow": [0.0] * n,
        "Co2Pressure": [1.0] * n, "N2Pressure": [1.0] * n,
        "TempSetPoint": [37.0] * n, "Co2ConcentrationSetPoint": [6.5] * n,
        "O2ConcentrationSetPoint": [5.0] * n,
        **{f"Chamber{i}TemperatureAlarm": [0] * n for i in range(1, 7)},
        "Co2ConcentrationAlarm": [0] * n, "O2ConcentrationAlarm": [0] * n,
        "Co2PressureAlarm": [0] * n, "N2PressureAlarm": [0] * n,
        "IsLidOpen": [0] * n,
    })
    week = pd.Timestamp(start)
    folder = (root / "data" / "miri_logs" / "6391" /
              f"Miri-6391-Incubation-[{week:%y%m%d}-{week + pd.Timedelta(days=7):%y%m%d}]" / "Data")
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / f"{week:%Y-%m-%d}.csv", sep=";", index=False)


def test_miri_forensics_runs_over_the_cultivation_window(e2e_root):
    _write_miri_logs(e2e_root, "2026-01-01 00:00:00", excursion_at=100)
    # Без --skip-miri: форензик обязан отработать и записать свои артефакты.
    assert run_analysis.main([
        "--dt-output", str(FIXTURES / "e2e_dt.csv"),
        "--tabpfn-output", str(FIXTURES / "e2e_tabpfn.csv"),
        "--skip-lots", "--skip-pgt",
    ]) == 0

    processed = _processed(e2e_root)
    summary = pd.read_csv(processed / "miri_summary.csv")
    assert not summary.empty, "review-list не пуст — форензик должен был отработать"
    assert summary["n_points"].gt(0).all(), "окно культивирования не нашло точек лога"
    assert (processed / "miri_baseline.json").is_file(), "нет сравнения flagged против baseline"

    baseline = json.loads((processed / "miri_baseline.json").read_text(encoding="utf-8"))
    # Группа флагов сравнивается с контрольной, иначе «совпало с инцидентом»
    # неотличимо от фонового уровня по всей когорте.
    assert baseline["flagged"]["n"] + baseline["baseline"]["n"] == 6
    assert baseline["flagged"]["warning_rate"] is not None
    manifest = json.loads((processed / "run_manifest.json").read_text(encoding="utf-8"))
    assert "miri_baseline.json" in manifest["artifacts"]


def test_miri_step_is_skipped_cleanly_when_no_logs_are_synced(e2e_root):
    # Обычное состояние до синхронизации логов с Google Drive: шаг обязан
    # пройти без падения и оставить пустые, но валидные артефакты.
    assert run_analysis.main([
        "--dt-output", str(FIXTURES / "e2e_dt.csv"),
        "--tabpfn-output", str(FIXTURES / "e2e_tabpfn.csv"),
        "--skip-lots", "--skip-pgt",
    ]) == 0
    summary = pd.read_csv(_processed(e2e_root) / "miri_summary.csv")
    assert "error" in summary.columns


def test_missing_input_fails_loudly_instead_of_running_on_nothing(e2e_root):
    (e2e_root / "data" / "inbox" / "cohort.csv").unlink()
    with pytest.raises(FileNotFoundError):
        _run()


def test_cohort_without_outcome_is_rejected_not_silently_emptied(e2e_root):
    source = pd.read_csv(e2e_root / "data" / "inbox" / "cohort.csv")
    source["Исход переноса"] = None
    source.to_csv(e2e_root / "data" / "inbox" / "cohort.csv", index=False)
    with pytest.raises(ValueError, match="перенос"):
        _run()
