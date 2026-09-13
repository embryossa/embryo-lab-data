"""scripts/preflight.py и scripts/check_setup.py — проверки окружения.

Оба скрипта существуют ровно для того, чтобы МЯГКИЙ сбой окружения стал видимым
до прогона (preflight.py: 19.07.2026 без pymupdf молча упали все 11 разборов
ПГТ-А). Поэтому здесь проверяется именно их способность отличить «всё на месте»
от «дыра в окружении» и вернуть корректный код возврата — а не то, что они
просто импортируются.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import check_setup  # noqa: E402
import preflight  # noqa: E402


# ---------------------------------------------------------------- preflight

def test_requirements_parser_strips_comments_and_keeps_specifiers(tmp_path):
    path = tmp_path / "requirements.txt"
    path.write_text(
        "# заголовок\n"
        "pandas>=2.0\n"
        "\n"
        "openpyxl>=3.1        # комментарий в конце строки\n"
        "scikit-learn\n"
        "python-calamine>=0.2\n",
        encoding="utf-8",
    )
    assert preflight._parse_requirements(path) == [
        ("pandas", ">=2.0"),
        ("openpyxl", ">=3.1"),
        ("scikit-learn", ""),
        ("python-calamine", ">=0.2"),
    ]


def test_real_requirements_file_parses_and_every_name_is_installed():
    parsed = preflight._parse_requirements(REPO / "requirements.txt")
    assert parsed, "requirements.txt не разобрался"
    # Тот же список, что читает preflight на живой машине: если сюда добавят
    # пакет с опечаткой в имени, проверка окружения начнёт врать про «не установлен».
    assert preflight.check_packages() == []


def test_soft_fail_imports_are_all_importable_here():
    # Именно этот список отделяет «у клиники нет отчётов ПГТ-А» от «нет pymupdf».
    assert preflight.check_soft_imports() == []
    assert preflight.SOFT_FAIL_IMPORTS["pymupdf"] == "fitz"


def test_missing_package_is_reported_not_swallowed(tmp_path, monkeypatch):
    path = tmp_path / "requirements.txt"
    path.write_text("pandas>=2.0\nembryo-nonexistent-package>=1.0\n", encoding="utf-8")
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    problems = preflight.check_packages()
    assert len(problems) == 1
    assert "НЕ УСТАНОВЛЕН" in problems[0] and "embryo-nonexistent-package" in problems[0]


def test_version_below_specifier_is_reported(tmp_path, monkeypatch):
    path = tmp_path / "requirements.txt"
    path.write_text("pandas>=99.0\n", encoding="utf-8")
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    problems = preflight.check_packages()
    assert len(problems) == 1 and problems[0].startswith("ВЕРСИЯ: pandas")


def test_yaml_loader_treats_absent_and_empty_file_as_empty_config(tmp_path):
    assert preflight._load_yaml(tmp_path / "нет-такого.yaml") == {}
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert preflight._load_yaml(empty) == {}


def test_deep_merge_overrides_leaf_and_keeps_sibling_keys():
    base = {"external_tools": {"tabpfn": {"launch": True, "project_dir": ""},
                               "digital_twin": {"launch": True}}}
    merged = preflight._deep_merge(base, {"external_tools": {"tabpfn": {"project_dir": "C:/t"}}})
    assert merged["external_tools"]["tabpfn"] == {"launch": True, "project_dir": "C:/t"}
    assert merged["external_tools"]["digital_twin"] == {"launch": True}
    assert base["external_tools"]["tabpfn"]["project_dir"] == "", "_deep_merge не должен мутировать base"


def _write_tool_config(root: Path, tools: dict) -> None:
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "config" / "settings.yaml").write_text(
        yaml.safe_dump({"external_tools": tools}, allow_unicode=True), encoding="utf-8")


@pytest.fixture
def no_live_dt(monkeypatch):
    """Живой вызов службы DT 7.1 подменяется: проверяется логика preflight, а не эта машина."""
    calls = []
    monkeypatch.setattr(preflight, "check_digital_twin_api", lambda cfg: calls.append(cfg) or [])
    return calls


def test_disabled_tools_are_skipped_not_reported_as_broken(tmp_path, monkeypatch, no_live_dt):
    _write_tool_config(tmp_path, {
        "digital_twin": {"api": {"ca_cert": "server.crt"}},
        "tabpfn": {"launch": False},
        "dnn_verifier": {"enabled": False},
        "tabpfn_regression": {"enabled": False},
    })
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    assert preflight.check_external_tools(None) == []
    # Digital Twin не имеет гейта: API 7.1 проверяется всегда, для любой клиники.
    assert no_live_dt == [{"api": {"ca_cert": "server.crt"}}]


def test_digital_twin_is_checked_only_through_the_71_api(tmp_path, monkeypatch):
    _write_tool_config(tmp_path, {"digital_twin": {"api": {"ca_cert": str(tmp_path / "нет.crt")}}})
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    problems = preflight.check_external_tools(None)
    assert len(problems) == 1 and "нет сертификата API" in problems[0]
    # Явный запрос батча 7.0 — проблема, а не молчаливый переход на API.
    _write_tool_config(tmp_path, {"digital_twin": {"source": "batch", "run_cmd": "RUN_BATCH.bat {input}"}})
    problems = preflight.check_external_tools(None)
    assert len(problems) == 1 and "батч 7.0 снят" in problems[0]


def test_enabled_tool_with_empty_or_missing_path_is_reported(tmp_path, monkeypatch, no_live_dt):
    real_dir = tmp_path / "tabpfn_project"
    real_dir.mkdir()
    _write_tool_config(tmp_path, {
        "digital_twin": {"api": {}},
        # results_dir указывает в никуда, project_dir существует -> ровно одна проблема
        "tabpfn": {"launch": True, "project_dir": str(real_dir),
                   "results_dir": str(tmp_path / "нет-такой-папки")},
        # enabled, но пути вообще не заданы -> три проблемы
        "dnn_verifier": {"enabled": True},
        "tabpfn_regression": {"enabled": False},
    })
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    problems = preflight.check_external_tools(None)

    assert any("TabPFN: путь не существует" in p for p in problems)
    assert sum("DNN (EVO): не задан" in p for p in problems) == 3
    assert not any("project_dir" in p and p.startswith("TabPFN:") for p in problems)


def test_preflight_main_returns_zero_when_tools_are_skipped(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["preflight.py", "--skip-tools"])
    assert preflight.main() == 0
    assert "окружение в порядке" in capsys.readouterr().out


def test_preflight_main_returns_one_when_a_package_is_missing(tmp_path, monkeypatch, capsys):
    (tmp_path / "requirements.txt").write_text("embryo-nonexistent-package>=1.0\n", encoding="utf-8")
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["preflight.py", "--skip-tools"])
    assert preflight.main() == 1
    assert "ПРОБЛЕМ: 1" in capsys.readouterr().out


@pytest.mark.parametrize("argv, expected", [
    (["preflight.py", "--skip-tools", "--fertimed"], "fertimed"),
    (["preflight.py", "--skip-tools", "--ggrc"], "ggrc"),
    (["preflight.py", "--skip-tools", "--clinic", "skyfert"], "skyfert"),
    (["preflight.py", "--skip-tools"], "A (по умолчанию)"),
])
def test_preflight_resolves_clinic_shortcuts(argv, expected, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", argv)
    preflight.main()
    assert f"клиника : {expected}" in capsys.readouterr().out


# -------------------------------------------------------------- check_setup

@pytest.fixture
def isolated_check_setup(tmp_path, monkeypatch):
    """ROOT в tmp: иначе _settings подмешает НАСТОЯЩИЙ config/machine.local.yaml этой машины."""
    monkeypatch.setattr(check_setup, "ROOT", tmp_path)
    return tmp_path


def test_check_setup_counts_missing_paths_and_honours_gates(isolated_check_setup, tmp_path, capsys):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    cert = tmp_path / "server.crt"
    cert.write_text("", encoding="utf-8")
    (config_dir / "settings.yaml").write_text(yaml.safe_dump({"external_tools": {
        "digital_twin": {"api": {"ca_cert": str(cert)}},
        "tabpfn": {"project_dir": str(tmp_path / "нет-tabpfn")},
        "dnn_verifier": {"enabled": False, "model_path": "", "scaler_path": ""},
        "tabpfn_regression": {"enabled": False},
    }}, allow_unicode=True), encoding="utf-8")

    # TabPFN project + BAT + .py = 3 отсутствующих; выключенные DNN/TabPFN-3 в список
    # проверок вообще не попадают. Digital Twin — только сертификат службы 7.1.
    assert check_setup.check(config_dir, "тест") == 3
    out = capsys.readouterr().out
    assert "[OK] Digital Twin 7.1 API certificate" in out
    assert "Digital Twin project" not in out
    assert "DNN" not in out



def test_check_setup_rejects_an_explicit_70_batch_source(isolated_check_setup, tmp_path, capsys):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "settings.yaml").write_text(yaml.safe_dump({"external_tools": {
        "digital_twin": {"source": "batch", "project_dir": str(tmp_path)},
        "tabpfn": {"project_dir": str(tmp_path)},
    }}, allow_unicode=True), encoding="utf-8")
    assert check_setup.check(config_dir, "тест") == 1
    assert "батч 7.0 снят" in capsys.readouterr().out


def test_check_setup_reports_ready_environment(isolated_check_setup, tmp_path, capsys):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    cert, tab = tmp_path / "server.crt", tmp_path / "tabpfn"
    cert.write_text("", encoding="utf-8")
    tab.mkdir()
    (tab / "run_tabpfn_batch.bat").write_text("", encoding="utf-8")
    (tab / "run_tabpfn_batch.py").write_text("", encoding="utf-8")
    (config_dir / "settings.yaml").write_text(yaml.safe_dump({"external_tools": {
        "digital_twin": {"api": {"ca_cert": str(cert)}},
        "tabpfn": {"project_dir": str(tab)},
    }}, allow_unicode=True), encoding="utf-8")

    assert check_setup.check(config_dir, "тест") == 0
    assert "Окружение готово" in capsys.readouterr().out


def test_check_setup_merges_clinic_local_over_base(isolated_check_setup, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "settings.yaml").write_text(yaml.safe_dump({"external_tools": {
        "digital_twin": {"api": {"ca_cert": "", "n_sim": 2000}}}}, allow_unicode=True), encoding="utf-8")
    (config_dir / "settings.local.yaml").write_text(yaml.safe_dump({"external_tools": {
        "digital_twin": {"api": {"ca_cert": "C:/real/server.crt"}}}}, allow_unicode=True), encoding="utf-8")

    cfg = check_setup._settings(config_dir)
    assert cfg["external_tools"]["digital_twin"]["api"] == {"ca_cert": "C:/real/server.crt", "n_sim": 2000}


def _fake_clinic_tree(root: Path, clinics: dict[str, bool]) -> None:
    """clinics: код -> все ли пути существуют."""
    for name, healthy in clinics.items():
        config_dir = root / "config" / "clinics" / name if name else root / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        cert, tab = root / f"dt_{name or 'base'}.crt", root / f"tab_{name or 'base'}"
        if healthy:
            cert.write_text("", encoding="utf-8")
            tab.mkdir()
            (tab / "run_tabpfn_batch.bat").write_text("", encoding="utf-8")
            (tab / "run_tabpfn_batch.py").write_text("", encoding="utf-8")
        (config_dir / "settings.yaml").write_text(yaml.safe_dump({"external_tools": {
            "digital_twin": {"api": {"ca_cert": str(cert)}},
            "tabpfn": {"project_dir": str(tab)},
        }}, allow_unicode=True), encoding="utf-8")


def test_check_setup_main_returns_zero_when_the_base_clinic_is_ready(
        isolated_check_setup, tmp_path, monkeypatch):
    _fake_clinic_tree(tmp_path, {"": True})
    monkeypatch.setattr(sys, "argv", ["check_setup.py"])
    assert check_setup.main() == 0


def test_check_setup_main_walks_every_clinic_and_fails_on_the_broken_one(
        isolated_check_setup, tmp_path, monkeypatch, capsys):
    _fake_clinic_tree(tmp_path, {"": True, "fertimed": True, "ggrc": False})
    monkeypatch.setattr(sys, "argv", ["check_setup.py", "--all-clinics"])
    assert check_setup.main() == 1
    out = capsys.readouterr().out
    assert "клиника A (base)" in out and "клиника fertimed" in out and "клиника ggrc" in out
    assert "[MISSING]" in out


def test_check_setup_main_can_target_one_clinic(isolated_check_setup, tmp_path, monkeypatch, capsys):
    _fake_clinic_tree(tmp_path, {"fertimed": True, "ggrc": False})
    monkeypatch.setattr(sys, "argv", ["check_setup.py", "--clinic", "fertimed"])
    assert check_setup.main() == 0
    assert "ggrc" not in capsys.readouterr().out


def test_every_shipped_clinic_config_is_loadable_by_check_setup():
    # Конфиг клиники, который не читается, ломает прогон этой клиники целиком —
    # дешевле поймать здесь, чем на живой выгрузке.
    for config_dir in [REPO / "config", *sorted((REPO / "config" / "clinics").iterdir())]:
        if not (config_dir / "settings.yaml").is_file():
            continue
        cfg = check_setup._settings(config_dir)
        assert "external_tools" in cfg, config_dir
        assert "digital_twin" in cfg["external_tools"], config_dir
        # Ни одна клиника не несёт батч 7.0: вход DT — только API 7.1.
        dt_cfg = cfg["external_tools"]["digital_twin"]
        assert dt_cfg.get("api", {}).get("timepoints") == ["transfer", "retrieval"], config_dir
        assert not any(k in dt_cfg for k in ("source", "run_cmd", "project_dir", "results_dir")), config_dir
        assert "tabpfn" in cfg["external_tools"], config_dir
