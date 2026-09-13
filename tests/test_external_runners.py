"""Адаптеры внешних инструментов: _launch, tabpfn_runner, tabpfn_regression_runner.

Сами DT/TabPFN/TabPFN-3 живут в своих окружениях и в тестах не участвуют. Проверяется
наш слой вокруг них — тот, что решает «результат свежий или это файл прошлого прогона»,
«команда получила {input}/{output}/{run_id}» и «что уходит дальше по пайплайну».
Ровно здесь ошибка стоит дороже всего: подхваченный старый файл выглядит как
успешный прогон и даёт отчёт по чужим данным (CLAUDE.md: «Старый файл текущим
результатом не считается»).

Внешний процесс подменяется настоящим дочерним python-процессом, а не моком: так
проверяется и подстановка шаблона, и запуск, и чтение результата.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
import pytest

from embryo import _launch, tabpfn_regression_runner, tabpfn_runner


def _writer_script(tmp_path: Path, body: str) -> Path:
    """Мини-скрипт, изображающий внешний инструмент."""
    path = tmp_path / "fake_tool.py"
    path.write_text(body, encoding="utf-8")
    return path


def _cmd(script: Path, *args: str) -> str:
    quoted = " ".join(args)
    return f'"{sys.executable}" "{script}" {quoted}'


# --------------------------------------------------------------- _launch

def test_subst_fills_input_output_and_repo_root():
    cmd = _launch._subst("tool {input} -o {output} --root {repo_root}", "C:/in.xlsx", "C:/out.csv")
    assert '"C:/in.xlsx"' in cmd and '"C:/out.csv"' in cmd
    assert "{repo_root}" not in cmd
    # {repo_root} — корень ЭТОГО репозитория, чтобы run_cmd переживал переезд машины.
    assert (Path(cmd.split("--root ", 1)[1]) / "src" / "embryo").is_dir()


def test_subst_leaves_placeholders_alone_when_no_value_given():
    assert _launch._subst("tool {input}", None, None) == "tool {input}"


def test_child_env_forces_utf8_and_keeps_console_codepage():
    env = _launch._child_env()
    assert env["PYTHONIOENCODING"] == "utf-8"
    # Внешний RUN_BATCH.bat иначе делает chcp 1251 на ВСЮ консоль на часы прогона.
    assert env["DT_KEEP_CODEPAGE"] == "1"
    assert env is not os.environ


def test_run_external_without_command_is_a_no_op():
    assert _launch.run_external("", ".") is None
    assert _launch.run_external(None, ".") is None


def test_run_external_runs_the_command_and_propagates_failure(tmp_path):
    marker = tmp_path / "ran.txt"
    script = _writer_script(tmp_path, f"open(r'{marker}', 'w').write('ok')\n")
    assert _launch.run_external(_cmd(script), str(tmp_path)) is True
    assert marker.read_text() == "ok"

    # Ненулевой код внешнего инструмента обязан всплыть, а не быть проглочен:
    # иначе пайплайн пойдёт искать результат, которого нет.
    failing = _writer_script(tmp_path, "raise SystemExit(3)\n")
    with pytest.raises(subprocess.CalledProcessError):
        _launch.run_external(_cmd(failing), str(tmp_path))


# --------------------------------------------------------- tabpfn_runner

_TABPFN_ROWS = ("patient_id,p_tabpfn,ci_low,ci_high,run_id\n"
                "A#20260101,0.7,0.6,0.8,{run_id}\n"
                "B#20260102,0.3,0.2,0.45,{run_id}\n")


def _tabpfn_cfg(tmp_path: Path, results: Path, *, rows: str = _TABPFN_ROWS) -> dict:
    script = _writer_script(tmp_path, (
        "import sys, pathlib\n"
        "out, run_id = sys.argv[1], sys.argv[2]\n"
        f"pathlib.Path(out).write_text({rows!r}.replace('{{run_id}}', run_id), encoding='utf-8')\n"
    ))
    return {
        "launch": True, "mode": "batch", "project_dir": str(tmp_path),
        "results_dir": str(results),
        "run_cmd": _cmd(script, "{output}", "{run_id}") + " {input}",
        "output": str(results / "tabpfn_predictions.csv"),
    }


def test_tabpfn_batch_run_produces_validated_frame_with_ci_width(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    frame = tabpfn_runner.launch_and_read(_tabpfn_cfg(tmp_path, results),
                                          input_path=str(tmp_path / "model_input.xlsx"))
    assert frame["patient_id"].tolist() == ["A#20260101", "B#20260102"]
    assert frame["ci_width_pp"].tolist() == pytest.approx([20.0, 25.0])


def test_tabpfn_batch_requires_all_three_placeholders(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    cfg = _tabpfn_cfg(tmp_path, results)
    cfg["run_cmd"] = cfg["run_cmd"].replace(" {input}", "")
    with pytest.raises(ValueError, match=r"\{input\}"):
        tabpfn_runner.launch_and_read(cfg, input_path=str(tmp_path / "in.xlsx"))


def test_tabpfn_batch_requires_an_input_path(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    with pytest.raises(ValueError, match="explicit input path"):
        tabpfn_runner.launch_and_read(_tabpfn_cfg(tmp_path, results), input_path=None)


def test_tabpfn_rejects_output_stamped_with_another_run(tmp_path):
    # Единственная защита от «инструмент дописал старый файл»: run_id этого запуска.
    results = tmp_path / "results"
    results.mkdir()
    cfg = _tabpfn_cfg(tmp_path, results, rows=_TABPFN_ROWS.replace("{run_id}", "чужой-run"))
    with pytest.raises(ValueError, match="run_id mismatch"):
        tabpfn_runner.launch_and_read(cfg, input_path=str(tmp_path / "in.xlsx"))


def test_tabpfn_reports_missing_output_instead_of_reusing_an_old_one(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    cfg = _tabpfn_cfg(tmp_path, results)
    cfg["run_cmd"] = _cmd(_writer_script(tmp_path, "pass\n"), "{output}", "{run_id}") + " {input}"
    with pytest.raises(FileNotFoundError, match="explicit current-run output"):
        tabpfn_runner.launch_and_read(cfg, input_path=str(tmp_path / "in.xlsx"))


def test_read_tabpfn_validates_the_contract(tmp_path):
    path = tmp_path / "tabpfn_predictions.csv"
    path.write_text("patient_id,p_tabpfn\nA,0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        tabpfn_runner.read_tabpfn(explicit_path=path)

    path.write_text("patient_id,p_tabpfn,ci_low,ci_high\nA,0.5,0.9,0.2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        tabpfn_runner.read_tabpfn(explicit_path=path)   # ci_low > ci_high


def test_read_tabpfn_needs_a_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        tabpfn_runner.read_tabpfn(explicit_path=tmp_path / "нет.csv")
    with pytest.raises(ValueError, match="results_dir"):
        tabpfn_runner.read_tabpfn()
    with pytest.raises(FileNotFoundError, match="no TabPFN results"):
        tabpfn_runner.read_tabpfn(results_dir=str(tmp_path))


# ---------------------------------------------- tabpfn_regression_runner

_REGRESSION_COLS = ["TPFN3_mii_p10", "TPFN3_mii_p25", "TPFN3_mii_p50",
                    "TPFN3_mii_p75", "TPFN3_mii_p90", "TPFN3_mii_interval80_width"]


def _regression_rows() -> str:
    header = ",".join(["cycle_uid", "Patient Full Name", "DOB", *_REGRESSION_COLS])
    return (f"{header}\n"
            "CYC-0000000000000001,Пациент Один,1990-01-01,4,5,7,9,11,7\n"
            "CYC-0000000000000002,Пациент Два,1988-02-02,2,3,4,6,7,5\n")


def test_regression_runner_requires_output_input_and_placeholders(tmp_path):
    with pytest.raises(ValueError, match="явный output"):
        tabpfn_regression_runner.run_tabpfn_regression({}, "in.xlsx")
    with pytest.raises(ValueError, match=r"\{input\}"):
        tabpfn_regression_runner.run_tabpfn_regression(
            {"output": str(tmp_path / "o.csv"), "run_cmd": "tool"}, "in.xlsx")
    with pytest.raises(ValueError, match="явный input_path"):
        tabpfn_regression_runner.run_tabpfn_regression(
            {"output": str(tmp_path / "o.csv"), "run_cmd": "tool {input} {output}"}, None)


def test_regression_runner_reads_and_validates_a_fresh_output(tmp_path):
    out = tmp_path / "regression_full.csv"
    script = _writer_script(tmp_path, (
        "import sys, pathlib\n"
        f"pathlib.Path(sys.argv[1]).write_text({_regression_rows()!r}, encoding='utf-8')\n"
    ))
    cfg = {"output": str(out), "project_dir": str(tmp_path),
           "run_cmd": _cmd(script, "{output}") + " {input}"}
    frame = tabpfn_regression_runner.run_tabpfn_regression(cfg, str(tmp_path / "model_input.xlsx"))
    assert len(frame) == 2


def test_regression_runner_rejects_a_stale_output(tmp_path):
    out = tmp_path / "regression_full.csv"
    out.write_text(_regression_rows(), encoding="utf-8")
    os.utime(out, (time.time() - 3600, time.time() - 3600))
    cfg = {"output": str(out), "project_dir": str(tmp_path),
           "run_cmd": _cmd(_writer_script(tmp_path, "pass\n")) + " {input} {output}"}
    with pytest.raises(FileNotFoundError, match="не создал свежий output"):
        tabpfn_regression_runner.run_tabpfn_regression(cfg, str(tmp_path / "in.xlsx"))


def test_extract_predictions_strips_phi_down_to_cycle_uid_and_quantiles():
    frame = pd.read_csv(pd.io.common.StringIO(_regression_rows()))
    clean, stages = tabpfn_regression_runner.extract_predictions(frame)
    assert stages == ["mii"]
    # Скрипт TabPFN-3 пишет прогноз поверх исходного кадра с ФИО/ДР — дальше по
    # пайплайну не должно уйти ничего, кроме cycle_uid и TPFN3_*.
    assert list(clean.columns) == ["cycle_uid", *_REGRESSION_COLS]
    assert "Patient Full Name" not in clean.columns
