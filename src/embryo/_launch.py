"""_launch.py — запуск внешнего инструмента (DT/TabPFN/DNN) в его окружении/каталоге.
run_cmd — шаблон команды; {input}/{output} подставляются, если присутствуют.
blocking=True — ждём завершения (батч-инструменты: RUN_BATCH.bat, DNN-скрипт).
blocking=False — фоновый запуск (Popen) для серверов (Streamlit TabPFN)."""
from __future__ import annotations
import os
import subprocess


def _subst(cmd, input_path, output_path):
    if input_path and "{input}" in cmd:
        cmd = cmd.replace("{input}", f'"{input_path}"')
    if output_path and "{output}" in cmd:
        cmd = cmd.replace("{output}", f'"{output_path}"')
    return cmd


def _child_env():
    # Внешние батники (напр. RUN_BATCH.bat) сами делают `chcp 1251`, а их питон-процесс
    # без явного PYTHONIOENCODING кодирует кириллицу под текущую кодовую страницу
    # консоли — родительский терминал ждёт UTF-8, отсюда мусор в выводе.
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_external(run_cmd, project_dir, input_path=None, output_path=None, blocking=True):
    """blocking -> True (завершился) ; non-blocking -> объект Popen ; run_cmd пуст -> None."""
    if not run_cmd:
        return None
    cmd = _subst(run_cmd, input_path, output_path)
    env = _child_env()
    if blocking:
        # stdin=DEVNULL: некоторые батники (напр. RUN_BATCH.bat) заканчиваются
        # безусловным `pause`; без EOF на stdin это подвесит headless-запуск.
        subprocess.run(cmd, shell=True, check=True, cwd=project_dir, stdin=subprocess.DEVNULL, env=env)
        return True
    return subprocess.Popen(cmd, shell=True, cwd=project_dir, stdin=subprocess.DEVNULL, env=env)
