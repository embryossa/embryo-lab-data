"""
dt_runner.py — ВНЕШНИЙ IVF Digital Twin (своё окружение/проект).
Авто-режим: (1) запускаем проект DT его командой (settings.external_tools.digital_twin.run_cmd),
(2) читаем самый свежий dt_analytics_*.csv из results_dir. DT пишет 92 колонки
(_key/Preg/PRAI/DIGITAL TWIN + ансамбль).
"""
from __future__ import annotations
import time
import pandas as pd
from ._io import load_latest, latest_file_after
from ._launch import run_external


def launch_digital_twin(cfg: dict, input_path: str | None = None) -> bool:
    return run_external(cfg.get("run_cmd"), cfg.get("project_dir", "."), input_path)


def read_dt(results_dir: str, prefer_outcome=False,
            outcome_glob="dt_analytics_with_outcome_*.csv",
            ready_glob="dt_analytics_ready_*.csv"):
    """Читает свежий результат DT. По умолчанию — ready, не with_outcome.

    Подтверждено (2026-07-17): DT-проект пишет dt_analytics_with_outcome_*.csv так,
    что Preg=0 трактуется как «нет исхода» и строка целиком выпадает (не только
    значение) — "With outcome (Preg): 9 (+9/-0)" при 12 реальных Preg=0 в этом же
    прогоне. Это нарушает "0 — известное наблюдение, не пропуск" (CLAUDE.md).
    ready содержит все строки текущего прогона (Preg у отрицательных — NaN, но сама
    строка на месте) — reconcile.build_dt_predictions уже восстанавливает Preg из
    OPU через combine_first, так что использование ready не теряет 0-исходы.
    prefer_outcome=True оставлен на случай, если внешний скрипт когда-нибудь
    починят.
    """
    try:
        return load_latest(results_dir, outcome_glob if prefer_outcome else ready_glob)
    except FileNotFoundError:
        return load_latest(results_dir, ready_glob)


def launch_and_read(cfg: dict, input_path: str | None = None,
                    prefer_outcome: bool = False) -> pd.DataFrame:
    """Запускает DT и принимает только результат, созданный текущим запуском."""
    started = time.time()
    if cfg.get("launch", True):
        launch_digital_twin(cfg, input_path)
    patterns = (["dt_analytics_with_outcome_*.csv", "dt_analytics_ready_*.csv"]
                if prefer_outcome else ["dt_analytics_ready_*.csv"])
    for pattern in patterns:
        f = latest_file_after(cfg["results_dir"], pattern, started)
        if f is not None:
            return pd.read_csv(f)
    raise FileNotFoundError(
        "Digital Twin не создал свежий dt_analytics_*.csv в текущем запуске; "
        "старый результат намеренно не используется."
    )
