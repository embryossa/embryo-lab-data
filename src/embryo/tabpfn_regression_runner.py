"""
tabpfn_regression_runner.py — ОПЦИОНАЛЬНАЯ независимая TabPFN-3 regression модель
(tabpfn3_ivf_regression_report_CPU_16K.py) для количественного прогноза стадий
воронки (MII/2PN/Bl/Good Bl/Frozen), своё окружение (.venv в проекте tabpfn).
По умолчанию enabled: true (settings.external_tools.tabpfn_regression), но сбой
этого шага в scripts/run_analysis.py всегда мягкий — эта модель дополняет
DT/TabPFN, а не заменяет их реконciliation по исходу переноса.

Скрипт пишет предсказания поверх ИСХОДНОГО OPU-фрейма (полный ФИО/DOB/муж) — вход
должен быть model_input.xlsx (data/processed/local_private/), который уже несёт ту
же чувствительность и уже содержит cycle_uid. Явный output этого раннера тоже
обязан указывать в local_private/: PHI никогда не покидает эту функцию — наружу
отдаётся только cycle_uid + TPFN3_* колонки через extract_predictions().
"""
from __future__ import annotations

from pathlib import Path
import time

import pandas as pd

from ._launch import run_external
from .contracts import detect_regression_stages, validate_regression_quantiles


def run_tabpfn_regression(cfg: dict, input_path: str | None = None) -> pd.DataFrame:
    out = cfg.get("output")
    if not out:
        raise ValueError("tabpfn_regression требует явный output (местный private-путь)")
    out = str(Path(out).resolve())
    cmd = cfg.get("run_cmd")
    if not cmd or any(token not in cmd for token in ("{input}", "{output}")):
        raise ValueError("tabpfn_regression run_cmd must forward {input} and {output}")
    if not input_path:
        raise ValueError("tabpfn_regression требует явный input_path (model_input.xlsx)")

    started = time.time()
    run_external(cmd, cfg.get("project_dir", "."), input_path, output_path=out, blocking=True)

    if not (Path(out).exists() and Path(out).stat().st_mtime >= started - 2):
        raise FileNotFoundError(f"TabPFN-3 regression не создал свежий output: {out}")

    frame = pd.read_csv(out, encoding="utf-8-sig")
    validate_regression_quantiles(frame)
    return frame


def extract_predictions(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Strip PHI down to cycle_uid + TPFN3_* prediction columns for downstream use."""
    stages = detect_regression_stages(frame)
    keep = ["cycle_uid"] + [c for c in frame.columns if c.startswith("TPFN3_")]
    return frame[keep].copy(), stages
