"""
dnn_runner.py — ОПЦИОНАЛЬНАЯ независимая DNN-модель (Evo_calculations.py) для
перекрёстной проверки Digital Twin и TabPFN. Обычная DNN (НЕ KAN), своё окружение
(иная версия torch), запуск в PyCharm-проекте. По умолчанию enabled: false
(settings.external_tools.dnn_verifier). Только локально, пока не перенесена.

Авто-режим требует сохранённый scaler, запускает скрипт с явным output и
валидирует current-run контракт до присоединения результата.
"""
from __future__ import annotations
from pathlib import Path
import time
import pandas as pd
from ._launch import run_external
from .contracts import assert_unique


def run_dnn_verifier(cfg: dict, input_path: str | None = None) -> pd.DataFrame:
    scaler = cfg.get("scaler_path")
    if not scaler:
        raise ValueError("DNN verifier is invalid without the saved training scaler_path")
    if not Path(scaler).is_file():
        raise FileNotFoundError(f"DNN training scaler not found: {scaler}")
    out = cfg.get("output") or None
    cmd = cfg.get("run_cmd")
    if not out:
        raise ValueError("DNN verifier requires an explicit output path")
    out = str(Path(out).resolve())
    if not cmd or any(token not in cmd for token in ("{input}", "{output}", "{scaler}")):
        raise ValueError("DNN run_cmd must forward {input}, {output}, and {scaler}")
    cmd = cmd.replace("{scaler}", f'"{scaler}"')
    if "{model}" in cmd:
        model = cfg.get("model_path")
        if not model:
            raise ValueError("DNN run_cmd uses {model}, but model_path is not configured")
        if not Path(model).exists():
            raise FileNotFoundError(f"DNN saved model not found: {model}")
        cmd = cmd.replace("{model}", f'"{model}"')
    started = time.time()
    launched = run_external(cmd, cfg.get("project_dir", "."),
                            input_path, output_path=out)
    if launched and Path(out).exists() and Path(out).stat().st_mtime >= started - 2:
        frame = pd.read_csv(out)
        required = {"patient_id", "Исход переноса EVO",
                    "Вероятность положительного исхода", "dnn_validation_status"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"DNN output contract missing columns: {sorted(missing)}")
        assert_unique(frame, "patient_id", "DNN output")
        if not frame["dnn_validation_status"].eq("valid_saved_scaler").all():
            raise ValueError("DNN output was not produced with a validated saved scaler")
        probability = pd.to_numeric(frame["Вероятность положительного исхода"], errors="raise")
        if ((probability < 0) | (probability > 1)).any():
            raise ValueError("DNN output probability must be in [0, 1]")
        return frame
    raise FileNotFoundError(f"DNN did not create its explicit output: {out}")
