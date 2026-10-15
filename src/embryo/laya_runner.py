"""
laya_runner.py — Laya (laya_kan_ft) как внешний компаратор Digital Twin.

Не член ансамбля DT L1–L7 и не замена DT: вторая независимая оценка, обученная
только на KAN (лабораторные счётчики). Валидация на 1 387 циклах Teable (2026-10-15):
внутрицентровый AUC Laya 0.555 против DT 0.559, а флаг `laya_disagreement_high`
отмечает циклы, где Brier DT выше (+0.026 [0.004; 0.047]) — повод перепроверить.

Своё окружение (.venv проекта Laya), запуск адаптера external/laya/laya_batch.py.
На вход уходят только cycle_uid, счётчики воронки и DIGITAL TWIN — без ID/ФИО/дат.
Мягкий шаг: сбой не блокирует пайплайн (как TabPFN-3 regression и DNN).
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from ._launch import run_external
from .contracts import assert_unique

INPUT_COLUMNS = ["cycle_uid", "Age", "Attempt", "N folicules OPU", "OCC", "MII", "Inseminated",
                 "2pN", "real_cleav", "Bl", "Good Bl", "real_cryo", "DIGITAL TWIN"]
OUTPUT_COLUMNS = ["Laya", "laya_raw", "laya_assessable", "laya_minus_dt_pp",
                  "laya_disagreement_high", "laya_flags", "laya_restored"]


def run_laya(cfg: dict, predictions: pd.DataFrame, private_dir: Path) -> pd.DataFrame:
    cmd = cfg.get("run_cmd")
    if not cmd or "{input}" not in cmd or "{output}" not in cmd:
        raise ValueError("Laya run_cmd must forward {input} and {output} (config/machine.local.yaml)")
    if "cycle_uid" not in predictions.columns:
        raise ValueError("Laya needs cycle_uid in predictions")
    cols = [c for c in INPUT_COLUMNS if c in predictions.columns]
    frame = predictions[cols].copy()
    frame = frame[frame["cycle_uid"].notna()]
    assert_unique(frame, "cycle_uid", "Laya input")
    private_dir.mkdir(parents=True, exist_ok=True)
    inp = (private_dir / "laya_input.csv").resolve()
    out = (private_dir / "laya_predictions.csv").resolve()
    frame.to_csv(inp, index=False, encoding="utf-8-sig")
    if cfg.get("centre"):
        cmd += f' --centre "{cfg["centre"]}"'
    if cfg.get("restore", False):
        cmd += " --restore"
    started = time.time()
    run_external(cmd, cfg.get("project_dir", "."), str(inp), output_path=str(out))
    if not out.exists() or out.stat().st_mtime < started - 2:
        raise FileNotFoundError(f"Laya did not create its output: {out}")
    res = pd.read_csv(out, encoding="utf-8-sig")
    missing = {"cycle_uid", *OUTPUT_COLUMNS}.difference(res.columns)
    if missing:
        raise ValueError(f"Laya output contract missing columns: {sorted(missing)}")
    assert_unique(res, "cycle_uid", "Laya output")
    if set(res["cycle_uid"].astype(str)) != set(frame["cycle_uid"].astype(str)):
        raise ValueError("Laya output cycles differ from its input")
    for col in ("Laya", "laya_raw"):
        p = pd.to_numeric(res[col], errors="raise")
        if ((p < 0) | (p > 1)).any():
            raise ValueError(f"Laya {col} must be in [0, 1]")
    return res


def join_laya(predictions: pd.DataFrame, laya: pd.DataFrame | None) -> pd.DataFrame:
    if laya is None or laya.empty:
        return predictions
    left = predictions.drop(columns=[c for c in OUTPUT_COLUMNS if c in predictions.columns])
    right = laya[["cycle_uid", *OUTPUT_COLUMNS]].copy()
    right["cycle_uid"] = right["cycle_uid"].astype(str)
    left = left.copy()
    left["__uid"] = left["cycle_uid"].astype("string")
    out = left.merge(right.rename(columns={"cycle_uid": "__uid"}), on="__uid", how="left",
                     validate="one_to_one")
    return out.drop(columns="__uid")


def coverage(predictions: pd.DataFrame) -> dict:
    ok = predictions.get("laya_assessable")
    scored = int(pd.to_numeric(predictions.get("Laya"), errors="coerce").notna().sum()) \
        if "Laya" in predictions.columns else 0
    return {"scored": scored, "cohort": int(len(predictions)),
            "not_assessable": int((ok == False).sum()) if ok is not None else None,  # noqa: E712
            "disagreement_high": int((predictions.get("laya_disagreement_high") == True).sum())  # noqa: E712
            if "laya_disagreement_high" in predictions.columns else 0}
