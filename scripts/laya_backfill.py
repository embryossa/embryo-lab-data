"""laya_backfill.py — пересчитать Laya для уже завершённого прогона без повторного DT/TabPFN.

Нужен, когда у центра появилась калибровка Laya после прогона (PERSONA, 2026-10-20):
тот же `laya_runner.run_laya` с текущими `external_tools.laya` клиники, результат
заменяет колонки Laya в `dt_predictions.csv`; статус и покрытие — в `run_manifest.json`
(`laya_backfill` — отметка пересчёта). HTML-отчёт и agent_packets не меняются.

    python scripts/laya_backfill.py --persona 2024
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_analysis", ROOT / "scripts" / "run_analysis.py")
ra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ra)

from embryo import _io, laya_runner  # noqa: E402


def _cell(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)) or v is pd.NA:
        return ""
    if isinstance(v, bool):
        return str(v)
    return repr(float(v)) if isinstance(v, (int, float)) else str(v)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Пересчитать Laya для завершённого прогона")
    parser.add_argument("--persona", metavar="YEAR", required=True)
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    cfg = ra._settings(ROOT / "config" / "clinics" / "persona")
    ra._persona_year_run(cfg, args.persona, None)
    processed = ra._project_path(cfg["paths"]["processed_dir"])
    private = ra._project_path(cfg["paths"]["local_private_dir"])
    lcfg = cfg["external_tools"]["laya"]

    pred_path = processed / "dt_predictions.csv"
    typed = pd.read_csv(pred_path, low_memory=False)
    laya = laya_runner.run_laya(lcfg, typed, private)
    # Остальные колонки — строками, без смены форматов (как dnn_backfill.py).
    predictions = pd.read_csv(pred_path, dtype=str, keep_default_na=False)
    joined = laya_runner.join_laya(predictions, laya)
    for col in laya_runner.OUTPUT_COLUMNS:
        joined[col] = joined[col].map(_cell)
    _io.atomic_write_csv(joined, pred_path, index=False)

    cov = laya_runner.coverage(laya_runner.join_laya(typed, laya))
    manifest_path = processed / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["laya_status"] = "ok" if lcfg.get("centre") else "ok_uncalibrated"
    manifest["laya_coverage"] = cov
    manifest["laya_centre"] = lcfg.get("centre") or None
    manifest["laya_backfill"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _io.atomic_write_text(json.dumps(manifest, ensure_ascii=False, indent=2), manifest_path)
    print(f"[laya-backfill] {args.persona}: {cov}")
    return 0 if cov.get("scored_centre_calibrated") else 1


if __name__ == "__main__":
    sys.exit(main())
