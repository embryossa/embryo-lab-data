"""dnn_backfill.py — досчитать DNN/EVO для уже завершённого прогона без повторного DT/TabPFN.

Берёт вход прогона (`local_private/model_input.xlsx`) и сохранённый выход TabPFN-3
regression, готовит вход DNN тем же `_dnn_input`, что и `run_analysis.py` (гейты
`external_tools.dnn_verifier` клиники), запускает EVO и дописывает колонки `DNN`,
`dnn_validation_status` в `dt_predictions.csv`; статус и покрытие — в `run_manifest.json`
(`dnn_backfill` — отметка, что DNN досчитана позже основного прогона). HTML-отчёт и
agent_packets не меняются.

    python scripts/dnn_backfill.py --persona 2024
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

from embryo import _io, dnn_runner, tabpfn_regression_runner  # noqa: E402

DNN_COLUMNS = ["DNN", "dnn_validation_status"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Досчитать DNN для завершённого прогона")
    parser.add_argument("--persona", metavar="YEAR", required=True)
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    cfg = ra._settings(ROOT / "config" / "clinics" / "persona")
    ra._persona_year_run(cfg, args.persona, None)
    processed = ra._project_path(cfg["paths"]["processed_dir"])
    private = ra._project_path(cfg["paths"]["local_private_dir"])
    dcfg = cfg["external_tools"]["dnn_verifier"]

    regression_path = private / "tabpfn_regression_full.csv"
    clean_regression = None
    if regression_path.is_file():
        clean_regression, _ = tabpfn_regression_runner.extract_predictions(
            pd.read_csv(regression_path, low_memory=False))
    dnn_input_path, imputed = ra._dnn_input(private / "model_input.xlsx", clean_regression, dcfg, private)
    dnn_frame = dnn_runner.run_dnn_verifier(dcfg, str(dnn_input_path))

    # Все колонки строками: dt_predictions.csv переписывается без смены форматов
    # (ведущие нули ID, представление чисел), меняются только колонки DNN.
    pred_path = processed / "dt_predictions.csv"
    predictions = pd.read_csv(pred_path, dtype=str, keep_default_na=False)
    predictions = predictions.drop(columns=[c for c in DNN_COLUMNS if c in predictions.columns])
    joined = ra._join_dnn(predictions, dnn_frame)
    joined["DNN"] = joined["DNN"].map(lambda v: "" if pd.isna(v) else repr(float(v)))
    joined["dnn_validation_status"] = joined["dnn_validation_status"].fillna("")
    _io.atomic_write_csv(joined, pred_path, index=False)

    scored = int(joined["DNN"].ne("").sum())
    manifest_path = processed / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dnn_status"] = "valid_saved_scaler"
    manifest["dnn_coverage"] = {"scored": scored, "cohort": int(len(joined)), "model_imputed": imputed}
    manifest["dnn_backfill"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _io.atomic_write_text(json.dumps(manifest, ensure_ascii=False, indent=2), manifest_path)
    print(f"[dnn-backfill] {args.persona}: DNN {scored}/{len(joined)}, импутации {imputed}")
    return 0 if scored else 1


if __name__ == "__main__":
    sys.exit(main())
