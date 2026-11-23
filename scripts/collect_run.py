"""Собирает выходы одного прогона клиники в отдельную папку, чтобы не путать их
с другими прогонами и клиниками.

    python scripts/collect_run.py                 # клиника A (Two-Line) -> data/processed/Two-Line/<дата>_<run_id>/

Файлы КОПИРУЮТСЯ: пайплайн, replay и teable_sync продолжают читать data/processed/.
local_private (ID/ФИО, PDF ПГТ) не копируется. Кросс-прогонные файлы (RUN_JOURNAL.md,
run_history.jsonl) остаются на месте.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

# Выходы клиники A в корне data/processed.
TOP_FILES = [
    "DT_Analytics_Report.html", "DT_Analytics_Report.xlsx", "OPU_table_enriched.xlsx",
    "dt_predictions.csv", "tabpfn_predictions.csv", "evo_predictions.csv",
    "review_list.csv", "kpi_computed.csv", "clinical_context.csv",
    "miri_events.csv", "miri_summary.csv", "miri_baseline.json",
    "lot_events.csv", "lot_summary.csv", "lot_baseline.json",
    "pgt_summary.csv", "pgt_embryo_events.csv",
    "ozema_cycles.csv", "ozema_genetic_counseling_candidates.csv",
    "regression_qc.csv", "regression_calibration.csv", "run_manifest.json",
]
DIRS = ["agent_packets", "off_cohort"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="Two-Line", help="папка клиники внутри data/processed")
    ap.add_argument("--logs", nargs="*", default=None,
                    help="логи прогона (по умолчанию logs/*<YYYYMMDD>*.log)")
    args = ap.parse_args()

    manifest = json.loads((PROCESSED / "run_manifest.json").read_text(encoding="utf-8"))
    run_id = manifest["run_id"]
    day = run_id[:8]
    out = PROCESSED / args.dest / f"{day[:4]}-{day[4:6]}-{day[6:]}_{run_id}"
    out.mkdir(parents=True, exist_ok=True)

    copied, missing = [], []
    for name in TOP_FILES:
        src = PROCESSED / name
        if src.exists():
            shutil.copy2(src, out / name)
            copied.append(name)
        else:
            missing.append(name)
    for d in DIRS:
        src = PROCESSED / d
        if src.is_dir():
            shutil.copytree(src, out / d, dirs_exist_ok=True)
            copied.append(d + "/")
    viewer = PROCESSED / "_viewer" / "two_line.html"
    if viewer.exists():
        shutil.copy2(viewer, out / "cards_two_line.html")
        copied.append("cards_two_line.html")
    logs = [Path(p) for p in args.logs] if args.logs is not None else sorted((ROOT / "logs").glob(f"*{day}*"))
    if logs:
        (out / "logs").mkdir(exist_ok=True)
        for p in logs:
            p = p if p.is_absolute() else ROOT / p
            shutil.copy2(p, out / "logs" / p.name)
        copied.append(f"logs/ ({len(logs)})")

    print(f"[collect] {out}")
    print(f"[collect] скопировано: {', '.join(copied)}")
    if missing:
        print(f"[collect] нет в этом прогоне: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
