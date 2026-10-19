"""PERSONA: сырые годовые выгрузки -> staged-файлы по годам + обезличенный вход демо.

    python scripts/persona_stage.py                       # все *.xlsx из paths.inbox_dir
    python scripts/persona_stage.py --years 2024 2025     # только эти годы (по "Дата Программы")

Пишет (правила — docs/PERSONA.md):
  <staged_dir>/persona_<год>.xlsx      вход scripts/run_analysis.py --clinic persona (PHI, local_private)
  <demo_export_dir>/persona_<год>.csv  вход ivf-lab-kpi-ews-demo (без ФИО/даты рождения)
В консоль — только агрегаты: число программ, блоков, сверка ∑, покрытие исхода и AFC.
Ничего не запускает дальше: пайплайны стартуют отдельно, после подтверждения владельца.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embryo import _io, ingest, privacy  # noqa: E402
from embryo.clinics import persona  # noqa: E402

CONFIG = ROOT / "config" / "clinics" / "persona"


def _path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def _summary(year_frame: pd.DataFrame, columns_cfg: dict) -> dict:
    programs = persona.program_frame(year_frame)
    parts = persona.lab_part_frame(year_frame, columns_cfg)
    # Сверка ∑ = кЭКО + И + DO по программам, где обе стороны известны.
    checks = {}
    for field in ("occ", "two_pn", "bl_total", "bl_good"):
        total = pd.to_numeric(programs[f"total_{field}"], errors="coerce")
        known = total.notna() & programs[field].notna()
        checks[field] = f"{int((total[known] == programs.loc[known, field]).sum())}/{int(known.sum())}"
    orphan_total = pd.to_numeric(year_frame["total_occ"], errors="coerce").gt(0) & ~year_frame["has_lab"].astype(bool)
    transferred = pd.to_numeric(programs["transferred"], errors="coerce").gt(0)
    eligible = transferred & programs["Preg"].notna()
    return {
        "programs": int(len(year_frame)),
        "by_category": year_frame["treatment_type"].value_counts().head(12).to_dict(),
        "programs_with_lab_block": int(len(programs)),
        "lab_rows_by_part": parts["lab_part"].value_counts().to_dict(),
        "sum_check_equal/known": checks,
        "total_occ>0_without_lab_block": int(orphan_total.sum()),
        "fresh_transfer_programs": int(transferred.sum()),
        "dt_eligible_(transfer+outcome)": int(eligible.sum()),
        "Preg_among_eligible": programs.loc[eligible, "Preg"].value_counts().to_dict(),
        "afc_from_occ_among_eligible": int(programs.loc[eligible, "afc_from_occ"].notna().sum()),
        "amh_observed_among_eligible": int(programs.loc[eligible, "amh"].notna().sum()),
        "bmi_observed_among_eligible": int(programs.loc[eligible, "bmi"].notna().sum()),
        "donor_oocytes_among_eligible": int(programs.loc[eligible, "oocyte_source"].isin(["donor", "own+donor"]).sum()),
        "fet_programs_with_outcome": int((year_frame["treatment_type"].fillna("").str.startswith("КРИО")
                                          & year_frame["Preg"].notna()).sum()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inbox", help="папка с годовыми выгрузками (по умолчанию paths.inbox_dir)")
    parser.add_argument("--years", nargs="*", type=int, help="какие годы писать (по умолчанию все)")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    with (CONFIG / "settings.yaml").open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    columns_cfg = ingest.load_config(str(CONFIG / "columns.yaml"))
    inbox = _path(args.inbox or cfg["paths"]["inbox_dir"])
    files = sorted(p for p in inbox.glob("*.xlsx") if not p.name.startswith("~$"))
    if not files:
        raise SystemExit(f"нет выгрузок в {inbox}")

    staged, report = persona.stage_exports(files, columns_cfg, cfg)
    print(f"[persona] книг: {len(files)}; {json.dumps(report, ensure_ascii=False)}")
    secret = privacy.load_or_create_secret(_path(cfg["privacy"]["secret_file"]))
    staged_dir, demo_dir = _path(cfg["paths"]["staged_dir"]), _path(cfg["paths"]["demo_export_dir"])
    staged_dir.mkdir(parents=True, exist_ok=True)
    demo_dir.mkdir(parents=True, exist_ok=True)

    years = sorted(int(y) for y in staged["year"].dropna().unique())
    for year in (args.years or years):
        part = staged.loc[staged["year"].eq(year)].reset_index(drop=True)
        if part.empty:
            print(f"[persona] {year}: программ нет")
            continue
        out = _io.atomic_write_excel({"staged": part}, staged_dir / f"persona_{year}.xlsx")
        demo = persona.demo_export(part, secret, columns_cfg)
        demo_out = _io.atomic_write_csv(demo, demo_dir / f"persona_{year}.csv")
        print(f"[persona] {year}: {json.dumps(_summary(part, columns_cfg), ensure_ascii=False)}")
        print(f"[persona] {year}: staged -> {out}; demo -> {demo_out} ({len(demo)} строк)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
