"""
skyfert_kpi_test.py — standalone KPI/data-only test run for the Skyfert clinic
export. Separate from scripts/run_analysis.py's --clinic dispatch on purpose
(owner request, 2026-07-21): no shared code with src/embryo/clinics/fertimed.py,
no external DT/TabPFN/DNN launch, no HTML report. Just: load the raw file,
resolve/normalize it through src/embryo/clinics/skyfert.py, compute the
laboratory Vienna-style KPI (kpi.py, generic/shared) and OZEMA (ozema.py,
generic/shared), and write local CSVs to eyeball before deciding whether to
wire Skyfert into the full pipeline.

Simplified variant confirmed by the owner: no MIRI, no consumable lots, no
PGT-A, no embryologist/doctor competency split (Skyfert's export has none of
those columns). See docs/SKYFERT.md.

Usage:
    python scripts/skyfert_kpi_test.py "C:\\path\\to\\SkyFert.xlsx"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embryo import ingest, kpi, ozema, privacy, _io  # noqa: E402
from embryo.clinics import skyfert  # noqa: E402

DEFAULT_INPUT = r"C:\Users\User\Desktop\IVF\SkyFert\SkyFert.xlsx"
CONFIG_DIR = ROOT / "config" / "clinics" / "skyfert"


def _settings() -> dict:
    import yaml
    with open(CONFIG_DIR / "settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _project_path(cfg: dict, key: str) -> Path:
    return ROOT / cfg["paths"][key]


def _eligible_cycles(normalized: pd.DataFrame) -> pd.DataFrame:
    """Та же граница когорты, что scripts/run_analysis.py._eligible_cycles
    (transferred>0 & известный исход) -- переиспользована как логика (3
    строки), не импортом run_analysis.py, чтобы этот скрипт оставался
    полностью самостоятельным."""
    transferred = pd.to_numeric(normalized["transferred"], errors="coerce")
    mask = transferred.gt(0) & normalized["Preg"].notna()
    selected = normalized.loc[mask].copy()
    if selected.empty:
        raise ValueError("нет строк одновременно с переносом >0 и заполненным исходом")
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Skyfert -- отдельный KPI/данные тест")
    parser.add_argument("input", nargs="?", default=DEFAULT_INPUT)
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    print(f"[input] {input_path}")

    cfg = _settings()
    columns_cfg = ingest.load_config(str(CONFIG_DIR / "columns.yaml"))
    processed = _project_path(cfg, "processed_dir")
    private = _project_path(cfg, "local_private_dir")
    processed.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    raw = ingest.load_table(str(input_path))
    print(f"[raw] {len(raw)} строк, {len(raw.columns)} колонок")

    normalized, unknown = skyfert.prepare(raw, columns_cfg, cfg)
    if unknown:
        print(f"[warn] неизвестные колонки (пропущены при резолве): {unknown}")
    print(f"[normalized] {len(normalized)} строк после резолва/нормализации")

    secret = privacy.load_or_create_secret(ROOT / cfg["privacy"]["secret_file"])
    lab_cycles = normalized.copy()
    lab_cycles["_key"] = skyfert.build_cycle_key(lab_cycles)
    lab_cycles = privacy.add_cycle_uid(lab_cycles, "_key", secret)
    print(f"[cycle_uid] {lab_cycles['cycle_uid'].nunique()} уникальных cycle_uid "
          f"(из {len(lab_cycles)} строк) -- коллизий нет")

    crosswalk = privacy.build_crosswalk(lab_cycles, "patient_id")
    _io.atomic_write_csv(crosswalk, private / "cycle_uid_crosswalk.csv")
    _io.atomic_write_csv(
        lab_cycles.drop(columns=[c for c in lab_cycles.columns if c.startswith("_")]),
        private / "skyfert_lab_population.csv",
    )

    eligible = _eligible_cycles(lab_cycles)
    print(f"[eligible] {len(eligible)} строк с переносом>0 и известным исходом "
          f"(из {len(lab_cycles)} широкой lab-популяции)")

    cycle = kpi.compute_kpi(eligible)
    cycle_out = cycle.drop(columns=[c for c in cycle.columns if c.startswith("_")])
    _io.atomic_write_csv(cycle_out, private / "skyfert_kpi_cycle.csv")

    slices = kpi.aggregate_kpi_slices(
        cycle,
        method_col="fertilization_method" if "fertilization_method" in cycle.columns else None,
    )
    for name, frame in slices.items():
        _io.atomic_write_csv(frame, processed / f"kpi_{name}.csv")
    print(f"[kpi] срезы: {list(slices)} -> data/processed/skyfert/kpi_*.csv")

    ozema_cycles = ozema.build_ozema_table(lab_cycles, columns_cfg)
    ozema_candidates = ozema.genetic_counseling_candidates(ozema_cycles)
    _io.atomic_write_csv(
        ozema_cycles.drop(columns=[c for c in ("patient_id", "patient_name") if c in ozema_cycles.columns]),
        processed / "ozema_cycles.csv",
    )
    _io.atomic_write_csv(ozema_candidates, processed / "ozema_genetic_counseling_candidates.csv")
    n_flagged = int(ozema_cycles["flagged"].sum())
    print(f"[ozema] {n_flagged} циклов с флагом из {len(ozema_cycles)} "
          f"(OMA всегда 'н/д' -- в выгрузке Skyfert нет MII)")

    print("\n[overall KPI]")
    overall = slices.get("overall", pd.DataFrame())
    if not overall.empty:
        print(overall[["metric", "rate", "ci_low_95", "ci_high_95", "n_cycles", "status"]]
              .to_string(index=False))

    # Владелец подтвердил (2026-07-21): считать как есть, не подменять формулу.
    # "Число дробящихся на 3 день" в выгрузке Skyfert превышает "2 pN" на 57% строк
    # (см. docs/SKYFERT.md) -- похоже на то, что МИС считает ВСЕ дробящиеся ооциты
    # (в т.ч. аномальное 1PN/3PN оплодотворение), а не только вышедшие из 2PN-чека.
    # ClR/BlR/TGBDR используют "2 pN" знаменателем по формуле Vienna (kpi.py, общий
    # для всех клиник) -> для Skyfert они системно завышены. Это проблема выгрузки
    # МИС клиники, не баг этого пайплайна -- повод для разговора с их разработчиками.
    two_pn_num = pd.to_numeric(cycle.get("two_pn"), errors="coerce")
    cleav_num = pd.to_numeric(cycle.get("cleavage_d3"), errors="coerce")
    valid_pair = two_pn_num.notna() & cleav_num.notna()
    n_exceed = int((cleav_num.loc[valid_pair] > two_pn_num.loc[valid_pair]).sum())
    if n_exceed:
        print(f"\n[data quality] дробящихся>2pN на {n_exceed}/{int(valid_pair.sum())} циклов "
              f"eligible-когорты -> ClR/BlR/TGBDR для Skyfert завышены относительно Vienna-"
              f"методологии (2 pN как знаменатель). Считается как есть по решению владельца "
              f"(2026-07-21), не корректируется этим скриптом. См. docs/SKYFERT.md.")

    print(f"\nГотово. Результаты: {processed} (агрегаты) и {private} (по циклам, локально).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
