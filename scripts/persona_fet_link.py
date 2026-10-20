"""persona_fet_link.py — связь криопереносов PERSONA с пункциями (persona.link_fets).

Читает staged-файлы всех лет (scripts/persona_stage.py) и пишет:
  data/processed/persona/fet_links.csv                     только псевдонимы (fet_uid, opu_cycle_uid)
  data/processed/persona/local_private/fet_links_patients.csv  ключ пациентки/пункции (PHI) — для
                                                           локальных Teable и карточек
В консоль — только агрегаты.

    python scripts/persona_fet_link.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embryo import _io, privacy  # noqa: E402
from embryo.clinics import persona  # noqa: E402

CONFIG = ROOT / "config" / "clinics" / "persona"


def _path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def main() -> int:
    settings = yaml.safe_load((CONFIG / "settings.yaml").read_text(encoding="utf-8"))
    staged_dir = _path(settings["paths"]["staged_dir"])
    staged = pd.concat([pd.read_excel(p) for p in sorted(staged_dir.glob("persona_*.xlsx"))], ignore_index=True)
    secret = privacy.load_or_create_secret(_path(settings["privacy"]["secret_file"]))
    links = persona.link_fets(staged, secret)

    # Локальный слой: ключ пункции = _key прогона (ПАЦИЕНТКА#ГГГГММДД) — по нему Teable
    # связывает криоперенос с записью Cycles; ФИО — для поиска во вьюере.
    fet = staged[staged["treatment_type"].astype("string").str.contains("КРИО", na=False)
                 & ~staged["has_lab"].map(lambda v: str(v).strip().lower() in ("true", "1", "1.0"))]
    d = pd.to_datetime(fet["program_date"], errors="coerce").fillna(pd.to_datetime(fet["date_et"], errors="coerce"))
    fet = fet.assign(_uid=[privacy.make_cycle_uid(f"{str(p).strip().upper()}#{x:%Y%m%d}#fet", secret)
                           if pd.notna(x) and pd.notna(p) else None for p, x in zip(fet["patient_id"], d)])
    who = fet.dropna(subset=["_uid"]).drop_duplicates("_uid").set_index("_uid")
    private = links[["fet_uid", "opu_date"]].copy()
    private["patient_id"] = private["fet_uid"].map(who["patient_id"])
    private["patient_name"] = private["fet_uid"].map(who["patient_name"])
    opu_day = pd.to_datetime(private["opu_date"], errors="coerce").dt.strftime("%Y%m%d")
    private["opu_key"] = (private["patient_id"].astype("string").str.strip().str.upper() + "#" + opu_day).where(
        opu_day.notna())
    out = ROOT / "data" / "processed" / "persona" / "fet_links.csv"
    _io.atomic_write_csv(links, out, index=False)
    _io.atomic_write_csv(private.drop(columns="opu_date"),
                         _path(settings["paths"]["local_private_dir"]) / "fet_links_patients.csv", index=False)

    summary = {
        "fet_programs": int(len(links)),
        "link_status": links["link_status"].value_counts().to_dict(),
        "remaining_check_linked": links.loc[links["opu_cycle_uid"].notna(), "remaining_check"].value_counts().to_dict(),
        "consumed_from": links["consumed_from"].value_counts().to_dict(),
        "with_outcome": int(links["Preg"].notna().sum()),
        "pregnancies": int(pd.to_numeric(links["Preg"], errors="coerce").sum()),
        "linked_opus": int(links["opu_cycle_uid"].nunique()),
        "surrogate_transfers": int(links["surrogate"].sum()),
    }
    print(f"[fet-link] -> {out}")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
