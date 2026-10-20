"""persona_dt_calibration_export.py — объединённый файл калибровки DT 7.1 для PERSONA (все годы с прогоном).

Формат — шаблон «Калибровка переносов» программы IVF Digital Twin 7.1
(desktop/clinic_intake.TRANSFER_COLUMNS): программа клиники сама считает по нему прогнозы
своей моделью и подбирает сдвиг интерсепта. Экспорт прогнозов с квитанциями (FORECAST_COLUMNS)
здесь не подходит: квитанции привязаны к пользователю Windows и отпечатку модели этого ПК.

Строка = свежий перенос с известным исходом из когорты DT годовых прогонов.
Входы модели — ровно те, что ушли в DT 7.1 в прогоне (`local_private/dt71_api_output.csv`,
с импутациями, согласованными владельцем: AFC = ОКК × 1.16, АМГ/ИМТ по формулам пайплайна,
пустая попытка = 1, источник спермы выведен). Фолликулы — пусто (в выгрузке их нет).
Идентификаторы — псевдонимы: patient_id = HMAC пациентки (секрет пайплайна PERSONA),
cycle_id = cycle_uid; ФИО, даты рождения, номера карт в файл не попадают.

    python scripts/persona_dt_calibration_export.py --out "C:/Users/User/Desktop/IVF/PERSONA"
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from embryo import privacy  # noqa: E402
from embryo.clinics.persona import run_years  # noqa: E402

BASE = ROOT / "data" / "processed" / "persona"
YEARS = tuple(run_years(BASE))
SECRET = BASE / "local_private" / "cycle_uid.key"
COLUMNS = ['patient_id', 'cycle_id', 'transfer_id', 'transfer_month', 'oocyte_source', 'transfer_kind',
           'embryo_stage', 'embryos_transferred', 'pgt_a', 'age', 'amh', 'afc', 'bmi', 'attempt',
           'sperm_source', 'follicles', 'okk', 'mii', 'pn2', 'blasts_total', 'blasts_good', 'euploid',
           'outcome_clinical_pregnancy', 'outcome_status']


def _int(v):
    return "" if pd.isna(v) else str(int(round(float(v))))


def _dec(v, nd=2):
    return "" if pd.isna(v) else f"{float(v):.{nd}f}".rstrip("0").rstrip(".")


def stage(day, bl) -> tuple[str, str | None]:
    """Стадия переносимого эмбриона по дню переноса; без дня — по наличию бластоцист."""
    if pd.notna(day):
        if day >= 5:
            # день 5–7 без единой бластоцисты в цикле — переносили небластулировавший эмбрион
            return ("morula", "day5_without_blastocyst") if bl == 0 else ("blastocyst", None)
        return ("morula", None) if day == 4 else ("cleavage", None)
    return ("blastocyst", "stage_from_bl") if bl and bl > 0 else ("cleavage", "stage_from_bl")


def build_year(year: str, secret: bytes, notes: Counter) -> pd.DataFrame:
    mi = pd.read_excel(BASE / year / "local_private" / "model_input.xlsx")
    dt = pd.read_csv(BASE / year / "local_private" / "dt71_api_output.csv", low_memory=False)
    pred = pd.read_csv(BASE / year / "dt_predictions.csv", low_memory=False, usecols=["cycle_uid"])
    keep = ["cycle_uid", "age", "amh", "afc", "bmi", "attempt_number", "sperm_source",
            "known_okk", "known_mii", "known_pn2", "known_blasts", "known_good"]
    df = mi[mi["cycle_uid"].isin(pred["cycle_uid"])].merge(
        dt[keep].add_prefix("dt_").rename(columns={"dt_cycle_uid": "cycle_uid"}), on="cycle_uid", how="left",
        validate="one_to_one")
    rows = []
    for _, r in df.iterrows():
        pid = str(r["patient_id"]).strip().upper()
        has_dt = pd.notna(r["dt_age"])
        if not has_dt:
            notes["inputs_from_export_no_dt"] += 1   # служба DT отклонила цикл: сырые значения выгрузки
        g = lambda dtc, raw: r[dtc] if has_dt else r.get(raw)  # noqa: E731
        okk, mii, pn2 = g("dt_known_okk", "occ"), g("dt_known_mii", "mii"), g("dt_known_pn2", "two_pn")
        bl, good = g("dt_known_blasts", "bl_total"), g("dt_known_good", "bl_good")
        if pd.notna(mii) and pd.notna(okk) and mii > okk:
            okk = mii; notes["okk_raised_to_mii_thawed"] += 1
        if pd.notna(mii) and pd.notna(pn2) and pn2 > mii:
            mii = np.nan; notes["mii_blank_pn2_gt_mii_ivf"] += 1   # кЭКО: 2PN из неденудированных
        st, why = stage(pd.to_numeric(r["transfer_day"], errors="coerce"), bl)
        if why:
            notes[why] += 1
        month = pd.to_datetime(r["date_et"], errors="coerce")
        if pd.isna(month):
            month = pd.to_datetime(r["date_opu"], errors="coerce"); notes["month_from_opu"] += 1
        src = str(r["oocyte_source"]).strip().lower()
        sperm = g("dt_sperm_source", "sperm_source")
        rows.append({
            "patient_id": privacy.make_cycle_uid(pid.split("#")[0] + "#patient", secret).replace("CYC-", "PAT-"),
            "cycle_id": r["cycle_uid"], "transfer_id": f"{r['cycle_uid']}-ET",
            "transfer_month": month.strftime("%Y-%m"),
            "oocyte_source": "own" if src == "own" else "donor",
            "transfer_kind": "fresh", "embryo_stage": st,
            "embryos_transferred": _int(r["transferred"]), "pgt_a": "0",
            "age": _dec(g("dt_age", "age"), 1), "amh": _dec(g("dt_amh", "amh"), 2),
            "afc": _int(g("dt_afc", "afc_from_occ")), "bmi": _dec(g("dt_bmi", "bmi"), 1),
            "attempt": _int(g("dt_attempt_number", "attempt_no")),
            "sperm_source": "" if pd.isna(sperm) else str(sperm),
            "follicles": "", "okk": _int(okk), "mii": _int(mii), "pn2": _int(pn2),
            "blasts_total": _int(bl), "blasts_good": _int(good), "euploid": "",
            "outcome_clinical_pregnancy": _int(r["Preg"]), "outcome_status": "known",
            "_year": year, "_own_donor_mixed": src == "own+donor",
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    secret = privacy.load_or_create_secret(SECRET)
    notes: Counter = Counter()
    df = pd.concat([build_year(y, secret, notes) for y in YEARS], ignore_index=True)
    notes["own_plus_donor_as_donor"] = int(df["_own_donor_mixed"].sum())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"PERSONA_DT71_calibration_transfers_{YEARS[0]}_{YEARS[-1]}.csv"
    df[COLUMNS].to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")
    by_year = df.groupby("_year").agg(rows=("cycle_id", "size"),
                                      pregnancies=("outcome_clinical_pregnancy", lambda s: int(s.astype(int).sum())))
    print(f"[export] {path} rows={len(df)}")
    print(by_year.to_string())
    print("notes:", dict(notes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
