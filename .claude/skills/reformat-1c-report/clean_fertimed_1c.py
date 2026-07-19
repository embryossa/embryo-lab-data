#!/usr/bin/env python3
"""Standalone, deterministic cleaner for Fertimed's raw 1C MIS export (.xlsx).

Self-contained on purpose: no imports from this project's `src/embryo` package,
so this single file (plus SKILL.md next to it) can be copied into another
project and still run unmodified. It is NOT part of the analytics pipeline —
`scripts/run_analysis.py` / `src/embryo/clinics/fertimed.py` never import this
file and are unaffected by anything here.

Rules below are owner-approved (2026-07-19 техзадание), applied on the
ORIGINAL column headers/order — no column renamed, added, or removed; only
row filtering and in-place cell fills/recodes on existing columns.

Row filter — a row is kept only if, applied in this order:
  1. "Исход переноса" is non-blank.
  2. "Число инсеминированных" is present and numeric (else the row is
     dropped immediately — this is the "only rows where fertilization was
     actually attempted" gate).
  3. "Лечение" (case-insensitive, trimmed) is one of: криоперенос, икси,
     эко, эко + икси, получение як (реципиент). Blank, "Только пункция",
     "Инсеминация", "ПГТ на размороженных эмбрионах" are dropped. "Лечение"
     is NOT used for anything else (it is known to be an unreliable marker
     of the actual fertilization method — see IVF/ICSI flags below).

Cell fills / recodes on the surviving cohort (only real gaps are filled,
except where noted as a full column recompute). Fill order matters here —
MII is filled first, then OCC falls back to (now-complete) MII, then
follicles falls back to (now-complete) OCC, which guarantees zero blanks
left in all three columns (the cohort filter already guarantees "Число
инсеминированных" is present on every row, so the chain always has a root):
  - "Число MII"             blank -> "Число инсеминированных" (same rule as
    the main pipeline: use MII if actually known, else Inseminated — see
    CLAUDE.md). Applied unconditionally to every row now, not just pure IVF.
  - "Число ОКК"             blank -> "Число MII" (post-fill).
  - "Количество фолликулов" blank -> "Число ОКК" (post-fill). "КАФ" is never
    used — it is known to be entered unreliably.
  - "№ попытки"             blank -> 1
  - "День переноса"         blank -> 5
  - "АМГ"                   blank -> imputed via the same fixed formula as
    `normalize.impute_amh` (age/AFC/OCC based, clipped to [floor, ceil])
  - "ИМТ" / "BMI"           blank -> fixed constant (only if such a column
    exists in the export; the current Fertimed format has none, no-op)
  - "Эмбриолог, проводивший оплодотворение"  blank -> "Эмбриолог на
    пункции", and if that is also blank -> "Эмбриолог на переносе" (this
    priority chain is the KPI-relevant embryologist; the direct column is
    frequently lost on export).
  - "Врач на пункции"       blank -> "Врач на переносе" (this is the
    KPI-relevant/primary doctor).
  - "Заморожено эмбрионов"  FULL RECOMPUTE (not just gap-fill) ->
    "Заморожено эмбрионов" + "Заморожено бластоцист", overwriting the
    existing value. "Заморожено бластоцист" itself is left untouched.
  - "Возраст"                FULL RECOMPUTE -> parsed to a plain number
    ("34 года"/"26 лет"/"31 год" -> 34/26/31). Cells that don't parse are
    left blank (not fabricated) in the output column; a separate internal
    parse with a neutral fallback feeds the AMH formula only (see below).
  - "Исход переноса"         FULL RECOMPUTE -> binary code: 1 if the cell
    text contains "+28 дней" anywhere (a confirmed clinical pregnancy —
    singleton or multiple, doesn't matter), else 0 (biochemical pregnancy
    or no pregnancy). The cohort filter above already guarantees this cell
    is non-blank for every surviving row.

Columns the owner has said are simply not used for this report (donor
codes, surrogate PHI, birth/complication outcomes, SMF, FET, PGT-TVP/ET,
КАФ, etc.) are intentionally left completely untouched in the output — kept
in the file, at their original values and position, just never read by any
rule above.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- established Fertimed rules (owner-approved 2026-07-19 техзадание; keep
# in sync with config/clinics/fertimed/columns.yaml + src/embryo/clinics/
# fertimed.py if those rules ever change, but this file must stay
# import-free from them) ---

COL_TREATMENT = "Лечение"
COL_OUTCOME = "Исход переноса"
COL_INSEMINATED = "Число инсеминированных"
COL_ATTEMPT_NO = "№ попытки"
COL_TRANSFER_DAY = "День переноса"
COL_FOLLICLES = "Количество фолликулов"
COL_OCC = "Число ОКК"
COL_AGE = "Возраст"
COL_AMH = "АМГ"
COL_BMI_CANDIDATES = ("ИМТ", "BMI")
COL_MII = "Число MII"
COL_EMBRYOLOGIST_FERT = "Эмбриолог, проводивший оплодотворение"
COL_EMBRYOLOGIST_PUNCTURE = "Эмбриолог на пункции"
COL_EMBRYOLOGIST_TRANSFER = "Эмбриолог на переносе"
COL_DOCTOR_PUNCTURE = "Врач на пункции"
COL_DOCTOR_TRANSFER = "Врач на переносе"
COL_FROZEN_EMBRYOS = "Заморожено эмбрионов"
COL_FROZEN_BLASTS = "Заморожено бластоцист"

ALLOWED_TREATMENT = {
    "криоперенос", "икси", "эко", "эко + икси", "получение як (реципиент)",
}

PREGNANCY_MARKER = "+28"

# Same fixed parameters as config/clinics/fertimed/settings.yaml's amh_impute/
# bmi_impute blocks (owner-approved, "правила не менять" per CLAUDE.md).
AMH_A_AFC = 0.16
AMH_B_OCC = 0.05
AMH_C_AGE = 0.08
AMH_INTERCEPT = 0.5
AMH_FLOOR = 0.1
AMH_CEIL = 12.0
BMI_CONSTANT = 24.0

_AGE_RE = re.compile(r"(\d+)")

# Required for the row filter / fill rules to even make sense; if any of
# these is missing the export isn't the format this script was built for.
REQUIRED_COLUMNS = [
    COL_TREATMENT, COL_OUTCOME, COL_INSEMINATED, COL_ATTEMPT_NO,
    COL_TRANSFER_DAY, COL_FOLLICLES, COL_OCC, COL_AGE, COL_MII,
    COL_EMBRYOLOGIST_FERT,
    COL_EMBRYOLOGIST_PUNCTURE, COL_EMBRYOLOGIST_TRANSFER,
    COL_DOCTOR_PUNCTURE, COL_DOCTOR_TRANSFER,
    COL_FROZEN_EMBRYOS, COL_FROZEN_BLASTS,
]


def _is_blank(series: pd.Series) -> pd.Series:
    return series.isna() | (series.astype(str).str.strip() == "")


def _parse_age_for_formula(value) -> float:
    """"34 года" -> 34.0, used ONLY as an input to the AMH formula below —
    never written back to the output table as-is (see _parse_age_to_number
    for the actual output column value, which does not fabricate a default)."""
    if pd.isna(value):
        return 35.0  # same neutral default as normalize.impute_amh's row.get() fallback
    m = _AGE_RE.search(str(value))
    return float(m.group(1)) if m else 35.0


def _parse_age_to_number(value):
    """"34 года"/"26 лет"/"31 год" -> 34/26/31 for the real output column.
    Unparseable/blank cells stay blank (NaN) — no fabricated default here,
    unlike the AMH-formula-only helper above."""
    if pd.isna(value):
        return np.nan
    m = _AGE_RE.search(str(value))
    return float(m.group(1)) if m else np.nan


def _cohort_mask(df: pd.DataFrame) -> tuple[pd.Series, dict]:
    outcome_present = ~_is_blank(df[COL_OUTCOME])
    inseminated_present = pd.to_numeric(df[COL_INSEMINATED], errors="coerce").notna()
    treatment_ok = df[COL_TREATMENT].astype(str).str.strip().str.lower().isin(ALLOWED_TREATMENT)

    # Step-wise breakdown, applied in the owner-specified order, purely for
    # diagnostics — the final mask is the same AND regardless of order.
    step1 = outcome_present
    step2 = step1 & inseminated_present
    step3 = step2 & treatment_ok
    breakdown = {
        "dropped_missing_outcome": int((~step1).sum()),
        "dropped_missing_inseminated": int((step1 & ~inseminated_present).sum()),
        "dropped_bad_treatment": int((step2 & ~treatment_ok).sum()),
    }
    return step3, breakdown


def _fill_attempt_no(df: pd.DataFrame) -> None:
    df[COL_ATTEMPT_NO] = pd.to_numeric(df[COL_ATTEMPT_NO], errors="coerce").fillna(1)


def _fill_transfer_day(df: pd.DataFrame) -> None:
    df[COL_TRANSFER_DAY] = pd.to_numeric(df[COL_TRANSFER_DAY], errors="coerce").fillna(5)


def _fill_mii_from_inseminated(df: pd.DataFrame) -> None:
    mii = pd.to_numeric(df[COL_MII], errors="coerce")
    inseminated = pd.to_numeric(df[COL_INSEMINATED], errors="coerce")
    df[COL_MII] = mii.where(mii.notna(), inseminated)


def _fill_occ_from_mii(df: pd.DataFrame) -> None:
    occ = pd.to_numeric(df[COL_OCC], errors="coerce")
    mii = pd.to_numeric(df[COL_MII], errors="coerce")
    df[COL_OCC] = occ.where(occ.notna(), mii)


def _fill_follicles_from_occ(df: pd.DataFrame) -> None:
    follicles = pd.to_numeric(df[COL_FOLLICLES], errors="coerce")
    occ = pd.to_numeric(df[COL_OCC], errors="coerce")
    df[COL_FOLLICLES] = follicles.where(follicles.notna(), occ)


def _fill_amh(df: pd.DataFrame) -> None:
    if COL_AMH not in df.columns:
        return
    amh = pd.to_numeric(df[COL_AMH], errors="coerce")
    mask = amh.isna()
    if not mask.any():
        return
    age_for_formula = df.loc[mask, COL_AGE].map(_parse_age_for_formula)
    occ = pd.to_numeric(df.loc[mask, COL_OCC], errors="coerce").fillna(0)
    afc = pd.to_numeric(df.loc[mask, COL_FOLLICLES], errors="coerce").fillna(0)
    estimate = (AMH_A_AFC * afc + AMH_B_OCC * occ
                - AMH_C_AGE * (age_for_formula - 35).clip(lower=0) + AMH_INTERCEPT)
    amh.loc[mask] = estimate.clip(lower=AMH_FLOOR, upper=AMH_CEIL)
    df[COL_AMH] = amh


def _fill_bmi(df: pd.DataFrame) -> None:
    bmi_col = next((c for c in COL_BMI_CANDIDATES if c in df.columns), None)
    if bmi_col is None:
        return
    bmi = pd.to_numeric(df[bmi_col], errors="coerce")
    df[bmi_col] = bmi.fillna(BMI_CONSTANT)


def _fill_embryologist_fertilization(df: pd.DataFrame) -> None:
    filled = df[COL_EMBRYOLOGIST_FERT].copy()
    blank = _is_blank(filled)
    filled.loc[blank] = df.loc[blank, COL_EMBRYOLOGIST_PUNCTURE]
    still_blank = _is_blank(filled)
    filled.loc[still_blank] = df.loc[still_blank, COL_EMBRYOLOGIST_TRANSFER]
    df[COL_EMBRYOLOGIST_FERT] = filled


def _fill_doctor_puncture(df: pd.DataFrame) -> None:
    blank = _is_blank(df[COL_DOCTOR_PUNCTURE])
    df.loc[blank, COL_DOCTOR_PUNCTURE] = df.loc[blank, COL_DOCTOR_TRANSFER]


def _sum_frozen_embryos(df: pd.DataFrame) -> None:
    frozen = pd.to_numeric(df[COL_FROZEN_EMBRYOS], errors="coerce").fillna(0)
    blasts = pd.to_numeric(df[COL_FROZEN_BLASTS], errors="coerce").fillna(0)
    df[COL_FROZEN_EMBRYOS] = frozen + blasts


def _overwrite_age_numeric(df: pd.DataFrame) -> None:
    df[COL_AGE] = df[COL_AGE].map(_parse_age_to_number)


def _encode_outcome(df: pd.DataFrame) -> None:
    text = df[COL_OUTCOME].astype(str)
    df[COL_OUTCOME] = text.str.contains(PREGNANCY_MARKER, regex=False).astype(int)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"во входной таблице нет ожидаемых колонок Fertimed 1C: {missing}")

    n_before = len(df)
    mask, cohort_breakdown = _cohort_mask(df)
    kept = df.loc[mask].reset_index(drop=True).copy()
    n_after = len(kept)

    n_attempt_filled = int(pd.to_numeric(kept[COL_ATTEMPT_NO], errors="coerce").isna().sum())
    n_transfer_day_filled = int(pd.to_numeric(kept[COL_TRANSFER_DAY], errors="coerce").isna().sum())
    n_mii_filled = int(pd.to_numeric(kept[COL_MII], errors="coerce").isna().sum())
    n_occ_filled = int(pd.to_numeric(kept[COL_OCC], errors="coerce").isna().sum())
    n_follicles_filled = int(pd.to_numeric(kept[COL_FOLLICLES], errors="coerce").isna().sum())
    n_amh_filled = int(pd.to_numeric(kept.get(COL_AMH), errors="coerce").isna().sum()) if COL_AMH in kept else 0
    bmi_col = next((c for c in COL_BMI_CANDIDATES if c in kept.columns), None)
    n_bmi_filled = int(pd.to_numeric(kept[bmi_col], errors="coerce").isna().sum()) if bmi_col else 0
    n_embryologist_filled = int(_is_blank(kept[COL_EMBRYOLOGIST_FERT]).sum())
    n_doctor_filled = int(_is_blank(kept[COL_DOCTOR_PUNCTURE]).sum())
    n_age_unparsed = int(kept[COL_AGE].map(_parse_age_to_number).isna().sum())

    _fill_attempt_no(kept)
    _fill_transfer_day(kept)
    _fill_mii_from_inseminated(kept)
    _fill_occ_from_mii(kept)
    _fill_follicles_from_occ(kept)
    _fill_amh(kept)
    _fill_bmi(kept)
    _fill_embryologist_fertilization(kept)
    _fill_doctor_puncture(kept)
    _sum_frozen_embryos(kept)
    _overwrite_age_numeric(kept)

    n_pregnant = int(kept[COL_OUTCOME].astype(str).str.contains(PREGNANCY_MARKER, regex=False).sum())
    _encode_outcome(kept)

    stats = {
        "rows_in": n_before,
        "rows_out": n_after,
        "rows_dropped_by_cohort_filter": n_before - n_after,
        **cohort_breakdown,
        "attempt_no_filled": n_attempt_filled,
        "transfer_day_filled": n_transfer_day_filled,
        "mii_filled": n_mii_filled,
        "occ_filled": n_occ_filled,
        "follicles_filled": n_follicles_filled,
        "amh_imputed": n_amh_filled,
        "bmi_imputed": n_bmi_filled,
        "embryologist_fert_filled": n_embryologist_filled,
        "doctor_puncture_filled": n_doctor_filled,
        "age_unparsed_left_blank": n_age_unparsed,
        "outcome_pregnant": n_pregnant,
        "outcome_not_pregnant": n_after - n_pregnant,
    }
    return kept, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="путь к исходному .xlsx выгрузки 1С Fertimed")
    parser.add_argument("--output", help="путь к результату (по умолчанию "
                         "'<имя_входа>_cleaned.xlsx' рядом со входным файлом)")
    parser.add_argument("--sheet", default=0, help="имя/индекс листа (по умолчанию первый)")
    args = parser.parse_args(argv)

    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        print(f"[error] входной файл не найден: {input_path}", file=sys.stderr)
        return 1

    output_path = (Path(args.output).resolve() if args.output
                   else input_path.with_name(f"{input_path.stem}_cleaned.xlsx"))

    df = pd.read_excel(input_path, sheet_name=args.sheet)
    try:
        cleaned, stats = clean(df)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_excel(output_path, index=False)

    print(f"[ok] строк было: {stats['rows_in']}, осталось: {stats['rows_out']} "
          f"(отфильтровано: исход пуст={stats['dropped_missing_outcome']}, "
          f"инсеминированных пусто={stats['dropped_missing_inseminated']}, "
          f"лечение вне когорты={stats['dropped_bad_treatment']})", file=sys.stderr)
    print(f"[ok] заполнено пропусков — № попытки: {stats['attempt_no_filled']}, "
          f"День переноса: {stats['transfer_day_filled']}, "
          f"Число MII (из Число инсеминированных): {stats['mii_filled']}, "
          f"Число ОКК (из Число MII): {stats['occ_filled']}, "
          f"Количество фолликулов (из Число ОКК): {stats['follicles_filled']}, "
          f"АМГ: {stats['amh_imputed']}, ИМТ/BMI: {stats['bmi_imputed']}, "
          f"Эмбриолог(оплодотворение): {stats['embryologist_fert_filled']}, "
          f"Врач на пункции: {stats['doctor_puncture_filled']}", file=sys.stderr)
    print(f"[ok] Возраст перезаписан числом (не распарсено и оставлено пустым: "
          f"{stats['age_unparsed_left_blank']}); Заморожено эмбрионов = "
          f"Заморожено эмбрионов + Заморожено бластоцист (перезаписано для всех строк); "
          f"Исход переноса перекодирован в 0/1 (беременность=1: {stats['outcome_pregnant']}, "
          f"0: {stats['outcome_not_pregnant']})", file=sys.stderr)
    # Final stdout line is exactly the absolute output path, nothing else —
    # the calling agent relays this line back to the user as-is.
    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
