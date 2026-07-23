"""
ggrc.py — ingest/normalize adapter for the GGRC (Armenia) clinic export.

Rules confirmed with the owner 2026-07-23 (rationale — docs/GGRC.md). Unlike
every other clinic in this project, GGRC hands us the "eligible cohort" and
the "wide lab population" as TWO SEPARATE files instead of one file sliced by
a transferred>0/Preg-known filter:

  - "ivf_master_13 (transfer).xlsx" — owner-curated cycles with a transfer and
    a known outcome. Resolved by :func:`prepare`, used for the full DT/TabPFN/
    DNN pipeline (mirrors every other clinic's ``kpi_cycles``).
  - "ivf_master_13 kpi.xlsx" (sheet "IVF Master DB") — wider lab-stage
    population (includes freeze-all/no-transfer cycles). Resolved by
    :func:`prepare_wide`, used only for KPI aggregation (embryologist/doctor
    competency tables) and OZEMA (mirrors every other clinic's ``lab_cycles``).
    Carries no outcome column at all -- fine, KPI_SPECS/OZEMA never read
    "Preg".

Both files share the same "OPU Code" + "OPU Date" for a given physical cycle
(verified against the real export: all 29 transfer-file OPU Codes exist in
the kpi-wide file with matching OPU Date), so the GENERIC ``_key`` keying in
``scripts/run_analysis.py::_add_opu_aliases`` (patient_id + date_opu) already
makes cycle_uid consistent between the two frames -- no Skyfert-style custom
``build_cycle_key`` needed here.

This module also doubles as the clinic's PGT-A adapter (see
``_CLINIC_PGT_MODULES`` in scripts/run_analysis.py): GGRC's export already
carries per-cycle PGT-A counts as plain columns (Biopsy Done/Tested
Embryos/Euploid/...), so :func:`analyze_reports` builds ``pgt_summary``
directly from those instead of parsing downloaded PDFs like clinic A/Fertimed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import ingest, normalize

# Two columns the draft script (Draft_for_NN.py) got wrong -- see docs/GGRC.md
# "Deviations from the draft script":
#   - it set "Заморожено эмбрионов" (frozen) = good-quality-blastocyst count,
#     even though the export carries real frozen counts under these four
#     "Cryo: ..." columns.
#   - it duplicated cleavage_d3 onto "Число эмбрионов 5 дня" (embryos_d5);
#     this adapter leaves embryos_d5 unmapped (NaN) instead, since it is
#     optional and not part of any KPI/DT-required field.
_CRYO_RAW_COLUMNS = [
    "Cryo: Day 5 Embryos", "Cryo: Day 6 Embryos", "Cryo: Day 7 Embryos",
    "Cryo: Other Blastocysts",
]

# GGRC's export already carries per-cycle PGT-A counts as plain columns --
# no PDF/portal step needed (unlike clinic A/Fertimed). Exposed so
# scripts/run_analysis.py can widen its PGT worklist for this clinic only.
PGT_RAW_COLUMNS = [
    "Biopsy Done", "Tested Embryos", "Not Tested Embryos",
    "Euploid", "Mosaic", "Failed/NA", "Pending Result", "Aneuploid",
]

# No per-embryo granularity from embedded cohort-level counts (only totals),
# so the PGT "events" table this adapter produces is always empty. Matches
# the shape src/embryo/pgt.py's EMBRYO_COLUMNS is used for, kept empty here.
EMBRYO_COLUMNS: list[str] = []


def _classify_outcome(value) -> float:
    """"1 No, 1 Yes" / "No, Yes" / "Biochemical, No" -> composite per-transfer
    tokens in one cell. Preg=1 if ANY token contains "yes" (case-insensitive,
    owner-confirmed 2026-07-23) -- "biochemical" and "no" alone are not a
    positive outcome. Empty/unparseable -> NaN (row is dropped downstream by
    the shared transferred>0 & Preg.notna() eligibility filter)."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if not s:
        return np.nan
    return 1.0 if "yes" in s.lower() else 0.0


def _follicles_estimate(df: pd.DataFrame) -> pd.Series:
    """No real follicle-count/AFC column in GGRC's export (owner-confirmed
    2026-07-23): estimate follicles = round(Oocytes N * 1.44). This makes
    OCCR read as a near-constant ~0.69 for every cycle -- expected, not a
    bug; document as an estimate, not a measurement, see docs/GGRC.md."""
    occ = pd.to_numeric(df["occ"], errors="coerce")
    return (occ * 1.44).round()


def _frozen_sum(df: pd.DataFrame) -> pd.Series:
    """Sum of the four raw "Cryo: ..." columns (min_count=1 so all-empty
    stays NaN rather than a fabricated 0) -- see module docstring for why
    this replaces the draft script's placeholder mapping."""
    cols = [c for c in _CRYO_RAW_COLUMNS if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    return numeric.sum(axis=1, min_count=1)


def _patient_name_fallback(df: pd.DataFrame) -> pd.Series:
    """4/29 transfer-file rows have a blank "Patients name" -- the external
    Digital Twin project skips any row with a blank ФИО outright. OPU Code is
    an internal cycle code, not PHI, so reusing it as a placeholder (same
    reasoning as Skyfert's ЭМК-as-ФИО fix) introduces no new PHI exposure."""
    if "patient_name" not in df.columns:
        return "GGRC-" + df["patient_id"].astype("string")
    name = df["patient_name"].astype("string")
    blank = name.isna() | (name.str.strip() == "")
    return name.where(~blank, "GGRC-" + df["patient_id"].astype("string"))


def _ensure_numeric_default(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    """Like a fillna, but safe when ``col`` never resolved at all (e.g.
    "attempt_no" -> "Attempts" is absent from the kpi-wide file entirely, and
    "transfer_day" has no source column in either GGRC file) -- df.get(col)
    would return a bare ``None``, not a Series, and crash pd.to_numeric/.fillna."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def _common_prepare(raw: pd.DataFrame, columns_cfg: dict, required_here: list[str]) -> tuple[pd.DataFrame, list]:
    """Resolve columns and hard-fail on THIS caller's own required set before
    touching any column (e.g. _follicles_estimate assumes "occ" exists) --
    the two GGRC files have different real required sets (see module
    docstring), so this check must happen ahead of the shared helpers below,
    not after."""
    resolved, unknown, _missing = ingest.resolve_columns(raw, columns_cfg)
    hard_missing = [c for c in required_here if c not in resolved.columns]
    if hard_missing:
        raise ValueError(f"нет обязательных колонок (GGRC): {hard_missing}")

    df = resolved.copy()
    df["follicles"] = _follicles_estimate(df)
    df["frozen"] = _frozen_sum(df)
    df["patient_name"] = _patient_name_fallback(df)
    if "fertilization_method" in df.columns:
        df["fertilization_method"] = df["fertilization_method"].astype("string").str.strip()
    return df, unknown


def prepare(raw: pd.DataFrame, columns_cfg: dict, settings_cfg: dict) -> tuple[pd.DataFrame, list]:
    """Eligible-cohort file ("(transfer)") -> canonical form. Same return
    shape as every other clinic adapter in this project."""
    required_here = ["patient_id", "age", "occ", "inseminated", "two_pn",
                      "cleavage_d3", "bl_total", "bl_good"]
    df, unknown = _common_prepare(raw, columns_cfg, required_here)

    df["attempt_no"] = _ensure_numeric_default(df, "attempt_no", 1)
    df["transfer_day"] = _ensure_numeric_default(df, "transfer_day", 5)
    if "transfer_outcome" not in df.columns:
        raise ValueError("нет обязательной колонки исхода переноса (Pregnancy) в transfer-файле GGRC")
    df["Preg"] = df["transfer_outcome"].apply(_classify_outcome)

    df = normalize.impute_amh(df, settings_cfg)
    df = normalize.impute_bmi(df, settings_cfg)
    return df, unknown


def prepare_wide(raw: pd.DataFrame, columns_cfg: dict, settings_cfg: dict) -> tuple[pd.DataFrame, list]:
    """Wide lab-population file ("kpi") -> canonical form. No outcome
    parsing (the file has no Pregnancy-equivalent column and KPI_SPECS/OZEMA
    never read "Preg"), no cohort-mask filtering -- every lab-evaluable row
    is kept, matching this file's role as the wide competency/OZEMA
    population (see module docstring)."""
    required_here = ["patient_id", "age", "occ"]
    df, unknown = _common_prepare(raw, columns_cfg, required_here)

    df["attempt_no"] = _ensure_numeric_default(df, "attempt_no", 1)
    df["transfer_day"] = _ensure_numeric_default(df, "transfer_day", 5)

    df = normalize.impute_amh(df, settings_cfg)
    df = normalize.impute_bmi(df, settings_cfg)
    return df, unknown


def _num(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(np.nan, index=index)
    return pd.to_numeric(series, errors="coerce")


def analyze_reports(
    cohort: pd.DataFrame,
    reports_dir,
    cycle_uid_col: str = "cycle_uid",
    name_col: str = "Patient Full Name",
    date_col_opu: str = "Date OPU",
    date_col_et: str = "Date ET",
) -> pd.DataFrame:
    """PGT-A summary straight from GGRC's embedded per-cycle columns --
    NO PDF parsing, NO portal, ``reports_dir`` is accepted only to match
    ``embryo.pgt.analyze_reports``'s call signature (scripts/run_analysis.py
    dispatches through ``_CLINIC_PGT_MODULES`` uniformly) and is otherwise
    unused. ``cohort`` must already carry ``PGT_RAW_COLUMNS`` (see
    scripts/run_analysis.py's ``_CLINIC_PGT_EXTRA_COLUMNS``)."""
    if cycle_uid_col not in cohort.columns:
        raise KeyError(f"нет обязательной псевдонимизированной колонки {cycle_uid_col!r}")

    biopsy_done = _num(cohort.get("Biopsy Done"), cohort.index)
    tested = _num(cohort.get("Tested Embryos"), cohort.index)
    not_tested = _num(cohort.get("Not Tested Embryos"), cohort.index)
    euploid = _num(cohort.get("Euploid"), cohort.index)
    aneuploid = _num(cohort.get("Aneuploid"), cohort.index)
    mosaic = _num(cohort.get("Mosaic"), cohort.index)
    failed_na = _num(cohort.get("Failed/NA"), cohort.index)
    pending = _num(cohort.get("Pending Result"), cohort.index)

    n_biopsied = biopsy_done.where(biopsy_done.notna(), tested.add(not_tested, fill_value=np.nan))
    n_other = failed_na.add(pending, fill_value=0.0).where(
        failed_na.notna() | pending.notna(), np.nan)

    records = []
    for i in cohort.index:
        n = n_biopsied.loc[i]
        rec = {
            "cycle_uid": cohort.loc[i, cycle_uid_col],
            "report_date": None,
            "pgt_run_id": None,
            "match_reason": "embedded_columns",
            "n_biopsied": None if pd.isna(n) else int(round(n)),
            "n_euploid": None if pd.isna(euploid.loc[i]) else int(round(euploid.loc[i])),
            "n_aneuploid": None if pd.isna(aneuploid.loc[i]) else int(round(aneuploid.loc[i])),
            "n_mosaic": None if pd.isna(mosaic.loc[i]) else int(round(mosaic.loc[i])),
            "n_other": None if pd.isna(n_other.loc[i]) else int(round(n_other.loc[i])),
            "euploid_rate": (round(float(euploid.loc[i]) / n, 4)
                              if n and not pd.isna(n) and n > 0 and not pd.isna(euploid.loc[i])
                              else None),
        }
        if not n or pd.isna(n) or n == 0:
            rec["pgt_note"] = "биопсия не проводилась / нет данных ПГТ-А"
        records.append(rec)

    result = pd.DataFrame.from_records(records)
    result.attrs["events"] = pd.DataFrame(columns=["cycle_uid", *EMBRYO_COLUMNS])
    result.attrs["unmatched"] = []
    return result
