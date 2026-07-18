"""OZEMA — per-cycle oocyte-to-embryo arrest flags.

Methodology reproduced from the owner's OZEMA sheet (OPU table, Google Drive
`198CHO8OEhJthRtCYGw8BvOwSC3WtT4JTy19lQa132Dw`), citing Van Der Kelen et al.,
Hum Reprod 2026. This is a per-cycle ONE-SIDED EXACT BINOMIAL SIGNIFICANCE TEST
(alpha=0.05) against Vienna-style competency thresholds — not a raw
below-the-line check:

  OMA  (oocyte maturation arrest)   MII / OCC     significantly below 75%
  FF   (fertilization failure)      2PN / MII     significantly below 60%  (excl. TESE/SMF)
  ZA   (zygote arrest)              Cleaved / 2PN significantly below 95%
  EEA  (early embryo arrest)        Bl / 2PN      significantly below 40%

A stage needs n>=MIN_N to be testable; below that it is untestable/underpowered
(rendered "н/д", the project's existing missing-data convention — deliberately
NOT the literal string "n/a": that token is in pandas' default ``read_csv``
``na_values`` list and would silently turn back into a real ``NaN`` on any later
CSV re-read, exactly the kind of silent-missingness bug this pipeline avoids
elsewhere). Never silently treated as a pass and never dropped from the table.
A patient with >=2 flagged cycles meets the OZEMA recurrence criterion: a
candidate for a genetic-counseling DISCUSSION, never a diagnosis (CLAUDE.md: no
diagnoses/prescriptions from this pipeline).

OZEMA deliberately runs on the WIDER lab-stage population (no transfer/outcome
gate, no freeze-all/NO-retrieval/open-cycle exclusion — that exclusion set is
scoped only to the embryologist/doctor competency dashboard, see kpi.py's
``competency_exclusion_mask``). ``MIN_N`` alone naturally turns a freeze-all or
NO-retrieval cycle's untestable stages into "н/д" without any special-casing,
matching the owner's own sheet where a freeze-all cycle is still legitimately
flagged (e.g. FF on a Cryo cycle with 2PN=0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .kpi import surgical_sperm_mask

MIN_N = 4
ALPHA = 0.05
METHODOLOGY_CITATION = (
    "Van Der Kelen et al., Hum Reprod 2026; Vienna Consensus-style one-sided "
    "binomial test, alpha=0.05"
)
STAGE_LABELS = {
    "OMA": "Oocyte maturation arrest — MII/OCC значимо ниже 75%",
    "FF": "Fertilization failure — 2PN/MII значимо ниже 60% (искл. TESE/SMF)",
    "ZA": "Zygote arrest — дробление/2PN значимо ниже 95%",
    "EEA": "Early embryo arrest — Bl/2PN значимо ниже 40%",
}
_STAGE_ORDER = ("OMA", "FF", "ZA", "EEA")
_UNFLAGGED = "—"  # within thresholds, no significant arrest
_UNTESTABLE = "н/д"  # underpowered (n<MIN_N) -- NOT "n/a": collides with pandas' na_values


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _one_sided_flag(k: float, n: float, p0: float) -> bool | None:
    """True if k/n is significantly below p0 (one-sided binomial); None if untestable."""
    if pd.isna(k) or pd.isna(n) or n < MIN_N:
        return None
    k_int, n_int = int(round(k)), int(round(n))
    if k_int < 0 or k_int > n_int:
        return None
    return bool(binomtest(k_int, n_int, p0, alternative="less").pvalue < ALPHA)


def build_ozema_table(df: pd.DataFrame, columns_cfg: dict) -> pd.DataFrame:
    """Compute per-cycle OZEMA flags.

    ``df`` must carry the canonical columns produced by ``normalize.normalize``
    (cycle_uid, patient_id, date_opu, age, occ, mii, two_pn, cleavage_d3,
    bl_total, inseminated, embryologist, doctor; sperm_source optional). One
    output row per input row, aligned on ``df.index``.
    """
    occ = _numeric(df, "occ")
    mii = _numeric(df, "mii")
    two_pn = _numeric(df, "two_pn")
    cleavage = _numeric(df, "cleavage_d3")
    bl = _numeric(df, "bl_total")
    inseminated = _numeric(df, "inseminated")
    # Same "known MII wins, else inseminated" convention as kpi.py's FR denominator
    # for this all-ICSI lab (CLAUDE.md), applied here to the FF stage only.
    ff_den = mii.where(mii.notna() & mii.gt(0), inseminated)
    excl_surgical = surgical_sperm_mask(df, columns_cfg)

    stage_inputs = {
        "OMA": (mii, occ, 0.75, None),
        "FF": (two_pn, ff_den, 0.60, excl_surgical),
        "ZA": (cleavage, two_pn, 0.95, None),
        "EEA": (bl, two_pn, 0.40, None),
    }

    flags_per_row: list[str] = []
    stage_result: dict[str, list[str]] = {code: [] for code in _STAGE_ORDER}
    for idx in df.index:
        codes = []
        any_testable = False
        for code in _STAGE_ORDER:
            num_series, den_series, threshold, excl_mask = stage_inputs[code]
            if excl_mask is not None and bool(excl_mask.loc[idx]):
                stage_result[code].append("excluded")
                continue
            flagged = _one_sided_flag(num_series.loc[idx], den_series.loc[idx], threshold)
            if flagged is None:
                stage_result[code].append(_UNTESTABLE)
                continue
            any_testable = True
            stage_result[code].append("flag" if flagged else "ok")
            if flagged:
                codes.append(code)
        if codes:
            flags_per_row.append("+".join(codes))
        elif any_testable:
            flags_per_row.append(_UNFLAGGED)
        else:
            flags_per_row.append(_UNTESTABLE)

    out = pd.DataFrame(index=df.index)
    out["cycle_uid"] = df.get("cycle_uid")
    out["patient_id"] = df.get("patient_id")
    out["patient_name"] = df.get("patient_name")
    out["date_opu"] = df.get("date_opu")
    out["age"] = _numeric(df, "age")
    out["COC"] = occ
    out["MII"] = mii
    out["2PN"] = two_pn
    out["Cleav"] = cleavage
    out["Bl"] = bl
    out["ozema_flag"] = flags_per_row
    for code in _STAGE_ORDER:
        out[f"{code}_result"] = stage_result[code]
    out["embryologist"] = df.get("embryologist")
    out["doctor"] = df.get("doctor")

    flagged_mask = ~out["ozema_flag"].isin([_UNFLAGGED, _UNTESTABLE])
    out["flagged"] = flagged_mask
    if "patient_id" in out.columns:
        counts = out.loc[flagged_mask].groupby(out["patient_id"]).size()
        out["flagged_cycles_n"] = out["patient_id"].map(counts).fillna(0).astype(int)
    else:
        out["flagged_cycles_n"] = 0

    out["monitoring_note"] = [
        _monitoring_note(flag, n) for flag, n in zip(out["ozema_flag"], out["flagged_cycles_n"])
    ]
    return out


def _monitoring_note(flag: str, flagged_cycles_n: int) -> str:
    if flag in (_UNFLAGGED, _UNTESTABLE):
        return ""
    if flagged_cycles_n >= 2:
        return (f"Повторный паттерн ({flagged_cycles_n} циклов с флагом) — критерий OZEMA "
                f"выполнен: кандидат на обсуждение генетического консультирования.")
    return (f"Единичный цикл ниже ожидаемой компетенции ({flag}). Наблюдение; "
            f"повтор будет соответствовать критерию OZEMA.")


def genetic_counseling_candidates(ozema_table: pd.DataFrame) -> pd.DataFrame:
    """Patient-level (raw ID) rows meeting the >=2-flagged-cycles OZEMA criterion.

    Repeat attempts share one raw ``patient_id`` (see opu_table.py), so the
    recurrence rule is inherently patient-level, not cycle-level. Local-only:
    never passed to an agent packet at this granularity.
    """
    if ozema_table.empty or "patient_id" not in ozema_table.columns:
        return ozema_table.iloc[0:0].copy()
    candidates = ozema_table.loc[ozema_table["flagged_cycles_n"] >= 2].copy()
    sort_cols = [c for c in ("patient_id", "date_opu") if c in candidates.columns]
    return candidates.sort_values(sort_cols) if sort_cols else candidates
