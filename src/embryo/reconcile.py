"""Deterministic DT -> TabPFN -> OPU reconciliation on the accepted unique ID."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from ._io import atomic_write_csv, atomic_write_excel
from .contracts import (DT_PREDICTIONS, TABPFN_OUTPUT, assert_unique, normalized_key,
                        validate_binary_outcome, validate_ci)
from .ingest import load_table
from .privacy import add_cycle_uid, build_crosswalk


DT_KEY_CANDIDATES = ("_key", "patient_id", "ID")
# "_key" (ID+Date OPU composite, see scripts/run_analysis.py:_add_opu_aliases) takes
# priority: a patient with repeat cycles shares one raw "ID" across rows, so joining
# on "ID" would silently collapse distinct cycles into one. "ID" itself stays the raw
# passport ID for human-facing display (final OPU_table_enriched.xlsx) and is not
# mutated anywhere in this pipeline.
OPU_KEY_CANDIDATES = ("_key", "ID", "patient_id")


def _find_col(df: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"{label}: none of the identifier columns exists: {candidates}")


def _with_join_key(df: pd.DataFrame, source_col: str, label: str) -> pd.DataFrame:
    out = df.copy()
    out["_join"] = normalized_key(out[source_col], label)
    assert_unique(out, "_join", label)
    return out


def _numeric(series: pd.Series | object, index: pd.Index) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce")
    return pd.Series(series, index=index, dtype="float64")


def _ci_text(lo: object, hi: object) -> object:
    if pd.isna(lo) or pd.isna(hi):
        return pd.NA
    return f"{float(lo) * 100:.1f}\u2013{float(hi) * 100:.1f}%"


def build_dt_predictions(dt: pd.DataFrame, tabpfn: pd.DataFrame | None,
                         opu: pd.DataFrame, secret: bytes,
                         threshold_pp: float = 20.0) -> pd.DataFrame:
    """Build the authoritative analytics table with explicit join status.

    DT is the base population. TabPFN and OPU are left-joined, so an absent model
    result is visible rather than silently dropping a cycle. Zero probabilities and
    zero outcomes remain values; only nulls are treated as missing.
    """
    dt_key = _find_col(dt, DT_KEY_CANDIDATES, "Digital Twin")
    opu_key = _find_col(opu, OPU_KEY_CANDIDATES, "OPU")
    d = _with_join_key(dt, dt_key, "Digital Twin")
    o = _with_join_key(opu, opu_key, "OPU")

    if "DIGITAL TWIN" not in d.columns:
        raise ValueError("Digital Twin output has no 'DIGITAL TWIN' column")

    # Keep the DT analytics payload, but expose one canonical public join key.
    out = d.copy()
    out["patient_id"] = out[dt_key].astype("string").str.strip()

    opu_fields = [c for c in ("Preg", "Date OPU", "Date ET", "Embryologist", "Doctor",
                              "IVF/ICSI") if c in o.columns]
    opu_lookup = o[["_join", *opu_fields]].rename(
        columns={c: f"{c}__opu" for c in opu_fields}
    )
    out = out.merge(opu_lookup, on="_join", how="left", validate="one_to_one", indicator="_opu_merge")
    out["opu_join_status"] = out.pop("_opu_merge").map(
        {"both": "matched", "left_only": "missing", "right_only": "unexpected"}
    ).astype("string")

    # Transfer outcome: DT value wins if present; OPU fills only a true missing value.
    dt_preg = _numeric(out["Preg"], out.index) if "Preg" in out.columns else _numeric(np.nan, out.index)
    opu_preg = (_numeric(out["Preg__opu"], out.index)
                if "Preg__opu" in out.columns else _numeric(np.nan, out.index))
    out["Preg"] = dt_preg.combine_first(opu_preg)
    validate_binary_outcome(out["Preg"], "reconciled Preg")
    out["DIGITAL TWIN"] = _numeric(out["DIGITAL TWIN"], out.index)

    # Copy non-conflicting locally useful OPU fields into the reconciled table.
    for field in opu_fields:
        source = f"{field}__opu"
        if field == "Preg" or source not in out.columns:
            continue
        if field in out.columns:
            out[field] = out[field].combine_first(out[source])
        else:
            out[field] = out[source]

    # DT's own CSV already carries placeholder PRAI/ci_low/ci_high columns (empty
    # unless DT was fed a prior "Predictions" column, which this pipeline never does).
    # Drop them so the merge below doesn't silently suffix both into PRAI_x/PRAI_y.
    out = out.drop(columns=[c for c in ("PRAI", "ci_low", "ci_high") if c in out.columns],
                   errors="ignore")

    if tabpfn is None:
        out["PRAI"] = np.nan
        out["ci_low"] = np.nan
        out["ci_high"] = np.nan
        out["tabpfn_join_status"] = "not_run"
    else:
        TABPFN_OUTPUT.validate(tabpfn)
        validate_ci(tabpfn)
        t = _with_join_key(tabpfn, "patient_id", "TabPFN")
        selected = t[["_join", "p_tabpfn", "ci_low", "ci_high"]].copy()
        selected = selected.rename(columns={"p_tabpfn": "PRAI"})
        out = out.merge(selected, on="_join", how="left", validate="one_to_one",
                        indicator="_tab_merge")
        out["tabpfn_join_status"] = out.pop("_tab_merge").map(
            {"both": "matched", "left_only": "missing", "right_only": "unexpected"}
        ).astype("string")
        for col in ("PRAI", "ci_low", "ci_high"):
            out[col] = _numeric(out[col], out.index)

    out["PRAI CI (TabPFN), %"] = [
        _ci_text(lo, hi) for lo, hi in zip(out["ci_low"], out["ci_high"])
    ]
    out["ci_width_pp"] = (out["ci_high"] - out["ci_low"]) * 100.0
    out["ci_wide"] = out["ci_width_pp"].gt(float(threshold_pp)).fillna(False)
    out = add_cycle_uid(out, "patient_id", secret)

    # Private implementation columns and duplicate OPU merge columns do not belong
    # to the stable table.
    out = out.drop(columns=[c for c in out.columns if c.endswith("__opu")], errors="ignore")
    out = out.drop(columns=["_join"], errors="ignore")
    lead = list(DT_PREDICTIONS.required)
    out = out[lead + [c for c in out.columns if c not in lead]]
    DT_PREDICTIONS.validate(out)
    return out


def enrich_opu_table(opu: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Reverse-join predictions into the local OPU table; never sent to Claude."""
    opu_key = _find_col(opu, OPU_KEY_CANDIDATES, "OPU")
    o = _with_join_key(opu, opu_key, "OPU")
    p = _with_join_key(predictions, "patient_id", "dt_predictions")
    fields = ["PRAI", "DIGITAL TWIN", "PRAI CI (TabPFN), %", "cycle_uid",
              "ci_width_pp", "ci_wide"]
    lookup = p[["_join", *fields]].rename(columns={c: f"__pred_{c}" for c in fields})
    out = o.merge(lookup, on="_join", how="left", validate="one_to_one")
    for field in fields:
        incoming = f"__pred_{field}"
        # A fresh model result replaces a stale value; missing new output preserves it.
        if field in out.columns:
            out[field] = out[incoming].combine_first(out[field])
        else:
            out[field] = out[incoming]
    return out.drop(columns=["_join", *[f"__pred_{c}" for c in fields]])


def build_review_list(predictions: pd.DataFrame,
                      analytics_review: pd.DataFrame | None = None,
                      priority_cutoff: int = 2) -> pd.DataFrame:
    """Union analytics review cases with every TabPFN interval wider than threshold."""
    assert_unique(predictions, "patient_id", "dt_predictions")
    selected_keys: set[str] = set(
        normalized_key(
            predictions.loc[predictions["ci_wide"].fillna(False), "patient_id"],
            "ci_wide.patient_id",
        ).tolist()
    )
    if analytics_review is not None and not analytics_review.empty:
        key = _find_col(analytics_review, ("patient_id", "_key", "ID"), "analytics review")
        selected = analytics_review
        if "priority" in selected.columns:
            priority = pd.to_numeric(selected["priority"], errors="coerce")
            selected = selected.loc[priority.le(priority_cutoff)]
        if not selected.empty:
            selected_keys.update(normalized_key(selected[key], "analytics review").tolist())
    prediction_keys = normalized_key(predictions["patient_id"], "dt_predictions.patient_id")
    return predictions.loc[prediction_keys.isin(selected_keys)].copy()


def write_reconciled_outputs(predictions: pd.DataFrame, opu: pd.DataFrame,
                             predictions_path: str | Path,
                             opu_path: str | Path,
                             crosswalk_path: str | Path) -> dict[str, Path]:
    """Validate and atomically commit the three authoritative local artifacts."""
    DT_PREDICTIONS.validate(predictions)
    crosswalk = build_crosswalk(predictions, "patient_id")
    outputs = {"dt_predictions": atomic_write_csv(predictions, predictions_path),
               "crosswalk": atomic_write_csv(crosswalk, crosswalk_path)}
    opu_path = Path(opu_path)
    if opu_path.suffix.lower() == ".csv":
        outputs["opu"] = atomic_write_csv(opu, opu_path)
    else:
        outputs["opu"] = atomic_write_excel({"OPU table": opu}, opu_path)
    return outputs


# Backward-compatible helpers used by the earlier orchestrator.
def reconcile(dt_path: str, opu_path: str, dt_key="_key",
              dt_key_fallback="patient_id", opu_key="ID") -> pd.DataFrame:
    dt = load_table(dt_path)
    opu = load_table(opu_path)
    d = _with_join_key(dt, _find_col(dt, (dt_key, dt_key_fallback), "DT"), "DT")
    o = _with_join_key(opu, opu_key, "OPU")
    result = d.merge(o, on="_join", how="inner", suffixes=("_dt", "_opu"),
                     validate="one_to_one")
    preg_col = "Preg_dt" if "Preg_dt" in result.columns else "Preg"
    return result.loc[result[preg_col].notna()].copy() if preg_col in result else result


def _parse_ci_width_pp(val) -> float:
    if pd.isna(val):
        return np.nan
    values = re.findall(r"\d+(?:[.,]\d+)?", str(val))
    nums = [float(x.replace(",", ".")) for x in values]
    return abs(nums[1] - nums[0]) if len(nums) >= 2 else (nums[0] if nums else np.nan)


def add_ci_width(df: pd.DataFrame, tabpfn_path: str | None = None,
                 ci_col="PRAI CI (TabPFN), %", threshold_pp: float = 20.0,
                 key="patient_id") -> pd.DataFrame:
    out = df.copy()
    if tabpfn_path:
        tab = load_table(tabpfn_path)
        TABPFN_OUTPUT.validate(tab)
        t = _with_join_key(tab, key, "TabPFN")
        t["ci_width_pp"] = (pd.to_numeric(t["ci_high"], errors="coerce")
                            - pd.to_numeric(t["ci_low"], errors="coerce")) * 100
        if "_join" not in out:
            out["_join"] = normalized_key(out[key], "analytics")
        out = out.merge(t[["_join", "ci_width_pp"]], on="_join", how="left",
                        validate="one_to_one")
    elif "ci_width_pp" not in out and ci_col in out:
        out["ci_width_pp"] = out[ci_col].map(_parse_ci_width_pp)
    if "ci_width_pp" not in out:
        out["ci_width_pp"] = np.nan
    out["ci_wide"] = pd.to_numeric(out["ci_width_pp"], errors="coerce").gt(threshold_pp).fillna(False)
    return out
