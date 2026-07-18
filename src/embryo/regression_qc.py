"""Actual-vs-predicted QC for the TabPFN-3 quantitative-stage regression.

Independent of reconcile.py (DT -> TabPFN-classifier -> OPU, by transfer outcome):
this compares OBSERVED funnel counts against the regression's P10-P90 patient
quantiles, per cycle_uid. A stage falling outside its interval is a QA signal for
model-auditor/lead-analyst, never a cause and never a substitute for the DT/TabPFN
outcome-probability reconciliation.

blasts_early and blasts_after_fert both predict the same observed quantity ("Bl")
at two different stages of knowledge (before/after the 2PN count is known) — both
are compared against "Bl" independently; the gap between them is itself informative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STAGE_ACTUAL_COL = {
    "mii": "MII",
    "pn2": "2pN",
    "blasts_early": "Bl",
    "blasts_after_fert": "Bl",
    "good": "Good Bl",
    "frozen": "Cryo",
}


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_regression_qc(
    kpi_cycles: pd.DataFrame,
    predictions: pd.DataFrame,
    stages_present: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join actual counts (kpi_cycles) against regression quantiles (predictions).

    Returns (per-cycle wide QC frame keyed by cycle_uid, cohort-level calibration
    summary: one row per stage with n_evaluated/share_in_interval80/mae_p50).
    """
    actual_cols = ["cycle_uid"] + [
        STAGE_ACTUAL_COL[s] for s in stages_present if STAGE_ACTUAL_COL[s] in kpi_cycles.columns
    ]
    actual = kpi_cycles[list(dict.fromkeys(actual_cols))].copy()
    out = predictions.merge(actual, on="cycle_uid", how="left", validate="one_to_one")

    calibration_rows = []
    out_of_interval = pd.Series([[] for _ in range(len(out))], index=out.index, dtype="object")
    for stage in stages_present:
        actual_col = STAGE_ACTUAL_COL[stage]
        if actual_col not in out.columns:
            continue
        p10 = _numeric(out[f"TPFN3_{stage}_p10"])
        p50 = _numeric(out[f"TPFN3_{stage}_p50"])
        p90 = _numeric(out[f"TPFN3_{stage}_p90"])
        fact = _numeric(out[actual_col])

        out[f"{stage}_actual"] = fact
        out[f"{stage}_p10"] = p10
        out[f"{stage}_p50"] = p50
        out[f"{stage}_p90"] = p90
        known = fact.notna() & p10.notna() & p90.notna()
        in_interval = pd.Series(pd.NA, index=out.index, dtype="boolean")
        in_interval.loc[known] = (fact.loc[known] >= p10.loc[known]) & (fact.loc[known] <= p90.loc[known])
        out[f"{stage}_in_interval80"] = in_interval

        outside = known & ~in_interval.fillna(False)
        for idx in out.index[outside]:
            out_of_interval.loc[idx] = out_of_interval.loc[idx] + [stage]

        n_evaluated = int(known.sum())
        share = float(in_interval.loc[known].mean()) if n_evaluated else np.nan
        mae = float((fact.loc[known] - p50.loc[known]).abs().mean()) if n_evaluated else np.nan
        calibration_rows.append({
            "stage": stage, "n_evaluated": n_evaluated,
            "share_in_interval80": share, "mae_p50": mae,
        })

    out["out_of_interval"] = out_of_interval
    calibration = pd.DataFrame(
        calibration_rows, columns=["stage", "n_evaluated", "share_in_interval80", "mae_p50"]
    )
    return out, calibration
