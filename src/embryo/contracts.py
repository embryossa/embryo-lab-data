"""Versioned, executable contracts for model-boundary tables."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Contract:
    name: str
    version: str
    required: tuple[str, ...]
    unique_key: str | None = None
    numeric: tuple[str, ...] = ()
    probability: tuple[str, ...] = ()
    binary: tuple[str, ...] = ()

    def validate(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.required if c not in df.columns]
        if missing:
            raise ValueError(f"{self.name} v{self.version}: missing columns {missing}")
        if self.unique_key:
            assert_unique(df, self.unique_key, f"{self.name} v{self.version}")
        for col in self.numeric:
            if col not in df.columns:
                continue
            raw = df[col]
            parsed = pd.to_numeric(raw, errors="coerce")
            invalid = raw.notna() & parsed.isna()
            if invalid.any():
                raise ValueError(f"{self.name} v{self.version}: non-numeric values in {col}")
            if np.isinf(parsed.dropna()).any():
                raise ValueError(f"{self.name} v{self.version}: infinite values in {col}")
        for col in self.probability:
            if col not in df.columns:
                continue
            values = pd.to_numeric(df[col], errors="coerce")
            if ((values.dropna() < 0) | (values.dropna() > 1)).any():
                raise ValueError(f"{self.name} v{self.version}: {col} must be in [0, 1]")
        for col in self.binary:
            if col not in df.columns:
                continue
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            if not values.isin([0, 1]).all():
                raise ValueError(f"{self.name} v{self.version}: {col} must be 0, 1, or missing")


DT_OUTPUT = Contract(
    "digital_twin_output", "1.1", ("_key", "DIGITAL TWIN"),
    unique_key="_key", numeric=("DIGITAL TWIN",), probability=("DIGITAL TWIN",),
)
TABPFN_OUTPUT = Contract(
    "tabpfn_output", "1.1",
    ("patient_id", "p_tabpfn", "ci_low", "ci_high"),
    unique_key="patient_id", numeric=("p_tabpfn", "ci_low", "ci_high"),
    probability=("p_tabpfn", "ci_low", "ci_high"),
)
DT_PREDICTIONS = Contract(
    "dt_predictions", "1.1",
    ("patient_id", "cycle_uid", "Preg", "PRAI", "DIGITAL TWIN",
     "PRAI CI (TabPFN), %", "ci_width_pp", "ci_wide",
     "tabpfn_join_status", "opu_join_status"),
    unique_key="patient_id",
    numeric=("Preg", "PRAI", "DIGITAL TWIN", "ci_width_pp"),
    probability=("PRAI", "DIGITAL TWIN"), binary=("Preg", "ci_wide"),
)

SCHEMA_VERSIONS: Mapping[str, str] = {
    DT_OUTPUT.name: DT_OUTPUT.version,
    TABPFN_OUTPUT.name: TABPFN_OUTPUT.version,
    DT_PREDICTIONS.name: DT_PREDICTIONS.version,
    "tabpfn_regression_output": "1.0",
}

# TabPFN-3 quantitative-stage regression (tabpfn3_ivf_regression_report_CPU_16K.py).
# A stage can be absent from a given run if its DEV/OPU feature had no signal (the
# external script skips that RegressionTask rather than erroring) — the contract
# therefore validates "whichever stages are present", never a fixed column set.
REGRESSION_STAGES: tuple[str, ...] = (
    "mii", "pn2", "blasts_early", "blasts_after_fert", "good", "frozen",
)
REGRESSION_QUANTILE_SUFFIXES: tuple[str, ...] = (
    "p10", "p25", "p50", "p75", "p90", "interval80_width",
)


def detect_regression_stages(df: pd.DataFrame) -> list[str]:
    """Stages where all 6 TPFN3_<stage>_<suffix> columns are present in df."""
    return [
        stage for stage in REGRESSION_STAGES
        if all(f"TPFN3_{stage}_{suffix}" in df.columns for suffix in REGRESSION_QUANTILE_SUFFIXES)
    ]


def validate_regression_quantiles(df: pd.DataFrame) -> list[str]:
    """Validate the TabPFN-3 regression output and return the stages present.

    Requires a unique ``cycle_uid`` and, for each present stage, numeric quantile
    columns monotonically ordered p10<=p25<=p50<=p75<=p90 on non-missing rows.
    """
    assert_unique(df, "cycle_uid", "tabpfn_regression_output")
    stages = detect_regression_stages(df)
    if not stages:
        raise ValueError(
            "tabpfn_regression_output: no regression stage columns present "
            f"(expected TPFN3_<stage>_<suffix> for one of {REGRESSION_STAGES})"
        )
    order = ("p10", "p25", "p50", "p75", "p90")
    for stage in stages:
        cols = {suffix: f"TPFN3_{stage}_{suffix}" for suffix in REGRESSION_QUANTILE_SUFFIXES}
        parsed = {suffix: pd.to_numeric(df[col], errors="coerce") for suffix, col in cols.items()}
        for suffix, values in parsed.items():
            raw = df[cols[suffix]]
            invalid = raw.notna() & values.isna()
            if invalid.any():
                raise ValueError(
                    f"tabpfn_regression_output: non-numeric values in {cols[suffix]}"
                )
        complete = pd.concat(parsed.values(), axis=1).notna().all(axis=1)
        prev = parsed[order[0]]
        for suffix in order[1:]:
            current = parsed[suffix]
            bad = complete & (prev > current)
            if bad.any():
                rows = df.index[bad].tolist()[:10]
                raise ValueError(
                    f"tabpfn_regression_output: {stage} quantiles not ordered "
                    f"(p10<=p25<=p50<=p75<=p90) in rows {rows}"
                )
            prev = current
    return stages


def normalized_key(series: pd.Series, label: str) -> pd.Series:
    """Normalize an identifier without converting missing values into the string ``NAN``."""
    out = series.astype("string").str.strip().str.upper()
    out = out.mask(out.eq(""), pd.NA)
    if out.isna().any():
        rows = out.index[out.isna()].tolist()[:10]
        raise ValueError(f"{label}: empty identifier in rows {rows}")
    return out


def assert_unique(df: pd.DataFrame, key: str, contract_name: str) -> None:
    if key not in df.columns:
        raise KeyError(f"{contract_name}: missing key {key}")
    normalized = normalized_key(df[key], f"{contract_name}.{key}")
    dup = normalized.duplicated(keep=False)
    if dup.any():
        vals = normalized.loc[dup].unique()[:10].tolist()
        raise ValueError(f"{contract_name}: non-unique {key}: {vals}")


def validate_ci(df: pd.DataFrame) -> None:
    """Validate ordering of non-missing TabPFN confidence limits."""
    lo = pd.to_numeric(df["ci_low"], errors="coerce")
    hi = pd.to_numeric(df["ci_high"], errors="coerce")
    unpaired = lo.isna() ^ hi.isna()
    if unpaired.any():
        raise ValueError(f"tabpfn_output: unpaired CI limit in rows {unpaired[unpaired].index.tolist()[:10]}")
    bad = lo.notna() & hi.notna() & (lo > hi)
    if bad.any():
        raise ValueError(f"tabpfn_output: ci_low > ci_high in rows {bad[bad].index.tolist()[:10]}")
    p = pd.to_numeric(df["p_tabpfn"], errors="coerce")
    outside = p.notna() & lo.notna() & hi.notna() & ((p < lo) | (p > hi))
    if outside.any():
        raise ValueError(f"tabpfn_output: p_tabpfn outside CI in rows {outside[outside].index.tolist()[:10]}")
    if np.isinf(lo).any() or np.isinf(hi).any():
        raise ValueError("tabpfn_output: confidence limits must be finite")


def validate_binary_outcome(series: pd.Series, label: str = "Preg") -> None:
    values = pd.to_numeric(series, errors="coerce")
    invalid_text = series.notna() & values.isna()
    invalid_value = values.notna() & ~values.isin([0, 1])
    if invalid_text.any() or invalid_value.any():
        rows = series.index[invalid_text | invalid_value].tolist()[:10]
        raise ValueError(f"{label} must be binary 0/1 or missing; invalid rows {rows}")
