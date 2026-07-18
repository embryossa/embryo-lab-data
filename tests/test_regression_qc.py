from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embryo.contracts import detect_regression_stages, validate_regression_quantiles
from embryo.regression_qc import build_regression_qc


def _predictions(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_detect_regression_stages_requires_all_six_columns():
    df = pd.DataFrame({
        "cycle_uid": ["c1"],
        "TPFN3_mii_p10": [1.0], "TPFN3_mii_p25": [2.0], "TPFN3_mii_p50": [3.0],
        "TPFN3_mii_p75": [4.0], "TPFN3_mii_p90": [5.0],
        # interval80_width missing -> "mii" must not count as present
        "TPFN3_pn2_p10": [1.0], "TPFN3_pn2_p25": [1.5], "TPFN3_pn2_p50": [2.0],
        "TPFN3_pn2_p75": [2.5], "TPFN3_pn2_p90": [3.0], "TPFN3_pn2_interval80_width": [2.0],
    })
    assert detect_regression_stages(df) == ["pn2"]


def test_validate_regression_quantiles_rejects_unordered_quantiles():
    df = _predictions([
        {"cycle_uid": "c1",
         "TPFN3_mii_p10": 5.0, "TPFN3_mii_p25": 2.0, "TPFN3_mii_p50": 3.0,
         "TPFN3_mii_p75": 4.0, "TPFN3_mii_p90": 6.0, "TPFN3_mii_interval80_width": 1.0},
    ])
    with pytest.raises(ValueError, match="not ordered"):
        validate_regression_quantiles(df)


def test_validate_regression_quantiles_accepts_ordered_quantiles():
    df = _predictions([
        {"cycle_uid": "c1",
         "TPFN3_mii_p10": 1.0, "TPFN3_mii_p25": 2.0, "TPFN3_mii_p50": 3.0,
         "TPFN3_mii_p75": 4.0, "TPFN3_mii_p90": 5.0, "TPFN3_mii_interval80_width": 4.0},
    ])
    assert validate_regression_quantiles(df) == ["mii"]


def test_validate_regression_quantiles_raises_when_no_stage_present():
    df = pd.DataFrame({"cycle_uid": ["c1"], "unrelated": [1]})
    with pytest.raises(ValueError, match="no regression stage columns"):
        validate_regression_quantiles(df)


def test_build_regression_qc_flags_actual_outside_interval():
    kpi_cycles = pd.DataFrame([
        {"cycle_uid": "c1", "MII": 9.0, "Bl": 2.0},   # MII outside [1,5]; Bl inside both blast stages
        {"cycle_uid": "c2", "MII": 3.0, "Bl": 2.0},   # MII inside [1,5]
    ])
    predictions = pd.DataFrame([
        {"cycle_uid": "c1",
         "TPFN3_mii_p10": 1.0, "TPFN3_mii_p25": 2.0, "TPFN3_mii_p50": 3.0,
         "TPFN3_mii_p75": 4.0, "TPFN3_mii_p90": 5.0, "TPFN3_mii_interval80_width": 4.0,
         "TPFN3_blasts_early_p10": 1.0, "TPFN3_blasts_early_p25": 1.5, "TPFN3_blasts_early_p50": 2.0,
         "TPFN3_blasts_early_p75": 2.5, "TPFN3_blasts_early_p90": 3.0, "TPFN3_blasts_early_interval80_width": 2.0},
        {"cycle_uid": "c2",
         "TPFN3_mii_p10": 1.0, "TPFN3_mii_p25": 2.0, "TPFN3_mii_p50": 3.0,
         "TPFN3_mii_p75": 4.0, "TPFN3_mii_p90": 5.0, "TPFN3_mii_interval80_width": 4.0,
         "TPFN3_blasts_early_p10": 1.0, "TPFN3_blasts_early_p25": 1.5, "TPFN3_blasts_early_p50": 2.0,
         "TPFN3_blasts_early_p75": 2.5, "TPFN3_blasts_early_p90": 3.0, "TPFN3_blasts_early_interval80_width": 2.0},
    ])
    qc, calibration = build_regression_qc(kpi_cycles, predictions, ["mii", "blasts_early"])

    c1 = qc.set_index("cycle_uid").loc["c1"]
    assert c1["mii_in_interval80"] == False
    assert c1["blasts_early_in_interval80"] == True
    assert c1["out_of_interval"] == ["mii"]

    c2 = qc.set_index("cycle_uid").loc["c2"]
    assert c2["out_of_interval"] == []

    mii_row = calibration.set_index("stage").loc["mii"]
    assert mii_row["n_evaluated"] == 2
    assert mii_row["share_in_interval80"] == 0.5
