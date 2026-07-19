from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from embryo.history import append_record, build_record, load_history


def _kpi_cycles():
    return pd.DataFrame({
        "age": [35, 41], "follicles": [10, 5], "occ": [8, 4], "mii": [None, 0],
        "inseminated": [8, 4], "two_pn": [6, 2], "cleavage_d3": [6, 2],
        "bl_total": [3, 1], "bl_good": [2, 0], "three_pn": [0, 1], "degrade": [0, 1],
    })


def _manifest():
    return {
        "run_id": "20260719T000000Z", "created_utc": "2026-07-19T00:00:00+00:00",
        "input_sha256": "abc123", "n_cycles": 2, "dnn_status": "valid_saved_scaler",
        "regression_status": "ok",
    }


def test_build_record_has_no_cycle_uid_or_patient_columns():
    review_list = pd.DataFrame({"cycle_uid": ["cy_1"]})
    record = build_record(_manifest(), _kpi_cycles(), review_list)

    assert record["run_id"] == "20260719T000000Z"
    assert record["n_cycles"] == 2
    assert record["review_list_n"] == 1
    assert "OCCR" in record["kpi"]
    assert record["kpi"]["OCCR"]["numerator"] == pytest.approx(12.0)
    assert record["kpi"]["OCCR"]["denominator"] == pytest.approx(15.0)
    dumped = str(record)
    assert "cycle_uid" not in dumped
    assert "patient_id" not in dumped


def test_build_record_embeds_baseline_and_regression_calibration():
    miri_baseline = {"flagged": {"n": 3, "warning_rate": np.float64(0.5)}, "baseline": {"n": 2, "warning_rate": None}}
    regression_calibration = pd.DataFrame([
        {"stage": "mii", "n_evaluated": 10, "share_in_interval80": 0.8, "mae_p50": 1.2},
    ])
    record = build_record(_manifest(), _kpi_cycles(), miri_baseline=miri_baseline,
                          regression_calibration=regression_calibration)

    assert record["miri_baseline"]["flagged"]["warning_rate"] == pytest.approx(0.5)
    assert record["regression_calibration"] == [
        {"stage": "mii", "n_evaluated": 10, "share_in_interval80": 0.8, "mae_p50": 1.2}
    ]


def test_append_and_load_history_round_trips_across_runs(tmp_path: Path):
    path = tmp_path / "run_history.jsonl"
    first = build_record({**_manifest(), "run_id": "run1"}, _kpi_cycles())
    second = build_record({**_manifest(), "run_id": "run2"}, _kpi_cycles())

    append_record(first, path)
    append_record(second, path)

    history = load_history(path)
    assert len(history) == 2
    assert history["run_id"].tolist() == ["run1", "run2"]
    assert "kpi_OCCR_rate" in history.columns


def test_load_history_returns_empty_frame_when_no_file_yet(tmp_path: Path):
    history = load_history(tmp_path / "does_not_exist.jsonl")
    assert history.empty


def test_append_record_survives_nan_and_numpy_types(tmp_path: Path):
    path = tmp_path / "run_history.jsonl"
    record = build_record(_manifest(), pd.DataFrame())  # empty cohort -> NaN rates
    append_record(record, path)
    history = load_history(path)
    assert len(history) == 1
