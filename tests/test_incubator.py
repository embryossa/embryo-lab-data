from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embryo.incubator import (
    EVENT_COLUMNS,
    analyze_patients,
    analyze_window_events,
    compare_flagged_vs_baseline,
    cultivation_window,
)


def _miri_frame(times: list[str]) -> pd.DataFrame:
    n = len(times)
    data = {
        "Date": pd.to_datetime(times),
        **{f"Temp{i}": [37.0] * n for i in range(1, 13)},
        "Co2Concentration": [6.5] * n,
        "O2Concentration": [5.0] * n,
        "Co2Flow": [0.0] * n,
        "N2Flow": [0.0] * n,
        "Co2Pressure": [1.0] * n,
        "N2Pressure": [1.0] * n,
        "TempSetPoint": [37.0] * n,
        "Co2ConcentrationSetPoint": [6.5] * n,
        "O2ConcentrationSetPoint": [5.0] * n,
        **{f"Chamber{i}TemperatureAlarm": [0] * n for i in range(1, 7)},
        "Co2ConcentrationAlarm": [0] * n,
        "O2ConcentrationAlarm": [0] * n,
        "Co2PressureAlarm": [0] * n,
        "N2PressureAlarm": [0] * n,
        "IsLidOpen": [0] * n,
    }
    return pd.DataFrame(data)


def test_default_cultivation_window_is_six_days():
    start, end = cultivation_window("2026-01-01 08:00:00")
    assert end - start == pd.Timedelta(days=6)


def test_real_episodes_include_duration_deviation_recovery_and_severity():
    times = [
        "2026-01-01 08:00:00",
        "2026-01-01 08:00:30",
        "2026-01-01 08:01:00",
        "2026-01-01 08:01:30",
        "2026-01-01 08:02:00",
        "2026-01-01 08:02:30",
    ]
    logs = _miri_frame(times)
    logs.loc[1:2, "Temp1"] = [37.7, 38.2]
    logs.loc[4, "Co2ConcentrationAlarm"] = 1
    logs.loc[2:3, "IsLidOpen"] = 1

    summary, events = analyze_window_events(
        logs,
        times[0],
        days=3 / 1440,  # end at 08:03, avoiding a synthetic boundary gap
        cycle_uid="cy_test",
    )

    temp = events.query("event_type == 'temperature_excursion' and sensor == 'Temp1'").iloc[0]
    assert temp["start"] == pd.Timestamp(times[1])
    assert temp["end"] == pd.Timestamp(times[2])
    assert temp["duration_minutes"] == pytest.approx(1.0)
    assert temp["max_deviation"] == pytest.approx(1.2)
    assert temp["recovery_time_minutes"] == pytest.approx(0.5)
    assert temp["severity"] == "critical"  # >= 2 x temperature tolerance

    alarm = events.query("event_type == 'alarm'").iloc[0]
    assert alarm["duration_minutes"] == pytest.approx(0.5)
    assert alarm["severity"] == "warning"

    lid = events.query("event_type == 'lid_open'").iloc[0]
    assert lid["duration_minutes"] == pytest.approx(1.0)
    assert lid["recovery_time_minutes"] == pytest.approx(0.0)

    assert summary["warning"] is True
    assert summary["critical_count"] == 1
    assert summary["event_count"] == 3
    assert set(EVENT_COLUMNS).issubset(events.columns)
    assert set(events["cycle_uid"]) == {"cy_test"}


def test_zero_alarm_and_values_at_setpoint_do_not_raise_warning():
    times = ["2026-01-01 08:00:00", "2026-01-01 08:00:30"]
    logs = _miri_frame(times)
    summary, events = analyze_window_events(logs, times[0], days=1 / 1440)

    assert events.empty
    assert summary["event_count"] == 0
    assert summary["warning_count"] == 0
    assert summary["critical_count"] == 0
    assert summary["warning"] is False
    assert summary["max_severity"] == "none"


def test_internal_cadence_gap_is_one_episode_with_missing_duration():
    times = [
        "2026-01-01 08:00:00",
        "2026-01-01 08:00:30",
        "2026-01-01 08:06:00",
        "2026-01-01 08:06:30",
    ]
    logs = _miri_frame(times)
    summary, events = analyze_window_events(
        logs,
        times[0],
        days=7 / 1440,
        gap_factor=3,
        severity_thresholds={"critical_gap_minutes": 5.0},
    )

    gaps = events.query("event_type == 'gap'")
    assert len(gaps) == 1
    gap = gaps.iloc[0]
    assert gap["start"] == pd.Timestamp("2026-01-01 08:01:00")
    assert gap["end"] == pd.Timestamp("2026-01-01 08:06:00")
    assert gap["duration_minutes"] == pytest.approx(5.0)
    assert gap["severity"] == "critical"
    assert summary["gap_count"] == 1
    assert summary["warning"] is True


def test_analyze_patients_is_keyed_only_by_cycle_uid_and_exposes_events(tmp_path: Path):
    times = ["2026-01-01 08:00:00", "2026-01-01 08:00:30"]
    logs = _miri_frame(times)
    logs.loc[0, "O2Concentration"] = 6.2
    folder = tmp_path / "6391" / "Miri-6391-Incubation-[260101-260107]" / "Data"
    folder.mkdir(parents=True)
    logs.to_csv(folder / "2026.01.01_data.csv", sep=";", index=False, date_format="%Y.%m.%d %H:%M:%S")

    patients = pd.DataFrame(
        [{"cycle_uid": "cy_abc123", "ID": "must-not-leak", "Date OPU": times[0]}]
    )
    result = analyze_patients(patients, str(tmp_path), days=1 / 1440)

    assert result.loc[0, "cycle_uid"] == "cy_abc123"
    assert "ID" not in result.columns
    assert "patient_id" not in result.columns
    assert result.loc[0, "warning"]
    events = result.attrs["events"]
    assert not events.empty
    assert set(events["cycle_uid"]) == {"cy_abc123"}
    assert "must-not-leak" not in events.to_string()


def test_analyze_patients_rejects_direct_identifier_as_the_only_key(tmp_path: Path):
    patients = pd.DataFrame([{"ID": "123", "Date OPU": "2026-01-01"}])
    with pytest.raises(KeyError, match="cycle_uid"):
        analyze_patients(patients, str(tmp_path))


def test_gap_reason_classifies_short_vs_long_outages_at_the_8_hour_boundary():
    # Internal gap of ~2h (well under 8h): a lost ability to WRITE logs (e.g. a power
    # outage) -- a real adverse event, never "missing data". Window ends right after
    # the last log point (as in test_internal_cadence_gap_...) so there is no extra
    # trailing gap to confuse the assertion.
    short_times = ["2026-01-01 08:00:00", "2026-01-01 08:00:30", "2026-01-01 10:00:30"]
    short_logs = _miri_frame(short_times)
    _, short_events = analyze_window_events(
        short_logs, short_times[0], days=(121.5 / 1440),
        severity_thresholds={"gap_reason_threshold_minutes": 480.0},
    )
    short_gap = short_events.query("event_type == 'gap'").iloc[0]
    assert short_gap["duration_minutes"] < 480.0
    assert short_gap["gap_reason"] == "power_or_logging_outage"

    # Internal gap of ~9h (over 8h): an explicit stop of the incubator or of logging.
    long_times = ["2026-01-01 08:00:00", "2026-01-01 08:00:30", "2026-01-01 17:00:30"]
    long_logs = _miri_frame(long_times)
    _, long_events = analyze_window_events(
        long_logs, long_times[0], days=(541.5 / 1440),
        severity_thresholds={"gap_reason_threshold_minutes": 480.0},
    )
    long_gap = long_events.query("event_type == 'gap'").iloc[0]
    assert long_gap["duration_minutes"] >= 480.0
    assert long_gap["gap_reason"] == "incubator_or_logging_stopped"

    # A completely absent log (test_absent_log_is_a_critical_full_window_gap's
    # scenario) also carries gap_reason -- a 6-day absence is well over the threshold.
    _, missing_events = analyze_window_events(
        pd.DataFrame(), short_times[0], severity_thresholds={"gap_reason_threshold_minutes": 480.0},
    )
    assert missing_events.iloc[0]["gap_reason"] == "incubator_or_logging_stopped"

    # non-gap episodes carry no gap_reason
    temp_times = ["2026-01-01 08:00:00", "2026-01-01 08:00:30", "2026-01-01 08:01:00"]
    temp_logs = _miri_frame(temp_times)
    temp_logs.loc[1, "Temp1"] = 39.0
    _, temp_events = analyze_window_events(temp_logs, temp_times[0], days=2 / 1440)
    excursion = temp_events.query("event_type == 'temperature_excursion'").iloc[0]
    assert pd.isna(excursion["gap_reason"])


def test_absent_log_is_a_critical_full_window_gap():
    summary, events = analyze_window_events(
        pd.DataFrame(),
        "2026-01-01 08:00:00",
        cycle_uid="cy_missing",
    )
    assert len(events) == 1
    assert events.loc[0, "event_type"] == "gap"
    assert events.loc[0, "duration_minutes"] == pytest.approx(6 * 24 * 60)
    assert events.loc[0, "severity"] == "critical"
    assert summary["warning"] is True
    assert summary["critical_count"] == 1


def test_baseline_comparison_splits_flagged_from_the_rest_of_the_cohort():
    summary = pd.DataFrame([
        {"cycle_uid": "cy_flag1", "warning": True, "critical_count": 1},
        {"cycle_uid": "cy_flag2", "warning": False, "critical_count": 0},
        {"cycle_uid": "cy_rest1", "warning": False, "critical_count": 0},
        {"cycle_uid": "cy_rest2", "warning": False, "critical_count": 0},
        {"cycle_uid": "cy_rest3", "warning": True, "critical_count": 0},
    ])
    result = compare_flagged_vs_baseline(summary, ["cy_flag1", "cy_flag2"])

    assert result["flagged"] == {
        "n": 2, "n_excluded_no_coverage": 0,
        "warning_rate": pytest.approx(0.5), "critical_rate": pytest.approx(0.5),
    }
    assert result["baseline"] == {
        "n": 3, "n_excluded_no_coverage": 0,
        "warning_rate": pytest.approx(1 / 3), "critical_rate": pytest.approx(0.0),
    }


def test_baseline_comparison_excludes_cycles_with_no_synced_coverage():
    # A missing local download ("нет логов") must not be counted as an observed
    # absence of incubator events in either group's denominator.
    summary = pd.DataFrame([
        {"cycle_uid": "cy_flag1", "warning": True, "critical_count": 1, "error": None},
        {"cycle_uid": "cy_rest1", "warning": False, "critical_count": 0, "error": "нет логов"},
        {"cycle_uid": "cy_rest2", "warning": False, "critical_count": 0, "error": None},
    ])
    result = compare_flagged_vs_baseline(summary, ["cy_flag1"])

    assert result["flagged"]["n"] == 1
    assert result["baseline"]["n"] == 1
    assert result["baseline"]["n_excluded_no_coverage"] == 1


def test_baseline_comparison_handles_empty_summary():
    result = compare_flagged_vs_baseline(pd.DataFrame(), ["cy_flag1"])
    assert result == {"flagged": None, "baseline": None, "note": "нет данных"}
