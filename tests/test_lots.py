from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embryo.lots import (
    EVENT_COLUMNS,
    analyze_patients,
    analyze_window_events,
    build_lot_timeline,
    compare_flagged_vs_baseline,
    cultivation_window,
)


def _changes(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_default_cultivation_window_is_six_days():
    start, end = cultivation_window("2026-01-01 08:00:00")
    assert end - start == pd.Timedelta(days=6)


def test_build_lot_timeline_derives_periods_from_change_history():
    items = pd.DataFrame([
        {"record_id": "recItem1", "Наименование": "CSCM-NXC", "атрибут": "среда",
         "Производитель": "Irvine"},
    ])
    changes = _changes([
        {"record_id": "recC1", "Record": "CSCM-NXC", "All Barcodes": "LOT-A",
         "Change Date": "2026-01-01", "расходные материалы Link": ["recItem1"]},
        {"record_id": "recC2", "Record": "CSCM-NXC", "All Barcodes": "LOT-B",
         "Change Date": "2026-01-05", "расходные материалы Link": ["recItem1"]},
    ])
    timeline = build_lot_timeline(items, changes)

    assert list(timeline["lot"]) == ["LOT-A", "LOT-B"]
    assert timeline.loc[0, "period_end"] == pd.Timestamp("2026-01-05")
    assert pd.isna(timeline.loc[1, "period_end"])
    assert timeline.loc[0, "attribute"] == "среда"
    assert timeline.loc[0, "manufacturer"] == "Irvine"


def test_single_overlapping_period_is_stable():
    items = pd.DataFrame([{"record_id": "recItem1", "Наименование": "PVP", "атрибут": "среда",
                          "Производитель": "Irvine"}])
    changes = _changes([
        {"record_id": "recC1", "Record": "PVP", "All Barcodes": "LOT-A",
         "Change Date": "2025-12-01", "расходные материалы Link": ["recItem1"]},
    ])
    timeline = build_lot_timeline(items, changes)

    summary, events = analyze_window_events(timeline, "2026-01-01 08:00:00", days=6,
                                             cycle_uid="cy_test")
    assert summary["n_items_tracked"] == 1
    assert summary["n_items_changed"] == 0
    assert summary["lot_change_flag"] is False
    row = events.iloc[0]
    assert row["event_type"] == "stable"
    assert row["lot_before"] == row["lot_after"] == "LOT-A"
    assert set(EVENT_COLUMNS).issubset(events.columns)


def test_lot_change_inside_window_is_flagged():
    items = pd.DataFrame([{"record_id": "recItem1", "Наименование": "CSCM-NXC",
                          "атрибут": "среда", "Производитель": "Irvine"}])
    changes = _changes([
        {"record_id": "recC1", "Record": "CSCM-NXC", "All Barcodes": "LOT-A",
         "Change Date": "2025-12-30", "расходные материалы Link": ["recItem1"]},
        {"record_id": "recC2", "Record": "CSCM-NXC", "All Barcodes": "LOT-B",
         "Change Date": "2026-01-03", "расходные материалы Link": ["recItem1"]},
    ])
    timeline = build_lot_timeline(items, changes)

    summary, events = analyze_window_events(timeline, "2026-01-01 08:00:00", days=6,
                                             cycle_uid="cy_test")
    assert summary["n_items_changed"] == 1
    assert summary["lot_change_flag"] is True
    assert summary["changed_items"] == ["CSCM-NXC"]
    change = events.iloc[0]
    assert change["event_type"] == "lot_change"
    assert change["lot_before"] == "LOT-A"
    assert change["lot_after"] == "LOT-B"
    assert change["change_date"] == pd.Timestamp("2026-01-03")


def test_lot_change_outside_window_does_not_flag():
    items = pd.DataFrame([{"record_id": "recItem1", "Наименование": "CSCM-NXC",
                          "атрибут": "среда", "Производитель": "Irvine"}])
    changes = _changes([
        {"record_id": "recC1", "Record": "CSCM-NXC", "All Barcodes": "LOT-A",
         "Change Date": "2025-12-01", "расходные материалы Link": ["recItem1"]},
        {"record_id": "recC2", "Record": "CSCM-NXC", "All Barcodes": "LOT-B",
         "Change Date": "2025-12-15", "расходные материалы Link": ["recItem1"]},
    ])
    timeline = build_lot_timeline(items, changes)

    summary, events = analyze_window_events(timeline, "2026-01-01 08:00:00", days=6,
                                             cycle_uid="cy_test")
    assert summary["n_items_tracked"] == 1
    assert summary["n_items_changed"] == 0
    row = events.iloc[0]
    assert row["event_type"] == "stable"
    assert row["lot_before"] == row["lot_after"] == "LOT-B"


def test_no_timeline_data_is_a_note_not_a_crash():
    summary, events = analyze_window_events(pd.DataFrame(), "2026-01-01 08:00:00",
                                             cycle_uid="cy_missing")
    assert events.empty
    assert summary["n_items_tracked"] == 0
    assert summary["lot_change_flag"] is False
    assert summary["note"] == "нет данных о лотах"


def test_analyze_patients_is_keyed_only_by_cycle_uid_and_exposes_events():
    items = pd.DataFrame([{"record_id": "recItem1", "Наименование": "CSCM-NXC",
                          "атрибут": "среда", "Производитель": "Irvine"}])
    changes = _changes([
        {"record_id": "recC1", "Record": "CSCM-NXC", "All Barcodes": "LOT-A",
         "Change Date": "2025-12-30", "расходные материалы Link": ["recItem1"]},
        {"record_id": "recC2", "Record": "CSCM-NXC", "All Barcodes": "LOT-B",
         "Change Date": "2026-01-03", "расходные материалы Link": ["recItem1"]},
    ])
    timeline = build_lot_timeline(items, changes)

    patients = pd.DataFrame(
        [{"cycle_uid": "cy_abc123", "ID": "must-not-leak", "Date OPU": "2026-01-01 08:00:00"}]
    )
    result = analyze_patients(patients, timeline, days=6)

    assert result.loc[0, "cycle_uid"] == "cy_abc123"
    assert "ID" not in result.columns
    assert "patient_id" not in result.columns
    assert result.loc[0, "lot_change_flag"]
    events = result.attrs["events"]
    assert not events.empty
    assert set(events["cycle_uid"]) == {"cy_abc123"}
    assert "must-not-leak" not in events.to_string()


def test_analyze_patients_rejects_direct_identifier_as_the_only_key():
    patients = pd.DataFrame([{"ID": "123", "Date OPU": "2026-01-01"}])
    with pytest.raises(KeyError, match="cycle_uid"):
        analyze_patients(patients, pd.DataFrame())


def test_baseline_comparison_splits_flagged_from_the_rest_of_the_cohort():
    summary = pd.DataFrame([
        {"cycle_uid": "cy_flag1", "lot_change_flag": True},
        {"cycle_uid": "cy_flag2", "lot_change_flag": False},
        {"cycle_uid": "cy_rest1", "lot_change_flag": False},
        {"cycle_uid": "cy_rest2", "lot_change_flag": False},
        {"cycle_uid": "cy_rest3", "lot_change_flag": True},
    ])
    result = compare_flagged_vs_baseline(summary, ["cy_flag1", "cy_flag2"])

    assert result["flagged"] == {"n": 2, "n_excluded_no_coverage": 0, "lot_change_rate": pytest.approx(0.5)}
    assert result["baseline"] == {"n": 3, "n_excluded_no_coverage": 0,
                                  "lot_change_rate": pytest.approx(1 / 3)}


def test_baseline_comparison_excludes_cycles_with_no_resolvable_timeline():
    summary = pd.DataFrame([
        {"cycle_uid": "cy_flag1", "lot_change_flag": True, "error": None, "note": None},
        {"cycle_uid": "cy_rest1", "lot_change_flag": False, "error": None, "note": "нет данных о лотах"},
        {"cycle_uid": "cy_rest2", "lot_change_flag": False, "error": None, "note": None},
    ])
    result = compare_flagged_vs_baseline(summary, ["cy_flag1"])

    assert result["flagged"]["n"] == 1
    assert result["baseline"]["n"] == 1
    assert result["baseline"]["n_excluded_no_coverage"] == 1


def test_baseline_comparison_handles_empty_summary():
    result = compare_flagged_vs_baseline(pd.DataFrame(), ["cy_flag1"])
    assert result == {"flagged": None, "baseline": None, "note": "нет данных"}
