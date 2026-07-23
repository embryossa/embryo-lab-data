"""Синтетические фикстуры и тесты для GGRC (Armenia) (см.
src/embryo/clinics/ggrc.py и docs/GGRC.md). Покрывает ingest/normalize
(ggrc.prepare/prepare_wide), embedded-PGT-адаптер (ggrc.analyze_reports) и
клинико-специфичные ветки scripts/run_analysis.py (_CLINIC_NORMALIZERS/
_CLINIC_PGT_MODULES/_CLINIC_WIDE_LOADERS/_CLINIC_PGT_EXTRA_COLUMNS).

Никаких реальных ФИО/данных пациентов — только синтетические строки."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

from embryo import ingest
from embryo.clinics import ggrc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_analysis import (  # noqa: E402
    _CLINIC_NORMALIZERS, _CLINIC_PGT_EXTRA_COLUMNS, _CLINIC_PGT_MODULES,
    _CLINIC_WIDE_LOADERS, _add_opu_aliases,
)

ROOT = Path(__file__).resolve().parents[1]
COLUMNS_CFG = ingest.load_config(str(ROOT / "config/clinics/ggrc/columns.yaml"))
SETTINGS_CFG = {
    "amh_impute": {"a_afc": 0.16, "b_occ": 0.05, "c_age": 0.08, "intercept": 0.5, "floor": 0.1, "ceil": 12.0},
    "bmi_impute": {"constant": 24.0},
}


def _transfer_raw(**overrides) -> pd.DataFrame:
    base = {
        "OPU Code": ["T001", "T002", "T003", "T004"],
        "Patients name": ["Test Patient A", None, "Test Patient C", "Test Patient D"],
        "Age": [28, 32, 35, 30],
        "Attempts": [1, 2, 1, None],
        "OPU Date": pd.to_datetime(["2026-01-10", "2026-01-11", "2026-01-12", "2026-01-13"]),
        "Oocytes N": [10, 8, 6, 12],
        "MII": [8, 6, 5, 10],
        "number of inseminated oocytes": [8, 6, 5, 10],
        "2pN": [6, 4, 3, 8],
        "3pN or more": [0, 1, 0, 0],
        "Degrade": [0, 0, 1, 0],
        "number of embryos dividing on day 3": [5, 3, 2, 6],
        "CombinedDays": [3, 2, 1, 4],
        "Good quality BL": [2, 1, 0, 3],
        "Number of transfered embryos": [1, 1, 2, 1],
        "Pregnancy": ["Yes", "No", "1 No, 1 Yes", "Biochemical, No"],
        "Fertilization Method": ["ICSI", "ICSI", "IVF", "ICSI"],
        "Sperm Source": ["Own Fresh", "Own Fresh", "Own Frozen", "Own Fresh"],
        "Doctor": ["Dr A", "Dr A", "Dr B", "Dr A"],
        "Embryologist IVF/ICSI": ["Emb X", "Emb Y", "Emb X", "Emb Y"],
        "Cryo: Day 5 Embryos": [1, 1, 0, 2],
        "Cryo: Day 6 Embryos": [0, 0, 0, 1],
        "Cryo: Day 7 Embryos": [0, 0, 0, 0],
        "Cryo: Other Blastocysts": [0, 0, 0, 0],
        "Biopsy Done": [2, 0, 1, 3],
        "Tested Embryos": [2, 0, 1, 3],
        "Not Tested Embryos": [0, 0, 0, 0],
        "Euploid": [1, 0, 1, 2],
        "Mosaic": [0, 0, 0, 0],
        "Failed/NA": [0, 0, 0, 0],
        "Pending Result": [0, 0, 0, 0],
        "Aneuploid": [1, 0, 0, 1],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _wide_raw(**overrides) -> pd.DataFrame:
    base = {
        "OPU Code": ["T001", "T002", "T003", "T004", "W005"],
        "Patients name": ["Test Patient A", "Test Patient B", "Test Patient C", "Test Patient D", "Test Patient E"],
        "Age": [28, 32, 35, 30, 29],
        "OPU Date": pd.to_datetime(["2026-01-10", "2026-01-11", "2026-01-12", "2026-01-13", "2026-01-14"]),
        "Oocytes N": [10, 8, 6, 12, 9],
        "MII": [8, 6, 5, 10, 7],
        "2pN": [6, 4, 3, 8, 5],
        "3pN or more": [0, 1, 0, 0, 0],
        "Degrade": [0, 0, 1, 0, 0],
        "D5-D6-D7": [3, 2, 1, 4, 2],
        "Good quality BL": [2, 1, 0, 3, 1],
        "Fertilization Method": ["ICSI", "ICSI", "IVF", "ICSI", "ICSI"],
        "Sperm Source": ["Own Fresh", "Own Fresh", "Own Frozen", "Own Fresh", "Own Fresh"],
        "Doctor": ["Dr A", "Dr A", "Dr B", "Dr A", "Dr B"],
        "Embryologist IVF/ICSI": ["Emb X", "Emb Y", "Emb X", "Emb Y", "Emb X"],
        "Cryo: Day 5 Embryos": [1, 1, 0, 2, 0],
        "Cryo: Day 6 Embryos": [0, 0, 0, 1, 1],
        "Cryo: Day 7 Embryos": [0, 0, 0, 0, 0],
        "Cryo: Other Blastocysts": [0, 0, 0, 0, 0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _prepared_transfer():
    df, unknown = ggrc.prepare(_transfer_raw(), COLUMNS_CFG, SETTINGS_CFG)
    return df


def _prepared_wide():
    df, unknown = ggrc.prepare_wide(_wide_raw(), COLUMNS_CFG, SETTINGS_CFG)
    return df


def test_ggrc_registered_in_clinic_dispatch_tables():
    assert _CLINIC_NORMALIZERS["ggrc"] is ggrc.prepare
    assert _CLINIC_PGT_MODULES["ggrc"] is ggrc
    assert _CLINIC_WIDE_LOADERS["ggrc"] is ggrc.prepare_wide
    assert _CLINIC_PGT_EXTRA_COLUMNS["ggrc"] == ggrc.PGT_RAW_COLUMNS


@pytest.mark.parametrize("value,expected", [
    ("Yes", 1.0), ("No", 0.0), ("1 No, 1 Yes", 1.0), ("No, No", 0.0),
    ("No, No, Yes", 1.0), ("Biochemical, No", 0.0), ("Biochemical", 0.0),
    ("", None), (None, None),
])
def test_outcome_composite_token_classification(value, expected):
    result = ggrc._classify_outcome(value)
    if expected is None:
        assert pd.isna(result)
    else:
        assert result == expected


def test_outcome_parsed_on_full_cohort():
    df = _prepared_transfer()
    got = df.set_index("patient_id")["Preg"].to_dict()
    assert got == {"T001": 1.0, "T002": 0.0, "T003": 1.0, "T004": 0.0}


def test_follicles_is_estimated_from_occ():
    df = _prepared_transfer()
    row = df.loc[df["patient_id"] == "T001"].iloc[0]
    assert row["follicles"] == round(10 * 1.44)


def test_frozen_sums_cryo_columns():
    df = _prepared_transfer()
    row = df.loc[df["patient_id"] == "T004"].iloc[0]
    assert row["frozen"] == 2 + 1 + 0 + 0  # Cryo Day5+Day6+Day7+Other


def test_patient_name_fallback_uses_opu_code():
    df = _prepared_transfer()
    row = df.loc[df["patient_id"] == "T002"].iloc[0]
    assert row["patient_name"] == "GGRC-T002"
    assert not df["patient_name"].isna().any()


def test_attempt_and_transfer_day_defaults():
    df = _prepared_transfer()
    row = df.loc[df["patient_id"] == "T004"].iloc[0]
    assert row["attempt_no"] == 1     # missing -> default 1
    assert (df["transfer_day"] == 5).all()  # no such column at all -> constant 5


def test_missing_required_column_raises():
    raw = _transfer_raw().drop(columns=["Oocytes N"])
    with pytest.raises(ValueError):
        ggrc.prepare(raw, COLUMNS_CFG, SETTINGS_CFG)


def test_prepare_wide_has_no_outcome_column_and_no_row_filtering():
    df = _prepared_wide()
    assert "Preg" not in df.columns
    assert len(df) == 5  # every lab-evaluable row kept, including W005 (no matching transfer row)


def test_prepare_wide_bl_total_resolves_from_d5_d6_d7_alias():
    df = _prepared_wide()
    assert not df["bl_total"].isna().any()
    row = df.loc[df["patient_id"] == "T001"].iloc[0]
    assert row["bl_total"] == 3


def test_prepare_wide_attempt_no_defaults_to_one_when_column_absent():
    df = _prepared_wide()
    assert (df["attempt_no"] == 1).all()


def test_cycle_uid_key_consistent_between_transfer_and_wide_files():
    """GGRC has no Skyfert-style custom build_cycle_key: the generic
    patient_id + date_opu keying in _add_opu_aliases must line up between
    ggrc.prepare() (eligible cohort) and ggrc.prepare_wide() (wide lab
    population, a SEPARATE file) for the same physical cycle, since
    analytics._restore_true_counts/_lab_kpi_slices join them by cycle_uid."""
    transfer_keyed = _add_opu_aliases(_prepared_transfer(), clinic="ggrc")
    wide_keyed = _add_opu_aliases(_prepared_wide(), clinic="ggrc")

    t_keys = transfer_keyed.set_index("patient_id")["_key"]
    w_keys = wide_keyed.set_index("patient_id")["_key"]
    shared = t_keys.index.intersection(w_keys.index)
    assert len(shared) == 4  # T001..T004
    pd.testing.assert_series_equal(t_keys.loc[shared], w_keys.loc[shared], check_names=False)


def test_analyze_reports_builds_summary_from_embedded_columns():
    df = _prepared_transfer()
    df["cycle_uid"] = [f"uid_{i}" for i in range(len(df))]
    summary = ggrc.analyze_reports(df, reports_dir="unused")

    row = summary.loc[summary["cycle_uid"] == "uid_0"].iloc[0]
    assert row["n_biopsied"] == 2
    assert row["n_euploid"] == 1
    assert row["n_aneuploid"] == 1
    assert row["euploid_rate"] == 0.5

    zero_row = summary.loc[summary["cycle_uid"] == "uid_1"].iloc[0]
    assert zero_row["n_biopsied"] == 0
    assert pd.isna(zero_row["euploid_rate"])
    assert zero_row["pgt_note"] == "биопсия не проводилась / нет данных ПГТ-А"

    assert summary.attrs["events"].empty
    assert summary.attrs["unmatched"] == []


def test_analyze_reports_requires_cycle_uid_column():
    df = _prepared_transfer()
    with pytest.raises(KeyError):
        ggrc.analyze_reports(df, reports_dir="unused")
