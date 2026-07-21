"""Синтетические фикстуры и тесты для Skyfert (см. src/embryo/clinics/skyfert.py
и docs/SKYFERT.md, "Полный прогон"). Покрывает как ingest/normalize (skyfert.prepare),
так и клинико-специфичные ветки scripts/run_analysis.py, добавленные для
--clinic skyfert (регистрация в _CLINIC_NORMALIZERS, cycle-key через
build_cycle_key вместо ID+Date OPU, placeholder ФИО=ЭМК для внешнего DT)."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

from embryo import ingest
from embryo.clinics import skyfert

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_analysis import _CLINIC_NORMALIZERS, _add_dt_aliases, _add_opu_aliases  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COLUMNS_CFG = ingest.load_config(str(ROOT / "config/clinics/skyfert/columns.yaml"))
SETTINGS_CFG = {
    "amh_impute": {"a_afc": 0.16, "b_occ": 0.05, "c_age": 0.08, "intercept": 0.5, "floor": 0.1, "ceil": 12.0},
    "bmi_impute": {"constant": 24.0},
}


def _raw_rows(**overrides) -> pd.DataFrame:
    base = {
        # "5822" repeats across physically different patients in the real export
        # (documented collision: ages 34/42/44/.../43 within one ЭМК) — mirrored
        # here on two rows with different ages to exercise build_cycle_key.
        "ЭМК": ["5822", "5822", "9001", "9002", "9003"],
        "Возраст": [34, 42, 28, 31, 27],
        "№ попытки": [1, 1, 1, None, 1],
        "ЭКО": ["ИКСИ", "ЭКО", "ИКСИ", "ИКСИ", None],
        "Количество фолликулов": [12, 10, 8, None, 7],
        "Число ОКК": [10, 8, 6, 5, 5],
        "Число инсеминированных": [8, 6, 4, 4, 4],
        "2 pN": [6, 4, 3, 3, 3],
        "Число дробящихся на 3 день": [5, 3, 2, 2, 2],
        "Число эмбрионов 5 дня": [4, 2, 2, 2, 1],
        "Число Bl": [3, 2, 1, 1, 1],
        "Число Bl хор.кач-ва": [2, 1, 1, 1, 1],
        "День переноса": [5, 5, 5, None, 5],
        "Исход переноса": [
            "Беременность клиническая", "Беременность не наступила",
            "Беременность не наступила", "Беременность не наступила", None,
        ],
        "Перенесено эмбрионов": [1, 1, 1, 1, 0],
        "Заморожено эмбрионов": [0, 0, 0, 0, 2],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _prepared():
    raw = _raw_rows()
    df, unknown = skyfert.prepare(raw, COLUMNS_CFG, SETTINGS_CFG)
    return df


def test_skyfert_registered_as_clinic_normalizer():
    assert _CLINIC_NORMALIZERS["skyfert"] is skyfert.prepare


def test_follicles_fallback_and_defaults():
    df = _prepared()
    row = df.loc[df["patient_id"] == "9002"].iloc[0]
    assert row["follicles"] == row["occ"] == 5    # follicles missing -> falls back to occ
    assert row["attempt_no"] == 1                 # missing -> default 1
    assert row["transfer_day"] == 5                # missing -> default 5


def test_outcome_and_amh_bmi_imputed():
    df = _prepared()
    assert df.loc[df["patient_id"] == "5822", "Preg"].iloc[0] == 1.0
    assert df.loc[df["patient_id"] == "5822", "Preg"].iloc[1] == 0.0
    # Skyfert's export has no AMH/BMI columns at all -> generic normalize.normalize()
    # must impute both for every row (docs/SKYFERT.md).
    assert df["amh_imputed"].all()
    assert df["bmi_imputed"].all()
    assert df["bmi"].eq(24.0).all()


def test_missing_required_column_raises():
    raw = _raw_rows().drop(columns=["Число ОКК"])
    with pytest.raises(ValueError):
        skyfert.prepare(raw, COLUMNS_CFG, SETTINGS_CFG)


def test_add_opu_aliases_uses_build_cycle_key_not_bare_id_for_skyfert():
    df = _prepared()
    out = _add_opu_aliases(df, clinic="skyfert")
    expected = skyfert.build_cycle_key(df)
    assert out["_key"].equals(expected)
    # The two ЭМК-5822 rows must NOT collapse to the same _key (unlike the
    # generic "elif 'ID' in out.columns: out['_key'] = out['ID']" fallback,
    # which would give both the bare "5822").
    fives = out.loc[out["ID"] == "5822", "_key"]
    assert fives.nunique() == 2


def test_cycle_key_consistent_across_full_population_and_eligible_subset():
    """Regression test for the bug fixed alongside wiring Skyfert into
    run_analysis.py: build_cycle_key must use df.index (stable, inherited from
    the one-time RangeIndex assigned at ingest.load_table), not a fresh
    positional np.arange(len(df)) recomputed on whatever slice it is handed —
    otherwise the same physical cycle gets a different _key depending on
    whether _add_opu_aliases(clinic="skyfert") is called on the full lab
    population (lab_cycles) or the transferred>0 & Preg.notna() eligible
    subset (kpi_cycles), breaking OZEMA <-> KPI/DT/TabPFN cross-referencing."""
    normalized = _prepared()
    lab_cycles = _add_opu_aliases(normalized, clinic="skyfert")

    transferred = pd.to_numeric(normalized["transferred"], errors="coerce")
    eligible_mask = transferred.gt(0) & normalized["Preg"].notna()
    eligible = normalized.loc[eligible_mask].copy()
    assert len(eligible) < len(normalized)  # the fixture must actually exercise a real subset
    kpi_cycles = _add_opu_aliases(eligible, clinic="skyfert")

    shared_index = kpi_cycles.index
    pd.testing.assert_series_equal(
        kpi_cycles.loc[shared_index, "_key"], lab_cycles.loc[shared_index, "_key"])


def test_dt_fio_placeholder_uses_patient_id_via_generic_alias():
    """Mirrors the `if args.clinic == "skyfert": kpi_cycles["Patient Full Name"] =
    kpi_cycles["patient_id"]` line in run_analysis.py:main() — verifies it rides
    on the existing generic _DT_ALIASES mapping ("Patient Full Name" -> "ФИО")
    without needing a Skyfert-specific alias function."""
    df = _prepared()
    df["Patient Full Name"] = df["patient_id"]
    out = _add_dt_aliases(df)
    assert (out["ФИО"] == out["patient_id"]).all()
    assert not (out["ФИО"].astype(str).str.strip() == "").any()
