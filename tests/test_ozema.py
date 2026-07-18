from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embryo.ozema import build_ozema_table, genetic_counseling_candidates


_CFG = {
    "competency_exclusion": {
        "no_retrieval_tokens": ["NO"],
        "freeze_all_tokens": ["Cryo"],
        "surgical_sperm_tokens": ["TESE", "SMF"],
    }
}


def _row(**overrides) -> dict:
    base = {
        "cycle_uid": "cy1", "patient_id": "P1", "patient_name": "Patient One",
        "date_opu": "2026-01-01", "age": 32, "occ": 10, "mii": 8, "two_pn": 6,
        "cleavage_d3": 5, "bl_total": 3, "inseminated": 8,
        "embryologist": "A", "doctor": "X", "sperm_source": None,
    }
    base.update(overrides)
    return base


def test_underpowered_stage_is_marked_not_available_not_a_pass():
    frame = pd.DataFrame([_row(occ=2, mii=1, two_pn=1, cleavage_d3=1, bl_total=1, inseminated=1)])
    out = build_ozema_table(frame, _CFG)
    # Deliberately not the literal "n/a": that token is in pandas' default
    # read_csv na_values and would silently become real NaN on a later re-read.
    assert out.loc[0, "ozema_flag"] == "н/д"
    assert out.loc[0, "OMA_result"] == "н/д"
    assert not out.loc[0, "flagged"]


def test_clearly_below_threshold_with_enough_n_is_flagged():
    # 2PN/MII = 0/10 -- unambiguously below the 60% FF threshold with n=10.
    frame = pd.DataFrame([_row(mii=10, two_pn=0, inseminated=10, cleavage_d3=0, bl_total=0)])
    out = build_ozema_table(frame, _CFG)
    assert "FF" in out.loc[0, "ozema_flag"]
    assert out.loc[0, "flagged"]


def test_clearly_above_threshold_is_unflagged_dash():
    # every stage comfortably above its threshold with plenty of n
    frame = pd.DataFrame([_row(occ=20, mii=19, two_pn=18, cleavage_d3=18, bl_total=15, inseminated=19)])
    out = build_ozema_table(frame, _CFG)
    assert out.loc[0, "ozema_flag"] == "—"
    assert not out.loc[0, "flagged"]


def test_ff_excludes_surgical_sperm_rows_only():
    # Same unambiguous 2PN=0/MII=10 fertilization-failure signal, but surgical sperm
    # retrieval (TESE) must exclude it from FF specifically -- other stages still test.
    frame = pd.DataFrame([_row(mii=10, two_pn=0, inseminated=10, cleavage_d3=0, bl_total=0,
                               sperm_source="TESE")])
    out = build_ozema_table(frame, _CFG)
    assert "FF" not in out.loc[0, "ozema_flag"]
    assert out.loc[0, "FF_result"] == "excluded"


def test_genetic_counseling_recurrence_requires_at_least_two_flagged_cycles():
    flagged_row = dict(mii=10, two_pn=0, inseminated=10, cleavage_d3=0, bl_total=0)
    frame = pd.DataFrame([
        _row(cycle_uid="cy1", patient_id="P1", **flagged_row),
        _row(cycle_uid="cy2", patient_id="P1", **flagged_row),
        _row(cycle_uid="cy3", patient_id="P2", **flagged_row),
    ])
    out = build_ozema_table(frame, _CFG)
    assert out.loc[out["patient_id"] == "P1", "flagged_cycles_n"].unique().tolist() == [2]
    assert out.loc[out["patient_id"] == "P2", "flagged_cycles_n"].unique().tolist() == [1]

    candidates = genetic_counseling_candidates(out)
    assert set(candidates["patient_id"]) == {"P1"}
    assert "P2" not in set(candidates["patient_id"])


def test_freeze_all_cycle_can_still_be_flagged_no_population_exclusion():
    # Unlike the embryologist/doctor competency tables, OZEMA does NOT drop
    # freeze-all/NO-retrieval/open-cycle rows -- MIN_N alone governs testability.
    frame = pd.DataFrame([_row(mii=10, two_pn=0, inseminated=10, cleavage_d3=0, bl_total=None)])
    out = build_ozema_table(frame, _CFG)
    assert "FF" in out.loc[0, "ozema_flag"]
