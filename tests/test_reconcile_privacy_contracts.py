import json

import pandas as pd
import pytest

from embryo.contracts import TABPFN_OUTPUT
from embryo.privacy import (build_crosswalk, deidentify_records,
                            write_agent_bundle)
from embryo.reconcile import (build_dt_predictions, enrich_opu_table,
                              write_reconciled_outputs)


def _tables():
    dt = pd.DataFrame({
        "_key": ["ab-1", "AB-2", "AB-3"],
        "Preg": [0, 1, pd.NA],
        "DIGITAL TWIN": [0.0, 0.7, 0.3],
    })
    tab = pd.DataFrame({
        "patient_id": ["AB-1", "ab-2"],
        "p_tabpfn": [0.0, 0.5],
        "ci_low": [0.0, 0.30],
        "ci_high": [0.0, 0.55],
    })
    opu = pd.DataFrame({
        "ID": ["AB-1", "AB-2", "AB-3"],
        "Patient Full Name": ["One", "Two", "Three"],
        "Preg": [1, 0, 0],
        "Date OPU": ["2026-01-01", "2026-01-02", "2026-01-03"],
    })
    return dt, tab, opu


def test_real_join_preserves_zero_and_fills_only_missing_outcome():
    dt, tab, opu = _tables()
    out = build_dt_predictions(dt, tab, opu, b"secret", threshold_pp=20)

    first = out.loc[out["patient_id"] == "ab-1"].iloc[0]
    assert first["Preg"] == 0                 # DT zero is known; OPU 1 cannot replace it
    assert first["PRAI"] == 0                 # probability zero is not missing
    assert first["DIGITAL TWIN"] == 0
    assert first["PRAI CI (TabPFN), %"] == "0.0–0.0%"
    assert not bool(first["ci_wide"])

    second = out.loc[out["patient_id"] == "AB-2"].iloc[0]
    assert second["ci_width_pp"] == pytest.approx(25)
    assert bool(second["ci_wide"])

    third = out.loc[out["patient_id"] == "AB-3"].iloc[0]
    assert third["Preg"] == 0                  # true missing DT value is filled from OPU
    assert third["tabpfn_join_status"] == "missing"
    assert pd.isna(third["PRAI"])
    assert out["cycle_uid"].str.match(r"^CYC-[0-9A-F]{16}$").all()


def test_reverse_join_returns_predictions_to_the_local_opu_table():
    dt, tab, opu = _tables()
    pred = build_dt_predictions(dt, tab, opu, b"secret", threshold_pp=20)
    enriched = enrich_opu_table(opu, pred)
    assert enriched.loc[enriched["ID"] == "AB-1", "PRAI"].iloc[0] == 0
    assert "cycle_uid" in enriched


def test_atomic_reconciled_outputs(tmp_path):
    dt, tab, opu = _tables()
    pred = build_dt_predictions(dt, tab, opu, b"secret")
    enriched = enrich_opu_table(opu, pred)
    paths = write_reconciled_outputs(
        pred, enriched, tmp_path / "dt_predictions.csv",
        tmp_path / "OPU_enriched.xlsx", tmp_path / "cycle_crosswalk.csv",
    )
    assert all(path.exists() for path in paths.values())
    assert set(pd.read_csv(paths["crosswalk"]).columns) >= {"cycle_uid", "patient_id"}


def test_agent_packet_has_allowlist_and_crosswalk_stays_separate(tmp_path):
    dt, tab, opu = _tables()
    pred = build_dt_predictions(dt, tab, opu, b"secret")
    safe = deidentify_records(pred, ["cycle_uid", "Preg", "PRAI", "ci_wide"])
    crosswalk = build_crosswalk(pred, "patient_id")
    packet = tmp_path / "agent.json"
    local = tmp_path / "local_crosswalk.csv"
    write_agent_bundle(safe, {"n_cycles": len(safe)}, packet, crosswalk, local)
    payload = json.loads(packet.read_text(encoding="utf-8"))
    assert set(payload["patients"][0]) == {"cycle_uid", "Preg", "PRAI", "ci_wide"}
    assert "patient_id" not in packet.read_text(encoding="utf-8")
    assert "patient_id" in pd.read_csv(local).columns

    with pytest.raises(ValueError, match="PHI"):
        deidentify_records(pred, ["patient_id", "Preg"])
    with pytest.raises(ValueError, match="PHI"):
        write_agent_bundle(pred[["cycle_uid", "patient_id"]], {}, packet)


def test_contract_rejects_duplicate_and_bad_ci_probability():
    bad = pd.DataFrame({
        "patient_id": ["A", "a"], "p_tabpfn": [0.2, 1.2],
        "ci_low": [0.1, 0.1], "ci_high": [0.3, 0.3],
    })
    with pytest.raises(ValueError):
        TABPFN_OUTPUT.validate(bad)

