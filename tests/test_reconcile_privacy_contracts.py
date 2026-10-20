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


def test_manual_opu_probabilities_never_survive_the_pipeline():
    """PRAI/DIGITAL TWIN typed into the OPU master are ignored even when a model gave nothing."""
    from embryo.analytics import build_merged
    dt, tab, opu = _tables()
    opu = opu.assign(**{"PRAI": [0.9, 0.9, 0.9], "DIGITAL TWIN": [0.8, 0.8, 0.8],
                        "PRAI CI (TabPFN), %": ["old", "old", "old"]})
    pred = build_dt_predictions(dt, tab, opu, b"secret", threshold_pp=20)
    enriched = enrich_opu_table(opu, pred)
    third = enriched.loc[enriched["ID"] == "AB-3"].iloc[0]   # TabPFN did not score AB-3
    assert pd.isna(third["PRAI"]) and pd.isna(third["PRAI CI (TabPFN), %"])
    assert enriched.loc[enriched["ID"] == "AB-2", "DIGITAL TWIN"].iloc[0] == pytest.approx(0.7)
    assert list(enriched.columns[:4]) == list(opu.columns[:4])   # sheet layout kept
    _, df_u, _ = build_merged(pred.assign(_key=pred["patient_id"]), enriched.assign(_key=enriched["ID"]))
    assert not [c for c in df_u.columns if c.endswith("_opu") and c.split("_opu")[0] in
                ("PRAI", "DIGITAL TWIN", "PRAI CI (TabPFN), %", "ci_width_pp", "ci_wide", "cycle_uid")]
    assert pd.isna(df_u.loc[df_u["patient_id"] == "AB-3", "PRAI"].iloc[0])


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



def test_tabpfn_ensemble_spread_is_carried_separately_from_funnel_range():
    # ci_low/ci_high — диапазон по неизвестным стадиям воронки (0 при полной воронке);
    # разброс членов ансамбля TabPFN приходит отдельными колонками ens_* (адаптер 2026-10-20).
    dt, tab, opu = _tables()
    tab = tab.assign(ens_low=[0.0, 0.40], ens_high=[0.0, 0.62], ens_sd=[0.0, 0.08], n_members=[8, 8])
    out = build_dt_predictions(dt, tab, opu, b"secret", threshold_pp=20).set_index("patient_id")
    assert out.loc["AB-2", "tabpfn_ens_width_pp"] == pytest.approx(22)
    assert out.loc["AB-2", "ci_width_pp"] == pytest.approx(25)
    assert out.loc["AB-2", "tabpfn_n_members"] == 8
    assert pd.isna(out.loc["AB-3", "tabpfn_ens_low"])


def test_tabpfn_output_without_ensemble_columns_still_joins():
    dt, tab, opu = _tables()
    out = build_dt_predictions(dt, tab, opu, b"secret", threshold_pp=20)
    assert out["tabpfn_ens_width_pp"].isna().all()
