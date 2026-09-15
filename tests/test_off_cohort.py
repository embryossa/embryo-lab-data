"""Прогноз для циклов вне когорты отчёта (embryo.off_cohort + Teable/карточки).

Решение владельца 2026-09-15: 2PN>0 — все модели; OCC>0 при 2PN=0 — только воронка
DT без вероятности; OCC=0 — модели не применяются. Слой не должен смешиваться с
когортой отчёта ни в карточках, ни в базе.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "teable"))

from embryo import off_cohort  # noqa: E402

import cards_build  # noqa: E402
import teable_common as tc  # noqa: E402
import teable_sync  # noqa: E402

from test_teable_sync_run import Args, RecordingTeable, _cfg  # noqa: E402
import teable_bootstrap as bootstrap  # noqa: E402


def _cycles() -> pd.DataFrame:
    return pd.DataFrame({
        "cycle_uid": ["CYC-R", "CYC-A", "CYC-B", "CYC-C", "CYC-D", "CYC-E"],
        "_key": ["R#1", "A#1", "B#1", "C#1", "D#1", "E#1"],
        "occ":         [8, 10, 6, 0, 7, 9],
        "two_pn":      [5, 4, 0, 0, 3, 5],
        "transferred": [1, np.nan, np.nan, np.nan, 1, 1],
        "Preg":        [1, np.nan, np.nan, np.nan, np.nan, 0],
    })


def test_classify_splits_tiers_by_lab_data_and_skips_reported_cycles():
    out = off_cohort.classify(_cycles(), reported_uids={"CYC-R"}).set_index("cycle_uid")
    assert "CYC-R" not in out.index
    assert out.loc["CYC-A", ["model_tier", "cohort_scope"]].tolist() == ["full", "no_transfer"]
    assert out.loc["CYC-B", ["model_tier", "cohort_scope"]].tolist() == ["funnel", "no_transfer_no_fertilization"]
    # Ноль — известное наблюдение: OCC=0 не «нет данных», а «ооциты не получены».
    assert out.loc["CYC-C", ["model_tier", "cohort_scope"]].tolist() == ["none", "no_oocytes"]
    assert out.loc["CYC-D", "cohort_scope"] == "transfer_pending_outcome"
    assert out.loc["CYC-E", "cohort_scope"] == "cohort_not_in_last_run"
    assert out["cohort_scope_label"].notna().all()


def test_funnel_tier_keeps_funnel_and_drops_every_pregnancy_probability():
    dt = pd.DataFrame({"_key": ["B#1"], "cycle_uid": ["CYC-B"], "DIGITAL TWIN": [0.3],
                       "clinical_probability": [0.3], "p_per_transfer": [0.2], "bayes_mean": [0.25],
                       "p_kat_raw": [0.4], "banking_p_per_mii": [0.05],
                       "med_mii": [5.0], "p025_blasts": [0.0], "p_no_blast": [0.4],
                       "dominant_cluster": [1], "ohss_moderate": [0.1]})
    out = off_cohort.strip_to_funnel(dt).iloc[0]
    for column in ("DIGITAL TWIN", "clinical_probability", "p_per_transfer", "bayes_mean",
                   "p_kat_raw", "banking_p_per_mii"):
        assert pd.isna(out[column]), column
    assert out["med_mii"] == 5.0 and out["p_no_blast"] == 0.4 and out["dominant_cluster"] == 1


def test_restore_facts_takes_true_counts_from_the_opu_table():
    cycles = off_cohort.classify(_cycles().assign(mii=[7, 9, 5, 0, 6, 8]), reported_uids=set())
    pred = pd.DataFrame({"cycle_uid": ["CYC-C"], "OCC": [np.nan]})
    out = off_cohort.restore_facts(pred, cycles).iloc[0]
    assert out["OCC"] == 0 and out["MII"] == 0
    assert out["model_tier"] == "none"


def test_dnn_input_reads_blank_et_as_zero_and_keeps_real_transfers(tmp_path):
    source = tmp_path / "model_input.xlsx"
    pd.DataFrame({"cycle_uid": ["CYC-A", "CYC-D"], "Перенесено эмбрионов": [np.nan, 2]}).to_excel(source, index=False)
    path, n = off_cohort.dnn_input_with_zero_transfer(source, tmp_path)
    out = pd.read_excel(path)
    assert n == 1 and path != source
    assert out["Перенесено эмбрионов"].tolist() == [0, 2]
    assert out["et_zero_for_dnn"].tolist() == [True, False]
    assert pd.isna(pd.read_excel(source).loc[0, "Перенесено эмбрионов"]), "исходный вход не трогается"


def _write_runs(root: Path) -> Path:
    main = root / "processed"
    (main / "off_cohort").mkdir(parents=True)
    pd.DataFrame({"cycle_uid": ["CYC-R"], "patient_id": ["R#1"], "Preg": [1], "PRAI": [0.6],
                  "DIGITAL TWIN": [0.55], "OCC": [8], "Date OPU": ["2026-01-05"]}
                 ).to_csv(main / "dt_predictions.csv", index=False)
    pd.DataFrame({
        "cycle_uid": ["CYC-A", "CYC-C"], "patient_id": ["A#1", "C#1"], "Preg": [np.nan, np.nan],
        "PRAI": [0.4, np.nan], "DIGITAL TWIN": [0.45, np.nan], "med_mii": [8.0, np.nan],
        "OCC": [10, 0], "Date OPU": ["2026-02-01", "2026-02-02"],
        "model_tier": ["full", "none"], "cohort_scope": ["no_transfer", "no_oocytes"],
        "cohort_scope_label": [off_cohort.SCOPE_LABELS["no_transfer"], off_cohort.SCOPE_LABELS["no_oocytes"]],
    }).to_csv(main / "off_cohort" / "dt_predictions.csv", index=False)
    (main / "off_cohort" / "run_manifest.json").write_text('{"scope": "off_cohort"}', encoding="utf-8")
    return main


def test_supplemental_run_is_synced_as_its_own_run_without_empty_predictions(tmp_path):
    main = _write_runs(tmp_path)
    tb = RecordingTeable()
    bootstrap.apply_schema(tb, "bse1", "Asia/Tashkent")
    clinic = tc.Clinic({"code": "test", "name": "Тест", "base_id": "bse1", "processed_dir": str(main)}, {})
    assert clinic.supplemental_dirs == [str(main / "off_cohort")]

    teable_sync.sync_clinic(_cfg(), tb, clinic, Args())
    runs = tb.records["Runs"]
    assert len(runs) == 2 and "off_cohort" in runs[1]["Run ID"]
    keys = [r["Ключ"].split("::")[0] for r in tb.records["Predictions"]]
    assert sorted(keys) == ["CYC-A", "CYC-R"], "у цикла без моделей не должно быть строки прогноза"
    labels = {r["cycle_uid"]: r.get("Охват прогноза") for r in tb.records["Cycles"]}
    assert labels["CYC-R"] == "Когорта отчёта"
    assert labels["CYC-C"] == off_cohort.SCOPE_LABELS["no_oocytes"]
    no_model = next(r for r in tb.records["Cycles"] if r["cycle_uid"] == "CYC-C")
    assert "Последний прогон" not in no_model


def test_cards_mark_off_cohort_cycles_and_do_not_duplicate_archive_rows(tmp_path):
    main = _write_runs(tmp_path)
    clinic = tc.Clinic({"code": "test", "name": "Тест", "base_id": "", "processed_dir": str(main)}, {})
    # Строка базы, заведённая OPU-таблицей без cycle_uid, лежит в архиве под join-ключом.
    archive = {"A#1": {"uid": "A#1", "key": "A#1", "opu": "2026-02-01", "name": "—"}}
    payload = cards_build.build_payload(clinic, archive=archive)
    by_uid = {c["uid"]: c for c in payload["cycles"]}
    assert set(by_uid) == {"CYC-R", "CYC-A", "CYC-C"} and payload["n_archived"] == 0
    assert by_uid["CYC-R"]["scope"] is None
    assert by_uid["CYC-A"]["tier"] == "full" and by_uid["CYC-A"]["src"] == "off_cohort"
    assert payload["n_off"] == 2
