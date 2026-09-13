"""Digital Twin 7.1 через REST API: сопоставление входа, подстановки с флагами, разбор ответа.

Служба в тестах подменяется объектом с тем же интерфейсом, что DTApiClient. Проверяется
наш слой: что уходит в службу (без ID/ФИО/дат), как строки ответа ложатся в контракт
DT_OUTPUT и почему воронка «прогноз против факта» берётся из точки после пункции.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from embryo import analytics, contracts, dt_api, reconcile


def _source() -> pd.DataFrame:
    return pd.DataFrame({
        "_key": ["P1#20260301", "P2#20260302", "P3#20260303"],
        "cycle_uid": ["CYC-AAAA", "CYC-BBBB", "CYC-CCCC"],
        "patient_name": ["Имя Фамилия", "Другая", "Третья"],
        "age": [30, 41, 35], "amh": [3.1, 0.9, 2.0], "amh_imputed": [False, True, True],
        "bmi": [22.0, 24.0, 24.0], "bmi_imputed": [False, True, True],
        "afc": [14, np.nan, np.nan], "follicles": [15, 6, np.nan], "attempt_no": [1, 2, np.nan],
        "sperm_source": ["N", "SMF", None],
        "occ": [12, 5, 8], "mii": [10, 4, 7], "inseminated": [10, 4, 7], "two_pn": [8, 3, 9],
        "cleavage_d3": [8, 3, 6], "bl_total": [4, 1, 3], "bl_good": [3, 0, 2], "frozen": [3, 0, 2],
        "Preg": [1, 0, 1],
    })


API_CFG = {"n_sim": 800, "impute": {"afc_constant": 12}}


def test_patient_mapping_and_explicit_imputation_flags():
    src = _source()
    first, flags = dt_api.build_patient(src.iloc[0], "transfer", API_CFG)
    assert first == {"age": 30.0, "amh": 3.1, "afc": 14, "bmi": 22.0, "attempt": 1, "sperm_source": "ejaculate",
                     "follicles": 15, "known_okk": 12, "known_mii": 10, "known_pn2": 8,
                     "known_blasts": 4, "known_good": 3}
    assert flags == {"amh_imputed": False, "bmi_imputed": False, "afc_source": "afc",
                     "attempt_imputed": False, "sperm_source_inferred": False, "okk_raised_for_thawed": False}

    second, flags = dt_api.build_patient(src.iloc[1], "retrieval", API_CFG)
    assert second["afc"] == 6 and flags["afc_source"] == "follicles"
    assert second["sperm_source"] == "testicular_OA" and flags["sperm_source_inferred"] is True
    assert set(k for k in second if k.startswith("known_")) == {"known_okk", "known_mii"}

    third, flags = dt_api.build_patient(src.iloc[2], "transfer", API_CFG)
    assert third["afc"] == 12 and flags["afc_source"] == "constant"
    assert third["attempt"] == 1 and flags["attempt_imputed"] is True
    assert third["sperm_source"] == "ejaculate" and flags["sperm_source_inferred"] is True
    assert "follicles" not in third  # неизвестно -> не передаётся, а не 0


def test_zero_stage_count_is_sent_as_observed_zero():
    src = _source()
    patient, _ = dt_api.build_patient(src.iloc[1], "transfer", API_CFG)
    assert patient["known_good"] == 0


def test_batches_carry_no_identifiers_and_two_timepoints():
    bodies, flags = dt_api.build_batches(_source(), API_CFG)
    assert len(bodies) == 1 and bodies[0]["options"] == {"n_sim": 800}
    ids = [c["cycle_id"] for c in bodies[0]["cycles"]]
    assert ids[:2] == ["CYC-AAAA.transfer", "CYC-AAAA.retrieval"] and len(ids) == 6
    raw = json.dumps(bodies, ensure_ascii=False)
    for private in ("P1#", "20260301", "Имя", "Фамилия"):
        assert private not in raw
    assert list(flags["cycle_uid"]) == ["CYC-AAAA", "CYC-BBBB", "CYC-CCCC"]


def test_large_cohort_is_split_into_api_sized_batches():
    src = pd.concat([_source()] * 100, ignore_index=True)
    src["cycle_uid"] = [f"CYC-{i:05d}" for i in range(len(src))]
    bodies, _ = dt_api.build_batches(src, API_CFG)
    assert [len(b["cycles"]) for b in bodies] == [500, 100]


COLUMNS = ["record_id", "timestamp", "patient_id", "age", "known_blasts", "med_okk", "med_mii",
           "med_pn2", "med_blasts", "med_good", "p025_blasts", "p975_blasts", "p_per_transfer",
           "p_overall_cycle", "bayes_mean", "p_kat_raw", "p_nvsa", "p_csdi", "dominant_cluster",
           "clinical_probability", "befe_per_transfer", "cycle_probability", "gnn_prob",
           "gnn_ensemble_prob", "gnn_weight", "layer_L5", "layer_L6", "befe_ci_low", "befe_ci_high",
           "befe_reliability"]


def _values(uid: str, timepoint: str, i: int) -> dict:
    transfer = timepoint == "transfer"
    return {"record_id": None, "timestamp": "2026-09-13T10:00:00", "patient_id": f"{uid}.{timepoint}",
            "age": 30 + i, "known_blasts": 4 if transfer else None, "med_okk": 12.0, "med_mii": 10.0,
            "med_pn2": 8.0 if transfer else 7.0, "med_blasts": 4.0 if transfer else 5.0,
            "med_good": 3.0 if transfer else 2.5, "p025_blasts": 1, "p975_blasts": 9,
            "p_per_transfer": .41, "p_overall_cycle": .7, "bayes_mean": .44, "p_kat_raw": .5,
            "p_nvsa": .48, "p_csdi": .46, "dominant_cluster": 0,
            "clinical_probability": .52 if transfer else .47, "befe_per_transfer": .52,
            "cycle_probability": .8, "gnn_prob": .55, "gnn_ensemble_prob": .53, "gnn_weight": .35,
            "layer_L5": True, "layer_L6": True, "befe_ci_low": .3, "befe_ci_high": .7, "befe_reliability": 61}


def _results(src: pd.DataFrame, invalid: str | None = None) -> list[dict]:
    rows = []
    for i, uid in enumerate(src["cycle_uid"]):
        for timepoint in ("transfer", "retrieval"):
            if uid == invalid and timepoint == "transfer":
                rows.append({"cycle_id": f"{uid}.{timepoint}", "status": "invalid", "seed": 1,
                             "error": {"code": "invalid_patient", "fields": ["cascade:known_pn2"]},
                             "model_fingerprint": None, "values": None})
            else:
                rows.append({"cycle_id": f"{uid}.{timepoint}", "status": "complete", "seed": 7 + i,
                             "error": None, "model_fingerprint": "fp", "values": _values(uid, timepoint, i)})
    return [{"id": "11111111-1111-1111-1111-111111111111", "columns": COLUMNS, "rows": rows}]


def test_output_contract_funnel_from_retrieval_and_70_aliases():
    import warnings
    src = _source()
    _, flags = dt_api.build_batches(src, API_CFG)
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        out = dt_api.to_dt_output(src, _results(src, invalid="CYC-CCCC"), flags)
    contracts.DT_OUTPUT.validate(out)
    assert not out.columns.duplicated().any()
    # Имя — только локальная подпись узлов отчёта, как эхо ФИО в 7.0; в службу не уходит.
    assert list(out["patient_name"]) == list(src["patient_name"])
    assert list(out["_key"]) == list(src["_key"])
    first = out.iloc[0]
    assert first["DIGITAL TWIN"] == .52 and first["DIGITAL TWIN (retrieval)"] == .47
    # med_blasts — прогноз после пункции (5.0), а не переданное модели фактическое число (4.0)
    assert first["med_blasts"] == 5.0 and first["transfer_med_blasts"] == 4.0 and first["med_bl"] == 5.0
    assert first["p_gnn_ens"] == .53 and bool(first["csdi_available"]) is True and first["w_gnn"] == .35
    assert first["OCC"] == 12 and first["Bl"] == 4 and first["real_cryo"] == 3
    assert first["dt_api_model_fingerprint"] == "fp" and first["dt_funnel_timepoint"] == "retrieval"
    assert "patient_id" not in out.columns and "record_id" not in out.columns
    assert first["dt_timepoint_used"] == "transfer"
    # «Источник NN» Teable/карточек: происхождение строки вместо описания ансамбля 7.0.
    assert first["nn_source"] == "IVF Digital Twin 7.1 REST API | model_fingerprint=fp"
    failed = out.iloc[2]
    assert failed["nn_source"] == "IVF Digital Twin 7.1 REST API | model_fingerprint=fp"
    # transfer отклонена -> вся строка из retrieval, с флагом (владелец, 2026-09-13)
    assert failed["dt_api_status"] == "invalid" and failed["dt_timepoint_used"] == "retrieval_fallback"
    assert failed["DIGITAL TWIN"] == .47 and failed["DIGITAL TWIN (retrieval)"] == .47
    assert failed["known_blasts"] is None or pd.isna(failed["known_blasts"])
    assert "cascade:known_pn2" in failed["dt_api_error"] and failed["dt_api_retrieval_status"] == "complete"
    assert failed["attempt_imputed"] and failed["afc_source"] == "constant"
    strict = dt_api.to_dt_output(src, _results(src, invalid="CYC-CCCC"), flags, fallback=False)
    assert pd.isna(strict.iloc[2]["DIGITAL TWIN"]) and strict.iloc[2]["dt_timepoint_used"] == "none"
    assert dt_api.run_summary(out)["timepoint_used"] == {"transfer": 2, "retrieval_fallback": 1}


def test_thawed_oocytes_raise_model_okk_with_flag_both_timepoints():
    src = _source()
    src.loc[1, "mii"] = 9  # MII > ОКК(5): добавлены размороженные ооциты
    for timepoint in ("transfer", "retrieval"):
        patient, flags = dt_api.build_patient(src.iloc[1], timepoint, API_CFG)
        assert patient["known_okk"] == 9 and patient["known_mii"] == 9 and flags["okk_raised_for_thawed"] is True
    patient, flags = dt_api.build_patient(src.iloc[0], "transfer", API_CFG)
    assert patient["known_okk"] == 12 and flags["okk_raised_for_thawed"] is False
    patient, _ = dt_api.build_patient(src.iloc[1], "retrieval", {**API_CFG, "impute": {"okk_for_thawed_oocytes": False}})
    assert patient["known_okk"] == 5
    _, flags_frame = dt_api.build_batches(src, API_CFG)
    out = dt_api.to_dt_output(src, _results(src), flags_frame)
    # фактические ОКК в эхе отчёта не меняются
    assert out.iloc[1]["OCC"] == 5 and bool(out.iloc[1]["okk_raised_for_thawed"])


def test_output_feeds_reconcile_and_analytics_unchanged():
    src = _source()
    _, flags = dt_api.build_batches(src, API_CFG)
    dt = dt_api.to_dt_output(src, _results(src), flags)
    opu = src.rename(columns={"occ": "OCC", "mii": "MII", "two_pn": "2pN", "bl_total": "Bl", "bl_good": "Good Bl"})
    opu["ID"] = opu["_key"]
    predictions = reconcile.build_dt_predictions(dt, None, opu, b"secret")
    contracts.DT_PREDICTIONS.validate(predictions)
    assert predictions["DIGITAL TWIN"].tolist() == [.52, .52, .52]
    conv = analytics.conversion_rates(predictions)
    bl = conv.set_index("step").loc["2PN→Bl"]
    # predicted: sum(med_blasts)/sum(med_pn2) retrieval-точки = 15/21, real = 8/20
    assert bl["predicted"] == round(15 / 21, 3) and bl["real"] == round(8 / 20, 3)


class FakeClient:
    def __init__(self, src):
        self.src, self.deleted, self.bodies, self.polls = src, [], [], 0

    def health(self):
        return {"api_version": "1", "status": "ready"}

    def submit(self, body):
        self.bodies.append(body)
        return {"id": "11111111-1111-1111-1111-111111111111", "total": len(body["cycles"]), "invalid": 0}

    def wait(self, batch_id, poll_seconds, timeout_seconds, sleep):
        assert poll_seconds == 15 and timeout_seconds == 360 * 60
        return _results(self.src)[0]

    def delete(self, batch_id):
        self.deleted.append(batch_id)


def test_orchestration_deletes_batch_after_download():
    src = _source()
    client = FakeClient(src)
    out = dt_api.run_digital_twin_api(src, {"api": API_CFG}, client=client)
    assert len(out) == 3 and client.deleted == ["11111111-1111-1111-1111-111111111111"]
    summary = dt_api.run_summary(out)
    assert summary["status"] == {"complete": 3} and summary["imputed"]["amh_imputed"] == 2
    assert summary["afc_source"] == {"afc": 1, "follicles": 1, "constant": 1}


def test_dry_check_reports_invalid_rows_without_scoring():
    src = _source()
    src.loc[1, "two_pn"] = 6  # 2PN > MII(4): transfer отклоняется, retrieval (ОКК+MII) считается
    src.loc[2, "mii"] = 9     # MII > ОКК(8): размороженные ооциты -> ОКК для модели = MII, строка валидна

    class ValidatingClient(FakeClient):
        def submit(self, body):
            raise AssertionError("dry check must not queue scoring")

        def validate(self, body):
            rows = []
            for c in body["cycles"]:
                p = c["patient"]
                order = [p.get(k) for k in ("known_okk", "known_mii", "known_pn2", "known_blasts", "known_good")]
                known = [v for v in order if v is not None]
                bad = any(b > a for a, b in zip(known, known[1:]))
                rows.append({"cycle_id": c["cycle_id"], "status": "invalid" if bad else "valid",
                             "error": {"code": "invalid_patient", "fields": ["cascade:known_pn2"]} if bad else None})
            return {"rows": rows}

    out = dt_api.check_digital_twin_api(src, {"api": {**API_CFG, "seconds_per_calculation": 36}},
                                        client=ValidatingClient(src))
    assert out["cycles"] == 3 and out["calculations"] == 5 and out["estimated_hours"] == 0.1
    assert out["timepoints"]["transfer"] == {"valid": 2, "invalid": 1,
                                             "errors": {"invalid_patient:cascade:known_pn2": 1}}
    assert {r["cycle_uid"] for r in out["invalid"]} == {"CYC-BBBB"}
    assert out["dt_coverage"] == {"transfer": 2, "retrieval_fallback": 1, "none": 0}
    assert out["imputed"]["amh_imputed"] == 2 and out["imputed"]["okk_raised_for_thawed"] == 1


def test_api_71_is_the_only_digital_twin_entry(capsys):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "run_analysis_mod", Path(__file__).resolve().parents[1] / "scripts" / "run_analysis.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Без какой-либо настройки — 7.1 через API; флага версии больше нет.
    assert "7.1" in mod.describe_dt_source({})
    assert not hasattr(mod, "apply_dt_version")
    with pytest.raises(SystemExit):
        mod.main(["--dt-version", "7.0"])
    assert dt_api.api_settings({"api": API_CFG}) == API_CFG
    assert dt_api.api_settings({"source": "api", "api": API_CFG}) == API_CFG
    # Явный батч 7.0 — ошибка, а не тихий переход; хвосты 7.0 только предупреждение.
    for source in ("batch", "7.0"):
        with pytest.raises(dt_api.DTApiError, match="батч 7.0 снят"):
            dt_api.api_settings({"source": source})
    assert dt_api.api_settings({"run_cmd": "RUN_BATCH.bat {input}", "project_dir": "C:/dt"}) == {}
    assert "run_cmd, project_dir" in capsys.readouterr().err


def test_teable_run_version_follows_the_manifest(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "teable"))
    import teable_sync
    assert teable_sync.run_dt_version(str(tmp_path), "v7.0") == "v7.0"
    (tmp_path / "run_manifest.json").write_text(json.dumps(
        {"digital_twin": dt_api.run_summary(pd.DataFrame({"dt_api_status": ["complete"]}))}), encoding="utf-8")
    assert teable_sync.run_dt_version(str(tmp_path), "v7.0") == "v7.1 (REST API)"


def test_not_ready_service_stops_the_run():
    client = FakeClient(_source())
    client.health = lambda: {"api_version": "1", "status": "license_required"}
    with pytest.raises(dt_api.DTApiError, match="не готова"):
        dt_api.run_digital_twin_api(_source(), {"api": API_CFG}, client=client)


class _Response:
    def __init__(self, code, body):
        self.status_code, self._body = code, body

    def json(self):
        return self._body


class _Session:
    def __init__(self, responses):
        self.responses, self.headers, self.calls = list(responses), {}, []

    def request(self, method, url, timeout, **kwargs):
        self.calls.append((method, url))
        return self.responses.pop(0)


def test_client_waits_for_results_and_surfaces_api_errors():
    running = {"id": "b", "total": 2, "complete": 1, "failed": 0, "invalid": 0}
    session = _Session([_Response(202, running), _Response(200, running),
                        _Response(200, {"id": "b", "rows": [], "columns": []})])
    client = dt_api.DTApiClient("https://127.0.0.1:8843/api/v1/", "ivf_api_x", "ca.crt", session=session)
    assert session.trust_env is False and session.verify == "ca.crt"
    assert session.headers["Authorization"] == "Bearer ivf_api_x"
    sleeps = []
    assert client.wait("b", poll_seconds=15, sleep=sleeps.append)["id"] == "b" and sleeps == [15]
    session.responses = [_Response(403, {"error": {"code": "scope_forbidden", "fields": []}})]
    with pytest.raises(dt_api.DTApiError, match="scope_forbidden"):
        client.health()


def test_client_times_out_instead_of_hanging():
    running = {"id": "b", "total": 2, "complete": 0, "failed": 0, "invalid": 0}
    session = _Session([_Response(202, running), _Response(200, running)] * 3)
    client = dt_api.DTApiClient("https://x/api/v1", "ivf_api_x", "ca.crt", session=session)
    ticks = iter([0, 10, 99999])
    with pytest.raises(dt_api.DTApiError, match="не завершился"):
        client.wait("b", timeout_seconds=60, sleep=lambda s: None, clock=lambda: next(ticks))


@pytest.mark.skipif(not hasattr(__import__("ctypes"), "WinDLL"), reason="DPAPI только на Windows")
def test_key_is_stored_under_user_dpapi_not_in_clear(tmp_path, monkeypatch):
    monkeypatch.delenv("EMBRYO_DT_API_KEY", raising=False)
    key = "ivf_api_00000000-0000-0000-0000-000000000000.secret-part"
    path = dt_api.save_key(key + "\n", tmp_path / "dt-api.key")
    assert key.encode() not in path.read_bytes()
    assert dt_api.load_key({"key_file": str(path)}) == key
    with pytest.raises(dt_api.DTApiError):
        dt_api.save_key("not-a-key", tmp_path / "x.key")
    with pytest.raises(dt_api.DTApiError, match="нет ключа"):
        dt_api.load_key({"key_file": str(tmp_path / "missing.key")})


def test_tabpfn_compares_against_this_runs_digital_twin_not_the_export(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "run_analysis_mod", Path(__file__).resolve().parents[1] / "scripts" / "run_analysis.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # В выгрузке — старые значения DIGITAL TWIN, вписанные в OPU-таблицу прежними прогонами.
    model_input = pd.DataFrame({"_key": ["A#20260101", "B#20260102"], "DIGITAL TWIN": [0.99, 0.88],
                                "OCC": [10, 8]})
    dt = pd.DataFrame({"_key": [" a#20260101 ", "B#20260102"], "DIGITAL TWIN": [0.52, None]})
    path, n_fresh = mod._tabpfn_input_with_fresh_dt(model_input, dt, tmp_path)
    written = pd.read_excel(path)
    assert n_fresh == 1 and written["DIGITAL TWIN"].iloc[0] == 0.52
    # DT не дал прогноза — пусто, а не старое число выгрузки.
    assert pd.isna(written["DIGITAL TWIN"].iloc[1])
    assert list(written["OCC"]) == [10, 8] and list(model_input["DIGITAL TWIN"]) == [0.99, 0.88]
