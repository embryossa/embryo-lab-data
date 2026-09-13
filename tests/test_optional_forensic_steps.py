"""Необязательные шаги прогона: DNN-проверка и Airtable-форензик лотов.

Оба выключены по умолчанию и включаются только у части клиник — поэтому и
ломаются незаметно: прогон продолжается, а в отчёте молча пусто. Здесь
проверяется, что каждый из них либо даёт результат, либо отказывает ГРОМКО, с
внятной причиной, и что сеть/внешний процесс подменяются, а не вызываются.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time

import pandas as pd
import pytest

from embryo import dnn_runner, lots


def _script(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "fake_dnn.py"
    path.write_text(body, encoding="utf-8")
    return path


def _cmd(script: Path, *args: str) -> str:
    return f'"{sys.executable}" "{script}" ' + " ".join(args)


_DNN_ROWS = (
    "patient_id,Исход переноса EVO,Вероятность положительного исхода,dnn_validation_status\n"
    "E001#20260101,1,0.66,valid_saved_scaler\n"
    "E002#20260102,0,0.21,valid_saved_scaler\n"
)


def _dnn_cfg(tmp_path: Path, rows: str = _DNN_ROWS) -> dict:
    scaler = tmp_path / "training_scaler.joblib"
    scaler.write_bytes(b"stub")
    model = tmp_path / "model.keras"
    model.write_bytes(b"stub")
    out = tmp_path / "evo_predictions.csv"
    script = _script(tmp_path, (
        "import sys, pathlib\n"
        f"pathlib.Path(sys.argv[1]).write_text({rows!r}, encoding='utf-8')\n"
    ))
    return {
        "scaler_path": str(scaler), "model_path": str(model), "output": str(out),
        "project_dir": str(tmp_path),
        "run_cmd": _cmd(script, "{output}") + " {input} {scaler} {model}",
    }


# ---------------------------------------------------------------- DNN

def test_dnn_runs_and_validates_its_output_contract(tmp_path):
    frame = dnn_runner.run_dnn_verifier(_dnn_cfg(tmp_path), str(tmp_path / "model_input.xlsx"))
    assert frame["patient_id"].tolist() == ["E001#20260101", "E002#20260102"]
    assert frame["Вероятность положительного исхода"].tolist() == [0.66, 0.21]


def test_dnn_without_a_saved_training_scaler_is_refused(tmp_path):
    # Правило проекта: DNN недействительна без сохранённого scaler обучения.
    cfg = _dnn_cfg(tmp_path)
    cfg["scaler_path"] = ""
    with pytest.raises(ValueError, match="scaler_path"):
        dnn_runner.run_dnn_verifier(cfg, "in.xlsx")

    cfg["scaler_path"] = str(tmp_path / "нет.joblib")
    with pytest.raises(FileNotFoundError, match="scaler not found"):
        dnn_runner.run_dnn_verifier(cfg, "in.xlsx")


def test_dnn_requires_explicit_output_and_all_placeholders(tmp_path):
    cfg = _dnn_cfg(tmp_path)
    cfg["output"] = ""
    with pytest.raises(ValueError, match="explicit output"):
        dnn_runner.run_dnn_verifier(cfg, "in.xlsx")

    cfg = _dnn_cfg(tmp_path)
    cfg["run_cmd"] = "tool {input} {output}"          # нет {scaler}
    with pytest.raises(ValueError, match=r"\{scaler\}"):
        dnn_runner.run_dnn_verifier(cfg, "in.xlsx")


def test_dnn_with_model_placeholder_needs_an_existing_model(tmp_path):
    cfg = _dnn_cfg(tmp_path)
    cfg["model_path"] = ""
    with pytest.raises(ValueError, match="model_path is not configured"):
        dnn_runner.run_dnn_verifier(cfg, "in.xlsx")

    cfg = _dnn_cfg(tmp_path)
    cfg["model_path"] = str(tmp_path / "нет.keras")
    with pytest.raises(FileNotFoundError, match="saved model not found"):
        dnn_runner.run_dnn_verifier(cfg, "in.xlsx")


def test_dnn_output_produced_without_a_validated_scaler_is_rejected(tmp_path):
    rows = _DNN_ROWS.replace("valid_saved_scaler", "fitted_on_inference")
    with pytest.raises(ValueError, match="validated saved scaler"):
        dnn_runner.run_dnn_verifier(_dnn_cfg(tmp_path, rows), "in.xlsx")


def test_dnn_output_missing_contract_columns_is_rejected(tmp_path):
    rows = "patient_id,Вероятность положительного исхода\nE001,0.5\n"
    with pytest.raises(ValueError, match="contract missing columns"):
        dnn_runner.run_dnn_verifier(_dnn_cfg(tmp_path, rows), "in.xlsx")


def test_dnn_probability_outside_zero_one_is_rejected(tmp_path):
    rows = _DNN_ROWS.replace("0.66", "1.4")
    with pytest.raises(ValueError, match=r"probability must be in \[0, 1\]"):
        dnn_runner.run_dnn_verifier(_dnn_cfg(tmp_path, rows), "in.xlsx")


def test_dnn_duplicate_patient_id_is_rejected(tmp_path):
    rows = _DNN_ROWS.replace("E002#20260102", "E001#20260101")
    with pytest.raises(ValueError):
        dnn_runner.run_dnn_verifier(_dnn_cfg(tmp_path, rows), "in.xlsx")


def test_dnn_stale_output_is_not_accepted_as_this_run(tmp_path):
    cfg = _dnn_cfg(tmp_path)
    Path(cfg["output"]).write_text(_DNN_ROWS, encoding="utf-8")
    os.utime(cfg["output"], (time.time() - 3600, time.time() - 3600))
    cfg["run_cmd"] = _cmd(_script(tmp_path, "pass\n")) + " {input} {output} {scaler} {model}"
    with pytest.raises(FileNotFoundError, match="did not create its explicit output"):
        dnn_runner.run_dnn_verifier(cfg, "in.xlsx")


# -------------------------------------------------------------- лоты

class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_fetch_table_pages_through_every_record(monkeypatch):
    pages = [
        {"records": [{"id": "rec1", "fields": {"Название": "среда culture"}}], "offset": "p2"},
        {"records": [{"id": "rec2", "fields": {"Название": "масло"}}]},
    ]
    seen = []

    def fake_get(url, headers, params, timeout):
        seen.append(params.get("offset"))
        return FakeResponse(pages[len(seen) - 1])

    monkeypatch.setattr(lots.requests, "get", fake_get)
    frame = lots.fetch_table("appX", "tblY", "токен")
    assert frame["record_id"].tolist() == ["rec1", "rec2"]
    assert seen == [None, "p2"], "вторая страница должна запрашиваться по offset"


def test_fetch_table_without_a_token_fails_with_a_pointer_to_the_config(monkeypatch):
    with pytest.raises(ValueError, match="settings.local.yaml"):
        lots.fetch_table("appX", "tblY", "")


def test_fetch_consumable_lots_pulls_both_tables(monkeypatch):
    calls = []

    def fake_fetch(base_id, table_id, token, page_size=100):
        calls.append(table_id)
        return pd.DataFrame({"record_id": [table_id]})

    monkeypatch.setattr(lots, "fetch_table", fake_fetch)
    cfg = {"consumable_lots": {"airtable": {"base_id": "appX", "token": "t",
                                            "items_table_id": "tblItems",
                                            "changes_table_id": "tblChanges"}}}
    items, changes = lots.fetch_consumable_lots(cfg)
    assert calls == ["tblItems", "tblChanges"]
    assert items["record_id"].iloc[0] == "tblItems"
    assert changes["record_id"].iloc[0] == "tblChanges"


def test_select_name_flattens_airtable_cell_shapes():
    assert lots._select_name({"name": "среда"}) == "среда"
    assert lots._select_name([{"name": "среда"}, {"name": "масло"}]) == "среда, масло"
    assert lots._select_name("готовая строка") == "готовая строка"
    assert lots._select_name(None) is None
    assert lots._select_name("") is None
    assert lots._select_name([]) is None


def test_first_link_id_reads_both_link_shapes():
    assert lots._first_link_id([{"id": "rec1"}]) == "rec1"
    assert lots._first_link_id(["rec1"]) == "rec1"
    assert lots._first_link_id([]) is None
    assert lots._first_link_id(None) is None


def test_cultivation_window_matches_the_configured_six_days():
    start, end = lots.cultivation_window("2026-01-01")
    assert (end - start) == pd.Timedelta(days=6)
    start, end = lots.cultivation_window("2026-01-01", days=3)
    assert (end - start) == pd.Timedelta(days=3)
