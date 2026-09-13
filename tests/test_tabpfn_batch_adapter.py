"""Адаптер external/tabpfn/run_tabpfn_batch.py — headless-батч TabPFN.

Файл живёт в ЧУЖОМ проекте (кладётся рядом с app_tabpfn.py) и импортирует его
модули, которых здесь нет — они подменяются заглушками, чтобы проверить наш
собственный слой: разбор аргументов, сопоставление колонок, обращение с нулями
и атомарную запись результата.

Что здесь важнее всего: ноль — наблюдение, а не пропуск; MII никогда не берётся
из «числа инсеминированных»; ключом строки идёт составной "_key", а не голый
паспортный ID (иначе повторные попытки одной пациентки схлопнутся в один цикл);
результат пишется атомарно и со штампом run_id, по которому tabpfn_runner
отличает свежий файл от чужого.
"""
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "external" / "tabpfn" / "run_tabpfn_batch.py"


@pytest.fixture(scope="module")
def adapter():
    """Загрузить адаптер с заглушками вместо модулей внешнего приложения TabPFN."""
    stubs = {"mc_features": types.ModuleType("mc_features"),
             "tabpfn_backend": types.ModuleType("tabpfn_backend")}
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = spec_from_file_location("run_tabpfn_batch", SCRIPT)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def test_column_map_prefers_the_composite_cycle_key(adapter):
    # "_key" (ID+Дата пункции) должен стоять раньше голого ID: повторные попытки
    # одной пациентки делят один паспортный ID и иначе схлопнутся.
    assert adapter.COLMAP["pid"][0] == "_key"
    assert "patient_id" in adapter.COLMAP["pid"]


def test_mii_is_never_aliased_to_the_inseminated_count(adapter):
    # MII необязателен; подмена его «числом инсеминированных» испортила бы MIIR
    # и вход модели одновременно.
    assert adapter.COLMAP["mii"] == ["mii", "MII", "Число MII"]
    assert not any("инсемин" in name.lower() for name in adapter.COLMAP["mii"])


def test_pick_returns_the_first_present_column_or_an_empty_series(adapter):
    frame = pd.DataFrame({"MII": [7, 5], "Число ОКК": [10, 8]})
    assert adapter._pick(frame, adapter.COLMAP["mii"]).tolist() == [7, 5]
    absent = adapter._pick(frame, ["нет-такой", "и-такой"])
    assert len(absent) == len(frame) and absent.isna().all()


def test_known_count_keeps_zero_and_reports_unknown_as_none(adapter):
    assert adapter._known_count(0) == 0        # ноль — наблюдение, не пропуск
    assert adapter._known_count(7.0) == 7
    assert adapter._known_count(None) is None
    assert adapter._known_count(np.nan) is None


@pytest.mark.parametrize("bad", [-1, 2.5, float("inf")])
def test_known_count_rejects_impossible_embryology_counts(bad, adapter):
    with pytest.raises(ValueError, match="non-negative integer"):
        adapter._known_count(bad)


def test_blank_if_missing_writes_an_empty_cell_not_the_string_nan(adapter):
    assert adapter._blank_if_missing(None) == ""
    assert adapter._blank_if_missing(np.nan) == ""
    assert adapter._blank_if_missing(0) == 0
    assert adapter._blank_if_missing("Emb A") == "Emb A"


def test_atomic_csv_leaves_no_partial_file_and_uses_utf8_bom(adapter, tmp_path):
    target = tmp_path / "out" / "tabpfn_predictions.csv"
    frame = pd.DataFrame({"patient_id": ["E001#20260101"], "p_tabpfn": [0.7],
                          "ci_low": [0.6], "ci_high": [0.8]})
    written = adapter._atomic_csv(frame, target)

    assert written == target and target.is_file()
    assert target.read_bytes().startswith(b"\xef\xbb\xbf"), "нужен BOM: файл читают в Excel"
    # Временный файл записи не остаётся рядом с результатом.
    assert [p.name for p in target.parent.iterdir()] == [target.name]
    assert pd.read_csv(target, encoding="utf-8-sig")["patient_id"].iloc[0] == "E001#20260101"


def test_atomic_csv_overwrites_an_existing_result(adapter, tmp_path):
    target = tmp_path / "tabpfn_predictions.csv"
    adapter._atomic_csv(pd.DataFrame({"a": [1]}), target)
    adapter._atomic_csv(pd.DataFrame({"a": [2]}), target)
    assert pd.read_csv(target, encoding="utf-8-sig")["a"].tolist() == [2]


def test_load_reads_both_csv_and_excel(adapter, tmp_path):
    frame = pd.DataFrame({"MII": [7]})
    frame.to_csv(tmp_path / "a.csv", index=False)
    frame.to_excel(tmp_path / "a.xlsx", index=False)
    assert adapter._load(str(tmp_path / "a.csv"))["MII"].iloc[0] == 7
    assert adapter._load(str(tmp_path / "a.xlsx"))["MII"].iloc[0] == 7


def test_args_accept_the_flags_the_pipeline_actually_sends(adapter, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_tabpfn_batch.py", "--input", "in.xlsx",
                                      "--output", "out.csv", "--run-id", "abc123"])
    ns = adapter._args()
    assert (ns.input_path, ns.output_path, ns.run_id) == ("in.xlsx", "out.csv", "abc123")


def test_args_still_accept_the_legacy_positional_form(adapter, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_tabpfn_batch.py", "in.xlsx", "out.csv"])
    ns = adapter._args()
    assert (ns.input_path, ns.output_path) == ("in.xlsx", "out.csv")
    assert ns.run_id, "run_id должен генерироваться, если не передан"


def test_args_without_input_fail_instead_of_scoring_nothing(adapter, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_tabpfn_batch.py"])
    with pytest.raises(SystemExit):
        adapter._args()


def test_config_defaults_do_not_require_a_config_file(adapter):
    cfg = adapter._cfg()
    assert cfg["target"] == "Исход переноса"
    assert cfg["backend"] and cfg["device"] and cfg["max_n"] > 0


def test_audit_log_columns_cover_the_contract_the_pipeline_reads(adapter):
    # tabpfn_runner валидирует ровно эти поля в результате прогона.
    for required in ("patient_id", "p_tabpfn", "ci_low", "ci_high"):
        assert required in adapter.LOG_COLUMNS
