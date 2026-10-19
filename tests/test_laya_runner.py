"""Laya как внешний компаратор DT: адаптер external/laya/laya_batch.py + embryo.laya_runner.

Модель Laya живёт в своём окружении (torch) и в тестах не участвует: вместо
service/laya_ivf.py подкладывается заглушка с тем же контрактом assess(). Сам
адаптер и раннер запускаются настоящим дочерним python-процессом, как в
test_external_runners.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from embryo import laya_runner

ROOT = Path(__file__).resolve().parents[1]

STUB = '''
def assess(cycles, centre=None, restore=False):
    rows = []
    for c in cycles:
        inp = c["inputs"]
        ok = inp.get("Age") is not None and inp.get("OCC") not in (None, 0.0)
        rows.append({"cycle_uid": c["cycle_uid"], "laya_probability": 0.4,
                     "laya_probability_centre_calibrated": 0.45 if centre else None,
                     "assessable": ok, "restored": ["attempt_number"] if restore else [],
                     "flags": [] if ok else ["required input unknown: age_years"],
                     "difference_laya_minus_dt": (0.45 - c["dt_probability"]) if ok and "dt_probability" in c else None,
                     "disagreement_high": True if ok and "dt_probability" in c else None})
    return {"backend": "stub", "centre": centre, "results": rows}
'''


def _cfg(tmp_path, centre="Two-Line"):
    service = tmp_path / "service"
    service.mkdir()
    (service / "laya_ivf.py").write_text(STUB, encoding="utf-8")
    cmd = (f'"{sys.executable}" "{ROOT / "external" / "laya" / "laya_batch.py"}" '
           f'--service-dir "{service}" --input {{input}} --output {{output}}')
    return {"enabled": True, "project_dir": str(service), "run_cmd": cmd,
            "centre": centre, "restore": True}


def _predictions():
    return pd.DataFrame({
        "cycle_uid": ["CYC-A", "CYC-B"], "patient_id": ["secret-1", "secret-2"],
        "Age": [33, None], "Attempt": [1, 2], "N folicules OPU": [10, 8], "OCC": [8, 6],
        "Inseminated": [7, 5], "2pN": [5, 4], "DIGITAL TWIN": [0.30, 0.50]})


def test_run_and_join_laya(tmp_path):
    pred = _predictions()
    out = laya_runner.run_laya(_cfg(tmp_path), pred, tmp_path / "private")
    joined = laya_runner.join_laya(pred, out)
    a = joined.set_index("cycle_uid")
    assert a.loc["CYC-A", "Laya"] == pytest.approx(0.45)
    assert a.loc["CYC-A", "laya_minus_dt_pp"] == pytest.approx(15.0)
    assert bool(a.loc["CYC-A", "laya_disagreement_high"]) is True
    # непригодный цикл: вероятности не публикуются, флаг объясняет почему
    assert pd.isna(a.loc["CYC-B", "Laya"]) and pd.isna(a.loc["CYC-B", "laya_raw"])
    assert "required input" in a.loc["CYC-B", "laya_flags"]
    assert laya_runner.coverage(joined) == {"scored": 1, "scored_centre_calibrated": 1, "cohort": 2,
                                            "not_assessable": 1, "disagreement_high": 1}


def test_coverage_without_centre_counts_raw_probability(tmp_path):
    # Клиника без калибровки центра (PERSONA): Laya пуста, laya_raw — нет; покрытие не нулевое.
    pred = _predictions()
    joined = laya_runner.join_laya(pred, laya_runner.run_laya(_cfg(tmp_path, centre=None), pred,
                                                              tmp_path / "private"))
    cov = laya_runner.coverage(joined)
    assert cov["scored"] == 1 and cov["scored_centre_calibrated"] == 0


def test_laya_input_has_no_identifiers(tmp_path):
    laya_runner.run_laya(_cfg(tmp_path), _predictions(), tmp_path / "private")
    sent = pd.read_csv(tmp_path / "private" / "laya_input.csv", encoding="utf-8-sig")
    assert "patient_id" not in sent.columns
    assert set(sent.columns) <= set(laya_runner.INPUT_COLUMNS)


def test_without_centre_only_raw(tmp_path):
    out = laya_runner.run_laya(_cfg(tmp_path, centre=""), _predictions(), tmp_path / "private")
    assert out["Laya"].isna().all()
    assert out["laya_raw"].notna().sum() == 1


def test_run_cmd_needs_placeholders(tmp_path):
    with pytest.raises(ValueError):
        laya_runner.run_laya({"run_cmd": "python x.py"}, _predictions(), tmp_path)


def test_missing_output_is_an_error(tmp_path):
    cfg = {"run_cmd": f'"{sys.executable}" -c "pass" {{input}} {{output}}', "project_dir": str(tmp_path)}
    with pytest.raises(FileNotFoundError):
        laya_runner.run_laya(cfg, _predictions(), tmp_path / "private")
