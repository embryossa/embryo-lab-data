from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from embryo.dnn_runner import run_dnn_verifier


ROOT = Path(__file__).resolve().parents[1]


def _load_dnn_module():
    path = ROOT / "external" / "dnn" / "Evo_calculations.py"
    spec = spec_from_file_location("evo_calculations_test", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dnn_requires_saved_scaler(tmp_path):
    with pytest.raises(ValueError, match="scaler_path"):
        run_dnn_verifier({"run_cmd": None}, "input.csv")

    module = _load_dnn_module()
    with pytest.raises(ValueError, match="требуется путь"):
        module._load_scaler("", module.SELECTED)

    wrong = StandardScaler().fit(np.ones((3, 2)))
    path = tmp_path / "wrong.joblib"
    joblib.dump(wrong, path)
    with pytest.raises(ValueError, match="ожидает 2 признаков"):
        module._load_scaler(str(path), module.SELECTED)


def test_tabpfn_adapter_keeps_zero_and_does_not_alias_inseminated_to_mii():
    source = (ROOT / "external" / "tabpfn" / "run_tabpfn_batch.py").read_text(encoding="utf-8")
    assert '"mii": ["mii", "MII", "Число MII"]' in source
    assert "if pd.isna(value):" in source and "return int(number)" in source
    assert "--run-id" in source and "_atomic_csv" in source
