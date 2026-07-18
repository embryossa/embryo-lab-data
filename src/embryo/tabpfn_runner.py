"""Launch/read the external TabPFN adapter with current-run provenance."""
from __future__ import annotations

from pathlib import Path
import time
import uuid

import pandas as pd

from ._io import latest_file_after
from ._launch import run_external
from .contracts import TABPFN_OUTPUT, validate_ci


def _validate(frame: pd.DataFrame, expected_run_id: str | None = None) -> pd.DataFrame:
    TABPFN_OUTPUT.validate(frame)
    validate_ci(frame)
    out = frame.copy()
    for col in ("p_tabpfn", "ci_low", "ci_high"):
        out[col] = pd.to_numeric(out[col], errors="raise")
    if expected_run_id is not None and "run_id" in out.columns:
        found = set(out["run_id"].dropna().astype(str))
        if found != {expected_run_id}:
            raise ValueError(f"TabPFN output run_id mismatch: expected {expected_run_id}, got {found}")
    out["ci_width_pp"] = (out["ci_high"] - out["ci_low"]) * 100.0
    return out


def read_tabpfn(results_dir: str | None = None,
                 pattern: str = "tabpfn_predictions*.csv",
                 explicit_path: str | Path | None = None,
                 expected_run_id: str | None = None) -> pd.DataFrame:
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.is_file():
            raise FileNotFoundError(f"TabPFN explicit output not found: {path}")
    else:
        if not results_dir:
            raise ValueError("results_dir or explicit_path is required")
        files = list(Path(results_dir).glob(pattern))
        if not files:
            raise FileNotFoundError(f"no TabPFN results {pattern} in {results_dir}")
        path = max(files, key=lambda p: p.stat().st_mtime)
    return _validate(pd.read_csv(path), expected_run_id)


def launch_and_read(cfg: dict, input_path: str | None = None) -> pd.DataFrame:
    results_dir = Path(cfg["results_dir"])
    pattern = cfg.get("predictions_glob", "tabpfn_predictions*.csv")
    mode = cfg.get("mode", "server")
    started = time.time()

    if mode == "server":
        run_external(cfg.get("run_cmd"), cfg.get("project_dir", "."), input_path,
                     blocking=False)
        deadline = started + int(cfg.get("timeout_s", 900))
        while time.time() < deadline:
            fresh = latest_file_after(str(results_dir), pattern, started)
            if fresh is not None:
                return read_tabpfn(explicit_path=fresh)
            time.sleep(3)
        raise TimeoutError("TabPFN did not produce a current-run result before timeout")

    run_id = uuid.uuid4().hex
    configured_output = cfg.get("output")
    explicit = ((Path(configured_output) if configured_output
                 else results_dir / f"tabpfn_predictions_{run_id}.csv").resolve())
    cmd = cfg.get("run_cmd")
    if cfg.get("launch", True):
        if not input_path:
            raise ValueError("TabPFN batch requires an explicit input path")
        if not cmd or any(token not in cmd for token in ("{input}", "{output}", "{run_id}")):
            raise ValueError("TabPFN batch run_cmd must forward {input}, {output}, and {run_id}")
        cmd = cmd.replace("{run_id}", run_id)
    if cfg.get("launch", True):
        run_external(cmd, cfg.get("project_dir", "."), input_path,
                     output_path=str(explicit), blocking=True)

    # Preferred contract: run_cmd contains {input}/{output}/{run_id} and writes exact path.
    if explicit.is_file() and explicit.stat().st_mtime >= started - 2:
        return read_tabpfn(explicit_path=explicit,
                           expected_run_id=run_id if cfg.get("launch", True) else None)
    raise FileNotFoundError(f"TabPFN did not create its explicit current-run output: {explicit}")
