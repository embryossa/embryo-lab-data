"""Machine-local inputs of the research scripts in reports/.

The scripts read source tables and external projects (IVF Digital Twin, the
TabPFN training context, TabFM work folder) that live outside this repository
and are not public. Their locations are kept in
`config/research_paths.local.json` (git-ignored; template:
`config/research_paths.local.json.example`) or in an `EMBRYO_<KEY>` environment
variable, which wins over the file. Standard library only: some scripts run in
their own virtual environments.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config" / "research_paths.local.json"


def local_path(key: str) -> Path:
    env = os.environ.get(f"EMBRYO_{key.upper()}")
    if env:
        return Path(env)
    paths = json.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() else {}
    if not paths.get(key):
        raise KeyError(f"'{key}' is not set: add it to {CONFIG} "
                       f"or set EMBRYO_{key.upper()}")
    return Path(paths[key])
