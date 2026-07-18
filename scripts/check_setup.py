"""Read-only проверка локальных путей и обязательных внешних артефактов."""
from __future__ import annotations

import sys
from pathlib import Path
import yaml

if sys.platform == "win32":
    # Detached/piped stdout on Windows falls back to the ANSI locale codepage
    # (cp1251) instead of the console's actual codepage, garbling Cyrillic output.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parents[1]


def _merge(a, b):
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = v
    return a


def main() -> int:
    with (ROOT / "config/settings.yaml").open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    local = ROOT / "config/settings.local.yaml"
    if local.exists():
        with local.open(encoding="utf-8") as f:
            _merge(cfg, yaml.safe_load(f))

    checks = []
    ext = cfg["external_tools"]
    checks.append(("Digital Twin project", Path(ext["digital_twin"]["project_dir"])))
    checks.append(("Digital Twin results", Path(ext["digital_twin"]["results_dir"])))
    checks.append(("TabPFN project", Path(ext["tabpfn"]["project_dir"])))
    checks.append(("TabPFN batch BAT", Path(ext["tabpfn"]["project_dir"]) / "run_tabpfn_batch.bat"))
    checks.append(("TabPFN batch Python", Path(ext["tabpfn"]["project_dir"]) / "run_tabpfn_batch.py"))
    dnn = ext.get("dnn_verifier", {})
    if dnn.get("enabled"):
        checks.append(("DNN model", Path(dnn.get("model_path", ""))))
        checks.append(("DNN training scaler", Path(dnn.get("scaler_path", ""))))

    failed = 0
    for label, path in checks:
        ok = path.exists()
        print(f"[{'OK' if ok else 'MISSING'}] {label}: {path}")
        failed += int(not ok)
    if failed:
        print(f"\nНе пройдено проверок: {failed}. Исправьте config/settings.local.yaml.")
        return 1
    print("\nОкружение готово к /run-analysis.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
