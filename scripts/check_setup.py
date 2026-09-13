"""Read-only проверка локальных путей и обязательных внешних артефактов."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml

if sys.platform == "win32":
    # Detached/piped stdout on Windows falls back to the ANSI locale codepage
    # (cp1251) instead of the console's actual codepage, garbling Cyrillic output.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embryo import dt_api  # noqa: E402


def _merge(a, b):
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = v
    return a


def _settings(config_dir: Path) -> dict:
    with (config_dir / "settings.yaml").open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # config/machine.local.yaml — машинно-зависимые пути к внешним инструментам,
    # общие для всех клиник на этом ПК (см. config/machine.local.yaml.example).
    machine = ROOT / "config" / "machine.local.yaml"
    if machine.exists():
        with machine.open(encoding="utf-8") as f:
            _merge(cfg, yaml.safe_load(f) or {})
    local = config_dir / "settings.local.yaml"
    if local.exists():
        with local.open(encoding="utf-8") as f:
            _merge(cfg, yaml.safe_load(f) or {})
    return cfg


def check(config_dir: Path, label: str) -> int:
    print(f"=== {label} ({config_dir}) ===")
    cfg = _settings(config_dir)

    checks = []
    ext = cfg["external_tools"]
    # Digital Twin — только 7.1 через REST API (батч 7.0 снят 2026-09-13): здесь только
    # сертификат службы; ключ и живой ответ проверяет scripts/preflight.py.
    try:
        api = dt_api.api_settings(ext.get("digital_twin"))
    except dt_api.DTApiError as exc:
        print(f"[FAIL] {exc}")
        return 1
    print("[INFO] Digital Twin 7.1 через REST API: " + (api.get("url") or "https://127.0.0.1:8843/api/v1"))
    checks.append(("Digital Twin 7.1 API certificate", Path(api.get("ca_cert") or "")))
    checks.append(("TabPFN project", Path(ext["tabpfn"]["project_dir"])))
    checks.append(("TabPFN batch BAT", Path(ext["tabpfn"]["project_dir"]) / "run_tabpfn_batch.bat"))
    checks.append(("TabPFN batch Python", Path(ext["tabpfn"]["project_dir"]) / "run_tabpfn_batch.py"))
    dnn = ext.get("dnn_verifier", {})
    if dnn.get("enabled"):
        checks.append(("DNN model", Path(dnn.get("model_path", ""))))
        checks.append(("DNN training scaler", Path(dnn.get("scaler_path", ""))))
    regression = ext.get("tabpfn_regression", {})
    if regression.get("enabled"):
        checks.append(("TabPFN-3 regression project", Path(regression.get("project_dir", ""))))

    failed = 0
    for check_label, path in checks:
        ok = bool(str(path)) and path.exists()
        print(f"[{'OK' if ok else 'MISSING'}] {check_label}: {path}")
        failed += int(not ok)
    if failed:
        print(f"\nНе пройдено проверок: {failed}. Исправьте config/machine.local.yaml "
              f"(и, при необходимости, {config_dir / 'settings.local.yaml'}).")
    else:
        print("\nОкружение готово к /run-analysis.")
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinic", default=None,
                         help="проверить клиничный конфиг, напр. --clinic fertimed; "
                              "без флага — только базовая клиника (config/settings.yaml)")
    parser.add_argument("--all-clinics", action="store_true",
                         help="проверить базовую клинику и все config/clinics/*")
    args = parser.parse_args()

    targets = []
    if args.all_clinics:
        targets.append((ROOT / "config", "клиника A (base)"))
        clinics_dir = ROOT / "config" / "clinics"
        if clinics_dir.is_dir():
            for clinic_dir in sorted(p for p in clinics_dir.iterdir() if p.is_dir()):
                targets.append((clinic_dir, f"клиника {clinic_dir.name}"))
    elif args.clinic:
        targets.append((ROOT / "config" / "clinics" / args.clinic, f"клиника {args.clinic}"))
    else:
        targets.append((ROOT / "config", "клиника A (base)"))

    failed_total = 0
    for config_dir, label in targets:
        failed_total += check(config_dir, label)
        print()
    return 1 if failed_total else 0


if __name__ == "__main__":
    sys.exit(main())
