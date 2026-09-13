#!/usr/bin/env python
"""preflight.py — проверка окружения ПЕРЕД прогоном run_analysis.py.

Зачем отдельный шаг. Часть пакетов проекта отказывает не контрактной ошибкой, а
мягко: без `pymupdf` шаг ПГТ-А просто пишет `[warn] PGT skipped` в stderr, и
результат читается как «отчётов у клиники нет», хотя на деле дыра в окружении
(так и произошло 19.07.2026 — молча упали все 11 разборов ПГТ-А). Такой же
разрыв бывает между `requirements.txt` и фактически установленным: `pytest`
объявлен, но 28.07.2026 в активном окружении его не оказалось.

Проверяет:
  1. пакеты из requirements.txt — установлены и удовлетворяют спецификатору версии;
  2. импортируемость пакетов, чьё отсутствие даёт мягкий сбой, а не падение;
  3. пути к внешним инструментам (DT/TabPFN/DNN/TabPFN-3) из machine.local.yaml
     и настроек клиники — с учётом гейтов enabled/launch.

Запуск:  python scripts/preflight.py [--clinic fertimed]
Код возврата: 0 — можно запускать прогон; 1 — есть блокирующие проблемы.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Пакеты, отсутствие которых НЕ роняет пайплайн, а тихо отключает шаг.
# Имя дистрибутива -> имя модуля для импорта.
SOFT_FAIL_IMPORTS = {
    "pymupdf": "fitz",
    "plotly": "plotly",
    "networkx": "networkx",
    "scikit-learn": "sklearn",
    "python-calamine": "python_calamine",
    "markdown": "markdown",
}


def _parse_requirements(path: Path) -> list[tuple[str, str]]:
    """[(имя дистрибутива, спецификатор версии)] из requirements.txt."""
    out: list[tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z0-9._-]+)\s*(.*)$", line)
        if m:
            out.append((m.group(1), m.group(2).strip()))
    return out


def check_packages() -> list[str]:
    from importlib.metadata import PackageNotFoundError, version
    try:
        from packaging.requirements import Requirement
    except ImportError:
        Requirement = None

    problems: list[str] = []
    for name, spec in _parse_requirements(ROOT / "requirements.txt"):
        try:
            installed = version(name)
        except PackageNotFoundError:
            problems.append(f"НЕ УСТАНОВЛЕН: {name}{spec}")
            continue
        if spec and Requirement is not None:
            try:
                req = Requirement(f"{name}{spec}")
                if not req.specifier.contains(installed, prereleases=True):
                    problems.append(f"ВЕРСИЯ: {name} установлен {installed}, требуется {spec}")
            except Exception:
                pass
        print(f"  ok  {name:22s} {installed}")
    return problems


def check_soft_imports() -> list[str]:
    import importlib
    problems: list[str] = []
    for dist, module in SOFT_FAIL_IMPORTS.items():
        try:
            importlib.import_module(module)
            print(f"  ok  import {module:20s} ({dist})")
        except Exception as exc:
            problems.append(f"НЕ ИМПОРТИРУЕТСЯ: {module} ({dist}) — {type(exc).__name__}: {exc}")
    return problems


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def check_external_tools(clinic: str | None) -> list[str]:
    cfg = _load_yaml(ROOT / "config" / "settings.yaml")
    if clinic:
        cfg = _deep_merge(cfg, _load_yaml(ROOT / "config" / "clinics" / clinic / "settings.yaml"))
        cfg = _deep_merge(cfg, _load_yaml(ROOT / "config" / "clinics" / clinic / "settings.local.yaml"))
    else:
        cfg = _deep_merge(cfg, _load_yaml(ROOT / "config" / "settings.local.yaml"))
    cfg = _deep_merge(cfg, _load_yaml(ROOT / "config" / "machine.local.yaml"))

    tools = cfg.get("external_tools", {}) or {}
    problems: list[str] = []
    # Digital Twin — только 7.1 через REST API, для всех клиник (батч 7.0 снят 2026-09-13).
    problems += check_digital_twin_api(tools.get("digital_twin") or {})

    # (секция, человекочитаемое имя, гейт, обязательные пути)
    checks = [
        ("tabpfn", "TabPFN", "launch", ["project_dir", "results_dir"]),
        ("dnn_verifier", "DNN (EVO)", "enabled", ["project_dir", "model_path", "scaler_path"]),
        ("tabpfn_regression", "TabPFN-3 regression", "enabled", ["project_dir"]),
    ]
    for section, label, gate, path_keys in checks:
        tool = tools.get(section) or {}
        if not tool.get(gate, False):
            print(f"  --  {label}: выключен ({gate}=false) — пропуск")
            continue
        for key in path_keys:
            value = tool.get(key)
            if not value:
                problems.append(f"{label}: не задан {key} (см. config/machine.local.yaml)")
                continue
            if not Path(value).exists():
                problems.append(f"{label}: путь не существует — {key}={value}")
            else:
                print(f"  ok  {label}: {key}")
    return problems


def check_digital_twin_api(dt_cfg: dict) -> list[str]:
    """7.1 через REST API: сертификат, ключ под DPAPI и живой авторизованный ответ службы.

    Сетевой вызов только на заданный локальный адрес службы; данных пациентов не содержит.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from embryo import dt_api
    try:
        client = dt_api.DTApiClient.from_config(dt_api.api_settings(dt_cfg))
        health = client.health()
        if health.get("status") != "ready":
            return [f"Digital Twin API: служба ответила {health} (лицензия 7.1?)"]
        client._call("GET", "/analytics/batches")
    except Exception as exc:
        return [f"Digital Twin API: {type(exc).__name__}: {exc}"]
    print(f"  ok  Digital Twin 7.1 API: {client.url} (ключ «Аналитика» принят)")
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка окружения перед прогоном")
    parser.add_argument("--clinic", help="config/clinics/<name> (fertimed, ggrc, skyfert)")
    parser.add_argument("--fertimed", action="store_true", help="ярлык для --clinic fertimed")
    parser.add_argument("--ggrc", action="store_true", help="ярлык для --clinic ggrc")
    parser.add_argument("--skip-tools", action="store_true",
                        help="не проверять пути внешних инструментов")
    args = parser.parse_args()
    clinic = args.clinic or ("fertimed" if args.fertimed else "ggrc" if args.ggrc else None)

    print(f"[preflight] python  : {sys.executable}")
    print(f"[preflight] клиника : {clinic or 'A (по умолчанию)'}")

    problems: list[str] = []
    print("\n[1/3] пакеты из requirements.txt")
    problems += check_packages()
    print("\n[2/3] импорт пакетов с мягким отказом")
    problems += check_soft_imports()
    if args.skip_tools:
        print("\n[3/3] внешние инструменты — пропущено (--skip-tools)")
    else:
        print("\n[3/3] внешние инструменты")
        problems += check_external_tools(clinic)

    if problems:
        print(f"\n[preflight] ПРОБЛЕМ: {len(problems)}")
        for item in problems:
            print(f"  - {item}")
        print("\nПочинить обычно так:  python -m pip install -r requirements.txt")
        print("Пути внешних инструментов — config/machine.local.yaml")
        return 1
    print("\n[preflight] окружение в порядке — можно запускать прогон")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
