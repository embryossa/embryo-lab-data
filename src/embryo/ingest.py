"""
ingest.py — загрузка выгрузок и резолв колонок по config/columns.yaml.
Неизвестная колонка -> предупреждение (не молчаливое угадывание, CLAUDE.md §0.2).
"""
from __future__ import annotations
from pathlib import Path
import warnings
import pandas as pd
import yaml


def load_config(path: str = "config/columns.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _alias_index(columns_cfg: dict) -> dict:
    """Строит {нормализованный_алиас: каноническое_имя}."""
    idx = {}
    for canon, spec in columns_cfg["canonical"].items():
        idx[canon.lower()] = canon
        for alias in spec.get("aliases", []):
            idx[str(alias).strip().lower()] = canon
    return idx


def load_table(path: str) -> pd.DataFrame:
    """Читает .csv/.xlsx (calamine -> openpyxl fallback).
    Полностью пустые строки (Excel-паддинг за пределами реальных данных) отбрасываются:
    строка без единого значения — не наблюдение с пропуском, а артефакт экспорта."""
    p = Path(path)
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        try:
            df = pd.read_excel(p, engine="calamine")
        except Exception:
            df = pd.read_excel(p, engine="openpyxl")
    # bool-колонки (напр. чекбоксы Excel) читаются как False даже в строках-паддинге,
    # поэтому "полностью пустая строка" проверяется без их учёта.
    non_bool = df.columns[df.dtypes != "bool"]
    blank = df[non_bool].isna().all(axis=1)
    # Формула Excel (напр. "Age" от DOB) может вернуть 0 на строке-паддинге за
    # пределами реальных данных, а не NaN — тогда строка не проходит строгую
    # проверку "все NaN", хотя не несёт ни одного реального наблюдения (реальные
    # строки этого проекта всегда заполняют минимум ~14 полей, паддинг — не больше 1).
    stray = df[non_bool].notna().sum(axis=1) <= 1
    return df.loc[~(blank | stray)].reset_index(drop=True)


def resolve_columns(df: pd.DataFrame, columns_cfg: dict) -> tuple[pd.DataFrame, list, list]:
    """
    Переименовывает колонки в канонические. Возвращает (df, unknown, missing_required).
    """
    idx = _alias_index(columns_cfg)
    rename, unknown, collisions = {}, [], []
    used = set()
    for col in df.columns:
        key = str(col).strip().lower()
        if key in idx:
            canon = idx[key]
            if canon in used:                       # 2+ входных колонки -> один канон
                collisions.append((col, canon))     # оставляем первую, эту не переименовываем
                continue
            rename[col] = canon
            used.add(canon)
        else:
            unknown.append(col)
    out = df.rename(columns=rename)
    if collisions:
        warnings.warn(f"Коллизии канона (оставлена первая): {collisions}")

    required = [c for c, s in columns_cfg["canonical"].items() if s.get("required")]
    missing = [c for c in required if c not in out.columns]

    if unknown:
        warnings.warn(f"Неизвестные колонки (пропущены при резолве): {unknown}")
    if missing:
        warnings.warn(f"ОТСУТСТВУЮТ обязательные канонические колонки: {missing}")
    return out, unknown, missing
