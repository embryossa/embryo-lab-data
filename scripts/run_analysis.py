"""Единый строгий оркестратор полного локального анализа для Claude Code."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

if sys.platform == "win32":
    # Detached/piped stdout on Windows falls back to the ANSI locale codepage
    # (cp1251) instead of the console's actual codepage, garbling Cyrillic output.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embryo import (analytics, clinical, consult, contracts, dnn_runner, dt_runner, history,
                    incubator, ingest, kpi, lots, normalize, opu_table, ozema, pgt, privacy,
                    reconcile, regression_qc, report, tabpfn_regression_runner,
                    tabpfn_runner)  # noqa: E402
from embryo import _io  # noqa: E402
from embryo.clinics import fertimed, fertimed_pgt, ggrc, skyfert  # noqa: E402

# Клиники с собственным ingest/normalize-адаптером (см. src/embryo/clinics/).
# Без --clinic (клиника A) поведение не меняется ни на йоту.
_CLINIC_NORMALIZERS = {"fertimed": fertimed.prepare, "skyfert": skyfert.prepare, "ggrc": ggrc.prepare}
# Клиники со своим ПГТ-А парсером (другой вендор/формат PDF, см.
# src/embryo/clinics/fertimed_pgt.py). GGRC — не PDF, а колонки, уже встроенные
# в её выгрузку (ggrc.analyze_reports читает их напрямую, см. docs/GGRC.md).
# Без --clinic используется общий embryo.pgt (eMap Cycle Report, клиника A) —
# поведение не меняется.
_CLINIC_PGT_MODULES = {"fertimed": fertimed_pgt, "ggrc": ggrc}
# Клиники, чья широкая lab-популяция (источник ВСЕХ таблиц KPI + OZEMA) приходит
# отдельным ФАЙЛОМ, а не срезом того же входного файла. Значение — clinic-adapter
# функция той же формы prepare(raw, columns_cfg, cfg) -> (df, unknown),
# см. ggrc.prepare_wide.
_CLINIC_WIDE_LOADERS = {"ggrc": ggrc.prepare_wide}
# Клиники, чья широкая lab-популяция берётся из ТОГО ЖЕ входного файла, но
# другим фильтром строк, чем отчётная когорта. Для Fertimed это принципиально:
# fertimed.prepare гейтует строки заполненным исходом переноса, а в июньской
# выгрузке исход стоит почти только у криопереносов — без отдельного загрузчика
# "широкая" популяция совпадала с отчётной когортой (34 строки из 105), и
# лабораторные KPI считались по переносам размороженных эмбрионов вместо
# свежих циклов. Тот же контракт возврата, что и у _CLINIC_WIDE_LOADERS.
_CLINIC_WIDE_FROM_SOURCE = {"fertimed": fertimed.prepare_wide}
# Клиники, чья широкая lab-популяция несёт дополнительные "сырые" колонки,
# нужные клиничному ПГТ-модулю выше (embedded PGT-A counts для GGRC — см.
# ggrc.PGT_RAW_COLUMNS/ggrc.analyze_reports).
_CLINIC_PGT_EXTRA_COLUMNS = {"ggrc": ggrc.PGT_RAW_COLUMNS}


def _merge(base: dict, override: dict) -> dict:
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def _settings(config_dir: Path) -> dict:
    with (config_dir / "settings.yaml").open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    # config/machine.local.yaml — машинно-зависимые пути к внешним инструментам
    # (DT/TabPFN/DNN/TabPFN-3), ОБЩИЕ для всех клиник на этом ПК. Подмешивается
    # раньше клиничного settings.local.yaml, чтобы редкий клиничный override
    # (напр. отдельный run_cmd) мог выиграть при необходимости. Единственный файл,
    # который нужно поменять при переносе всего проекта на новую машину — см.
    # config/machine.local.yaml.example.
    machine_cfg = ROOT / "config" / "machine.local.yaml"
    if machine_cfg.exists():
        with machine_cfg.open(encoding="utf-8") as stream:
            _merge(cfg, yaml.safe_load(stream) or {})
    local = config_dir / "settings.local.yaml"
    if local.exists():
        with local.open(encoding="utf-8") as stream:
            _merge(cfg, yaml.safe_load(stream) or {})
    return cfg


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _input_path(cfg: dict, explicit: str | None) -> Path:
    if explicit:
        path = _project_path(explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    inbox = _project_path(cfg["paths"]["inbox_dir"])
    candidates = []
    for pattern in ("*.xlsx", "*.xls", "*.csv"):
        candidates.extend(inbox.glob(pattern))
    # "*_cleaned.xlsx" — фиксированный, документированный суффикс отдельного
    # skill'а .claude/skills/reformat-1c-report (SKILL.md: "независимые
    # процессы", основной пайплайн не должен его читать/автоопределять,
    # даже если файл лежит в том же inbox_dir и оказался новее сырой выгрузки).
    candidates = [p for p in candidates if not p.stem.endswith("_cleaned")
                  and not p.name.startswith("~$")]
    if not candidates:
        raise FileNotFoundError(f"нет Excel/CSV в {inbox}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


_OPU_ALIASES = {
    "patient_id": "ID", "patient_name": "Patient Full Name", "date_opu": "Date OPU",
    "date_et": "Date ET", "age": "Age", "attempt_no": "# Attempt",
    "follicles": "N folicules OPU", "afc": "Antral follicule count", "occ": "OCC",
    "mii": "MII", "inseminated": "Inseminated", "two_pn": "2pN",
    "three_pn": "3pN", "degrade": "Deg", "cleavage_d3": "Cleavage",
    "bl_total": "Bl", "bl_good": "Good Bl", "frozen": "Cryo",
    "transferred": "ET", "transfer_day": "Day of ET", "amh": "AMH", "bmi": "BMI",
    "fertilization_method": "IVF/ICSI", "embryologist": "Embryologist", "doctor": "Doctor",
}


def _add_opu_aliases(frame: pd.DataFrame, clinic: str | None = None) -> pd.DataFrame:
    out = frame.copy()
    for canonical, alias in _OPU_ALIASES.items():
        if canonical in out.columns:
            if alias in out.columns:
                out[alias] = out[alias].combine_first(out[canonical])
            else:
                out[alias] = out[canonical]
    if "Preg" in out.columns:
        out["Preg"] = pd.to_numeric(out["Preg"], errors="coerce")
    if "IVF/ICSI" not in out.columns:
        out["IVF/ICSI"] = "ICSI"
    else:
        out["IVF/ICSI"] = out["IVF/ICSI"].fillna("ICSI")
    # "_key" = ID + Date OPU: a patient can have repeat cycles (attempts) sharing one
    # passport ID, so the join/cycle identity must be per-cycle, not per-patient. "ID"
    # itself is left untouched (real passport ID) for human-facing OPU output.
    if clinic == "skyfert":
        # Skyfert has no date columns at all, and its "ЭМК" (patient_id) repeats
        # across physically different patients within one export (documented
        # collision example in docs/SKYFERT.md) — the ID+Date OPU key below, and
        # its bare-ID fallback, are both wrong here. skyfert.build_cycle_key()
        # (patient_id + positional row sequence) is the only reliable per-cycle
        # identity for this clinic; must be checked before the generic branches.
        out["_key"] = skyfert.build_cycle_key(out)
    elif "patient_id" in out.columns and "date_opu" in out.columns:
        date_part = pd.to_datetime(out["date_opu"], errors="coerce").dt.strftime("%Y%m%d")
        key = (out["patient_id"].astype("string").str.strip().str.upper()
               + "#" + date_part.fillna("NA"))
        # Fertimed only: 1C carries the original fresh "Дата пункции" forward onto
        # every subsequent cryo-transfer row for that retrieval, so a patient with
        # several separate FET attempts from one OPU collides on ID+Date OPU alone
        # (confirmed on the "6 месяцев 2026" export, owner decision 2026-07-19).
        # Clinic A's key is left untouched to preserve cycle_uid continuity across
        # runs (MIRI/lot history is keyed by it there; Fertimed has neither).
        if clinic == "fertimed" and "date_et" in out.columns:
            et_part = pd.to_datetime(out["date_et"], errors="coerce").dt.strftime("%Y%m%d")
            key = key + "#" + et_part.fillna("NA")
        out["_key"] = key
    elif "ID" in out.columns:
        out["_key"] = out["ID"]
    return out


# dt_postprocess.py (external DT project) reads its batch Excel via fixed Russian
# headers (COL_ID/COL_FIO/... in that project, not ours) instead of the English
# OPU aliases above; a row with a blank ФИО/Возраст is silently skipped rather than
# erroring, so this mapping must stay in sync with that project's COL_* constants.
_DT_ALIASES = {
    # "_key" (ID+Date OPU composite), not raw "ID": DT echoes this column back verbatim
    # as its own row identifier, and repeat-attempt patients share one raw ID.
    "_key": "Номер карты пациента", "Patient Full Name": "ФИО", "DOB": "Дата рождения",
    "AMH": "АМГ", "Antral follicule count": "КАФ", "MII": "Число MII",
    "Cryo": "Заморожено бластоцист",  # confirmed by owner: cryo is always blastocyst-stage here
    "Date OPU": "Дата пункции", "Date ET": "Дата переноса", "Day of ET": "День переноса",
}


# Текст исхода, который внешний DT разбирает ПРАВИЛЬНО. Его binarize_outcome
# (dt_postprocess.py:97) начинается со строки
#     if pd.isna(val) or str(val).strip() in ("", "0", "нет", "Нет"): return np.nan
# — то есть "Нет" объявляется НЕИЗВЕСТНЫМ исходом, и строка выпадает из
# dt_analytics_with_outcome_*.csv целиком; ветка `if s in ("нет","no"): return 0`
# пятью строками ниже недостижима. Пока мы отдавали "Да"/"Нет", в отчёт попадали
# ТОЛЬКО беременности: все пять последних выходов DT на этой машине (289/289/10/
# 113/15 строк, клиника A и Fertimed) содержат Preg=1 и ни одного Preg=0.
# Июньская когорта Fertimed: 34 цикла (15 беременностей, 19 неудач) -> 15 строк.
# "Беременность не наступила" первый guard пропускает, а следующая же проверка
# ("не наступила" in s) даёт 0 — то же значение, тот же DT, без правок внешнего
# проекта. Решение владельца 2026-07-31: чинить у себя, для всех клиник.
# Колонка идёт ТОЛЬКО во вход DT; человекочитаемая таблица OPU её не берёт.
DT_OUTCOME_TEXT = {1.0: "Беременность +28 дней", 0.0: "Беременность не наступила"}


def _add_dt_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for source, target in _DT_ALIASES.items():
        if source in out.columns and target not in out.columns:
            out[target] = out[source]
    preg = pd.to_numeric(out.get("Preg"), errors="coerce")
    out["Исход переноса"] = preg.map(DT_OUTCOME_TEXT).fillna("")
    return out


def _normalise_source(path: Path, columns_cfg: dict, cfg: dict, clinic: str | None = None):
    raw = ingest.load_table(str(path))
    normalizer = _CLINIC_NORMALIZERS.get(clinic)
    if normalizer:
        normalized, unknown = normalizer(raw, columns_cfg, cfg)
    else:
        resolved, unknown, missing = ingest.resolve_columns(raw, columns_cfg)
        if missing:
            raise ValueError(f"нет обязательных колонок: {missing}")
        normalized = normalize.normalize(resolved, columns_cfg, cfg)
    return raw, normalized, unknown


def _eligible_cycles(normalized: pd.DataFrame) -> pd.DataFrame:
    transferred = pd.to_numeric(normalized["transferred"], errors="coerce")
    mask = transferred.gt(0) & normalized["Preg"].notna()
    selected = normalized.loc[mask].copy()
    if selected.empty:
        raise ValueError("нет строк одновременно с переносом ET>0 и заполненным исходом")
    computed = kpi.compute_kpi(selected)
    if computed.empty:
        raise ValueError("после принятого фильтра 2PN!=0 не осталось циклов")
    return computed


def _model_fallback_map(cfg: dict) -> dict:
    """canonical column -> column holding its model-only substitute."""
    quirks = cfg.get("quirks", {}) or {}
    mapping = dict(quirks.get("model_fallback_columns") or {})
    legacy = quirks.get("mii_model_fallback_column")
    if legacy and "mii" not in mapping:
        mapping["mii"] = legacy
    return mapping


def _apply_model_fallbacks(normalized: pd.DataFrame, mapping: dict) -> tuple[pd.DataFrame, dict]:
    """Substitute the model-only values into the canonical columns.

    Applied to the pre-KPI frame on purpose: everything derived from these
    columns (OCCR/MIIR/3pNR, the Russian DT feature names, Model_KPIScore) is
    then recomputed from the substituted values, so the model cohort stays
    internally consistent instead of mixing filled counts with rates computed
    off the true ones.  The laboratory cohort never sees this frame.
    """
    applied: dict = {}
    if not mapping:
        return normalized, applied
    out = normalized.copy()
    for canon, source in mapping.items():
        if canon not in out.columns or source not in out.columns:
            continue
        target = pd.to_numeric(out[canon], errors="coerce")
        fallback = pd.to_numeric(out[source], errors="coerce")
        filled = target.isna() & fallback.notna()
        if not bool(filled.any()):
            continue
        out[canon] = target.where(target.notna(), fallback)
        applied[canon] = int(filled.sum())
    return out, applied


def _dedupe_cycle_keys(frame: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.Series]:
    """Схлопывает строки с одинаковым "_key", оставляя наиболее полную.

    Владелец (2026-07-31): при повторе ключа берём строку, содержащую больше
    информации. Подтверждённый случай — июньская выгрузка Fertimed, пациентка с
    одной пункцией и одним переносом двумя строками, различающимися только
    заполненным/пустым "Число ОКК".

    Полнота = число непустых значений в строке. Ничего не СЛИВАЕТСЯ: берётся одна
    целая строка, а не собранная из двух — иначе воронка одного цикла оказалась бы
    склеена из двух разных записей МИС. При равной полноте выигрывает первая по
    порядку выгрузки (детерминированно между прогонами).

    Возвращает (кадр без дублей, булева маска оставленных строк). Маска нужна,
    чтобы к модельному кадру применить ТО ЖЕ решение позиционно: считать полноту
    там заново нельзя — model-подстановки заполняют пропуски, и «более полной»
    могла бы оказаться другая строка, а когорты обязаны остаться построчно
    идентичными.
    """
    if "_key" not in frame.columns:
        return frame, pd.Series(True, index=frame.index)
    duplicated = frame["_key"].duplicated(keep=False)
    if not bool(duplicated.any()):
        return frame, pd.Series(True, index=frame.index)
    completeness = frame.notna().sum(axis=1)
    order = pd.DataFrame({"_key": frame["_key"], "_completeness": completeness})
    # sort stable -> при равной полноте сохраняется исходный порядок строк
    keep_index = (order.sort_values("_completeness", ascending=False, kind="stable")
                       .drop_duplicates("_key", keep="first").index)
    mask = frame.index.isin(keep_index)
    dropped = frame.loc[~mask]
    print(f"[dedupe:{label}] снято дублей ключа: {len(dropped)} "
          f"(ключи: {', '.join(sorted(set(dropped['_key'].astype(str))))})",
          file=sys.stderr)
    return frame.loc[mask].copy(), pd.Series(mask, index=frame.index)


def _prepare_opu(cfg: dict, input_path: Path, columns_cfg: dict, clinic: str | None = None) -> pd.DataFrame:
    configured = _project_path(cfg["paths"]["opu_table"])
    source = configured if configured.is_file() else input_path
    _, normalized, _ = _normalise_source(source, columns_cfg, cfg, clinic=clinic)
    aliased = _add_opu_aliases(normalized, clinic=clinic)
    # Кадр OPU читается ОТДЕЛЬНЫМ путём (может быть другой файл — paths.opu_table),
    # поэтому дублирующийся "_key" надо снять и здесь: reconcile.build_dt_predictions
    # джойнит по нему прогнозы с исходами и требует уникальности. Правило то же, что
    # и для когорт (наиболее полная строка), поэтому выживает та же запись.
    deduped, _mask = _dedupe_cycle_keys(aliased, "кадр OPU")
    return deduped


def _read_dt(path: str | None, cfg: dict, model_input: Path) -> pd.DataFrame:
    if path:
        frame = ingest.load_table(str(_project_path(path)))
        contracts.DT_OUTPUT.validate(frame)
        return frame
    dt_cfg = cfg["external_tools"]["digital_twin"]
    target = dt_cfg.get("input_target")
    if target:
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_input, target)
    return dt_runner.launch_and_read(dt_cfg, str(model_input))


def _read_tabpfn(path: str | None, cfg: dict, model_input: Path) -> pd.DataFrame:
    if path:
        return tabpfn_runner.read_tabpfn(explicit_path=_project_path(path))
    return tabpfn_runner.launch_and_read(cfg["external_tools"]["tabpfn"], str(model_input))


def _join_dnn(predictions: pd.DataFrame, dnn: pd.DataFrame | None) -> pd.DataFrame:
    if dnn is None or dnn.empty:
        return predictions
    left = predictions.copy()
    right = dnn.copy()
    left["__dnn_key"] = left["patient_id"].astype("string").str.strip().str.upper()
    right["__dnn_key"] = right["patient_id"].astype("string").str.strip().str.upper()
    keep = ["__dnn_key", "Вероятность положительного исхода", "dnn_validation_status"]
    right = right[keep].rename(columns={"Вероятность положительного исхода": "DNN"})
    out = left.merge(right, on="__dnn_key", how="left", validate="one_to_one")
    return out.drop(columns="__dnn_key")


def _records(frame: pd.DataFrame | None) -> list[dict]:
    return [] if frame is None else frame.replace({np.nan: None}).to_dict("records")


def _final_excel(ares: dict, predictions: pd.DataFrame, miri_summary: pd.DataFrame,
                 miri_events: pd.DataFrame, lot_summary: pd.DataFrame,
                 lot_events: pd.DataFrame, regression_qc_frame: pd.DataFrame,
                 regression_calibration: pd.DataFrame, ozema_cycles: pd.DataFrame,
                 ozema_candidates: pd.DataFrame, pgt_summary: pd.DataFrame,
                 pgt_events: pd.DataFrame, target: Path) -> None:
    df_u = ares["df_u"]
    preg = df_u.loc[pd.to_numeric(df_u.get("Preg"), errors="coerce").notna()].copy()
    regression_sheet = regression_qc_frame.copy()
    if "out_of_interval" in regression_sheet.columns:
        # Excel cells can't hold a Python list; the JSON agent packet keeps the list form.
        regression_sheet["out_of_interval"] = regression_sheet["out_of_interval"].map(
            lambda stages: ", ".join(stages) if isinstance(stages, list) else stages)
    sheets = {
        "Summary": analytics.build_summary(df_u),
        "FullData": df_u.drop(columns=[c for c in df_u if c.startswith("_")], errors="ignore"),
        "WithOutcomes": preg.drop(columns=[c for c in preg if c.startswith("_")], errors="ignore"),
        "Calibration": ares["calibration"], "Conversions": ares["conversion"],
        "ReviewList": ares["review_list"],
        "KPI Cycle": ares["kpi_cycle"].drop(columns=[c for c in ares["kpi_cycle"] if c.startswith("_")], errors="ignore"),
        "Reconciliation": predictions[[c for c in ["patient_id", "cycle_uid", "opu_join_status",
                                                     "tabpfn_join_status", "ci_width_pp", "ci_wide"]
                                               if c in predictions]],
        "MIRI Summary": miri_summary,
        "MIRI Events": miri_events,
        "Lot Summary": lot_summary,
        "Lot Events": lot_events,
        "Regression QC": regression_sheet,
        "Regression Calibration": regression_calibration,
        "OZEMA Cycles": ozema_cycles,
        "OZEMA Genetic Counseling": ozema_candidates,
        "PGT Summary": pgt_summary,
        "PGT Embryo Events": pgt_events,
    }
    labels = {"overall": "KPI Overall", "time": "KPI Time", "method": "KPI Method",
              "embryologist": "KPI Embryologist", "doctor": "KPI Doctor"}
    for key, table in ares["kpi_slices"].items():
        sheets[labels[key]] = table
    _io.atomic_write_excel(sheets, target)


def _pgt_event_map(pgt_events: pd.DataFrame) -> dict:
    """Per-cycle embryo list for the agent packet, keyed by a within-cycle ordinal.

    ``sample_id`` is the genetics lab's OWN embryo code and typically embeds a
    name fragment (e.g. "KUC4" from "Kuchkorova") — it must never reach the agent
    packet. Only ``embryo_<n>`` crosses that boundary; the raw lab sample_id stays
    in the local-only pgt_embryo_events.csv.
    """
    if pgt_events.empty:
        return {}
    keep_pgt = [c for c in
                ["category", "karyotype", "cnv_qc_status", "overall_noise", "sdi", "sd_robust",
                 "interpretation", "recommendation_category"]
                if c in pgt_events.columns]
    event_map = {}
    for uid, group in pgt_events.groupby("cycle_uid"):
        records = group[keep_pgt].to_dict("records")
        for i, rec in enumerate(records, start=1):
            rec["embryo"] = f"embryo_{i}"
        event_map[uid] = records
    return event_map


def _agent_packet(ares: dict, miri_summary: pd.DataFrame, miri_events: pd.DataFrame,
                  lot_summary: pd.DataFrame, lot_events: pd.DataFrame,
                  regression_qc_frame: pd.DataFrame, regression_calibration: pd.DataFrame,
                  outdir: Path, crosswalk: pd.DataFrame, crosswalk_path: Path,
                  dnn_status: str, regression_status: str,
                  ozema_cycles: pd.DataFrame, ozema_candidates: pd.DataFrame,
                  pgt_summary: pd.DataFrame, pgt_events: pd.DataFrame,
                  clinical_cycles: pd.DataFrame, clinical_cohort: dict,
                  miri_baseline: dict | None = None, lot_baseline: dict | None = None) -> Path:
    review = ares["review_list"].copy()
    if review.empty:
        review = ares["df_u"].head(0).copy()
    if not miri_summary.empty:
        review = review.merge(miri_summary, on="cycle_uid", how="left", suffixes=("", "_miri"))
    event_map = {}
    if not miri_events.empty:
        keep = ["event_type", "sensor", "duration_minutes", "max_deviation", "severity",
                "recovery_time_minutes"]
        event_map = {uid: group[keep].to_dict("records")
                     for uid, group in miri_events.groupby("cycle_uid")}
    review["miri_events"] = review.get("cycle_uid", pd.Series(index=review.index, dtype="object")).map(event_map)
    if not lot_summary.empty:
        keep_summary = ["cycle_uid", "n_items_tracked", "n_items_changed", "changed_items",
                        "lot_change_flag"]
        review = review.merge(lot_summary[[c for c in keep_summary if c in lot_summary]],
                              on="cycle_uid", how="left", suffixes=("", "_lots"))
    lot_event_map = {}
    if not lot_events.empty:
        keep_events = ["item_name", "attribute", "event_type", "lot_before", "lot_after", "change_date"]
        lot_event_map = {uid: group[keep_events].to_dict("records")
                         for uid, group in lot_events.groupby("cycle_uid")}
    review["lot_events"] = review.get("cycle_uid", pd.Series(index=review.index, dtype="object")).map(lot_event_map)
    if not regression_qc_frame.empty and "out_of_interval" in regression_qc_frame.columns:
        review = review.merge(regression_qc_frame[["cycle_uid", "out_of_interval"]],
                              on="cycle_uid", how="left", suffixes=("", "_regression"))
    else:
        review["out_of_interval"] = [[] for _ in range(len(review))]
    if not pgt_summary.empty:
        pgt_cols = ["cycle_uid", "n_biopsied", "n_euploid", "n_aneuploid", "n_mosaic",
                    "euploid_rate", "qc_pass_rate", "mean_overall_noise_euploid",
                    "max_overall_noise_euploid", "mean_sdi_euploid", "max_sdi_euploid",
                    "mean_sd_robust_euploid", "max_sd_robust_euploid",
                    "analysis_settings_variable", "pgt_note",
                    # Fertimed-only (f-genetics.com report has no CNV QC/analysis-settings
                    # fields above, but does carry a transfer recommendation + case note).
                    "n_tested", "n_not_tested", "n_recommended",
                    "n_recommended_with_consent", "n_not_recommended", "case_note"]
        review = review.merge(pgt_summary[[c for c in pgt_cols if c in pgt_summary]],
                              on="cycle_uid", how="left", suffixes=("", "_pgt"))
    pgt_event_map = _pgt_event_map(pgt_events)
    review["pgt_embryos"] = review.get("cycle_uid", pd.Series(index=review.index, dtype="object")).map(pgt_event_map)
    # Клинический контекст по-цикловый: только для строк review-list, чтобы
    # clinical-context-analyst мог связать конкретный протокол/дозу/диагноз с
    # уже отобранным проблемным циклом. Когортное распределение — ниже, в cohort.
    if not clinical_cycles.empty:
        clinical_cols = [c for c in clinical.PACKET_COLUMNS if c in clinical_cycles.columns]
        review = review.merge(clinical_cycles[["cycle_uid", *clinical_cols]],
                              on="cycle_uid", how="left", suffixes=("", "_clinical"))
    # Pre-classified per-patient packet (prognosis level, confidence grade, OHSS risk,
    # response phenotype, banking urgency) that dt-narrator's contract expects — Python
    # classifies via consult.py's fixed thresholds, the agent only narrates the result.
    # build_narrative_packet's own output is already PHI-safe (cycle_uid + numeric flags).
    review["narrative"] = [consult.build_narrative_packet(r) for r in review.to_dict("records")]
    allowed = [
        "age", "Age", "Preg", "PRAI", "DIGITAL TWIN", "DNN", "ci_width_pp", "ci_wide",
        "priority", "reason", "p_per_transfer", "p_kat_raw", "p_nvsa", "p_csdi",
        "OCC", "MII", "2pN", "Bl", "Good Bl", "event_count", "warning_count",
        "critical_count", "gap_count", "max_severity", "warning", "miri_events",
        "n_items_tracked", "n_items_changed", "changed_items", "lot_change_flag",
        "lot_events", "narrative", "out_of_interval",
        "n_biopsied", "n_euploid", "n_aneuploid", "n_mosaic", "euploid_rate",
        "qc_pass_rate", "mean_overall_noise_euploid", "max_overall_noise_euploid",
        "mean_sdi_euploid", "max_sdi_euploid", "mean_sd_robust_euploid",
        "max_sd_robust_euploid", "analysis_settings_variable", "pgt_embryos", "pgt_note",
        "n_tested", "n_not_tested", "n_recommended", "n_recommended_with_consent",
        "n_not_recommended", "case_note",
        # Описательный клинический слой (см. src/embryo/clinical.py). Коды МКБ-10
        # и параметры протокола — аналитические признаки, не идентификаторы:
        # ни ФИО, ни ID, ни дат рождения здесь нет, только производные "сколько
        # дней от старта цикла до пункции".
        *clinical.PACKET_COLUMNS,
    ]
    safe = privacy.deidentify_records(review, allowed)
    # Privacy-safe counts only: OZEMA's own per-cycle table carries raw patient_id/
    # name (local-only, see ozema.py), never passed to an agent at that granularity.
    ozema_flagged = int(ozema_cycles.get("flagged", pd.Series(dtype=bool)).sum()) if not ozema_cycles.empty else 0
    ozema_candidate_patients = int(ozema_candidates["patient_id"].nunique()) if not ozema_candidates.empty else 0
    # Cluster/KAT-vs-PRAI/similarity-graph analysis added to the report: cohort-level
    # only (no patient identifiers), so the lead-analyst can reference and explain
    # it in the final synthesis instead of it living only inside the HTML charts.
    cohort = {
        "n_cycles": int(ares["n"]), "n_with_outcome": int(ares["n_with_outcome"]),
        "dnn_status": dnn_status, "regression_status": regression_status,
        "kpi_aggregate": _records(ares["kpi_slices"].get("overall")),
        "calibration": _records(ares["calibration"]),
        "conversion": _records(ares["conversion"]),
        "regression_calibration": _records(regression_calibration),
        "ozema_flagged_cycles": ozema_flagged,
        "ozema_genetic_counseling_candidate_patients": ozema_candidate_patients,
        # Flagged (review-list) vs baseline (rest of the dated cohort) rate, so a
        # forensics agent can tell a real co-occurrence from cohort-wide base-rate
        # noise instead of only ever seeing the flagged group in isolation.
        "miri_baseline_comparison": miri_baseline,
        "lot_baseline_comparison": lot_baseline,
        # Клинический контекст когорты: распределения протоколов, доз, дней МЦ и
        # групп МКБ-10 вместе с заполненностью каждого поля. Пусто у клиник, чья
        # выгрузка этих полей не содержит.
        "clinical_context": clinical_cohort or None,
        **report.build_analysis_summary(ares["df_u"]),
    }
    target = outdir / "analysis_bundle.json"
    privacy.write_agent_bundle(safe, cohort, target, crosswalk, crosswalk_path)
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="End-to-end embryo laboratory analysis")
    parser.add_argument("input", nargs="?")
    parser.add_argument("--dt-output", help="детерминированный replay без запуска DT")
    parser.add_argument("--tabpfn-output", help="детерминированный replay без запуска TabPFN")
    parser.add_argument("--regression-output",
                        help="детерминированный replay без запуска TabPFN-3 regression")
    parser.add_argument("--skip-miri", action="store_true")
    parser.add_argument("--skip-lots", action="store_true")
    parser.add_argument("--skip-pgt", action="store_true")
    parser.add_argument("--clinic",
                        help="config/clinics/<name> вместо config/ (клиника A по умолчанию, "
                             "поведение без флага не меняется)")
    parser.add_argument("--fertimed", action="store_true",
                        help="ярлык для --clinic fertimed")
    parser.add_argument("--ggrc", action="store_true",
                        help="ярлык для --clinic ggrc")
    args = parser.parse_args(argv)
    if args.fertimed and not args.clinic:
        args.clinic = "fertimed"
    if args.ggrc and not args.clinic:
        args.clinic = "ggrc"

    os.chdir(ROOT)
    config_dir = ROOT / "config" if not args.clinic else ROOT / "config" / "clinics" / args.clinic
    cfg = _settings(config_dir)
    columns_cfg = ingest.load_config(str(config_dir / "columns.yaml"))
    input_path = _input_path(cfg, args.input)
    processed = _project_path(cfg["paths"]["processed_dir"])
    private = _project_path(cfg["paths"]["local_private_dir"])
    packets = _project_path(cfg["paths"]["agent_packets_dir"])
    for directory in (processed, private, packets):
        directory.mkdir(parents=True, exist_ok=True)
    print(f"[input] {input_path}")

    source_raw, normalized, unknown = _normalise_source(input_path, columns_cfg, cfg, clinic=args.clinic)
    kpi_cycles = _eligible_cycles(normalized)
    secret = privacy.load_or_create_secret(_project_path(cfg["privacy"]["secret_file"]))
    # "_key" (ID+Date OPU) must exist before hashing cycle_uid: a repeat-attempt
    # patient shares one raw patient_id across cycles, so cycle_uid needs the
    # per-cycle composite, not the per-patient raw ID.
    kpi_cycles = _add_opu_aliases(kpi_cycles, clinic=args.clinic)
    if args.clinic == "skyfert":
        # Skyfert's export has no name column at all, and the external Digital
        # Twin project skips any row with a blank "ФИО" outright (dt_postprocess.py:
        # "if fio.lower() in ('итого','') or not fio: skip"). The clinic already
        # de-identifies its export down to "ЭМК" (patient_id) only, so using it as
        # the ФИО placeholder introduces no new PHI exposure (owner-confirmed
        # 2026-07-21) — it only exists to keep DT from dropping every row.
        kpi_cycles["Patient Full Name"] = kpi_cycles["patient_id"]
    # Повторяющийся "_key" — не всегда дефект схемы: МИС может записать один цикл
    # двумя строками разной полноты. Решение владельца (2026-07-31) — оставить
    # наиболее полную строку; маска нужна, чтобы то же решение позиционно
    # применилось к модельному кадру (см. _dedupe_cycle_keys).
    kpi_cycles, kpi_keep_mask = _dedupe_cycle_keys(kpi_cycles, "когорта отчёта")
    kpi_cycles = privacy.add_cycle_uid(kpi_cycles, "_key", secret)

    # Wider lab-stage population (NOT gated by transferred>0/Preg known): the source
    # of EVERY Vienna KPI table and of OZEMA, all of which judge lab performance
    # regardless of whether a transfer happened (matches the owner's Vienna-style
    # dashboard and OZEMA sheet, which flag freeze-all/no-ET cycles too; reaffirmed
    # 2026-07-31). Three ways to obtain it:
    #   * separate FILE (GGRC, see docs/GGRC.md) — _CLINIC_WIDE_LOADERS;
    #   * same file, own row filter (Fertimed, whose report cohort is gated by a
    #     filled transfer outcome) — _CLINIC_WIDE_FROM_SOURCE;
    #   * otherwise the normalized frame as is — it is already ungated for clinics
    #     whose normalizer applies no outcome filter.
    wide_loader = _CLINIC_WIDE_LOADERS.get(args.clinic)
    wide_from_source = _CLINIC_WIDE_FROM_SOURCE.get(args.clinic)
    if wide_loader:
        wide_path = _project_path(cfg["paths"]["kpi_wide_file"])
        wide_raw = ingest.load_table(str(wide_path))
        wide_normalized, _ = wide_loader(wide_raw, columns_cfg, cfg)
        lab_cycles = _add_opu_aliases(wide_normalized, clinic=args.clinic)
    elif wide_from_source:
        wide_normalized, _ = wide_from_source(source_raw, columns_cfg, cfg)
        lab_cycles = _add_opu_aliases(wide_normalized, clinic=args.clinic)
    else:
        lab_cycles = _add_opu_aliases(normalized, clinic=args.clinic)
    lab_cycles, _ = _dedupe_cycle_keys(lab_cycles, "широкая lab-популяция")
    lab_cycles = privacy.add_cycle_uid(lab_cycles, "_key", secret)

    # Описательный клинический контекст цикла: тип протокола стимуляции, суммарная
    # доза ФСГ, триггер, дни МЦ, длительность стимуляции, базальный ФСГ, КАФ, коды
    # МКБ-10 и причина завершения программы. По-цикловый слой строится на когорте
    # отчёта (kpi_cycles) — иначе доли по диагнозам и исходы считались бы по разным
    # знаменателям. Ни в KPI, ни в фильтрацию, ни во вход DT/TabPFN слой не входит.
    #
    # Исключение — причина завершения программы. В когорте отчёта она по определению
    # почти всегда "Перенос эмбрионов", а содержательные причины (остановка развития,
    # отсутствие оплодотворения, овуляция) живут в строках, до переноса не дошедших.
    # Брать для них "широкую lab-популяцию" пайплайна НЕЛЬЗЯ: у Fertimed она уже
    # отфильтрована клиничным _cohort_mask (требует заполненного исхода переноса) и
    # в июньском прогоне давала ровно те же 34 строки, что и когорта отчёта — блок
    # выходил с честной на вид, но неверной подписью "без гейта". Поэтому сюда
    # подаётся резолвнутая выгрузка ДО когортного фильтра: 105 строк июня против 34,
    # и в них 8 остановок развития, 3 овуляции, 2 отсутствия оплодотворения.
    clinical_cycles = pd.DataFrame(columns=["cycle_uid"])
    clinical_cohort: dict = {}
    if clinical.has_clinical_columns(kpi_cycles):
        diagnosis_groups = clinical.load_diagnosis_groups(config_dir / "diagnosis_groups.yaml")
        export_population, _, _ = ingest.resolve_columns(source_raw, columns_cfg)
        clinical_cycles, clinical_cohort = clinical.build_clinical_context(
            kpi_cycles, diagnosis_groups, export_population=export_population)
        _io.atomic_write_csv(clinical_cycles, processed / "clinical_context.csv")
        print(f"[clinical] клинический контекст: {len(clinical_cycles)} циклов когорты отчёта, "
              f"причина завершения по {len(export_population)} строкам выгрузки "
              f"(до когортного фильтра), справочник диагнозов: "
              f"{'да' if diagnosis_groups else 'нет'}")
    ozema_cycles = ozema.build_ozema_table(lab_cycles, columns_cfg)
    ozema_candidates = ozema.genetic_counseling_candidates(ozema_cycles)
    _io.atomic_write_csv(ozema_cycles, processed / "ozema_cycles.csv")
    _io.atomic_write_csv(ozema_candidates, processed / "ozema_genetic_counseling_candidates.csv")

    # Клиника B (и любая другая с quirks.model_fallback_columns): вход DT/TabPFN/
    # DNN/TabPFN-3 получает заполненную цепочку MII->OCC->фолликулы, чтобы
    # нейросети не теряли циклы с неполной выгрузкой. Это ТОЛЬКО model_input:
    # kpi_computed.csv / OPU_table_enriched.xlsx, лабораторные OCCR/MIIR/3pNR и
    # OZEMA продолжают считаться по фактически измеренному (df kpi_cycles не
    # трогается). Подставлять производные значения в знаменатель Vienna KPI
    # нельзя: OCCR=OCC/фолликулы и MIIR=MII/OCC вырождаются ровно в 1.0.
    # См. src/embryo/clinics/fertimed.py.
    fallback_map = _model_fallback_map(cfg)
    model_normalized, model_fallbacks = _apply_model_fallbacks(normalized, fallback_map)
    if model_fallbacks:
        model_source = _eligible_cycles(model_normalized)
        model_source = _add_opu_aliases(model_source, clinic=args.clinic)
        if args.clinic == "skyfert":
            model_source["Patient Full Name"] = model_source["patient_id"]
        # То же решение по дублям, что и в лабораторной когорте, применённое
        # ПОЗИЦИОННО. Пересчитывать полноту здесь нельзя: model-подстановки
        # заполняют пропуски, поэтому «более полной» могла бы оказаться другая
        # строка пары — и когорты разъехались бы при идентичных ключах.
        if not bool(kpi_keep_mask.all()):
            if len(model_source) != len(kpi_keep_mask):
                raise ValueError("модельная когорта до дедупликации не совпала с лабораторной")
            model_source = model_source.loc[kpi_keep_mask.to_numpy()].copy()
        model_source = privacy.add_cycle_uid(model_source, "_key", secret)
        # The substitutions touch no column the cohort filter reads, so the two
        # cohorts must stay row-for-row identical; a drift here would silently
        # mis-join model output back onto the laboratory cohort.
        if list(model_source["cycle_uid"]) != list(kpi_cycles["cycle_uid"]):
            raise ValueError("model-подстановки изменили состав когорты — join разъедется")
        print(f"[model-input] подстановки только для моделей: {model_fallbacks}")
    else:
        model_source = kpi_cycles
    model_input = _add_dt_aliases(model_source)
    model_input_path = private / "model_input.xlsx"
    _io.atomic_write_excel({"model_input": model_input}, model_input_path)
    _io.atomic_write_csv(kpi_cycles, processed / "kpi_computed.csv")
    _io.atomic_write_text(json.dumps({"schema_version": "1.0", "unknown_columns": list(map(str, unknown))},
                                     ensure_ascii=False, indent=2),
                          packets / "schema_diagnostic.json")

    opu = _prepare_opu(cfg, input_path, columns_cfg, clinic=args.clinic)
    dt = _read_dt(args.dt_output, cfg, model_input_path)
    if "_key" in dt.columns:
        # DT's own "ready" export accumulates every patient it has ever processed
        # (see dt_runner.read_dt docstring) — scope it down to this run's cycles so
        # unrelated historical rows never enter dt_predictions.csv/review-list.
        current_keys = set(kpi_cycles["_key"].astype("string").str.strip().str.upper())
        dt_keys = dt["_key"].astype("string").str.strip().str.upper()
        dt = dt.loc[dt_keys.isin(current_keys)].copy()
    tab = _read_tabpfn(args.tabpfn_output, cfg, model_input_path)
    predictions = reconcile.build_dt_predictions(
        dt, tab, opu, secret, threshold_pp=cfg["thresholds"]["tabpfn_ci_width_pp"])

    dnn_frame = None
    dnn_status = "disabled"
    dnn_coverage: dict | None = None
    dcfg = cfg["external_tools"].get("dnn_verifier", {})
    if dcfg.get("enabled"):
        # EVO — необязательная кросс-проверка (см. settings.local.yaml), не должна
        # гейтить пайплайн: тот же мягкий паттерн, что и у tabpfn_regression ниже.
        try:
            dnn_frame = dnn_runner.run_dnn_verifier(dcfg, str(model_input_path))
            dnn_status = "valid_saved_scaler"
            predictions = _join_dnn(predictions, dnn_frame)
            # Успешный запуск ещё не означает покрытую когорту: Evo отбрасывает
            # строку с пропуском в любом из 18 признаков обучения, поэтому статус
            # "valid_saved_scaler" может стоять при почти пустой колонке DNN в
            # отчёте (Fertimed, июнь 2026: 2 строки из 34 — у криопереносов не
            # заполнены "Число Bl"/"Число Bl хор.кач-ва"). Считаем покрытие явно,
            # иначе молчаливо пустая колонка читается как "модель не запускалась".
            scored = int(pd.to_numeric(predictions.get("DNN"), errors="coerce").notna().sum()) \
                if "DNN" in predictions.columns else 0
            dnn_coverage = {"scored": scored, "cohort": int(len(predictions))}
            if scored < len(predictions):
                print(f"[warn] DNN оценила {scored} из {len(predictions)} циклов когорты: "
                      "остальные строки отброшены из-за пропусков в признаках обучения",
                      file=sys.stderr)
        except Exception as exc:
            dnn_status = f"failed: {type(exc).__name__}: {exc}"
            print(f"[warn] DNN verifier skipped: {dnn_status}", file=sys.stderr)

    # TabPFN-3 quantitative-stage regression: independent QA layer (actual vs P10-P90
    # per funnel stage), never the outcome-probability reconciliation above. Failure is
    # intentionally soft — this model augments DT/TabPFN/DNN, it does not gate them.
    regression_status = "disabled"
    regression_qc_frame = pd.DataFrame(columns=["cycle_uid", "out_of_interval"])
    regression_calibration = pd.DataFrame(columns=["stage", "n_evaluated", "share_in_interval80", "mae_p50"])
    rcfg = cfg["external_tools"].get("tabpfn_regression", {})
    if rcfg.get("enabled"):
        try:
            if args.regression_output:
                raw_regression = pd.read_csv(_project_path(args.regression_output), encoding="utf-8-sig")
                contracts.validate_regression_quantiles(raw_regression)
            else:
                raw_regression = tabpfn_regression_runner.run_tabpfn_regression(rcfg, str(model_input_path))
            clean_regression, stages_present = tabpfn_regression_runner.extract_predictions(raw_regression)
            regression_qc_frame, regression_calibration = regression_qc.build_regression_qc(
                kpi_cycles, clean_regression, stages_present)
            regression_status = "ok"
        except Exception as exc:
            regression_status = f"failed: {type(exc).__name__}: {exc}"
            print(f"[warn] TabPFN-3 regression skipped: {regression_status}", file=sys.stderr)

    enriched = reconcile.enrich_opu_table(opu, predictions)
    if args.clinic == "skyfert":
        # enrich_opu_table/ensure_opu_schema only ever look at the Title-Case
        # "Patient Full Name" OPU column; DT's own echo of our ЭМК placeholder
        # comes back as its lowercase "patient_name" field (visible in the HTML
        # report/review-list via report._first_present's fallback), which
        # ensure_opu_schema's fixed column selection silently drops. Without
        # this, the human-facing OPU_table_enriched.xlsx would show a blank
        # name next to a populated "ID" (ЭМК) column.
        enriched["Patient Full Name"] = enriched["ID"]
    exact_opu = opu_table.ensure_opu_schema(enriched)
    crosswalk_path = _project_path(cfg["privacy"]["crosswalk_file"])
    outputs = reconcile.write_reconciled_outputs(
        predictions, exact_opu, processed / "dt_predictions.csv",
        processed / "OPU_table_enriched.xlsx", crosswalk_path)
    # Add local names to the crosswalk; they never enter the packet.
    crosswalk = privacy.build_crosswalk(predictions, "patient_id")
    if "Patient Full Name" in enriched:
        names = enriched[["ID", "Patient Full Name"]].copy()
        names["__key"] = names["ID"].astype("string").str.strip().str.upper()
        crosswalk["__key"] = crosswalk["patient_id"].astype("string").str.strip().str.upper()
        crosswalk = crosswalk.merge(names[["__key", "Patient Full Name"]], on="__key", how="left").drop(columns="__key")
        _io.atomic_write_csv(crosswalk, crosswalk_path)

    xlsx = processed / "DT_Analytics_Report.xlsx"
    # lab_cycles (широкая популяция) и kpi_cycles (отчётная когорта) — разные
    # наборы строк у Fertimed/GGRC: первый даёт все таблицы KPI, второй нужен
    # построчно, по cycle_uid, чтобы вернуть в df_u истинные до-DT счётчики и
    # найти строки с перенесённой воронкой. Подставлять один вместо другого
    # нельзя: у Fertimed криопереносы есть только в отчётной когорте, и
    # _restore_true_counts по широкой популяции обнулил бы весь df_u.
    ares = analytics.run_analytics(predictions, exact_opu, xlsx,
                                   lab_population=lab_cycles, columns_cfg=columns_cfg,
                                   cohort_population=kpi_cycles)
    _io.atomic_write_csv(ares["review_list"], processed / "review_list.csv")

    date_col = "Date OPU"
    review_patients = pd.DataFrame(columns=["cycle_uid", date_col])
    if not ares["review_list"].empty and date_col in ares["review_list"]:
        review_patients = ares["review_list"][["cycle_uid", date_col]].dropna(subset=[date_col]).copy()
    # Same universe review_list is drawn from (transferred + known outcome), used as
    # the control/baseline group for MIRI and consumable-lot forensics below: without
    # it, "an excursion/lot change co-occurred with a flagged cycle" cannot be told
    # apart from ordinary cohort-wide base-rate noise.
    dated_cohort = pd.DataFrame(columns=["cycle_uid", date_col])
    if date_col in kpi_cycles.columns:
        dated_cohort = (kpi_cycles[["cycle_uid", date_col]].dropna(subset=[date_col])
                        .drop_duplicates(subset="cycle_uid").copy())

    miri_summary = pd.DataFrame(columns=[
        "cycle_uid", "window_start", "window_end", "n_points", "event_count",
        "warning_count", "critical_count", "max_severity", "warning", "gap_count",
        "max_event_duration_minutes", "error",
    ])
    miri_events = pd.DataFrame(columns=incubator.EVENT_COLUMNS)
    miri_baseline: dict | None = None
    miri_enabled = cfg.get("miri_logs", {}).get("enabled", True)
    if not args.skip_miri and miri_enabled and not review_patients.empty:
        mcfg = cfg["miri_logs"]
        miri_summary = incubator.analyze_patients(
            review_patients, str(_project_path(mcfg["base_dir"])), mcfg["incubator"],
            date_col=date_col, days=cfg["thresholds"]["cultivation_window_days"],
            setpoints=mcfg.get("setpoints"), sep=mcfg.get("sep", ";"),
            dt_fmt=mcfg.get("datetime_format", "%Y.%m.%d %H:%M:%S"),
            tolerances=mcfg.get("tolerances"))
        miri_events = miri_summary.attrs.get("events", miri_events)
        if not dated_cohort.empty:
            # Local-only (globs already-synced MIRI CSVs, no network): free to run
            # over the full dated cohort, not just review-list, purely to get a
            # control group for the comparison below.
            miri_full_summary = incubator.analyze_patients(
                dated_cohort, str(_project_path(mcfg["base_dir"])), mcfg["incubator"],
                date_col=date_col, days=cfg["thresholds"]["cultivation_window_days"],
                setpoints=mcfg.get("setpoints"), sep=mcfg.get("sep", ";"),
                dt_fmt=mcfg.get("datetime_format", "%Y.%m.%d %H:%M:%S"),
                tolerances=mcfg.get("tolerances"))
            miri_baseline = incubator.compare_flagged_vs_baseline(
                miri_full_summary, review_patients["cycle_uid"])
    _io.atomic_write_csv(miri_summary, processed / "miri_summary.csv")
    _io.atomic_write_csv(miri_events, processed / "miri_events.csv")
    if miri_baseline is not None:
        _io.atomic_write_text(json.dumps(miri_baseline, ensure_ascii=False, indent=2),
                              processed / "miri_baseline.json")

    lot_summary = pd.DataFrame(columns=[
        "cycle_uid", "window_start", "window_end", "n_items_tracked",
        "n_items_changed", "changed_items", "lot_change_flag", "error", "note",
    ])
    lot_events = pd.DataFrame(columns=lots.EVENT_COLUMNS)
    lot_baseline: dict | None = None
    if not args.skip_lots and cfg.get("consumable_lots", {}).get("enabled") and not review_patients.empty:
        items, changes = lots.fetch_consumable_lots(cfg)
        timeline = lots.build_lot_timeline(items, changes)
        lot_summary = lots.analyze_patients(
            review_patients, timeline, date_col=date_col,
            days=cfg["thresholds"]["cultivation_window_days"])
        lot_events = lot_summary.attrs.get("events", lot_events)
        if not dated_cohort.empty:
            # The Airtable timeline above is already fetched once for the whole run;
            # widening the population it is checked against costs no extra API calls.
            lot_full_summary = lots.analyze_patients(
                dated_cohort, timeline, date_col=date_col,
                days=cfg["thresholds"]["cultivation_window_days"])
            lot_baseline = lots.compare_flagged_vs_baseline(
                lot_full_summary, review_patients["cycle_uid"])
    _io.atomic_write_csv(lot_summary, processed / "lot_summary.csv")
    _io.atomic_write_csv(lot_events, processed / "lot_events.csv")
    if lot_baseline is not None:
        _io.atomic_write_text(json.dumps(lot_baseline, ensure_ascii=False, indent=2),
                              processed / "lot_baseline.json")
    _io.atomic_write_csv(regression_qc_frame, processed / "regression_qc.csv")
    _io.atomic_write_csv(regression_calibration, processed / "regression_calibration.csv")

    pgt_summary = pd.DataFrame(columns=[
        "cycle_uid", "report_date", "pgt_run_id", "match_reason", "n_biopsied",
        "n_euploid", "n_aneuploid", "n_mosaic", "n_other", "euploid_rate",
        "n_qc_pass", "qc_pass_rate", "software_versions", "analysis_settings_variable",
        "aneuploid_chromosomes", "mean_overall_noise_euploid", "max_overall_noise_euploid",
        "mean_sdi_euploid", "max_sdi_euploid", "mean_sd_robust_euploid",
        "max_sd_robust_euploid", "mean_overall_noise_all", "max_overall_noise_all", "pgt_note",
    ])
    pgt_module = _CLINIC_PGT_MODULES.get(args.clinic, pgt)
    pgt_events = pd.DataFrame(columns=pgt_module.EMBRYO_COLUMNS)
    pcfg = cfg.get("pgt", {})
    if not args.skip_pgt and pcfg.get("enabled") and not kpi_cycles.empty:
        reports_dir = _project_path(pcfg.get("reports_dir", "data/processed/local_private/pgt_reports"))
        worklist_cols = (["cycle_uid", "Patient Full Name", "Date OPU", "Date ET"]
                          + _CLINIC_PGT_EXTRA_COLUMNS.get(args.clinic, []))
        worklist_parts = [kpi_cycles[[c for c in worklist_cols if c in kpi_cycles.columns]]]
        if not ares["review_list"].empty:
            worklist_parts.append(
                ares["review_list"][[c for c in worklist_cols if c in ares["review_list"].columns]])
        worklist = pd.concat(worklist_parts, ignore_index=True).drop_duplicates(subset="cycle_uid")
        if reports_dir.is_dir():
            try:
                pgt_summary = pgt_module.analyze_reports(worklist, reports_dir,
                                                          date_col_opu="Date OPU", date_col_et="Date ET")
                pgt_events = pgt_summary.attrs.get("events", pgt_events)
                unmatched = pgt_summary.attrs.get("unmatched", [])
                if unmatched:
                    print(f"[warn] PGT: {len(unmatched)} отчёт(ов) не сопоставлены с cycle_uid: "
                          f"{unmatched}", file=sys.stderr)
            except Exception as exc:
                print(f"[warn] PGT skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
        else:
            print(f"[warn] PGT: reports_dir {reports_dir} не найден, шаг пропущен", file=sys.stderr)
    _io.atomic_write_csv(pgt_summary, processed / "pgt_summary.csv")
    _io.atomic_write_csv(pgt_events, processed / "pgt_embryo_events.csv")

    _final_excel(ares, predictions, miri_summary, miri_events, lot_summary, lot_events,
                 regression_qc_frame, regression_calibration, ozema_cycles, ozema_candidates,
                 pgt_summary, pgt_events, xlsx)

    html = processed / "DT_Analytics_Report.html"
    report_result = report.build_report(
        ares, kpi_cycles, miri_summary, miri_events,
        lot_summary=lot_summary, lot_events=lot_events,
        regression_qc_frame=regression_qc_frame, regression_calibration=regression_calibration,
        ozema_cycles=ozema_cycles, ozema_candidates=ozema_candidates,
        miri_baseline=miri_baseline, lot_baseline=lot_baseline,
        out_html=str(html))
    packet = _agent_packet(ares, miri_summary, miri_events, lot_summary, lot_events,
                           regression_qc_frame, regression_calibration,
                           packets, crosswalk, crosswalk_path, dnn_status, regression_status,
                           ozema_cycles, ozema_candidates, pgt_summary, pgt_events,
                           clinical_cycles, clinical_cohort,
                           miri_baseline, lot_baseline)

    artifact_paths = [Path(outputs["dt_predictions"]), Path(outputs["opu"]), xlsx, html,
                      processed / "review_list.csv", processed / "miri_events.csv",
                      processed / "lot_events.csv", processed / "regression_qc.csv",
                      processed / "ozema_cycles.csv", processed / "pgt_summary.csv",
                      processed / "pgt_embryo_events.csv",
                      processed / "ozema_genetic_counseling_candidates.csv", packet]
    if not clinical_cycles.empty:
        artifact_paths.append(processed / "clinical_context.csv")
    if miri_baseline is not None:
        artifact_paths.append(processed / "miri_baseline.json")
    if lot_baseline is not None:
        artifact_paths.append(processed / "lot_baseline.json")
    manifest = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema_versions": dict(contracts.SCHEMA_VERSIONS),
        "input_sha256": _sha256(input_path), "n_cycles": int(ares["n"]),
        "dnn_status": dnn_status, "regression_status": regression_status,
        # Сколько циклов когорты реально получили прогноз DNN: "valid_saved_scaler"
        # сам по себе этого не говорит (см. расчёт dnn_coverage выше).
        "dnn_coverage": dnn_coverage,
        "n_lab_population": int(len(lab_cycles)),
        "warnings": report_result["warnings"],
        "artifacts": {p.name: {"path": str(p), "sha256": _sha256(p)}
                      for p in artifact_paths if p.is_file()},
    }
    _io.atomic_write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                          processed / "run_manifest.json")

    # Cross-run history: aggregate-only (no cycle_uid, no patient rows), appended to
    # a jsonl that survives across runs (data/processed is not deleted between runs,
    # only individual known-named artifacts are overwritten) so KPI/calibration/
    # forensic trend can be read across many runs instead of re-derived from one.
    history_record = history.build_record(
        manifest, kpi_cycles, ares["review_list"], miri_baseline, lot_baseline,
        regression_calibration, ozema_candidates)
    history_path = history.append_record(history_record, processed / "run_history.jsonl")
    print(f"[done] cycles={ares['n']} review={len(ares['review_list'])} report={html} "
          f"history={history_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FAILED] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
