#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""teable_sync.py — push /run-analysis results of every clinic into Teable.

Reads the finished files in each clinic's `data/processed` folder and writes
them into that clinic's own Teable base (schema: teable_schema.py), then rebuilds
the local patient-card viewer. Nothing is recomputed here — the pipeline's
numbers are uploaded as they are.

    Runs                  one record per analysis run (snapshot anchor)
    Cycles                cycle master, upserted by join key — never duplicated
    Predictions           every model output for one cycle in one run
    Prediction vs Reality funnel prediction-vs-fact with ±30% flags
    Regression QC         TabPFN-3 quantitative funnel forecast (P10/P50/P90)
    Review                the analytics review-list (priority / reason)
    OZEMA                 per-cycle arrest flags (OMA/FF/ZA/EEA)
    Calibration           per-run × model discriminative metrics
    Files                 the processed CSV/XLSX files as attachments

Idempotent between runs:
  * a Run is keyed by the content hash of dt_predictions.csv — re-syncing the
    same files creates no duplicate run (--force overrides);
  * Cycles are upserted by join key, so one cycle stays one record forever and
    history accumulates in Predictions / PvR / Regression QC across runs.

Usage:
    python teable_sync.py --all                  # every clinic in the registry
    python teable_sync.py --clinic fertimed
    python teable_sync.py --all --dry-run        # parse + report, write nothing
    python teable_sync.py --all --force          # load again despite same hash
    python teable_sync.py --all --no-files       # skip attachment upload
    python teable_sync.py --all --no-cards       # skip the viewer rebuild

Dependencies:  pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys

try:
    import pandas as pd
except ImportError:  # pragma: no cover - dependency guard
    sys.exit("Missing dependency 'pandas'. Run: pip install -r requirements.txt")

import cards_build
import teable_common as tc
import teable_schema as ts


# ---------------------------------------------------------------------------
# Column maps — left side is the code's name, right side the candidates tried in
# order against the actual file headers.
# ---------------------------------------------------------------------------

OPU_COLS = {
    # "_key" is the cycle-level join key (raw_id#date_opu) shared with
    # dt_predictions.csv. Bare "ID" repeats across multi-attempt patients and
    # must never be used to join real values against predictions.
    "patient_id":   ["_key", "ID", "patient_id", "Patient ID"],
    "name":         ["Patient Full Name", "ФИО", "Имя", "Name"],
    "age":          ["Age", "Возраст"],
    "amh":          ["AMH", "АМГ"],
    "afc":          ["Antral follicule count", "AFC", "КАФ"],
    "bmi":          ["BMI", "ИМТ"],
    "attempt":      ["# Attempt", "Attempt", "Попытка"],
    "date_opu":     ["Date OPU", "Дата OPU", "Date of OPU"],
    "date_et":      ["Date ET", "Дата ET"],
    "follicles":    ["N folicules OPU", "N follicules OPU", "Количество фолликулов", "Follicles"],
    "occ":          ["OCC", "Число ОКК", "ОКК"],
    "mii":          ["MII", "Число инсеминированных", "МII"],
    "inseminated":  ["Inseminated", "Инсеминировано"],
    "pn2":          ["2pN", "2PN", "2pn", "Число 2pN"],
    "cleavage":     ["Cleavage", "Дробление"],
    "bl":           ["Bl", "Blast", "Бластоцисты", "Число Bl"],
    "good_bl":      ["Good Bl", "Good Blast", "Bl хор.кач-ва", "Число Bl хор.кач-ва"],
    "cryo":         ["Cryo", "Крио"],
    "et":           ["ET", "Перенос"],
    "day_et":       ["Day of ET", "День ET"],
    "preg":         ["Preg", "Исход", "Pregnancy"],
    # PRAI/DIGITAL TWIN из OPU-таблицы не читаются: вероятности только из PRED_COLS.
    "method":       ["IVF/ICSI"],
    "embryologist": ["Embryologist", "Эмбриолог"],
    "doctor":       ["Doctor", "Врач"],
}

PRED_COLS = {
    "patient_id":   ["_key", "patient_id", "ID", "Patient ID"],
    "cycle_uid":    ["cycle_uid"],
    "timestamp":    ["timestamp", "Timestamp", "created"],
    "med_okk":      ["med_okk", "med_OCC"],
    "med_mii":      ["med_mii"],
    "med_pn2":      ["med_pn2"],
    "med_blasts":   ["med_blasts", "med_bl"],
    "med_good":     ["med_good", "med_goodbl"],
    "med_euploid":  ["med_euploid"],
    "prai":         ["PRAI"],
    "ci_low":       ["ci_low", "rate_ci_low"],
    "ci_high":      ["ci_high", "rate_ci_high"],
    "ci_width":     ["ci_width_pp"],
    "ci_wide":      ["ci_wide"],
    "p_per_transfer": ["p_per_transfer"],
    "p_overall":    ["p_overall_cycle", "p_overall"],
    "p_cum":        ["p_cum_if_viable"],
    "p_viable":     ["p_viable"],
    "p_cancel":     ["p_cancel_risk", "p_cancel"],
    "bayes_mean":   ["bayes_mean"],
    "bayes_low":    ["bayes_ci_low"],
    "bayes_high":   ["bayes_ci_high"],
    "p_kat":        ["p_kat_raw", "p_kat", "KAT"],
    "p_nvsa":       ["p_nvsa", "NVSA"],
    "p_csdi":       ["p_csdi", "CSDI"],
    "p_gnn":        ["p_gnn_ens", "p_gnn_raw"],
    "gnn_available": ["gnn_available"],
    "csdi_available": ["csdi_available"],
    "dnn":          ["DNN"],
    "dnn_status":   ["dnn_validation_status"],
    "dt_headline":  ["DIGITAL TWIN", "p_digital_twin", "DT"],
    "ohss_mod":     ["ohss_moderate"],
    "ohss_sev":     ["ohss_severe"],
    "no_blast":     ["p_no_blast"],
    "no_good":      ["p_no_good_blast"],
    "bank_mii":     ["banking_p_per_mii"],
    "bank_eup":     ["banking_expected_euploid"],
    "nn_source":    ["nn_source"],
    "cluster":      ["dominant_cluster", "cluster"],
    "c0":           ["cluster_c0_prob"],
    "c1":           ["cluster_c1_prob"],
    "c2":           ["cluster_c2_prob"],
}

FILES = {
    "pred": "dt_predictions.csv",
    "opu": "OPU_table_enriched.xlsx",
    "tabpfn": "tabpfn_predictions.csv",
    "regression": "regression_qc.csv",
    "review": "review_list.csv",
    "ozema": "ozema_cycles.csv",
    "ozema_candidates": "ozema_genetic_counseling_candidates.csv",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pick(row, candidates):
    for c in candidates:
        if c in row and pd.notna(row[c]):
            return row[c]
    return None


def num(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        f = float(v)
    except (ValueError, TypeError):
        return None
    if f != f:
        return None
    return int(f) if f.is_integer() else f


def text(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    return s or None


def flag(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "да", "yes")
    return bool(v)


def norm_id(v):
    return str(v).strip().upper() if v is not None and pd.notna(v) else None


def iso_date(v):
    """Даты приходят двух видов: ISO из файлов пайплайна и "дд.мм.гггг" из выгрузок.

    ISO разбирается ПЕРВЫМ и без dayfirst: pandas 3.0.3 с ``dayfirst=True``
    читает "2026-01-05" как 1 мая (format="mixed" не помогает), то есть тихо
    переставляет день и месяц у каждой даты с днём ≤ 12 — примерно 40% строк,
    и карточка пациентки показывала бы чужую дату. Та же логика продублирована
    в cards_build._date; см. embryo/normalize.parse_dates.
    """
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        t = pd.to_datetime(pd.Series([v]), errors="coerce", format="ISO8601").iloc[0]
        if pd.isna(t):
            t = pd.to_datetime(pd.Series([v]), errors="coerce", dayfirst=True,
                               format="mixed").iloc[0]
    except (ValueError, TypeError):
        return None
    return None if pd.isna(t) else t.date().isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def preg_label(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "Неизвестно"
    try:
        return "Беременность" if int(float(v)) == 1 else "Нет"
    except (ValueError, TypeError):
        return "Неизвестно"


def flag30(real, pred):
    """±30% verdict, mirroring the report's funnel deviation rule."""
    if real in (None, 0) or pred is None:
        return "— н/д"
    return "✅ в норме" if abs(pred - real) / real <= 0.30 else "⚠️ вне ±30%"


def priority_label(v):
    p = num(v)
    if p is None:
        return None
    return {1: "1 Критично", 2: "2 Рекомендовано"}.get(int(p), "3 В норме")


def regression_stage_fields(r: dict | None) -> tuple[dict, int]:
    """Per-stage TabPFN-3 columns and how many stages fell outside P10–P90."""
    if not r:
        return {}, 0
    fields, outside = {}, 0
    for stage, cands in ts.REGRESSION_STAGES.items():
        for cand in cands:
            p50 = r.get(f"{cand}_p50")
            if p50 is None or pd.isna(p50):
                continue
            ok = flag(r.get(f"{cand}_in_interval80"))
            fields[f"{stage} факт"] = num(r.get(f"{cand}_actual"))
            fields[f"{stage} P10"] = num(r.get(f"{cand}_p10"))
            fields[f"{stage} P50"] = num(r.get(f"{cand}_p50"))
            fields[f"{stage} P90"] = num(r.get(f"{cand}_p90"))
            fields[f"{stage} интервал"] = ("✅ в P10–P90" if ok is True else
                                           "⚠️ вне P10–P90" if ok is False else "— н/д")
            outside += 1 if ok is False else 0
            break
    return fields, outside


def read_any(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        if path.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(path)
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - a broken optional input is not fatal
        print(f"  [!] {os.path.basename(path)}: {exc}")
        return pd.DataFrame()


def index_by(df, col):
    if df.empty or col not in df.columns:
        return {}
    return {str(k): v for k, v in df.set_index(col).to_dict("index").items()}


def clean(fields: dict) -> dict:
    return {k: v for k, v in fields.items() if v is not None}


# ---------------------------------------------------------------------------
# One clinic
# ---------------------------------------------------------------------------

def run_dt_version(proc: str, default: str) -> str:
    """Версия DT этого прогона — из его run_manifest.json, а не из статичного конфига клиники.

    С 2026-09-13 новые прогоны всех клиник идут только через API 7.1, но в Runs остаются
    прежние прогоны батча 7.0 — поэтому подпись берётся из файла прогона."""
    try:
        with open(os.path.join(proc, "run_manifest.json"), encoding="utf-8") as fh:
            source = (json.load(fh).get("digital_twin") or {}).get("source") or ""
    except (OSError, ValueError):
        return default
    if "7.1" in source:
        return "v7.1 (REST API)"
    if "7.0" in source:
        return "v7.0"
    return default


def sync_clinic(cfg, tb: tc.Teable, clinic: tc.Clinic, args) -> bool:
    proc = clinic.processed_dir
    f_pred = os.path.join(proc, FILES["pred"])
    if not os.path.exists(f_pred):
        print(f"  [skip] нет {FILES['pred']} в {proc}")
        return False
    if not clinic.configured:
        print("  [skip] у клиники нет base_id — сначала teable_bootstrap.py")
        return False

    tables = tc.resolve_tables(tb, clinic.base_id)
    missing = [k for k in ts.table_keys() if k not in tables]
    if missing:
        print(f"  [skip] в базе нет таблиц {missing} — запустите teable_bootstrap.py")
        return False

    # ---- run identity: content hash of the predictions file ---------------
    run_hash = sha256_file(f_pred)[:12]
    now = dt.datetime.now()
    run_id = f"{clinic.code}-{now:%Y%m%d-%H%M}-{run_hash}"
    cycles_only = getattr(args, "cycles_only", False)
    existing_runs = [r["fields"].get("Run ID") for r in tb.all_records(tables["Runs"])]
    if (any(k and k.endswith(run_hash) for k in existing_runs)
            and not args.force and not cycles_only):
        print(f"  [skip] эти результаты (hash {run_hash}) уже в Teable. "
              f"Перезалить — --force.")
        return False

    # ---- load every source -------------------------------------------------
    df_pred = read_any(f_pred)
    df_opu = read_any(os.path.join(proc, FILES["opu"]))
    tabpfn = index_by(read_any(os.path.join(proc, FILES["tabpfn"])), "patient_id")
    regression = index_by(read_any(os.path.join(proc, FILES["regression"])), "cycle_uid")
    review = index_by(read_any(os.path.join(proc, FILES["review"])), "cycle_uid")
    ozema = index_by(read_any(os.path.join(proc, FILES["ozema"])), "cycle_uid")
    candidates = set(index_by(
        read_any(os.path.join(proc, FILES["ozema_candidates"])), "cycle_uid"))

    # cycle join key -> {"real": {...}, "pred": {...}, "uid": ...}
    patients: dict = {}

    def slot(pid):
        return patients.setdefault(pid, {"real": {}, "pred": {}, "uid": None})

    for _, r in df_opu.iterrows():
        pid = norm_id(pick(r, OPU_COLS["patient_id"]))
        if not pid:
            continue
        d = slot(pid)["real"]
        for key, cands in OPU_COLS.items():
            d[key] = pick(r, cands)

    for _, r in df_pred.iterrows():
        pid = norm_id(pick(r, PRED_COLS["patient_id"]))
        if not pid:
            continue
        entry = slot(pid)
        for key, cands in PRED_COLS.items():
            entry["pred"][key] = pick(r, cands)
        entry["uid"] = text(pick(r, PRED_COLS["cycle_uid"])) or pid
        # dt_predictions also carries the joined real funnel; use it wherever the
        # OPU master did not supply a value (e.g. clinics without an OPU master).
        for key, cands in OPU_COLS.items():
            if entry["real"].get(key) is None:
                entry["real"][key] = pick(r, cands)

    n_pat = len(patients)
    n_out = sum(1 for p in patients.values() if pd.notna(p["real"].get("preg"))
                and p["real"].get("preg") is not None)
    n_pos = sum(1 for p in patients.values() if str(p["real"].get("preg")) in ("1", "1.0"))
    n_review = sum(1 for p in patients.values() if str(p["uid"]) in review)
    n_ozema = sum(1 for p in patients.values()
                  if flag((ozema.get(str(p["uid"])) or {}).get("flagged")))
    print(f"  Parsed: {n_pat} patients, {n_out} with outcome (+{n_pos}); "
          f"review {n_review}, OZEMA flags {n_ozema}. Run: {run_id}")

    if args.dry_run:
        print("  [dry-run] no data written.")
        return False

    # ---- 1) Run ------------------------------------------------------------
    if cycles_only:
        print("  [cycles-only] обновляю только таблицу Cycles, новый прогон не создаётся.")
    else:
        tb.create(tables["Runs"], clean({
            "Run ID": run_id,
            "Дата прогона": now.isoformat(),
            "Клиника": clinic.name,
            "Входной файл": FILES["pred"],
            "Пациентов всего": n_pat,
            "С исходом": n_out,
            "Положительных исходов": n_pos,
            "В ревью-листе": n_review,
            "Флагов OZEMA": n_ozema,
            "Версия DT": run_dt_version(proc, clinic.dt_version),
            "Карточки (вьюер)": clinic.card_url(tc.cards_base_url(cfg)),
            "Заметки": f"sha256[pred]={run_hash}; clinic={clinic.code}; "
                       f"источник={proc}",
        }))

    # ---- 2) Cycles: upsert by join key ------------------------------------
    cyc_index = {r["fields"].get("ID пациента"): r["id"]
                 for r in tb.all_records(tables["Cycles"])}
    # How many runs each cycle already appeared in — counted from Predictions
    # keys so that it stays correct on repeated (--force) syncs.
    prior_runs: dict = {}
    for rec in tb.all_records(tables["Predictions"], projection=["Ключ"]):
        composite = str(rec["fields"].get("Ключ") or "")
        if "::" in composite:
            prior_runs[composite.split("::", 1)[0]] = \
                prior_runs.get(composite.split("::", 1)[0], 0) + 1

    cards_url = tc.cards_base_url(cfg)
    creates, updates = [], []
    for pid, p in patients.items():
        real, pred, uid = p["real"], p["pred"], p["uid"]
        cl = text(pred.get("cluster"))
        vrow = review.get(str(uid)) or {}
        orow = ozema.get(str(uid)) or {}
        tp = tabpfn.get(str(text(pred.get("patient_id")) or "")) or {}
        _, outside = regression_stage_fields(regression.get(str(uid)))
        prai = num(pred.get("prai"))
        fields = clean({
            "ID пациента": pid,
            "Имя": text(real.get("name")),
            "cycle_uid": uid,
            "Клиника": clinic.name,
            "Карточка": clinic.card_url(cards_url, uid),
            "Возраст": num(real.get("age")),
            "AMH": num(real.get("amh")),
            "AFC": num(real.get("afc")),
            "BMI": num(real.get("bmi")),
            "Попытка": num(real.get("attempt")),
            "Дата OPU": iso_date(real.get("date_opu")),
            "Дата ET": iso_date(real.get("date_et")),
            "N фолликулов": num(real.get("follicles")),
            "OCC": num(real.get("occ")),
            "MII": num(real.get("mii")),
            "Инсеминировано": num(real.get("inseminated")),
            "2pN": num(real.get("pn2")),
            "Cleavage": num(real.get("cleavage")),
            "Bl": num(real.get("bl")),
            "Good Bl": num(real.get("good_bl")),
            "Cryo": num(real.get("cryo")),
            "ET": num(real.get("et")),
            "День ET": num(real.get("day_et")),
            "IVF/ICSI": text(real.get("method")),
            "Эмбриолог": text(real.get("embryologist")),
            "Врач": text(real.get("doctor")),
            "Исход": preg_label(real.get("preg")),
            "Кластер": ts.CLUSTER_LABELS.get(str(cl).replace(".0", "").lower()) if cl else None,
            # Latest-run snapshot — what the "Поиск пациенток" tab shows
            "PRAI (посл.)": prai if prai is not None else num(tp.get("p_tabpfn")),
            "DIGITAL TWIN (посл.)": num(pred.get("dt_headline")),
            "Ширина CI, пп": num(pred.get("ci_width")),
            "Приоритет ревью": priority_label(vrow.get("priority")) if vrow else None,
            "Флаг OZEMA": text(orow.get("ozema_flag")) if orow else None,
            "Есть флаг OZEMA": flag(orow.get("flagged")) if orow else None,
            "Стадий вне интервала": outside if regression.get(str(uid)) else None,
            # In --cycles-only mode the run bookkeeping is left untouched: that
            # mode refreshes fields, it does not record a new run.
            "Прогонов": None if cycles_only else prior_runs.get(str(uid), 0) + 1,
            "Последний прогон": None if cycles_only else run_id,
        })
        if pid in cyc_index:
            updates.append((cyc_index[pid], fields))
        else:
            creates.append(fields)
    made = tb.create_records(tables["Cycles"], creates)
    for fields, rec in zip(creates, made):
        cyc_index[fields["ID пациента"]] = rec["id"]
    tb.update_records(tables["Cycles"], updates)
    print(f"  Cycles: +{len(creates)} новых, {len(updates)} обновлено.")

    if cycles_only:
        print("  [cycles-only] готово, остальные таблицы не тронуты.")
        return True

    # ---- 3) Predictions ----------------------------------------------------
    rows = []
    for pid, p in patients.items():
        pred, uid = p["pred"], p["uid"]
        if not pred:
            continue
        tp = tabpfn.get(str(text(pred.get("patient_id")) or ""), {})
        cl = text(pred.get("cluster"))
        rows.append(clean({
            "Ключ": f"{uid}::{run_id}",
            "Пациентка": pid,
            "Прогон": run_id,
            "Timestamp": now.isoformat(),
            "med OCC": num(pred.get("med_okk")),
            "med MII": num(pred.get("med_mii")),
            "med 2PN": num(pred.get("med_pn2")),
            "med Bl": num(pred.get("med_blasts")),
            "med Good": num(pred.get("med_good")),
            "med Euploid": num(pred.get("med_euploid")),
            "PRAI (TabPFN)": num(pred.get("prai")) if pred.get("prai") is not None
                             else num(tp.get("p_tabpfn")),
            "PRAI CI низ": num(pred.get("ci_low")) if pred.get("ci_low") is not None
                           else num(tp.get("ci_low")),
            "PRAI CI верх": num(pred.get("ci_high")) if pred.get("ci_high") is not None
                            else num(tp.get("ci_high")),
            "Ширина CI, пп": num(pred.get("ci_width")),
            "Широкий CI": flag(pred.get("ci_wide")),
            "DIGITAL TWIN": num(pred.get("dt_headline")),
            "P per transfer": num(pred.get("p_per_transfer")),
            "P overall cycle": num(pred.get("p_overall")),
            "P cum if viable": num(pred.get("p_cum")),
            "Bayes mean": num(pred.get("bayes_mean")),
            "Bayes CI низ": num(pred.get("bayes_low")),
            "Bayes CI верх": num(pred.get("bayes_high")),
            "KAT": num(pred.get("p_kat")),
            "NVSA": num(pred.get("p_nvsa")),
            "CSDI": num(pred.get("p_csdi")),
            "GNN": num(pred.get("p_gnn")),
            "DNN": num(pred.get("dnn")),
            "Статус DNN": text(pred.get("dnn_status")),
            "GNN доступен": flag(pred.get("gnn_available")),
            "CSDI доступен": flag(pred.get("csdi_available")),
            "Кластер": ts.CLUSTER_LABELS.get(str(cl).replace(".0", "").lower()) if cl else None,
            "C0 prob": num(pred.get("c0")), "C1 prob": num(pred.get("c1")),
            "C2 prob": num(pred.get("c2")),
            "P viable": num(pred.get("p_viable")),
            "P cancel": num(pred.get("p_cancel")),
            "OHSS умеренный": num(pred.get("ohss_mod")),
            "OHSS тяжёлый": num(pred.get("ohss_sev")),
            "P без бластоцист": num(pred.get("no_blast")),
            "P без хор. бластоцист": num(pred.get("no_good")),
            "Banking P на MII": num(pred.get("bank_mii")),
            "Banking ожид. эуплоидных": num(pred.get("bank_eup")),
            "Источник NN": text(pred.get("nn_source")),
        }))
    tb.create_records(tables["Predictions"], rows)
    print(f"  Predictions: {len(rows)} записей.")

    # ---- 4) Prediction vs Reality -----------------------------------------
    rows = []
    for pid, p in patients.items():
        real, pred, uid = p["real"], p["pred"], p["uid"]
        vrow = review.get(str(uid), {})
        occ_r, occ_p = num(real.get("occ")), num(pred.get("med_okk"))
        bl_r, bl_p = num(real.get("bl")), num(pred.get("med_blasts"))
        gb_r, gb_p = num(real.get("good_bl")), num(pred.get("med_good"))
        # PRAI — только выход TabPFN этого прогона (dt_predictions), не столбец OPU-таблицы.
        kat, prai = num(pred.get("p_kat")), num(pred.get("prai"))
        rows.append(clean({
            "Ключ": f"{uid}::{run_id}", "Пациентка": pid, "Прогон": run_id,
            "OCC факт": occ_r, "OCC прогноз": occ_p, "OCC ±30%": flag30(occ_r, occ_p),
            "Bl факт": bl_r, "Bl прогноз": bl_p, "Bl ±30%": flag30(bl_r, bl_p),
            "Good Bl факт": gb_r, "Good Bl прогноз": gb_p,
            "Good ±30%": flag30(gb_r, gb_p),
            "KAT−PRAI (пп)": (round((kat - prai) * 100, 1)
                              if kat is not None and prai is not None else None),
            "Исход": preg_label(real.get("preg")),
            "Приоритет ревью": priority_label(vrow.get("priority")) if vrow else None,
            "Основание": text(vrow.get("reason")) if vrow else None,
        }))
    tb.create_records(tables["Prediction vs Reality"], rows)
    print(f"  Prediction vs Reality: {len(rows)} записей.")

    # ---- 5) Regression QC (TabPFN-3) --------------------------------------
    rows = []
    for pid, p in patients.items():
        uid = p["uid"]
        r = regression.get(str(uid))
        if not r:
            continue
        stage_fields, outside = regression_stage_fields(r)
        fields = {"Ключ": f"{uid}::{run_id}", "Пациентка": pid, "Прогон": run_id,
                  **stage_fields, "Стадий вне интервала": outside}
        rows.append(clean(fields))
    tb.create_records(tables["Regression QC"], rows)
    print(f"  Regression QC: {len(rows)} записей.")

    # ---- 6) Review-list ----------------------------------------------------
    rows = []
    for pid, p in patients.items():
        real, pred, uid = p["real"], p["pred"], p["uid"]
        v = review.get(str(uid))
        if not v:
            continue
        rows.append(clean({
            "Ключ": f"{uid}::{run_id}", "Пациентка": pid, "Прогон": run_id,
            "Приоритет": priority_label(v.get("priority")),
            "Основание": text(v.get("reason")),
            "PRAI": num(v.get("PRAI")),
            "CI низ": num(v.get("ci_low")), "CI верх": num(v.get("ci_high")),
            "Ширина CI, пп": num(v.get("ci_width_pp")),
            "Широкий CI": flag(v.get("ci_wide")),
            "DIGITAL TWIN": num(v.get("DIGITAL TWIN")),
            "Исход": preg_label(v.get("Preg")),
            "Дата OPU": iso_date(v.get("Date OPU")),
            "Дата ET": iso_date(v.get("Date ET")),
            "Карточка": clinic.card_url(cards_url, uid),
        }))
    tb.create_records(tables["Review"], rows)
    print(f"  Review: {len(rows)} записей.")

    # ---- 7) OZEMA ----------------------------------------------------------
    rows = []
    for pid, p in patients.items():
        uid = p["uid"]
        o = ozema.get(str(uid))
        if not o:
            continue
        rows.append(clean({
            "Ключ": f"{uid}::{run_id}", "Пациентка": pid, "Прогон": run_id,
            "Дата OPU": iso_date(o.get("date_opu")), "Возраст": num(o.get("age")),
            "COC": num(o.get("COC")), "MII": num(o.get("MII")),
            "2PN": num(o.get("2PN")), "Cleav": num(o.get("Cleav")),
            "Bl": num(o.get("Bl")),
            "Флаг": text(o.get("ozema_flag")), "Есть флаг": flag(o.get("flagged")),
            "OMA": text(o.get("OMA_result")), "FF": text(o.get("FF_result")),
            "ZA": text(o.get("ZA_result")), "EEA": text(o.get("EEA_result")),
            "Флагов у пациентки": num(o.get("flagged_cycles_n")),
            "Кандидат на генетическое консультирование": str(uid) in candidates,
            "Примечание": text(o.get("monitoring_note")),
            "Эмбриолог": text(o.get("embryologist")), "Врач": text(o.get("doctor")),
        }))
    tb.create_records(tables["OZEMA"], rows)
    print(f"  OZEMA: {len(rows)} записей "
          f"({len(candidates)} кандидатов на генетическое консультирование).")

    # ---- 8) Calibration ----------------------------------------------------
    write_calibration(tb, tables["Calibration"], patients, run_id)

    # ---- 9) Files ----------------------------------------------------------
    if not args.no_files and clinic.upload_files:
        upload_files(tb, tables["Files"], proc, run_id, run_hash, now)

    print(f"  Done. Run '{run_id}' -> base {clinic.base_id}.")
    return True


def write_calibration(tb, table_id, patients, run_id):
    """Discriminative metrics per model. Kept for the report's Калибровка table
    only — CLAUDE.md forbids leaning on these numbers in the interpretation."""
    try:
        from sklearn.metrics import roc_auc_score, brier_score_loss
    except ImportError:
        print("  [info] scikit-learn не установлен — калибровка пропущена.")
        return
    model_cols = {"MC": "p_per_transfer", "Bayes": "bayes_mean", "KAT": "p_kat",
                  "NVSA": "p_nvsa", "CSDI": "p_csdi", "GNN": "p_gnn",
                  "DT": "dt_headline", "TabPFN": "prai", "DNN": "dnn"}
    rows_in = [(p["pred"], p["real"].get("preg")) for p in patients.values()
               if p["real"].get("preg") is not None and pd.notna(p["real"].get("preg"))]
    if len(rows_in) < 5:
        print(f"  Calibration: пропущена ({len(rows_in)} пациенток с исходом).")
        return
    y = [int(float(o)) for _, o in rows_in]
    out = []
    for mname, mkey in model_cols.items():
        pairs = [(num(pr.get(mkey)), yy) for (pr, _), yy in zip(rows_in, y)]
        pairs = [(a, b) for a, b in pairs if a is not None]
        if len(pairs) < 5:
            continue
        pv = [a for a, _ in pairs]
        yv = [b for _, b in pairs]
        try:
            auc = roc_auc_score(yv, pv) if len(set(yv)) > 1 else None
        except ValueError:
            auc = None
        out.append(clean({
            "Ключ": f"{run_id}::{mname}", "Прогон": run_id, "Модель": mname,
            "Brier": round(brier_score_loss(yv, pv), 4),
            "AUC-ROC": round(auc, 4) if auc is not None else None,
            "Acc@0.5": round(sum(1 for a, b in zip(pv, yv)
                                 if (a >= 0.5) == (b == 1)) / len(yv), 4),
            "Mean pred": round(sum(pv) / len(pv), 4),
            "Real rate": round(sum(yv) / len(yv), 4),
            "N": len(yv),
        }))
    tb.create_records(table_id, out)
    print(f"  Calibration: {len(out)} моделей ({len(rows_in)} с известным исходом).")


def upload_files(tb, table_id, proc, run_id, run_hash, now):
    fil_field = next((f["id"] for f in tb.fields(table_id) if f["type"] == "attachment"),
                     None)
    fil_index = {r["fields"].get("Имя файла"): r["id"] for r in tb.all_records(table_id)}
    try:
        entries = sorted(e for e in os.listdir(proc)
                         if os.path.isfile(os.path.join(proc, e)) and not e.startswith("."))
    except FileNotFoundError:
        entries = []
    n = 0
    for name in entries:
        path = os.path.join(proc, name)
        nrows = None
        if name.lower().endswith((".csv", ".xlsx", ".xls")):
            try:
                nrows = len(read_any(path))
            except Exception:  # noqa: BLE001 - a row count is nice to have, not required
                pass
        display_name = f"{name} ({run_hash})"
        stem = os.path.splitext(name)[0]
        known = {c["name"] for c in ts.FILE_TYPE_CHOICES}
        fields = clean({
            "Имя файла": display_name,
            "Тип": stem if stem in known else "other",
            "Прогон": run_id,
            "Размер, КБ": round(os.path.getsize(path) / 1024, 1),
            "Строк": nrows,
            "Путь (локальный)": path,
            "SHA-256": sha256_file(path)[:16],
            "Загружено": now.isoformat(),
        })
        if display_name in fil_index:
            rec_id = fil_index[display_name]
            tb.update(table_id, rec_id, fields)
        else:
            rec_id = tb.create(table_id, fields)["id"]
            fil_index[display_name] = rec_id
        if fil_field:
            tb.upload_attachment(table_id, rec_id, fil_field, path)
        n += 1
    print(f"  Files: {n} вложений.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=tc.DEFAULT_CONFIG)
    ap.add_argument("--clinic", help="clinic code from the registry")
    ap.add_argument("--all", action="store_true", help="every clinic in the registry")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-files", action="store_true")
    ap.add_argument("--no-cards", action="store_true",
                    help="do not rebuild the local patient-card viewer")
    ap.add_argument("--cycles-only", action="store_true",
                    help="refresh only the Cycles table (fields added later, "
                         "latest-run snapshot) without recording a new run")
    args = ap.parse_args()

    if not args.clinic and not args.all:
        sys.exit("Pass --clinic <code> or --all.")

    cfg = tc.load_config(args.config)
    tb = tc.connect(cfg, dry_run=args.dry_run)

    synced = []
    for clinic in tc.clinics(cfg, args.clinic):
        print(f"\n=== {clinic.name} ({clinic.code}) ===")
        try:
            if sync_clinic(cfg, tb, clinic, args):
                synced.append(clinic.code)
        except Exception as exc:  # noqa: BLE001 - one clinic must not kill the rest
            print(f"  [!] {clinic.name}: {exc}")

    if not args.no_cards and not args.dry_run:
        print("\n=== Карточки пациенток ===")
        try:
            cards_build.build(cfg, only=args.clinic, history=True)
        except Exception as exc:  # noqa: BLE001 - the viewer is a bonus, not a blocker
            print(f"  [!] сборка карточек не удалась: {exc}")

    print(f"\nDone. Синхронизировано клиник: {len(synced)}"
          + (f" ({', '.join(synced)})" if synced else ""))


if __name__ == "__main__":
    main()
