#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cards_build.py — per-cycle patient cards ("вьюер") for each clinic.

Builds ONE self-contained HTML page per clinic from that clinic's
`data/processed` files, plus an index page with a clinic switcher. The page
lets you pick a patient by ФИО / ID / cycle_uid and shows the same structured
information the final `/run-analysis` HTML report gives — funnel fact vs Digital
Twin medians vs TabPFN-3 P10–P90, every pregnancy-probability model, cluster and
cycle risks, the review-list verdict, OZEMA arrest flags — for that one cycle,
positioned against the rest of the clinic's cohort.

If the clinic's Teable base is reachable, the per-cycle prediction history
across previous runs is embedded too, so one cycle can be traced run to run.

The page is ACCUMULATIVE, not a snapshot of the last run: `data/processed/
<clinic>/` is a per-clinic folder, so every run overwrites `dt_predictions.csv`
in it, and building only from that file made whole cohorts vanish from the
viewer whenever a run covered a narrower slice (Fertimed, 31.07.2026: a June
run of 34 cycles replaced the six-month 250, while the Teable base still held
all 282). Cycles missing from the current run are therefore rebuilt from the
base itself — `Cycles` + `Predictions` + `Regression QC` + `Review` + `OZEMA`
together carry enough for a full card — and marked `archived` so a reconstructed
card is never mistaken for a fresh one. Without a reachable base the page falls
back to the current run alone, exactly as before.

Privacy: the pages carry ФИО and are written under `data/processed/_viewer`,
which lives inside the gitignored `data/processed` tree and is served only by
the local nginx container on 127.0.0.1. They are the local layer — never an
agent packet.

Usage:
    python cards_build.py --all
    python cards_build.py --clinic fertimed --no-history
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import sys

try:
    import pandas as pd
except ImportError:  # pragma: no cover - dependency guard
    sys.exit("Missing dependency 'pandas'. Run: pip install -r requirements.txt")

import teable_common as tc
import teable_schema as ts

# Layout, typography and component shapes still follow src/embryo/report.py, but
# the palette here is deliberately its own: the owner asked for a soft olive card
# viewer (2026-07-28). report.py keeps the warm Claude.ai palette pinned by
# CLAUDE.md — do NOT propagate these tokens back into the report.
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif"

STAGE_SOURCES = [
    # display, real column in dt_predictions, DT median column, regression key
    ("Фолликулы", "N folicules OPU", None, None),
    ("ООК", "OCC", "med_okk", None),
    ("MII", "MII", "med_mii", "mii"),
    ("2PN", "2pN", "med_pn2", "pn2"),
    ("Бласт", "Bl", "med_blasts", "blasts_after_fert"),
    ("Хор.бласт", "Good Bl", "med_good", "good"),
    ("Витрифицировано", "Cryo", None, "frozen"),
]

MODEL_ROW = [
    # key in the cycle's prob dict, label, short explanation for the tooltip
    ("prai", "PRAI (TabPFN)", "калиброванная вероятность беременности на перенос"),
    ("dt", "DIGITAL TWIN", "сводная оценка ансамбля"),
    ("mc", "MC per transfer", "Монте-Карло пайплайн, на перенос"),
    ("cum", "MC кумулятивно", "накопленная при жизнеспособном эмбрионе"),
    ("bayes", "Bayes", "иерархическая байесовская оценка"),
    ("kat", "KAT", "KAN + FT-Transformer, сырой выход"),
    ("nvsa", "NVSA", "нейро-выживаемость"),
    ("csdi", "CSDI", "диффузионная импутация"),
    ("gnn", "GNN", "граф похожих циклов"),
    ("dnn", "DNN", "независимая проверочная сеть"),
    ("laya", "Laya", "внешний компаратор DT (laya_kan_ft), калибровка по центру; не член ансамбля"),
]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _num(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return int(f) if f.is_integer() else round(f, 6)


def _str(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return s or None


def _date(v):
    """ISO сначала, "дд.мм.гггг" — запасным ходом (см. teable_sync.iso_date).

    dayfirst=True применённый к ISO переставляет день с месяцем: "2026-01-05"
    прочиталось бы как 1 мая, и в карточке стояла бы чужая дата.
    """
    try:
        t = pd.to_datetime(pd.Series([v]), errors="coerce", format="ISO8601").iloc[0]
        if pd.isna(t):
            t = pd.to_datetime(pd.Series([v]), errors="coerce", dayfirst=True,
                               format="mixed").iloc[0]
    except (ValueError, TypeError):
        return None
    return None if pd.isna(t) else t.date().isoformat()


def _bool(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "да", "yes")
    return bool(v)


def _laya_avail(ok, high, gap, flags) -> dict:
    """Состояние Laya для карточки. laya_ok: None — Laya в прогоне не было,
    False — цикл для неё непригоден (флаги объясняют почему)."""
    if ok is None or (isinstance(ok, float) and pd.isna(ok)):
        # Teable не отдаёт снятый чекбокс: «непригоден» тогда видно по флагам Laya.
        if _str(flags):
            return {"laya_ok": False, "laya_high": None, "laya_gap": None, "laya_flags": _str(flags)}
        return {"laya_ok": None}
    ok = ok if isinstance(ok, bool) else str(ok).strip().lower() in ("true", "1")
    hi = high if isinstance(high, bool) else (str(high).strip().lower() in ("true", "1")
                                              if high is not None and not (isinstance(high, float) and pd.isna(high)) else None)
    return {"laya_ok": ok, "laya_high": hi, "laya_gap": _num(gap), "laya_flags": _str(flags)}


def _pick(row, *names):
    for n in names:
        if n in row.index and pd.notna(row[n]):
            return row[n]
    return None


def _read(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        if path.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(path)
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - a broken optional input must not stop the build
        print(f"    [!] {os.path.basename(path)}: {exc}")
        return pd.DataFrame()


def _index_by(df, col):
    if df.empty or col not in df.columns:
        return {}
    return {str(k): v for k, v in df.set_index(col).to_dict("index").items()}


# ---------------------------------------------------------------------------
# per-clinic payload
# ---------------------------------------------------------------------------

def fetch_record_links(tb: tc.Teable, base_id: str, teable_url: str) -> dict:
    """cycle_uid -> direct URL of that cycle's record in Teable.

    Closes the loop: the Teable row links out to the card, and the card links
    back to the row, so either side can be the starting point.
    """
    try:
        tables = tc.resolve_tables(tb, base_id)
        table_id = tables["Cycles"]
        views = tb.views(table_id)
        # Prefer the patient-search tab so the row opens in the view the owner
        # actually works in; fall back to whatever view exists first.
        view = next((v for v in views if "Поиск" in v["name"]), views[0] if views else None)
        recs = tb.all_records(table_id, projection=["cycle_uid"])
    except Exception as exc:  # noqa: BLE001 - a missing back-link is not fatal
        print(f"    [!] ссылки на записи Teable недоступны: {exc}")
        return {}
    out = {}
    for rec in recs:
        uid = rec.get("fields", {}).get("cycle_uid")
        if not uid:
            continue
        path = f"{teable_url.rstrip('/')}/base/{base_id}/{table_id}"
        if view:
            path += f"/{view['id']}"
        out[str(uid)] = f"{path}?recordId={rec['id']}"
    return out


def build_payload(clinic: tc.Clinic, history: dict | None = None,
                  record_links: dict | None = None, archive: dict | None = None) -> dict:
    # Основной прогон, затем дополнительные (циклы вне когорты отчёта, scripts/
    # run_off_cohort.py). Цикл, уже собранный из более раннего источника, не дублируется.
    cycles: list = []
    for proc in [clinic.processed_dir, *getattr(clinic, "supplemental_dirs", [])]:
        seen = {c["uid"] for c in cycles}
        cycles.extend(c for c in _run_cycles(proc, history, record_links) if c["uid"] not in seen)
    if not cycles and not archive:
        return {}

    # Циклы, которых в этом прогоне нет, но которые есть в базе, восстанавливаются
    # из неё (см. fetch_archive). Строки текущего прогона имеют приоритет: они
    # собраны из файлов, а не из денормализованного снимка, поэтому полнее.
    fresh = {c["uid"] for c in cycles}
    # Строка базы без cycle_uid (заведена OPU-таблицей до первого прогноза) лежит в
    # архиве под join-ключом — сверяем и по нему, иначе цикл задвоится.
    fresh_keys = {str(c.get("key") or "").upper() for c in cycles}
    n_archived = 0
    for uid, card in (archive or {}).items():
        if uid in fresh or str(card.get("key") or "").upper() in fresh_keys:
            continue
        card = dict(card)
        card["hist"] = ((history or {}).get(str(uid))
                        or (history or {}).get(str(card.get("key") or "").upper(), []))
        card["rec"] = (record_links or {}).get(str(uid))
        cycles.append(card)
        n_archived += 1

    # Прогон, сделанный до подключения Laya, не несёт её колонок в dt_predictions.csv;
    # значение тогда берётся из самой поздней строки Predictions этого цикла в базе.
    for c in cycles:
        if c["prob"].get("laya") is None and c["avail"].get("laya_ok") is None:
            last = next((h for h in reversed(c.get("hist") or [])
                         if (h.get("laya_av") or {}).get("laya_ok") is not None), None)
            if last:
                c["prob"]["laya"] = _num(last.get("laya"))
                c["avail"].update(last["laya_av"])

    cycles.sort(key=lambda c: (c["opu"] or "", c["name"]), reverse=True)
    return {
        "clinic": clinic.name, "code": clinic.code,
        "built": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "source": os.path.basename(clinic.processed_dir.rstrip("\\/")) or "processed",
        "n_fresh": len(fresh), "n_archived": n_archived,
        "n_off": sum(1 for c in cycles if c.get("scope")),
        "cycles": cycles,
    }


def _run_cycles(proc: str, history: dict | None, record_links: dict | None) -> list:
    """Карточки циклов одной папки прогона (dt_predictions.csv и соседние файлы)."""
    pred = _read(os.path.join(proc, "dt_predictions.csv"))
    if pred.empty:
        return []
    source = os.path.basename(proc.rstrip("\\/")) or "processed"

    tabpfn = _index_by(_read(os.path.join(proc, "tabpfn_predictions.csv")), "patient_id")
    reg = _index_by(_read(os.path.join(proc, "regression_qc.csv")), "cycle_uid")
    review = _index_by(_read(os.path.join(proc, "review_list.csv")), "cycle_uid")
    ozema = _index_by(_read(os.path.join(proc, "ozema_cycles.csv")), "cycle_uid")
    opu = _read(os.path.join(proc, "OPU_table_enriched.xlsx"))
    opu_idx = _index_by(opu, "_key") if "_key" in opu.columns else {}

    cycles = []
    for _, r in pred.iterrows():
        uid = _str(r.get("cycle_uid")) or _str(r.get("_key")) or _str(r.get("patient_id"))
        key = _str(r.get("_key")) or _str(r.get("patient_id")) or uid
        extra = opu_idx.get(str(key), {})

        def _x(name):
            return extra.get(name) if isinstance(extra, dict) else None

        real = {
            "Фолликулы": _num(_pick(r, "N folicules OPU", "follicles_tvp")),
            "ООК": _num(_pick(r, "OCC", "known_okk")),
            "MII": _num(_pick(r, "MII", "known_mii")),
            "2PN": _num(_pick(r, "2pN", "known_pn2", "real_pn2")),
            "Бласт": _num(_pick(r, "Bl", "known_blasts", "real_bl")),
            "Хор.бласт": _num(_pick(r, "Good Bl", "known_good", "real_goodbl")),
            # 0 is a real observation here, so fall back on missingness only
            "Витрифицировано": (_num(_pick(r, "real_cryo")) if _num(_pick(r, "real_cryo")) is not None
                                else _num(_x("Cryo"))),
        }
        # No DT median for the cryo stage: med_warmed answers a different question
        # (embryos warmed, not vitrified) and must not be shown against Cryo.
        med = {"ООК": _num(r.get("med_okk")), "MII": _num(r.get("med_mii")),
               "2PN": _num(r.get("med_pn2")), "Бласт": _num(r.get("med_blasts")),
               "Хор.бласт": _num(r.get("med_good")), "Витрифицировано": None}

        tp = tabpfn.get(str(_str(r.get("patient_id")) or ""), {})
        prob = {
            "prai": _num(r.get("PRAI")),
            "ci_low": _num(_pick(r, "ci_low", "rate_ci_low")),
            "ci_high": _num(_pick(r, "ci_high", "rate_ci_high")),
            "ci_width": _num(r.get("ci_width_pp")),
            "ci_wide": _bool(r.get("ci_wide")),
            "dt": _num(r.get("DIGITAL TWIN")),
            "mc": _num(r.get("p_per_transfer")),
            "cum": _num(r.get("p_cum_if_viable")),
            "overall": _num(r.get("p_overall_cycle")),
            "bayes": _num(r.get("bayes_mean")),
            "bayes_lo": _num(r.get("bayes_ci_low")),
            "bayes_hi": _num(r.get("bayes_ci_high")),
            "kat": _num(r.get("p_kat_raw")),
            "nvsa": _num(r.get("p_nvsa")),
            "csdi": _num(r.get("p_csdi")),
            "gnn": _num(_pick(r, "p_gnn_ens", "p_gnn_raw")),
            "dnn": _num(r.get("DNN")),
            "laya": _num(r.get("Laya")),
            "tabpfn": _num(tp.get("p_tabpfn")) if tp else None,
        }
        risk = {
            "cluster": _num(r.get("dominant_cluster")),
            "c0": _num(r.get("cluster_c0_prob")), "c1": _num(r.get("cluster_c1_prob")),
            "c2": _num(r.get("cluster_c2_prob")),
            "viable": _num(r.get("p_viable")), "cancel": _num(r.get("p_cancel_risk")),
            "ohss_mod": _num(r.get("ohss_moderate")), "ohss_sev": _num(r.get("ohss_severe")),
            "no_blast": _num(r.get("p_no_blast")), "no_good": _num(r.get("p_no_good_blast")),
            "bank_mii": _num(r.get("banking_p_per_mii")),
            "bank_eup": _num(r.get("banking_expected_euploid")),
        }

        rrow = reg.get(str(uid), {})
        reg_out = {}
        for display, _real_col, _med_col, stage_keys in STAGE_SOURCES:
            if not stage_keys:
                continue
            for cand in (ts.REGRESSION_STAGES.get(display) or (stage_keys,)):
                if rrow.get(f"{cand}_p50") is not None and pd.notna(rrow.get(f"{cand}_p50")):
                    reg_out[display] = {
                        "p10": _num(rrow.get(f"{cand}_p10")),
                        "p50": _num(rrow.get(f"{cand}_p50")),
                        "p90": _num(rrow.get(f"{cand}_p90")),
                        "actual": _num(rrow.get(f"{cand}_actual")),
                        "ok": _bool(rrow.get(f"{cand}_in_interval80")),
                    }
                    break

        vrow = review.get(str(uid), {})
        orow = ozema.get(str(uid), {})
        preg = _num(r.get("Preg"))

        cycles.append({
            "uid": uid, "key": key,
            "pid": _str(_pick(r, "patient_id", "ID")),
            "name": _str(_pick(r, "patient_name", "Patient Full Name")) or "—",
            "opu": _date(r.get("Date OPU")), "et": _date(r.get("Date ET")),
            "age": _num(_pick(r, "age", "Age")), "amh": _num(r.get("amh")),
            "bmi": _num(r.get("bmi")), "afc": _num(r.get("afc")),
            "attempt": _num(_pick(r, "attempt_number", "Attempt")),
            "emb": _str(r.get("Embryologist")), "doc": _str(r.get("Doctor")),
            "method": _str(r.get("IVF/ICSI")), "sperm": _str(r.get("sperm_source")),
            "preg": None if preg is None else int(preg),
            "real": real, "med": med, "prob": prob, "risk": risk, "reg": reg_out,
            # Why a model is empty: False = it did not load in this run at all
            # (nothing to show for anybody), True = it ran, so an empty value
            # means it produced nothing for this particular cycle.
            "avail": {"gnn": _bool(r.get("gnn_available")),
                      "csdi": _bool(r.get("csdi_available")),
                      "dnn": _str(r.get("dnn_validation_status")),
                      **_laya_avail(r.get("laya_assessable"), r.get("laya_disagreement_high"),
                                    r.get("laya_minus_dt_pp"), r.get("laya_flags"))},
            "review": ({"priority": _num(vrow.get("priority")),
                        "reason": _str(vrow.get("reason"))} if vrow else None),
            "ozema": ({"flag": _str(orow.get("ozema_flag")),
                       "oma": _str(orow.get("OMA_result")), "ff": _str(orow.get("FF_result")),
                       "za": _str(orow.get("ZA_result")), "eea": _str(orow.get("EEA_result")),
                       "flagged_n": _num(orow.get("flagged_cycles_n")),
                       "note": _str(orow.get("monitoring_note"))} if orow else None),
            # Predictions rows are keyed by cycle_uid since this sync version;
            # older rows in an already-populated base use the upper-cased join
            # key, so both are looked up before giving up on the history.
            "hist": ((history or {}).get(str(uid))
                     or (history or {}).get(str(key).upper(), [])),
            "rec": (record_links or {}).get(str(uid)),
            # Только у дополнительного прогона вне когорты отчёта: почему прогноз
            # полный, урезан до воронки или отсутствует (см. embryo/off_cohort.py).
            "scope": _str(r.get("cohort_scope_label")),
            "tier": _str(r.get("model_tier")),
            "src": source,
        })
    return cycles


def fetch_history(tb: tc.Teable, base_id: str) -> dict:
    """cycle key -> [{run, ts, prai, dt, kat, bl}] across every synced run.

    Predictions rows are keyed "<cycle key>::<run id>"; the cycle key is the
    same join key the cards are built on, so history lines up per cycle.
    """
    try:
        tables = tc.resolve_tables(tb, base_id)
        recs = tb.all_records(tables["Predictions"])
    except Exception as exc:  # noqa: BLE001 - history is a bonus, never a blocker
        print(f"    [!] история прогонов недоступна: {exc}")
        return {}
    out: dict = {}
    for rec in recs:
        fl = rec.get("fields", {})
        composite = str(fl.get("Ключ") or "")
        if "::" not in composite:
            continue
        ckey, run = composite.split("::", 1)
        out.setdefault(ckey, []).append({
            "run": run, "ts": fl.get("Timestamp"),
            "prai": fl.get("PRAI (TabPFN)"), "dt": fl.get("DIGITAL TWIN"),
            "kat": fl.get("KAT"), "bl": fl.get("med Bl"),
            "laya": fl.get("Laya"),
            "laya_av": _laya_avail(fl.get("Laya пригодна"), fl.get("Расхождение высокое"),
                                   fl.get("Laya−DT, пп"), fl.get("Флаги Laya")),
        })
    for rows in out.values():
        rows.sort(key=lambda x: str(x.get("ts") or ""))
    return out


# ---------------------------------------------------------------------------
# archive: cycles that live in Teable but not in this run's processed files
# ---------------------------------------------------------------------------
#
# ЗАЧЕМ ЭТОТ СЛОЙ. `data/processed/<клиника>/` — папка на клинику, а не на
# прогон, поэтому каждый запуск перезаписывает в ней `dt_predictions.csv`.
# Пока карточки строились только из этого файла, вьюер показывал ровно
# последний прогон: 31.07.2026 прогон на июньской выгрузке (34 цикла) сменил
# шестимесячную (250), и полугодовые карточки исчезли со страницы, хотя в базе
# Teable оставались все 282 цикла. База накопительная — вьюер рядом с ней был
# не накопительным, и это расхождение сбивало при каждом прогоне на подвыборке.
#
# Теперь недостающие циклы восстанавливаются из самой базы. Данных там хватает
# на ПОЛНУЮ карточку, а не на урезанную: `Predictions` держит все модели,
# медианы DT, кластер и риски, `Regression QC` — P10/P50/P90 по стадиям,
# `Review` и `OZEMA` — свои вердикты, `Cycles` — личность, демографию и
# фактическую воронку. Берётся САМЫЙ ПОЗДНИЙ прогон каждого цикла.
#
# Восстановленный цикл помечается `archived` — читателю должно быть видно, что
# карточка собрана из базы по прошлому прогону, а не из файлов текущего.

_OUTCOME_TO_PREG = {"Беременность": 1, "Нет": 0}
_CLUSTER_TO_NUM = {"C0 Standard": 0.0, "C1 Poor": 1.0, "C2 High": 2.0}
_PRIORITY_TO_NUM = {"1 Критично": 1.0, "2 Рекомендовано": 2.0, "3 В норме": 3.0}
# Стадии Regression QC названы в базе по-русски теми же ярлыками, что и в карточке.
_ARCHIVE_REG_STAGES = ["MII", "2PN", "Бласт", "Хор.бласт", "Витрифицировано"]


def _run_order(fields: dict) -> str:
    """Ключ сортировки «какой прогон новее».

    У `Predictions` есть `Timestamp`, у `Review`/`Regression QC`/`OZEMA` — нет,
    поэтому запасной вариант — сам идентификатор прогона: он начинается с даты
    (`fertimed-20260731-1706-...`). Регистр разный между версиями синхронизатора
    («Fertimed-...» против «fertimed-...»), поэтому сравнение — по нижнему.
    """
    return (str(fields.get("Timestamp") or "") + "|" + str(fields.get("Прогон") or "")).lower()


def _newest_by_cycle(records: list) -> dict:
    """Префикс «Ключа» -> поля САМОЙ ПОЗДНЕЙ строки этого цикла.

    Строки помечены «<ключ цикла>::<id прогона>». Ключом цикла в новых прогонах
    служит `cycle_uid`, в старых — составной join-ключ; поэтому здесь просто
    сохраняются оба варианта префикса, а разрешение имени — на стороне вызова.
    """
    out: dict = {}
    for rec in records:
        fields = rec.get("fields", {})
        composite = str(fields.get("Ключ") or "")
        if not composite:
            continue
        key = composite.split("::", 1)[0]
        current = out.get(key)
        if current is None or _run_order(fields) >= _run_order(current):
            out[key] = fields
    return out


def _lookup(source: dict, *keys) -> dict:
    """Первое совпадение по любому из вариантов ключа цикла (uid / join-ключ)."""
    for key in keys:
        if not key:
            continue
        for variant in (str(key), str(key).upper()):
            if variant in source:
                return source[variant]
    return {}


def _interval_ok(value) -> bool | None:
    text = _str(value)
    if not text:
        return None
    if text.startswith("✅"):
        return True
    if text.startswith("⚠"):
        return False
    return None


def fetch_archive(tb: tc.Teable, base_id: str) -> dict:
    """cycle_uid -> карточка цикла, собранная из базы (для циклов вне прогона)."""
    try:
        tables = tc.resolve_tables(tb, base_id)
        cycle_rows = tb.all_records(tables["Cycles"])
        preds = _newest_by_cycle(tb.all_records(tables["Predictions"]))
        regs = _newest_by_cycle(tb.all_records(tables["Regression QC"]))
        revs = _newest_by_cycle(tb.all_records(tables["Review"]))
        ozem = _newest_by_cycle(tb.all_records(tables["OZEMA"]))
    except Exception as exc:  # noqa: BLE001 - архив опционален, прогон не блокирует
        print(f"    [!] архив базы недоступен: {exc}")
        return {}

    out: dict = {}
    for rec in cycle_rows:
        cf = rec.get("fields", {})
        key = _str(cf.get("ID пациента"))
        # Единичные старые строки базы лежат без cycle_uid (у Fertimed — одна,
        # из прогона 29.07.2026). Ключом карточки тогда служит составной
        # join-ключ: без него пациентка просто исчезала бы из вьюера. Коллизия
        # исключена — cycle_uid всегда начинается с "CYC-", join-ключ никогда.
        uid = _str(cf.get("cycle_uid")) or key
        if not uid:
            continue
        pf = _lookup(preds, uid, key)
        rf = _lookup(regs, uid, key)
        vf = _lookup(revs, uid, key)
        of = _lookup(ozem, uid, key)

        reg_out = {}
        for stage in _ARCHIVE_REG_STAGES:
            if rf.get(f"{stage} P50") is None:
                continue
            reg_out[stage] = {
                "p10": _num(rf.get(f"{stage} P10")), "p50": _num(rf.get(f"{stage} P50")),
                "p90": _num(rf.get(f"{stage} P90")), "actual": _num(rf.get(f"{stage} факт")),
                "ok": _interval_ok(rf.get(f"{stage} интервал")),
            }

        out[uid] = {
            "uid": uid, "key": key,
            "pid": key, "name": _str(cf.get("Имя")) or "—",
            "opu": _date(cf.get("Дата OPU")), "et": _date(cf.get("Дата ET")),
            "age": _num(cf.get("Возраст")), "amh": _num(cf.get("AMH")),
            "bmi": _num(cf.get("BMI")), "afc": _num(cf.get("AFC")),
            "attempt": _num(cf.get("Попытка")),
            "emb": _str(cf.get("Эмбриолог")), "doc": _str(cf.get("Врач")),
            "method": _str(cf.get("IVF/ICSI")), "sperm": None,
            "preg": _OUTCOME_TO_PREG.get(_str(cf.get("Исход"))),
            "real": {
                "Фолликулы": _num(cf.get("N фолликулов")), "ООК": _num(cf.get("OCC")),
                "MII": _num(cf.get("MII")), "2PN": _num(cf.get("2pN")),
                "Бласт": _num(cf.get("Bl")), "Хор.бласт": _num(cf.get("Good Bl")),
                "Витрифицировано": _num(cf.get("Cryo")),
            },
            "med": {"ООК": _num(pf.get("med OCC")), "MII": _num(pf.get("med MII")),
                    "2PN": _num(pf.get("med 2PN")), "Бласт": _num(pf.get("med Bl")),
                    "Хор.бласт": _num(pf.get("med Good")), "Витрифицировано": None},
            "prob": {
                "prai": _num(pf.get("PRAI (TabPFN)")),
                "ci_low": _num(pf.get("PRAI CI низ")), "ci_high": _num(pf.get("PRAI CI верх")),
                "ci_width": _num(pf.get("Ширина CI, пп")), "ci_wide": _bool(pf.get("Широкий CI")),
                "dt": _num(pf.get("DIGITAL TWIN")), "mc": _num(pf.get("P per transfer")),
                "cum": _num(pf.get("P cum if viable")), "overall": _num(pf.get("P overall cycle")),
                "bayes": _num(pf.get("Bayes mean")), "bayes_lo": _num(pf.get("Bayes CI низ")),
                "bayes_hi": _num(pf.get("Bayes CI верх")),
                "kat": _num(pf.get("KAT")), "nvsa": _num(pf.get("NVSA")),
                "csdi": _num(pf.get("CSDI")), "gnn": _num(pf.get("GNN")),
                "dnn": _num(pf.get("DNN")),
                "laya": _num(pf.get("Laya")),
                # p_tabpfn отдельной колонкой в базе не лежит: в неё уже сведён PRAI.
                "tabpfn": None,
            },
            "risk": {
                "cluster": _CLUSTER_TO_NUM.get(_str(cf.get("Кластер"))),
                "c0": _num(pf.get("C0 prob")), "c1": _num(pf.get("C1 prob")),
                "c2": _num(pf.get("C2 prob")),
                "viable": _num(pf.get("P viable")), "cancel": _num(pf.get("P cancel")),
                "ohss_mod": _num(pf.get("OHSS умеренный")), "ohss_sev": _num(pf.get("OHSS тяжёлый")),
                "no_blast": _num(pf.get("P без бластоцист")),
                "no_good": _num(pf.get("P без хор. бластоцист")),
                "bank_mii": _num(pf.get("Banking P на MII")),
                "bank_eup": _num(pf.get("Banking ожид. эуплоидных")),
            },
            "reg": reg_out,
            "avail": {"gnn": _bool(pf.get("GNN доступен")), "csdi": _bool(pf.get("CSDI доступен")),
                      "dnn": _str(pf.get("Статус DNN")),
                      **_laya_avail(pf.get("Laya пригодна"), pf.get("Расхождение высокое"),
                                    pf.get("Laya−DT, пп"), pf.get("Флаги Laya"))},
            "review": ({"priority": _PRIORITY_TO_NUM.get(_str(vf.get("Приоритет"))),
                        "reason": _str(vf.get("Основание"))} if vf else None),
            "ozema": ({"flag": _str(of.get("Флаг")), "oma": _str(of.get("OMA")),
                       "ff": _str(of.get("FF")), "za": _str(of.get("ZA")),
                       "eea": _str(of.get("EEA")),
                       "flagged_n": _num(of.get("Флагов у пациентки")),
                       "note": _str(of.get("Примечание"))} if of else None),
            "hist": [],
            "rec": None,
            # Метка честности: карточка собрана из базы по последнему прогону,
            # в котором цикл участвовал, а не из файлов текущего прогона.
            "archived": _str(cf.get("Последний прогон")) or "прошлый прогон",
        }
    return out


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _css() -> str:
    return """
/* Нежно-оливковая гамма. Акцент --accent намеренно приглушённый: он держит
   навигацию и шкалы, а насыщенность остаётся за статусами (красный/жёлтый/
   зелёный), иначе оливковый фон съедает сигнал «в норме / вне нормы». */
:root{
  --bg:#F6F7EE; --card:#FCFDF7; --line:#E2E5D4; --text:#2B2E22; --muted:#7C806B;
  --accent:#87965A; --accent-dark:#5F6B3A;
  --red:#A63A2E; --red-bg:#F3DCD6; --amber:#7E6416; --amber-bg:#EFE4BC;
  --green:#3E7A46; --green-bg:#DBE9D0; --info:#5F6B3A; --info-bg:#E7ECD6;
  --hero-a:#EDF0DF; --hero-b:#D9E1C2;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#16180F; --card:#20231A; --line:#353929; --text:#F1F3E7; --muted:#9AA089;
    --accent:#A9BC6E; --accent-dark:#87965A;
    --red:#EDA79D; --red-bg:#38130F; --amber:#E6C77F; --amber-bg:#2C2306;
    --green:#9CD290; --green-bg:#182811; --info:#C3D392; --info-bg:#252C16;
    --hero-a:#232819; --hero-b:#2E3620;
  }
}
*{box-sizing:border-box}
body{font-family:__FONT__;color:var(--text);background:var(--bg);margin:0;
     line-height:1.45;font-size:14px}
a{color:var(--accent-dark)}
header.hero{background:linear-gradient(135deg,var(--hero-a) 0%,var(--hero-b) 100%);
  padding:20px 22px 16px;border-bottom:3px solid var(--accent)}
header.hero .eyebrow{font-size:11px;font-weight:700;letter-spacing:.08em;
  color:var(--accent-dark);text-transform:uppercase;margin-bottom:5px}
header.hero h1{margin:0;font-size:22px;font-weight:800}
header.hero .meta{display:flex;gap:8px;flex-wrap:wrap;margin-top:9px;align-items:center}
.pill{background:color-mix(in srgb, var(--accent) 18%, transparent);
      border-radius:999px;padding:4px 12px;font-size:12px}
.pill a{text-decoration:none}
.app{display:grid;grid-template-columns:330px 1fr;gap:0;min-height:calc(100vh - 96px)}
.side{border-right:1px solid var(--line);background:var(--card);
      max-height:calc(100vh - 96px);overflow:auto;position:sticky;top:0}
.side-head{padding:12px;border-bottom:1px solid var(--line);position:sticky;top:0;
           background:var(--card);z-index:2}
.search{width:100%;padding:9px 12px;border:1px solid var(--line);border-radius:8px;
        background:var(--bg);color:var(--text);font-size:13px}
.side-count{font-size:11px;color:var(--muted);margin-top:7px}
.pat{padding:9px 12px;border-bottom:1px solid var(--line);cursor:pointer}
.pat:hover{background:color-mix(in srgb, var(--accent) 7%, var(--card))}
.pat.on{background:color-mix(in srgb, var(--accent) 14%, var(--card));
        box-shadow:inset 3px 0 0 var(--accent)}
.pat .pn{font-weight:700;font-size:13px}
.pat .ps{font-size:11px;color:var(--muted);margin-top:1px;display:flex;gap:8px;
         flex-wrap:wrap;align-items:center}
.pane{padding:20px 24px 60px;max-width:1100px}
.eyebrow{font-size:11px;font-weight:700;letter-spacing:.08em;color:var(--muted);
         text-transform:uppercase;margin-bottom:5px}
h2.card-h1{font-size:24px;font-weight:800;margin:0 0 6px}
h3{font-size:14px;margin:0 0 10px}
.sec{background:var(--card);border:1px solid var(--line);border-radius:10px;
     padding:15px 17px;margin-bottom:14px}
.muted{color:var(--muted)}
.stats{display:flex;flex-wrap:wrap;background:var(--card);border:1px solid var(--line);
       border-radius:10px;margin:12px 0 14px;overflow:hidden}
.jstat{flex:1 1 120px;padding:11px 15px;border-right:1px solid var(--line)}
.jstat:last-child{border-right:none}
.jsl{font-size:10.5px;font-weight:700;letter-spacing:.06em;color:var(--muted);
     text-transform:uppercase}
.jsv{font-size:19px;font-weight:800;margin-top:3px}
.badge{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11px;
       font-weight:600;white-space:nowrap}
.badge-ok{background:var(--green-bg);color:var(--green)}
.badge-warn{background:var(--amber-bg);color:var(--amber)}
.badge-crit{background:var(--red-bg);color:var(--red)}
.badge-info{background:var(--info-bg);color:var(--info)}
.prob-pill{display:inline-block;padding:4px 13px;border-radius:999px;font-weight:800;
           font-size:17px}
.prob-low{background:var(--red-bg);color:var(--red)}
.prob-mid{background:var(--amber-bg);color:var(--amber)}
.prob-high{background:var(--green-bg);color:var(--green)}
.ci-track{display:block;position:relative;width:150px;height:6px;background:var(--line);
          border-radius:3px;margin-top:7px}
.ci-range{position:absolute;top:0;height:100%;background:var(--accent);opacity:.55;
          border-radius:3px}
.ci-tick{position:absolute;top:-3px;width:2px;height:12px;background:var(--text);
         border-radius:1px}
table.tbl{border-collapse:collapse;width:100%;font-size:12.5px}
table.tbl th{text-align:left;padding:7px 8px;border-bottom:2px solid var(--line);
             white-space:nowrap;text-transform:uppercase;font-size:10.5px;
             letter-spacing:.05em;color:var(--muted)}
table.tbl td{padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:middle}
.table-wrap{overflow-x:auto}
.models{display:grid;grid-template-columns:repeat(auto-fill,minmax(148px,1fr));gap:9px}
.model{border:1px solid var(--line);border-radius:9px;padding:9px 11px;background:var(--bg)}
.model .ml{font-size:10.5px;font-weight:700;text-transform:uppercase;color:var(--muted);
           letter-spacing:.04em}
.model .mv{font-size:17px;font-weight:800;margin:3px 0 5px}
.mbar{height:5px;background:var(--line);border-radius:3px;overflow:hidden}
.mbar > span{display:block;height:100%;border-radius:3px;background:var(--accent)}
.model-off{border-style:dashed;background:transparent}
.mv-none{color:var(--muted)}
.mnote{font-size:10.5px;color:var(--muted);line-height:1.25;margin-top:1px}
.cohort{margin-top:4px}
.crow{display:grid;grid-template-columns:150px 1fr 118px;gap:10px;align-items:center;
      padding:5px 0}
.ctrack{position:relative;height:8px;background:var(--line);border-radius:4px}
.cfill{position:absolute;top:0;left:0;height:100%;background:var(--accent);opacity:.45;
       border-radius:4px}
.cmed{position:absolute;top:-3px;width:2px;height:14px;background:var(--muted)}
.cme{position:absolute;top:-4px;width:3px;height:16px;background:var(--text);
     border-radius:2px}
.clab{font-size:12px}.cval{font-size:11.5px;color:var(--muted);text-align:right}
.spark{display:flex;align-items:flex-end;gap:3px;height:22px;margin-top:5px}
.spark-bar{display:block;width:9px;background:var(--line);border-radius:1px;min-height:2px}
.spark-ok{background:var(--green)} .spark-bad{background:var(--red)}
.note{border-left:4px solid var(--accent);background:var(--info-bg);color:var(--text);
      padding:10px 13px;border-radius:0 7px 7px 0;font-size:12.5px;margin-top:9px}
.empty{padding:60px 20px;text-align:center;color:var(--muted)}
.kv{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:7px 16px;
    font-size:12.5px}
.kv b{font-weight:700}
.foot{font-size:11px;color:var(--muted);margin-top:22px}
@media(max-width:900px){.app{grid-template-columns:1fr}.side{position:static;
  max-height:320px}}
""".replace("__FONT__", FONT)


_JS = r"""
const $ = s => document.querySelector(s);
let SEL = null;

const pct = v => v === null || v === undefined ? '—' : (v * 100).toFixed(1) + '%';
const n1  = v => v === null || v === undefined ? '—' : (Math.round(v * 10) / 10).toString();
const n0  = v => v === null || v === undefined ? '—' : Math.round(v).toString();
const esc = s => String(s === null || s === undefined ? '' : s)
  .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
const tier = v => v < 0.40 ? 'low' : (v < 0.60 ? 'mid' : 'high');

/* Percentile of `v` inside the cohort values `arr` (share of cycles at or below). */
function percentile(arr, v){
  const xs = arr.filter(x => x !== null && x !== undefined).sort((a,b) => a-b);
  if (!xs.length || v === null || v === undefined) return null;
  let i = 0; while (i < xs.length && xs[i] <= v) i++;
  return i / xs.length;
}
function median(arr){
  const xs = arr.filter(x => x !== null && x !== undefined).sort((a,b) => a-b);
  if (!xs.length) return null;
  const m = Math.floor(xs.length / 2);
  return xs.length % 2 ? xs[m] : (xs[m-1] + xs[m]) / 2;
}

function outcomeBadge(p){
  if (p === 1) return "<span class='badge badge-ok'>Беременность</span>";
  if (p === 0) return "<span class='badge badge-crit'>Нет</span>";
  return "<span class='badge badge-info'>Исход неизвестен</span>";
}

function renderList(q){
  const query = (q || '').trim().toLowerCase();
  const rows = DATA.cycles.filter(c => !query ||
    (c.name + ' ' + (c.pid||'') + ' ' + (c.uid||'')).toLowerCase().includes(query));
  /* Состав страницы называется явно: сколько циклов из текущего прогона и
     сколько восстановлено из базы. Иначе рост числа циклов после прогона на
     подвыборке читается как «выгрузка выросла». */
  const arch = DATA.n_archived || DATA.n_off
    ? ' (' + DATA.n_fresh + ' из прогонов' + (DATA.n_off ? ', из них ' + DATA.n_off + ' вне отчёта' : '')
      + (DATA.n_archived ? ' + ' + DATA.n_archived + ' из базы' : '') + ')' : '';
  $('#count').textContent = rows.length + ' из ' + DATA.cycles.length + ' циклов' + arch;
  $('#list').innerHTML = rows.map(c => {
    const p = c.prob.prai;
    const pillCls = p === null || p === undefined ? '' : ' badge-' +
      (tier(p) === 'low' ? 'crit' : tier(p) === 'mid' ? 'warn' : 'ok');
    const rev = c.review && c.review.priority === 1
      ? "<span class='badge badge-crit'>P1</span>" : '';
    /* Цикл не из текущего прогона, а восстановленный из базы — это должно быть
       видно в списке, иначе архив выглядит как часть свежей выгрузки. */
    const arch = c.archived ? "<span class='badge badge-info'>архив</span>"
      : (c.scope ? "<span class='badge badge-warn'>вне отчёта</span>" : '');
    return `<div class="pat${c.uid===SEL?' on':''}" data-uid="${esc(c.uid)}">
      <div class="pn">${esc(c.name)}</div>
      <div class="ps"><span>${esc(c.opu || '—')}</span><span>${esc(c.pid || '')}</span>
        <span class="badge${pillCls}">${pct(p)}</span>${rev}${arch}</div></div>`;
  }).join('') || "<div class='empty'>Ничего не найдено</div>";
  document.querySelectorAll('.pat').forEach(el =>
    el.onclick = () => select(el.dataset.uid, true));
}

/* Why a model shows no number. An empty cell must never read as "the model
   said nothing here" when in fact the model never ran in this batch. */
function missingReason(c, k){
  const av = c.avail || {};
  if (k === 'gnn' && av.gnn === false) return 'не поднялась в прогоне';
  if (k === 'csdi' && av.csdi === false) return 'не поднялась в прогоне';
  if (k === 'dnn') return av.dnn ? 'нет оценки для цикла' : 'не запускалась';
  if (k === 'laya') return av.laya_ok === false ? 'цикл непригоден для Laya' : 'не запускалась';
  return 'нет значения';
}

function modelsBlock(c){
  const tiles = MODELS.map(([k, label, hint]) => {
    const v = c.prob[k];
    const absent = v === null || v === undefined;
    const w = absent ? 0 : Math.max(2, Math.min(100, v*100));
    const body = absent
      ? `<div class="mv mv-none">—</div>
         <div class="mnote">${esc(missingReason(c, k))}</div>`
      : `<div class="mv">${pct(v)}</div>
         <div class="mbar"><span style="width:${w}%"></span></div>`;
    return `<div class="model${absent ? ' model-off' : ''}" title="${esc(hint)}">
      <div class="ml">${esc(label)}</div>${body}</div>`;
  }).join('');
  const down = MODELS.filter(([k]) => {
    const av = c.avail || {};
    return (k === 'gnn' && av.gnn === false) || (k === 'csdi' && av.csdi === false);
  }).map(([, label]) => label);
  const warn = down.length
    ? `<div class="note">В этом прогоне не поднялась модель: <b>${down.map(esc).join(', ')}</b>
       — пустое значение относится ко всей выборке, а не к этой пациентке.
       Причина ищется на стороне проекта Digital Twin (флаг <code>gnn_available</code> /
       <code>csdi_available</code> в <code>dt_predictions.csv</code>).</div>`
    : '';
  const av = c.avail || {};
  let laya = '';
  if (av.laya_ok === true && av.laya_high === true)
    laya = `<div class="note">Laya и DIGITAL TWIN заметно расходятся
       (Laya − DT = ${av.laya_gap === null || av.laya_gap === undefined ? '—' : av.laya_gap.toFixed(1)} пп
       после калибровки обеих по центру). На когорте RBMO и в базе Teable в этой зоне ошибка DT
       (Brier) выше — повод перепроверить входы цикла, не вердикт.</div>`;
  else if (av.laya_ok === false && av.laya_flags)
    laya = `<div class="note">Laya не оценивает этот цикл: ${esc(av.laya_flags)}</div>`;
  return `<div class="models">${tiles}</div>${warn}${laya}`;
}

function flag30(real, pred){
  if (real === null || real === undefined || !real || pred === null || pred === undefined)
    return "<span class='muted'>— н/д</span>";
  return Math.abs(pred - real) / real <= 0.30
    ? "<span class='badge badge-ok'>✅ ±30%</span>"
    : "<span class='badge badge-crit'>⚠️ вне ±30%</span>";
}

function funnelBlock(c){
  const rows = STAGES.map(s => {
    const real = c.real[s], med = c.med[s], r = c.reg[s];
    const interval = !r ? "<span class='muted'>—</span>"
      : (r.ok === true ? "<span class='badge badge-ok'>✅ в P10–P90</span>"
      : (r.ok === false ? "<span class='badge badge-crit'>⚠️ вне P10–P90</span>"
      : "<span class='muted'>—</span>"));
    const band = r ? `${n1(r.p10)} – ${n1(r.p50)} – ${n1(r.p90)}` : '—';
    return `<tr><td><b>${esc(s)}</b></td><td>${n0(real)}</td><td>${n1(med)}</td>
      <td>${flag30(real, med)}</td><td class="muted">${band}</td><td>${interval}</td></tr>`;
  }).join('');
  const vals = STAGES.map(s => c.real[s]);
  const peak = Math.max(...vals.filter(v => v !== null && v !== undefined), 0);
  const spark = STAGES.map(s => {
    const v = c.real[s], r = c.reg[s];
    const cls = r && r.ok === false ? ' spark-bad' : (r && r.ok === true ? ' spark-ok' : '');
    const h = (v === null || v === undefined || peak <= 0) ? 2
      : Math.max(6, Math.round(v / peak * 100)) + '%';
    return `<span class="spark-bar${cls}" style="height:${h===2?'2px':h}"></span>`;
  }).join('');
  return `<div class="table-wrap"><table class="tbl">
    <thead><tr><th>Стадия</th><th>Факт</th><th>Медиана DT</th><th>Расхождение</th>
      <th>TabPFN-3 P10 · P50 · P90</th><th>Интервал</th></tr></thead>
    <tbody>${rows}</tbody></table></div><div class="spark">${spark}</div>
    <div class="muted" style="font-size:11.5px;margin-top:6px">
      Медиана DT — Монте-Карло пайплайн Digital Twin; P10–P90 — независимый
      количественный прогноз TabPFN-3 (QA-слой, не заменяет реконсиляцию по исходу).</div>`;
}

function cohortBlock(c){
  /* Вероятности сравниваются только внутри когорты отчёта: у циклов вне её
     (c.scope) это оценка для гипотетического переноса, другая величина. */
  const specs = [
    ['PRAI (TabPFN)', x => x.scope ? null : x.prob.prai, pct],
    ['DIGITAL TWIN',  x => x.scope ? null : x.prob.dt,   pct],
    ['ООК',           x => x.real['ООК'],       n0],
    ['2PN',           x => x.real['2PN'],       n0],
    ['Бластоцисты',   x => x.real['Бласт'],     n0],
    ['Хор. бластоцисты', x => x.real['Хор.бласт'], n0],
    ['Возраст',       x => x.age, n0],
    ['AMH',           x => x.amh, n1],
  ];
  return `<div class="cohort">` + specs.map(([label, get, fmt]) => {
    const arr = DATA.cycles.map(get);
    const v = get(c), p = percentile(arr, v), med = median(arr);
    if (v === null || v === undefined || p === null)
      return `<div class="crow"><div class="clab">${esc(label)}</div>
        <div class="ctrack"></div><div class="cval">—</div></div>`;
    const mx = Math.max(...arr.filter(x => x !== null && x !== undefined));
    const posMe = mx > 0 ? Math.min(100, v / mx * 100) : 0;
    const posMed = mx > 0 ? Math.min(100, med / mx * 100) : 0;
    return `<div class="crow">
      <div class="clab">${esc(label)}</div>
      <div class="ctrack"><div class="cfill" style="width:${posMe}%"></div>
        <div class="cmed" style="left:${posMed}%" title="медиана когорты"></div>
        <div class="cme" style="left:${posMe}%" title="эта пациентка"></div></div>
      <div class="cval">${fmt(v)} · ${Math.round(p*100)}-й перц.</div></div>`;
  }).join('') + `</div><div class="muted" style="font-size:11.5px;margin-top:8px">
    Тонкая метка — медиана когорты этой клиники, жирная — текущий цикл.
    Перцентиль — доля циклов клиники со значением не выше.</div>`;
}

function riskBlock(c){
  const r = c.risk;
  const cl = r.cluster === null || r.cluster === undefined ? '—' : CLUSTERS[r.cluster] || r.cluster;
  return `<div class="kv">
    <div>Кластер: <b>${esc(cl)}</b> <span class="muted">(C0 ${pct(r.c0)} · C1 ${pct(r.c1)} · C2 ${pct(r.c2)})</span></div>
    <div>Жизнеспособный эмбрион: <b>${pct(r.viable)}</b></div>
    <div>Риск отмены: <b>${pct(r.cancel)}</b></div>
    <div>Без бластоцист: <b>${pct(r.no_blast)}</b></div>
    <div>Без хор. бластоцист: <b>${pct(r.no_good)}</b></div>
    <div>OHSS умеренный / тяжёлый: <b>${pct(r.ohss_mod)}</b> / <b>${pct(r.ohss_sev)}</b></div>
    <div>Banking: P на MII <b>${pct(r.bank_mii)}</b></div>
    <div>Banking: ожид. эуплоидных <b>${n1(r.bank_eup)}</b></div></div>`;
}

function reviewBlock(c){
  if (!c.review || c.review.priority === null) return '';
  const p = c.review.priority;
  const cls = p === 1 ? 'badge-crit' : (p === 2 ? 'badge-warn' : 'badge-ok');
  const lab = p === 1 ? '1 Критично' : (p === 2 ? '2 Рекомендовано' : '3 В норме');
  return `<div class="sec"><h3>Ревью-лист</h3>
    <span class="badge ${cls}">${lab}</span>
    <div style="margin-top:8px">${esc(c.review.reason || '')}</div>
    ${c.prob.ci_wide ? `<div class="note">Широкий доверительный интервал TabPFN
      (${n1(c.prob.ci_width)} пп) — прогноз по этому циклу нестабилен; в таких случаях
      пайплайн поднимает форензику логов MIRI и смен лотов расходников.</div>` : ''}</div>`;
}

function ozemaBlock(c){
  if (!c.ozema) return '';
  const o = c.ozema;
  const cell = v => v === 'flag' ? "<span class='badge badge-crit'>flag</span>"
    : (v === 'ok' ? "<span class='badge badge-ok'>ok</span>"
    : "<span class='muted'>н/д</span>");
  const flagged = o.flag && o.flag !== '—' && o.flag !== 'н/д';
  return `<div class="sec"><h3>OZEMA — арест-паттерны цикла</h3>
    <div style="margin-bottom:9px">Флаг цикла:
      ${flagged ? `<span class='badge badge-crit'>${esc(o.flag)}</span>`
                : `<span class='badge badge-ok'>${esc(o.flag || '—')}</span>`}
      ${o.flagged_n ? `<span class="muted"> · флагов у пациентки: ${n0(o.flagged_n)}</span>` : ''}
    </div>
    <div class="table-wrap"><table class="tbl">
      <thead><tr><th>OMA</th><th>FF</th><th>ZA</th><th>EEA</th></tr></thead>
      <tbody><tr><td>${cell(o.oma)}</td><td>${cell(o.ff)}</td><td>${cell(o.za)}</td>
        <td>${cell(o.eea)}</td></tr></tbody></table></div>
    ${o.note ? `<div class="note">${esc(o.note)}</div>` : ''}
    <div class="muted" style="font-size:11.5px;margin-top:8px">
      Односторонний биномиальный тест (α=0.05) против порогов 75%/60%/95%/40%,
      n&lt;4 → «н/д». Это не диагноз.</div></div>`;
}

function historyBlock(c){
  const h = c.hist || [];
  const same = DATA.cycles.filter(x => x.pid && x.pid === c.pid && x.uid !== c.uid);
  if (!h.length && !same.length) return '';
  let out = '<div class="sec"><h3>Прослеживание</h3>';
  if (h.length){
    out += `<div class="table-wrap"><table class="tbl">
      <thead><tr><th>Прогон</th><th>Когда</th><th>PRAI</th><th>DIGITAL TWIN</th>
        <th>KAT</th><th>Laya</th><th>med Bl</th></tr></thead><tbody>` +
      h.map(x => `<tr><td>${esc(x.run)}</td>
        <td class="muted">${esc((x.ts||'').slice(0,16).replace('T',' '))}</td>
        <td>${pct(x.prai)}</td><td>${pct(x.dt)}</td><td>${pct(x.kat)}</td>
        <td>${pct(x.laya)}</td><td>${n1(x.bl)}</td></tr>`).join('') + '</tbody></table></div>';
    if (h.length === 1)
      out += `<div class="muted" style="font-size:11.5px;margin-top:6px">
        Этот цикл пока присутствует в одном прогоне — строки добавляются
        при каждой следующей синхронизации.</div>`;
  }
  if (same.length){
    out += `<h3 style="margin-top:14px">Другие циклы этой пациентки</h3>` +
      same.map(x => `<div class="crow" style="grid-template-columns:110px 1fr 110px">
        <div class="clab">${esc(x.opu || '—')}</div>
        <div><a href="#uid=${esc(x.uid)}" onclick="select('${esc(x.uid)}',true)">
          попытка ${n0(x.attempt)} · ООК ${n0(x.real['ООК'])} → бласт ${n0(x.real['Бласт'])}</a></div>
        <div class="cval">${pct(x.prob.prai)}</div></div>`).join('');
  }
  return out + '</div>';
}

function select(uid, push){
  const c = DATA.cycles.find(x => x.uid === uid);
  SEL = uid;
  renderList($('#q').value);
  if (!c){ $('#pane').innerHTML = "<div class='empty'>Цикл не найден</div>"; return; }
  if (push) history.replaceState(null, '', '#uid=' + encodeURIComponent(uid));
  const p = c.prob.prai;
  const pill = (p === null || p === undefined)
    ? "<span class='muted'>прогноз недоступен</span>"
    : `<span class="prob-pill prob-${tier(p)}">${pct(p)}</span>`;
  let ci = '';
  if (c.prob.ci_low !== null && c.prob.ci_high !== null && c.prob.ci_low !== undefined){
    const lo = Math.max(0, Math.min(100, c.prob.ci_low*100));
    const hi = Math.max(0, Math.min(100, c.prob.ci_high*100));
    const me = Math.max(0, Math.min(100, (p||0)*100));
    ci = `<div class="ci-track"><span class="ci-range" style="left:${lo}%;
      width:${Math.max(1,hi-lo)}%"></span>
      <span class="ci-tick" style="left:${me}%"></span></div>
      <div class="muted" style="font-size:11.5px;margin-top:4px">95% ДИ TabPFN
      ${pct(c.prob.ci_low)} – ${pct(c.prob.ci_high)} (${n1(c.prob.ci_width)} пп)</div>`;
  }
  $('#pane').innerHTML = `
    <div class="eyebrow">${esc(DATA.clinic)} · карточка цикла</div>
    <h2 class="card-h1">${esc(c.name)}</h2>
    <div class="muted" style="margin-bottom:4px">ID ${esc(c.pid || '—')} ·
      cycle_uid <code>${esc(c.uid)}</code>${c.rec ? ` · <a href="${esc(c.rec)}"
      target="_blank">открыть запись в Teable</a>` : ''}</div>
    <div style="margin:9px 0 4px">${outcomeBadge(c.preg)}</div>
    ${c.archived ? `<div class="note">Карточка собрана из базы Teable по прогону
      <code>${esc(c.archived)}</code> — этого цикла нет в текущей выгрузке
      (<code>${esc(DATA.source)}</code>). Числа те же, что были синхронизированы
      тогда; поля, которых база не хранит (источник спермы, отдельный выход
      TabPFN), здесь пусты — это отсутствие в архиве, а не отсутствие в цикле.</div>` : ''}
    ${c.scope ? `<div class="note"><b>${esc(c.scope)}.</b> Цикла нет в последнем
      прогоне отчёта (туда входят переносы с известным исходом), поэтому он не
      участвует в KPI, калибровке и ревью-листе. ${c.tier === 'full'
        ? 'Вероятности беременности — оценка моделей для переноса эмбриона этого цикла;' +
          ' если переноса не было, исходом она не проверяется.'
        : c.tier === 'funnel'
        ? 'Зигот нет (2PN=0), поэтому вероятность беременности не считалась: показана' +
          ' только ожидаемая воронка Digital Twin из точки после пункции против факта.'
        : 'Модели к циклу не применялись — данных для прогноза нет.'}</div>` : ''}
    <div class="stats">
      <div class="jstat"><div class="jsl">Возраст</div><div class="jsv">${n0(c.age)}</div></div>
      <div class="jstat"><div class="jsl">AMH</div><div class="jsv">${n1(c.amh)}</div></div>
      <div class="jstat"><div class="jsl">BMI</div><div class="jsv">${n1(c.bmi)}</div></div>
      <div class="jstat"><div class="jsl">Попытка</div><div class="jsv">${n0(c.attempt)}</div></div>
      <div class="jstat"><div class="jsl">OPU</div><div class="jsv" style="font-size:14px">${esc(c.opu||'—')}</div></div>
      <div class="jstat"><div class="jsl">ET</div><div class="jsv" style="font-size:14px">${esc(c.et||'—')}</div></div>
    </div>
    <div class="sec"><h3>Вероятность беременности</h3>
      ${pill}${ci}
      <div style="margin-top:13px">${modelsBlock(c)}</div>
      <div class="muted" style="font-size:11.5px;margin-top:9px">
        Модели показаны как есть, без взаимного ранжирования: на когортах этого
        размера разница между ними статистически неотличима от шума.</div></div>
    <div class="sec"><h3>Воронка цикла: факт против прогнозов</h3>${funnelBlock(c)}</div>
    <div class="sec"><h3>Риски и структура цикла</h3>${riskBlock(c)}</div>
    ${reviewBlock(c)}
    ${ozemaBlock(c)}
    <div class="sec"><h3>Место цикла в когорте клиники</h3>${cohortBlock(c)}</div>
    ${historyBlock(c)}
    <div class="sec"><h3>Протокол</h3><div class="kv">
      <div>Метод: <b>${esc(c.method || '—')}</b></div>
      <div>Эмбриолог: <b>${esc(c.emb || '—')}</b></div>
      <div>Врач: <b>${esc(c.doc || '—')}</b></div>
      <div>Источник спермы: <b>${esc(c.sperm || '—')}</b></div>
      <div>AFC: <b>${n0(c.afc)}</b></div></div></div>
    <div class="foot">${c.archived
        ? `Источник: база Teable, прогон <code>${esc(c.archived)}</code>`
        : `Данные прогона: ${esc(c.src || DATA.source)}`} · собрано ${esc(DATA.built)}.
      Числа приходят посчитанными из пайплайна; страница только отображает их.</div>`;
  window.scrollTo(0, 0);
}

function boot(){
  $('#q').addEventListener('input', e => renderList(e.target.value));
  const hash = decodeURIComponent(location.hash.replace(/^#/, ''));
  let start = DATA.cycles.length ? DATA.cycles[0].uid : null;
  if (hash.startsWith('uid=')){
    start = hash.slice(4);
  } else if (hash.startsWith('id=') || hash.startsWith('q=')){
    const q = hash.slice(hash.indexOf('=') + 1);
    $('#q').value = q;
    const hit = DATA.cycles.find(c =>
      (c.name + ' ' + (c.pid||'')).toLowerCase().includes(q.toLowerCase()));
    if (hit) start = hit.uid;
  }
  renderList($('#q').value);
  if (start) select(start, false);
}
document.addEventListener('DOMContentLoaded', boot);
"""


def render_clinic(payload: dict, base_url: str, teable_url: str, base_id: str) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    stages = json.dumps([s[0] for s in STAGE_SOURCES], ensure_ascii=False)
    models = json.dumps(MODEL_ROW, ensure_ascii=False)
    clusters = json.dumps({0: "C0 Standard", 1: "C1 Poor", 2: "C2 High"}, ensure_ascii=False)
    n = len(payload.get("cycles", []))
    n_rev = sum(1 for c in payload["cycles"] if c.get("review"))
    n_out = sum(1 for c in payload["cycles"] if c.get("preg") is not None)
    base_link = (f"<span class='pill'><a href='{html.escape(teable_url)}/base/"
                 f"{html.escape(base_id)}' target='_blank'>Открыть базу в Teable</a></span>"
                 if base_id else "")
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(payload['clinic'])} — карточки циклов</title>
<style>{_css()}</style></head><body>
<header class="hero"><div class="eyebrow">Digital Twin · локальная база прогнозов</div>
<h1>{html.escape(payload['clinic'])} — карточки циклов</h1>
<div class="meta">
  <span class="pill">{n} циклов</span>
  <span class="pill">{n_out} с известным исходом</span>
  <span class="pill">{n_rev} в ревью-листе</span>
  <span class="pill">собрано {html.escape(payload['built'])}</span>
  <span class="pill"><a href="{html.escape(base_url)}/index.html">Все клиники</a></span>
  {base_link}
</div></header>
<div class="app">
  <aside class="side">
    <div class="side-head">
      <input id="q" class="search" placeholder="Поиск: ФИО, ID или cycle_uid…"
             autocomplete="off">
      <div id="count" class="side-count"></div>
    </div>
    <div id="list"></div>
  </aside>
  <main id="pane" class="pane"><div class="empty">Выберите пациентку слева</div></main>
</div>
<script>
const DATA={data};
const STAGES={stages};
const MODELS={models};
const CLUSTERS={clusters};
{_JS}
</script></body></html>
"""


def render_index(entries: list[dict], base_url: str) -> str:
    cards = "".join(
        f"""<a class="sec" style="display:block;text-decoration:none;color:inherit"
             href="{html.escape(base_url)}/{html.escape(e['code'])}.html">
        <h3 style="margin:0 0 4px">{html.escape(e['clinic'])}</h3>
        <div class="muted" style="font-size:12.5px">{e['n']} циклов ·
        {e['n_out']} с исходом · {e['n_rev']} в ревью-листе ·
        собрано {html.escape(e['built'])}</div></a>""" for e in entries)
    if not cards:
        cards = "<div class='empty'>Нет собранных клиник</div>"
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Карточки циклов — все клиники</title>
<style>{_css()}</style></head><body>
<header class="hero"><div class="eyebrow">Digital Twin · локальная база прогнозов</div>
<h1>Карточки циклов</h1>
<div class="meta"><span class="pill">{len(entries)} клиник</span></div></header>
<div class="pane" style="max-width:760px">{cards}
<div class="foot">Страницы обновляются командой <code>/teable-sync</code> вместе с
заливкой в Teable. Данные локальные, наружу не уходят.</div></div>
</body></html>
"""


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def build(cfg: dict, only: str | None = None, history: bool = True) -> list[dict]:
    out_dir = tc.cards_root(cfg)
    os.makedirs(out_dir, exist_ok=True)
    base_url = tc.cards_base_url(cfg)
    teable_url = cfg.get("teable_url", "http://127.0.0.1:3000")
    tb = tc.connect(cfg) if history else None

    entries = []
    for clinic in tc.clinics(cfg, only):
        hist, links, archive = {}, {}, {}
        if tb is not None and clinic.configured:
            hist = fetch_history(tb, clinic.base_id)
            links = fetch_record_links(tb, clinic.base_id, teable_url)
            archive = fetch_archive(tb, clinic.base_id)
        payload = build_payload(clinic, hist, links, archive)
        if not payload:
            print(f"  [skip] {clinic.name}: нет ни dt_predictions.csv в "
                  f"{clinic.processed_dir}, ни циклов в базе")
            continue
        page = os.path.join(out_dir, f"{clinic.code}.html")
        tmp = page + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(render_clinic(payload, base_url, teable_url, clinic.base_id))
        os.replace(tmp, page)
        entry = {"code": clinic.code, "clinic": clinic.name,
                 "n": len(payload["cycles"]), "built": payload["built"],
                 "n_out": sum(1 for c in payload["cycles"] if c.get("preg") is not None),
                 "n_rev": sum(1 for c in payload["cycles"] if c.get("review"))}
        entry["n_arch"] = payload.get("n_archived", 0)
        entries.append(entry)
        arch = (f", из них {entry['n_arch']} из базы (прошлые прогоны)"
                if entry["n_arch"] else "")
        print(f"  cards: {clinic.name} -> {entry['n']} циклов{arch} "
              f"({round(os.path.getsize(page)/1024)} КБ) {base_url}/{clinic.code}.html")

    # The index always lists every clinic already built, not just this run's.
    known = {e["code"]: e for e in entries}
    for clinic in tc.clinics(cfg):
        if clinic.code in known:
            continue
        if os.path.exists(os.path.join(out_dir, f"{clinic.code}.html")):
            known[clinic.code] = {"code": clinic.code, "clinic": clinic.name,
                                  "n": 0, "n_out": 0, "n_rev": 0, "built": "ранее"}
    idx = os.path.join(out_dir, "index.html")
    with open(idx + ".tmp", "w", encoding="utf-8") as fh:
        fh.write(render_index(list(known.values()), base_url))
    os.replace(idx + ".tmp", idx)
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=tc.DEFAULT_CONFIG)
    ap.add_argument("--clinic", help="clinic code (default: every clinic)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-history", action="store_true",
                    help="skip the per-cycle run history read from Teable")
    args = ap.parse_args()
    cfg = tc.load_config(args.config)
    build(cfg, only=args.clinic, history=not args.no_history)
    print("Done.")


if __name__ == "__main__":
    main()
