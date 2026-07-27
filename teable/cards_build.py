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
    try:
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
                  record_links: dict | None = None) -> dict:
    proc = clinic.processed_dir
    pred = _read(os.path.join(proc, "dt_predictions.csv"))
    if pred.empty:
        return {}

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
                      "dnn": _str(r.get("dnn_validation_status"))},
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
        })

    cycles.sort(key=lambda c: (c["opu"] or "", c["name"]), reverse=True)
    return {
        "clinic": clinic.name, "code": clinic.code,
        "built": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "source": os.path.basename(clinic.processed_dir.rstrip("\\/")) or "processed",
        "cycles": cycles,
    }


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
        })
    for rows in out.values():
        rows.sort(key=lambda x: str(x.get("ts") or ""))
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
  $('#count').textContent = rows.length + ' из ' + DATA.cycles.length + ' циклов';
  $('#list').innerHTML = rows.map(c => {
    const p = c.prob.prai;
    const pillCls = p === null || p === undefined ? '' : ' badge-' +
      (tier(p) === 'low' ? 'crit' : tier(p) === 'mid' ? 'warn' : 'ok');
    const rev = c.review && c.review.priority === 1
      ? "<span class='badge badge-crit'>P1</span>" : '';
    return `<div class="pat${c.uid===SEL?' on':''}" data-uid="${esc(c.uid)}">
      <div class="pn">${esc(c.name)}</div>
      <div class="ps"><span>${esc(c.opu || '—')}</span><span>${esc(c.pid || '')}</span>
        <span class="badge${pillCls}">${pct(p)}</span>${rev}</div></div>`;
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
  return `<div class="models">${tiles}</div>${warn}`;
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
  const specs = [
    ['PRAI (TabPFN)', x => x.prob.prai, pct],
    ['DIGITAL TWIN',  x => x.prob.dt,   pct],
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
        <th>KAT</th><th>med Bl</th></tr></thead><tbody>` +
      h.map(x => `<tr><td>${esc(x.run)}</td>
        <td class="muted">${esc((x.ts||'').slice(0,16).replace('T',' '))}</td>
        <td>${pct(x.prai)}</td><td>${pct(x.dt)}</td><td>${pct(x.kat)}</td>
        <td>${n1(x.bl)}</td></tr>`).join('') + '</tbody></table></div>';
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
    <div class="foot">Данные прогона: ${esc(DATA.source)} · собрано ${esc(DATA.built)}.
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
        hist, links = {}, {}
        if tb is not None and clinic.configured:
            hist = fetch_history(tb, clinic.base_id)
            links = fetch_record_links(tb, clinic.base_id, teable_url)
        payload = build_payload(clinic, hist, links)
        if not payload:
            print(f"  [skip] {clinic.name}: нет dt_predictions.csv в {clinic.processed_dir}")
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
        entries.append(entry)
        print(f"  cards: {clinic.name} -> {entry['n']} циклов "
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
