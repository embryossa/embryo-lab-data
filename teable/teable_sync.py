#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
teable_sync.py — Auto-upload IVF Digital Twin /run-analysis results into Teable.

Reads the intermediate files produced by dt_batch.py (the /run-analysis command)
from the `processed` folder and pushes them into a per-clinic Teable base:

    Runs                 — one record per analysis run (batch snapshot)
    Cycles               — patient/cycle master (upserted by patient ID)
    Predictions          — DT model outputs, one record per patient per run
    Prediction vs Reality — funnel prediction-vs-fact + review priority
    Calibration          — per-run model metrics (Brier / AUC / Acc / ...)
    Files                — the raw CSV/XLSX files themselves, as attachments

Idempotent between runs:
  * a Run is keyed by content hash of dt_predictions.csv — re-running sync on the
    same files does NOT create a duplicate run (use --force to override);
  * Cycles are upserted by patient ID (never duplicated across runs);
  * Predictions / PvR / Calibration / Files are keyed per (patient|run) or
    (run|model) or (file|run) so repeated syncs update in place.

Usage:
    python teable_sync.py                 # uses teable_config.json next to this file
    python teable_sync.py --config C:\\path\\to\\teable_config.json
    python teable_sync.py --dry-run       # parse + report, write nothing
    python teable_sync.py --force         # create a new run even if hash seen
    python teable_sync.py --no-files      # skip attachment upload

Dependencies:  pip install requests pandas openpyxl
Optional (for calibration metrics):  pip install scikit-learn numpy
"""

import argparse
import hashlib
import json
import os
import sys
import datetime as dt

try:
    import requests
except ImportError:
    sys.exit("Missing dependency 'requests'. Run: pip install requests pandas openpyxl")

try:
    import pandas as pd
except ImportError:
    sys.exit("Missing dependency 'pandas'. Run: pip install requests pandas openpyxl")


# ---------------------------------------------------------------------------
# Column maps — adjust these to match your actual file headers if they differ.
# Left side = your file column; the code tries each candidate in order.
# ---------------------------------------------------------------------------

OPU_COLS = {
    # "_key" is the cycle-level join key (raw_id#date_opu) shared with
    # dt_predictions.csv's own "patient_id"/"_key" column. Bare "ID" repeats
    # across multi-attempt patients and must not be used to join real vs pred.
    "patient_id":   ["_key", "ID", "patient_id", "Patient ID"],
    "name":         ["Patient Full Name", "ФИО", "Имя", "Name"],
    "dob":          ["DOB", "Дата рождения"],
    "age":          ["Age", "Возраст"],
    "amh":          ["AMH", "АМГ"],
    "afc":          ["AFC", "КАФ"],
    "bmi":          ["BMI", "ИМТ"],
    "attempt":      ["Attempt", "Попытка"],
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
    "prai":         ["PRAI"],
    "dt_headline":  ["DIGITAL TWIN", "DigitalTwin", "DT"],
}

PRED_COLS = {
    "patient_id":   ["_key", "patient_id", "ID", "Patient ID"],
    "timestamp":    ["timestamp", "Timestamp", "created"],
    "med_okk":      ["med_okk", "med_OCC"],
    "med_mii":      ["med_mii"],
    "med_pn2":      ["med_pn2"],
    "med_blasts":   ["med_blasts", "med_bl"],
    "med_good":     ["med_good"],
    "p_per_transfer": ["p_per_transfer"],
    "p_overall":    ["p_overall_cycle", "p_overall"],
    "bayes_mean":   ["bayes_mean"],
    "p_kat":        ["p_kat_raw", "p_kat", "KAT"],
    "p_nvsa":       ["p_nvsa", "NVSA"],
    "p_csdi":       ["p_csdi", "CSDI"],
    "dt_headline":  ["DIGITAL TWIN", "p_digital_twin", "DT"],
    "ohss_risk":    ["ohss_risk", "OHSS"],
    "p_cancel":     ["p_cancel"],
    "banking":      ["banking", "Esteves", "banking_strategy"],
    "cluster":      ["dominant_cluster", "cluster"],
}

# Single-select label maps
CLUSTER_LABELS = {"0": "C0 Standard", "1": "C1 Poor", "2": "C2 High",
                  "c0": "C0 Standard", "c1": "C1 Poor", "c2": "C2 High"}


# ---------------------------------------------------------------------------
# Teable API client
# ---------------------------------------------------------------------------

class Teable:
    def __init__(self, url, token, dry_run=False):
        self.url = url.rstrip("/")
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}",
                               "Accept": "application/json"})
        self.dry_run = dry_run

    def _get(self, path, **kw):
        r = self.s.get(self.url + path, timeout=60, **kw)
        r.raise_for_status()
        return r.json()

    def tables(self, base_id):
        return self._get(f"/api/base/{base_id}/table")

    def fields(self, table_id):
        return self._get(f"/api/table/{table_id}/field")

    def all_records(self, table_id):
        """Fetch every record (paged), fieldKeyType=name."""
        out, skip = [], 0
        while True:
            page = self._get(f"/api/table/{table_id}/record",
                             params={"take": 1000, "skip": skip, "fieldKeyType": "name"})
            recs = page.get("records", [])
            out.extend(recs)
            if len(recs) < 1000:
                break
            skip += 1000
        return out

    def create(self, table_id, fields):
        if self.dry_run:
            return {"id": "DRYRUN"}
        r = self.s.post(f"{self.url}/api/table/{table_id}/record",
                        json={"fieldKeyType": "name", "typecast": True,
                              "records": [{"fields": fields}]}, timeout=60)
        r.raise_for_status()
        return r.json()["records"][0]

    def update(self, table_id, record_id, fields):
        if self.dry_run:
            return {"id": record_id}
        r = self.s.patch(f"{self.url}/api/table/{table_id}/record/{record_id}",
                         json={"fieldKeyType": "name", "typecast": True,
                               "record": {"fields": fields}}, timeout=60)
        r.raise_for_status()
        return r.json()

    def upload_attachment(self, table_id, record_id, field_id, filepath):
        if self.dry_run:
            return {"id": record_id}
        with open(filepath, "rb") as fh:
            r = self.s.post(
                f"{self.url}/api/table/{table_id}/record/{record_id}/{field_id}/uploadAttachment",
                files={"file": (os.path.basename(filepath), fh)}, timeout=300)
        r.raise_for_status()
        return r.json()


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
        return int(f) if f.is_integer() else f
    except (ValueError, TypeError):
        return None


def norm_id(v):
    return str(v).strip().upper() if v is not None and pd.notna(v) else None


def iso_date(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return pd.Timestamp(v).date().isoformat()
    except (ValueError, TypeError):
        return None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def preg_label(v):
    if v is None or pd.isna(v):
        return "Неизвестно"
    try:
        return "Беременность" if int(float(v)) == 1 else "Нет"
    except (ValueError, TypeError):
        return "Неизвестно"


def flag30(real, pred):
    if real in (None, 0) or pred is None:
        return "— н/д"
    return "✅ в норме" if abs(pred - real) / real <= 0.30 else "⚠️ вне ±30%"


def read_any(path):
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Main sync
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--config", default=os.path.join(here, "teable_config.json"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-files", action="store_true")
    ap.add_argument("--base", help="override base_id (e.g. per clinic)")
    ap.add_argument("--clinic", help="override clinic name")
    ap.add_argument("--processed", help="override processed_dir (e.g. a clinic subfolder)")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = json.load(fh)

    if args.base:
        cfg["base_id"] = args.base
    if args.clinic:
        cfg["clinic"] = args.clinic
    if args.processed:
        cfg["processed_dir"] = args.processed

    proc = cfg["processed_dir"]
    fnames = cfg.get("files", {})
    f_pred = os.path.join(proc, fnames.get("dt_predictions", "dt_predictions.csv"))
    f_opu = os.path.join(proc, fnames.get("opu_table", "OPU_table_enriched.xlsx"))

    tb = Teable(cfg.get("teable_url", "https://app.teable.ai"), cfg["token"],
                dry_run=args.dry_run)

    # Resolve table IDs by name (portable across per-clinic bases)
    tmap = {t["name"].split(" —")[0].strip(): t["id"] for t in tb.tables(cfg["base_id"])}
    def tid(short):
        for k, v in tmap.items():
            if k.lower().startswith(short.lower()):
                return v
        raise SystemExit(f"Table starting with '{short}' not found in base.")

    T_RUNS, T_CYC = tid("Runs"), tid("Cycles")
    T_PRED, T_PVR = tid("Predictions"), tid("Prediction vs Reality")
    T_CAL, T_FIL = tid("Calibration"), tid("Files")

    # ---- Run identity (content hash → idempotent between runs) --------------
    if not os.path.exists(f_pred):
        sys.exit(f"Predictions file not found: {f_pred}")
    run_hash = sha256_file(f_pred)[:12]
    now = dt.datetime.now()
    run_id = f"{cfg.get('clinic', 'RUN')}-{now:%Y%m%d-%H%M}-{run_hash}"

    existing_runs = {r["fields"].get("Run ID"): r["id"] for r in tb.all_records(T_RUNS)}
    dup = next((rid for k, rid in existing_runs.items()
                if k and k.endswith(run_hash)), None)
    if dup and not args.force:
        print(f"[skip] These results (hash {run_hash}) are already in Teable as a run. "
              f"Use --force to load again.")
        return

    # ---- Load data ---------------------------------------------------------
    df_pred = read_any(f_pred)
    df_opu = read_any(f_opu) if os.path.exists(f_opu) else pd.DataFrame()

    # Build patient dict keyed by normalized id, merging OPU (real) + preds
    patients = {}

    def slot(pid):
        return patients.setdefault(pid, {"real": {}, "pred": {}})

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
        d = slot(pid)["pred"]
        for key, cands in PRED_COLS.items():
            d[key] = pick(r, cands)

    n_pat = len(patients)
    n_out = sum(1 for p in patients.values()
                if p["real"].get("preg") is not None and pd.notna(p["real"].get("preg")))
    n_pos = sum(1 for p in patients.values()
                if str(p["real"].get("preg")) in ("1", "1.0"))
    print(f"Parsed: {n_pat} patients, {n_out} with outcome (+{n_pos}). Run: {run_id}")

    if args.dry_run:
        print("[dry-run] no data written.")
        return

    # ---- 1) Run record -----------------------------------------------------
    run_rec = tb.create(T_RUNS, {
        "Run ID": run_id,
        "Дата прогона": now.isoformat(),
        "Входной файл": os.path.basename(f_pred),
        "Пациентов всего": n_pat,
        "С исходом": n_out,
        "Положительных исходов": n_pos,
        "Версия DT": cfg.get("dt_version", "v7.0"),
        "Заметки": f"sha256[pred]={run_hash}; clinic={cfg.get('clinic','')}",
    })
    run_ref = run_id  # link-by-name via typecast

    # ---- 2) Cycles upsert by patient id ------------------------------------
    cyc_index = {r["fields"].get("ID пациента"): r["id"] for r in tb.all_records(T_CYC)}
    for pid, p in patients.items():
        real, pred = p["real"], p["pred"]
        cl = str(real.get("cluster") if real.get("cluster") is not None
                 else pred.get("cluster") or "").strip().lower()
        fields = {
            "ID пациента": pid,
            "Имя": (str(real.get("name")) if real.get("name") else None),
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
            "Исход": preg_label(real.get("preg")),
            "Кластер": CLUSTER_LABELS.get(cl.replace("c", "").strip()
                                          if cl else "", None) or CLUSTER_LABELS.get(cl, None),
        }
        fields = {k: v for k, v in fields.items() if v is not None}
        if pid in cyc_index:
            tb.update(T_CYC, cyc_index[pid], fields)
        else:
            rec = tb.create(T_CYC, fields)
            cyc_index[pid] = rec["id"]

    # ---- 3) Predictions per patient per run --------------------------------
    for pid, p in patients.items():
        pred = p["pred"]
        if not pred:
            continue
        tb.create(T_PRED, {
            "Ключ": f"{pid}::{run_id}",
            "Пациентка": pid,
            "Прогон": run_ref,
            "Timestamp": now.isoformat(),
            "med OCC": num(pred.get("med_okk")),
            "med MII": num(pred.get("med_mii")),
            "med 2PN": num(pred.get("med_pn2")),
            "med Bl": num(pred.get("med_blasts")),
            "med Good": num(pred.get("med_good")),
            "P per transfer": num(pred.get("p_per_transfer")),
            "P overall cycle": num(pred.get("p_overall")),
            "Bayes mean": num(pred.get("bayes_mean")),
            "KAT": num(pred.get("p_kat")),
            "NVSA": num(pred.get("p_nvsa")),
            "CSDI": num(pred.get("p_csdi")),
            "DIGITAL TWIN": num(pred.get("dt_headline")),
            "OHSS risk": num(pred.get("ohss_risk")),
            "P cancel": num(pred.get("p_cancel")),
            "Banking": (str(pred.get("banking")) if pred.get("banking") is not None else None),
        })

    # ---- 4) Prediction vs Reality ------------------------------------------
    for pid, p in patients.items():
        real, pred = p["real"], p["pred"]
        occ_r, occ_p = num(real.get("occ")), num(pred.get("med_okk"))
        bl_r, bl_p = num(real.get("bl")), num(pred.get("med_blasts"))
        gb_r, gb_p = num(real.get("good_bl")), num(pred.get("med_good"))
        kat, prai = num(pred.get("p_kat")), num(real.get("prai"))
        kat_prai = round((kat - prai) * 100, 1) if (kat is not None and prai is not None) else None
        tb.create(T_PVR, {
            "Ключ": f"{pid}::{run_id}",
            "Пациентка": pid,
            "Прогон": run_ref,
            "OCC факт": occ_r, "OCC прогноз": occ_p, "OCC ±30%": flag30(occ_r, occ_p),
            "Bl факт": bl_r, "Bl прогноз": bl_p, "Bl ±30%": flag30(bl_r, bl_p),
            "Good Bl факт": gb_r, "Good Bl прогноз": gb_p, "Good ±30%": flag30(gb_r, gb_p),
            "KAT−PRAI (пп)": kat_prai,
        })

    # ---- 5) Calibration metrics (optional; needs sklearn) ------------------
    try:
        from sklearn.metrics import roc_auc_score, brier_score_loss
        model_cols = {"MC": "p_per_transfer", "Bayes": "bayes_mean", "KAT": "p_kat",
                      "NVSA": "p_nvsa", "CSDI": "p_csdi", "DT": "dt_headline"}
        rows = [(p["pred"], p["real"].get("preg")) for p in patients.values()
                if p["real"].get("preg") is not None and pd.notna(p["real"].get("preg"))]
        written = 0
        if len(rows) >= 5:
            y = [int(float(o)) for _, o in rows]
            for mname, mkey in model_cols.items():
                ps = [num(pr.get(mkey)) for pr, _ in rows]
                pairs = [(a, b) for a, b in zip(ps, y) if a is not None]
                if len(pairs) < 5:
                    continue
                pv = [a for a, _ in pairs]; yv = [b for _, b in pairs]
                try:
                    auc = roc_auc_score(yv, pv) if len(set(yv)) > 1 else None
                except ValueError:
                    auc = None
                brier = brier_score_loss(yv, pv)
                acc = sum(1 for a, b in zip(pv, yv) if (a >= 0.5) == (b == 1)) / len(yv)
                tb.create(T_CAL, {
                    "Ключ": f"{run_id}::{mname}",
                    "Прогон": run_ref, "Модель": mname,
                    "Brier": round(brier, 4),
                    "AUC-ROC": round(auc, 4) if auc is not None else None,
                    "Acc@0.5": round(acc, 4),
                    "Mean pred": round(sum(pv) / len(pv), 4),
                    "Real rate": round(sum(yv) / len(yv), 4),
                    "N": len(yv),
                })
                written += 1
        print(f"Calibration: {written} model(s) written ({len(rows)} patients with known outcome).")
    except ImportError:
        print("[info] scikit-learn not installed — calibration metrics skipped.")

    # ---- 6) Files: attach every file in processed_dir to this run ---------
    if not args.no_files and cfg.get("upload_files", True):
        fil_field = next((f["id"] for f in tb.fields(T_FIL) if f["type"] == "attachment"), None)
        fil_index = {r["fields"].get("Имя файла"): r["id"] for r in tb.all_records(T_FIL)}
        try:
            entries = sorted(e for e in os.listdir(proc)
                              if os.path.isfile(os.path.join(proc, e)) and not e.startswith("."))
        except FileNotFoundError:
            entries = []
        n_files = 0
        for name in entries:
            path = os.path.join(proc, name)
            nrows = None
            if name.lower().endswith((".csv", ".xlsx", ".xls")):
                try:
                    nrows = len(read_any(path))
                except Exception:
                    pass
            display_name = f"{name} ({run_hash})"
            fields = {
                "Имя файла": display_name,
                "Тип": os.path.splitext(name)[0],
                "Прогон": run_ref,
                "Размер, КБ": round(os.path.getsize(path) / 1024, 1),
                "Строк": nrows,
                "Путь (локальный)": path,
                "SHA-256": sha256_file(path)[:16],
                "Загружено": now.isoformat(),
            }
            if display_name in fil_index:
                rec_id = fil_index[display_name]
                tb.update(T_FIL, rec_id, fields)
            else:
                rec = tb.create(T_FIL, fields)
                rec_id = rec["id"]
                fil_index[display_name] = rec_id
            if fil_field:
                tb.upload_attachment(T_FIL, rec_id, fil_field, path)
            n_files += 1
        print(f"Files attached: {n_files}.")

    print(f"Done. Run '{run_id}' synced to Teable base {cfg['base_id']}.")


if __name__ == "__main__":
    main()
