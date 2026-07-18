"""Headless TabPFN adapter using the installed application's own model modules.

Place this file beside ``app_tabpfn.py``. The authoritative result is an explicit,
atomically written snapshot; the application's append-only log remains local audit
history and is never used to discover the result of the current run.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import tempfile
import uuid

import numpy as np
import pandas as pd

import mc_features as mcf
import tabpfn_backend as tb


APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "tabpfn_cache"
CONFIG = APP_DIR / "tabpfn_config.json"
LOG_DIR = APP_DIR / "tabpfn_logs"
LOG_PATH = LOG_DIR / "tabpfn_predictions.csv"
DEFAULT_CONTEXT = r"C:/Users/User/Desktop/IVF/AI/DNN/обучение и валидация/all_df_with_KPI.xlsx"
DEFAULT_TARGET = "Исход переноса"

LOG_COLUMNS = ["timestamp", "record_id", "clinic", "patient_name", "patient_id",
               "backend", "mode", "age", "amh", "height_cm", "weight_kg", "bmi",
               "afc", "follicles", "attempt", "okk", "mii", "pn2", "blasts", "good",
               "context_n", "n_scored", "p_tabpfn", "ci_low", "ci_high", "dt_compare"]
COLMAP = {
    "age": ["age", "Age", "Возраст"], "amh": ["amh", "AMH", "АМГ"],
    "bmi": ["bmi", "BMI", "ИМТ"], "afc": ["afc", "AFC", "АФК", "Antral follicule count"],
    "attempt": ["attempt", "Attempt", "attempt_number", "№ попытки", "№ попытки ЭКО", "# Attempt"],
    "foll": ["N folicules OPU", "follicles", "follicles_tvp", "Количество фолликулов"],
    "okk": ["okk", "OCC", "OKK", "Число ОКК"],
    # MII is optional and must never be inferred from the number inseminated.
    "mii": ["mii", "MII", "Число MII"],
    "pn2": ["pn2", "2pN", "2 pN"],
    "blasts": ["blasts_total", "Bl", "Число Bl", "blasts"],
    "good": ["blasts_good", "Good Bl", "Число Bl хор.кач-ва", "good"],
    # "_key" (ID+Date OPU composite) takes priority over raw patient_id/ID: a patient
    # with repeat cycles shares one raw ID across rows, so the caller (embryo-lab-data)
    # disambiguates cycles via "_key" and expects it echoed back as this run's identity.
    "pid": ["_key", "patient_id", "ID", "id"],
    "pname": ["Patient Full Name", "patient_name", "name"],
    "dt": ["DIGITAL TWIN", "digital_twin"],
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TabPFN IVF headless batch")
    parser.add_argument("legacy_input", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("legacy_output", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("--input", dest="input_path")
    parser.add_argument("--output", dest="output_path")
    parser.add_argument("--run-id", default=None)
    ns = parser.parse_args()
    ns.input_path = ns.input_path or ns.legacy_input
    ns.output_path = ns.output_path or ns.legacy_output
    if not ns.input_path:
        parser.error("--input is required")
    ns.run_id = ns.run_id or uuid.uuid4().hex
    if not ns.output_path:
        ns.output_path = str(LOG_DIR / f"tabpfn_predictions_{ns.run_id}.csv")
    return ns


def _cfg() -> dict:
    c = json.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() else {}
    return {"backend": c.get("backend", "local"), "device": c.get("device", "cpu"),
            "key": c.get("key", ""), "ctx": c.get("context_path", DEFAULT_CONTEXT),
            "target": c.get("target", DEFAULT_TARGET), "max_n": int(c.get("max_n", 5000)),
            "seed": int(c.get("seed", 42)), "n_sim": int(c.get("n_sim", 2000)),
            "max_query": int(c.get("max_query", 400)), "clinic": c.get("clinic", "")}


def _pick(df: pd.DataFrame, names: list[str]) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(pd.NA, index=df.index, dtype="object")


def _known_count(value: object) -> int | None:
    """Null means unknown; an observed zero remains exactly zero."""
    if pd.isna(value):
        return None
    number = float(value)
    if not np.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError(f"embryology count must be a non-negative integer, got {value!r}")
    return int(number)


def _blank_if_missing(value: object) -> object:
    return "" if value is None or pd.isna(value) else value


def _append_audit(row: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    need_header = not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0
    with LOG_PATH.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=LOG_COLUMNS)
        if need_header:
            writer.writeheader()
        writer.writerow({k: _blank_if_missing(row.get(k)) for k in LOG_COLUMNS})


def _atomic_csv(frame: pd.DataFrame, target: str | Path) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=target.suffix,
                                    dir=target.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        frame.to_csv(tmp, index=False, encoding="utf-8-sig")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


def _load(path: str) -> pd.DataFrame:
    return pd.read_csv(path) if str(path).lower().endswith(".csv") else pd.read_excel(path)


def main() -> None:
    args = _args()
    cfg = _cfg()
    frame = _load(args.input_path)
    if not os.path.isfile(cfg["ctx"]):
        raise FileNotFoundError(f"TabPFN context file not found: {cfg['ctx']}")

    patient_id = _pick(frame, COLMAP["pid"]).astype("string").str.strip()
    if patient_id.isna().any() or patient_id.eq("").any():
        raise ValueError("TabPFN input requires a non-empty patient_id/ID for every row")
    normalized = patient_id.str.upper()
    if normalized.duplicated().any():
        raise ValueError("TabPFN input patient_id/ID must be unique")

    print(f"[tabpfn] run={args.run_id} rows={len(frame)} context={cfg['ctx']}")
    tb.apply_token(cfg["key"])
    clf, info, _ = tb.build_or_load_fitted(
        cache_dir=str(CACHE_DIR), ctx_path=cfg["ctx"], feature_names=mcf.NN_FEATURE_NAMES,
        target=cfg["target"], max_n=cfg["max_n"], balanced=True, seed=cfg["seed"],
        device=cfg["device"], model_path=tb.resolve_model_path() or None,
        fit_mode="fit_with_cache", force_rebuild=False,
    )

    numeric = {k: pd.to_numeric(_pick(frame, names), errors="coerce")
               for k, names in COLMAP.items() if k not in ("pid", "pname", "dt")}
    patient_name = _pick(frame, COLMAP["pname"]).astype("string")
    dt_compare = pd.to_numeric(_pick(frame, COLMAP["dt"]), errors="coerce")
    output: list[dict] = []

    for pos in range(len(frame)):
        age_value = numeric["age"].iloc[pos]
        age = float(age_value) if pd.notna(age_value) else 34.0
        amh_value = numeric["amh"].iloc[pos]
        bmi_value = numeric["bmi"].iloc[pos]
        afc_value = numeric["afc"].iloc[pos]
        attempt_value = numeric["attempt"].iloc[pos]
        amh = float(amh_value) if pd.notna(amh_value) else 2.0
        bmi = float(bmi_value) if pd.notna(bmi_value) else 23.0
        afc = int(afc_value) if pd.notna(afc_value) else 12
        attempt = int(attempt_value) if pd.notna(attempt_value) else 1
        follicle_value = numeric["foll"].iloc[pos]
        follicles = int(follicle_value) if pd.notna(follicle_value) else afc
        known = {k: _known_count(numeric[k].iloc[pos])
                 for k in ("okk", "mii", "pn2", "blasts", "good")}

        simulated = mcf.simulate_features(
            age=age, amh=amh, bmi=bmi, afc=afc, attempt=attempt, follicles=follicles,
            known_okk=known["okk"], known_mii=known["mii"], known_pn2=known["pn2"],
            known_blasts=known["blasts"], known_good=known["good"],
            n_sim=cfg["n_sim"], seed=cfg["seed"],
        )
        result = tb.score_distribution(clf, simulated["features"],
                                       max_query=cfg["max_query"], seed=cfg["seed"])
        probability = float(result["p_median"])
        ci_low, ci_high = float(result["ci_low"]), float(result["ci_high"])
        if (not np.isfinite([probability, ci_low, ci_high]).all()
                or not (0 <= ci_low <= probability <= ci_high <= 1)):
            raise ValueError(
                f"invalid TabPFN probability/CI for {patient_id.iloc[pos]}: "
                f"p={probability}, CI=[{ci_low}, {ci_high}]"
            )
        record = {"run_id": args.run_id, "patient_id": patient_id.iloc[pos],
                  "p_tabpfn": round(probability, 4), "ci_low": round(ci_low, 4),
                  "ci_high": round(ci_high, 4)}
        output.append(record)
        _append_audit({
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"), "run_id": args.run_id,
            "record_id": str(uuid.uuid4()), "clinic": cfg["clinic"],
            "patient_name": patient_name.iloc[pos], "patient_id": patient_id.iloc[pos],
            "backend": cfg["backend"], "mode": "batch_dist", "age": age, "amh": amh,
            "bmi": bmi, "afc": afc, "follicles": follicles, "attempt": attempt,
            "okk": known["okk"], "mii": known["mii"], "pn2": known["pn2"],
            "blasts": known["blasts"], "good": known["good"], "context_n": info["n_used"],
            "n_scored": result["n_scored"], "p_tabpfn": round(probability, 4),
            "ci_low": round(ci_low, 4), "ci_high": round(ci_high, 4),
            "dt_compare": (round(float(dt_compare.iloc[pos]), 4)
                           if pd.notna(dt_compare.iloc[pos]) else None),
        })
        print(f"  [{pos + 1}/{len(frame)}] p={probability:.3f} CI [{ci_low:.3f};{ci_high:.3f}]")

    target = _atomic_csv(pd.DataFrame(output), args.output_path)
    print(f"[tabpfn] snapshot -> {target}")


if __name__ == "__main__":
    main()
