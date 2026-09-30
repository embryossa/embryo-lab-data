#!/usr/bin/env python3
"""Publication run: IVF Digital Twin 7.1 (REST API) on "OPU table for publication.xlsx".

DT only — no TabPFN, no TabPFN-3 regression, no DNN, no PGT, no MIRI/lots, no agents.
The cohort is built with the unchanged embryo-lab-data code path of scripts/run_analysis.py
(clinic A config): normalize -> _eligible_cycles (ET>0, known Preg, 2PN!=0) -> OPU aliases
-> dedupe -> HMAC cycle_uid; DT via dt_api.run_digital_twin_api (transfer + retrieval views,
retrieval fallback). data/processed of the main pipeline is neither read nor written.

  python run_dt.py --check   # service-side validation only, nothing computed
  python run_dt.py           # full run

Identifiers (_key, patient_name) are written ONLY to
data/processed/local_private/publication_60_dt71/; this folder receives cycle_uid only.
"""
import argparse, hashlib, json, sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PRIV = ROOT / "data/processed/local_private/publication_60_dt71"
SRC = ROOT / "data/inbox/publication_60/OPU table for publication.xlsx"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_analysis as ra                               # noqa: E402
from embryo import _io, contracts, dt_api, ingest, privacy  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--check", action="store_true")
args = p.parse_args()
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

cfg = ra._settings(ROOT / "config")
columns_cfg = ingest.load_config(str(ROOT / "config" / "columns.yaml"))
raw, normalized, unknown = ra._normalise_source(SRC, columns_cfg, cfg)
cohort = ra._eligible_cycles(normalized)
cohort = ra._add_opu_aliases(cohort)
cohort, _ = ra._dedupe_cycle_keys(cohort, "publication_60")
secret = privacy.load_or_create_secret(ra._project_path(cfg["privacy"]["secret_file"]))
cohort = privacy.add_cycle_uid(cohort, "_key", secret)
if ra._model_fallback_map(cfg):
    raise SystemExit("unexpected model fallbacks in clinic A config")
dt_cfg = cfg["external_tools"]["digital_twin"]
print(f"[cohort] source rows {len(raw)}, known outcome {int(normalized['Preg'].notna().sum())}, "
      f"eligible cycles {len(cohort)}")

if args.check:
    print(json.dumps(dt_api.check_digital_twin_api(cohort, dt_cfg), ensure_ascii=False, indent=2, default=str))
    sys.exit(0)

PRIV.mkdir(parents=True, exist_ok=True)
started = datetime.now().isoformat(timespec="seconds")
frame = dt_api.run_digital_twin_api(cohort, dt_cfg)
contracts.DT_OUTPUT.validate(frame)
_io.atomic_write_csv(frame, PRIV / "dt71_api_output.csv")          # replay file, has _key
_io.atomic_write_csv(cohort[["_key", "cycle_uid"]], PRIV / "crosswalk.csv")

keep_obs = ["cycle_uid", "Preg", "transferred", "transfer_day", "age", "attempt_no", "amh", "afc", "bmi",
            "follicles", "occ", "mii", "inseminated", "two_pn", "cleavage_d3", "bl_total", "bl_good", "frozen",
            "sperm_source", "fert_method"]
obs = cohort[[c for c in keep_obs if c in cohort.columns]]
drop = {"_key", "patient_name", "Patient Full Name", "patient_id", "ФИО"}
dt_pub = frame[[c for c in frame.columns if c not in drop and not c.startswith("_")]]
dt_pub = dt_pub.drop(columns=[c for c in dt_pub.columns if c in obs.columns and c != "cycle_uid"])
_io.atomic_write_csv(obs.merge(dt_pub, on="cycle_uid", how="left", validate="one_to_one"),
                     OUT / "dt71_predictions.csv")
manifest = {"source": SRC.name, "source_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
            "started": started, "finished": datetime.now().isoformat(timespec="seconds"),
            "rows_source": int(len(raw)), "known_outcome": int(normalized["Preg"].notna().sum()),
            "eligible_cycles": int(len(cohort)), "api": {k: v for k, v in dt_api.api_settings(dt_cfg).items()
                                                         if k not in ("key_file", "ca_cert")},
            "dt": dt_api.run_summary(frame), "unknown_columns": list(map(str, unknown)),
            "not_run": ["TabPFN", "TabPFN-3 regression", "DNN", "PGT-A", "MIRI", "consumable lots", "agents"]}
(OUT / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print(json.dumps(manifest["dt"], ensure_ascii=False, indent=2))
