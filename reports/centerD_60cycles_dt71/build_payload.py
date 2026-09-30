#!/usr/bin/env python3
"""Step 1 of the separate DT 7.1 run on the 60-cycle center D (Two-Line) cohort.

Source: data/inbox/publication_60/OPU table for publication.xlsx (Google Drive
"OPU table for publication.xlsx", sha256 6192d779...). The cohort is built with the
unchanged embryo-lab-data code path of run_analysis.py (clinic A config = Two-Line):
normalize -> _eligible_cycles (ET>0 and known Preg; KPI filter 2PN!=0) -> OPU aliases
-> dedupe -> HMAC cycle_uid -> dt_api.build_batches (transfer + retrieval views,
n_sim 2000, documented imputations flagged).

Writes (no identifiers): api_payload.json, imputation_flags.csv, outcomes.csv.
The cycle_uid <-> _key crosswalk stays in data/processed/local_private/centerD_60cycles_dt71/.
Nothing in data/processed of the main pipeline is read or overwritten.
"""
import json, sys, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_analysis as ra                      # noqa: E402
from embryo import dt_api, ingest, privacy     # noqa: E402

SRC = ROOT / "data/inbox/publication_60/OPU table for publication.xlsx"
cfg = ra._settings(ROOT / "config")
columns_cfg = ingest.load_config(str(ROOT / "config" / "columns.yaml"))
raw, normalized, unknown = ra._normalise_source(SRC, columns_cfg, cfg)
n_known = int(normalized["Preg"].notna().sum())
elig = ra._eligible_cycles(normalized)
elig = ra._add_opu_aliases(elig)
elig, _ = ra._dedupe_cycle_keys(elig, "centerD_60")
secret = privacy.load_or_create_secret(ra._project_path(cfg["privacy"]["secret_file"]))
elig = privacy.add_cycle_uid(elig, "_key", secret)
fb = ra._model_fallback_map(cfg)
if fb:
    raise SystemExit("clinic has model fallbacks; not expected for clinic A")
api_cfg = dt_api.api_settings(cfg["external_tools"]["digital_twin"])
bodies, flags = dt_api.build_batches(elig, api_cfg)

(OUT / "api_payload.json").write_text(json.dumps(bodies, ensure_ascii=False, indent=1), encoding="utf-8")
flags.to_csv(OUT / "imputation_flags.csv", index=False)
keep = ["cycle_uid", "Preg", "transferred", "transfer_day", "age", "attempt_no", "amh", "afc", "bmi",
        "follicles", "occ", "mii", "inseminated", "two_pn", "cleavage_d3", "bl_total", "bl_good", "frozen",
        "sperm_source", "fert_method", "amh_imputed", "bmi_imputed"]
elig[[c for c in keep if c in elig.columns]].to_csv(OUT / "outcomes.csv", index=False)
priv = ROOT / "data/processed/local_private/centerD_60cycles_dt71"
priv.mkdir(parents=True, exist_ok=True)
elig[["_key", "cycle_uid"]].to_csv(priv / "crosswalk.csv", index=False)
src_sha = hashlib.sha256(SRC.read_bytes()).hexdigest()
json.dump({"source": SRC.name, "source_sha256": src_sha, "rows_source": int(len(raw)),
           "known_outcome": n_known, "eligible_cycles": int(len(elig)),
           "calculations": sum(len(b["cycles"]) for b in bodies), "api_options": bodies[0]["options"],
           "unknown_columns": list(map(str, unknown))},
          open(OUT / "payload_manifest.json", "w"), ensure_ascii=False, indent=2)
print(json.load(open(OUT / "payload_manifest.json")))
