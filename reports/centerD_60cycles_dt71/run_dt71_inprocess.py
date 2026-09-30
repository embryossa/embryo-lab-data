#!/usr/bin/env python3
"""Step 2 of the separate DT 7.1 run on the 60-cycle center D (Two-Line) cohort.

Scores api_payload.json (step 1, build_payload.py) with the unchanged IVF Digital Twin 7.1
analytics code path of integration/analytics_worker.py:
    analytics.validate_cycle -> seed = default_seed(cycle_id)
    -> desktop.service.Service.compute(patient, cycle_id, adapt=False, retain=False)
    -> desktop.history_export.build_row
with the batch options pinned as the worker pins them (no clinic prior, centre calibration
and reliability overrides removed). Only the REST transport and the SQLite/DPAPI queue are
not used. The licence is the clinic's own: license.lic is copied from the desktop data folder
into the private work folder; nothing is bypassed. The output has the structure of
GET /analytics/batches/{id}/results, so embryo_lab_data dt_api.to_dt_output maps it unchanged.
Input and output contain only cycle_uid pseudonyms and numeric features.

Run with the Digital Twin 7.1 runtime (Windows):
  "<IVF_Clinic_Desktop>\\runtime\\python.exe" run_dt71_inprocess.py "<IVF_Clinic_Desktop>"
Optional: a number as the 2nd argument limits the run to that many calculations (smoke test).
"""
import contextlib, io, json, os, shutil, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(sys.argv[1]).resolve()
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None
PAYLOAD = HERE / "api_payload.json"
OUTFILE = HERE / ("dt71_results.json" if LIMIT is None else f"_smoke_{LIMIT}.json")
WORK = HERE.parents[1] / "data" / "processed" / "local_private" / "centerD_60cycles_dt71" / "dt71_work"
WORK.mkdir(parents=True, exist_ok=True)

lic = WORK / "license.lic"
if not lic.exists():
    src = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "IVF Clinic Desktop" / "license.lic"
    if not src.exists():
        sys.exit(f"no licence found at {src}; activate IVF Digital Twin 7.1 on this PC first")
    shutil.copyfile(src, lic)

sys.path[:0] = [str(ROOT), str(ROOT / "src")]
os.chdir(ROOT)
os.environ["IVF_DATA_DIR"] = str(WORK)
from integration import analytics                          # noqa: E402
from desktop.service import Service                        # noqa: E402
from desktop.calibration import model_fingerprint          # noqa: E402
from desktop.history_export import build_row, COLUMNS      # noqa: E402

service = Service(WORK)
if not service.license()["valid"]:
    sys.exit("the IVF Digital Twin 7.1 licence is not valid on this PC")
bodies = json.loads(PAYLOAD.read_text(encoding="utf-8"))
out, t0, n = [], time.time(), 0
for b_i, body in enumerate(bodies):
    options = analytics._options(body["options"])
    prior = options.get("clinic_prior")
    (WORK / "clinic-settings.json").write_text(json.dumps({"enabled": bool(prior), "batches": prior or []}))
    for name in ("center-calibration.json", "reliability.json"):
        with contextlib.suppress(FileNotFoundError):
            (WORK / name).unlink()
    fp = model_fingerprint(WORK)
    rows = []
    for item in body["cycles"]:
        if LIMIT is not None and n >= LIMIT:
            break
        cycle_id = item["cycle_id"]
        seed = item.get("seed", analytics.default_seed(cycle_id))
        patient, error = analytics.validate_cycle(item["patient"])
        rec = {"cycle_id": cycle_id, "seed": seed, "model_fingerprint": None, "values": None, "error": None}
        if patient is None:
            rec.update(status="invalid", error=error)
        else:
            patient.update(n_sim=options["n_sim"], seed=seed)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = service.compute(patient, cycle_id, adapt=False, retain=False)
                values = build_row(result)
                values["record_id"] = None
                rec.update(status="complete", values=values, model_fingerprint=fp)
            except Exception as exc:
                rec.update(status="failed", error={"code": "calculation_failed", "fields": [type(exc).__name__]})
                traceback.print_exc(limit=3)
        rows.append(rec)
        n += 1
        print(f"{n}  {cycle_id.rsplit('.', 1)[-1]:9s} {rec['status']}  {time.time() - t0:.0f}s", flush=True)
    out.append({"id": f"inprocess-{b_i}", "state": "complete", "options": options, "model_fingerprint": fp,
                "scoring": "automatic_research", "clinician_reviewed": False, "columns": COLUMNS, "rows": rows})
service.pool.shutdown(wait=False); service.llm_pool.shutdown(wait=False)
OUTFILE.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
print("written", OUTFILE, f"{time.time() - t0:.0f}s")
