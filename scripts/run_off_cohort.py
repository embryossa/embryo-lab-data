"""Прогноз моделей для циклов OPU-таблицы вне когорты отчёта (см. src/embryo/off_cohort.py).

Дополняет, а не заменяет scripts/run_analysis.py: берёт циклы, которых нет в
dt_predictions.csv последнего прогона, и считает для них Digital Twin 7.1, TabPFN PRAI,
TabPFN-3 regression и DNN по уровню лабораторных данных. KPI, калибровка, review-list,
MIRI/лоты/ПГТ и agent packet здесь не строятся. Все выходы — в отдельных папках, файлы
основного прогона не перезаписываются:

    data/processed/off_cohort/                dt_predictions.csv, OPU_table_enriched.xlsx,
                                              tabpfn_predictions.csv, regression_qc.csv,
                                              ozema_cycles.csv, run_manifest.json
    data/processed/local_private/off_cohort/  model_input*.xlsx, выходы DT/TabPFN-3/DNN

Пока только клиника A (Two-Line, config/): у остальных клиник свои адаптеры когорты.

    python scripts/run_off_cohort.py "data/inbox/OPU table.xlsx" --dt-check
    python scripts/run_off_cohort.py "data/inbox/OPU table.xlsx"
    python scripts/run_off_cohort.py "data/inbox/OPU table.xlsx" \
        --dt-output data/processed/local_private/off_cohort/dt71_api_output.csv
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_analysis as ra  # noqa: E402
from embryo import (_io, contracts, dt_api, ingest, kpi, off_cohort, opu_table, ozema,  # noqa: E402
                    privacy, reconcile, regression_qc, tabpfn_regression_runner, tabpfn_runner)

OUT_SUBDIR = "off_cohort"


def _dt_cfg(cfg: dict, timepoints: list[str]) -> dict:
    dt_cfg = copy.deepcopy(cfg["external_tools"]["digital_twin"])
    dt_cfg.setdefault("api", {})["timepoints"] = timepoints
    return dt_cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input")
    parser.add_argument("--dt-output", help="replay: сохранённый выход DT этого слоя")
    parser.add_argument("--tabpfn-output", help="replay: сохранённый выход TabPFN")
    parser.add_argument("--regression-output", help="replay: полный выход TabPFN-3 regression")
    parser.add_argument("--skip-dnn", action="store_true")
    parser.add_argument("--dt-check", action="store_true",
                        help="проверка пакетов службой DT без расчёта; ничего не пишет")
    args = parser.parse_args(argv)

    cfg = ra._settings(ROOT / "config")
    columns_cfg = ingest.load_config(str(ROOT / "config" / "columns.yaml"))
    input_path = ra._input_path(cfg, args.input)
    processed_main = ra._project_path(cfg["paths"]["processed_dir"])
    processed = processed_main / OUT_SUBDIR
    private = ra._project_path(cfg["paths"]["local_private_dir"]) / OUT_SUBDIR
    print(f"[input] {input_path}")

    _, normalized, _ = ra._normalise_source(input_path, columns_cfg, cfg)
    cycles = ra._add_opu_aliases(normalized)
    cycles, _ = ra._dedupe_cycle_keys(cycles, "OPU-таблица")
    secret = privacy.load_or_create_secret(ra._project_path(cfg["privacy"]["secret_file"]))
    cycles = privacy.add_cycle_uid(cycles, "_key", secret)

    main_pred = processed_main / "dt_predictions.csv"
    if not main_pred.is_file():
        raise FileNotFoundError(f"нет {main_pred}: сначала основной прогон run_analysis.py")
    reported = set(pd.read_csv(main_pred, usecols=["cycle_uid"])["cycle_uid"].astype(str))
    off = off_cohort.classify(cycles, reported)
    if off.empty:
        print("[off-cohort] все циклы таблицы уже в прогоне пайплайна — считать нечего")
        return 0
    print(f"[off-cohort] вне прогона {len(off)} из {len(cycles)} циклов; "
          f"охват: {off['cohort_scope'].value_counts().to_dict()}; "
          f"уровень моделей: {off['model_tier'].value_counts().to_dict()}")

    full = off.loc[off["model_tier"] == off_cohort.TIER_FULL]
    funnel = off.loc[off["model_tier"] == off_cohort.TIER_FUNNEL]
    none = off.loc[off["model_tier"] == off_cohort.TIER_NONE]
    # Признаки модели (русские имена, Model_KPIScore) — тем же compute_kpi, что в основном
    # прогоне; фильтр 2PN!=0 внутри него у уровня full ничего не снимает.
    full_src = kpi.compute_kpi(full) if not full.empty else full
    if len(full_src) != len(full):
        raise ValueError("compute_kpi снял строки уровня full — классификация разошлась с фильтром 2PN")

    dt_full_cfg = _dt_cfg(cfg, ["transfer", "retrieval"])
    dt_funnel_cfg = _dt_cfg(cfg, ["retrieval"])
    if args.dt_check:
        report = {}
        if not full_src.empty:
            report["full"] = dt_api.check_digital_twin_api(full_src, dt_full_cfg)
        if not funnel.empty:
            report["funnel"] = dt_api.check_digital_twin_api(funnel, dt_funnel_cfg)
        for tier, summary in report.items():
            print(f"[dt-check:{tier}] циклов {summary['cycles']}, расчётов {summary['calculations']} "
                  f"(~{summary['estimated_hours']} ч), отклонено {len(summary['invalid'])}, "
                  f"покрытие {summary['dt_coverage']}")
            for item in summary["invalid"]:
                print(f"    {item['cycle_uid']} [{item['timepoint']}]: {item['error']}")
        return 0

    for directory in (processed, private):
        directory.mkdir(parents=True, exist_ok=True)
    model_input = ra._add_dt_aliases(full_src)
    model_input_path = Path(_io.atomic_write_excel({"model_input": model_input},
                                                   private / "model_input.xlsx"))

    # ---- Digital Twin 7.1 --------------------------------------------------------------
    if args.dt_output:
        dt = ingest.load_table(str(ra._project_path(args.dt_output)))
        contracts.DT_OUTPUT.validate(dt)
    else:
        parts = []
        if not full_src.empty:
            parts.append(dt_api.run_digital_twin_api(full_src, dt_full_cfg))
        if not funnel.empty:
            parts.append(off_cohort.strip_to_funnel(dt_api.run_digital_twin_api(funnel, dt_funnel_cfg)))
        if not none.empty:
            parts.append(off_cohort.empty_dt_rows(none))
        dt = pd.concat(parts, ignore_index=True, sort=False)
        contracts.DT_OUTPUT.validate(dt)
        print(f"[dt-api] результат сохранён для replay: "
              f"{_io.atomic_write_csv(dt, private / 'dt71_api_output.csv')}")
    dt = dt.loc[dt["cycle_uid"].isin(set(off["cycle_uid"]))].copy()

    # ---- TabPFN PRAI (только full) -----------------------------------------------------
    tab = None
    if not full_src.empty:
        dt_for_tab = dt.loc[dt["cycle_uid"].isin(set(full_src["cycle_uid"]))]
        tab_input, n_fresh = ra._tabpfn_input_with_fresh_dt(model_input, dt_for_tab, private)
        print(f"[tabpfn] опорный DIGITAL TWIN этого слоя: {n_fresh}/{len(model_input)}")
        if args.tabpfn_output:
            tab = tabpfn_runner.read_tabpfn(explicit_path=ra._project_path(args.tabpfn_output))
        else:
            tcfg = dict(cfg["external_tools"]["tabpfn"], output=str(private / "tabpfn_predictions.csv"))
            tab = tabpfn_runner.launch_and_read(tcfg, str(tab_input))
        _io.atomic_write_csv(tab, processed / "tabpfn_predictions.csv")

    opu = ra._prepare_opu(cfg, input_path, columns_cfg)
    opu = opu.loc[opu["_key"].isin(set(off["_key"]))].copy()
    predictions = reconcile.build_dt_predictions(
        dt, tab, opu, secret, threshold_pp=cfg["thresholds"]["tabpfn_ci_width_pp"])

    # ---- TabPFN-3 regression (только full, мягкий сбой) ----------------------------------
    regression_status, regression_frame, clean_regression = "not_applicable", pd.DataFrame(), None
    rcfg = dict(cfg["external_tools"].get("tabpfn_regression", {}),
                output=str(private / "tabpfn_regression_full.csv"))
    if rcfg.get("enabled") and not full_src.empty:
        try:
            if args.regression_output:
                raw = pd.read_csv(ra._project_path(args.regression_output), encoding="utf-8-sig")
                contracts.validate_regression_quantiles(raw)
            else:
                raw = tabpfn_regression_runner.run_tabpfn_regression(rcfg, str(model_input_path))
            clean_regression, stages = tabpfn_regression_runner.extract_predictions(raw)
            regression_frame, _ = regression_qc.build_regression_qc(full_src, clean_regression, stages)
            regression_status = "ok"
        except Exception as exc:  # noqa: BLE001 - QA-слой не блокирует прогноз
            regression_status = f"failed: {type(exc).__name__}: {exc}"
            print(f"[warn] TabPFN-3 regression skipped: {regression_status}", file=sys.stderr)
    if not regression_frame.empty:
        sheet = regression_frame.copy()
        sheet["out_of_interval"] = sheet["out_of_interval"].map(
            lambda s: ", ".join(s) if isinstance(s, list) else s)
        _io.atomic_write_csv(sheet, processed / "regression_qc.csv")

    # ---- DNN (только full, мягкий сбой) ----------------------------------------------------
    dnn_status = "disabled"
    dcfg = dict(cfg["external_tools"].get("dnn_verifier", {}), output=str(private / "evo_predictions.csv"))
    if dcfg.get("enabled") and not args.skip_dnn and not full_src.empty:
        try:
            dnn_input, _ = ra._dnn_input_with_bl_imputation(model_input_path, clean_regression, dcfg, private)
            dnn_input, n_et_zero = off_cohort.dnn_input_with_zero_transfer(dnn_input, private)
            print(f"[dnn] 'Перенесено эмбрионов' = 0 при пустом ET: {n_et_zero} циклов (только вход DNN)")
            from embryo import dnn_runner
            predictions = ra._join_dnn(predictions, dnn_runner.run_dnn_verifier(dcfg, str(dnn_input)))
            dnn_status = "valid_saved_scaler"
        except Exception as exc:  # noqa: BLE001 - кросс-проверка не блокирует прогноз
            dnn_status = f"failed: {type(exc).__name__}: {exc}"
            print(f"[warn] DNN skipped: {dnn_status}", file=sys.stderr)

    predictions = off_cohort.restore_facts(predictions, off)
    enriched = opu_table.ensure_opu_schema(reconcile.enrich_opu_table(opu, predictions))
    enriched["cohort_scope_label"] = enriched["_key"].map(off.set_index("_key")["cohort_scope_label"])
    outputs = reconcile.write_reconciled_outputs(
        predictions, enriched, processed / "dt_predictions.csv",
        processed / "OPU_table_enriched.xlsx", private / "cycle_uid_crosswalk.csv")

    # OZEMA — на всей таблице (как широкая lab-популяция основного прогона), срез по слою.
    ozema_all = ozema.build_ozema_table(cycles, columns_cfg)
    if "cycle_uid" in ozema_all.columns:
        _io.atomic_write_csv(ozema_all.loc[ozema_all["cycle_uid"].isin(set(off["cycle_uid"]))],
                             processed / "ozema_cycles.csv")

    dt_summary = dt_api.run_summary(dt) if "dt_api_status" in dt.columns else {}
    dt_summary["source"] = "IVF Digital Twin 7.1 REST API (analytics connection)"
    manifest = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "off_cohort",
        "note": "Циклы вне когорты отчёта: не входят в KPI, калибровку, review-list и agent packet.",
        "input_sha256": ra._sha256(input_path),
        "main_run_predictions_sha256": ra._sha256(main_pred),
        "n_cycles": int(len(off)),
        "cohort_scope": {k: int(v) for k, v in off["cohort_scope"].value_counts().items()},
        "model_tier": {k: int(v) for k, v in off["model_tier"].value_counts().items()},
        "digital_twin": dt_summary,
        "tabpfn_scored": int(pd.to_numeric(predictions["PRAI"], errors="coerce").notna().sum()),
        "regression_status": regression_status,
        "dnn_status": dnn_status,
        "artifacts": {Path(p).name: str(p) for p in outputs.values()},
    }
    _io.atomic_write_text(json.dumps(manifest, ensure_ascii=False, indent=2), processed / "run_manifest.json")
    print(f"[done] off-cohort: {len(off)} циклов -> {processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
