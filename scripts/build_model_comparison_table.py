"""Собрать единый CSV для сравнения прогноза стадий эмбриологического протокола
двух моделей — TabPFN-3 regression и Digital Twin (мех. MC-воронка `med_*`) — по
циклам Fertimed. Один ряд = один cycle_uid; факт и оба прогноза рядом, без
предвыбора стадий — сравнение и интерпретацию для статьи делает пользователь сам.

Источники (оба уже cycle_uid-keyed, никаких ID/ФИО не добавляется):
  dt_predictions.csv   — Digital Twin funnel (med_okk/mii/pn2/blasts/good) + факт
  regression_qc.csv    — TabPFN-3 regression P10-P90 по стадиям + факт (mii/pn2/
                          blasts_early/blasts_after_fert/good/frozen)

Запуск:
    python scripts/build_model_comparison_table.py
    python scripts/build_model_comparison_table.py --run-dir data/processed/fertimed \
        --output data/processed/fertimed/model_comparison_tabpfn3_vs_digitaltwin.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from embryo._io import atomic_write_csv  # noqa: E402

TABPFN3_QUANTILE_STAGES = {
    "mii": "MII",
    "pn2": "2PN",
    "blasts_early": "Bl_early",
    "blasts_after_fert": "Bl_after_fert",
    "good": "GoodBl",
    "frozen": "Frozen",
}


def _quantile_cols(prefix: str) -> list[str]:
    return [f"{prefix}_p10", f"{prefix}_p25", f"{prefix}_p50", f"{prefix}_p75", f"{prefix}_p90",
            f"{prefix}_interval80_width"]


def build(dt: pd.DataFrame, rq: pd.DataFrame) -> pd.DataFrame:
    merged = dt.merge(rq, on="cycle_uid", how="outer", suffixes=("_dt", "_rq"))

    out = pd.DataFrame({"cycle_uid": merged["cycle_uid"]})

    # covariates (no ID/ФИО/record_id/clinic_name/timestamp — pseudonymized per CLAUDE.md)
    covariate_cols = {
        "age": "age", "amh": "amh", "afc": "afc", "bmi": "bmi",
        "attempt_number": "attempt_number", "sperm_source": "sperm_source",
        "IVF/ICSI": "ivf_icsi", "Embryologist": "embryologist", "Doctor": "doctor",
    }
    for src, dst in covariate_cols.items():
        out[dst] = merged.get(src)

    # --- OCC: Digital Twin only, TabPFN-3 does not forecast this stage ---
    out["occ_actual"] = merged.get("OCC")
    out["dt_occ_pred"] = merged.get("med_okk")
    out["dt_occ_p025"] = merged.get("p025_okk")
    out["dt_occ_p975"] = merged.get("p975_okk")
    out["dt_occ_input_known"] = merged.get("known_okk").notna() if "known_okk" in merged else None

    # --- MII ---
    out["mii_actual"] = merged.get("mii_actual")  # strict actual (NaN when truly unmeasured)
    out["mii_actual_fr_fallback"] = merged.get("MII_dt", merged.get("MII"))  # FR MII-or-Inseminated fallback, for reference only
    out["dt_mii_pred"] = merged.get("med_mii")  # NOTE: pass-through of known_mii in this run, not a genuine forecast — see script docstring/README note
    out["dt_mii_input_known"] = merged.get("known_mii").notna() if "known_mii" in merged else None
    for c in _quantile_cols("TPFN3_mii"):
        out[c.lower().replace("tpfn3_", "tabpfn3_")] = merged.get(c)
    out["tabpfn3_mii_in_interval80"] = merged.get("mii_in_interval80")

    # --- 2PN ---
    out["pn2_actual"] = merged.get("pn2_actual")
    out["dt_pn2_pred"] = merged.get("med_pn2")
    for c in _quantile_cols("TPFN3_pn2"):
        out[c.lower().replace("tpfn3_", "tabpfn3_")] = merged.get(c)
    out["tabpfn3_pn2_in_interval80"] = merged.get("pn2_in_interval80")

    # --- Blastocysts (total / "after fertilization" — closest match to DT's single Bl stage) ---
    out["bl_actual"] = merged.get("blasts_after_fert_actual", merged.get("Bl_dt", merged.get("Bl")))
    out["dt_bl_pred"] = merged.get("med_blasts")  # NOTE: ~0 correlation with actual in the 2026-07-19 run — flagged, not silently trusted
    out["dt_bl_p025"] = merged.get("p025_blasts")
    out["dt_bl_p975"] = merged.get("p975_blasts")
    for c in _quantile_cols("TPFN3_blasts_after_fert"):
        out[c.lower().replace("tpfn3_", "tabpfn3_")] = merged.get(c)
    out["tabpfn3_blasts_after_fert_in_interval80"] = merged.get("blasts_after_fert_in_interval80")

    # --- Blastocysts, early/day-5 read — TabPFN-3 only, no DT equivalent stage ---
    out["bl_early_actual"] = merged.get("blasts_early_actual")
    for c in _quantile_cols("TPFN3_blasts_early"):
        out[c.lower().replace("tpfn3_", "tabpfn3_")] = merged.get(c)
    out["tabpfn3_blasts_early_in_interval80"] = merged.get("blasts_early_in_interval80")

    # --- Good-quality blastocysts ---
    out["good_bl_actual"] = merged.get("good_actual", merged.get("Good Bl_dt", merged.get("Good Bl")))
    out["dt_good_bl_pred"] = merged.get("med_good")  # NOTE: ~0 correlation with actual in the 2026-07-19 run — flagged, not silently trusted
    out["dt_good_bl_p025"] = merged.get("p025_good")
    out["dt_good_bl_p975"] = merged.get("p975_good")
    for c in _quantile_cols("TPFN3_good"):
        out[c.lower().replace("tpfn3_", "tabpfn3_")] = merged.get(c)
    out["tabpfn3_good_in_interval80"] = merged.get("good_in_interval80")

    # --- Frozen/cryopreserved — TabPFN-3 only; DT's med_warmed is a different concept (thaw, not freeze) ---
    out["frozen_actual"] = merged.get("frozen_actual")
    for c in _quantile_cols("TPFN3_frozen"):
        out[c.lower().replace("tpfn3_", "tabpfn3_")] = merged.get(c)
    out["tabpfn3_frozen_in_interval80"] = merged.get("frozen_in_interval80")

    return out


def _print_data_quality_flags(dt: pd.DataFrame) -> None:
    def corr(pred: str, actual: str) -> float:
        d = dt[[pred, actual]].dropna()
        return d[pred].corr(d[actual]) if len(d) > 5 else float("nan")

    print("\n=== Data-quality flags (Digital Twin med_* fields, run 2026-07-19) ===")
    print(f"- dt_mii_pred (med_mii): pass-through of known_mii input in every row "
          f"(MAE=0, r=1.00 vs the FR-fallback MII column) — NOT a genuine forecast, "
          f"exclude from any MII predictive-accuracy comparison.")
    print(f"- dt_bl_pred (med_blasts): r={corr('med_blasts', 'Bl'):.3f} vs actual Bl — "
          f"near zero, systematic +bias — verify before quoting as a working forecast.")
    print(f"- dt_good_bl_pred (med_good): r={corr('med_good', 'Good Bl'):.3f} vs actual Good Bl — "
          f"near zero, systematic +bias — verify before quoting as a working forecast.")
    print(f"- dt_occ_pred (med_okk): mixed — {dt['known_okk'].notna().sum()}/{len(dt)} rows "
          f"are pass-through (OCC already known at prediction time); remaining rows are a "
          f"genuine follicle-count-based forecast. Use dt_occ_input_known to split before pooling.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="data/processed/fertimed")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    dt = pd.read_csv(run_dir / "dt_predictions.csv")
    rq = pd.read_csv(run_dir / "regression_qc.csv")

    out = build(dt, rq)
    target = Path(args.output) if args.output else run_dir / "model_comparison_tabpfn3_vs_digitaltwin.csv"
    atomic_write_csv(out, target)

    print(f"Wrote {len(out)} rows x {len(out.columns)} cols -> {target}")
    _print_data_quality_flags(dt)


if __name__ == "__main__":
    main()
