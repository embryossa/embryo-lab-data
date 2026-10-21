"""persona_metrics_report.py — сводка метрик прогонов PERSONA по всем годам с прогоном (markdown для статьи).

Только агрегаты: run_manifest.json, agent_packets/analysis_bundle.json (KPI) и
dt_predictions.csv (вероятности моделей + исход). Все числа считает этот скрипт.

Дискриминация и калибровка — на общей выборке циклов, где есть оценка всех моделей и
исход (клиническая беременность по УЗИ, включая внематочную). 95% ДИ AUC и ΔAUC —
бутстреп по циклам (2000 повторов, seed 20261020), ΔAUC — парный.
Дизайн (решение владельца 2026-10-21): годы до persona.prospective_from_year в
settings.yaml клиники — ретроспективная когорта, с него — проспективная; pooled
считается внутри каждой когорты и никогда через их границу.
Laya с калибровкой центра оценивается вне выборки: в ретроспективной когорте сдвиг
интерсепта logit подбирается на других ретроспективных годах (leave-one-year-out), как в
Laya/scripts/42_centre_persona.py; в проспективной — на всей ретроспективной когорте.

    python scripts/persona_metrics_report.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from embryo.clinics.persona import run_years  # noqa: E402

BASE = ROOT / "data" / "processed" / "persona"
YEARS = tuple(run_years(BASE))
SPAN = f"{YEARS[0]}–{YEARS[-1]}" if YEARS else "—"
with (ROOT / "config" / "clinics" / "persona" / "settings.yaml").open(encoding="utf-8") as _s:
    PROSPECTIVE_FROM = int((yaml.safe_load(_s).get("persona") or {}).get("prospective_from_year", 9999))
RETRO = tuple(y for y in YEARS if int(y) < PROSPECTIVE_FROM)
PROS = tuple(y for y in YEARS if int(y) >= PROSPECTIVE_FROM)


def _span(years) -> str:
    return f"{years[0]}–{years[-1]}" if len(years) > 1 else years[0]


def cohort(year: str) -> str:
    return "prospective" if year in PROS else "retrospective"


# Строки таблиц: годы ретроспективы, её pooled, годы проспективы, её pooled (pooled —
# только при ≥2 годах когорты). Label -> годы.
ROWS: list[tuple[str, tuple[str, ...]]] = []
for _years, _name in ((RETRO, "retrospective"), (PROS, "prospective")):
    ROWS += [(f"{y} ({_name})", (y,)) for y in _years]
    if len(_years) > 1:
        ROWS.append((f"{_span(_years)} {_name}, pooled", _years))
EPS = 1e-6
N_BOOT = 2000
MODELS = {  # подпись -> колонка dt_predictions
    "DT 7.1 clinical probability": "DIGITAL TWIN",
    "DT 7.1 BEFE (L7)": "befe_per_transfer",
    "TabPFN (PRAI)": "PRAI",
    "DNN (EVO)": "DNN",
    "Laya (raw)": "laya_raw",
}
REF = "DT 7.1 clinical probability"
KPI_ORDER = ("FR", "ClR", "BlR", "TGBDR", "DegR", "MIIR", "OCCR", "3pNR")


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-x))


def intercept(x, y) -> float:
    lo, hi = -10.0, 10.0
    for _ in range(80):
        a = (lo + hi) / 2
        lo, hi = (a, hi) if (y - expit(a + x)).sum() > 0 else (lo, a)
    return (lo + hi) / 2


def slope(p, y) -> float:
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0, 0])


def load() -> pd.DataFrame:
    frames = []
    for y in YEARS:
        d = pd.read_csv(BASE / y / "dt_predictions.csv", low_memory=False)
        keep = ["Preg", "Bl", "clinical_probability_kind", *MODELS.values()]
        frames.append(d[keep].assign(year=y))
    d = pd.concat(frames, ignore_index=True)
    for c in ["Preg", "Bl", *MODELS.values()]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["Preg", *MODELS.values()]).reset_index(drop=True)
    # Laya с калибровкой PERSONA вне выборки: ретроспектива — сдвиг с других
    # ретроспективных лет; проспектива — со всей ретроспективной когорты.
    d["laya_oos"] = np.nan
    for y in YEARS:
        tr = d["year"].isin(RETRO) & d["year"].ne(y)
        if not tr.any():
            continue
        a = intercept(logit(d.loc[tr, "laya_raw"]), d.loc[tr, "Preg"].to_numpy())
        test = d["year"].eq(y)
        d.loc[test, "laya_oos"] = expit(logit(d.loc[test, "laya_raw"]) + a)
    return d


SUBSETS = ("All fresh transfers", "Transfers with ≥1 blastocyst (Bl > 0)")


def subsets(d: pd.DataFrame, name: str):
    mask = d["Bl"].gt(0) if name == SUBSETS[1] else pd.Series(True, index=d.index)
    for label, years in ROWS:
        yield label, d[mask & d["year"].isin(years)]


def boot_idx(n: int, rng):
    return [rng.integers(0, n, n) for _ in range(N_BOOT)]


def auc_ci(y, p, idx):
    vals = [roc_auc_score(y[i], p[i]) for i in idx if y[i].min() != y[i].max()]
    return np.percentile(vals, [2.5, 97.5])


def fmt_ci(v, lo, hi, nd=3):
    return f"{v:.{nd}f} [{lo:.{nd}f}–{hi:.{nd}f}]"


def discrimination(d: pd.DataFrame) -> list[str]:
    out = []
    rng = np.random.default_rng(20261020)
    for name in SUBSETS:
        out += [f"### {name}", "",
                "| Year | n (events) | " + " | ".join(MODELS) + " |",
                "|---|---|" + "---|" * len(MODELS)]
        delta = [f"#### Paired ΔAUC vs {REF} — {name}", "",
                 "| Year | " + " | ".join(m for m in MODELS if m != REF) + " |",
                 "|---|" + "---|" * (len(MODELS) - 1)]
        for y, s in subsets(d, name):
            yy = s["Preg"].to_numpy().astype(int)
            idx = boot_idx(len(s), rng)
            cells, dcells = [], []
            ref_p = s[MODELS[REF]].to_numpy()
            for m, col in MODELS.items():
                p = s[col].to_numpy()
                cells.append(fmt_ci(roc_auc_score(yy, p), *auc_ci(yy, p, idx)))
                if m != REF:
                    diffs = [roc_auc_score(yy[i], p[i]) - roc_auc_score(yy[i], ref_p[i])
                             for i in idx if yy[i].min() != yy[i].max()]
                    dv = roc_auc_score(yy, p) - roc_auc_score(yy, ref_p)
                    lo, hi = np.percentile(diffs, [2.5, 97.5])
                    dcells.append(f"{dv:+.3f} [{lo:+.3f}; {hi:+.3f}]")
            out.append(f"| {y} | {len(s)} ({yy.sum()}) | " + " | ".join(cells) + " |")
            delta.append(f"| {y} | " + " | ".join(dcells) + " |")
        out += [""] + delta + [""]
    return out


def calibration(d: pd.DataFrame) -> list[str]:
    cols = {**MODELS, "Laya (PERSONA-calibrated, out of sample)": "laya_oos"}
    out = []
    for name in SUBSETS:
        out += [f"### {name}", "",
                "| Year | Model | n | Observed | Mean predicted | O/E | Brier | Calibration slope |",
                "|---|---|---|---|---|---|---|---|"]
        for y, s in subsets(d, name):
            yy = s["Preg"].to_numpy().astype(int)
            for m, col in cols.items():
                p = s[col].to_numpy()
                if np.isnan(p).any():
                    continue
                out.append(f"| {y} | {m} | {len(s)} | {yy.mean():.3f} | {p.mean():.3f} | "
                           f"{yy.mean() / p.mean():.2f} | {brier_score_loss(yy, p):.4f} | {slope(p, yy):.2f} |")
        out.append("")
    return out


def runs_section() -> list[str]:
    out = ["## 1. Runs and components", "",
           "| Year | Cohort | Lab population (method blocks) | DT cohort (fresh transfers, known outcome) | "
           "Pregnancies | DT 7.1: transfer / retrieval fallback / none | TabPFN | TabPFN-3 regression | "
           "DNN scored | Laya scored | Laya high disagreement | Review-list |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    kpi_rows = {}
    for y in YEARS:
        m = json.loads((BASE / y / "run_manifest.json").read_text(encoding="utf-8"))
        b = json.loads((BASE / y / "agent_packets" / "analysis_bundle.json").read_text(encoding="utf-8"))
        d = pd.read_csv(BASE / y / "dt_predictions.csv", low_memory=False)
        rv = pd.read_csv(BASE / y / "review_list.csv", low_memory=False)
        tp = m["digital_twin"]["timepoint_used"]
        preg = int(pd.to_numeric(d["Preg"], errors="coerce").sum())
        dnn = (m.get("dnn_coverage") or {}).get("scored", 0)
        lc = m.get("laya_coverage") or {}
        out.append(f"| {y} | {cohort(y)} | {m['n_lab_population']} | {m['n_cycles']} | {preg} | "
                   f"{tp.get('transfer', 0)} / {tp.get('retrieval_fallback', 0)} / {tp.get('none', 0)} | "
                   f"{int(pd.to_numeric(d['PRAI'], errors='coerce').notna().sum())} | {m['regression_status']} | "
                   f"{dnn} | {lc.get('scored', 0)} | {lc.get('disagreement_high', 0)} | {len(rv)} |")
        kpi_rows[y] = {k["metric"]: k for k in b["cohort"]["kpi_aggregate"]}
    out += ["", "## 2. Vienna laboratory KPIs (pooled Σ numerator / Σ denominator, Wilson 95% CI)", "",
            "Lab population: all method blocks with laboratory work, freeze-all included (clinic decision). "
            "OCCR: follicle counts (total follicles on trigger day) are recorded only from the 2026 export and only "
            "in a minority of retrievals, so OCCR is shown where follicles are known; 3pNR not mapped.", "",
            "| KPI | Competency | Benchmark | " + " | ".join(YEARS) + " |",
            "|---|---|---|" + "---|" * len(YEARS)]
    for k in KPI_ORDER:
        first = next((kpi_rows[y][k] for y in YEARS if k in kpi_rows[y]), None)
        if not first:
            continue
        cells = []
        for y in YEARS:
            r = kpi_rows[y].get(k) or {}
            if not r.get("denominator"):
                cells.append("n/a")
                continue
            cells.append(f"{100 * r['rate']:.1f}% ({int(r['numerator'])}/{int(r['denominator'])}; "
                         f"{100 * r['ci_low_95']:.1f}–{100 * r['ci_high_95']:.1f})")
        comp = f"{'≥' if first['direction'] == 'lower' else '≤'}{100 * first['competency']:.0f}%" \
            if first.get("competency") is not None else "—"
        bench = f"{'≥' if first['direction'] == 'lower' else '≤'}{100 * first['benchmark']:.0f}%" \
            if first.get("benchmark") is not None else "—"
        out.append(f"| {k} | {comp} | {bench} | " + " | ".join(cells) + " |")
    return out + [""]


def laya_centre_section() -> list[str]:
    path = ROOT.parent / "Laya" / "reports" / "persona_centre_calibration.csv"
    out = ["## 5. Laya centre calibration for PERSONA", "",
           "Intercept-only shift on the logit scale (slope fixed at 1), the same method as the four RBMO centres. "
           "Laya: all Laya-assessable fresh transfers; DT: `per_transfer` cycles only (DT returns a current-cycle "
           "probability for transfers without blastocysts). Validation: leave-one-year-out (shift fitted on the two "
           "other years).", ""]
    if not path.is_file():
        return out + ["_reports/persona_centre_calibration.csv not found — run Laya/scripts/42_centre_persona.py._", ""]
    c = pd.read_csv(path)
    out += ["| Model | Test year | Fitted on | n (events) | Shift | Mean pred raw → calibrated | Observed | "
            "O/E raw → calibrated | Brier raw → calibrated |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in c.iterrows():
        out.append(f"| {r['model']} | {r['test']} | {r['fit_on']} | {r['n']} ({r['events']}) | {r['shift']:+.4f} | "
                   f"{r['raw_mean_pred']:.3f} → {r['cal_mean_pred']:.3f} | {r['raw_observed']:.3f} | "
                   f"{r['raw_O/E']:.2f} → {r['cal_O/E']:.2f} | {r['raw_brier']:.4f} → {r['cal_brier']:.4f} |")
    return out + [""]


def design_section(d: pd.DataFrame) -> list[str]:
    """Дизайн для статьи: только факты из манифестов и выходов прогонов."""
    manifests = {y: json.loads((BASE / y / "run_manifest.json").read_text(encoding="utf-8")) for y in YEARS}
    prints = {y: set(m["digital_twin"].get("model_fingerprints") or []) for y, m in manifests.items()}
    retro_prints = set().union(*(prints[y] for y in RETRO)) if RETRO else set()
    out = ["## Study design", ""]
    if RETRO:
        out.append(f"- **Retrospective cohort:** {_span(RETRO)} — fresh transfers with a known outcome "
                   f"(n = {int(d['year'].isin(RETRO).sum())} in the common sample).")
    if not PROS:
        return out + ["- No prospective year has a finished run yet.", ""]
    span = _span(PROS)
    dates = []
    for y in PROS:
        p = pd.read_csv(BASE / y / "dt_predictions.csv", low_memory=False)
        col = next((c for c in ("Date OPU", "date_opu") if c in p.columns), None)
        if col:
            dates.append(pd.to_datetime(p[col], errors="coerce"))
    rng = pd.concat(dates).agg(["min", "max"]) if dates else None
    period = (f"retrievals {rng['min']:%Y-%m-%d} to {rng['max']:%Y-%m-%d}" if rng is not None
              and rng.notna().all() else f"year {span}")
    same_dt = all(prints[y] and prints[y] <= retro_prints for y in PROS)
    out += [f"- **Prospective cohort:** {span} ({period}; n = {int(d['year'].isin(PROS).sum())} fresh transfers "
            "with a known outcome in the common sample). It is analysed as a prospective study and reported "
            "separately: no metric pools prospective and retrospective cycles.",
            "- Nothing used for prediction was fitted on prospective data: the Laya PERSONA centre shift was fitted "
            f"on {_span(RETRO) if RETRO else 'earlier years'} (out-of-sample evaluation here uses the shift fitted on "
            "the whole retrospective cohort); the AFC-from-OCC coefficient (1.16) was fitted on another clinic "
            "(Fertimed); the DT 7.1 model fingerprint in the prospective run "
            + ("is identical to the retrospective runs." if same_dt else "DIFFERS from the retrospective runs — check before publication."),
            "- Outcomes are used only for evaluation. Follow-up is incomplete for the most recent retrievals: transfers "
            "without a recorded ultrasound/hCG result are not in the cohort (see the staging summary of the run).",
            "- The prospective export adds observed AFC and total follicles on trigger day for a minority of cycles "
            "(section 6); where present, observed AFC replaces the OCC-based estimate in the DT input.", ""]
    return out


def main() -> None:
    d = load()
    n_bl0 = int(d["Bl"].eq(0).sum())
    ciw = pd.concat([pd.to_numeric(pd.read_csv(BASE / y / "dt_predictions.csv", low_memory=False)["ci_width_pp"],
                                   errors="coerce") for y in YEARS])
    ci_small, ci_n, ci_max = int(ciw.lt(0.1).sum()), int(ciw.notna().sum()), float(ciw.max())
    lines = [f"# PERSONA {SPAN}: run metrics and model discrimination", "",
             f"_Generated {date.today().isoformat()} by `embryo-lab-data/scripts/persona_metrics_report.py` "
             "from run outputs (aggregates only; no identifiers)._", "",
             "Outcome: clinical pregnancy confirmed by ultrasound (including ectopic), per fresh transfer; "
             "not live birth. Cryo-transfers are not linked to retrievals and are not included.", ""]
    lines += design_section(d)
    lines += runs_section()
    lines += ["## 3. Discrimination (AUC-ROC, 95% bootstrap CI)", "",
              f"Common sample: fresh transfers with a known outcome and an estimate from every model "
              f"(n = {len(d)}; {n_bl0} transfers without blastocysts). Bootstrap over cycles, {N_BOOT} resamples; "
              "ΔAUC is paired (same resamples).", "",
              "Reading note: for transfers without blastocysts DT 7.1 returns the probability of the current "
              "cycle (`clinical_probability_kind = current_cycle`, value 0), not a per-transfer probability; the "
              "all-transfer AUC therefore partly reflects whether blastocysts were available. The Bl > 0 subset "
              "is the per-transfer comparison. Laya calibration is monotone, so calibrated and raw Laya have the "
              "same AUC. On `per_transfer` cycles the DT 7.1 clinical probability equals BEFE (L7), hence identical values in the Bl > 0 subset; on transfers without blastocysts the value 0 also drives the low all-transfer calibration slope of the DT clinical probability.", ""]
    lines += discrimination(d)
    lines += ["## 4. Calibration in the large", "",
              "O/E = observed / mean predicted. Calibration slope = coefficient of logit(p) in a logistic model "
              "of the outcome (1 = ideal). Laya (raw) has no centre calibration; the PERSONA-calibrated Laya "
              "is evaluated out of sample (leave-one-year-out shift).", ""]
    lines += calibration(d)
    lines += laya_centre_section()
    dnn_imp, afc_src = {}, {}
    for y in YEARS:
        m = json.loads((BASE / y / "run_manifest.json").read_text(encoding="utf-8"))
        dnn_imp[y] = (m.get("dnn_coverage") or {}).get("model_imputed") or {}
        afc_src[y] = m["digital_twin"].get("afc_source") or {}
    bl_from_p50 = sum(int(v.get("bl", 0)) for v in dnn_imp.values())
    lines += ["## 6. Inputs imputed for PERSONA (owner-approved)", "",
              "- DT 7.1: AFC = OCC × 1.16 where no AFC is recorded (for donor-oocyte cycles the donor OCC); "
              "observed AFC where recorded (own-oocyte cycles, from the 2026 export); missing AMH/BMI by the "
              "pipeline formulas; missing attempt number = 1; sperm source inferred. AFC source per year: "
              + "; ".join(f"{y} {v}" for y, v in afc_src.items()) + ".",
              "- DNN (EVO): follicles = rint(OCC / 0.846) where follicles are not recorded; inseminated = MII "
              f"for donor-oocyte cycles; missing blastocyst count from the TabPFN-3 P50 in {bl_from_p50} cycle(s). "
              "Where imputed, OCC/follicles is ≈0.846 and carries no information. Per year: "
              + "; ".join(f"{y} {v}" for y, v in dnn_imp.items()) + ".",
              "- Laya: the same follicle/inseminated rule (`--impute-dt71`), attempt = 1 if missing.",
              f"- TabPFN stage range (`ci_width_pp`; historically labelled a 95% CI) is below 0.1 pp in {ci_small} of "
              f"{ci_n} cohort cycles (max {ci_max:.2f} pp): it is the spread over Monte-Carlo rows of UNKNOWN funnel "
              "stages and is exactly zero when every stage is known — not a model confidence interval. Model "
              "uncertainty (spread across TabPFN ensemble members, `tabpfn_ens_*`) is recorded from the "
              "2026-10-20 adapter onward.", ""]
    out = BASE / f"PERSONA_metrics_{YEARS[0]}_{YEARS[-1]}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[metrics] -> {out}")


if __name__ == "__main__":
    main()
