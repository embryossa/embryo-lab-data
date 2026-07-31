"""Deterministic local port of the DataLore analytics notebook.

The module keeps the notebook's merge, calibration, funnel and review logic,
but makes cohort conversion/KPI statistics reproducible and batch-safe:
aggregates are ratios of summed counts (not means of per-cycle percentages),
the review list is built only after TabPFN ``ci_wide`` is present, and the
Excel report is atomically replaced after a complete successful write.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._io import atomic_write_excel
from .kpi import (DOCTOR_METRICS, EMBRYOLOGIST_METRICS, aggregate_kpi_slices, aggregate_kpis,
                  compute_kpi, compute_rates, competency_exclusion_mask,
                  embryologist_only_exclusion_mask, funnel_exclusion_mask)

try:
    from sklearn.metrics import brier_score_loss, roc_auc_score
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False


PROB_MODELS = {
    "MC": "p_per_transfer", "Bayes": "bayes_mean", "KAT": "p_kat_raw",
    "NVSA": "p_nvsa", "CSDI": "p_csdi", "DT": "DIGITAL TWIN", "PRAI": "PRAI",
}
FUNNEL = {
    "OCC": ("med_okk", "OCC"), "MII": ("med_mii", "MII"),
    "2PN": ("med_pn2", "2pN"), "Bl": ("med_blasts", "Bl"),
    "Good": ("med_good", "Good Bl"),
}


def _num(value):
    return pd.to_numeric(value, errors="coerce")


def _one_num(value) -> float:
    return _num(pd.Series([value])).iloc[0]


def _normalise_key(series: pd.Series) -> pd.Series:
    # StringDtype preserves missing IDs; ordinary astype(str) silently creates
    # a joinable literal "NAN".
    return series.astype("string").str.strip().str.upper()


def _bool_series(value: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(value):
        return value.fillna(False)
    normal = value.astype("string").str.strip().str.lower()
    return normal.isin({"1", "true", "yes", "y", "да"})


def build_merged(dt: pd.DataFrame, opu: pd.DataFrame,
                 dt_key: str = "_key", opu_key: str = "_key"):
    dt = dt.copy()
    opu = opu.copy()
    kdt = dt_key if dt_key in dt.columns else "patient_id"
    if kdt not in dt.columns:
        raise KeyError(f"Digital Twin key not found: {dt_key!r} or 'patient_id'")
    kopu = opu_key if opu_key in opu.columns else "ID"
    if kopu not in opu.columns:
        raise KeyError(f"OPU key not found: {opu_key!r} or 'ID'")
    opu_key = kopu
    dt["_join"] = _normalise_key(dt[kdt])
    opu["_join"] = _normalise_key(opu[opu_key])
    df = pd.merge(dt, opu, on="_join", how="left", suffixes=("", "_opu"),
                  validate="many_to_one")
    if "timestamp" in df.columns:
        df_u = df.sort_values("timestamp").drop_duplicates("_join", keep="last").copy()
    else:
        df_u = df.drop_duplicates("_join", keep="last").copy()
    preg = _num(df_u["Preg"]) if "Preg" in df_u.columns else pd.Series(np.nan, index=df_u.index)
    df_u["Preg"] = preg
    df_preg = df_u.loc[df_u["Preg"].notna()].copy()
    return df, df_u, df_preg


def within_30(pred, real):
    pred, real = _num(pred), _num(real)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_error = (pred - real).abs() / real.replace(0, np.nan)
    return relative_error <= 0.30


def calibration_metrics(df_preg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = _num(df_preg["Preg"]) if "Preg" in df_preg else pd.Series(dtype=float)
    if y.notna().sum() < 6:
        return pd.DataFrame()
    for name, col in PROB_MODELS.items():
        if col not in df_preg.columns:
            continue
        p = _num(df_preg[col])
        mask = p.notna() & y.notna()
        if mask.sum() < 4:
            continue
        yy, pp = y.loc[mask].astype(int), p.loc[mask].clip(0, 1)
        rec = {
            "model": name, "n": int(mask.sum()),
            "mean_pred": round(float(pp.mean()), 3),
            "real_rate": round(float(yy.sum() / len(yy)), 3),
            "acc@0.5": round(float(((pp >= 0.5).astype(int) == yy).mean()), 3),
        }
        if _HAVE_SK:
            rec["brier"] = round(float(brier_score_loss(yy, pp)), 4)
            rec["auc"] = round(float(roc_auc_score(yy, pp)), 3) if yy.nunique() == 2 else None
        rows.append(rec)
    return pd.DataFrame(rows)


def _aggregate_ratio(frame: pd.DataFrame, numerator: str,
                     denominator: str) -> tuple[float, float, float]:
    if numerator not in frame.columns or denominator not in frame.columns:
        return np.nan, np.nan, np.nan
    n = _num(frame[numerator])
    d = _num(frame[denominator])
    valid = n.notna() & d.notna() & d.gt(0)
    if not valid.any():
        return np.nan, np.nan, np.nan
    n_sum, d_sum = float(n.loc[valid].sum()), float(d.loc[valid].sum())
    return n_sum / d_sum, n_sum, d_sum


def conversion_rates(df_u: pd.DataFrame) -> pd.DataFrame:
    """Real/predicted funnel conversions using sums of stage counts."""
    steps = [
        ("OCC→MII", "MII", "OCC", "med_mii", "med_okk"),
        ("MII→2PN", "2pN", "MII", "med_pn2", "med_mii"),
        ("2PN→Bl", "Bl", "2pN", "med_blasts", "med_pn2"),
        ("Bl→Good", "Good Bl", "Bl", "med_good", "med_blasts"),
    ]
    rows = []
    for name, real_n, real_d, pred_n, pred_d in steps:
        real, rn, rd = _aggregate_ratio(df_u, real_n, real_d)
        predicted, pn, pd_ = _aggregate_ratio(df_u, pred_n, pred_d)
        rows.append({
            "step": name,
            "real": round(real, 3) if pd.notna(real) else None,
            "real_numerator": rn, "real_denominator": rd,
            "predicted": round(predicted, 3) if pd.notna(predicted) else None,
            "predicted_numerator": pn, "predicted_denominator": pd_,
        })
    return pd.DataFrame(rows)


def auto_interpret(df_u: pd.DataFrame) -> pd.DataFrame:
    """Apply notebook review rules after joined TabPFN CI flags are present."""
    priorities, reasons = [], []
    ci_wide = (_bool_series(df_u["ci_wide"]) if "ci_wide" in df_u.columns
               else pd.Series(False, index=df_u.index))
    for pos, (_, row) in enumerate(df_u.iterrows()):
        priority, why = 3, []
        preg, mc = _one_num(row.get("Preg")), _one_num(row.get("p_per_transfer"))
        if pd.notna(preg) and preg == 0 and pd.notna(mc) and mc > 0.60:
            priority = 1
            why.append("ложноположительный (MC>60%, исход отрицательный)")

        real_bl, predicted_bl = _one_num(row.get("Bl")), _one_num(row.get("med_blasts"))
        if (pd.notna(real_bl) and real_bl > 0 and pd.notna(predicted_bl)
                and predicted_bl > 3 * real_bl):
            priority = 1
            why.append("прогноз бластоцист >3× реального")

        off = 0
        for predicted_col, real_col in FUNNEL.values():
            real = _one_num(row.get(real_col))
            predicted = _one_num(row.get(predicted_col))
            if pd.notna(real) and real > 0 and pd.notna(predicted):
                off += int(abs(predicted - real) / real > 0.30)
        kat, prai = _one_num(row.get("p_kat_raw")), _one_num(row.get("PRAI"))
        divergence = abs(kat - prai) * 100 if pd.notna(kat) and pd.notna(prai) else 0.0
        if priority != 1 and off >= 2:
            priority = 2
            why.append(f"воронка >30% на {off} этапах")
        if priority != 1 and divergence > 20:
            priority = 2
            why.append(f"KAT−PRAI расхождение {divergence:.0f} пп")
        if bool(ci_wide.iloc[pos]):
            if priority != 1:
                priority = 2
            why.append("ширина CI TabPFN >20 пп")
        priorities.append(priority)
        reasons.append("; ".join(why) or "в пределах ожиданий")
    out = df_u.copy()
    out["priority"] = priorities
    out["reason"] = reasons
    return out


def build_summary(df_u: pd.DataFrame) -> pd.DataFrame:
    def col(*names):
        for name in names:
            if name in df_u.columns:
                return df_u[name]
        return pd.Series([np.nan] * len(df_u), index=df_u.index)

    def real_pred(real_col, pred_col):
        real, pred = _num(col(real_col)), _num(col(pred_col))
        return [f"{'' if pd.isna(a) else int(a)} / {'' if pd.isna(b) else int(b)}"
                for a, b in zip(real, pred)]

    def flag(real_col, pred_col):
        ok = within_30(col(pred_col), col(real_col))
        real = _num(col(real_col))
        return ["—" if pd.isna(r) else ("✅" if bool(o) else "⚠️")
                for r, o in zip(real, ok)]

    def pct(name):
        return (_num(col(name)) * 100).round(1)

    summary = pd.DataFrame({
        "Пациент": col("patient_name", "Patient Full Name").astype("string").str[:25],
        "ID": col("patient_id", "ID"),
        "cycle_uid": col("cycle_uid"),
        "Возраст": col("age", "Age"),
        "OCC (р/п)": real_pred("OCC", "med_okk"), "OCC": flag("OCC", "med_okk"),
        "Bl (р/п)": real_pred("Bl", "med_blasts"), "Bl": flag("Bl", "med_blasts"),
        "MC%": pct("p_per_transfer"), "KAT%": pct("p_kat_raw"),
        "PRAI%": pct("PRAI"), "DT%": pct("DIGITAL TWIN"),
        "Кластер": col("dominant_cluster").map({0: "C0 Standard", 1: "C1 Poor", 2: "C2 High"}),
        "Исход": _num(col("Preg")).map({1: "✅", 0: "❌"}).fillna("—"),
        "Priority": col("priority"), "Reason": col("reason"),
    })
    summary["KAT−PRAI"] = [
        (f"+{delta:.1f} пп" if delta >= 0 else f"{delta:.1f} пп") if pd.notna(delta) else "—"
        for delta in (summary["KAT%"] - summary["PRAI%"])
    ]
    return summary


_KPI_ALIASES: dict[str, tuple[str, ...]] = {
    "age": ("age", "Age", "Возраст"),
    "attempt_no": ("attempt_no", "# Attempt", "№ попытки"),
    "follicles": ("follicles", "N folicules OPU", "Количество фолликулов"),
    "occ": ("occ", "OCC", "Число ОКК"),
    "mii": ("mii", "MII"),
    "inseminated": ("inseminated", "Inseminated", "Число инсеминированных"),
    "two_pn": ("two_pn", "2pN", "2 pN"),
    "three_pn": ("three_pn", "3pN", "3 pN"),
    "degrade": ("degrade", "Deg"),
    "cleavage_d3": ("cleavage_d3", "Cleavage", "Число дробящихся на 3 день"),
    "embryos_d5": ("embryos_d5", "Число эмбрионов 5 дня"),
    "bl_total": ("bl_total", "Bl", "Число Bl"),
    "bl_good": ("bl_good", "Good Bl", "Число Bl хор.кач-ва"),
    "frozen": ("frozen", "Cryo", "Заморожено эмбрионов"),
    "transferred": ("transferred", "ET", "Перенесено эмбрионов"),
    "date_opu": ("date_opu", "Date OPU"),
    "fert_method": ("fert_method", "IVF/ICSI"),
    "embryologist": ("embryologist", "Embryologist"),
    "doctor": ("doctor", "Doctor"),
}


_TRUE_COUNT_CANONICALS = (
    "age", "attempt_no", "follicles", "occ", "mii", "inseminated",
    "two_pn", "three_pn", "degrade", "cleavage_d3", "embryos_d5",
    "bl_total", "bl_good", "frozen", "transferred",
)


def _restore_true_counts(df_u: pd.DataFrame, lab_population: pd.DataFrame | None) -> pd.DataFrame:
    """Digital Twin's own reconciled export can substitute a model-side
    reconstruction for a raw count that was genuinely missing at input — e.g.
    OCC and MII both unknown in our data, DT echoes back a matching non-null
    pair in its own analytics-ready export (confirmed on Fertimed cycles: both
    NaN pre-DT, identical non-null values post-DT). That substitution must
    never be read back as an observed value for Vienna KPI or funnel-
    conversion purposes (owner confirmed 2026-07-18).

    Overwrites df_u's raw count columns with the true pre-DT values from
    ``lab_population`` (the same source ``kpi_computed.csv`` is written from),
    keyed by cycle_uid — a genuinely missing true value becomes NaN again,
    not whatever DT reconstructed. Leaves DT's own prediction/med_* columns
    untouched (conversion_rates' "predicted" side still reads those). No-op
    if either frame lacks cycle_uid.
    """
    if (lab_population is None or lab_population.empty
            or "cycle_uid" not in lab_population.columns
            or "cycle_uid" not in df_u.columns):
        return df_u
    out = df_u.copy()
    true_by_uid = lab_population.drop_duplicates("cycle_uid").set_index("cycle_uid")
    for canonical, aliases in _KPI_ALIASES.items():
        if canonical not in _TRUE_COUNT_CANONICALS or canonical not in true_by_uid.columns:
            continue
        true_values = pd.to_numeric(out["cycle_uid"].map(true_by_uid[canonical]), errors="coerce")
        for alias in aliases:
            if alias in out.columns:
                out[alias] = true_values
    return out


def _canonical_kpi_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for canonical, aliases in _KPI_ALIASES.items():
        for alias in aliases:
            if alias in frame.columns:
                out[canonical] = frame[alias]
                break
    # Keep local linkage and slicing fields alongside canonical counts.
    for name in ("cycle_uid", "patient_id", "ID"):
        if name in frame.columns:
            out[name] = frame[name]
    return out


def _funnel_excluded_cycle_uids(lab_population: pd.DataFrame | None,
                                columns_cfg: dict | None) -> set:
    """cycle_uid set flagged by funnel_exclusion_mask (e.g. Fertimed cryo-transfer
    rows whose 2PN/Bl counts were carried over from the original retrieval batch).
    Computed from ``lab_population`` (has ``treatment_type``) so it can be reused
    to filter ``df_u``-derived tables (funnel conversions, cohort KPI) that don't
    carry that column themselves. Empty for a clinic without the relevant config
    (e.g. clinic A) — a no-op there."""
    if (lab_population is None or lab_population.empty or columns_cfg is None
            or "cycle_uid" not in lab_population.columns):
        return set()
    mask = funnel_exclusion_mask(lab_population, columns_cfg)
    return set(lab_population.loc[mask, "cycle_uid"]) if mask.any() else set()


def _lab_kpi_slices(lab_population: pd.DataFrame, columns_cfg: dict) -> dict[str, pd.DataFrame]:
    """ALL Vienna laboratory KPI tables from the WIDER lab-stage population.

    Laboratory KPI describe what the lab did with the oocytes it received, so
    every lab-evaluable cycle counts regardless of whether a transfer happened
    or how it ended (owner reaffirmed 2026-07-31; matches the owner's
    Vienna-style dashboard). Until then only the embryologist/doctor competency
    tables were built here while overall/time/method still came from the
    transfer-gated cohort — on Fertimed's June export that cohort is 32/34
    cryo-transfer rows, so "KPI Overall" described frozen-embryo transfers, not
    the laboratory.

    Three nested exclusion sets, each narrower than the last:
      * ``competency_exclusion_mask`` — both tables (still-open cycles with no
        Bl value yet, NO-retrieval, freeze-all).
      * ``funnel_exclusion_mask`` — embryologist metrics only: this row's
        2PN/Bl counts were carried over by the MIS from another retrieval.
      * ``embryologist_only_exclusion_mask`` — embryologist metrics only:
        retrieval-only cycles that never reached insemination, whose
        follicles/OCC/MII are still a real retrieval for OCCR/MIIR.

    ``lab_population`` already carries the true columns.yaml canonical names
    (occ/mii/two_pn/.../fertilization_method/sperm_source) straight from
    ``normalize.normalize`` — unlike ``_canonical_kpi_frame``'s output, which is
    analytics.py's own internal shorthand for the DT/OPU-merged frame. No
    translation needed here; ``kpi.compute_rates``/``competency_exclusion_mask``
    read the columns.yaml names directly.
    """
    # Doctor metrics (OCCR/MIIR) read occ/follicles/mii, which still reflect a
    # real, single retrieval event by that doctor even on a row whose
    # fertilisation-stage counts are stale or absent — owner confirmed
    # 2026-07-18 (stale funnel) and 2026-07-31 (retrieval-only) that these must
    # NOT be excluded for doctor KPI.
    base_excluded = competency_exclusion_mask(lab_population, columns_cfg)
    embryologist_excluded = (base_excluded
                             | funnel_exclusion_mask(lab_population, columns_cfg)
                             | embryologist_only_exclusion_mask(lab_population, columns_cfg))
    # Doctor rates keep rows whose 2PN is blank because the cycle never reached
    # the fertilisation check: OCCR/MIIR live entirely above that stage. The
    # embryologist frame keeps the standing 2PN filter.
    lab_rates = compute_rates(lab_population)
    doctor_rates = compute_rates(lab_population, drop_without_two_pn=False)
    # Stable sheet schema: the cohort slices used to come from _canonical_kpi_frame,
    # whose method column is "fert_method". The lab population carries the
    # columns.yaml name instead, so alias it rather than renaming the output.
    for frame in (lab_rates, doctor_rates):
        if "fert_method" not in frame.columns and "fertilization_method" in frame.columns:
            frame["fert_method"] = frame["fertilization_method"]
    embryologist_eligible = lab_rates.loc[
        ~embryologist_excluded.reindex(lab_rates.index, fill_value=False)]
    doctor_eligible = doctor_rates.loc[
        ~base_excluded.reindex(doctor_rates.index, fill_value=False)]

    # overall/time/method mix both metric groups, so each group is aggregated on
    # its own correctly-scoped cohort and the two halves are recombined — the
    # same pattern _kpi_tables already used for the transfer-gated cohort.
    slice_kwargs = dict(
        time_col="date_opu" if "date_opu" in lab_rates.columns else None,
        method_col="fert_method" if "fert_method" in lab_rates.columns else None,
    )
    doctor_slices = aggregate_kpi_slices(doctor_eligible, **slice_kwargs)
    embryologist_slices = aggregate_kpi_slices(embryologist_eligible, **slice_kwargs)
    out: dict[str, pd.DataFrame] = {}
    for key in ("overall", "time", "method"):
        doc_df, emb_df = doctor_slices.get(key), embryologist_slices.get(key)
        if doc_df is None or emb_df is None:
            continue
        doc_rows = doc_df[doc_df["metric"].isin(DOCTOR_METRICS)]
        emb_rows = emb_df[~emb_df["metric"].isin(DOCTOR_METRICS)]
        out[key] = pd.concat([doc_rows, emb_rows], ignore_index=True)
    if "embryologist" in embryologist_eligible.columns:
        out["embryologist"] = aggregate_kpis(embryologist_eligible, "embryologist",
                                             metrics=EMBRYOLOGIST_METRICS)
    if "doctor" in doctor_eligible.columns:
        out["doctor"] = aggregate_kpis(doctor_eligible, "doctor", metrics=DOCTOR_METRICS)
    return out


def _kpi_tables(df_u: pd.DataFrame, lab_population: pd.DataFrame | None = None,
                columns_cfg: dict | None = None,
                cohort_population: pd.DataFrame | None = None,
                ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    canonical = _canonical_kpi_frame(df_u)
    cycle = compute_kpi(canonical)
    if cycle.empty:
        return cycle, {"overall": pd.DataFrame()}
    # When a lab population is supplied, EVERY Vienna KPI table comes from it
    # (see _lab_kpi_slices) — laboratory performance is not conditioned on a
    # transfer having happened. The transfer-gated tables below remain the
    # fallback for a caller that has no lab population (e.g. run_analytics
    # invoked directly from a notebook with dt/opu only).
    if lab_population is not None and not lab_population.empty and columns_cfg is not None:
        return cycle, _lab_kpi_slices(lab_population, columns_cfg)
    # funnel_exclusion_mask needs "treatment_type", which df_u/canonical/cycle don't
    # carry (OPU-enriched frame keeps only the fixed OPU-table contract) — computed
    # instead from cohort_population (same source rows, still has treatment_type) and
    # applied here by cycle_uid. Doctor metrics (OCCR/MIIR) never exclude these
    # rows (their occ/follicles/mii still reflect one real retrieval event); only
    # embryologist metrics (2PN/Bl-stage) do (owner confirmed 2026-07-18) — each
    # metric group computed from its own correctly-scoped cohort, then recombined.
    excluded_uids = _funnel_excluded_cycle_uids(cohort_population, columns_cfg)
    kwargs = dict(
        time_col="date_opu" if "date_opu" in cycle else None,
        method_col="fert_method" if "fert_method" in cycle else None,
        embryologist_col="embryologist" if "embryologist" in cycle else None,
        doctor_col="doctor" if "doctor" in cycle else None,
    )
    doctor_slices = aggregate_kpi_slices(cycle, **kwargs)
    if excluded_uids and "cycle_uid" in cycle.columns:
        embryologist_slices = aggregate_kpi_slices(
            cycle.loc[~cycle["cycle_uid"].isin(excluded_uids)], **kwargs)
        slices = {}
        for key in set(doctor_slices) | set(embryologist_slices):
            doc_df, emb_df = doctor_slices.get(key), embryologist_slices.get(key)
            if doc_df is None or doc_df.empty or "metric" not in doc_df.columns:
                slices[key] = emb_df if emb_df is not None else doc_df
            elif emb_df is None or emb_df.empty:
                slices[key] = doc_df
            else:
                doc_rows = doc_df[doc_df["metric"].isin(DOCTOR_METRICS)]
                emb_rows = emb_df[~emb_df["metric"].isin(DOCTOR_METRICS)]
                slices[key] = pd.concat([doc_rows, emb_rows], ignore_index=True)
    else:
        slices = doctor_slices
    return cycle, slices


def _review_list(df_u: pd.DataFrame) -> pd.DataFrame:
    base = _num(df_u["priority"]).le(2) if "priority" in df_u else pd.Series(False, index=df_u.index)
    wide = _bool_series(df_u["ci_wide"]) if "ci_wide" in df_u else pd.Series(False, index=df_u.index)
    wanted = [
        "cycle_uid", "patient_name", "Patient Full Name", "patient_id", "ID", "age", "Age",
        "priority", "reason", "PRAI", "PRAI CI (TabPFN), %", "ci_low", "ci_high",
        "ci_width_pp", "ci_wide", "DIGITAL TWIN", "DNN", "Preg", "Date OPU", "Date ET",
    ]
    columns = [name for name in wanted if name in df_u.columns]
    review = df_u.loc[base | wide, columns].copy()
    return review.sort_values([c for c in ["priority", "ci_width_pp"] if c in review.columns],
                              ascending=[True, False][:len([c for c in ["priority", "ci_width_pp"] if c in review.columns])])


def run_analytics(dt: pd.DataFrame, opu: pd.DataFrame,
                  out_xlsx: str | Path = "data/processed/DT_Analytics_Report.xlsx",
                  lab_population: pd.DataFrame | None = None,
                  columns_cfg: dict | None = None,
                  cohort_population: pd.DataFrame | None = None) -> dict:
    """Run local analytics and atomically create the multi-sheet Excel report.

    Two source populations, deliberately kept apart:

    * ``cohort_population`` — the report cohort (transferred>0 & outcome known),
      the same rows ``df_u`` was built from. Used to restore true pre-DT counts
      and to spot rows whose funnel counts are stale, both of which have to be
      keyed row-for-row against ``df_u``.
    * ``lab_population`` — the wider lab-stage population (no transfer gate).
      When supplied together with ``columns_cfg`` it is the source of every
      Vienna KPI table (see ``_lab_kpi_slices``).

    Passing only ``lab_population`` keeps the old behaviour of using it for
    both, which is correct exactly when it is a superset of the cohort.
    """
    merged, df_u, _ = build_merged(dt, opu)
    if cohort_population is None:
        cohort_population = lab_population
    df_u = _restore_true_counts(df_u, cohort_population)
    df_u = auto_interpret(df_u)  # deliberately after all joined CI fields exist
    df_preg = df_u.loc[_num(df_u["Preg"]).notna()].copy()
    summary = build_summary(df_u)
    calibration = calibration_metrics(df_preg)
    # Same funnel-inconsistent (e.g. cryo-transfer) rows excluded from embryologist
    # KPI (_kpi_tables) must also stay out of the MII→2PN/2PN→Bl/Bl→Good funnel
    # steps — all sum the same 2PN/Bl-style counts. OCC→MII is doctor-scoped
    # (mirrors MIIR) and never excludes these rows: occ/mii still reflect one
    # real retrieval event by that doctor (owner confirmed 2026-07-18). Empty
    # excluded set is a no-op (e.g. clinic A).
    excluded_uids = _funnel_excluded_cycle_uids(cohort_population, columns_cfg)
    conversions = conversion_rates(df_u)
    if excluded_uids and "cycle_uid" in df_u.columns:
        restricted = conversion_rates(df_u.loc[~df_u["cycle_uid"].isin(excluded_uids)])
        doctor_step = conversions.loc[conversions["step"] == "OCC→MII"]
        embryologist_steps = restricted.loc[restricted["step"] != "OCC→MII"]
        conversions = pd.concat([doctor_step, embryologist_steps], ignore_index=True)
    review = _review_list(df_u)
    kpi_cycle, kpi_slices = _kpi_tables(df_u, lab_population, columns_cfg, cohort_population)

    drop_internal = [col for col in df_u.columns if col.startswith("_")]
    sheets: dict[str, pd.DataFrame] = {
        "Summary": summary,
        "FullData": df_u.drop(columns=drop_internal, errors="ignore"),
        "WithOutcomes": df_preg.drop(
            columns=[c for c in df_preg.columns if c.startswith("_")], errors="ignore"),
        "Calibration": calibration,
        "Conversions": conversions,
        "ReviewList": review,
        "KPI Cycle": kpi_cycle.drop(
            columns=[c for c in kpi_cycle.columns if c.startswith("_")], errors="ignore"),
    }
    sheet_names = {
        "overall": "KPI Overall", "time": "KPI Time", "method": "KPI Method",
        "embryologist": "KPI Embryologist", "doctor": "KPI Doctor",
    }
    for key, table in kpi_slices.items():
        sheets[sheet_names[key]] = table
    report = atomic_write_excel(sheets, out_xlsx)
    return {
        "report": str(report), "n": len(df_u), "n_with_outcome": len(df_preg),
        "calibration": calibration, "conversion": conversions,
        "review_list": review, "df_u": df_u, "kpi_cycle": kpi_cycle,
        "kpi_slices": kpi_slices,
    }
