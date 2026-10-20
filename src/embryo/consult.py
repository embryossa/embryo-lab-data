"""
consult.py — детерминированный слой консультанта (перенос llm_consultant.py Tier 0/1
под облачного Claude). Принцип сохранён: Python КЛАССИФИЦИРУЕТ, Claude ОБЪЯСНЯЕТ.
Здесь НЕТ вызовов LLM (Ollama убран) — модуль собирает pre-classified «брифинг-пакеты»
(JSON), которые агент `consultant` (Claude Opus) превращает в клинический нарратив.

Пакеты:
  build_flags(row)            — клинические категории по порогам ЭКО (bands из оригинала)
  build_narrative_packet(row) — компактный per-patient пакет для нарратора (Tier 0)
  build_ensemble_packet(row)  — слои L1..L7 для аналитика ансамбля (Tier 1)
  build_cohort_packet(...)    — когортный пакет: KPI vs Вена, калибровка, review-list, MIRI
"""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd


def _g(row, *names, default=None):
    for n in names:
        if n in row and pd.notna(row[n]):
            try:
                return float(row[n])
            except (TypeError, ValueError):
                return row[n]
    return default


def build_flags(row: dict, primary_prob=("DIGITAL TWIN", "bayes_mean", "p_per_transfer")) -> dict:
    """Детерминированная классификация категорий (пороги — из llm_consultant.py)."""
    f: dict[str, Any] = {}

    p = _g(row, *primary_prob)
    if p is not None:
        pp = p * 100
        f["prognosis_level"] = ("favorable" if pp >= 55 else "good" if pp >= 40
                                else "moderate" if pp >= 25 else "low")

    # Неопределённость модели TabPFN = разброс вероятности между членами ансамбля
    # (tabpfn_ens_*, адаптер с 2026-10-20). Прежний `ci_width_pp` — НЕ ДИ: это разброс по
    # Монте-Карло-вариантам НЕИЗВЕСТНЫХ стадий воронки, ровно 0 при полной воронке
    # (проверено на всех клиниках 2026-10-20). Он идёт отдельным полем и в категорию
    # надёжности не входит. Пороги 15/30 пп перенесены с прежнего поля — проверить на
    # первых прогонах с ансамблем.
    w = _g(row, "tabpfn_ens_width_pp")
    if w is None:
        lo, hi = _g(row, "tabpfn_ens_low"), _g(row, "tabpfn_ens_high")
        if lo is not None and hi is not None:
            w = round((hi - lo) * 100, 1)
    funnel = _g(row, "ci_width_pp")
    if funnel is None:
        lo, hi = _g(row, "ci_low"), _g(row, "ci_high")
        if lo is not None and hi is not None:
            funnel = (hi - lo) * 100
    if funnel is not None:
        f["funnel_unknown_range_pp"] = round(float(funnel), 1)
    # Вырожденная ширина — непосчитанный разброс, а не бесконечно точный прогноз
    # (прецедент: два цикла Fertimed с 0.01 пп получили оценку A). 0.1 пп заведомо ниже
    # любого реального разброса вероятности беременности.
    _DEGENERATE_CI_PP = 0.1
    if w is not None and w >= _DEGENERATE_CI_PP:
        f["CI_width_pp"] = round(float(w), 1)
        f["CI_source"] = "tabpfn_ensemble"
        f["CI_category"] = ("low_statistical_uncertainty" if w < 15
                            else "moderate_statistical_uncertainty" if w < 30
                            else "high_statistical_uncertainty")
    elif w is not None:
        f["CI_width_pp"] = round(float(w), 1)
        f["CI_note"] = ("Разброс ансамбля TabPFN не рассчитан (вырожденная ширина "
                        f"{round(float(w), 2)} пп) — как сигнал не используется")
    else:
        f["CI_note"] = ("Разброс ансамбля TabPFN в этом прогоне не рассчитан; "
                        "funnel_unknown_range_pp — диапазон по неизвестным стадиям воронки, не ДИ модели")

    # [FIX spread] PRAI — второй независимый оценщик, присутствующий у каждого
    # цикла, но его не было в списке. В реальных когортах из семи перечисленных
    # колонок заполнена ровно одна («DIGITAL TWIN»), поэтому len(mp) никогда не
    # достигал 2 и agreement_category тоже не вычислялась.
    mp = [v for v in (_g(row, "p_per_transfer"), _g(row, "bayes_mean"), _g(row, "p_kat_raw"),
                      _g(row, "p_nvsa"), _g(row, "p_csdi"), _g(row, "p_gnn_ens"),
                      _g(row, "DIGITAL TWIN"), _g(row, "PRAI")) if v is not None]
    if len(mp) >= 2:
        sp = round((max(mp) - min(mp)) * 100, 1)
        f["ensemble_spread_pp"] = sp
        f["agreement_category"] = ("high_model_agreement" if sp < 15
                                   else "moderate_model_agreement" if sp < 30
                                   else "high_model_disagreement")

    # [FIX grade] Раньше при пустых ci_cat и agr обе строки были "", условие
    # `"" != "high_..."` истинно, и КАЖДЫЙ цикл получал «B». На реальных
    # когортах это давало 467 «B» из 467 — константу, которую нарратор
    # объяснял врачу как результат оценки надёжности.
    ci_cat, agr = f.get("CI_category", ""), f.get("agreement_category", "")
    if not ci_cat and not agr:
        f["confidence_grade"] = "insufficient_evidence"
        f["confidence_grade_reason"] = "no_uncertainty_signal"
    elif ci_cat == "low_statistical_uncertainty" and agr == "high_model_agreement":
        f["confidence_grade"] = "A"
        f["confidence_grade_reason"] = "narrow_CI_and_model_agreement"
    elif ci_cat != "high_statistical_uncertainty" and agr != "high_model_disagreement":
        f["confidence_grade"] = "B"
        f["confidence_grade_reason"] = (
            "moderate_CI" if ci_cat == "moderate_statistical_uncertainty" else
            "moderate_model_agreement" if agr == "moderate_model_agreement" else
            "partial_signal")
    else:
        f["confidence_grade"] = "C"
        f["confidence_grade_reason"] = (
            "wide_CI" if ci_cat == "high_statistical_uncertainty"
            else "model_disagreement")

    p_sev, p_mod = _g(row, "ohss_severe"), _g(row, "ohss_moderate")
    if p_sev is not None and p_sev >= 0.10:
        f["OHSS_risk"] = "high"
    elif (p_sev is not None and p_sev >= 0.05) or (p_mod is not None and p_mod >= 0.20):
        f["OHSS_risk"] = "moderate"
    elif p_sev is not None or p_mod is not None:
        f["OHSS_risk"] = "low"

    p_nb = _g(row, "p_no_blast")
    if p_nb is not None:
        f["empty_cycle_risk"] = ("high" if p_nb >= 0.35 else "moderate" if p_nb >= 0.20 else "low")

    cl = _g(row, "dominant_cluster")
    if cl is not None:
        f["response_phenotype"] = {0: "standard_responder", 1: "poor_responder",
                                   2: "high_responder"}.get(int(cl), "unknown")
        conf = _g(row, f"cluster_c{int(cl)}_prob")
        if conf is not None:
            f["phenotype_confidence_pct"] = round(conf * 100, 1)

    p_cancel = _g(row, "p_cancel_risk")
    if p_cancel is not None:
        f["cycle_management"] = ("high_cancellation_risk" if p_cancel >= 0.15
                                 else "monitor_closely" if p_cancel >= 0.07
                                 else "standard_cycle_expected")

    exp_e = _g(row, "banking_expected_euploid")
    age = _g(row, "age", "Age")
    if exp_e is not None:
        f["banking_expected_euploid"] = round(exp_e, 2)
        if age is not None:
            f["banking_age_urgency"] = ("high" if age >= 38 else "moderate" if age >= 35 else "low")
    return f


def build_narrative_packet(row: dict) -> dict:
    """Компактный per-patient пакет для нарратора Tier 0 (все числа готовы, Claude пишет текст)."""
    return {
        "цикл": {"cycle_uid": row.get("cycle_uid"),
                    "возраст": _g(row, "age", "Age"), "AMH": _g(row, "amh", "AMH"),
                    "AFC": _g(row, "afc", "Antral follicule count"), "попытка": _g(row, "attempt_number", "# Attempt")},
        "прогноз": {"P_per_transfer_%": _pct(_g(row, "p_per_transfer")),
                    "P_overall_cycle_%": _pct(_g(row, "p_overall_cycle")),
                    "DIGITAL_TWIN_%": _pct(_g(row, "DIGITAL TWIN")),
                    "PRAI_%": _pct(_g(row, "PRAI"))},
        "флаги": build_flags(row),
        "исход_факт": row.get("Preg") if pd.notna(row.get("Preg", np.nan)) else "неизвестен",
    }


def build_ensemble_packet(row: dict) -> dict:
    """Слои L1..L7 (Tier 1 — анализ согласованности/расхождений оценщиков)."""
    return {
        "L1_MC_prior": {"P_per_transfer_%": _pct(_g(row, "p_per_transfer")),
                        "P_overall_cycle_%": _pct(_g(row, "p_overall_cycle")),
                        "P_cancel_%": _pct(_g(row, "p_cancel_risk")),
                        "роль": "механистический приор (MC + FORTUNE/KPI воронка)"},
        "L3_KAT": {"P_KAT_%": _pct(_g(row, "p_kat_raw")),
                   "P_NVSA_correction_%": _pct(_g(row, "p_nvsa")),
                   "CI_low_%": _pct(_g(row, "ci_kat_low")), "CI_high_%": _pct(_g(row, "ci_kat_high")),
                   "роль": "KAN+FT-Transformer ансамбль (KAT); NVSA — его внутренняя "
                          "KPI-якорная коррекция, не отдельная модель"},
        "L4_кластер": {"доминирующий": _g(row, "dominant_cluster"),
                       "P_C0_%": _pct(_g(row, "cluster_c0_prob")),
                       "P_C1_%": _pct(_g(row, "cluster_c1_prob")),
                       "P_C2_%": _pct(_g(row, "cluster_c2_prob"))},
        "L5_CSDI": {"P_%": _pct(_g(row, "p_csdi")), "CI_low_%": _pct(_g(row, "csdi_ci_low")),
                    "CI_high_%": _pct(_g(row, "csdi_ci_high")),
                    "роль": "биологический якорь (диффузия); расхождение с L3/L6 >10пп = конфликт"},
        "L6_GAT": {"P_raw_%": _pct(_g(row, "p_gnn_raw")), "P_ens_%": _pct(_g(row, "p_gnn_ens")),
                   "вес": _g(row, "w_gnn"),
                   "роль": "graph attention transformer по графу пациентов; P_raw — сам GAT, "
                          "P_ens — уже смешан с KAT"},
        "L2_Bayes": {"bayes_mean_%": _pct(_g(row, "bayes_mean")),
                    "CI_low_%": _pct(_g(row, "bayes_ci_low")), "CI_high_%": _pct(_g(row, "bayes_ci_high")),
                    "роль": "Bayesian-обновлённый posterior над тем же прогоном L1 MC, "
                           "не отдельная архитектура"},
        "L7_BEFE": {"DIGITAL_TWIN_%": _pct(_g(row, "DIGITAL TWIN")),
                    "роль": "итоговый ансамбль (Bayesian Evidence Fusion Engine): L1+L3+L5+L6+L4"},
        "флаги_уверенности": {k: v for k, v in build_flags(row).items()
                              if k in ("CI_category", "agreement_category", "confidence_grade",
                                       "CI_width_pp", "CI_source", "funnel_unknown_range_pp",
                                       "ensemble_spread_pp")},
    }


def build_cohort_packet(kpi_summary: dict, calibration: dict, review_list,
                        incubator_findings=None, vienna_lcl=None) -> dict:
    """Когортный пакет — вход для ИТОГОВОГО слоя (объясняющее резюме всего разбора)."""
    return {
        "privacy": "Only cycle_uid is permitted; reverse identity join is local-only.",
        "KPI_vs_Вена": {"значения": kpi_summary, "LCL": vienna_lcl,
                        "бенчмарк": "Vienna Consensus PMID 28784335"},
        "калибровка_моделей": calibration,
        "review_list": review_list.to_dict("records") if hasattr(review_list, "to_dict") else review_list,
        "MIRI_форензик": incubator_findings or "не запускался",
    }


def _pct(x, nd=1):
    return round(float(x) * 100, nd) if x is not None and pd.notna(x) else None
