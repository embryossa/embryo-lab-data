"""consult.py — слой, где Python КЛАССИФИЦИРУЕТ, а агент только объясняет.

Особая цена ошибки: агент-нарратор пересказывает эти поля врачу как результат
оценки. Три дефекта в истории модуля выглядели одинаково — категория переставала
вычисляться и превращалась в константу, которую нарратор честно объяснял:
  * CI_category не считалась НИКОГДА (искали bayes_ci_low вместо готового ci_width_pp);
  * agreement_category не считалась НИКОГДА (в списке моделей не было PRAI, и
    заполненной оказывалась ровно одна из семи колонок);
  * confidence_grade был "B" у ВСЕХ 467 циклов (пустые категории проходили
    проверку `"" != "high_..."`).
Каждый из трёх закреплён отдельным тестом ниже.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from embryo import consult


def _row(**overrides) -> dict:
    base = {
        "cycle_uid": "CYC-0000000000000001",
        "age": 34, "amh": 2.2, "afc": 12, "attempt_number": 1,
        "DIGITAL TWIN": 0.62, "PRAI": 0.58, "ci_width_pp": 12.0, "Preg": 1,
    }
    base.update(overrides)
    return base


# ------------------------------------------------------------ прогноз

def test_prognosis_bands_follow_the_documented_thresholds():
    for prob, expected in [(0.60, "favorable"), (0.55, "favorable"), (0.45, "good"),
                           (0.40, "good"), (0.30, "moderate"), (0.25, "moderate"),
                           (0.10, "low")]:
        flags = consult.build_flags(_row(**{"DIGITAL TWIN": prob}))
        assert flags["prognosis_level"] == expected, prob


def test_prognosis_is_absent_when_no_probability_is_known():
    flags = consult.build_flags({"cycle_uid": "CYC-1"})
    assert "prognosis_level" not in flags


# ------------------------------------------- неопределённость TabPFN [FIX ci]

def test_ci_category_comes_from_the_tabpfn_ensemble_spread():
    assert consult.build_flags(_row(tabpfn_ens_width_pp=12.0))["CI_category"] == "low_statistical_uncertainty"
    assert consult.build_flags(_row(tabpfn_ens_width_pp=22.0))["CI_category"] == "moderate_statistical_uncertainty"
    flags = consult.build_flags(_row(tabpfn_ens_width_pp=35.0))
    assert flags["CI_category"] == "high_statistical_uncertainty"
    assert flags["CI_source"] == "tabpfn_ensemble"


def test_ensemble_width_is_derived_from_member_bounds():
    flags = consult.build_flags(_row(tabpfn_ens_low=0.30, tabpfn_ens_high=0.65))
    assert flags["CI_width_pp"] == 35.0
    assert flags["CI_category"] == "high_statistical_uncertainty"


def test_funnel_stage_range_is_not_a_model_ci():
    # ci_width_pp — разброс по неизвестным стадиям воронки (0 при полной воронке), а не
    # ДИ модели: отдельное поле, в категорию надёжности не входит.
    flags = consult.build_flags(_row(ci_width_pp=35.0))
    assert flags["funnel_unknown_range_pp"] == 35.0
    assert "CI_category" not in flags
    assert "не рассчитан" in flags["CI_note"]


def test_degenerate_ensemble_spread_is_reported_as_not_computed():
    flags = consult.build_flags(_row(tabpfn_ens_width_pp=0.01))
    assert "CI_category" not in flags
    assert "не рассчитан" in flags["CI_note"]
    assert flags["CI_width_pp"] == 0.0


def test_exactly_zero_ensemble_spread_is_degenerate_too():
    flags = consult.build_flags(_row(tabpfn_ens_width_pp=0.0))
    assert "CI_category" not in flags and "CI_note" in flags


# --------------------------------------------- согласие моделей [FIX spread]

def test_spread_counts_prai_alongside_digital_twin():
    # Регрессия: без PRAI в списке заполненной была ровно одна колонка из семи,
    # len(mp) не достигал 2, и agreement_category не считалась никогда.
    flags = consult.build_flags(_row(**{"DIGITAL TWIN": 0.62, "PRAI": 0.58}))
    assert flags["ensemble_spread_pp"] == 4.0
    assert flags["agreement_category"] == "high_model_agreement"


def test_spread_bands_separate_agreement_from_disagreement():
    assert consult.build_flags(_row(**{"DIGITAL TWIN": 0.80, "PRAI": 0.60})
                               )["agreement_category"] == "moderate_model_agreement"
    assert consult.build_flags(_row(**{"DIGITAL TWIN": 0.90, "PRAI": 0.30})
                               )["agreement_category"] == "high_model_disagreement"


def test_spread_is_absent_with_a_single_estimator():
    row = _row()
    row.pop("PRAI")
    assert "agreement_category" not in consult.build_flags(row)


# --------------------------------------------- оценка надёжности [FIX grade]

def test_no_uncertainty_signal_yields_insufficient_evidence_not_grade_b():
    # Регрессия: 467 циклов из 467 получали "B" — константу, которую нарратор
    # объяснял врачу как результат оценки надёжности.
    flags = consult.build_flags({"cycle_uid": "CYC-1", "DIGITAL TWIN": 0.5})
    assert flags["confidence_grade"] == "insufficient_evidence"
    assert flags["confidence_grade_reason"] == "no_uncertainty_signal"


def test_grade_a_requires_both_narrow_ci_and_model_agreement():
    flags = consult.build_flags(_row(tabpfn_ens_width_pp=10.0, **{"DIGITAL TWIN": 0.62, "PRAI": 0.60}))
    assert flags["confidence_grade"] == "A"
    assert flags["confidence_grade_reason"] == "narrow_CI_and_model_agreement"


def test_degenerate_ci_cannot_earn_grade_a():
    # Именно так два цикла Fertimed получили высшую оценку за непосчитанный ДИ.
    flags = consult.build_flags(_row(tabpfn_ens_width_pp=0.01, **{"DIGITAL TWIN": 0.62, "PRAI": 0.60}))
    assert flags["confidence_grade"] != "A"


def test_grade_c_on_wide_ci_or_model_disagreement():
    wide = consult.build_flags(_row(tabpfn_ens_width_pp=40.0))
    assert wide["confidence_grade"] == "C" and wide["confidence_grade_reason"] == "wide_CI"
    disagree = consult.build_flags(_row(tabpfn_ens_width_pp=10.0, **{"DIGITAL TWIN": 0.9, "PRAI": 0.3}))
    assert disagree["confidence_grade"] == "C"
    assert disagree["confidence_grade_reason"] == "model_disagreement"


# ------------------------------------------------------- клинические флаги

def test_ohss_risk_bands():
    assert consult.build_flags(_row(ohss_severe=0.12))["OHSS_risk"] == "high"
    assert consult.build_flags(_row(ohss_severe=0.06))["OHSS_risk"] == "moderate"
    assert consult.build_flags(_row(ohss_moderate=0.25))["OHSS_risk"] == "moderate"
    assert consult.build_flags(_row(ohss_severe=0.01))["OHSS_risk"] == "low"
    assert "OHSS_risk" not in consult.build_flags(_row())


def test_empty_cycle_and_cancellation_and_phenotype_flags():
    flags = consult.build_flags(_row(p_no_blast=0.40, p_cancel_risk=0.18,
                                     dominant_cluster=1, cluster_c1_prob=0.83))
    assert flags["empty_cycle_risk"] == "high"
    assert flags["cycle_management"] == "high_cancellation_risk"
    assert flags["response_phenotype"] == "poor_responder"
    assert flags["phenotype_confidence_pct"] == 83.0


def test_unknown_cluster_label_does_not_crash():
    assert consult.build_flags(_row(dominant_cluster=7))["response_phenotype"] == "unknown"


def test_banking_urgency_uses_age_bands():
    assert consult.build_flags(_row(age=39, banking_expected_euploid=1.2)
                               )["banking_age_urgency"] == "high"
    assert consult.build_flags(_row(age=36, banking_expected_euploid=1.2)
                               )["banking_age_urgency"] == "moderate"
    assert consult.build_flags(_row(age=30, banking_expected_euploid=1.2)
                               )["banking_age_urgency"] == "low"


# ------------------------------------------------------------- пакеты

def test_narrative_packet_matches_the_dt_narrator_contract():
    packet = consult.build_narrative_packet(_row())
    assert set(packet) == {"цикл", "прогноз", "флаги", "исход_факт"}
    assert packet["цикл"]["cycle_uid"] == "CYC-0000000000000001"
    assert packet["прогноз"]["DIGITAL_TWIN_%"] == 62.0
    assert packet["прогноз"]["PRAI_%"] == 58.0
    assert packet["исход_факт"] == 1


def test_narrative_packet_says_outcome_unknown_instead_of_nan():
    packet = consult.build_narrative_packet(_row(Preg=np.nan))
    assert packet["исход_факт"] == "неизвестен"


def test_narrative_packet_carries_no_identifiers_beyond_cycle_uid():
    # Пакет уходит агенту-нарратору: ФИО/ID/_key из строки просачиваться не должны.
    packet = consult.build_narrative_packet(
        _row(patient_id="E001#20260101", **{"Patient Full Name": "Пациент Один", "_key": "E001"}))
    blob = repr(packet)
    for leaked in ("E001", "Пациент Один"):
        assert leaked not in blob


def test_narrative_packet_handles_a_row_with_almost_nothing_in_it():
    packet = consult.build_narrative_packet({"cycle_uid": "CYC-1"})
    assert packet["цикл"]["возраст"] is None
    assert packet["прогноз"]["DIGITAL_TWIN_%"] is None
    assert packet["исход_факт"] == "неизвестен"


def test_ensemble_packet_exposes_every_documented_layer():
    packet = consult.build_ensemble_packet(_row(p_per_transfer=0.5, p_kat_raw=0.55,
                                                p_csdi=0.48, p_gnn_ens=0.52, bayes_mean=0.51,
                                                dominant_cluster=0, cluster_c0_prob=0.7))
    assert set(packet) >= {"L1_MC_prior", "L2_Bayes", "L3_KAT", "L4_кластер",
                           "L5_CSDI", "L6_GAT", "L7_BEFE", "флаги_уверенности"}
    assert packet["L7_BEFE"]["DIGITAL_TWIN_%"] == 62.0
    assert set(packet["флаги_уверенности"]) <= {
        "CI_category", "agreement_category", "confidence_grade",
        "CI_width_pp", "CI_source", "funnel_unknown_range_pp", "ensemble_spread_pp"}


def test_cohort_packet_states_the_privacy_boundary_and_serialises_review_list():
    review = pd.DataFrame({"cycle_uid": ["CYC-1"], "priority": [1]})
    packet = consult.build_cohort_packet({"FR": 0.75}, {"brier": 0.2}, review)
    assert "cycle_uid" in packet["privacy"]
    assert packet["review_list"] == [{"cycle_uid": "CYC-1", "priority": 1}]
    assert packet["MIRI_форензик"] == "не запускался"


def test_percent_helper_keeps_zero_and_drops_missing():
    assert consult._pct(0.0) == 0.0          # 0 — наблюдение, не пропуск
    assert consult._pct(None) is None
    assert consult._pct(np.nan) is None
    assert consult._pct(0.6789) == 67.9


def test_value_lookup_skips_missing_and_returns_first_present():
    assert consult._g({"a": np.nan, "b": 5}, "a", "b") == 5.0
    assert consult._g({"a": "текст"}, "a") == "текст"
    assert consult._g({}, "a", default="нет") == "нет"
