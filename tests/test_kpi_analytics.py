from pathlib import Path

import pandas as pd
import pytest

from embryo.analytics import conversion_rates, run_analytics
from embryo.kpi import (DOCTOR_METRICS, EMBRYOLOGIST_METRICS, aggregate_kpis, competency_exclusion_mask,
                        funnel_exclusion_mask,
                        compute_kpi, compute_rates, surgical_sperm_mask)


_EXCLUSION_CFG = {
    "competency_exclusion": {
        "no_retrieval_tokens": ["NO"],
        "freeze_all_tokens": ["Cryo"],
        "surgical_sperm_tokens": ["TESE", "SMF"],
    }
}


def _cycles(**overrides):
    values = {
        "age": [35, 41],
        "follicles": [10, 5],
        "occ": [8, 4],
        "mii": [None, 0],
        "inseminated": [8, 4],
        "two_pn": [6, 2],
        "cleavage_d3": [6, 2],
        "bl_total": [3, 1],
        "bl_good": [2, 0],
        "three_pn": [0, 1],
        "degrade": [0, 1],
    }
    values.update(overrides)
    return pd.DataFrame(values)


def test_mii_optional_zero_is_observed_and_fr_falls_back_to_inseminated():
    out = compute_rates(_cycles())
    assert out["FR"].tolist() == pytest.approx([0.75, 0.50])
    assert pd.isna(out.iloc[0]["MIIR"])
    assert out.iloc[1]["MIIR"] == 0.0  # known zero, not a missing value
    assert out["OCCR"].tolist() == pytest.approx([0.8, 0.8])
    model_frame = compute_kpi(_cycles())
    assert model_frame["Частота получения ОКК"].equals(model_frame["OCCR"])
    assert model_frame["Частота зрелости ОКК"].equals(model_frame["MIIR"])


def test_two_pn_zero_filter_is_preserved():
    source = _cycles(two_pn=[0, 2])
    out = compute_rates(source)
    assert out.index.tolist() == [1]


def test_model_score_is_separate_and_exact_training_formula():
    out = compute_kpi(_cycles().iloc[[0]])
    # age<=36:5, follicles 8..15:3, inseminated>7:5,
    # FR>0.65:5, good blastocysts<=2:3.
    assert out.iloc[0]["Model_KPIScore"] == 21
    assert out.iloc[0]["KPIScore"] == 21
    assert "Vienna_KPIScore" not in out.columns


def test_model_score_uses_training_fr_not_laboratory_fr():
    source = _cycles().iloc[[0]].copy()
    source["mii"] = 4
    source["inseminated"] = 8
    source["two_pn"] = 4
    out = compute_kpi(source).iloc[0]
    assert out["FR"] == pytest.approx(1.0)
    assert out["Model_FR"] == pytest.approx(0.5)
    assert out["Частота оплодотворения"] == pytest.approx(0.5)
    assert out["Model_KPIScore"] == 19


def test_cohort_rate_is_ratio_of_sums_and_controls_have_correct_direction():
    source = _cycles(
        follicles=[2, 10], occ=[1, 9], mii=[1, 9], inseminated=[1, 9],
        two_pn=[1, 9], cleavage_d3=[1, 9], bl_total=[1, 4], bl_good=[1, 3],
        three_pn=[0, 1], degrade=[0, 1],
    )
    aggregate = aggregate_kpis(source).set_index("metric")
    assert aggregate.loc["OCCR", "rate"] == pytest.approx(10 / 12)
    assert aggregate.loc["OCCR", "rate"] != pytest.approx((1 / 2 + 9 / 10) / 2)
    assert aggregate.loc["FR", "benchmark"] == 0.75
    assert aggregate.loc["FR", "direction"] == "lower"
    assert aggregate.loc["DegR", "direction"] == "upper"
    assert aggregate.loc["3pNR", "direction"] == "upper"
    assert 0 <= aggregate.loc["FR", "ci_low_95"] <= aggregate.loc["FR", "rate"]
    import math
    fr_rate = float(aggregate.loc["FR", "rate"])
    fr_ci_high = float(aggregate.loc["FR", "ci_high_95"])
    assert fr_rate <= fr_ci_high or math.isclose(fr_rate, fr_ci_high, abs_tol=1e-9)
    assert fr_ci_high <= 1.0 or math.isclose(fr_ci_high, 1.0, abs_tol=1e-9)


def test_conversion_uses_sum_over_sum_not_mean_of_cycle_percentages():
    frame = pd.DataFrame({"2pN": [1, 9], "MII": [2, 10]})
    result = conversion_rates(frame).set_index("step")
    assert result.loc["MII→2PN", "real"] == pytest.approx(10 / 12, abs=0.0005)


def test_ci_wide_is_added_to_review_after_joined_flag(tmp_path: Path):
    dt = pd.DataFrame({
        "_key": ["A", "B"], "Preg": [1, 1], "DIGITAL TWIN": [0.7, 0.6],
        "PRAI": [0.65, 0.55], "ci_wide": [False, True], "ci_width_pp": [10, 25],
    })
    opu = pd.DataFrame({
        "ID": ["A", "B"], "Age": [35, 36], "N folicules OPU": [10, 8],
        "OCC": [8, 6], "Inseminated": [8, 6], "2pN": [6, 5],
        "Cleavage": [6, 5], "Bl": [3, 2], "Good Bl": [2, 1],
    })
    report = tmp_path / "report.xlsx"
    result = run_analytics(dt, opu, report)
    assert report.exists()
    assert result["review_list"]["ID"].tolist() == ["B"]
    reviewed = result["df_u"].set_index("ID").loc["B"]
    assert reviewed["priority"] == 2
    assert "диапазон по неизвестным стадиям" in reviewed["reason"]
    assert "KPI Overall" in pd.ExcelFile(report).sheet_names


def test_aggregate_kpis_metrics_filter_restricts_to_requested_subset():
    source = _cycles(embryologist=["A", "B"], doctor=["X", "Y"])
    embryologist = aggregate_kpis(source, "embryologist", metrics=EMBRYOLOGIST_METRICS)
    doctor = aggregate_kpis(source, "doctor", metrics=DOCTOR_METRICS)
    assert set(embryologist["metric"]) == set(EMBRYOLOGIST_METRICS)
    assert set(doctor["metric"]) == set(DOCTOR_METRICS)
    # never mixed: doctor metrics never computed in the embryologist table or vice versa
    assert set(embryologist["metric"]).isdisjoint(DOCTOR_METRICS)
    assert set(doctor["metric"]).isdisjoint(EMBRYOLOGIST_METRICS)
    # unrestricted call still returns every KPI_SPECS entry (back-compat)
    full = aggregate_kpis(source, "embryologist")
    assert set(full["metric"]) == set(EMBRYOLOGIST_METRICS) | set(DOCTOR_METRICS)


def test_competency_exclusion_mask_flags_open_no_retrieval_and_freeze_all():
    frame = pd.DataFrame({
        "bl_total": [3, None, 2, 1],
        "fertilization_method": ["ICSI", "ICSI", "NO", "Cryo"],
    })
    mask = competency_exclusion_mask(frame, _EXCLUSION_CFG)
    assert mask.tolist() == [False, True, True, True]  # open-cycle, NO-retrieval, freeze-all


def test_surgical_sperm_mask_matches_only_configured_tokens():
    frame = pd.DataFrame({"sperm_source": ["TESE", "SMF", "ejaculate", None]})
    mask = surgical_sperm_mask(frame, _EXCLUSION_CFG)
    assert mask.tolist() == [True, True, False, False]


def test_funnel_exclusion_mask_flags_only_configured_treatment_tokens():
    frame = pd.DataFrame({"treatment_type": ["Криоперенос", "ИКСИ", "криоперенос", None]})
    cfg = {"funnel_exclusion": {"stale_funnel_tokens": ["Криоперенос"]}}
    mask = funnel_exclusion_mask(frame, cfg)
    assert mask.tolist() == [True, False, True, False]  # case-insensitive, exact token


def test_funnel_exclusion_mask_is_noop_without_treatment_type_column():
    frame = pd.DataFrame({"bl_total": [1, 2]})
    cfg = {"funnel_exclusion": {"stale_funnel_tokens": ["Криоперенос"]}}
    assert funnel_exclusion_mask(frame, cfg).tolist() == [False, False]


def test_open_cycle_falls_back_to_bl_total_without_closure_signal():
    """Клиники без "Причины завершения" продолжают жить по прежнему прокси."""
    frame = pd.DataFrame({
        "bl_total": [3, None],
        "fertilization_method": ["ICSI", "ICSI"],
    })
    assert competency_exclusion_mask(frame, _EXCLUSION_CFG).tolist() == [False, True]


def test_closure_signal_wins_over_missing_bl_total():
    """Пустой Bl у закрытого цикла — не открытый цикл: строка остаётся в KPI.

    Иначе из знаменателя уходят ровно те циклы, где бластоцист не получилось
    (и, отдельно, каждая строка "Только пункция", у которой Bl нет никогда,
    хотя OCCR/MIIR врача считаются выше стадии бластоцисты).
    """
    frame = pd.DataFrame({
        "bl_total": [None, None, 2],
        "clin_termination_reason": ["Остановка развития эмбрионов", None, "ПГТ"],
        "fertilization_method": ["ICSI", "ICSI", "ICSI"],
    })
    mask = competency_exclusion_mask(frame, _EXCLUSION_CFG)
    assert mask.tolist() == [False, True, False]  # закрыт / ещё идёт / закрыт


def test_embryologist_only_exclusion_keeps_row_for_doctor_table():
    from embryo.kpi import embryologist_only_exclusion_mask
    frame = pd.DataFrame({
        "treatment_type": ["Только пункция", "ИКСИ", "только пункция", None],
        "clin_termination_reason": ["Криоконсервация ооцитов", "ПГТ", "Донор", "Овуляция"],
    })
    cfg = {"competency_exclusion": {"embryologist_only_treatment_tokens": ["Только пункция"]}}
    mask = embryologist_only_exclusion_mask(frame, cfg)
    assert mask.tolist() == [True, False, True, False]
    # это НЕ competency-исключение: врачебная таблица такую строку сохраняет
    assert competency_exclusion_mask(frame, cfg).tolist() == [False, False, False, False]


def test_compute_rates_keeps_retrieval_only_rows_for_doctor_metrics():
    """OCCR/MIIR считаются выше стадии 2PN, поэтому пустой 2PN их не отменяет.

    Записанный 2PN=0 продолжает выпадать в обоих режимах — неподвижное
    решение владельца не затронуто.
    """
    frame = pd.DataFrame({
        "follicles": [10, 10, 10],
        "occ": [8, 8, 8],
        "mii": [6, 6, 6],
        "inseminated": [8, None, 8],
        "two_pn": [6, None, 0],
    })
    assert len(compute_rates(frame)) == 1
    doctor = compute_rates(frame, drop_without_two_pn=False)
    assert len(doctor) == 2                      # пустой 2PN сохранён, записанный 0 — нет
    assert doctor["OCCR"].tolist() == [0.8, 0.8]
    assert doctor["MIIR"].tolist() == [0.75, 0.75]
