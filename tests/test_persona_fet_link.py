"""Связь криопереноса с пункцией (persona.link_fets): правило владельца 2026-10-20 —
ближайшая предыдущая пункция с остатком эмбрионов, сверка с "Остатками эмбрионов",
перенос без пункции в данных хранится без связи, дубль строки выгрузки не списывает
эмбрионы дважды."""
from __future__ import annotations

import pandas as pd

from embryo.clinics import persona

SECRET = b"test-secret"


def _opu(pid, date, frozen):
    return {"patient_id": pid, "treatment_type": "ТВП", "has_lab": True, "program_date": date, "date_opu": date,
            "date_et": None, "ivf_frozen": None, "icsi_frozen": frozen, "donor_frozen": None}


def _fet(pid, date, thawed, remaining=None, moved=1, preg=1.0):
    return {"patient_id": pid, "treatment_type": "КРИО ПЕ", "has_lab": False, "program_date": date,
            "date_opu": None, "date_et": date, "ivf_frozen": None, "icsi_frozen": None, "donor_frozen": None,
            "thawed_fet": thawed, "transferred_fet": moved, "embryos_remaining": remaining, "Preg": preg,
            "surrogate": False, "transfer_day": 5}


def test_nearest_opu_with_stock_and_remaining_check():
    rows = [_opu("A", "2024-01-10", 1), _opu("A", "2024-03-10", 2),
            _fet("A", "2024-05-01", 1, remaining=2),      # ближайшая (март) -> остаток 1 + 1 (январь) = 2
            _fet("A", "2024-06-01", 1, remaining=1),      # март -> 0; всего 1
            _fet("A", "2024-07-01", 1, remaining=0),      # у марта остатка нет -> январь
            _fet("B", "2024-02-01", 1)]                   # пункции нет
    links = persona.link_fets(pd.DataFrame(rows), SECRET).set_index("fet_date")
    assert links.loc["2024-05-01", "opu_date"] == "2024-03-10"
    assert links.loc["2024-05-01", "link_status"] == "nearest_with_stock"
    assert links.loc["2024-05-01", "remaining_check"] == "match"
    assert links.loc["2024-06-01", "opu_date"] == "2024-03-10"
    assert links.loc["2024-07-01", "opu_date"] == "2024-01-10"
    assert links.loc["2024-07-01", "remaining_check"] == "match"
    assert links.loc["2024-02-01", "link_status"] == "no_opu_in_data"
    assert pd.isna(links.loc["2024-02-01", "opu_cycle_uid"])


def test_opu_uid_matches_run_cycle_uid_and_duplicates_consume_once():
    rows = [_opu("C", "2024-01-10", 2), _fet("C", "2024-02-01", 1, remaining=1),
            _fet("C", "2024-02-01", 1, remaining=1)]       # повтор строки выгрузки
    links = persona.link_fets(pd.DataFrame(rows), SECRET)
    assert len(links) == 1
    assert links.loc[0, "link_status"] == "single_opu"
    assert links.loc[0, "remaining_check"] == "match"
    from embryo import privacy
    assert links.loc[0, "opu_cycle_uid"] == privacy.make_cycle_uid("C#20240110", SECRET)
