"""Слой Teable: реестр клиник, схема базы и преобразование значений при заливке.

Сеть здесь не трогается — только чистая логика, которая решает, ЧТО поедет в
базу и под каким именем. Ошибка в ней тихая: карточка пациентки просто
показывает не то, а расхождение схемы с отчётом («Бласт» против «Бластоцисты»)
всплывает уже на живой заливке. Отдельно проверяются два кросс-модульных
контракта — воронка и стадии регрессии должны совпадать с отчётом и с
contracts.py, иначе Teable и HTML описывают один прогон по-разному.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "teable"))

import teable_common as tc  # noqa: E402
import teable_schema as ts  # noqa: E402
import teable_sync  # noqa: E402
from embryo import contracts, report  # noqa: E402


# --------------------------------------------------------- реестр клиник

def _config(tmp_path: Path, **overrides) -> Path:
    payload = {
        "teable_url": "http://127.0.0.1:3000",
        "token": "локальный-токен",
        "cards_url": "http://127.0.0.1:8788/",
        "cards_dir": "data/processed/_viewer",
        "clinics": [
            {"code": "fertimed", "name": "Fertimed", "base_id": "bseFERTIMED",
             "processed_dir": "data/processed/fertimed"},
            {"code": "ggrc", "name": "GGRC", "base_id": "",
             "processed_dir": "data/processed/ggrc", "timezone": "Asia/Yerevan"},
        ],
    }
    payload.update(overrides)
    path = tmp_path / "teable_config.local.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_clinic_resolves_relative_processed_dir_against_repo_root(tmp_path):
    cfg = tc.load_config(str(_config(tmp_path)))
    fertimed, ggrc = tc.clinics(cfg)
    assert Path(fertimed.processed_dir).is_absolute()
    assert Path(fertimed.processed_dir).parts[-2:] == ("processed", "fertimed")
    # Абсолютный путь берётся как есть — конфиг может указывать вне репозитория.
    absolute = tc.Clinic({"code": "x", "processed_dir": str(tmp_path)}, cfg)
    assert absolute.processed_dir == str(tmp_path)
    assert ggrc.timezone == "Asia/Yerevan"


def test_clinic_without_base_id_is_not_configured(tmp_path):
    cfg = tc.load_config(str(_config(tmp_path)))
    fertimed, ggrc = tc.clinics(cfg)
    assert fertimed.configured and not ggrc.configured


def test_card_url_points_at_the_clinic_page_and_anchors_the_cycle(tmp_path):
    cfg = tc.load_config(str(_config(tmp_path)))
    clinic = tc.clinics(cfg, only="Fertimed")[0]
    base = tc.cards_base_url(cfg)
    assert base == "http://127.0.0.1:8788"
    assert clinic.card_url(base) == "http://127.0.0.1:8788/fertimed.html"
    assert clinic.card_url(base, "CYC-1") == "http://127.0.0.1:8788/fertimed.html#uid=CYC-1"


def test_clinic_filter_matches_code_or_name_and_rejects_unknown(tmp_path):
    cfg = tc.load_config(str(_config(tmp_path)))
    assert [c.code for c in tc.clinics(cfg, only="fertimed")] == ["fertimed"]
    assert [c.code for c in tc.clinics(cfg, only="GGRC")] == ["ggrc"]
    with pytest.raises(SystemExit, match="not in the registry"):
        tc.clinics(cfg, only="нет-такой")


def test_missing_config_exits_with_a_pointer_to_the_example(tmp_path):
    with pytest.raises(SystemExit, match="teable_config.local.json.example"):
        tc.load_config(str(tmp_path / "нет.json"))


def test_config_without_clinics_exits(tmp_path):
    cfg = tc.load_config(str(_config(tmp_path, clinics=[])))
    with pytest.raises(SystemExit, match="No 'clinics'"):
        tc.clinics(cfg)


def test_save_base_id_persists_only_the_named_clinic(tmp_path):
    path = _config(tmp_path)
    cfg = tc.load_config(str(path))
    tc.save_base_id(cfg, "ggrc", "bseNEW")
    written = json.loads(path.read_text(encoding="utf-8"))
    by_code = {c["code"]: c["base_id"] for c in written["clinics"]}
    assert by_code == {"fertimed": "bseFERTIMED", "ggrc": "bseNEW"}
    with pytest.raises(KeyError, match="нет-такой"):
        tc.save_base_id(cfg, "нет-такой", "bseX")


def test_resolve_tables_strips_the_russian_suffix_from_display_names():
    class FakeClient:
        def tables(self, base_id):
            return [{"name": "Runs — Прогоны анализа", "id": "tblRuns"},
                    {"name": "Cycles — Пациентки / циклы", "id": "tblCycles"}]

    assert tc.resolve_tables(FakeClient(), "bse1") == {"Runs": "tblRuns", "Cycles": "tblCycles"}


# ---------------------------------------------------------------- схема

def test_schema_tables_are_unique_and_match_table_keys():
    tables = ts.schema()
    keys = [t["key"] for t in tables]
    assert keys == ts.table_keys()
    assert len(keys) == len(set(keys))
    for table in tables:
        names = [f["name"] for f in table["fields"]]
        assert len(names) == len(set(names)), f"дубль поля в таблице {table['key']}"


def test_every_view_references_an_existing_table_and_field():
    fields_by_table = {t["key"]: {f["name"] for f in t["fields"]} for t in ts.schema()}
    for table_key, view_list in ts.views().items():
        assert table_key in fields_by_table, table_key
        known = fields_by_table[table_key]
        for view in view_list:
            for column in view.get("columns", []):
                assert column in known, f"{table_key}/{view['name']}: нет поля {column}"
            for field_name, _op, _value in view.get("filter", []):
                assert field_name in known, f"{table_key}/{view['name']}: фильтр по {field_name}"
            for field_name, _direction in view.get("sort", []):
                assert field_name in known, f"{table_key}/{view['name']}: сортировка по {field_name}"


def test_funnel_stages_match_the_html_report():
    # Teable и отчёт обязаны описывать одну воронку одними словами в одном порядке.
    assert ts.FUNNEL_STAGES == report._JOURNAL_FUNNEL_STAGES
    assert ts.FUNNEL_REAL_COLUMN == report._JOURNAL_REAL_COLUMN


def test_regression_stage_candidates_exist_in_the_contract():
    known = set(contracts.REGRESSION_STAGES)
    for stage, candidates in ts.REGRESSION_STAGES.items():
        assert set(candidates) <= known, f"{stage}: {candidates} нет в contracts.REGRESSION_STAGES"


def test_ozema_uses_the_project_mandated_na_token():
    # "n/a" сталкивается с na_values pandas — по правилам проекта только "н/д".
    names = {c["name"] for c in ts.OZEMA_RESULT_CHOICES} | {c["name"] for c in ts.OZEMA_FLAG_CHOICES}
    assert "н/д" in names
    assert "n/a" not in {n.lower() for n in names}


def test_field_constructors_produce_valid_teable_specs():
    assert ts.num(2)["options"]["formatting"]["precision"] == 2
    assert ts.date(True, "Asia/Tashkent")["options"]["formatting"]["time"] == "HH:mm"
    assert ts.date(False)["options"]["formatting"]["time"] == "None"
    assert ts.link("Cycles")["_link_to"] == "Cycles"
    assert ts.f("Возраст", ts.num(0))["name"] == "Возраст"


# ----------------------------------------------- преобразование значений

def test_num_keeps_zero_and_drops_missing():
    assert teable_sync.num(0) == 0            # 0 — наблюдение, не пропуск
    assert teable_sync.num(3.0) == 3 and isinstance(teable_sync.num(3.0), int)
    assert teable_sync.num(2.5) == 2.5
    assert teable_sync.num(None) is None
    assert teable_sync.num(float("nan")) is None
    assert teable_sync.num("не число") is None


def test_text_and_flag_normalisation():
    assert teable_sync.text("  Emb A ") == "Emb A"
    assert teable_sync.text("   ") is None
    assert teable_sync.text(None) is None
    assert teable_sync.flag("да") is True and teable_sync.flag("True") is True
    assert teable_sync.flag("нет") is False
    assert teable_sync.flag(None) is None      # неизвестно ≠ False
    assert teable_sync.flag(1) is True


def test_norm_id_and_iso_date():
    assert teable_sync.norm_id(" e001#20260101 ") == "E001#20260101"
    assert teable_sync.norm_id(None) is None
    assert teable_sync.iso_date("2026-01-05") == "2026-01-05"
    assert teable_sync.iso_date("05.01.2026") == "2026-01-05"   # dayfirst, как в выгрузках
    assert teable_sync.iso_date("не дата") is None
    assert teable_sync.iso_date(None) is None


def test_preg_label_distinguishes_negative_from_unknown():
    assert teable_sync.preg_label(1) == "Беременность"
    assert teable_sync.preg_label(0) == "Нет"
    assert teable_sync.preg_label(None) == "Неизвестно"
    assert teable_sync.preg_label(float("nan")) == "Неизвестно"


def test_flag30_mirrors_the_report_deviation_rule():
    assert teable_sync.flag30(10, 12) == "✅ в норме"        # +20%
    assert teable_sync.flag30(10, 14) == "⚠️ вне ±30%"
    assert teable_sync.flag30(10, 13) == "✅ в норме"        # ровно 30% — в норме
    assert teable_sync.flag30(0, 5) == "— н/д"               # деления на ноль нет
    assert teable_sync.flag30(None, 5) == "— н/д"
    assert teable_sync.flag30(10, None) == "— н/д"


def test_priority_label_maps_to_the_schema_choices():
    known = {c["name"] for c in ts.PRIORITY_CHOICES}
    for value in (1, 2, 3, 9):
        assert teable_sync.priority_label(value) in known
    assert teable_sync.priority_label(None) is None


def test_regression_stage_fields_counts_only_stages_outside_the_interval():
    row = {
        "mii_p50": 7, "mii_p10": 4, "mii_p90": 11, "mii_actual": 8,
        "mii_in_interval80": True,
        "pn2_p50": 5, "pn2_p10": 3, "pn2_p90": 7, "pn2_actual": 9,
        "pn2_in_interval80": False,
        # good: прогноза нет вовсе — стадия пропускается целиком
        "good_in_interval80": None,
    }
    fields, outside = teable_sync.regression_stage_fields(row)
    assert outside == 1
    assert fields["MII интервал"] == "✅ в P10–P90"
    assert fields["2PN интервал"] == "⚠️ вне P10–P90"
    assert not any(k.startswith("Хор.бласт") for k in fields)
    known = {c["name"] for c in ts.INTERVAL_CHOICES}
    assert {v for k, v in fields.items() if k.endswith("интервал")} <= known


def test_regression_stage_fields_on_absent_row():
    assert teable_sync.regression_stage_fields(None) == ({}, 0)


def test_blast_stage_prefers_the_after_fertilisation_forecast():
    # Обе колонки предсказывают одно "Bl", но после известного 2PN — точнее.
    row = {"blasts_after_fert_p50": 3, "blasts_after_fert_p10": 2,
           "blasts_after_fert_p90": 5, "blasts_after_fert_actual": 3,
           "blasts_after_fert_in_interval80": True,
           "blasts_early_p50": 9, "blasts_early_p10": 8, "blasts_early_p90": 10,
           "blasts_early_actual": 3, "blasts_early_in_interval80": False}
    fields, outside = teable_sync.regression_stage_fields(row)
    assert fields["Бласт P50"] == 3 and outside == 0


def test_read_any_returns_empty_frame_for_missing_or_broken_file(tmp_path, capsys):
    assert teable_sync.read_any(str(tmp_path / "нет.csv")).empty
    broken = tmp_path / "broken.xlsx"
    broken.write_text("это не excel", encoding="utf-8")
    assert teable_sync.read_any(str(broken)).empty
    assert "[!]" in capsys.readouterr().out


def test_read_any_loads_csv_and_xlsx(tmp_path):
    frame = pd.DataFrame({"cycle_uid": ["CYC-1"], "PRAI": [0.5]})
    frame.to_csv(tmp_path / "a.csv", index=False)
    frame.to_excel(tmp_path / "a.xlsx", index=False)
    assert len(teable_sync.read_any(str(tmp_path / "a.csv"))) == 1
    assert len(teable_sync.read_any(str(tmp_path / "a.xlsx"))) == 1


def test_index_by_and_clean():
    frame = pd.DataFrame({"cycle_uid": ["CYC-1", "CYC-2"], "PRAI": [0.5, 0.7]})
    indexed = teable_sync.index_by(frame, "cycle_uid")
    assert indexed["CYC-2"]["PRAI"] == 0.7
    assert teable_sync.index_by(pd.DataFrame(), "cycle_uid") == {}
    assert teable_sync.index_by(frame, "нет-колонки") == {}
    assert teable_sync.clean({"a": 1, "b": None}) == {"a": 1}


def test_pick_takes_the_first_present_value():
    assert teable_sync.pick({"a": None, "b": 5}, ["a", "b"]) == 5
    assert teable_sync.pick({}, ["a"]) is None


def test_sha256_file_matches_the_manifest_hash_length(tmp_path):
    path = tmp_path / "artifact.csv"
    path.write_text("cycle_uid\nCYC-1\n", encoding="utf-8")
    digest = teable_sync.sha256_file(str(path))
    assert len(digest) == 64 and digest == teable_sync.sha256_file(str(path))


def test_sync_skips_a_clinic_without_predictions_or_base_id(tmp_path, capsys):
    class Args:
        pass

    clinic = tc.Clinic({"code": "x", "processed_dir": str(tmp_path)}, {})
    assert teable_sync.sync_clinic({}, None, clinic, Args()) is False
    assert "нет dt_predictions.csv" in capsys.readouterr().out

    pd.DataFrame({"cycle_uid": ["CYC-1"]}).to_csv(tmp_path / "dt_predictions.csv", index=False)
    assert teable_sync.sync_clinic({}, None, clinic, Args()) is False
    assert "нет base_id" in capsys.readouterr().out
