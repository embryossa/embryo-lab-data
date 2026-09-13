"""Заведение базы Teable и сборка карточек — против поддельного REST-клиента.

Настоящий Teable здесь не поднимается: клиент подменён объектом, который
записывает вызовы. Проверяется то, из-за чего заливка ломается молча — порядок
таблиц (link на ещё не созданную таблицу), повторный прогон (bootstrap обязан
быть идемпотентным и НЕ трогать вкладки, которые владелец настроил руками),
и содержимое карточки пациентки (в неё уходят ФИО и ID — она локальная, но её
собирает тот же код, что и пакеты агентам).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "teable"))

import cards_build  # noqa: E402
import teable_bootstrap as bootstrap  # noqa: E402
import teable_common as tc  # noqa: E402
import teable_schema as ts  # noqa: E402


class FakeTeable:
    """Клиент Teable без сети: помнит созданное и умеет отвечать как настоящий."""

    def __init__(self, tables: dict | None = None):
        self.state: dict[str, dict] = {}
        self.calls: list[tuple] = []
        for key, name in (tables or {}).items():
            self.state[key] = {"id": f"tbl{key}", "name": name, "fields": [], "views": []}

    # -- reads
    def tables(self, base_id):
        return [{"id": t["id"], "name": t["name"]} for t in self.state.values()]

    def _by_id(self, table_id):
        return next(t for t in self.state.values() if t["id"] == table_id)

    def fields(self, table_id):
        return list(self._by_id(table_id)["fields"])

    def views(self, table_id):
        return list(self._by_id(table_id)["views"])

    def all_records(self, table_id, projection=None):
        return self._by_id(table_id).get("records", [])

    # -- writes
    def create_table(self, base_id, body):
        key = body["name"].split(" —")[0].strip()
        # "type" отдаёт и настоящий API — apply_layout по нему ищет link-поля.
        fields = [{"id": f"fld{key}{i}", "name": f["name"], "type": f.get("type", "singleLineText")}
                  for i, f in enumerate(body["fields"])]
        self.state[key] = {"id": f"tbl{key}", "name": body["name"], "fields": fields, "views": []}
        self.calls.append(("create_table", body["name"], len(body["fields"])))
        return {"id": f"tbl{key}", "fields": fields}

    def create_field(self, table_id, body):
        table = self._by_id(table_id)
        table["fields"].append({"id": f"fld{len(table['fields'])}", "name": body["name"],
                                "type": body.get("type", "singleLineText")})
        self.calls.append(("create_field", table["name"], body["name"]))
        return table["fields"][-1]

    def create_view(self, table_id, body):
        table = self._by_id(table_id)
        table["views"].append({"id": f"viw{len(table['views'])}", "name": body["name"],
                               "body": body})
        self.calls.append(("create_view", table["name"], body["name"]))
        return table["views"][-1]

    def rename_table(self, base_id, table_id, name):
        self._by_id(table_id)["name"] = name
        self.calls.append(("rename_table", table_id, name))

    def move_table(self, base_id, table_id, anchor_id, position="after"):
        self.calls.append(("move_table", table_id, anchor_id))

    def move_view(self, table_id, view_id, anchor_id, position="after"):
        self.calls.append(("move_view", table_id, view_id))

    def set_column_meta(self, table_id, view_id, entries):
        self.calls.append(("set_column_meta", table_id, view_id, len(entries)))


# ------------------------------------------------------------- bootstrap

def test_bootstrap_creates_every_table_of_the_schema_on_an_empty_base():
    tb = FakeTeable()
    made = bootstrap.apply_schema(tb, "bse1", "Asia/Tashkent")
    assert made["tables"] == len(ts.table_keys())
    assert made["fields"] == 0                    # поля пришли вместе с таблицами
    created = [name.split(" —")[0].strip() for kind, name, _ in tb.calls if kind == "create_table"]
    assert created == ts.table_keys(), "порядок таблиц важен: link-цели создаются первыми"


def test_bootstrap_is_idempotent_on_an_already_built_base():
    tb = FakeTeable()
    bootstrap.apply_schema(tb, "bse1", "Asia/Tashkent")
    tb.calls.clear()
    made = bootstrap.apply_schema(tb, "bse1", "Asia/Tashkent")
    assert made == {"tables": 0, "fields": 0}
    assert tb.calls == [], "повторный bootstrap не должен трогать существующую базу"


def test_bootstrap_adds_only_the_fields_that_are_missing():
    tb = FakeTeable()
    bootstrap.apply_schema(tb, "bse1", "Asia/Tashkent")
    cycles = tb.state["Cycles"]
    dropped = cycles["fields"].pop()               # владелец удалил одно поле
    tb.calls.clear()

    made = bootstrap.apply_schema(tb, "bse1", "Asia/Tashkent")
    assert made["fields"] == 1
    assert [c for c in tb.calls if c[0] == "create_field"] == [
        ("create_field", cycles["name"], dropped["name"])]


def test_link_placeholder_is_resolved_to_a_real_foreign_table_id():
    resolved = bootstrap._resolve_link(ts.link("Cycles"), {"Cycles": "tblCycles"})
    assert resolved["options"]["foreignTableId"] == "tblCycles"
    assert "_link_to" not in resolved


def test_link_to_a_table_created_later_fails_loudly():
    # Иначе база собралась бы с битой связью и это всплыло бы только в интерфейсе.
    with pytest.raises(SystemExit, match="order of teable_schema.schema"):
        bootstrap._resolve_link(ts.link("Cycles"), {})


def test_dry_run_creates_nothing_but_still_resolves_link_targets(capsys):
    tb = FakeTeable()
    made = bootstrap.apply_schema(tb, "bse1", "Asia/Tashkent", dry_run=True)
    assert made["tables"] == len(ts.table_keys())
    assert tb.calls == []
    assert "[dry]" in capsys.readouterr().out


def test_views_are_created_once_and_never_overwritten():
    tb = FakeTeable()
    bootstrap.apply_schema(tb, "bse1", "Asia/Tashkent")
    made = bootstrap.apply_views(tb, "bse1")
    assert made == sum(len(v) for v in ts.views().values())

    tb.calls.clear()
    assert bootstrap.apply_views(tb, "bse1") == 0
    assert tb.calls == [], "вкладки владельца не должны переписываться"


def test_view_body_carries_resolved_sort_filter_and_hidden_columns():
    tb = FakeTeable()
    bootstrap.apply_schema(tb, "bse1", "Asia/Tashkent")
    bootstrap.apply_views(tb, "bse1")
    review_view = next(v for v in tb.state["Cycles"]["views"]
                       if "Ревью" in v["name"])["body"]
    assert review_view["filter"]["conjunction"] == "and"
    assert review_view["sort"]["manualSort"] is False
    # Всё, что не в списке колонок вкладки, прячется, но НЕ удаляется.
    hidden = [m for m in review_view["columnMeta"].values() if m["hidden"]]
    assert hidden and all(m["order"] >= 1000 for m in hidden)


def test_layout_renames_a_table_whose_display_name_changed_in_the_schema():
    tb = FakeTeable()
    bootstrap.apply_schema(tb, "bse1", "Asia/Tashkent")
    tb.state["Cycles"]["name"] = "Cycles — старое название"
    tb.calls.clear()

    assert bootstrap.apply_layout(tb, "bse1") >= 1
    renames = [c for c in tb.calls if c[0] == "rename_table"]
    assert renames and renames[0][2] == "Cycles — Пациентки / циклы"


def test_layout_on_a_freshly_built_base_has_nothing_to_move():
    tb = FakeTeable()
    bootstrap.apply_schema(tb, "bse1", "Asia/Tashkent")
    bootstrap.apply_views(tb, "bse1")
    tb.calls.clear()
    bootstrap.apply_layout(tb, "bse1")
    assert not [c for c in tb.calls if c[0] in ("rename_table", "move_table")]


# ------------------------------------------------------- REST-клиент

def test_dry_run_client_writes_nothing():
    client = tc.Teable("http://127.0.0.1:3000", "токен", dry_run=True)
    assert client.create_records("tbl1", [{"a": 1}, {"a": 2}]) == [{"id": "DRYRUN"}] * 2
    assert client.update("tbl1", "rec1", {"a": 1}) == {"id": "rec1"}
    assert client.update_records("tbl1", [("rec1", {"a": 1})]) is None
    assert client.upload_attachment("tbl1", "rec1", "fld1", "нет-файла") == {"id": "rec1"}


def test_client_sends_the_token_and_normalises_the_url():
    client = tc.Teable("http://127.0.0.1:3000/", "токен")
    assert client.url == "http://127.0.0.1:3000"
    assert client.s.headers["Authorization"] == "Bearer токен"


def test_connect_uses_the_default_url_when_config_omits_it():
    assert tc.connect({"token": "t"}).url == "http://127.0.0.1:3000"
    assert tc.connect({"token": "t", "teable_url": "http://host:1/"}).url == "http://host:1"


def test_create_records_on_an_empty_batch_does_nothing():
    assert tc.Teable("http://x", "t").create_records("tbl1", []) == []


# ----------------------------------------------------------- карточки

def _clinic_dir(tmp_path: Path) -> Path:
    proc = tmp_path / "processed"
    proc.mkdir()
    pd.DataFrame({
        "cycle_uid": ["CYC-1", "CYC-2"],
        "patient_id": ["E001#20260101", "E002#20260102"],
        "Patient Full Name": ["Пациент Один", "Пациент Два"],
        "Preg": [1, 0], "PRAI": [0.7, 0.3], "DIGITAL TWIN": [0.72, 0.35],
        "ci_low": [0.6, 0.2], "ci_high": [0.8, 0.45],
        "ci_width_pp": [20.0, 25.0], "ci_wide": [False, True],
        "Date OPU": ["2026-01-05", "2026-01-06"],
        "Date ET": ["2026-01-10", "2026-01-11"],
        "Age": [34, 39], "OCC": [8, 6], "MII": [7, 5], "2pN": [6, 3],
        "Bl": [3, 1], "Good Bl": [2, 0], "N folicules OPU": [10, 8],
        "Embryologist": ["Emb A", "Emb B"], "Doctor": ["Dr A", "Dr B"],
    }).to_csv(proc / "dt_predictions.csv", index=False)
    pd.DataFrame({"cycle_uid": ["CYC-2"], "priority": [1], "reason": ["широкий ДИ"]}
                 ).to_csv(proc / "review_list.csv", index=False)
    return proc


def test_payload_is_built_per_cycle_from_the_processed_folder(tmp_path):
    proc = _clinic_dir(tmp_path)
    clinic = tc.Clinic({"code": "test", "name": "Тест", "processed_dir": str(proc)}, {})
    payload = cards_build.build_payload(clinic)

    # Карточки идут от свежей пункции к старой — так владелец их и читает.
    assert [c["uid"] for c in payload["cycles"]] == ["CYC-2", "CYC-1"]
    assert payload["clinic"] == "Тест" and payload["n_fresh"] == 2
    flagged = next(c for c in payload["cycles"] if c["uid"] == "CYC-2")
    assert flagged["review"] == {"priority": 1, "reason": "широкий ДИ"}
    assert next(c for c in payload["cycles"] if c["uid"] == "CYC-1")["review"] is None


def test_payload_is_empty_when_the_clinic_has_no_run_yet(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    clinic = tc.Clinic({"code": "x", "processed_dir": str(empty)}, {})
    assert cards_build.build_payload(clinic) == {}


def test_payload_dates_are_not_day_month_swapped(tmp_path):
    proc = _clinic_dir(tmp_path)
    clinic = tc.Clinic({"code": "test", "processed_dir": str(proc)}, {})
    payload = cards_build.build_payload(clinic)
    by_uid = {c["uid"]: c for c in payload["cycles"]}
    # dayfirst=True на ISO прочитал бы "2026-01-05" как 1 мая — в карточке
    # пациентки стояла бы чужая дата пункции.
    assert by_uid["CYC-1"]["opu"] == "2026-01-05" and by_uid["CYC-1"]["et"] == "2026-01-10"
    assert by_uid["CYC-2"]["opu"] == "2026-01-06"
    assert "2026-05-01" not in json.dumps(payload, ensure_ascii=False)


def test_clinic_page_is_self_contained_html_with_the_payload_inlined(tmp_path):
    proc = _clinic_dir(tmp_path)
    clinic = tc.Clinic({"code": "test", "name": "Тест", "processed_dir": str(proc)}, {})
    payload = cards_build.build_payload(clinic)
    html_text = cards_build.render_clinic(payload, "http://127.0.0.1:8788",
                                          "http://127.0.0.1:3000", "bse1")
    assert html_text.lstrip().lower().startswith("<!doctype html")
    assert "CYC-1" in html_text
    # Страница открывается локально без сети: внешних <script src> быть не должно.
    assert '<script src="http' not in html_text.lower()


def test_index_page_lists_clinics_and_survives_an_empty_registry():
    entries = [{"code": "fertimed", "clinic": "Fertimed", "n": 34, "n_out": 34,
                "n_rev": 5, "built": "2026-08-29"}]
    page = cards_build.render_index(entries, "http://127.0.0.1:8788")
    assert "Fertimed" in page and "fertimed.html" in page
    assert "Нет собранных клиник" in cards_build.render_index([], "http://127.0.0.1:8788")


def test_record_links_degrade_to_empty_when_teable_is_unreachable(capsys):
    class Broken:
        def tables(self, base_id):
            raise ConnectionError("teable down")

    # Обратная ссылка «карточка -> строка в Teable» приятна, но её отсутствие
    # не должно ронять сборку карточек.
    assert cards_build.fetch_record_links(Broken(), "bse1", "http://127.0.0.1:3000") == {}
    assert "недоступны" in capsys.readouterr().out


def test_record_links_point_at_the_search_view():
    tb = FakeTeable({"Cycles": "Cycles — Пациентки / циклы"})
    tb.state["Cycles"]["views"] = [{"id": "viw0", "name": "Grid view"},
                                    {"id": "viw1", "name": "🔎 Поиск пациенток"}]
    tb.state["Cycles"]["records"] = [{"id": "rec1", "fields": {"cycle_uid": "CYC-1"}},
                                      {"id": "rec2", "fields": {}}]
    links = cards_build.fetch_record_links(tb, "bse1", "http://127.0.0.1:3000")
    assert links == {"CYC-1": "http://127.0.0.1:3000/base/bse1/tblCycles/viw1?recordId=rec1"}
