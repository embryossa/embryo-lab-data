"""Заливка прогона в Teable (teable_sync.sync_clinic) против поддельного клиента.

Сеть не трогается. Проверяется поведение, которое дороже всего чинить постфактум:
повторная заливка того же прогона не должна плодить дубли (идентичность прогона —
хэш содержимого dt_predictions.csv), --dry-run не должен писать ничего, а
отсутствие таблиц в базе обязано отправлять пользователя в teable_bootstrap,
а не падать на середине заливки.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "teable"))

import teable_bootstrap as bootstrap  # noqa: E402
import teable_common as tc  # noqa: E402
import teable_schema as ts  # noqa: E402
import teable_sync  # noqa: E402

from test_teable_bootstrap_and_cards import FakeTeable  # noqa: E402


class RecordingTeable(FakeTeable):
    """FakeTeable + запись строк, как настоящий клиент при заливке данных."""

    def __init__(self):
        super().__init__()
        self.records: dict[str, list[dict]] = {}

    def _key_of(self, table_id):
        return next(k for k, t in self.state.items() if t["id"] == table_id)

    def all_records(self, table_id, projection=None):
        return [{"id": f"rec{i}", "fields": f}
                for i, f in enumerate(self.records.get(self._key_of(table_id), []))]

    def create_records(self, table_id, rows):
        self.records.setdefault(self._key_of(table_id), []).extend(rows)
        return [{"id": f"rec{i}"} for i, _ in enumerate(rows)]

    def create(self, table_id, fields):
        return self.create_records(table_id, [fields])[0]

    def update(self, table_id, record_id, fields):
        return {"id": record_id}

    def update_records(self, table_id, updates):
        return None

    def upload_attachment(self, table_id, record_id, field_id, filepath):
        return {"id": record_id}


class Args:
    def __init__(self, **kw):
        self.dry_run = kw.get("dry_run", False)
        self.force = kw.get("force", False)
        self.cycles_only = kw.get("cycles_only", False)
        self.no_files = kw.get("no_files", True)


@pytest.fixture
def run_dir(tmp_path) -> Path:
    proc = tmp_path / "processed"
    proc.mkdir()
    pd.DataFrame({
        "cycle_uid": ["CYC-1", "CYC-2"],
        "patient_id": ["E001#20260101", "E002#20260102"],
        "Patient Full Name": ["Пациент Один", "Пациент Два"],
        "Preg": [1, 0], "PRAI": [0.7, 0.3], "DIGITAL TWIN": [0.72, 0.35],
        "ci_low": [0.6, 0.2], "ci_high": [0.8, 0.45],
        "ci_width_pp": [20.0, 25.0], "ci_wide": [False, True],
        "Date OPU": ["2026-01-05", "2026-01-06"], "Date ET": ["2026-01-10", "2026-01-11"],
        "Age": [34, 39], "OCC": [8, 6], "MII": [7, 5], "2pN": [6, 3],
        "Bl": [3, 1], "Good Bl": [2, 0], "N folicules OPU": [10, 8],
        "Embryologist": ["Emb A", "Emb B"], "Doctor": ["Dr A", "Dr B"],
    }).to_csv(proc / "dt_predictions.csv", index=False)
    pd.DataFrame({"cycle_uid": ["CYC-2"], "priority": [1], "reason": ["широкий ДИ"]}
                 ).to_csv(proc / "review_list.csv", index=False)
    return proc


@pytest.fixture
def built_base():
    tb = RecordingTeable()
    bootstrap.apply_schema(tb, "bse1", "Asia/Tashkent")
    return tb


def _clinic(run_dir: Path, base_id="bse1") -> tc.Clinic:
    return tc.Clinic({"code": "test", "name": "Тест", "base_id": base_id,
                      "processed_dir": str(run_dir)}, {})


def _cfg() -> dict:
    return {"cards_url": "http://127.0.0.1:8788", "teable_url": "http://127.0.0.1:3000"}


def test_sync_writes_a_run_and_one_row_per_cycle(built_base, run_dir, capsys):
    assert teable_sync.sync_clinic(_cfg(), built_base, _clinic(run_dir), Args()) is not False
    runs = built_base.records.get("Runs", [])
    cycles = built_base.records.get("Cycles", [])
    assert len(runs) == 1
    assert runs[0]["Пациентов всего"] == 2 and runs[0]["С исходом"] == 2
    assert runs[0]["В ревью-листе"] == 1
    assert len(cycles) == 2
    assert "Parsed: 2 patients" in capsys.readouterr().out


def test_second_sync_of_the_same_results_is_refused_without_force(built_base, run_dir, capsys):
    teable_sync.sync_clinic(_cfg(), built_base, _clinic(run_dir), Args())
    n_runs = len(built_base.records["Runs"])

    assert teable_sync.sync_clinic(_cfg(), built_base, _clinic(run_dir), Args()) is False
    assert "уже в Teable" in capsys.readouterr().out
    assert len(built_base.records["Runs"]) == n_runs, "дубль прогона в базе"


def test_force_reuploads_the_same_results(built_base, run_dir):
    teable_sync.sync_clinic(_cfg(), built_base, _clinic(run_dir), Args())
    teable_sync.sync_clinic(_cfg(), built_base, _clinic(run_dir), Args(force=True))
    assert len(built_base.records["Runs"]) == 2


def test_a_changed_run_is_recognised_as_new(built_base, run_dir):
    teable_sync.sync_clinic(_cfg(), built_base, _clinic(run_dir), Args())
    # Идентичность прогона — хэш содержимого dt_predictions.csv.
    frame = pd.read_csv(run_dir / "dt_predictions.csv")
    frame.loc[0, "PRAI"] = 0.71
    frame.to_csv(run_dir / "dt_predictions.csv", index=False)

    assert teable_sync.sync_clinic(_cfg(), built_base, _clinic(run_dir), Args()) is not False
    assert len(built_base.records["Runs"]) == 2


def test_dry_run_writes_nothing(built_base, run_dir, capsys):
    assert teable_sync.sync_clinic(_cfg(), built_base, _clinic(run_dir),
                                   Args(dry_run=True)) is False
    assert built_base.records == {}
    assert "[dry-run] no data written" in capsys.readouterr().out


def test_sync_sends_the_user_to_bootstrap_when_the_base_is_empty(run_dir, capsys):
    empty = RecordingTeable()
    assert teable_sync.sync_clinic(_cfg(), empty, _clinic(run_dir), Args()) is False
    out = capsys.readouterr().out
    assert "teable_bootstrap.py" in out and "нет таблиц" in out


def test_sync_skips_a_clinic_that_has_not_been_bootstrapped(built_base, run_dir, capsys):
    clinic = _clinic(run_dir, base_id="")
    assert teable_sync.sync_clinic(_cfg(), built_base, clinic, Args()) is False
    assert "нет base_id" in capsys.readouterr().out


def test_cycle_rows_carry_the_review_flag_and_unswapped_dates(built_base, run_dir):
    teable_sync.sync_clinic(_cfg(), built_base, _clinic(run_dir), Args())
    by_id = {c.get("ID пациента"): c for c in built_base.records["Cycles"]}
    assert set(by_id) == {"E001#20260101", "E002#20260102"}
    flagged = by_id["E002#20260102"]
    assert flagged["Приоритет ревью"] == "1 Критично"
    # dayfirst на ISO прочитал бы "2026-01-05" как 1 мая.
    assert by_id["E001#20260101"]["Дата OPU"] == "2026-01-05"


def test_every_field_written_exists_in_the_schema(built_base, run_dir):
    """Незнакомое имя поля Teable отвергает — и заливка падает на середине."""
    teable_sync.sync_clinic(_cfg(), built_base, _clinic(run_dir), Args())
    declared = {t["key"]: {f["name"] for f in t["fields"]} for t in ts.schema()}
    for table_key, rows in built_base.records.items():
        known = declared[table_key]
        for row in rows:
            unknown = set(row) - known
            assert not unknown, f"{table_key}: полей нет в схеме — {sorted(unknown)}"
