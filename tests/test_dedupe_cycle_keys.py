"""Дедупликация повторяющегося "_key" (scripts/run_analysis.py::_dedupe_cycle_keys).

Правило владельца (2026-07-31): при повторе ключа остаётся наиболее полная строка.
Проверяется то, чем ошибка здесь опасна: выбор не той строки теряет счётчики
воронки, а расхождение решения между лабораторной и модельной когортами молча
разъезжает join прогнозов с исходами.
"""
import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_run_analysis", ROOT / "scripts/run_analysis.py")
ra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ra)


def _pair() -> pd.DataFrame:
    """Реальный случай июньской выгрузки: один цикл двумя строками разной полноты."""
    return pd.DataFrame({
        "_key": ["A#1#2", "A#1#2", "B#3#4"],
        "occ": [12.0, None, 8.0],
        "two_pn": [6.0, 6.0, 4.0],
        "transferred": [1.0, 1.0, 1.0],
    })


def test_keeps_the_more_complete_row():
    out, mask = ra._dedupe_cycle_keys(_pair(), "тест")
    assert len(out) == 2
    assert out.loc[out["_key"] == "A#1#2", "occ"].iloc[0] == 12.0   # пустая строка снята
    assert list(mask) == [True, False, True]


def test_frames_without_duplicates_are_untouched():
    frame = pd.DataFrame({"_key": ["A", "B"], "occ": [1.0, 2.0]})
    out, mask = ra._dedupe_cycle_keys(frame, "тест")
    assert out.equals(frame)
    assert mask.all()


def test_tie_keeps_the_first_row_deterministically():
    """При равной полноте порядок выгрузки решает — иначе прогоны разъедутся."""
    frame = pd.DataFrame({"_key": ["A", "A"], "occ": [10.0, 20.0]})
    out, _ = ra._dedupe_cycle_keys(frame, "тест")
    assert out["occ"].tolist() == [10.0]
    # и это устойчиво при повторном вызове
    again, _ = ra._dedupe_cycle_keys(frame, "тест")
    assert again["occ"].tolist() == [10.0]


def test_missing_key_column_is_not_an_error():
    frame = pd.DataFrame({"occ": [1.0, 2.0]})
    out, mask = ra._dedupe_cycle_keys(frame, "тест")
    assert len(out) == 2 and mask.all()


def test_mask_transfers_the_same_decision_to_a_filled_model_frame():
    """Ядро правила: в модельном кадре пропуски заполнены, полнота уже другая.

    Если бы решение пересчитывалось там заново, выиграла бы вторая строка пары —
    и когорты перестали бы совпадать построчно.
    """
    lab = _pair()
    lab["mii"] = [None, None, 6.0]      # у обеих строк пары пропуск
    model = lab.copy()
    # model-подстановки заполняют пропуски только там, где они были: вторая строка
    # пары получает и ОКК, и MII, первая — ничего, и становится "менее полной".
    model.loc[1, "occ"] = 99.0
    model.loc[1, "mii"] = 5.0
    _, mask = ra._dedupe_cycle_keys(lab, "лаб")
    positional = model.loc[mask.to_numpy()]
    assert positional["occ"].tolist() == [12.0, 8.0]     # решение то же, что в лаборатории
    recomputed, _ = ra._dedupe_cycle_keys(model, "модель")
    assert recomputed["occ"].tolist() == [99.0, 8.0]     # пересчёт дал бы ДРУГУЮ строку


def test_every_frame_that_reaches_a_key_contract_is_deduped():
    """Кадр OPU читается отдельным путём и на первом прогоне 31.07.2026 остался
    без дедупликации — `reconcile.build_dt_predictions` упал на неуникальном
    `_join` уже ПОСЛЕ того, как отработали внешние DT и TabPFN. Три места, где
    строится ключ, должны сниматься от дублей все три.
    """
    source = (ROOT / "scripts/run_analysis.py").read_text(encoding="utf-8")
    # каждый вызов add_cycle_uid и возврат _prepare_opu должен быть прикрыт
    assert source.count("_dedupe_cycle_keys(") >= 4          # определение + 3 применения
    prepare_opu = source.split("def _prepare_opu(")[1].split("\ndef ")[0]
    assert "_dedupe_cycle_keys(" in prepare_opu


def test_all_duplicate_keys_collapse_to_one_row_each():
    frame = pd.DataFrame({
        "_key": ["A", "A", "A", "B", "B"],
        "occ": [None, 5.0, None, 1.0, None],
        "mii": [None, 4.0, 3.0, None, 2.0],
    })
    out, _ = ra._dedupe_cycle_keys(frame, "тест")
    assert out["_key"].tolist() == ["A", "B"]
    assert out.set_index("_key").loc["A", "occ"] == 5.0
    assert not out["_key"].duplicated().any()
