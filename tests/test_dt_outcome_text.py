"""Текст исхода, уходящий во внешний Digital Twin (run_analysis._add_dt_aliases).

Почему это отдельный тест: внешний DT молча ВЫБРАСЫВАЕТ строку, чей исход он не
распознал, а не падает. Пока мы отдавали "Нет", в отчёт попадали только
беременности — дефект прожил несколько прогонов обеих клиник и виден лишь по
числу строк. Здесь воспроизведён разбор внешнего проекта, чтобы регрессия
поймалась тестом, а не следующим отчётом.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_run_analysis", ROOT / "scripts/run_analysis.py")
ra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ra)


def _dt_binarize_outcome(val):
    """Дословная копия dt_postprocess.binarize_outcome внешнего проекта DT.

    Скопирована намеренно: внешний проект не импортируется этим репозиторием, а
    контракт между ними — именно текст в колонке "Исход переноса". Если DT
    изменит разбор, этот тест перестанет отражать реальность — тогда копию надо
    обновить, а не удалить.
    """
    if pd.isna(val) or str(val).strip() in ("", "0", "нет", "Нет"):
        return np.nan
    s = str(val).lower().strip()
    if "не наступила" in s:
        return 0
    if "беременность" in s or "клиническая" in s or "биохимическая" in s:
        return 1
    if s in ("да", "yes", "1"):
        return 1
    if s in ("нет", "no"):
        return 0
    return np.nan


def test_both_outcomes_survive_the_external_dt_parser():
    for preg, text in ra.DT_OUTCOME_TEXT.items():
        parsed = _dt_binarize_outcome(text)
        assert not pd.isna(parsed), f"DT потеряет исход {preg}: {text!r} -> nan"
        assert parsed == preg, f"DT прочитал {text!r} как {parsed}, ожидалось {preg}"


def test_plain_no_would_be_dropped_by_the_external_dt():
    """Фиксирует саму причину дефекта — чтобы правку не «упростили» обратно."""
    assert pd.isna(_dt_binarize_outcome("Нет"))
    assert _dt_binarize_outcome("Да") == 1


def test_dt_frame_carries_the_parseable_text_for_failures_too():
    frame = pd.DataFrame({"Preg": [1.0, 0.0, np.nan], "_key": ["A", "B", "C"]})
    out = ra._add_dt_aliases(frame)
    assert out["Исход переноса"].tolist() == [
        "Беременность +28 дней", "Беременность не наступила", ""]
    parsed = [_dt_binarize_outcome(v) for v in out["Исход переноса"]]
    assert parsed[0] == 1 and parsed[1] == 0 and pd.isna(parsed[2])


def test_unknown_outcome_stays_empty_and_is_not_invented():
    """Пустой исход должен остаться пустым: DT сам решит его пропустить."""
    out = ra._add_dt_aliases(pd.DataFrame({"Preg": [np.nan], "_key": ["A"]}))
    assert out["Исход переноса"].iloc[0] == ""


def test_our_own_config_still_reads_both_texts_correctly():
    """Тот же текст возвращается к нам через columns.yaml — он обязан бинаризоваться."""
    from embryo import ingest, normalize

    cfg = ingest.load_config(str(ROOT / "config/columns.yaml"))
    texts = pd.Series(list(ra.DT_OUTCOME_TEXT.values()))
    assert normalize.outcome_to_binary(texts, cfg["outcome"]).tolist() == [1.0, 0.0]
