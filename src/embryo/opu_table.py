"""
opu_table.py — канонический OPU table. На практике OPU table — МАСТЕР-таблица
(лежит в data/inbox/), а не генерируется с нуля; здесь фиксируем точные заголовки
(с реальными пробелами/Ё) и умеем добить недостающие поля из канонизированного df.
"""
from __future__ import annotations
import pandas as pd

OPU_COLUMNS = [
    "#", "Patient Full Name", "DOB", "ID", "Husband Full Name", "DOB 2", "ID 2",
    "cause of infertility", "Addition ", "Date OPU", "Date ET", "BMI", "AMH", "FSH",
    "E2 ", "LG", "P", "# Attempt", "Stimulation, days", "Start FSH dosage", "LH stim",
    "Sum stim dosage", "Trigger day/time", "Trigger dosage dec", "second trigger",
    "second trigger time", "Cetrotide 0.25, days", "second  prep",
    "Antral follicule count", "OPU time", "Age", "N folicules OPU", "OCC", "MII",
    "sperm", "IVF/ICSI", "Inseminated", "2pN", "3pN", "1pN", "0pN", "Deg",
    "Cleavage", "Bl", "Good Bl", "Cryo", "PGT", "ET", "Day of ET", "Embryo grade",
    "Endometrium pattern", "Endometrium on trigger day, mm", "Endometrium on ЕТ, mm",
    "Homocystein", "Preg", "Embryo implanted", "PRAI", "DIGITAL TWIN",
    "Embryologist", "Doctor", "PRAI CI (TabPFN), %", "_key",
]  # PRAI = TabPFN prob; DIGITAL TWIN = DT prob; PRAI CI = TabPFN CI-строка
# "_key" = ID+Date OPU composite (see scripts/run_analysis.py): repeat-attempt
# patients share one "ID", so downstream per-cycle joins (analytics.build_merged)
# need this instead of "ID" for a unique key. Kept as a trailing column here so
# it's visible for tracing without disturbing the established OPU sheet layout.


def ensure_opu_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Гарантирует наличие всех колонок OPU (недостающие -> пустые), порядок канонический."""
    out = df.copy()
    for c in OPU_COLUMNS:
        if c not in out.columns:
            out[c] = pd.NA
    return out[OPU_COLUMNS]


def enrich_predictions(opu: pd.DataFrame, predictions: pd.DataFrame,
                       opu_key: str = "ID", pred_key: str = "patient_id") -> pd.DataFrame:
    """Локально добавляет TabPFN/DT к OPU, сохраняя канонический порядок и PHI внутри ПК."""
    left = opu.copy()
    right = predictions.copy()
    if opu_key not in left or pred_key not in right:
        raise KeyError(f"нельзя обогатить OPU: нужны {opu_key} и {pred_key}")
    left["__join"] = left[opu_key].astype("string").str.strip().str.upper()
    right["__join"] = right[pred_key].astype("string").str.strip().str.upper()
    right = right.drop_duplicates("__join", keep="last")
    keep = ["__join", "PRAI", "DIGITAL TWIN", "PRAI CI (TabPFN), %"]
    left = left.drop(columns=[c for c in keep[1:] if c in left.columns], errors="ignore")
    out = left.merge(right[[c for c in keep if c in right]], on="__join", how="left")
    return ensure_opu_schema(out.drop(columns="__join", errors="ignore"))
