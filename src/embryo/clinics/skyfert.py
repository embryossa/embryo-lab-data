"""
skyfert.py — ingest/normalize adapter for the Skyfert clinic export
("SkyFert.xlsx", single sheet, single header row). Independent from
src/embryo/clinics/fertimed.py: different raw layout, different quirks, no
code shared beyond the generic ingest/normalize modules every clinic already
shares. Used both by scripts/skyfert_kpi_test.py (fast standalone KPI/OZEMA
smoke test) and by scripts/run_analysis.py --clinic skyfert (full DT/TabPFN +
HTML report run, added 2026-07-21 -- see docs/SKYFERT.md, "Полный прогон").

Reduced scope confirmed by the owner (2026-07-21, see docs/SKYFERT.md): no
MIRI/incubator logs, no consumable-lot tracking, no PGT-A, no
embryologist/doctor breakdown, no date columns at all.

One row = one cycle/attempt, same convention as every other clinic in this
project. "ЭМК" (patient card number) is NOT a reliable unique-patient key on
its own -- the same ЭМК recurs across rows with implausibly different ages
(e.g. 34/42/44 within one ЭМК), so it cannot be trusted to identify one
physical patient across the whole export. cycle_uid therefore hashes
ЭМК + a synthetic per-row sequence (see build_cycle_key), not raw patient_id
alone.
"""
from __future__ import annotations

import pandas as pd

from .. import ingest, normalize


def _follicles_fallback(df: pd.DataFrame) -> pd.Series:
    """Пусто -> occ (врачи часто не вносят количество фолликулов отдельно от
    числа полученных ооцитов), тот же паттерн, что у Fertimed."""
    follicles = pd.to_numeric(df["follicles"], errors="coerce")
    occ = pd.to_numeric(df["occ"], errors="coerce")
    return follicles.where(follicles.notna(), occ)


def _fillna_default(series: pd.Series, default: float) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _clean_method(df: pd.DataFrame) -> pd.Series:
    """"ЭКО" (IVF/ICSI/IVF+ICSI) пусто на ~8% строк -- оставлено как пропуск
    (NaN), не угадывается: реальной третьей категории тут нет, просто пробел
    в заполнении выгрузки."""
    if "fertilization_method" not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="string")
    method = df["fertilization_method"].astype("string").str.strip()
    return method.replace("", pd.NA)


def prepare(raw: pd.DataFrame, columns_cfg: dict, settings_cfg: dict) -> tuple[pd.DataFrame, list]:
    """Резолв колонок + Skyfert-специфичные fallback'ы. Возвращает
    (normalized_df, unknown_columns) -- та же форма, что и другие clinic
    adapters этого проекта."""
    resolved, unknown, missing = ingest.resolve_columns(raw, columns_cfg)
    if missing:
        raise ValueError(f"нет обязательных колонок: {missing}")

    df = resolved.copy()
    df["follicles"] = _follicles_fallback(df)
    df["attempt_no"] = _fillna_default(df["attempt_no"], 1)
    df["transfer_day"] = _fillna_default(df["transfer_day"], 5)
    df["fertilization_method"] = _clean_method(df)

    df = normalize.normalize(df, columns_cfg, settings_cfg)
    return df, unknown


def build_cycle_key(df: pd.DataFrame) -> pd.Series:
    """Ключ для privacy.add_cycle_uid: raw "ЭМК" повторяется у явно разных
    физических пациенток (см. модуль docstring) и нет ни одной колонки даты,
    которой можно было бы обычным способом (ID+Date OPU) отличить их друг от
    друга. Дополняем ЭМК порядковым номером строки ИСХОДНОГО файла --
    df.index, а не позиционный np.arange(len(df)): full-run пайплайн
    (scripts/run_analysis.py --clinic skyfert) вызывает build_cycle_key дважды
    на разных срезах одного и того же normalized (полная lab-популяция и
    только eligible-подмножество transferred>0 & Preg.notna()), и ingest.py
    гарантирует один сквозной RangeIndex, назначенный один раз при загрузке
    (load_table), не сбрасываемый ни resolve_columns, ни normalize.normalize,
    ни kpi.compute_kpi -- поэтому один и тот же физический ряд должен получать
    один и тот же _key независимо от того, на каком срезе вызвана эта функция;
    позиционный arange этого не гарантирует (пересчитывает 1..N внутри среза)."""
    patient = df["patient_id"].astype("string").str.strip().str.upper()
    row_seq = df.index.to_series().astype("string")
    return patient + "#" + row_seq
