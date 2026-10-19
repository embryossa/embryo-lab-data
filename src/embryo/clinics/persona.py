"""
persona.py — ingest/normalize adapter for the PERSONA clinic exports
("Новый отчет 2023", "Нов. отчет 2024/2025" — one workbook per year, data on the
sheet "нов<год>"). Rules confirmed by the owner 2026-10-19; rationale — docs/PERSONA.md.

Unlike every other clinic, one row of the export is one PROGRAM (ТВП, КРИО ПЕ,
Без ТВП, ...), and the laboratory funnel is split into method blocks:
``кЭКО`` (conventional IVF), ``И`` (ICSI/PICSI), ``DO`` (donor-oocyte recipients),
``ЗО`` (oocyte freezing, no fertilisation) and ``∑`` (= кЭКО + И + DO, verified on
all three years). The adapter therefore works in two steps:

  1. :func:`read_export` + :func:`stage_export` (driven by scripts/persona_stage.py)
     turn the raw sheet into a *staged* table: one row per program, canonical
     scalar columns plus per-block counts (``ivf_occ``, ``icsi_two_pn``, ...).
     Raw headers are resolved only through ``export:`` in
     config/clinics/persona/columns.yaml.
  2. :func:`prepare` / :func:`prepare_wide` read the staged file inside
     scripts/run_analysis.py, the same clinic-adapter contract as Skyfert/GGRC:
       - ``prepare`` — one row per program, funnel = sum of кЭКО/И/DO (report
         cohort for DT/TabPFN; run_analysis gates it by transfer + outcome);
       - ``prepare_wide`` — one row per program *block* (``lab_part`` ivf / icsi /
         donor): a mixed ЭКО/ИКСИ cycle becomes two sub-rows so FR is exact per
         method (owner decision); Σ/Σ rates are unchanged by the split.

``ЗО`` never enters the laboratory population (no fertilisation). ``Аномал.опл.``
is deliberately not mapped (it pools 1PN/3PN; owner decision). There is no AFC or
follicle column at all: ``afc_from_occ`` = own OCC x ``persona.afc_per_occ``
exists only as a Digital Twin input (owner decision, coefficient fitted on
Fertimed — see settings.yaml), never as a laboratory count.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .. import ingest, normalize, privacy

LAB_PARTS = ("ivf", "icsi", "donor")      # блоки, входящие в лабораторную воронку
OWN_OOCYTE_PARTS = ("ivf", "icsi", "oocyte_freeze")
FUNNEL_FIELDS = ("occ", "mii", "two_pn", "degrade", "cleavage_d3", "bl_total", "bl_good", "frozen")
# Колонки контекста программы, которые staging пишет сознательно (не "неизвестные"):
# в KPI/DT не входят, нужны для базы пациентов, карточек и разбора исходов.
CONTEXT_COLUMNS = {
    "program_date", "year", "sheet_year", "has_lab", "lab_part", "program", "status",
    "fertilization_method_raw", "embryologist_opu", "doctor_program", "oocyte_source_raw", "cai",
    "et_or_cancel", "termination_reason", "hcg_result", "us2_result", "heartbeat_us3",
    "attempt_in_clinic", "miri", "time_lapse", "cook", "surrogate", "donor_sperm", "transferred_fet",
    "amh_imputed", "bmi_imputed", "freeze_all",
}


def _clean_header(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _text(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NA, index=frame.index, dtype="string")
    s = frame[column].astype("string").str.strip()
    return s.mask(s == "")


# --------------------------------------------------------------------------- 1. raw -> staged

def find_data_sheet(path: str | Path, columns_cfg: dict) -> tuple[str, int]:
    """Имя листа "нов<год>" и год из имени. Ровно один такой лист, иначе ошибка."""
    pattern = re.compile(columns_cfg["export"]["sheet_pattern"])
    sheets = pd.ExcelFile(path, engine="calamine").sheet_names
    matches = [(s, int(m.group(1))) for s in sheets if (m := pattern.match(s.strip()))]
    if len(matches) != 1:
        raise ValueError(f"{Path(path).name}: ожидался ровно один лист по шаблону "
                         f"{pattern.pattern}, найдено {[s for s, _ in matches]}")
    return matches[0]


def read_export(path: str | Path, columns_cfg: dict) -> pd.DataFrame:
    """Строки программ листа "нов<год>" с нормализованными заголовками.

    Строка заголовка ищется по маркеру (в 2025 она вторая), протянутое
    форматирование (~90 тыс. пустых строк) отбрасывается: строка программы —
    та, где заполнена "Категория программы".
    """
    export = columns_cfg["export"]
    sheet, sheet_year = find_data_sheet(path, columns_cfg)
    raw = pd.read_excel(path, sheet_name=sheet, header=None, dtype=object, engine="calamine")
    marker = export["header_marker"]
    header_row = next((i for i in range(min(20, len(raw)))
                       if any(_clean_header(v) == marker for v in raw.iloc[i].values)), None)
    if header_row is None:
        raise ValueError(f"{Path(path).name}: строка заголовка с '{marker}' не найдена")
    headers = [_clean_header(v) for v in raw.iloc[header_row].values]
    blank_rule = export.get("blank_header_before") or {}
    for i, name in enumerate(headers):
        if name == blank_rule.get("column") and i > 0 and headers[i - 1] == "":
            headers[i - 1] = blank_rule["name"]
    frame = raw.iloc[header_row + 1:].copy()
    frame.columns = [h if h else f"<col{i}>" for i, h in enumerate(headers)]
    frame = frame.loc[:, [not c.startswith("<col") for c in frame.columns]]
    programs = _text(frame, marker).notna()
    frame = frame.loc[programs].reset_index(drop=True)
    frame["_sheet_year"] = sheet_year
    return frame


def _block_columns(columns: list[str], columns_cfg: dict) -> dict[str, dict[str, str]]:
    """{блок: {поле: колонка}} по регуляркам ``export.stems``; каждое поле — ровно одна колонка."""
    export = columns_cfg["export"]
    unprefixed = export.get("unprefixed") or {}
    only = export.get("block_fields") or {}
    out: dict[str, dict[str, str]] = {}
    for block, spec in export["blocks"].items():
        fields = only.get(block) or list(export["stems"])
        out[block] = {}
        for field in fields:
            stem = export["stems"][field]
            prefix = "" if field in unprefixed.get(block, []) else re.escape(spec["prefix"]) + " "
            rx = re.compile(f"^{prefix}{stem}$")
            hits = [c for c in columns if rx.match(c)]
            if len(hits) != 1:
                raise ValueError(f"блок {block} ({spec['prefix']}), поле {field}: "
                                 f"ожидалась одна колонка, найдено {hits}")
            out[block][field] = hits[0]
    return out


def _scalar_columns(columns: list[str], columns_cfg: dict) -> dict[str, str]:
    out = {}
    for canon, aliases in columns_cfg["export"]["scalar"].items():
        hits = [c for c in columns if c in aliases]
        if len(hits) > 1:
            raise ValueError(f"{canon}: несколько колонок-кандидатов {hits}")
        if hits:
            out[canon] = hits[0]
    return out


def _parse_decimal(series: pd.Series) -> pd.Series:
    """"1,2" / "1.2" / "1,2 нг" -> 1.2; нечисловое -> пропуск (не 0)."""
    text = series.astype("string").str.replace(",", ".", regex=False)
    value = pd.to_numeric(text.str.extract(r"(\d+(?:\.\d+)?)", expand=False), errors="coerce")
    return value.astype("float64")              # NaN, а не pd.NA: так ждут normalize/kpi


def _patient_id(name: pd.Series, dob: pd.Series) -> pd.Series:
    """Локальный ключ пациентки ФИО|дата рождения (PHI — только local_private).
    ИИН есть лишь в 2025 и в ключ не входит: ключ должен совпадать между годами."""
    n = (name.astype("string").str.upper().str.replace("Ё", "Е", regex=False)
         .str.replace(r"\s+", " ", regex=True).str.strip())
    d = pd.to_datetime(dob, errors="coerce").dt.strftime("%Y-%m-%d").astype("string")
    d = d.fillna(dob.astype("string").str.strip())
    return (n + "|" + d.fillna("NA")).where(n.notna())


def _outcome(frame: pd.DataFrame) -> pd.Series:
    """Клиническая беременность (УЗИ, включая внематочную) -> 1; ХГЧ или УЗИ "отр" -> 0;
    остальное (нет переноса, ХГЧ "пол" без УЗИ) -> пропуск."""
    hcg = _text(frame, "hcg_result").str.lower()
    us2 = _text(frame, "us2_result").str.lower()
    ectopic = _text(frame, "ectopic").str.startswith("1").fillna(False)
    preg = pd.Series(np.nan, index=frame.index, dtype=float)
    preg[hcg.eq("отр").fillna(False)] = 0.0
    preg[us2.eq("отр").fillna(False)] = 0.0
    preg[us2.eq("пол").fillna(False)] = 1.0
    preg[ectopic] = 1.0
    return preg


def stage_export(raw: pd.DataFrame, columns_cfg: dict, settings_cfg: dict) -> pd.DataFrame:
    """Строки программ одного листа -> staged-схема (канонические имена + блоки)."""
    cols = list(raw.columns)
    scalar = _scalar_columns(cols, columns_cfg)
    blocks = _block_columns(cols, columns_cfg)
    src = raw.rename(columns={v: k for k, v in scalar.items()})
    out = pd.DataFrame(index=raw.index)

    out["patient_id"] = _patient_id(_text(src, "patient_name"), src.get("dob", pd.Series(pd.NA, index=src.index)))
    out["patient_name"] = _text(src, "patient_name")
    out["program_date"] = pd.to_datetime(src.get("program_date"), errors="coerce")
    out["year"] = out["program_date"].dt.year.astype("Int64")
    out["sheet_year"] = raw["_sheet_year"]
    for c in ("treatment_type", "program", "status", "fertilization_method_raw", "embryologist",
              "embryologist_opu", "doctor", "doctor_program", "oocyte_source_raw", "cai",
              "et_or_cancel", "termination_reason", "hcg_result", "us2_result", "heartbeat_us3"):
        out[c] = _text(src, c)
    out["age"] = _num(src, "age")
    lo, hi = (settings_cfg.get("persona") or {}).get("bmi_plausible", [12, 70])
    bmi = _num(src, "bmi")
    out["bmi"] = bmi.where(bmi.between(lo, hi))
    out["amh"] = _parse_decimal(src["amh"]) if "amh" in src.columns else np.nan
    out["fsh"] = _parse_decimal(src["fsh"]) if "fsh" in src.columns else np.nan
    attempt = _num(src, "attempt_no")
    out["attempt_no"] = attempt.where(attempt > 0)
    out["attempt_in_clinic"] = _num(src, "attempt_in_clinic")
    for flag in ("miri", "time_lapse", "cook"):
        out[flag] = _num(src, flag).eq(1)
    out["surrogate"] = _text(src, "surrogate_raw").str.upper().str.contains("СМ").fillna(False)
    out["donor_sperm"] = _text(src, "donor_sperm_code").notna()
    tesa = (_text(src, "tesa_material").str.lower().eq("получили").fillna(False)
            | _text(src, "tesa_ivf_cryo").notna())
    sperm = pd.Series(pd.NA, index=src.index, dtype="string")
    sperm[out["donor_sperm"]] = "donor"
    sperm[tesa] = "TESA"
    out["sperm_source"] = sperm

    for block, fields in blocks.items():
        for field, column in fields.items():
            out[f"{block}_{field}"] = pd.to_numeric(raw[column], errors="coerce")
    for part in LAB_PARTS:
        out[f"{part}_present"] = out[f"{part}_occ"].notna()
        out[f"{part}_transferred"] = out[[f"{part}_et_d3_4", f"{part}_et_d5_6"]].sum(axis=1, min_count=1)

    has_lab = out[[f"{p}_present" for p in LAB_PARTS]].any(axis=1)
    out["has_lab"] = has_lab
    # Дата пункции = "Дата Программы" только у программ с лабораторным блоком:
    # у КРИО ПЕ та же колонка — дата переноса (проверено: Дата переноса − Дата
    # Программы = День ПЕ у ~98% свежих ТВП 2024/2025).
    out["date_opu"] = out["program_date"].where(has_lab)
    out["date_et"] = pd.to_datetime(src.get("date_et"), errors="coerce")
    out["transfer_day"] = _num(src, "transfer_day")
    out["transferred_fet"] = _num(src, "transferred_fet")
    out["Preg"] = _outcome(src)

    # Freeze-all (решение владельца 2026-10-19: исключать из таблиц компетентности,
    # как у Fertimed): свежий перенос отменён (ОП) и заморожен хотя бы один эмбрион.
    # Флаг программы, а не причины отмены: причины разнородны, в 2025 у 155 отмен
    # пусты, а лабораторные неудачи ("блок развития", "не оплодотворилось") почти
    # не дают заморозки и остаются в таблицах.
    program_frozen = out[[f"{p}_frozen" for p in LAB_PARTS]].sum(axis=1, min_count=1)
    out["freeze_all"] = out["et_or_cancel"].eq("ОП").fillna(False) & program_frozen.gt(0)

    own_occ = out[[f"{p}_occ" for p in OWN_OOCYTE_PARTS]].sum(axis=1, min_count=1)
    ratio = float((settings_cfg.get("persona") or {})["afc_per_occ"])
    out["afc_from_occ"] = (own_occ * ratio).round().where(own_occ > 0)
    return out


def stage_exports(paths: list[str | Path], columns_cfg: dict, settings_cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Все годовые книги -> одна staged-таблица без межфайловых дублей.

    Программа может попасть в два файла (в файле 2023 — 35 программ января-февраля
    2024, они же есть в файле 2024). Дубль определяется ключом пациентка + дата +
    категория программы; остаётся строка из книги того года, к которому относится
    "Дата Программы". Повторы внутри одного файла не трогаются — их снимает общий
    _dedupe_cycle_keys пайплайна (правило владельца: наиболее полная строка).
    """
    frames = [stage_export(read_export(p, columns_cfg), columns_cfg, settings_cfg) for p in paths]
    staged = pd.concat(frames, ignore_index=True)
    key = (staged["patient_id"].fillna("NA") + "#"
           + staged["program_date"].dt.strftime("%Y%m%d").fillna("NA") + "#"
           + staged["treatment_type"].fillna("NA"))
    own_year = staged["year"].eq(staged["sheet_year"]).fillna(False)
    order = staged.assign(_k=key, _own=~own_year).sort_values(["_k", "_own"], kind="stable")
    cross = order.groupby("_k")["sheet_year"].transform("nunique") > 1
    drop = order.index[cross & order.duplicated("_k", keep="first")]
    report = {"programs_read": int(len(staged)), "cross_file_duplicates_dropped": int(len(drop)),
              "programs_without_date": int(staged["program_date"].isna().sum())}
    staged = staged.drop(index=drop).reset_index(drop=True)
    return staged, report


# --------------------------------------------------------------------------- 2. staged -> pipeline

def _block_method(part: str, columns_cfg: dict | None) -> str | None:
    if columns_cfg and "export" in columns_cfg:
        return columns_cfg["export"]["blocks"][part].get("method")
    return {"ivf": "IVF", "icsi": "ICSI"}.get(part)


def _program_method(staged: pd.DataFrame) -> pd.Series:
    ivf, icsi = staged["ivf_present"].astype(bool), staged["icsi_present"].astype(bool)
    method = pd.Series(pd.NA, index=staged.index, dtype="string")
    method[ivf & ~icsi] = "IVF"
    method[icsi & ~ivf] = "ICSI"
    method[ivf & icsi] = "IVF+ICSI"
    return method


def _oocyte_source(own: pd.Series, donor: pd.Series) -> pd.Series:
    src = pd.Series(pd.NA, index=own.index, dtype="string")
    src[own & ~donor] = "own"
    src[donor & ~own] = "donor"
    src[own & donor] = "own+donor"
    return src


def _booleans(staged: pd.DataFrame) -> pd.DataFrame:
    """Excel хранит флаги как True/False/1/0 — привести к bool без NaN."""
    out = staged.copy()
    for c in [c for c in out.columns if c.endswith("_present")] + ["has_lab"]:
        if c in out.columns:
            out[c] = out[c].map(lambda v: str(v).strip().lower() in ("true", "1", "1.0"))
    return out


def program_frame(staged: pd.DataFrame) -> pd.DataFrame:
    """Одна строка = программа с лабораторным блоком; воронка = сумма кЭКО/И/DO."""
    s = _booleans(staged)
    s = s.loc[s["has_lab"]].copy()
    for field in FUNNEL_FIELDS:
        s[field] = s[[f"{p}_{field}" for p in LAB_PARTS]].apply(pd.to_numeric, errors="coerce").sum(
            axis=1, min_count=1)
    s["transferred"] = s[[f"{p}_transferred" for p in LAB_PARTS]].apply(pd.to_numeric, errors="coerce").sum(
        axis=1, min_count=1)
    # Инсеминированные: ОКК кЭКО + MII ИКСИ. Для блока DO метод не указан -> неизвестно.
    ins = s[["ivf_occ", "icsi_mii"]].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    s["inseminated"] = ins.where(~s["donor_present"])
    s["fertilization_method"] = _program_method(s)
    s["oocyte_source"] = _oocyte_source(s["ivf_present"] | s["icsi_present"], s["donor_present"])
    return s.reset_index(drop=True)


def lab_part_frame(staged: pd.DataFrame, columns_cfg: dict | None = None) -> pd.DataFrame:
    """Одна строка = блок программы (ivf / icsi / donor) — широкая lab-популяция."""
    s = _booleans(staged)
    parts = []
    for part in LAB_PARTS:
        rows = s.loc[s[f"{part}_present"]].copy()
        if rows.empty:
            continue
        for field in FUNNEL_FIELDS:
            rows[field] = pd.to_numeric(rows[f"{part}_{field}"], errors="coerce")
        rows["transferred"] = pd.to_numeric(rows[f"{part}_transferred"], errors="coerce")
        rows["inseminated"] = (rows["occ"] if part == "ivf" else rows["mii"] if part == "icsi"
                               else pd.Series(np.nan, index=rows.index))
        method = _block_method(part, columns_cfg)
        rows["fertilization_method"] = pd.Series(method if method else pd.NA, index=rows.index, dtype="string")
        rows["oocyte_source"] = "donor" if part == "donor" else "own"
        rows["lab_part"] = part
        parts.append(rows)
    if not parts:
        return pd.DataFrame(columns=list(s.columns) + ["lab_part"])
    out = pd.concat(parts)
    order = {p: i for i, p in enumerate(LAB_PARTS)}
    out = out.assign(_o=out["lab_part"].map(order)).rename_axis("_row").sort_values(["_row", "_o"], kind="stable")
    return out.drop(columns="_o").reset_index(drop=True)


def _resolve(frame: pd.DataFrame, columns_cfg: dict) -> tuple[pd.DataFrame, list]:
    """Проверка обязательных канонических колонок staged-файла. Блочные колонки
    (ivf_occ, ...) — структурные, в unknown не попадают."""
    canonical = set(columns_cfg["canonical"])
    _resolved, _unknown, missing = ingest.resolve_columns(frame[[c for c in frame.columns if c in canonical]],
                                                          columns_cfg)
    if missing:
        raise ValueError(f"нет обязательных колонок (PERSONA): {missing}")
    structural = re.compile(rf"^({'|'.join(LAB_PARTS + ('oocyte_freeze', 'total'))})_")
    known = canonical | CONTEXT_COLUMNS
    unknown = [c for c in frame.columns if c not in known and not structural.match(str(c))]
    return frame, unknown


def prepare(raw: pd.DataFrame, columns_cfg: dict, settings_cfg: dict) -> tuple[pd.DataFrame, list]:
    """Staged-файл -> отчётная когорта (одна строка на программу). Гейт по переносу
    и исходу применяет run_analysis._eligible_cycles, как у остальных клиник."""
    df, unknown = _resolve(program_frame(raw), columns_cfg)
    df = normalize.impute_amh(df, settings_cfg)
    df = normalize.impute_bmi(df, settings_cfg)
    return df, unknown


def prepare_wide(raw: pd.DataFrame, columns_cfg: dict, settings_cfg: dict) -> tuple[pd.DataFrame, list]:
    """Staged-файл -> широкая lab-популяция (строка на блок программы, без гейта исхода)."""
    df, unknown = _resolve(lab_part_frame(raw, columns_cfg), columns_cfg)
    if df.empty:
        raise ValueError("в staged-файле PERSONA нет программ с лабораторным блоком")
    df = normalize.impute_amh(df, settings_cfg)
    df = normalize.impute_bmi(df, settings_cfg)
    return df, unknown


# --------------------------------------------------------------------------- 3. demo export

DEMO_COLUMNS = {
    "Cycle ID": "cycle_ref", "Date OPU": "date_opu", "IVF/ICSI": "fertilization_method",
    "OCC": "occ", "MII": "mii", "2PN": "two_pn", "Deg": "degrade", "Cleavage": "cleavage_d3",
    "Bl": "bl_total", "Good Bl": "bl_good", "Cryo": "frozen", "ET": "transferred",
    "Embryologist": "embryologist", "Doctor": "doctor", "Age": "age", "AMH": "amh", "BMI": "bmi",
    "# Attempt": "attempt_no", "Oocyte source": "oocyte_source_demo", "Sperm Source": "sperm_source",
}


def demo_export(staged: pd.DataFrame, secret: bytes, columns_cfg: dict | None = None) -> pd.DataFrame:
    """Обезличенный вход ivf-lab-kpi-ews-demo: строка на блок программы, имена колонок —
    синонимы config/demo/column_aliases.yaml демо-проекта. Единственный идентификатор —
    HMAC (пациентка + дата пункции + блок); ФИО и дата рождения не выходят. АМГ/ИМТ —
    наблюдённые значения, без импутации."""
    parts = lab_part_frame(staged, columns_cfg)
    ref = (parts["patient_id"].astype("string") + "#"
           + pd.to_datetime(parts["date_opu"], errors="coerce").dt.strftime("%Y%m%d").fillna("NA")
           + "#" + parts["lab_part"])
    parts["cycle_ref"] = ref.map(lambda v: privacy.make_cycle_uid(v, secret))
    parts["oocyte_source_demo"] = parts["oocyte_source"].map({"donor": "Donor", "own": "Own"})
    parts["date_opu"] = pd.to_datetime(parts["date_opu"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = pd.DataFrame({demo: parts[src] for demo, src in DEMO_COLUMNS.items()})
    return out
