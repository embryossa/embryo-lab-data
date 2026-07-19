"""Parsing and cohort-level forensics for Fertimed's PGT-A reports.

Reports are downloaded manually from the genetics lab's own portal
(``my.f-genetics.com``) into a local folder and matched back to our cycles by
patient name — unlike clinic A's eMap Cycle Report (pa.ivf.uz), this vendor's
PDF carries the patient's real full name directly (``Пациент:``), not an
opaque slug ``Subject ID``, so matching uses that name field straight away.

Format is a compact one/two-page result letter, not a structured lab-system
export: a header block of ``Label \n value`` pairs followed by a table of
per-embryo rows (``T1``, ``T2``, ... sample ids) with free-text karyotype
(ISCN 2020), interpretation and transfer recommendation — no numeric CNV QC
metrics (noise/SDI/SD Robust), no software/analysis-settings fields, so the
richer cohort features clinic A's parser computes from those (QC noise,
analysis-settings variability) are simply not derivable here and are left as
``None``. Classification (euploid/aneuploid/mosaic) therefore comes from the
Russian free-text "Интерпретация" cell rather than an English summary field.

Matching, parsing and feature extraction all stay local; only the per-cycle
summary (:func:`analyze_reports`) — keyed by ``cycle_uid``, no name/DOB — is
safe to fold into an agent packet.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

import pandas as pd

from ..pgt import _lines, _pick_cycle_uid, normalize_name  # noqa: F401 (re-exported for callers)

EMBRYO_COLUMNS = [
    "cycle_uid", "sample_id", "category", "karyotype", "cnv_qc_status",
    "overall_noise", "sdi", "sd_robust", "total_reads", "filtered_cnv_reads",
    "software_version", "workflow_version", "interpretation", "recommendation",
    "recommendation_category",
]

_HEADER_LABELS = {
    "ID пациента в организации": "id_org",
    "ID пациента в лаборатории": "id_lab",
    "Пациент": "patient_raw",
    "Партнер": "partner",
    "Дата биопсии": "biopsy_dates_raw",
    "Дата получения": "received_dates_raw",
    "Дата выдачи": "issued_dates_raw",
    "Эмбриолог": "embryologist_raw",
    "Организация, город": "organization_raw",
    "Показания": "indications_raw",
}
_TABLE_MARKER = "Результат ПГТ-А"
_SAMPLE_ID_RE = re.compile(r"T\d+")
_NAME_SUFFIX_RE = re.compile(r"\s*\([^)]*\)\s*$")


def _split_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_header(lines: list[str]) -> tuple[dict, int]:
    table_idx = next(
        (i for i, line in enumerate(lines) if line.strip().startswith(_TABLE_MARKER)),
        len(lines),
    )
    fields: dict[str, str] = {}
    current_key: str | None = None
    current_vals: list[str] = []

    def flush():
        if current_key is not None:
            fields[current_key] = " ".join(v.strip() for v in current_vals).strip()

    for line in lines[:table_idx]:
        s = line.strip()
        if s == "Результат":
            continue
        if s in _HEADER_LABELS:
            flush()
            current_key = _HEADER_LABELS[s]
            current_vals = []
            continue
        current_vals.append(s)
    flush()

    patient_raw = fields.get("patient_raw", "")
    header = {
        "id_org": fields.get("id_org"),
        "id_lab": fields.get("id_lab"),
        "patient_name": _NAME_SUFFIX_RE.sub("", patient_raw).strip(),
        "patient_name_raw": patient_raw,
        "biopsy_dates": _split_list(fields.get("biopsy_dates_raw")),
        "received_dates": _split_list(fields.get("received_dates_raw")),
        "issued_dates": _split_list(fields.get("issued_dates_raw")),
        "embryologists": _split_list(fields.get("embryologist_raw")),
        "organization": fields.get("organization_raw"),
        "indications": _split_list(fields.get("indications_raw")),
    }
    return header, table_idx


def _is_recommendation_start(line: str) -> bool:
    s = line.strip()
    return s.startswith("Рекомендован") or s.startswith("Не рекомендован")


def _classify_recommendation(raw: str | None) -> str | None:
    if not raw:
        return None
    r = raw.strip()
    if r.startswith("Не рекомендован"):
        return "not_recommended"
    if r.startswith("Рекомендован с"):
        return "recommended_with_consent"
    if r.startswith("Рекомендован"):
        return "recommended"
    return "unknown"


def _classify_interpretation(text: str | None) -> str:
    if not text:
        return "no_result"
    t = text.strip().strip(".").lower()
    if t == "норма":
        return "euploid"
    if "мозаич" in t:
        return "mosaic"
    return "aneuploid"


def _parse_embryo_chunk(sample_id: str, chunk: list[str]) -> tuple[dict, list[str]]:
    """One ``T<n>`` row: karyotype (ISCN, wraps onto continuation lines that
    start with '(') then free-text interpretation then a recommendation drawn
    from a closed set of phrases, one of which itself wraps onto 2 more
    lines. Returns the embryo dict plus any trailing lines the chunk didn't
    need (a case-level note following the last row, if any)."""
    if not chunk:
        return {"sample_id": sample_id, "category": "not_tested", "karyotype": None,
                "interpretation": None, "recommendation": None,
                "recommendation_category": None}, []
    if len(chunk) <= 2 and any("на хранении" in c.lower() for c in chunk):
        return {"sample_id": sample_id, "category": "not_tested", "karyotype": None,
                "interpretation": " ".join(chunk), "recommendation": None,
                "recommendation_category": None}, []

    i = 1
    karyotype_lines = [chunk[0]]
    while i < len(chunk) and chunk[i].startswith("("):
        karyotype_lines.append(chunk[i])
        i += 1
    karyotype = " ".join(karyotype_lines)

    interpretation_lines = []
    while i < len(chunk) and not _is_recommendation_start(chunk[i]):
        interpretation_lines.append(chunk[i])
        i += 1
    interpretation = " ".join(interpretation_lines).strip()

    recommendation_raw = None
    if i < len(chunk):
        first_line = chunk[i]
        recommendation_lines = [first_line]
        i += 1
        if first_line not in ("Рекомендован", "Не рекомендован"):
            while i < len(chunk) and len(recommendation_lines) < 3:
                recommendation_lines.append(chunk[i])
                i += 1
        recommendation_raw = " ".join(recommendation_lines).strip()

    embryo = {
        "sample_id": sample_id,
        "karyotype": karyotype,
        "interpretation": interpretation,
        "category": _classify_interpretation(interpretation),
        "recommendation": recommendation_raw,
        "recommendation_category": _classify_recommendation(recommendation_raw),
    }
    return embryo, chunk[i:]


def _parse_embryo_table(lines: list[str], table_idx: int) -> tuple[list[dict], str | None]:
    rest = [line.strip() for line in lines[table_idx + 1:]]
    marker_positions = [i for i, line in enumerate(rest) if _SAMPLE_ID_RE.fullmatch(line)]
    if not marker_positions:
        return [], None
    bounds = list(zip(marker_positions, marker_positions[1:] + [len(rest)]))
    embryos = []
    case_note = None
    for idx, (start, end) in enumerate(bounds):
        sample_id = rest[start]
        chunk = [line for line in rest[start + 1:end] if line]
        embryo, leftover = _parse_embryo_chunk(sample_id, chunk)
        embryos.append(embryo)
        if idx == len(bounds) - 1 and leftover:
            case_note = " ".join(leftover).strip() or None
    return embryos, case_note


def parse_report(pdf_path: str | Path) -> dict:
    """Parse one f-genetics.com result-letter PDF into header + per-embryo rows."""
    raw_lines = _lines(pdf_path)
    lines = [
        line for line in raw_lines
        if line.strip() != "f-genetics.com"
        and not line.strip().startswith("Тестирование выполнено")
        and not line.strip().startswith("*")
    ]
    header, table_idx = _parse_header(lines)
    embryos, case_note = _parse_embryo_table(lines, table_idx)
    header["case_note"] = case_note
    header["embryos"] = embryos
    return header


def summarize_embryos(embryos: list[dict]) -> dict:
    """Cohort-level features for one report. Unlike clinic A's eMap Cycle
    Report, f-genetics.com's letter carries no CNV QC noise metrics or
    analysis-settings fields — those keys are kept (as ``None``) only so the
    output shares clinic A's column shape; euploid_rate and the transfer-
    recommendation counts are the only features actually available here."""
    n = len(embryos)
    by_category = Counter(e["category"] for e in embryos)
    n_not_tested = by_category.get("not_tested", 0)
    n_tested = n - n_not_tested
    by_recommendation = Counter(
        e["recommendation_category"] for e in embryos if e.get("recommendation_category")
    )
    return {
        "n_biopsied": n,
        "n_tested": n_tested,
        "n_not_tested": n_not_tested,
        "n_euploid": by_category.get("euploid", 0),
        "n_aneuploid": by_category.get("aneuploid", 0),
        "n_mosaic": by_category.get("mosaic", 0),
        "n_other": by_category.get("no_result", 0),
        "euploid_rate": round(by_category.get("euploid", 0) / n_tested, 4) if n_tested else None,
        "n_recommended": by_recommendation.get("recommended", 0),
        "n_recommended_with_consent": by_recommendation.get("recommended_with_consent", 0),
        "n_not_recommended": by_recommendation.get("not_recommended", 0),
        # Не предоставляется этим отчётом (в отличие от eMap Cycle Report клиники A):
        "n_qc_pass": None, "qc_pass_rate": None, "software_versions": [],
        "analysis_settings_variable": None, "aneuploid_chromosomes": {},
        "mean_overall_noise_euploid": None, "max_overall_noise_euploid": None,
        "mean_sdi_euploid": None, "max_sdi_euploid": None,
        "mean_sd_robust_euploid": None, "max_sd_robust_euploid": None,
        "mean_overall_noise_all": None, "max_overall_noise_all": None,
    }


def analyze_reports(
    cohort: pd.DataFrame,
    reports_dir: str | Path,
    cycle_uid_col: str = "cycle_uid",
    name_col: str = "Patient Full Name",
    date_col_opu: str = "Date OPU",
    date_col_et: str = "Date ET",
) -> pd.DataFrame:
    """Match downloaded f-genetics.com PDFs to ``cycle_uid`` by patient name
    and summarize each cohort. Same repeat-attempt disambiguation as clinic
    A's :func:`embryo.pgt.analyze_reports` (report date inside that attempt's
    ``Date OPU..Date ET`` window, else left unresolved) — reused directly
    from there via :func:`embryo.pgt._pick_cycle_uid`."""
    if cycle_uid_col not in cohort.columns:
        raise KeyError(f"нет обязательной псевдонимизированной колонки {cycle_uid_col!r}")
    if name_col not in cohort.columns:
        raise KeyError(f"нет колонки имени для локального сопоставления: {name_col!r}")

    by_name: dict[str, list[dict]] = {}
    for _, row in cohort.iterrows():
        key = normalize_name(row[name_col])
        if not key:
            continue
        by_name.setdefault(key, []).append(row.to_dict())

    summaries: list[dict] = []
    all_events: list[pd.DataFrame] = []
    unmatched: list[dict] = []
    matched_uids: set[str] = set()

    for pdf_path in sorted(Path(reports_dir).glob("*.pdf")):
        try:
            report = parse_report(pdf_path)
        except Exception as exc:  # pragma: no cover - defensive, mirrors clinic A's try/except at call site
            unmatched.append({"file": pdf_path.name, "reason": f"parse_error:{type(exc).__name__}"})
            continue
        key = normalize_name(report["patient_name"])
        candidates = by_name.get(key, [])
        if not candidates:
            unmatched.append({"file": pdf_path.name, "reason": "no_name_match"})
            continue
        report_date = None
        issued = [pd.to_datetime(d, format="%d.%m.%Y", errors="coerce") for d in report["issued_dates"]]
        issued = [d for d in issued if pd.notna(d)]
        if issued:
            report_date = max(issued)
        cycle_uid, match_reason = _pick_cycle_uid(candidates, report_date, date_col_opu, date_col_et)
        if cycle_uid is None:
            unmatched.append({"file": pdf_path.name, "reason": match_reason,
                              "n_candidates": len(candidates)})
            continue
        matched_uids.add(cycle_uid)
        summary = {
            "cycle_uid": cycle_uid,
            "report_date": report_date,
            "pgt_run_id": report.get("id_lab"),
            "match_reason": match_reason,
            "case_note": report.get("case_note"),
            "embryologists": report.get("embryologists"),
            **summarize_embryos(report["embryos"]),
        }
        summaries.append(summary)
        events = pd.DataFrame([
            {
                "cycle_uid": cycle_uid, "sample_id": e.get("sample_id"),
                "category": e.get("category"), "karyotype": e.get("karyotype"),
                "cnv_qc_status": None, "overall_noise": None, "sdi": None, "sd_robust": None,
                "total_reads": None, "filtered_cnv_reads": None,
                "software_version": None, "workflow_version": None,
                "interpretation": e.get("interpretation"),
                "recommendation": e.get("recommendation"),
                "recommendation_category": e.get("recommendation_category"),
            }
            for e in report["embryos"]
        ], columns=EMBRYO_COLUMNS)
        if not events.empty:
            all_events.append(events)

    for key, candidates in by_name.items():
        for c in candidates:
            if c[cycle_uid_col] not in matched_uids:
                summaries.append({"cycle_uid": c[cycle_uid_col], "n_biopsied": 0,
                                  "pgt_note": "нет скачанного отчёта ПГТ (f-genetics.com)"})

    result = pd.DataFrame(summaries)
    result.attrs["events"] = (
        pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame(columns=EMBRYO_COLUMNS)
    )
    result.attrs["unmatched"] = unmatched
    return result
