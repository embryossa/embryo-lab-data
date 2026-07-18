"""Parsing and cohort-level forensics for PGT-A (eMap Cycle Report) PDFs.

Reports are downloaded manually from the genetics lab's portal (pa.ivf.uz) into a
local folder and matched back to our own cycles by patient name — the portal has no
API and its own "Cycle" number is the lab's internal biopsy-batch id, unrelated to
our ``cycle_uid``. Matching, parsing and feature extraction all stay local; only the
per-cycle summary (:func:`analyze_reports`) — keyed by ``cycle_uid``, no name/DOB —
is safe to fold into an agent packet.

Font caveat: the "Sign-off comment" free text is Cyrillic rendered with a subset font
that has no usable ToUnicode CMap — every extraction library (pdfplumber, PyMuPDF)
returns mojibake for it, not a bug in this module. Classification therefore uses the
English "Summary result" / "CNV Status" field instead, which extracts cleanly and
carries the same conclusion (euploid / aneuploid / mosaic).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import statistics

import pandas as pd

try:
    import fitz  # PyMuPDF
    _HAVE_FITZ = True
except Exception:  # pragma: no cover - exercised only when PyMuPDF is absent
    _HAVE_FITZ = False


REGION_FIELDS = [
    "chromosome", "start", "end", "length", "start_band", "end_band",
    "type", "copy_number_call", "call_type", "confidence",
]
EMBRYO_COLUMNS = [
    "cycle_uid", "sample_id", "category", "karyotype", "cnv_qc_status",
    "overall_noise", "sdi", "sd_robust", "total_reads", "filtered_cnv_reads",
    "software_version", "workflow_version",
]
_HEADER_FIELD = re.compile(r"^([A-Za-z][A-Za-z \-()]*?):\s*(.*)$")


def normalize_name(name: object) -> str:
    """Collapse a patient's full name to the portal's slug form (no spaces, upper)."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    return re.sub(r"\s+", "", str(name)).strip().upper()


def _lines(pdf_path: str | Path) -> list[str]:
    if not _HAVE_FITZ:
        raise RuntimeError("пакет pymupdf не установлен: добавьте pymupdf в окружение")
    with fitz.open(pdf_path) as doc:
        # Each page's text already ends with its own "\n" — joining with an extra
        # "\n" would insert a phantom blank line at every page boundary, which shifts
        # the positional Regions-called row parsing below by one token per boundary.
        text = "".join(page.get_text() for page in doc)
    return [line.rstrip() for line in text.split("\n") if line.strip()]


def _field(lines: list[str], label: str) -> str | None:
    pattern = re.compile(rf"^{re.escape(label)}:\s*(.*)$")
    for line in lines:
        m = pattern.match(line.strip())
        if m:
            return m.group(1).strip() or None
    return None


def _to_number(value: str | None):
    if value is None:
        return None
    cleaned = value.replace(",", "").strip()
    try:
        return float(cleaned) if "." in cleaned else int(cleaned)
    except ValueError:
        return None


def _classify(summary_result: str | None) -> str:
    if not summary_result:
        return "no_result"
    text = summary_result.lower()
    if "mosaic" in text:
        return "mosaic"
    if "no abnormalities" in text:
        return "euploid"
    if "aneuploid" in text:
        return "aneuploid"
    return "other"


def _parse_regions(chunk: list[str]) -> list[dict]:
    try:
        start = chunk.index("Regions called") + 1
    except ValueError:
        return []
    # Skip the 10 header tokens (Chromosome/Start/.../Confidence), one per line.
    start += len(REGION_FIELDS)
    end = chunk.index("Analysis settings") if "Analysis settings" in chunk else len(chunk)
    values = [v for v in chunk[start:end] if v.strip()]
    regions = []
    for i in range(0, len(values) - len(REGION_FIELDS) + 1, len(REGION_FIELDS)):
        row = values[i : i + len(REGION_FIELDS)]
        if len(row) < len(REGION_FIELDS):
            break
        regions.append(dict(zip(REGION_FIELDS, (v.strip() for v in row))))
    return regions


def _parse_embryo_chunk(chunk: list[str]) -> dict:
    summary_result = _field(chunk, "Summary result")
    return {
        "sample_id": _field(chunk, "Sample ID"),
        "internal_id": _field(chunk, "Internal ID"),
        "sample_type": _field(chunk, "Sample type"),
        "unique_pool_id": _field(chunk, "Unique pool ID"),
        "sex": _field(chunk, "Sex"),
        "summary_result": summary_result,
        "category": _classify(summary_result),
        "karyotype": _field(chunk, "Karyotype"),
        "cnv_qc_status": _field(chunk, "CNV QC Status"),
        "overall_noise": _to_number(_field(chunk, "Overall noise")),
        "total_reads": _to_number(_field(chunk, "Total reads")),
        "filtered_cnv_reads": _to_number(_field(chunk, "Filtered CNV Reads")),
        "sdi": _to_number(_field(chunk, "Step-wise noise (SDI)")),
        "sd_robust": _to_number(_field(chunk, "SD Robust")),
        "regions": _parse_regions(chunk),
        "software_version": _field(chunk, "Software version"),
        "analysis_version": _field(chunk, "Analysis version"),
        "smoothing": _field(chunk, "Smoothing"),
        "segmental": _field(chunk, "Segmental"),
        "thresholds_gain": _field(chunk, "Thresholds (Gain)"),
        "thresholds_loss": _field(chunk, "Thresholds (Loss)"),
        "custom_thresholds": _field(chunk, "Custom copy-number thresholds"),
        "normalisation": _field(chunk, "Normalisation"),
        "hide_sex": _field(chunk, "Hide Sex"),
        "telomeric_drop_correction": _field(chunk, "Telomeric drop correction"),
        "workflow_version": _field(chunk, "Workflow Version"),
    }


def parse_report(pdf_path: str | Path) -> dict:
    """Parse one eMap Cycle Report PDF into subject header + per-embryo records."""
    lines = _lines(pdf_path)
    header = {
        "subject_id": _field(lines, "Subject ID"),
        "last_name": _field(lines, "Last name"),
        "first_name": _field(lines, "First name"),
        "dob": _field(lines, "Date of Birth"),
        "report_date": _field(lines, "Date"),
        "pgt_run_id": _field(lines, "Cycle"),
    }
    starts = [i for i, line in enumerate(lines) if line.strip() == "Sample detail"]
    bounds = list(zip(starts, starts[1:] + [len(lines)]))
    embryos = [_parse_embryo_chunk(lines[a:b]) for a, b in bounds]
    header["embryos"] = embryos
    return header


def summarize_embryos(embryos: list[dict]) -> dict:
    """Cohort-level features for one PGT-A report, e.g. the euploid subset's noise."""
    n = len(embryos)
    by_category = Counter(e["category"] for e in embryos)
    euploid = [e for e in embryos if e["category"] == "euploid"]
    qc_pass = [e for e in embryos if (e.get("cnv_qc_status") or "").upper() == "PASS"]

    def _stats(values: list[float], prefix: str) -> dict:
        clean = [v for v in values if v is not None]
        if not clean:
            return {f"mean_{prefix}": None, f"max_{prefix}": None}
        return {f"mean_{prefix}": round(statistics.mean(clean), 4), f"max_{prefix}": round(max(clean), 4)}

    settings_keys = {
        (e.get("software_version"), e.get("analysis_version"), e.get("thresholds_gain"),
         e.get("thresholds_loss"), e.get("normalisation"))
        for e in embryos
    }
    aneuploid_chromosomes = Counter(
        region["chromosome"]
        for e in embryos if e["category"] in {"aneuploid", "mosaic"}
        for region in e.get("regions", [])
    )

    summary = {
        "n_biopsied": n,
        "n_euploid": by_category.get("euploid", 0),
        "n_aneuploid": by_category.get("aneuploid", 0),
        "n_mosaic": by_category.get("mosaic", 0),
        "n_other": by_category.get("other", 0) + by_category.get("no_result", 0),
        "euploid_rate": round(by_category.get("euploid", 0) / n, 4) if n else None,
        "n_qc_pass": len(qc_pass),
        "qc_pass_rate": round(len(qc_pass) / n, 4) if n else None,
        "software_versions": sorted({e["software_version"] for e in embryos if e.get("software_version")}),
        "analysis_settings_variable": len(settings_keys) > 1,
        "aneuploid_chromosomes": dict(aneuploid_chromosomes),
    }
    summary.update(_stats([e["overall_noise"] for e in euploid], "overall_noise_euploid"))
    summary.update(_stats([e["sdi"] for e in euploid], "sdi_euploid"))
    summary.update(_stats([e["sd_robust"] for e in euploid], "sd_robust_euploid"))
    summary.update(_stats([e["overall_noise"] for e in embryos], "overall_noise_all"))
    return summary


def _pick_cycle_uid(candidates: list[dict], report_date, date_col_opu="Date OPU",
                     date_col_et="Date ET", slack_days: int = 30):
    """Disambiguate a name match against a patient's repeat attempts by report date."""
    if len(candidates) == 1:
        return candidates[0]["cycle_uid"], "unique_name_match"
    report_ts = pd.to_datetime(report_date, errors="coerce")
    if pd.isna(report_ts):
        return None, "ambiguous_name_no_report_date"
    scored = []
    for c in candidates:
        opu = pd.to_datetime(c.get(date_col_opu), errors="coerce")
        et = pd.to_datetime(c.get(date_col_et), errors="coerce")
        if pd.isna(opu):
            continue
        window_end = (et if not pd.isna(et) else opu) + pd.Timedelta(days=slack_days)
        if opu - pd.Timedelta(days=1) <= report_ts <= window_end:
            scored.append((abs((report_ts - opu).total_seconds()), c["cycle_uid"]))
    if len(scored) == 1:
        return scored[0][1], "matched_by_report_date"
    if not scored:
        return None, "ambiguous_name_no_date_window_match"
    scored.sort()
    return None, "ambiguous_name_multiple_date_window_matches"


def analyze_reports(
    cohort: pd.DataFrame,
    reports_dir: str | Path,
    cycle_uid_col: str = "cycle_uid",
    name_col: str = "Patient Full Name",
    date_col_opu: str = "Date OPU",
    date_col_et: str = "Date ET",
) -> pd.DataFrame:
    """Match downloaded PGT-A PDFs to ``cycle_uid`` by name and summarize each cohort.

    Filenames carry no patient info (portal names them ``results_<order-id>.pdf``);
    matching is done by the ``Subject ID`` parsed from each report's own text. A name
    with more than one cycle_uid (repeat attempts) is resolved by report date falling
    inside that attempt's ``Date OPU .. Date ET`` window; if that is still ambiguous
    the match is intentionally left unresolved rather than guessed, and surfaced in
    ``result.attrs["unmatched"]``.
    """
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
        report = parse_report(pdf_path)
        key = normalize_name(report["subject_id"])
        candidates = by_name.get(key, [])
        if not candidates:
            unmatched.append({"file": pdf_path.name, "reason": "no_name_match"})
            continue
        cycle_uid, match_reason = _pick_cycle_uid(candidates, report["report_date"], date_col_opu, date_col_et)
        if cycle_uid is None:
            unmatched.append({"file": pdf_path.name, "reason": match_reason,
                              "n_candidates": len(candidates)})
            continue
        matched_uids.add(cycle_uid)
        summary = {
            "cycle_uid": cycle_uid,
            "report_date": report["report_date"],
            "pgt_run_id": report["pgt_run_id"],
            "match_reason": match_reason,
            **summarize_embryos(report["embryos"]),
        }
        summaries.append(summary)
        events = pd.DataFrame([
            {"cycle_uid": cycle_uid, **{k: e.get(k) for k in EMBRYO_COLUMNS if k not in ("cycle_uid",)}}
            for e in report["embryos"]
        ], columns=EMBRYO_COLUMNS)
        if not events.empty:
            all_events.append(events)

    for key, candidates in by_name.items():
        for c in candidates:
            if c[cycle_uid_col] not in matched_uids:
                summaries.append({"cycle_uid": c[cycle_uid_col], "n_biopsied": 0,
                                  "pgt_note": "нет скачанного отчёта ПГТ"})

    result = pd.DataFrame(summaries)
    result.attrs["events"] = (
        pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame(columns=EMBRYO_COLUMNS)
    )
    result.attrs["unmatched"] = unmatched
    return result
