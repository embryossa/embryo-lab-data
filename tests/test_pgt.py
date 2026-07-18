from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

fitz = pytest.importorskip("fitz")

from embryo import pgt


def _write_pdf(path: Path, lines: list[str]) -> Path:
    """Synthesize a minimal single-page PDF whose text extracts one line per line,
    matching the real eMap Cycle Report's PyMuPDF layout (no real patient data)."""
    doc = fitz.open()
    page = doc.new_page()
    y = 50.0
    for line in lines:
        if y > 780:
            page = doc.new_page()
            y = 50.0
        page.insert_text((50, y), line, fontsize=10)
        y += 12.0
    doc.save(path)
    doc.close()
    return path


def _header(subject_id: str, report_date: str, pgt_run_id: str = "1") -> list[str]:
    return [
        "Subject information.",
        f"Subject ID: {subject_id}",
        "Last name: TEST",
        "First name: SUBJECT",
        "Date of Birth: 1990-01-01",
        "Sample summary",
        f"Cycle: {pgt_run_id}",
        f"Date: {report_date}",
    ]


def _embryo_block(sample_id: str, summary_result: str, karyotype: str,
                  overall_noise: float, sdi: float, sd_robust: float,
                  software_version: str = "2.1.2", regions: list[tuple] | None = None) -> list[str]:
    lines = [
        "Sample detail",
        f"Sample ID: {sample_id}",
        f"Internal ID: {sample_id}-INT",
        "Sample type: Trophectoderm",
        "Unique pool ID: 1",
        f"Summary result: {summary_result}",
        f"Karyotype: {karyotype}",
        "CNV QC Status: PASS",
        f"Overall noise: {overall_noise}",
        "Total reads: 1,000,000",
        "Filtered CNV Reads: 900,000",
        f"Step-wise noise (SDI): {sdi}",
        f"SD Robust: {sd_robust}",
        "Sign-off comment: (mojibake, unused for classification)",
        "Regions called",
        "Chromosome", "Start", "End", "Length", "Start band", "End band",
        "Type", "Copy-number call", "Call type", "Confidence",
    ]
    for chrom, gain_loss, cn_call in (regions or []):
        lines += [chrom, "Start", "End", "10.0 MB", "p1", "p2", gain_loss, cn_call, "automatic", "1.00"]
    lines += [
        "Analysis settings",
        f"Software version: {software_version}",
        "Analysis version: 1.0.0",
        "Thresholds (Gain): 2.7/2.5/2.5",
        "Thresholds (Loss): 1.5/1.5/1.3",
        "Normalisation: EmbryoMap NextSeq v1.3",
        "Hide Sex: No",
        "Workflow Version: EmbryoMap NextSeq v1.0",
    ]
    return lines


def test_normalize_name_collapses_whitespace_and_case():
    assert pgt.normalize_name("Turabekova  Surayyoxon") == "TURABEKOVASURAYYOXON"
    assert pgt.normalize_name(" turabekova surayyoxon ") == "TURABEKOVASURAYYOXON"
    assert pgt.normalize_name(None) == ""
    assert pgt.normalize_name(float("nan")) == ""


def test_parse_report_reads_euploid_and_aneuploid_embryos(tmp_path):
    lines = (
        _header("TESTOVATESTOVNA", "2026-01-01")
        + _embryo_block("1-A1", "No abnormalities detected", "46,XY", 0.16, 0.61, 0.18)
        + _embryo_block("1-A2", "Aneuploid", "45,XY,sseq[hg38](21)x1", 0.13, 0.82, 0.16,
                        regions=[("chr21", "Loss", "1.05")])
    )
    pdf = _write_pdf(tmp_path / "r.pdf", lines)
    report = pgt.parse_report(pdf)

    assert report["subject_id"] == "TESTOVATESTOVNA"
    assert report["report_date"] == "2026-01-01"
    assert len(report["embryos"]) == 2

    euploid, aneuploid = report["embryos"]
    assert euploid["category"] == "euploid"
    assert euploid["overall_noise"] == 0.16
    assert euploid["cnv_qc_status"] == "PASS"
    assert aneuploid["category"] == "aneuploid"
    assert aneuploid["regions"] == [{
        "chromosome": "chr21", "start": "Start", "end": "End", "length": "10.0 MB",
        "start_band": "p1", "end_band": "p2", "type": "Loss", "copy_number_call": "1.05",
        "call_type": "automatic", "confidence": "1.00",
    }]


def test_summarize_embryos_restricts_noise_stats_to_euploid_subset():
    embryos = [
        {"category": "euploid", "overall_noise": 0.10, "sdi": 0.5, "sd_robust": 0.1,
         "cnv_qc_status": "PASS", "software_version": "2.1.2", "analysis_version": "1.0",
         "thresholds_gain": "g", "thresholds_loss": "l", "normalisation": "n", "regions": []},
        {"category": "euploid", "overall_noise": 0.20, "sdi": 0.7, "sd_robust": 0.2,
         "cnv_qc_status": "PASS", "software_version": "2.1.2", "analysis_version": "1.0",
         "thresholds_gain": "g", "thresholds_loss": "l", "normalisation": "n", "regions": []},
        {"category": "aneuploid", "overall_noise": 0.90, "sdi": 2.0, "sd_robust": 0.9,
         "cnv_qc_status": "PASS", "software_version": "2.1.2", "analysis_version": "1.0",
         "thresholds_gain": "g", "thresholds_loss": "l", "normalisation": "n",
         "regions": [{"chromosome": "chr21", "type": "Loss"}]},
    ]
    summary = pgt.summarize_embryos(embryos)

    assert summary["n_biopsied"] == 3
    assert summary["n_euploid"] == 2
    assert summary["n_aneuploid"] == 1
    assert summary["euploid_rate"] == pytest.approx(round(2 / 3, 4))
    assert summary["qc_pass_rate"] == 1.0
    # High-noise aneuploid embryo must not pollute the euploid-only noise stats.
    assert summary["mean_overall_noise_euploid"] == pytest.approx(0.15)
    assert summary["max_overall_noise_euploid"] == pytest.approx(0.20)
    assert summary["mean_overall_noise_all"] == pytest.approx((0.10 + 0.20 + 0.90) / 3)
    assert summary["analysis_settings_variable"] is False
    assert summary["aneuploid_chromosomes"] == {"chr21": 1}


def test_summarize_embryos_flags_variable_analysis_settings():
    embryos = [
        {"category": "euploid", "overall_noise": 0.1, "sdi": 0.5, "sd_robust": 0.1,
         "cnv_qc_status": "PASS", "software_version": "2.1.2", "analysis_version": "1.0",
         "thresholds_gain": "g", "thresholds_loss": "l", "normalisation": "n", "regions": []},
        {"category": "euploid", "overall_noise": 0.1, "sdi": 0.5, "sd_robust": 0.1,
         "cnv_qc_status": "PASS", "software_version": "2.2.0", "analysis_version": "1.0",
         "thresholds_gain": "g", "thresholds_loss": "l", "normalisation": "n", "regions": []},
    ]
    assert pgt.summarize_embryos(embryos)["analysis_settings_variable"] is True


def test_summarize_embryos_handles_empty_cohort():
    summary = pgt.summarize_embryos([])
    assert summary["n_biopsied"] == 0
    assert summary["euploid_rate"] is None
    assert summary["mean_overall_noise_euploid"] is None


def _cohort(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_analyze_reports_matches_single_candidate_by_name(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    lines = (
        _header("PATIENTAONE", "2026-02-10")
        + _embryo_block("1-A1", "No abnormalities detected", "46,XX", 0.15, 0.5, 0.1)
    )
    _write_pdf(reports_dir / "results_1.pdf", lines)

    cohort = _cohort([
        {"cycle_uid": "CYC-1", "Patient Full Name": "Patient Aone",
         "Date OPU": "2026-02-01", "Date ET": "2026-02-06"},
    ])
    result = pgt.analyze_reports(cohort, reports_dir)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["cycle_uid"] == "CYC-1"
    assert row["n_biopsied"] == 1
    assert row["n_euploid"] == 1
    assert row["match_reason"] == "unique_name_match"
    assert not result.attrs["events"].empty
    assert result.attrs["unmatched"] == []


def test_analyze_reports_disambiguates_repeat_attempts_by_report_date(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    lines = (
        _header("REPEATPATIENT", "2026-05-05")
        + _embryo_block("2-A1", "Aneuploid", "45,X", 0.30, 1.0, 0.3)
    )
    _write_pdf(reports_dir / "results_2.pdf", lines)

    cohort = _cohort([
        {"cycle_uid": "CYC-OLD", "Patient Full Name": "Repeat Patient",
         "Date OPU": "2026-01-01", "Date ET": "2026-01-06"},
        {"cycle_uid": "CYC-NEW", "Patient Full Name": "Repeat Patient",
         "Date OPU": "2026-05-01", "Date ET": "2026-05-06"},
    ])
    result = pgt.analyze_reports(cohort, reports_dir)

    matched = result[result["n_biopsied"] > 0]
    assert len(matched) == 1
    assert matched.iloc[0]["cycle_uid"] == "CYC-NEW"
    assert matched.iloc[0]["match_reason"] == "matched_by_report_date"
    # The other attempt for the same patient stays in the result with no report.
    other = result[result["cycle_uid"] == "CYC-OLD"].iloc[0]
    assert other["pgt_note"] == "нет скачанного отчёта ПГТ"


def test_analyze_reports_leaves_ambiguous_match_unresolved(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    lines = (
        _header("AMBIGUOUSPATIENT", "2026-09-01")
        + _embryo_block("3-A1", "No abnormalities detected", "46,XX", 0.15, 0.5, 0.1)
    )
    _write_pdf(reports_dir / "results_3.pdf", lines)

    cohort = _cohort([
        {"cycle_uid": "CYC-A", "Patient Full Name": "Ambiguous Patient",
         "Date OPU": "2026-01-01", "Date ET": "2026-01-06"},
        {"cycle_uid": "CYC-B", "Patient Full Name": "Ambiguous Patient",
         "Date OPU": "2026-06-01", "Date ET": "2026-06-06"},
    ])
    result = pgt.analyze_reports(cohort, reports_dir)

    assert all(row["n_biopsied"] == 0 for _, row in result.iterrows())
    assert len(result.attrs["unmatched"]) == 1
    assert result.attrs["unmatched"][0]["file"] == "results_3.pdf"


def test_analyze_reports_reports_no_name_match(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    lines = _header("NOBODYINCOHORT", "2026-01-01") + _embryo_block(
        "4-A1", "No abnormalities detected", "46,XX", 0.15, 0.5, 0.1)
    _write_pdf(reports_dir / "results_4.pdf", lines)

    cohort = _cohort([
        {"cycle_uid": "CYC-1", "Patient Full Name": "Someone Else",
         "Date OPU": "2026-01-01", "Date ET": "2026-01-06"},
    ])
    result = pgt.analyze_reports(cohort, reports_dir)

    assert result.attrs["unmatched"] == [{"file": "results_4.pdf", "reason": "no_name_match"}]
    assert result.iloc[0]["pgt_note"] == "нет скачанного отчёта ПГТ"


def test_analyze_reports_requires_cycle_uid_and_name_columns(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    with pytest.raises(KeyError):
        pgt.analyze_reports(_cohort([{"Patient Full Name": "X"}]), reports_dir)
    with pytest.raises(KeyError):
        pgt.analyze_reports(_cohort([{"cycle_uid": "CYC-1"}]), reports_dir)
