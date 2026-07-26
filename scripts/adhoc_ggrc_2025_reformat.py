"""
adhoc_ggrc_2025_reformat.py — ONE-TIME standalone reformatter for the two new
GGRC 2025 exports ("предсказания 2025.xlsx" / "KPI 2025.xlsx"), which carry
the same *meaning* as the established "ivf_master_13 (transfer).xlsx" /
"ivf_master_13 kpi.xlsx" but a slightly different raw header layout (per
owner instruction 2026-07-25: do not touch the production clinic adapter
src/embryo/clinics/ggrc.py or config/clinics/ggrc/columns.yaml for this --
write a separate one-off script instead).

Differences handled here (verified against the real files):
  - "предсказания 2025.xlsx" (561 rows, transfer-file equivalent) spans
    2020-2025; only the 502 rows with OPU Date in 2025 are kept (matches the
    user's "for 2025" scope). All 502 already satisfy the standard pipeline
    eligibility filter (transferred>0 & known Pregnancy outcome).
  - "KPI 2025.xlsx" (1376 rows, kpi-wide-file equivalent) carries NO "OPU
    Date" column at all -- its OPU Number range (207373-209323) sits inside
    the transfer file's 2025 OPU Number range (207395-208989, 493/502
    overlap), consistent with the file already being a 2025-only export by
    construction (its filename), so no date-based row filter is applied to
    it (none possible/needed). One fully-blank padding row is dropped.
    CRITICAL: a missing "OPU Date" is not just a cosmetic gap -- scripts/
    run_analysis.py's ``_add_opu_aliases`` builds the cross-file join key
    "_key" as ``patient_id + "#" + date_opu`` (analytics._restore_true_counts
    matches df_u rows back to this wide file by that same key to pull
    verified raw counts). With OPU Date entirely absent, EVERY wide-file row
    collapses to the same "{OPU Code}#NA" key, which never matches the
    transfer file's real-dated keys -- silently NaNing out Age/OCC/MII/etc.
    across virtually the whole cohort (discovered via a downstream
    ``int(nan)`` crash in report.py's similar-cases graph, 2026-07-25 run).
    Fixed here by backfilling "OPU Date" via a join on "OPU Code" against
    the transfer file (which does carry real dates) -- 493/498 eligible
    2025 cycles recover a matching date this way; the remaining 5 genuinely
    do not appear in the wide file at all (real data gap, not an adapter
    bug -- those 5 correctly keep NaN "true counts", matching this
    pipeline's documented "genuinely missing -> NaN" semantics).
  - Column header renames (case-insensitive alias matching in
    ingest.resolve_columns already handles most naming drift, e.g. "Sperm
    source" vs "Sperm Source" -- only genuinely different strings need a
    rename here):
      "OPU Number"   -> "OPU Code"    (both files; "OPU Number.1" in the
                                        transfer file is a verified exact
                                        duplicate of "OPU Number", dropped)
      "fertilization" -> "Fertilization Method"  (both files, where present)
  - Neither new file carries the embedded PGT-A count columns (Biopsy Done/
    Tested Embryos/Euploid/...) that config/clinics/ggrc/columns.yaml's
    PGT_RAW_COLUMNS expects -- confirmed absent, not a bug (per owner:
    "информации по ПГТ тут нет"). ggrc.analyze_reports already degrades
    gracefully (n_biopsied=None -> pgt_note "нет данных ПГТ-А" per row) when
    these columns are missing, so nothing to patch here.
  - "предсказания 2025.xlsx" additionally carries a second block of already-
    computed Russian-named KPI/prediction columns (from a prior/foreign run
    of this data, e.g. "KPIScore", "Predictions", "Частота оплодотворения")
    -- these are dropped so they cannot be confused with this pipeline's own
    freshly computed numbers (CLAUDE.md: "числа не пересчитывать в LLM").
  - Real "number of follicles" data present in the transfer file is
    intentionally NOT threaded through: docs/GGRC.md records the owner's
    2026-07-23 decision to keep using the round(Oocytes N * 1.44) estimate
    for GGRC regardless (ggrc.py's `follicles` is not a resolvable column at
    all), so this script does not touch that behaviour.

Output: two standardized .xlsx files under data/inbox/ggrc/_2025_adapted/,
column layout matching the existing production files 1:1 (columns absent in
source added as NaN) so the untouched ggrc.prepare()/prepare_wide() resolve
them exactly as they resolve the originals. Run once, then point
run_analysis.py at the output (positional `input` arg for the transfer file,
config/clinics/ggrc/settings.local.yaml override for kpi_wide_file).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_TRANSFER = ROOT / "data/inbox/ggrc/transfer/предсказания 2025.xlsx"
IN_KPI_WIDE = ROOT / "data/inbox/ggrc/kpi_wide/KPI 2025.xlsx"
OUT_DIR = ROOT / "data/inbox/ggrc/_2025_adapted"

# Exact column layout of the existing production files (docs/GGRC.md /
# config/clinics/ggrc/columns.yaml source of truth) minus blank Excel
# spacer columns ("Unnamed: N") -- reproduced here so the adapted output has
# an identical, predictable shape without editing the production schema.
TRANSFER_TARGET_COLUMNS = [
    "OPU Code", "Doctor", "Embryologist IVF/ICSI", "Embryologist denudation",
    "Embryologist OPU", "OPU Date", "Patients name", "Age", "Husband name",
    "Oocyte Source", "Sperm Source", "Fertilization Method", "Oocytes N",
    "MII", "MI", "GV", "Abnormal, degr., etc.", "2pN", "1pN", "3pN or more",
    "Degrade", "Day 5", "Day 6-7", "number of embryos dividing on day 3",
    "number of inseminated oocytes", "CombinedDays", "Good quality BL",
    "description", "Attempts", "Pregnancy", "Number of transfered embryos",
    "Transfer: Surrogacy Program", "Transfer: Surrogate Mother",
    "Transfer: Doctor", "Transfer: Embryologist Vitrification",
    "Transfer: Embryologist Thawing", "Transfer: Embryologist Transfer",
    "Transfer: Morphology", "Transfer: Notes after Thawing",
    "Cryo: Date of OPU", "Cryo: Day 5 Embryos", "Cryo: Day 6 Embryos",
    "Cryo: Day 7 Embryos", "Cryo: Other Blastocysts",
    "Cryo: Vitrification Dates", "Biopsy Done", "Tested Embryos",
    "Not Tested Embryos", "Euploid", "Mosaic", "Failed/NA",
    "Pending Result", "Aneuploid", "Embryologist on Biopsy",
]

KPI_WIDE_TARGET_COLUMNS = [
    "OPU Code", "Doctor", "Embryologist IVF/ICSI", "Embryologist denudation",
    "Embryologist OPU", "Patients name", "Age", "Husband name",
    "Oocyte Source", "Sperm Source", "Fertilization Method", "Oocytes N",
    "MII", "MI", "GV", "Abnormal, degr., etc.", "2pN", "1pN", "3pN or more",
    "Degrade", "Day 5", "Day 6-7", "D5-D6-D7", "Good quality BL",
    "description", "Transfer: Attempt Quantity", "Transfer: Overall Pregnancies",
    "Transfer: Surrogacy Program", "Transfer: Surrogate Mother",
    "Transfer: Doctor", "Transfer: Embryologist Vitrification",
    "Transfer: Embryologist Thawing", "Transfer: Embryologist Transfer",
    "Transfer: Embryos Transferred", "Transfer: Morphology",
    "Transfer: Notes after Thawing", "OPU Date", "Cryo: Date of OPU",
    "Cryo: Day 5 Embryos", "Cryo: Day 6 Embryos", "Cryo: Day 7 Embryos",
    "Cryo: Other Blastocysts", "Cryo: Vitrification Dates", "Biopsy Done",
    "Tested Embryos", "Not Tested Embryos", "Euploid", "Mosaic",
    "Failed/NA", "Pending Result", "Aneuploid", "Embryologist on Biopsy",
]


def _rename_common(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    if "OPU Number" in df.columns:
        rename["OPU Number"] = "OPU Code"
    if "fertilization" in df.columns:
        rename["fertilization"] = "Fertilization Method"
    return df.rename(columns=rename)


def build_transfer_2025() -> pd.DataFrame:
    df = pd.read_excel(IN_TRANSFER, sheet_name="Sheet1")
    assert (df["OPU Number"] == df["OPU Number.1"]).all(), \
        "OPU Number.1 no longer an exact duplicate of OPU Number -- re-check source file"
    df = df.drop(columns=["OPU Number.1"])
    df = _rename_common(df)
    df["OPU Date"] = pd.to_datetime(df["OPU Date"], errors="coerce")
    before = len(df)
    df = df[df["OPU Date"].dt.year == 2025].copy()
    print(f"[transfer] {before} rows total -> {len(df)} rows with OPU Date in 2025")
    return df.reindex(columns=TRANSFER_TARGET_COLUMNS)


def build_kpi_wide_2025() -> pd.DataFrame:
    df = pd.read_excel(IN_KPI_WIDE, sheet_name="Sheet1")
    before = len(df)
    df = df[df["OPU Number"].notna()].copy()
    print(f"[kpi_wide] {before} rows -> {len(df)} rows after dropping blank padding row(s)")
    df = _rename_common(df)

    # Backfill "OPU Date" from the transfer file (see module docstring --
    # required for cross-file cycle_uid matching downstream, not cosmetic).
    date_source = pd.read_excel(IN_TRANSFER, sheet_name="Sheet1")
    date_map = (date_source.dropna(subset=["OPU Number"])
                .drop_duplicates("OPU Number")
                .set_index("OPU Number")["OPU Date"])
    df["OPU Date"] = df["OPU Code"].map(date_map)
    matched = df["OPU Date"].notna().sum()
    print(f"[kpi_wide] backfilled OPU Date for {matched}/{len(df)} rows from transfer file "
          f"({len(df) - matched} rows have no matching transfer-file cycle -- real data gap)")

    return df.reindex(columns=KPI_WIDE_TARGET_COLUMNS)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    transfer_out = build_transfer_2025()
    transfer_path = OUT_DIR / "transfer_2025.xlsx"
    transfer_out.to_excel(transfer_path, sheet_name="Sheet1", index=False)
    print(f"[write] {transfer_path} ({len(transfer_out)} rows)")

    kpi_wide_out = build_kpi_wide_2025()
    kpi_wide_path = OUT_DIR / "kpi_wide_2025.xlsx"
    kpi_wide_out.to_excel(kpi_wide_path, sheet_name="IVF Master DB", index=False)
    print(f"[write] {kpi_wide_path} ({len(kpi_wide_out)} rows)")


if __name__ == "__main__":
    main()
