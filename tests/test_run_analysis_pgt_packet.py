from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_analysis import _pgt_event_map  # noqa: E402
from embryo.pgt import EMBRYO_COLUMNS  # noqa: E402


def test_pgt_event_map_drops_lab_sample_id_and_uses_ordinal():
    # "KUC4"/"KUC5" mimic the genetics lab's own embryo code, which embeds a
    # surname fragment (e.g. "Kuchkorova") — this must never reach the agent
    # packet (see run_analysis._pgt_event_map docstring).
    events = pd.DataFrame([
        {"cycle_uid": "CYC-1", "sample_id": "18307-KUC4", "category": "euploid",
         "karyotype": "46,XX", "cnv_qc_status": "PASS", "overall_noise": 0.19,
         "sdi": 0.77, "sd_robust": 0.20},
        {"cycle_uid": "CYC-1", "sample_id": "18308-KUC5", "category": "aneuploid",
         "karyotype": "69,XXY,multiple", "cnv_qc_status": "PASS", "overall_noise": 0.17,
         "sdi": 1.53, "sd_robust": 0.18},
    ], columns=EMBRYO_COLUMNS)

    event_map = _pgt_event_map(events)

    assert set(event_map.keys()) == {"CYC-1"}
    records = event_map["CYC-1"]
    assert [r["embryo"] for r in records] == ["embryo_1", "embryo_2"]
    for record in records:
        assert "sample_id" not in record
        joined = " ".join(str(v) for v in record.values())
        assert "KUC" not in joined


def test_pgt_event_map_handles_empty_events():
    assert _pgt_event_map(pd.DataFrame(columns=EMBRYO_COLUMNS)) == {}
