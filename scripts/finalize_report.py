"""Локально вставляет текст lead-analyst в уже рассчитанный HTML без передачи PHI агенту."""
from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
import sys

if sys.platform == "win32":
    # Detached/piped stdout on Windows falls back to the ANSI locale codepage
    # (cp1251) instead of the console's actual codepage, garbling Cyrillic output.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embryo._io import atomic_write_text  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="data/processed/DT_Analytics_Report.html")
    parser.add_argument("--text-file", default="data/processed/agent_packets/lead_analysis.md")
    args = parser.parse_args()
    report = Path(args.report)
    text_path = Path(args.text_file)
    html = report.read_text(encoding="utf-8")
    text = text_path.read_text(encoding="utf-8")
    start, end = "<!-- LEAD_START -->", "<!-- LEAD_END -->"
    if start not in html or end not in html:
        raise ValueError("HTML report has no lead-analysis insertion contract")
    replacement = (start + "<div class='narrative' id='lead-analysis'>" +
                   escape(text) + "</div>" + end)
    before, rest = html.split(start, 1)
    _, after = rest.split(end, 1)
    atomic_write_text(before + replacement + after, report)
    print(f"[ok] lead-analyst text inserted into {report}")


if __name__ == "__main__":
    main()
