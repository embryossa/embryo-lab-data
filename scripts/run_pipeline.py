"""
run_pipeline.py — оркестратор ДЕТЕРМИНИРОВАННЫХ этапов (ingest -> normalize -> kpi -> filter).
Исходную таблицу берёт из data/inbox/ (последнюю по времени) либо из аргумента.

  python scripts/run_pipeline.py                      # последняя из data/inbox
  python scripts/run_pipeline.py data/sample/x.xlsx   # конкретный файл
"""
from __future__ import annotations
import sys
from pathlib import Path
import yaml

if sys.platform == "win32":
    # Detached/piped stdout on Windows falls back to the ANSI locale codepage
    # (cp1251) instead of the console's actual codepage, garbling Cyrillic output.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embryo import ingest, normalize, kpi  # noqa: E402
from embryo._io import latest_file, atomic_write_csv          # noqa: E402


def load_settings():
    with open("config/settings.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    local = Path("config/settings.local.yaml")   # машинно-зависимые переопределения
    if local.exists():
        with open(local, encoding="utf-8") as f:
            _deep_merge(cfg, yaml.safe_load(f) or {})
    return cfg


def _deep_merge(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def main(input_path: str | None):
    columns_cfg = ingest.load_config("config/columns.yaml")
    settings_cfg = load_settings()

    if input_path is None:
        inbox = settings_cfg["paths"]["inbox_dir"]
        f = (latest_file(inbox, "*.xlsx") or latest_file(inbox, "*.csv"))
        if f is None:
            print(f"[!] В {inbox} нет входных файлов (.xlsx/.csv)."); sys.exit(1)
        input_path = str(f)
    print(f"[in] {input_path}")

    raw = ingest.load_table(input_path)
    df, unknown, missing = ingest.resolve_columns(raw, columns_cfg)
    if missing:
        print(f"[!] Отсутствуют обязательные колонки: {missing}")

    df = normalize.normalize(df, columns_cfg, settings_cfg)
    df = kpi.compute_kpi(df)
    if "Preg" in df.columns:                          # §5.4: перенос был И исход заполнен
        df = df[df["Preg"].notna()]

    rep = kpi.validate_finite(df)
    bad = {c: r for c, r in rep.items()
           if r["inf"] or (r["nan"] and not r.get("optional", False))}
    print(f"[hygiene] NaN/inf в KPI: {bad if bad else 'нет'}")

    out = Path(settings_cfg["paths"]["processed_dir"]) / "kpi_computed.csv"
    atomic_write_csv(df, out)
    print(f"[ok] {len(df)} строк -> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
