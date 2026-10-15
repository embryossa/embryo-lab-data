#!/usr/bin/env python3
"""laya_batch.py — батч-адаптер Laya (laya_kan_ft) для embryo-lab-data.

Запускается в окружении проекта Laya (её .venv: свой torch), из
`embryo.laya_runner`. Вызывает тот же `service/laya_ivf.assess`, что и MCP
`laya_ivf`, поэтому guard'ы (обязательные входы, «перенос невозможен»),
правила restore и калибровка по центру одинаковы в MCP и в пайплайне.

Вход — CSV только с cycle_uid и лабораторными счётчиками (без ID/ФИО/дат) плюс
DIGITAL TWIN этого прогона. Выход — CSV по cycle_uid:
  Laya                    вероятность после калибровки по центру (пусто, если цикл
                          непригоден или центр не задан)
  laya_raw                сырая вероятность (пусто у непригодного цикла)
  laya_assessable         пригоден ли цикл для Laya
  laya_minus_dt_pp        Laya − DT, пп, обе после калибровки по центру
  laya_disagreement_high  |logit|-разрыв выше порога RBMO — повод перепроверить
  laya_flags              флаги Laya через « | »
  laya_restored           поля, восстановленные по правилам (restore)
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

# колонки dt_predictions.csv -> имена, которые понимает laya_ivf (как в RBMO-оценке Laya)
INPUTS = ["Age", "Attempt", "N folicules OPU", "OCC", "MII", "Inseminated", "2pN",
          "real_cleav", "Bl", "Good Bl", "real_cryo"]


def _val(v):
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--service-dir", required=True, help="папка Laya/service с laya_ivf.py")
    ap.add_argument("--centre", default=None, help="Fertimed | GGRC | SkyFert | Two-Line")
    ap.add_argument("--restore", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, str(Path(a.service_dir).resolve()))
    import laya_ivf as L  # noqa: E402 - путь сервиса известен только во время запуска

    frame = pd.read_csv(a.input, encoding="utf-8-sig")
    cycles = []
    for _, r in frame.iterrows():
        c = {"cycle_uid": str(r["cycle_uid"]),
             "inputs": {k: _val(r.get(k)) for k in INPUTS if k in frame.columns}}
        dt = _val(r.get("DIGITAL TWIN"))
        if dt is not None:
            c["dt_probability"] = dt
        cycles.append(c)
    res = L.assess(cycles, centre=a.centre or None, restore=a.restore)
    rows = []
    for row in res["results"]:
        ok = bool(row["assessable"])
        gap = row.get("difference_laya_minus_dt")
        rows.append({
            "cycle_uid": row["cycle_uid"],
            "Laya": row["laya_probability_centre_calibrated"] if ok else None,
            "laya_raw": row["laya_probability"] if ok else None,
            "laya_assessable": ok,
            "laya_minus_dt_pp": None if gap is None else round(100 * gap, 1),
            "laya_disagreement_high": row.get("disagreement_high"),
            "laya_flags": " | ".join(row["flags"]) or None,
            "laya_restored": ";".join(row["restored"]) or None,
        })
    out = pd.DataFrame(rows)
    out.to_csv(a.output, index=False, encoding="utf-8-sig")
    print(f"[laya] {len(out)} циклов, пригодны {int(out['laya_assessable'].sum())}, "
          f"центр={a.centre or '—'}, backend={res['backend']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
