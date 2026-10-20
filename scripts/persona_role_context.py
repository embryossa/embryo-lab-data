"""persona_role_context.py — факты оркестратора для ролей интерпретации прогона PERSONA за год.

Пишет в data/processed/persona/<год>/agent_packets/:
  persona_context.md            статусы компонентов, импутации, правила чтения PERSONA и факты,
                                посчитанные Python (только агрегаты и cycle_uid);
  narrator_input_batch_N.json   срезы patients[] из analysis_bundle.json по 10 циклов для dt-narrator.
Читает только выходы прогона (run_manifest.json, run.log, analysis_bundle.json, dt_predictions.csv)
— роли сами эти файлы не открывают (CLAUDE.md, «Приватность»).

    python -X utf8 scripts/persona_role_context.py <год>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BATCH = 10


def _mean(s) -> str:
    s = pd.to_numeric(s, errors="coerce")
    return f"{s.mean():.3f}" if s.notna().any() else "н/д"


def _preg(x) -> int:
    return int(pd.to_numeric(x["Preg"], errors="coerce").sum())


def build(year: str) -> str:
    root = ROOT / "data" / "processed" / "persona" / year
    ap = root / "agent_packets"
    m = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    b = json.loads((ap / "analysis_bundle.json").read_text(encoding="utf-8"))
    d = pd.read_csv(root / "dt_predictions.csv", low_memory=False)
    log_path = root / "run.log"
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""

    dt = m["digital_twin"]
    tp = dt.get("timepoint_used", {})
    lc = m.get("laya_coverage") or {}
    dc = m.get("dnn_coverage") or {}
    n = m["n_cycles"]
    invalid = re.findall(r"\[dt-api\] (CYC-[0-9A-F]+): invalid (\{.*\})", log)
    kind = d["clinical_probability_kind"].fillna("NA")
    bl = pd.to_numeric(d["Bl"], errors="coerce")
    g0, g1, gna = d[bl.eq(0)], d[bl.gt(0)], d[bl.isna()]
    cur = d[kind.eq("current_cycle")]
    stage_w = pd.to_numeric(d.get("ci_width_pp"), errors="coerce")
    ens_w = pd.to_numeric(d.get("tabpfn_ens_width_pp"), errors="coerce") if "tabpfn_ens_width_pp" in d \
        else pd.Series(dtype=float)
    dnn_status = str(m.get("dnn_status", ""))
    dnn_ok = dnn_status == "valid_saved_scaler" or dnn_status.startswith("ok")
    laya_centre = m.get("laya_centre") or ("PERSONA" if lc.get("scored_centre_calibrated") else None)

    L = [f"# Контекст прогона PERSONA {year} (для ролей интерпретации)", "",
         "Источник — оркестратор (`run_manifest.json`, `run.log`, агрегаты dt_predictions, посчитанные "
         "Python; `docs/PERSONA.md`). Без идентификаторов, кроме cycle_uid. Эти факты главнее общих "
         "правил skill там, где описывают настройку клиники.", "",
         "## Компоненты прогона", "", "| Компонент | Статус | Комментарий |", "|---|---|---|",
         f"| DT 7.1 REST API | ok | точка переноса {tp.get('transfer', 0)}/{n}, retrieval_fallback "
         f"{tp.get('retrieval_fallback', 0)}, без DT {tp.get('none', 0)} |",
         f"| TabPFN (PRAI) | ok | {int(pd.to_numeric(d['PRAI'], errors='coerce').notna().sum())}/{n}; "
         + (f"разброс ансамбля посчитан у {int(ens_w.notna().sum())}, медиана {ens_w.median():.1f} пп"
            if ens_w.notna().any() else "разброс ансамбля в этом прогоне не считался") + " |",
         f"| TabPFN-3 regression | {m.get('regression_status')} | воронка P10–P90 |",
         "| DNN/EVO | " + (f"ok | {dc.get('scored', 0)}/{dc.get('cohort', n)}; вход дозаполнен по правилу DT 7.1 "
                            f"(согласие владельца): {dc.get('model_imputed') or 'без импутаций'}"
                            if dnn_ok else f"мягкий сбой | {dnn_status[:160]}") + " |",
         f"| Laya | {m.get('laya_status')} | оценено {lc.get('scored', 0)}/{lc.get('cohort', n)}, "
         f"непригодны {lc.get('not_assessable', 0)}; сильное расхождение с DT — {lc.get('disagreement_high', 0)}; "
         + (f"калибровка центра {laya_centre} (сдвиг интерсепта, подобран на 2023–2025)" if laya_centre
            else "калибровки центра нет — только ранжирование") + " |",
         "| MIRI, лоты, ПГТ-А | выключены гейтом | у PERSONA этих источников нет |", ""]
    if invalid:
        tpu = d.set_index("cycle_uid")["dt_timepoint_used"].to_dict()
        blu = d.set_index("cycle_uid")["Bl"].to_dict()
        L += ["Отклонения службы DT касаются конкретной точки расчёта: " + "; ".join(
            f"{u} {e} → итоговая точка `{tpu.get(u, '?')}` (Bl = {blu.get(u, '?')})" for u, e in invalid)
            + ". `retrieval_fallback` — DIGITAL TWIN из точки «после пункции», где бластоцисты ещё "
            "неизвестны, не прогноз на день переноса; `none` — DT нет.", ""]
    imp = dt.get("imputed", {})
    L += [f"Импутация в DT (с согласия владельца): AFC = ОКК × 1.16 у {dt.get('afc_source', {}).get('afc_from_occ', 0)}; "
          f"АМГ у {imp.get('amh_imputed', 0)}, ИМТ у {imp.get('bmi_imputed', 0)}, номер попытки у "
          f"{imp.get('attempt_imputed', 0)}; источник спермы выведен у {imp.get('sperm_source_inferred', 0)}; "
          f"ОКК поднят до MII (размороженные ооциты) у {imp.get('okk_raised_for_thawed', 0)} — не ошибка данных.", "",
          "## Как читать данные PERSONA", "",
          "- AFC в выгрузке нет: в DT он оценён из ОКК. У DO-циклов это ОКК донора при возрасте реципиентки — прогноз DT для них условен вдвойне.",
          "- Laya: в `analysis_bundle.json` не выгружается — статус только из таблицы выше. "
          + ("Абсолютные вероятности откалиброваны по центру PERSONA." if laya_centre
             else "Абсолютные вероятности не интерпретировать, только ранжирование."),
          "- Фолликулов нет → OCCR недоступен. «Аномал.опл.» не сопоставлено → 3pNR недоступен.",
          f"- Freeze-all остаётся в таблицах компетентности (решение владельца; токены маски исключения у PERSONA пусты — "
          f"это не противоречие skill vienna-kpi). Lab-популяция — {m['n_lab_population']} строк-блоков; `n_cycles` в "
          f"`kpi_aggregate` — строки, валидные для метрики, поэтому меньше. Когорта DT — {n} свежих переносов с известным исходом.",
          "- Мост TGBDR = (Bl→Good) × BlR нельзя собирать из разных разделов: `conversion` посчитан на когорте DT, "
          "`kpi_aggregate` — на lab-популяции. Расхождение — разные популяции, не ошибка.",
          "- Смешанный ЭКО/ИКСИ в lab-популяции — две строки (блоки), в когорте DT — один цикл.",
          "- Криопереносы в этот прогон не входят (их связь с пункциями — отдельный слой базы, `fet_links.csv`).",
          "- Исход — клиническая беременность по УЗИ (включая внематочную), не живорождение.",
          "- MIRI / лоты / ПГТ-А пусты по гейту — это не «нет событий», а «нет источника».",
          "- `conversion` OCC→MII: «прогноз» — поля DT `med_mii`/`med_okk`; на точке переноса DT получает известные ОКК и MII, "
          "поэтому прогноз этого шага равен факту по построению.",
          "- Причина «ложноположительный (MC>60%, исход отрицательный)» использует MC (`p_per_transfer`), которая в пакет "
          "`patients` не выгружается.",
          "- OZEMA: кандидат на обсуждение генетического консультирования — ≥2 флагов у одной пациентки; не диагноз.",
          f"- `ci_width_pp` / `funnel_unknown_range_pp` — НЕ доверительный интервал: разброс TabPFN по неизвестным стадиям "
          f"воронки, ноль при полной воронке (здесь < 0.1 пп у {int(stage_w.lt(0.1).sum())} из {int(stage_w.notna().sum())}). "
          "Ветка review-list `ci_width_pp>20` срабатывает только при неполной воронке. Неопределённость модели — "
          "`CI_category` (`CI_source=tabpfn_ensemble`, разброс членов ансамбля TabPFN), если посчитана.", "",
          "## Факт оркестратора: два вида вероятности в колонке DIGITAL TWIN", "",
          "- `clinical_probability_kind`: " + ", ".join(f"{k} {v}" for k, v in kind.value_counts().items()) + ".",
          f"- Переносы с Bl = 0: n={len(g0)}, беременностей {_preg(g0)}; средний DIGITAL TWIN {_mean(g0['DIGITAL TWIN'])}, "
          f"PRAI {_mean(g0['PRAI'])}, MC {_mean(g0['p_per_transfer'])}, BEFE {_mean(g0['befe_per_transfer'])}.",
          f"- Переносы с Bl > 0: n={len(g1)}, беременностей {_preg(g1)} (доля {_preg(g1) / max(len(g1), 1):.3f}); "
          f"средний DIGITAL TWIN {_mean(g1['DIGITAL TWIN'])}, PRAI {_mean(g1['PRAI'])} (по циклам с непустым значением).",
          f"- Bl неизвестно: n={len(gna)}" + (f" ({', '.join(gna['cycle_uid'].astype(str))})" if len(gna) else "")
          + f". Всего беременностей {_preg(d)} из {len(d)}.",
          f"- Для `current_cycle` (n={len(cur)}) DIGITAL TWIN — вероятность текущего цикла (без бластоцист 0.0), не на перенос. "
          "Когортное среднее DIGITAL TWIN смешивает две величины; калибровку на перенос сравнивать только по Bl > 0.",
          "- Прогноза DT «на перенос» для переносов без бластоцист нет; MC и BEFE для них посчитаны, но смысл для переноса "
          "на стадии дробления не проверен.",
          "- Дробные числители прогнозной воронки — суммы медианных прогнозов; знаменатели прогноза — только циклы с `med_*`.",
          "- Прогнозная воронка (`med_*`) построена на всех циклах когорты, включая Bl = 0; разбивки по подгруппам в пакете нет.", ""]
    uids = {p["cycle_uid"] for p in b["patients"]}
    zero_rev = sorted(set(d.loc[kind.eq("current_cycle") & d["cycle_uid"].isin(uids), "cycle_uid"]))
    if zero_rev:
        L += [f"- Циклы review-list из группы current_cycle: {', '.join(zero_rev)}.", ""]
    (ap / "persona_context.md").write_text("\n".join(L), encoding="utf-8")

    p = b["patients"]
    for old in ap.glob("narrator_input_batch_*.json"):
        old.unlink()
    for i in range(0, len(p), BATCH):
        (ap / f"narrator_input_batch_{i // BATCH + 1}.json").write_text(
            json.dumps({"patients": p[i:i + BATCH]}, ensure_ascii=False, indent=1), encoding="utf-8")
    return (f"review={len(p)} batches={(len(p) + BATCH - 1) // BATCH} bl0={len(g0)} current_cycle={len(cur)} "
            f"bl_unknown={len(gna)} dnn={'ok' if dnn_ok else 'fail'} laya_centre={laya_centre}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or not sys.argv[1].isdigit():
        sys.exit("usage: persona_role_context.py <год>")
    print(build(sys.argv[1]))
