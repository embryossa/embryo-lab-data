---
name: ozema
description: OZEMA oocyte-to-embryo arrest flags — per-cycle binomial significance test, thresholds, genetic-counseling recurrence criterion. Preload into agents that interpret OZEMA output (lead-analyst).
---
# OZEMA — арест-паттерны цикла

Реализация: `src/embryo/ozema.py`. Методология и пороги перенесены дословно с листа
OZEMA мастер-таблицы OPU владельца (Google Drive), которая цитирует Van Der Kelen et
al., Hum Reprod 2026, и использует Vienna Consensus-style биномиальный тест (p<0,05).

## Флаги (по стадиям, каждый — отдельный односторонний точный биномиальный тест)

- **OMA** (Oocyte maturation arrest) — MII/OCC значимо ниже 75%.
- **FF** (Fertilization failure) — 2PN/MII значимо ниже 60% (искл. TESE/SMF из
  `sperm_source`).
- **ZA** (Zygote arrest) — дробящиеся/2PN значимо ниже 95%.
- **EEA** (Early embryo arrest) — Bl/2PN значимо ниже 40%.

Флаг ставится не просто когда наблюдаемая доля НИЖЕ порога, а когда это статистически
значимо (`scipy.stats.binomtest(k, n, p0, alternative="less")`, α=0.05) — при малом n
почти любое отклонение недостаточно значимо, поэтому:

- **n < 4** (`ozema.MIN_N`) → `"н/д"` — недостаточно наблюдений, стадию оценить нельзя.
  Это не «пропуск», не «ноль флагов» — прямо покажи как «нельзя оценить». (Намеренно не
  `"n/a"`: это значение — в списке `na_values` по умолчанию у `pandas.read_csv` и молча
  превратилось бы в `NaN` при повторном чтении CSV.)
- Ни одна стадия не флагована, но ≥1 тестируема → `"—"` — в пределах ожиданий.
- Несколько стадий флагованы одновременно → код через `+` (например `"OMA+ZA"`) —
  смешанный фенотип.

## Популяция

OZEMA считается на ШИРОКОЙ lab-стадийной популяции — БЕЗ гейта
transferred>0/Preg известен и БЕЗ exclusion-набора таблиц компетентности эмбриолог/врач
(open-cycle/NO-retrieval/freeze-all). Freeze-all (Cryo) цикл с 2PN=0 может и должен
получить флаг FF — это подтверждено самим листом-источником. Единственное собственное
исключение OZEMA — TESE/SMF из `sperm_source`, и только для стадии FF.

## Критерий генетического консультирования

Пациент (raw `patient_id` — повторные попытки делят один ID) с **≥2 флагованными
циклами** — кандидат на ОБСУЖДЕНИЕ направления на генетическую консультацию. Это НЕ
диагноз и не автоматическое назначение — формулируй как повод для клинического
обсуждения. Единичный флагованный цикл — просто наблюдение («recurrence would meet
OZEMA criteria»), не критерий сам по себе.

## Приватность

`ozema_cycles.csv`/`ozema_genetic_counseling_candidates.csv` несут raw `patient_id` и
`patient_name` — это ЛОКАЛЬНЫЕ артефакты (как OPU/review-list), не agent packet. В
пакет агентам попадают только агрегированные счётчики (`ozema_flagged_cycles`,
`ozema_genetic_counseling_candidate_patients`), без идентификаторов.
