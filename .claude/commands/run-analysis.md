---
description: Одной командой — авто-прогон DT/TabPFN, все расчёты, форензик и итоговая интерпретация
---
Полностью автоматический разбор по таблице из data/inbox (или $ARGUMENTS). Пользователь
участвует только этой командой — всё остальное делается само.

Следуй skill `run-analysis`.

1. Запусти авто-оркестратор: `python scripts/run_analysis.py $ARGUMENTS`
   Он САМ: берёт таблицу из data/inbox → АВТО-запускает Digital Twin и TabPFN (их командами)
   и читает свежие результаты → (опц.) DNN-проверку → KPI(Вена) → джойн DT↔OPU → ci_wide →
   аналитику (калибровка/конверсии/приоритеты/review-list, DT_Analytics_Report.xlsx) →
   форензик MIRI (вложенные логи, окно Date OPU..+6 дней) → HTML-отчёт (A/B/C/D).
   Если DT/TabPFN не запустились — сообщи пользователю про run_cmd/пути в settings и не выдумывай числа.
2. Интерпретация поверх готовых данных:
   - `kpi-analyst`: сверка KPI с LCL/Веной.
   - `model-auditor`: проверка согласования DT/TabPFN и CI.
   - `dt-narrator`: пер-пациентные разборы по `cycle_uid` для review-list ∪ ci_wide.
   - `incubator-forensics`: интерпретация инцидентов MIRI (гипотезы, не факты).
   - (опц.) `dnn-verifier`, если включён.
   Все агенты читают только `data/processed/agent_packets/`.
3. `lead-analyst` (Opus, свободно): итоговый текст всего разбора → передай его локальному
   Python для обратного join. Сохрани полученный текст в
   `data/processed/agent_packets/lead_analysis.md`, затем выполни:
   `python scripts/finalize_report.py`. Это атомарно вставит текст в локальный HTML,
   где `cycle_uid` уже сопоставлен с OPU без передачи ФИО/ID агенту.
