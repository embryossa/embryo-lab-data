---
description: Аналитика клиники PERSONA за год — staging выгрузок, preflight, dt-check, полный прогон DT/TabPFN/Laya, интерпретация и отчёт
argument-hint: <год> [--dt-output <csv> --tabpfn-output <csv>]
---
Прогон аналитики PERSONA за год `$ARGUMENTS` (первое слово — год, например `2024`).
Это `/run-analysis` для PERSONA: следуй skill `run-analysis` с отличиями ниже. Правила
клиники и решения владельца — `docs/PERSONA.md`.

**Перед запуском любого прогона, кроме staging, preflight и `--dt-check`, напиши владельцу,
что именно запускаешь и сколько это займёт, и дождись подтверждения.**

1. **Staging** (дёшево, без PHI в выводе): `python scripts/persona_stage.py`.
   Нужен, если в `data/inbox/PERSONA/` появились или обновились выгрузки или нет
   `data/processed/persona/local_private/staged/persona_<год>.xlsx`. Перескажи из вывода
   только агрегаты: программы, блоки, сверку ∑, когорту DT (перенос + исход).
2. **Preflight**: `python scripts/preflight.py --persona <год>` — должен вернуть 0.
3. **Сухая проверка DT**: `python scripts/run_analysis.py --persona <год> --dt-check`.
   Сообщи владельцу циклы, `dt_coverage`, отклонённые строки с причинами и оценку часов.
   Ожидаемо: `afc_source` = `afc_from_occ` (AFC = ОКК × 1.16, у DO-циклов — ОКК донора),
   `sperm_source_inferred` почти у всех (источник спермы в выгрузке не указан).
4. **Полный прогон после подтверждения** — в фоне с логом:
   `python scripts/run_analysis.py --persona <год> > data/processed/persona/<год>/run.log 2>&1`.
   Выходы — `data/processed/persona/<год>/`; годы друг друга не перезаписывают.
   Replay без повторного DT/TabPFN: добавь `--dt-output data/processed/persona/<год>/local_private/dt71_api_output.csv`
   и `--tabpfn-output` из того же прогона.
5. **Контроль компонентов** (CLAUDE.md): таблица `ok / мягкий сбой + причина / выключен гейтом`
   по `run_manifest.json`. У PERSONA выключены гейтом: MIRI, лоты, ПГТ-А. Laya работает без
   калибровки центра — ожидаемый статус `ok_uncalibrated`, покрытие — `laya_coverage.scored`
   (по `laya_raw`); ноль при включённой Laya — сбой.
6. **Интерпретация** — роли и порядок как в skill `run-analysis`; агенты читают только
   `data/processed/persona/<год>/agent_packets/`. Выходы ролей записываешь туда же.
7. **Финализация**:
   `python scripts/finalize_report.py --report data/processed/persona/<год>/DT_Analytics_Report.html --text-file data/processed/persona/<год>/agent_packets/lead_analysis.md`.

Что держать в голове при интерпретации PERSONA:
- AFC в выгрузке нет: в DT он оценён как ОКК × 1.16 (флаг `afc_source`). У DO-циклов это ОКК
  донора при возрасте реципиентки — прогноз DT для них условен вдвойне.
- Laya: абсолютные вероятности не интерпретировать (на 2023–2025 средний прогноз выше факта
  на 15–16 п.п.), только ранжирование. Restore Laya в пайплайне ставит фолликулы = ОКК —
  это не то же правило, что в демо-анализе Laya (`rint(ОКК/0.846)`); метрики дискриминации
  демо-анализа — `data/processed/persona/laya/PERSONA_Laya_*.md`.
- Фолликулов нет: OCCR недоступен; «Аномал.опл.» не сопоставлено — 3pNR недоступен.
- Freeze-all остаётся в таблицах компетентности (решение владельца): они на всей лабораторной работе.
- Смешанный ЭКО/ИКСИ в широкой lab-популяции — две строки (блоки), в когорте DT — один цикл.
- Криопереносы не связаны с пункциями — их исходы в этот прогон не входят.
