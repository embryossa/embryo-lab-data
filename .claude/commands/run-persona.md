---
description: Аналитика клиники PERSONA за год — staging выгрузок, preflight, dt-check, полный прогон DT/TabPFN/DNN/Laya, интерпретация, отчёт, связь криопереносов, Teable и карточки
argument-hint: <год> [--dt-output <csv> --tabpfn-output <csv>]
---
Прогон аналитики PERSONA за год `$ARGUMENTS` (первое слово — год, например `2026`).
Это `/run-analysis` для PERSONA: следуй skill `run-analysis` с отличиями ниже. Правила
клиники и решения владельца — `docs/PERSONA.md`.

**Перед запуском любого прогона, кроме staging, preflight и `--dt-check`, напиши владельцу,
что именно запускаешь и сколько это займёт, и дождись подтверждения.** Годы — по одному:
каждый год доводится до HTML-отчёта, прежде чем запускать следующий (решение владельца).

## 0. Новые выгрузки

Положить книги в `data/inbox/PERSONA/` (лист `нов<год>`, заголовок с «Категория программы»).
Повторная выгрузка уже обработанного года может лежать рядом со старой: staging возьмёт более
свежую по дате файла и назовёт пропущенную в `books_superseded`. Перед повторным прогоном года
перенести старую папку `data/processed/persona/<год>/` в архив (`<год>_prev_<дата>`), иначе
тексты ролей прошлого прогона останутся рядом с новыми пакетами.

## 1. Подготовка (дёшево, без подтверждения)

1. **Staging**: `python scripts/persona_stage.py`. Перескажи только агрегаты: программы по
   категориям, блоки, сверку ∑, когорту DT (перенос + исход), `books_superseded`. Ошибка
   «ожидался ровно один лист» или «строка заголовка не найдена», новые заголовки в выводе —
   стоп: сопоставление колонок не угадывать, спросить владельца.
2. **Preflight**: `python scripts/preflight.py --persona <год>` — должен вернуть 0.
3. **Сухая проверка DT**: `python scripts/run_analysis.py --persona <год> --dt-check`.
   Сообщи владельцу циклы, отклонённые строки с причинами и оценку часов (≈20.5 с на расчёт,
   2 расчёта на цикл). Ожидаемо: `afc_from_occ` у всех, `sperm_source_inferred` почти у всех.

## 2. Полный прогон (после подтверждения)

`python -u scripts/run_analysis.py --persona <год> > data/processed/persona/<год>/run.log 2>&1` в фоне.
Replay без повторного DT/TabPFN: `--dt-output data/processed/persona/<год>/local_private/dt71_api_output.csv`
и `--tabpfn-output` из того же прогона.

Что входит сразу (настройки `config/clinics/persona/settings.yaml`):
- DT 7.1 REST API (точка переноса, при отказе — retrieval_fallback);
- TabPFN с разбросом членов ансамбля (`tabpfn_ens_*`; +1–8 с на цикл);
- TabPFN-3 regression;
- DNN/EVO со входом, дозаполненным по правилу DT 7.1 (`dnn_verifier.impute_dt71`);
- Laya с калибровкой центра PERSONA (`laya.centre: "PERSONA"`, статус `ok`).

## 3. Контроль компонентов

Таблица `ok / мягкий сбой + причина / выключен гейтом` по `run_manifest.json`:
- **DT:** `digital_twin.timepoint_used`.
- **TabPFN:** покрытие PRAI и `tabpfn_ens_*` (должен быть заполнен у всех; пусто — адаптер TABPFN старый, `tabpfn_backend.member_spread` отсутствует).
- **TabPFN-3:** `regression_status`.
- **DNN:** `dnn_status=valid_saved_scaler`, `dnn_coverage.scored` = когорта, `model_imputed`.
- **Laya:** `laya_status=ok`, `laya_coverage.scored_centre_calibrated` ≈ `scored`.
- **MIRI / лоты / ПГТ-А:** выключены гейтом.

## 4. Интерпретация

1. `python -X utf8 scripts/persona_role_context.py <год>` → `agent_packets/persona_context.md`
   (факты оркестратора) и `narrator_input_batch_N.json` (по 10 циклов review-list).
   Прочитай контекст сам; вопросы ролей закрывай дописыванием фактов в этот файл.
2. Роли (агенты читают только `data/processed/persona/<год>/agent_packets/` и файлы ролей;
   запись — сами, в ту же папку):
   - `kpi-analyst`, `model-auditor` — модель из frontmatter роли; вход `analysis_bundle.json` + `persona_context.md`;
   - `dt-narrator` — по пачке `narrator_input_batch_N.json` → `dt_narrator_batch_N.md`;
     не больше 20 агентов одновременно; после всех пачек проверь, что каждый cycle_uid пачки описан;
   - `lead-analyst` → `lead_analysis.md` (сохрани копию `lead_analysis_v1_pre_critic.md`);
   - `interpretation-critic` → `interpretation_critic.md`; lead исправляет все замечания.
   Во всех заданиях: числа только из пакетов; факт и гипотеза раздельно; без диагнозов и
   оценки специалистов; только cycle_uid; MIRI/лоты/ПГТ-А — «нет источника»; калибровку DT на
   перенос сравнивать только по Bl > 0; `ci_width_pp` — диапазон по неизвестным стадиям, не ДИ.
3. **Финализация**: `python scripts/finalize_report.py --report data/processed/persona/<год>/DT_Analytics_Report.html --text-file data/processed/persona/<год>/agent_packets/lead_analysis.md`;
   `RUN_SUMMARY.md` года (таблица компонентов + ключевые факты).

## 5. База и выдача (после HTML-отчёта)

1. Связь криопереносов: `python scripts/persona_fet_link.py` (все годы staging).
2. Teable и карточки: `python teable/teable_bootstrap.py --clinic persona` (новые поля схемы),
   затем `python teable/teable_sync.py --clinic persona`. Годы PERSONA реестр находит сам
   (`year_root`): последний год — основной прогон, остальные — дополнительные.
3. Копия карточек для клиники (без ссылок на Teable, есть ФИО — передавать только защищённым
   каналом): `python teable/cards_build.py --clinic persona --standalone "C:/Users/User/Desktop/IVF/PERSONA/PERSONA_карточки_циклов_<годы>.html"`.
4. По запросу владельца:
   - метрики для статьи — `python scripts/persona_metrics_report.py`;
   - файл калибровки DT для клиники — `python scripts/persona_dt_calibration_export.py --out "C:/Users/User/Desktop/IVF/PERSONA"`.
   Оба скрипта берут все годы с прогоном.

Калибровку Laya PERSONA (`Laya/scripts/42_centre_persona.py --write`) автоматически не
перефитировать: сдвиг подобран на 2023–2025, новые годы — его внешняя проверка; перефит —
решение владельца.

## Что держать в голове при интерпретации PERSONA

- AFC в выгрузке нет: в DT он = ОКК × 1.16; у DO-циклов — ОКК донора при возрасте реципиентки.
- Фолликулов нет: OCCR недоступен; «Аномал.опл.» не сопоставлено — 3pNR недоступен.
- Freeze-all остаётся в таблицах компетентности (решение владельца).
- Смешанный ЭКО/ИКСИ в широкой lab-популяции — две строки (блоки), в когорте DT — один цикл.
- Переносы без бластоцист: DT возвращает вероятность текущего цикла (`current_cycle`, 0).
- Криопереносы в когорту отчёта не входят; их связь с пункциями — `fet_links.csv`, Teable FET.
