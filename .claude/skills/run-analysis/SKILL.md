---
name: run-analysis
description: End-to-end IVF laboratory run from inbox through external DT/TabPFN, local joins, KPI, MIRI, pseudonymized agent review and reports.
---
# Полный запуск анализа

Несколько клиник в одном проекте (см. `docs/FERTIMED.md`): без аргументов —
клиника A (поведение не меняется). `/run-analysis --fertimed` — прогон для
Fertimed (`config/clinics/fertimed/`, `--clinic fertimed` под капотом
`scripts/run_analysis.py`). У Fertimed нет MIRI/лотов расходников
(`miri_logs.enabled: false`, `consumable_lots.enabled: false` в её конфиге) —
шаги 2a/2b ниже относятся только к клинике A и для Fertimed пропускаются
автоматически. ПГТ-А клиники Fertimed (`https://my.f-genetics.com/#/cases/orders`)
уже автоматизирован (`pgt.enabled: true`, парсер `src/embryo/clinics/
fertimed_pgt.py`, см. `docs/FERTIMED.md`) — новые отчёты для пациентов review-list
∪ известного исхода без ещё скачанного файла в `pgt.reports_dir` нужно докачивать
интерактивно через claude-in-chrome по аналогии с шагом 2b клиники A (портал и
формат PDF другие, сама логика скачивания под Fertimed отдельно не описана здесь —
см. `docs/FERTIMED.md`), затем перезапустить тем же `--dt-output/--tabpfn-output`
replay.

`/run-analysis --ggrc` — прогон для GGRC (Armenia) (`config/clinics/ggrc/`,
`--clinic ggrc`, см. `docs/GGRC.md`). Тоже без MIRI/лотов расходников; ПГТ-А
не требует шага 2b вообще — данные уже в самой выгрузке, отдельного
докачивания с портала нет. У GGRC два отдельных входных файла (eligible-
когорта + широкая lab-популяция для KPI) вместо одного — см. docs/GGRC.md,
подготовка входа для этой клиники ничем не отличается для пользователя skill'а
(шаги ниже те же).

0. Pre-flight (дёшево, экономит целый повторный прогон):
   - **Обязательно выполни `python scripts/preflight.py $ARGUMENTS` и не начинай
     прогон, пока он не вернёт 0.** Скрипт сверяет фактически установленное с
     `requirements.txt`, отдельно проверяет импорт пакетов, чьё отсутствие даёт
     МЯГКИЙ отказ (нет исключения — просто тихо пропущенный шаг), и проверяет
     существование путей внешних инструментов из `config/machine.local.yaml` с
     учётом гейтов `launch`/`enabled`. Чинится обычно одной командой
     `python -m pip install -r requirements.txt`.
     Почему это не формальность — три подтверждённых случая расхождения
     окружения с `requirements.txt` на этой машине: `pymupdf` (19.07.2026, молча
     упали все 11 разборов ПГТ-А, выглядело как «у клиники нет отчётов»),
     `pytest` (28.07.2026, тесты нечем было прогнать), `python-calamine`
     (28.07.2026, чтение Excel втихую шло по медленному откату на openpyxl).
     Первый случай стоил ошибочного вывода в отчёте, а не только времени.
   - После шага ПГТ-А не делай вывод "клиника не тестирует генетику"/"не заказывали"
     только по `pgt_note`. Сверь число пациентов когорты, у которых в сырой выгрузке
     есть флаг заказа ПГТ (у Fertimed — `PGT-ET`/`pgt_et_flag` = "Да"; у клиники A
     такого явного флага в выгрузке нет, там кандидатами считается весь review-list
     ∪ известный исход), против числа реально сопоставленных `n_biopsied>0` записей.
     Если флагованных заметно больше, чем сопоставленных — отчёты просто ещё не
     скачаны (шаг 2b ниже не выполнялся), а не "не заказывались".
1. Из корня проекта выполни `python scripts/run_analysis.py $ARGUMENTS` (первый проход;
   если MIRI-логи для нужного окна ещё не синхронизированы локально, `miri_summary`
   просто отметит их отсутствием по этим `cycle_uid` — это не блокирует остальной пайплайн).
2. Не продолжай клиническую интерпретацию, если оркестратор завершился ошибкой контракта,
   не создал `dt_predictions.csv` или указал, что DNN недействительна из-за scaler.
2a. MIRI из Google Drive (коннектор MCP, интерактивно — не headless Python): открой
    `review_list.csv`, возьми оттуда только `cycle_uid` + `Date OPU` (это уже локальный
    файл, не агентский пакет — ID/ФИО там и так нет). Для каждой даты вычисли окно
    `Date OPU .. +cultivation_window_days` и определи, какие папки
    `Miri-6391-Incubation-[YYMMDD-YYMMDD]` в шаренной папке Google Drive пересекаются
    с этим окном (folder id в `config/settings.local.yaml: miri_logs.google_drive_folder_id`).
    Через `mcp__claude_ai_Google_Drive__search_files` с `parentId = '<folder_id>'` найди
    пересекающиеся неделные подпапки, затем их `Data`-подпапку; скачай отсутствующие/более
    новые `*.csv` через `download_file_content` и запиши в
    `data/miri_logs/6391/<та же папка>/Data/<тот же файл>.csv` (структура зеркальна —
    просто копия путей). Не скачивай папки вне пересекающихся окон — весь архив Drive
    качать не нужно. Если это не первый прогон и нужные папки уже локальны и не новее
    Drive-версии — пропусти скачивание.
    Если после этого шага появились новые/обновлённые MIRI CSV — перезапусти
    `python scripts/run_analysis.py $ARGUMENTS --dt-output <результат DT из первого
    прохода> --tabpfn-output <результат TabPFN из первого прохода>` (детерминированный
    replay без повторного запуска внешних DT/TabPFN — пересчитается только MIRI/lots/
    аналитика/отчёты с уже подтянутыми логами).
2b. ПГТ-А из личного кабинета pa.ivf.uz (claude-in-chrome, интерактивно — не headless):
    гейт `pgt.enabled` в `config/settings.yaml`. Собери когорту локально: cycle_uid +
    `Patient Full Name` + `Date OPU`/`Date ET` из `data/processed/review_list.csv`
    ∪ `data/processed/local_private/model_input.xlsx` (полная когорта с известным
    исходом переноса) — это уже локальные файлы, не агентский пакет. Для каждого
    пациента, у которого ещё нет файла в `pgt.reports_dir`, зайди в
    `https://pa.ivf.uz/orders`, найди по ФИО (склеенные Имя+Фамилия без пробела —
    формат `Subject ID` портала), и если статус `Provided` — скачай PDF в
    `pgt.reports_dir` (`data/processed/local_private/pgt_reports/`, никогда не
    коммитится). Имя файла не имеет значения — `pgt.analyze_reports` сопоставляет
    отчёт с `cycle_uid` по `Subject ID` внутри самого PDF, разрешая повторные попытки
    одной пациентки через окно `Date OPU..Date ET`. Пациентов без ПГТ или без
    готового результата — пропусти. После скачивания перезапусти тем же
    `--dt-output/--tabpfn-output` replay-командой (шаг 2a) — пересчитается PGT вместе
    с MIRI/lots.
3. Агентам разрешено читать только `data/processed/agent_packets/`. Файлы OPU,
   crosswalk, исходные таблицы и локальные результаты с ID/ФИО не передавать агентам.
4. Запусти роли по порядку: kpi-analyst, incubator-forensics, consumable-lots-forensics,
   pgt-forensics, dt-narrator для review-list, затем lead-analyst. Все промежуточные
   роли используют Haiku; lead-analyst — Opus.
   Для dt-narrator: возьми список `cycle_uid` из `review_list.csv` (он же в
   `analysis_bundle.json`), раздели его на пачки по 8–10 и вызови dt-narrator один раз на
   каждую пачку, передав ей в задании ровно её список `cycle_uid` (не по одному вызову на
   пациента). Пачки независимы — можно вызывать параллельно. При <=10 пациентах в
   review-list это один вызов.
4a. Сохрани итог lead-analyst в `agent_packets/lead_analysis.md` и запусти по нему
   `interpretation-critic` (Opus) — состязательную проверку интерпретации против тех же
   пакетов. Роль ищет ровно то, что дороже всего стоит клинике: числа, не совпадающие с
   пакетом; утверждения без основания; гипотезы, поданные как вывод; доли на вырожденных
   знаменателях; одно измерение, поданное как несколько независимых подтверждений.
   Замечания высокой серьёзности отработай до `finalize_report.py`: либо поправь текст
   через lead-analyst, либо, если замечание неверно, зафиксируй почему.
   Это не формальность. Проверка на реальном отчёте (Fertimed, 19.07.2026) дала 3
   замечания высокой серьёзности, включая стержневой аргумент: «три независимых
   измерения сходятся» — при том что два из трёх были одной и той же дробью (143/250 и
   как шаг воронки `2PN→Bl`, и как BlR), а главный вывод стоял на 48 циклах из 250, что
   в тексте нигде не названо. Панель параллельных синтезаторов этого не ловит — ловит
   именно недоверчивый одиночный проход.
   Пустой результат критика — нормальный исход, не повод искать замечания повторно.
5. Числа не пересчитывать в LLM. Итоговые гипотезы отделять от установленных фактов.
6. Обратное сопоставление `cycle_uid → ID/ФИО` выполняется только локальным Python при
   формировании окончательного XLSX/HTML, никогда не в контексте Claude.
7. Убедись, что `agent_packets/lead_analysis.md` содержит версию, отработанную по
   замечаниям критика, и запусти `python scripts/finalize_report.py`.

Основные результаты: `dt_predictions.csv`, `OPU_table_enriched.xlsx`,
`review_list.csv`, `miri_events.csv`, `lot_events.csv`, `pgt_summary.csv`,
`pgt_embryo_events.csv`, `ozema_cycles.csv`,
`ozema_genetic_counseling_candidates.csv`, `DT_Analytics_Report.xlsx`,
`DT_Analytics_Report.html` (вкладки: Обзор, KPI эмбриологов, KPI врачей, Модели и
калибровка (боксплот прогнозов по всем моделям + KAT vs PRAI, стиль
notebooks/reference/Data_Twin_analytics.ipynb), Кластерный анализ (donut/профиль/
исходы/P(беременность) по кластерам), Похожие случаи (граф сходства пациентов,
Similar Cases, Mann-Whitney intra/inter-cluster), Прогноз беременности
(PRAI+Digital Twin, воронка — из TabPFN-3 regression P50 этого отчёта),
TabPFN-3: прогноз протокола (per-stage P10-P90 QA), Review-list, MIRI 6391,
Партии расходников, OZEMA, Итоговая
интерпретация; стиль/палитра — Claude.ai warm/clay), `run_manifest.json`.
