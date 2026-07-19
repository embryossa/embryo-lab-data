# CLAUDE.md — правила проекта Embryo Lab Data Analytics

Перед работой прочитай этот файл и `docs/CONTRACTS.md`. Основной пользовательский вход —
skill `/run-analysis`; отдельный BAT проекта не создавать и не запускать.

## Неподвижные решения владельца

- Все циклы ICSI; benchmark FR фиксирован 75%.
- MII необязателен. FR использует MII, если MII фактически известен, иначе Inseminated.
- `OCCR=OCC/follicles` — «Частота получения ОКК» модели.
- `MIIR=MII/OCC` — отдельный maturity rate и считается только при известном MII.
- Ноль — известное наблюдение, не пропуск. Исключение строк с `2PN=0` сохраняется.
- AMH и BMI импутируются действующей логикой `normalize.py`; правила не менять.
- `Model_KPIScore` точно повторяет формулу обучения. `KPIScore` — только совместимый
  алиас на границе модели; не смешивать с лабораторными Vienna KPI.
- Когортный KPI = Σ numerator / Σ denominator, никогда не mean индивидуальных ratios.
- Для FR/ClR/BlR/TGBDR/OCCR/MIIR контроль нижний; для DegR/3pNR — верхний.
- В KPI-таблице обязательны numerator, denominator, 95% CI, direction, competency,
  benchmark, status и срезы по времени, IVF/ICSI, эмбриологу, врачу.
- Review-list формируется после join как `priority<=2 OR ci_width_pp>20`.
- DNN недействительна без сохранённого scaler обучения; `fit` на inference запрещён.
- Warning MIRI возникает только из реального эпизода severity warning/critical.
- MIRI: пропуск логов <8ч в окне культивирования — потеря возможности ПИСАТЬ логи
  (например, отключение питания), реальное неблагоприятное событие для анализа, а не
  «нет данных». Пропуск ≥8ч — явная остановка инкубатора или логирования. Порог —
  `incubator.DEFAULT_SEVERITY["gap_reason_threshold_minutes"]`; поле `gap_reason` на
  событиях `event_type=gap`, не заменяет существующую severity (10-минутный порог).
- KPI эмбриологов (FR, ClR, BlR, TGBDR, DegR, 3pNR) и KPI врачей (только OCCR, MIIR) —
  раздельные таблицы; никогда не сравнивать/не смешивать эмбриолога и врача в одной
  таблице или одном выводе. Ниже уровня компетентности — отдельный абзац «Поводы
  задуматься», не встроенный в таблицу. Эти две таблицы строятся на широкой
  lab-популяции (без гейта transferred>0) минус `kpi.competency_exclusion_mask`
  (открытые циклы без Bl, NO-retrieval, freeze-all/Cryo — токены в
  `config/columns.yaml`); это исключение больше нигде в пайплайне не применяется.
- OZEMA — арест-паттерны цикла (OMA/FF/ZA/EEA), односторонний биномиальный тест
  (α=0.05) против порогов 75%/60%/95%/40%, n<4 → «н/д» (не «n/a» — коллизия с
  pandas na_values). Методология и пороги — Van Der Kelen et al., Hum Reprod 2026,
  лист OZEMA мастер-таблицы OPU (Google Drive). ≥2 флагов у одного пациента —
  кандидат на обсуждение генетического консультирования; это НЕ диагноз. Работает
  без transferred>0-гейта и без exclusion-набора эмбриолог/врач таблиц (freeze-all
  циклы тоже могут получить флаг) — единственное собственное исключение OZEMA:
  TESE/SMF из `sperm_source`, и только для стадии FF.
- Итоговый HTML-отчёт (`report.py`) разбит на вкладки (Обзор, KPI эмбриологов,
  KPI врачей, Модели и калибровка, TabPFN: прогноз беременности, TabPFN-3: прогноз
  протокола, Review-list, MIRI 6391, Партии расходников, OZEMA, Итоговая
  интерпретация). Стиль/палитра — тёплая Claude.ai (`--clay:#d97757` и токены её
  дизайн-системы), не менять без прямого запроса владельца. MIRI-вкладка показывает
  только реальные эпизоды (без сводки по циклам). Вкладка «Прогноз
  беременности» берёт воронку реал/прогноз из P50 TabPFN-3 regression ЭТОГО отчёта
  — не из старых полей Digital Twin `med_*`; это два разных источника прогноза
  воронки, не смешивать.
  - Дискриминативные метрики моделей (Brier, ROC-AUC, Acc@0.5 и т.п.) сравниваются
  только в таблице «Калибровка» вкладки «Модели и калибровка». lead-analyst не
  описывает их и не опирается на них при формировании «Итоговой интерпретации» —
  на малых когортах этого проекта такие метрики статистически неотличимы от шума и
  подрывают доверие к честному по сути пайплайну, не давая взамен истинного сигнала
  о качестве моделей.
## Поток данных

1. Resolve колонок только через `config/columns.yaml`; неизвестные имена не угадывать.
2. Normalize Preg, AMH/BMI; отобрать строки с переносом и заполненным исходом.
3. Добавить локальный HMAC `cycle_uid`; ID/ФИО остаются в локальном слое.
4. Посчитать лабораторные KPI и отдельный Model_KPIScore.
5. Запустить свежие batch-результаты IVF Digital Twin и TabPFN. Старый файл текущим
   результатом не считать. Дополнительно (мягкий сбой, не блокирует пайплайн) запустить
   TabPFN-3 regression количественных стадий воронки и сверить факт против P10–P90 —
   независимый QA-слой, не заменяет реконсилиацию по исходу переноса.
6. Выполнить `TabPFN → dt_predictions.csv → OPU_table_enriched.xlsx` и проверку контрактов.
7. Перенести DataLore-аналитику локально: калибровка, funnel, flags, review-list,
   агрегированные KPI и срезы.
8. Для review-list ∪ wide CI разобрать MIRI в окне Date OPU..+6 дней: episodes,
   duration, max deviation, severity, gaps, recovery time.
8a. Для того же окна Date OPU..+N дней сверить смену лотов расходников (Airtable,
    см. «Внешние инструменты») — сколько лотов активно на позицию, была ли смена
    лота внутри окна (`event_type=lot_change`) как кандидат-confounder.
8b. Для той же когорты (review-list ∪ известный исход переноса) — при наличии
    скачанных отчётов ПГТ-А в `data/processed/local_private/pgt_reports/` —
    посчитать когортные признаки euploid rate, CNV QC noise (только по
    эуплоидным эмбрионам) и вариативность analysis settings; мягкий шаг, не
    блокирует пайплайн при отсутствии отчётов.
9. Атомарно записать CSV/XLSX/HTML/JSON и run manifest.
10. Агентам дать только `data/processed/agent_packets/`; reverse join выполнить Python
    локально после интерпретации.

## Приватность

Запрещено помещать в agent prompt/packet: ID, `_key`, `_join`, ФИО, DOB, сведения мужа,
точные идентификаторы записей. Разрешён `cycle_uid` и аналитические признаки по allow-list.
Ни один агент не должен читать `data/inbox`, `data/raw`, `data/processed/local_private`,
OPU master или полный `dt_predictions.csv`. Финальный XLSX/HTML с локально возвращёнными
идентификаторами формирует только Python.

## Внешние инструменты

- Digital Twin: `C:\Users\User\Desktop\IVF\AI\IVF Digital Twin Pro\IVF Digital Twin\RUN_BATCH.bat`.
- TabPFN: batch adapter из `external/tabpfn/`, установленный в существующий проект TabPFN.
- TabPFN-3 regression (количественный прогноз стадий MII/2PN/Bl/Good Bl/Frozen):
  `C:\Users\User\PycharmProjects\pythonProject\tabpfn\tabpfn3_ivf_regression_report_CPU_16K.py`,
  своё окружение `.venv` в том же проекте. Гейт `external_tools.tabpfn_regression.enabled`
  (по умолчанию включён); сбой этого шага мягкий и не блокирует остальной прогон.
- DNN model: `C:\Users\User\PycharmProjects\pythonProject\KAN\saved_model\my_best_model`;
  scaler задаётся отдельно в local settings. Текущий `training_scaler.joblib` —
  реконструкция (2026-07-17, см. `config/settings.local.yaml`), не сохранённый оригинал
  обучения; raw-выход EVO сравнивается с DT/TabPFN без отдельного калибратора.
- DataLore reference: `notebooks/reference/Data_Twin_analytics.ipynb`; выполняемый код —
  локальный Python-модуль, а не облачный notebook.
- Consumable lots: Airtable base `appTnM9Hp4x4jecrW` («расходные материалы» +
  «Barcode Changes History»), забирается живым pull по REST API (`src/embryo/lots.py`).
  Гейт `consumable_lots.enabled` + Personal Access Token — только в
  `config/settings.local.yaml`, в репозиторий не коммитятся.
- MIRI логи: авторитетный источник — шаренная папка Google Drive (folder id в
  `config/settings.local.yaml: miri_logs.google_drive_folder_id`), структура зеркальна
  `data/miri_logs/6391/Miri-6391-Incubation-[YYMMDD-YYMMDD]/Data/*.csv` (подтверждено
  2026-07-17: тот же разделитель `;`, те же колонки). Синхронизация — коннектор Google
  Drive MCP, интерактивно из Claude Code (см. `.claude/skills/run-analysis/SKILL.md`
  шаг 2a), не headless Python: скачиваются только папки недель, пересекающие окно
  `Date OPU..+cultivation_window_days` пациентов review-list текущего прогона, а не весь
  архив. В `data/miri_logs` больше не кладутся логи вручную.
  - ПГТ-А: личный кабинет `https://pa.ivf.uz/orders` (нет API, вход на этом ПК
  автоматический), вкладка Pdg — заказы биопсии трофэктодермы, скачиваемый PDF
  (eMap Cycle Report, IMC Genomics/Vitrolife) в разделе Results со статусом
  `Provided`. Скачивание и сопоставление с ФИО — интерактивно из Claude Code
  (claude-in-chrome, см. `.claude/skills/run-analysis/SKILL.md` шаг 2b), не headless.
  Парсинг — `src/embryo/pgt.py`; сопоставление с `cycle_uid` по имени пациента
  (Subject ID портала = Имя+Фамилия без пробела), повторные попытки одной
  пациентки разрешаются по дате отчёта внутри окна `Date OPU..Date ET`.
  Классификация euploid/aneuploid/mosaic — по английскому `Summary result`, не
  по русскоязычному `Sign-off comment` (у отчёта нечитаемый кириллический шрифт
  без ToUnicode CMap — ограничение самого PDF, подтверждено на pdfplumber и
  PyMuPDF). Только когорта review-list ∪ известный исход переноса. Сырые PDF и
  ФИО — только `data/processed/local_private/pgt_reports/`, не коммитятся, не
  попадают в agent packet (лабораторный `sample_id` эмбриона тоже — он содержит
  фрагмент ФИО, в пакет уходит только порядковый `embryo_N`); интерпретация —
  роль `pgt-forensics`.

## Роли Claude Code

Промежуточные роли используют `model: haiku`: схема, KPI, model QA, MIRI, нарратор,
лоты, DNN-проверка — однотипные задачи объяснения по готовому пакету, не требующие
дорогой модели. `lead-analyst` использует `model: opus` (целевой Opus 4.8 через
настроенный alias Claude Code) — единственная роль, синтезирующая всё воедино.
Python считает и классифицирует; агенты только объясняют. Факт и гипотезу всегда
разделять, диагнозов и назначений не давать.
