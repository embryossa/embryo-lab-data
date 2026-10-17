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
- Показатель выше 100% — НЕ «испорченные данные» и не повод забраковать строку.
  `ClR > 100%` — неоптимальный тайминг оценки оплодотворения: часть зигот не попала в
  окно визуализации пронуклеусов и не учтена как 2pN, но продробилась (вывод — про
  тайминг проверки, а не про правила подсчёта). `MII > OCC` (и следом прогноз воронки
  `OCC→MII` больше 1) — в программу добавлены размороженные ооциты, которых не было в
  OCC свежей пункции. Называть это «data integrity / counting mismatch» запрещено;
  биномиальный CI в таких строках не считается только потому, что математически не
  определён.
- Стадия «хорошие бластоцисты» существует в ДВУХ базах: Vienna `TGBDR = Good Bl / 2pN`
  (пороги 30%/40%) и доля от бластоцист `good/Bl` — так считают CSDI (`good_rate`) и шаг
  воронки `Bl→Good`. Не сравнивать их между собой и не читать `Bl→Good` против порогов
  TGBDR: одна и та же лаборатория даёт `Bl→Good` ≈70% и TGBDR ≈43% одновременно.
  Мост: `TGBDR = (Bl→Good) × BlR`. У KAT-фич и `calibrate_for_clinic` база — 2PN, как у
  Vienna; у CSDI `blast_rate = Bl/2pN` совпадает с BlR, расхождение только на одной стадии.
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
- Итоговый HTML-отчёт (`report.py`): состав вкладок, палитра и решения владельца по
  отчёту (стиль не менять без запроса, источник воронки, место дискриминативных
  метрик) — в skill `run-analysis`, раздел «Неподвижные решения владельца по
  HTML-отчёту»; правило о метриках для синтеза — в `.claude/agents/lead-analyst.md`.

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
10. Агентам дать только папку `agent_packets` текущего прогона (`data/processed/agent_packets/` для клиники A, `data/processed/<клиника>/agent_packets/` для остальных); reverse join выполнить Python
    локально после интерпретации.

## Мониторинг и контроль подключения компонентов (обязательно)

Владелец, 2026-10-02: каждый прогон сопровождается мониторингом и явной проверкой
подключения ВСЕХ компонентов. Мягкий сбой допустим для пайплайна, но не для отчёта о
прогоне: тихо пропущенный шаг выглядит как «данных нет» (прецедент — `pymupdf`,
19.07.2026).
- До расчёта: `scripts/preflight.py` = 0 и `--dt-check` со статусом службы `ready`.
- Во время: длинный прогон идёт в фоне с логом, лог проверяется, а не только код выхода.
- После: по каждому компоненту — DT 7.1 API, TabPFN, TabPFN-3 regression, DNN/EVO, Laya,
  Airtable-лоты, MIRI (Google Drive MCP), ПГТ-А портал (claude-in-chrome), Teable —
  сверить фактическое покрытие свежего прогона (`run_manifest.json`, число строк/циклов
  в выходах, mtime) с ожидаемым и привести пользователю таблицу статусов:
  `ok` / `мягкий сбой + причина` / `выключен гейтом`. Нулевое покрытие включённого
  компонента — сбой подключения, пока не доказано обратное
  (см. правило о несделанном шаге и отсутствии данных).

## Приватность

Запрещено помещать в agent prompt/packet: ID, `_key`, `_join`, ФИО, DOB, сведения мужа,
точные идентификаторы записей. Разрешён `cycle_uid` и аналитические признаки по allow-list.
Ни один агент не должен читать `data/inbox`, `data/raw`, `data/processed/local_private`,
OPU master или полный `dt_predictions.csv`. Финальный XLSX/HTML с локально возвращёнными
идентификаторами формирует только Python.

## Внешние инструменты

- Все абсолютные машинные пути к внешним инструментам ниже (Digital Twin, TabPFN,
  DNN, TabPFN-3 regression) живут ТОЛЬКО в `config/machine.local.yaml` (в
  .gitignore, общий для всех клиник — см. `config/machine.local.yaml.example`).
  Этот файл — не место для них; при переезде на новую машину меняется только
  `config/machine.local.yaml`, этот раздел не устаревает.
- Digital Twin: только 7.1 через локальный REST API (служба `IVFClinicIntegration71`,
  подключение «Аналитика», `src/embryo/dt_api.py`) — для всех клиник и всех команд;
  батч 7.0 (`RUN_BATCH.bat`) снят 2026-09-13. Политика пакета — `digital_twin.api` в
  `settings.yaml` клиники, сертификат службы — `digital_twin.api.ca_cert` в
  `config/machine.local.yaml`, ключ — под DPAPI (`python -m embryo.dt_api store-key`).
- TabPFN: batch adapter из `external/tabpfn/`, установленный в существующий проект TabPFN
  — путь в `external_tools.tabpfn.project_dir`.
- TabPFN-3 regression (количественный прогноз стадий MII/2PN/Bl/Good Bl/Frozen):
  `tabpfn3_ivf_regression_report_CPU_16K.py`, своё окружение `.venv` в том же
  проекте — путь в `external_tools.tabpfn_regression.project_dir`. Гейт
  `external_tools.tabpfn_regression.enabled` (по умолчанию включён); сбой этого
  шага мягкий и не блокирует остальной прогон.
- DNN model: путь в `external_tools.dnn_verifier.model_path`; scaler — в
  `dnn_verifier.scaler_path`. Текущий `training_scaler.joblib` — реконструкция
  (2026-07-17, см. `config/settings.local.yaml`), не сохранённый оригинал
  обучения; raw-выход EVO сравнивается с DT/TabPFN без отдельного калибратора.
- Laya (`laya_kan_ft`, проект Laya, своё `.venv`): внешний компаратор DT, НЕ член
  ансамбля L1–L7 и не замена DT; адаптер `external/laya/laya_batch.py` вызывает тот же
  `service/laya_ivf.assess`, что MCP `laya_ivf`. Гейт/центр/restore — `external_tools.laya`
  в settings.yaml клиники, пути — `config/machine.local.yaml`; мягкий сбой
  (`laya_status` в run_manifest). Валидация 2026-10-15 на 1 387 циклах Teable:
  внутрицентровый AUC Laya 0.555 vs DT 0.559; `laya_disagreement_high` — повод перепроверить
  цикл (в этой зоне Brier DT выше), не вердикт. Вероятность непригодного цикла не публикуется.
- DataLore reference: `notebooks/reference/Data_Twin_analytics.ipynb`; выполняемый код —
  локальный Python-модуль, а не облачный notebook.
- Consumable lots: Airtable-база лаборатории (ID базы/таблиц — в `config/settings.local.yaml`; «расходные материалы» +
  «Barcode Changes History»), забирается живым pull по REST API (`src/embryo/lots.py`).
  Гейт `consumable_lots.enabled` + Personal Access Token — только в
  `config/settings.local.yaml`, в репозиторий не коммитятся.
- MIRI логи: авторитетный источник — шаренная папка Google Drive (структура и формат
  CSV зеркальны `data/miri_logs/6391/`). Синхронизация — только интерактивно через
  Google Drive MCP и только недели окна пациентов review-list, не весь архив, не
  headless Python (процедура — `run-analysis` шаг 2a). Вручную в `data/miri_logs` логи
  не кладутся.
- ПГТ-А: личный кабинет `https://pa.ivf.uz/orders` (нет API) — скачивание и
  сопоставление только интерактивно через claude-in-chrome, не headless (процедура —
  `run-analysis` шаг 2b); парсинг — `src/embryo/pgt.py`. Классификация euploid/aneuploid/mosaic — по английскому `Summary result`, не
  по русскоязычному `Sign-off comment` (у отчёта нечитаемый кириллический шрифт
  без ToUnicode CMap — ограничение самого PDF, подтверждено на pdfplumber и
  PyMuPDF). Только когорта review-list ∪ известный исход переноса. Сырые PDF и
  ФИО — только `data/processed/local_private/pgt_reports/`, не коммитятся, не
  попадают в agent packet (лабораторный `sample_id` эмбриона тоже — он содержит
  фрагмент ФИО, в пакет уходит только порядковый `embryo_N`); интерпретация —
  роль `pgt-forensics`.

## Роли Claude Code

Модели ролей — во frontmatter `.claude/agents/*.md` (источник истины). Промежуточные
роли — дешёвые модели (однотипные объяснения по готовому пакету); синтезирующие роли
пинятся полным ID, а не alias `opus`, чтобы модель не менялась молча при сдвиге alias.
Файл агента один на все клиники.
Python считает и классифицирует; агенты только объясняют. Факт и гипотезу всегда
разделять, диагнозов и назначений не давать.