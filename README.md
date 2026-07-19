# Embryo Lab Data Analytics

Локальный end-to-end аналитический конвейер для лаборатории эмбриологии IVF/ICSI:
Excel/CSV на входе → нормализация и приватность → лабораторные KPI по Vienna
Consensus → внешние прогнозные модели (IVF Digital Twin, TabPFN, TabPFN-3,
опциональная DNN-проверка) → форензика инкубатора (MIRI), партий расходников
(Airtable) и ПГТ-А → псевдонимизированный разбор агентами Claude Code → единый
HTML/XLSX-отчёт с итоговой клинической интерпретацией.

Это система поддержки анализа для лаборатории, **не медицинское изделие и не
механизм автоматического назначения лечения**. Причинные объяснения в отчёте —
гипотезы, а не диагнозы.

---

## Что происходит внутри (сквозной поток)

Единственная точка входа — skill `/run-analysis` в Claude Code
(`.claude/skills/run-analysis/SKILL.md`), которая запускает
`scripts/run_analysis.py` и затем оркестрирует агентов. Отдельного BAT-файла
для основного проекта нет и не должно быть.

1. **Resolve колонок** — только через `config/columns.yaml`; неизвестные имена
   не угадываются (см. `src/embryo/ingest.py`).
2. **Normalize** — исход переноса (Preg), импутация AMH/BMI действующей
   логикой (`src/embryo/normalize.py`); отбираются строки с переносом и
   известным исходом. Ноль — это наблюдение, а не пропуск.
3. Локальный HMAC `cycle_uid` — ID/ФИО остаются только в локальном слое
   (`src/embryo/privacy.py`).
4. **Лабораторные KPI** (`src/embryo/kpi.py`) по методологии Vienna Consensus
   + отдельный `Model_KPIScore`, который точно повторяет формулу обучения
   модели (не путать с лабораторным KPI).
5. **Внешние модели** запускаются автоматически в своих окружениях
   (`src/embryo/_launch.py`, `dt_runner.py`, `tabpfn_runner.py`,
   `tabpfn_regression_runner.py`, `dnn_runner.py`):
   - **IVF Digital Twin** — ансамбль из 7 моделей (MC pipeline, Bayes,
     KAT, GAT, CSDI, TabPFN/PRAI, итоговый BEFE-ансамбль `DIGITAL TWIN`);
   - **TabPFN** — вероятность беременности на перенос (`PRAI`) с CI;
   - **TabPFN-3 regression** — независимый количественный QA-слой: прогноз
     P10–P90 по стадиям воронки (MII/2PN/Bl/Good Bl/Frozen), сверяется с
     фактом, не подменяет реконсилиацию по исходу переноса; сбой этого шага
     мягкий и не блокирует остальной пайплайн;
   - **DNN-проверка** (опционально, выключена по умолчанию) — независимая
     TensorFlow-модель, недействительна без сохранённого scaler обучения.
6. **Reconcile** DT → TabPFN → OPU по паспортному ID (`src/embryo/reconcile.py`)
   + проверка версионированных контрактов (`src/embryo/contracts.py`,
   `docs/CONTRACTS.md`) — при нарушении контракта, дубликате ключа или
   несвежем результате полный прогон прекращается.
7. **DataLore-аналитика перенесена в Python** (`src/embryo/analytics.py`) —
   детерминированный порт `notebooks/reference/Data_Twin_analytics.ipynb`:
   калибровка моделей (Brier/AUC), воронка факт/прогноз, кластерный анализ,
   граф сходства пациентов (Similar Cases), review-list.
8. **Форензика окна культивирования** (`Date OPU .. +cultivation_window_days`,
   по умолчанию 6 дней) для review-list ∪ широких CI:
   - **MIRI 6391** (`src/embryo/incubator.py`) — эпизоды, gaps, severity,
     recovery time, Western Electric rules по логам инкубатора;
   - **Партии расходников** (`src/embryo/lots.py`) — смена лота внутри окна
     как кандидат-confounder (не причина), источник — Airtable;
   - **ПГТ-А** (`src/embryo/pgt.py`) — euploid rate, CNV QC noise (только по
     эуплоидным эмбрионам), вариативность analysis settings, источник — PDF
     eMap Cycle Report с портала pa.ivf.uz.
9. **OZEMA** (`src/embryo/ozema.py`) — арест-паттерны цикла (OMA/FF/ZA/EEA),
   односторонний биномиальный тест против порогов Van Der Kelen et al. (Hum
   Reprod 2026); ≥2 флагов у одного пациента — кандидат на обсуждение
   генетического консультирования (не диагноз).
10. **Псевдонимизированный разбор агентами Claude Code** — промежуточные
    роли на Haiku (`kpi-analyst`, `incubator-forensics`,
    `consumable-lots-forensics`, `pgt-forensics`, `dt-narrator`) читают
    только `data/processed/agent_packets/analysis_bundle.json` (никаких
    ID/ФИО); финальный синтез делает `lead-analyst` на Opus.
11. **Атомарная запись** CSV/XLSX/HTML/JSON + run manifest с sha256 и
    версиями схем (`src/embryo/_io.py`).

### Итоговый HTML-отчёт (`report.py`)

Одна страница, 13 вкладок: Обзор · KPI эмбриологов · KPI врачей · Модели и
калибровка · Кластерный анализ · Похожие случаи · Прогноз беременности ·
TabPFN-3: прогноз протокола · Review-list · MIRI 6391 · Партии расходников ·
OZEMA · Итоговая интерпретация. Полностью локальный интерактивный Plotly,
стиль — тёплая палитра Claude.ai.

## Приватность и границы данных

- ID, `_key`, `_join`, ФИО, DOB, точные идентификаторы записей **никогда** не
  попадают в agent-контекст. Агентам доступен только
  `data/processed/agent_packets/` (allow-list аналитических признаков +
  `cycle_uid`).
- Обратное сопоставление `cycle_uid → ID/ФИО` выполняет только локальный
  Python при формировании финального XLSX/HTML — никогда в контексте Claude.
- Сырые данные (`data/raw/`, `data/inbox/`, `data/miri_logs/`,
  `data/processed/` целиком, включая `local_private/` с ПГТ-PDF и crosswalk)
  а также `config/machine.local.yaml` и `config/settings.local.yaml` /
  `config/clinics/*/settings.local.yaml` — в `.gitignore`, в этом репозитории их
  нет. Переносите их на новую машину отдельно и вручную, вне git.

## Структура репозитория

```
config/            settings.yaml (переносимые параметры) + columns.yaml (схема
                    колонок) + clinics/<name>/ (отдельный settings.yaml на
                    клинику) + machine.local.yaml.example (шаблон машинных
                    путей DT/TabPFN/DNN/TabPFN-3, общий для ВСЕХ клиник) +
                    settings.local.yaml.example (шаблон клиничных секретов/флагов)
src/embryo/         ядро пайплайна (ingest, normalize, kpi, incubator, lots,
                    pgt, ozema, analytics, reconcile, contracts, privacy,
                    report, *_runner.py — адаптеры внешних инструментов)
scripts/            run_analysis.py (оркестратор), finalize_report.py,
                    check_setup.py, run_pipeline.py
.claude/            skill /run-analysis, роли-агенты (Haiku/Opus), прочие
                    project skills (dt-ensemble, vienna-kpi, ozema, ...)
tests/               pytest: синтетические фикстуры, e2e replay, контракты
docs/               CONTRACTS.md, METHODOLOGY.md, TABPFN_INSTALL.md
notebooks/reference/ исходный DataLore-ноутбук (неизменный референс)
external/           batch-адаптеры для внешних проектов (TabPFN, DNN)
data/               inbox/raw/processed/miri_logs — не в git, кроме
                    data/sample/ (синтетический пример)
```

## Установка на новой машине

### 1. Клонирование и окружение

```powershell
git clone https://github.com/embryossa/embryo-lab-data.git
cd embryo-lab-data
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` покрывает только детерминированное ядро этого проекта
(pandas, numpy, openpyxl, scipy, scikit-learn, networkx, plotly, pymupdf и
т.д.). Torch/TabPFN/IVF Digital Twin/KAN живут в своих отдельных окружениях —
их сюда ставить не нужно.

### 2. Локальный конфиг

Машинные абсолютные пути к внешним инструментам (DT/TabPFN/DNN/TabPFN-3) —
**один файл на весь ПК**, общий для всех клиник:

```powershell
copy config\machine.local.yaml.example config\machine.local.yaml
```

Заполните все `project_dir`/`results_dir`/`model_path`/`scaler_path`/`run_cmd`
под новую машину — это единственный файл, который нужно редактировать, если
меняются только пути (сам репозиторий, обе клиники и config/settings.yaml
трогать не нужно). Проверить: `python scripts\check_setup.py --all-clinics`.

Отдельно — клиничные секреты/флаги (на клинику, НЕ про пути):

```powershell
copy config\settings.local.yaml.example config\settings.local.yaml
copy config\clinics\fertimed\settings.local.yaml.example config\clinics\fertimed\settings.local.yaml
```

Заполните под новую машину/аккаунт:

- `miri_logs.base_dir` или `miri_logs.google_drive_folder_id` — источник
  логов инкубатора MIRI 6391 (клиника A).
- `consumable_lots.airtable.token` — Personal Access Token с правами
  `data.records:read`, `schema.bases:read` на нужную Airtable-базу; без него
  форензик по лотам остаётся выключен (`enabled: false`).
- `pgt.enabled: true` — только когда соответствующая `local_private/pgt_reports/`
  наполнена и парсинг проверен (у каждой клиники свой личный кабинет/парсер).

`config/settings.yaml` и `config/clinics/*/settings.yaml` — версионированная,
переносимая часть конфига (структура папок, схема join, пороги, feature list
для DT) и НЕ содержат абсолютных путей. Их менять не нужно для переезда на
новую машину — только `machine.local.yaml` и клиничные `settings.local.yaml`.

### 3. Внешние инструменты (устанавливаются/копируются отдельно, не из этого репо)

- **IVF Digital Twin** — `RUN_BATCH.bat` в своём проекте
  (`external_tools.digital_twin.project_dir`).
- **TabPFN batch-адаптер** — скопируйте `external/tabpfn/run_tabpfn_batch.py`
  и `.bat` в окружение TabPFN на новой машине, см. `docs/TABPFN_INSTALL.md`.
- **TabPFN-3 regression** — отдельный проект/venv с
  `tabpfn3_ivf_regression_report_CPU_16K.py`; мягкий шаг, гейт
  `external_tools.tabpfn_regression.enabled`.
- **DNN-проверка** (опционально) — укажите реальный `training_scaler.joblib`
  обучения; без него независимая проверка признаётся недействительной, `fit`
  на inference запрещён контрактом.

### 4. Проверка и тесты

```powershell
python scripts\check_setup.py --all-clinics
pytest -q
```

`check_setup.py` проверяет, что все обязательные пути/инструменты видны с
этой машины (`--all-clinics` — сразу для клиники A и всех `config/clinics/*`;
`--clinic <name>` — только одна). `pytest` гоняет синтетические e2e-фикстуры
(`tests/fixtures/`, никаких реальных пациентов) и контрактные тесты приватности.

### 5. Данные

Положите входную таблицу в `data/inbox/OPU_table.xlsx` (или укажите другой
путь в `config/settings.yaml: paths.opu_table`). MIRI-логи, Airtable-токен и
ПГТ-PDF переносятся на новую машину отдельно, вне git — см. раздел
«Приватность и границы данных» выше.

## Рабочий запуск

Откройте корень проекта в Claude Code и выполните:

```
/run-analysis
```

Skill сама вызовет `python scripts/run_analysis.py`, синхронизирует MIRI
(Google Drive MCP) и ПГТ-А (claude-in-chrome) для окна культивирования
review-list, запустит промежуточные роли-агенты и `lead-analyst`, затем
`scripts/finalize_report.py`.

## Ключевые выходные файлы (`data/processed/`)

| Файл | Назначение |
|---|---|
| `dt_predictions.csv` | DT + TabPFN + OPU join |
| `OPU_table_enriched.xlsx` | локальный OPU с прогнозами |
| `review_list.csv` | приоритет 1/2 и/или широкий CI TabPFN |
| `miri_events.csv`, `lot_events.csv`, `pgt_summary.csv`, `pgt_embryo_events.csv` | форензика окна культивирования |
| `ozema_cycles.csv`, `ozema_genetic_counseling_candidates.csv` | арест-паттерны |
| `regression_qc.csv`, `regression_calibration.csv` | TabPFN-3 QA факт vs прогноз |
| `DT_Analytics_Report.xlsx` / `.html` | итоговый отчёт, 13 вкладок |
| `agent_packets/analysis_bundle.json` | единственный файл, доступный агентам |
| `agent_packets/lead_analysis.md` | финальная интерпретация Opus-агента |
| `run_manifest.json` | версии схем, sha256 артефактов, статус DNN/regression |
| `local_private/cycle_uid_crosswalk.csv` | обратный join ID↔cycle_uid — только локально |

Контракты каждого артефакта версионированы и описаны в `docs/CONTRACTS.md`;
методология KPI и расхождения с первоисточником Vienna — в
`docs/METHODOLOGY.md`.
