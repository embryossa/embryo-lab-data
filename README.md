# Embryo Lab Data Analytics

Локальный end-to-end анализ лаборатории эмбриологии: Excel/CSV → нормализация →
KPI → IVF Digital Twin + TabPFN + опциональная DNN → OPU → DataLore-аналитика →
MIRI 6391 → XLSX/HTML и псевдонимизированный разбор агентами Claude Code.

## Первый запуск на лабораторном ПК

```powershell
cd "C:\Users\User\Desktop\IVF\AI\embryo-lab-data"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy config\settings.local.yaml.example config\settings.local.yaml
python scripts\check_setup.py
pytest -q
```

Перед `check_setup.py`:

1. Скопируйте `external\tabpfn\run_tabpfn_batch.py` и `.bat` в
   `C:\Users\User\Desktop\IVF\AI\TABPFN\` — см. `docs/TABPFN_INSTALL.md`.
2. Укажите реальный scaler обучения DNN в `config/settings.local.yaml`. DNN выключена
   по умолчанию и не признаётся независимой проверкой без scaler.
3. Положите входной Excel/CSV в `data\inbox\`. OPU master по умолчанию —
   `data\inbox\OPU_table.xlsx`.
4. Настройте `miri_logs.base_dir`: локальная синхронизированная папка Google Drive `Miri`
   либо `data\miri_logs`. Структура `6391\Miri-6391-Incubation-[...]\Data\*.csv`
   читается рекурсивно и только на чтение.

## Рабочий запуск

Откройте корень проекта в Claude Code и выполните:

```text
/run-analysis data/inbox/OPU_table.xlsx
```

Отдельный BAT для основного проекта не нужен. Claude Code запускает единый Python-
оркестратор; BAT используется только как адаптер внутри внешнего проекта TabPFN.

Итоги находятся в `data/processed/`:

- `dt_predictions.csv` — реальный DT + TabPFN join;
- `OPU_table_enriched.xlsx` — локальный OPU с прогнозами;
- `review_list.csv` — P1/P2 и/или CI TabPFN >20 пп;
- `miri_events.csv` — эпизоды, gaps и recovery;
- `DT_Analytics_Report.xlsx` и `.html`;
- `agent_packets/analysis_bundle.json` — единственный файл пациентов, доступный агентам;
- `local_private/cycle_uid_crosswalk.csv` — обратный join, только локально.

Контракты описаны в `docs/CONTRACTS.md`. Референсный DataLore notebook сохранён в
`notebooks/reference/`, его детерминированная логика перенесена в `src/embryo/analytics.py`.

Это аналитическая система поддержки разбора, не медицинское изделие и не механизм
автоматического назначения лечения. Причинные объяснения в отчёте являются гипотезами.
