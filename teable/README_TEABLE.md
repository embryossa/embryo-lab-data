# IVF Digital Twin → Teable: настройка и автозагрузка

Организация результатов `/run-analysis` (batch `dt_batch.py`) в Teable, по одной базе на клинику, с автоматической загрузкой между запусками.

## ⚠️ 2026-07-21: переезд на self-hosted Teable

Облачный спейс `spc4wPGypHLxXSI6JWO` (`app.teable.ai`) — free-план, лимит **1000 строк
на весь спейс**. Уже существующая в том же спейсе база «CRYO» (не относится к этому
пайплайну) занимала 909 из 1000, так что реальная синхронизация обеих клиник
падала на середине с `402 Payment Required: "Exceed the maximum number of rows"`
— молча, без явного зависания, но и без полной записи данных.

Решение — self-hosted Teable в Docker (`teable/self-hosted/docker-compose.yaml`,
официальный standalone-образ `ghcr.io/teableio/teable`, без лимита строк). Все
bat-файлы и `teable_config.local.json` теперь по умолчанию пишут именно туда
(`http://127.0.0.1:3000`). Старый `teable_config.json` (облако) оставлен как есть
на случай, если понадобится вернуться — но в нём тоже был баг с именем файла
(`OPU_table_filled.xlsx` → должно быть `OPU_table_enriched.xlsx`), из-за которого
реальные данные пациенток (возраст, AMH, ОКК, исход) молча не подхватывались;
это тоже исправлено.

Поднять контейнеры (если выключены): `cd teable\self-hosted && docker compose up -d`.

## Что уже сделано

В локальном self-hosted Teable созданы две базы: **«Two-Line IVF — Digital Twin»**
(`bse4b0RSHeKI7ZGEdvC`) и **«Fertimed — Digital Twin»** (`bsef6WTgVoVeJeJD1N7`), у
каждой — шесть связанных таблиц:

| Таблица | Назначение | Источник |
|---|---|---|
| **Runs — Прогоны анализа** | Один запуск `/run-analysis` = одна запись (снимок). Якорь, к которому привязано всё остальное «между запусками». | метаданные прогона |
| **Cycles — Пациентки / циклы** | Мастер пациенток. Upsert по `ID пациента`, не дублируется между прогонами. | `OPU_table_filled.xlsx` + медианы из `dt_predictions.csv` |
| **Predictions — Прогнозы DT** | Выходы моделей: медианы воронки, P per transfer, Bayes, KAT, NVSA, CSDI, DIGITAL TWIN, OHSS, banking. Одна запись на пациентку на прогон. | `dt_predictions.csv` |
| **Prediction vs Reality — Прогноз vs Факт** | Воронка прогноз/факт с флагами ±30%, KAT−PRAI, приоритет ревью. | `dt_vs_real_comparison.csv` / расчёт |
| **Calibration — Калибровка моделей** | На прогон × модель: Brier, AUC-ROC, Acc@0.5, Mean pred, Real rate, N. | `DT_Analytics_Report.xlsx` (WithOutcomes) |
| **Files — Файлы прогонов** | Сами промежуточные CSV/XLSX как **вложения**, привязанные к прогону — «упорядочить файлы из папки». | все файлы из `processed` |

## Файлы в этом пакете

```
teable_sync.py                 главный скрипт синхронизации processed -> Teable
teable_config.example.json     шаблон конфига (скопируйте в teable_config.json)
teable_bootstrap.py            создать такую же схему в новой базе (для новой клиники)
requirements.txt               зависимости Python
run_sync.bat                   ручной запуск синхронизации (Windows)
run_analysis_and_sync.bat      /run-analysis + автозагрузка одной кнопкой
README_TEABLE.md               этот файл
```

## Шаг 1. Зависимости

```bat
pip install -r requirements.txt
```

`scikit-learn` и `numpy` нужны только для таблицы Calibration; без них синхронизация работает, метрики просто пропускаются.

## Шаг 2. Токен доступа Teable

1. Откройте `https://app.teable.ai` → аватар справа снизу → **Настройки** → **Personal access tokens** (Токены доступа).
2. Создайте токен. Права: как минимум `record|read`, `record|write`, `table|read`, а для загрузки файлов — доступ к нужному спейсу/базе. Проще выдать полный доступ к спейсу `spc4wPGypHLxXSI6JWO`.
3. Скопируйте токен (он показывается один раз).

> Токен — это как пароль. Держите его только в `teable_config.json` на своём ПК, не публикуйте и не отправляйте никому.

## Шаг 3. Конфиг

Текущий рабочий конфиг — `teable_config.local.json` (self-hosted, не в git):

```json
{
  "teable_url": "http://127.0.0.1:3000",
  "token": "ваш_локальный_токен",
  "base_id": "bse4b0RSHeKI7ZGEdvC",
  "clinic": "Two-Line IVF",
  "processed_dir": "C:\\Users\\User\\Desktop\\IVF\\AI\\embryo-lab-data\\data\\processed"
}
```

Если имена колонок в ваших файлах отличаются — поправьте словари `OPU_COLS` / `PRED_COLS` вверху `teable_sync.py` (там уже перечислены частые варианты, включая русские).

## Шаг 4. Первый запуск

```bat
python teable_sync.py --dry-run     REM разобрать и показать, ничего не писать
python teable_sync.py               REM залить в Teable
```

Скрипт напечатает, сколько пациенток и исходов распознал, и создаст прогон.

## Идемпотентность «между запусками»

- Каждый прогон помечается **хешем содержимого** `dt_predictions.csv`. Повторный запуск синхронизации на тех же файлах ничего не дублирует (`[skip] ... hash ...`). Нужно всё равно залить — `--force`.
- **Cycles** обновляются по `ID пациента` (одна пациентка — одна запись, накопление истории идёт через Predictions/PvR по разным прогонам).
- Предыдущие прогоны остаются в базе — так копится история между запусками.

## Шаг 5. Автозапуск

Вариант А — **вместе с анализом** (рекомендуется). Отредактируйте пути вверху `run_analysis_and_sync.bat` и запускайте его вместо ручного `/run-analysis`: он прогонит `dt_batch.py`, затем зальёт результат.

Вариант Б — **по расписанию** (Планировщик заданий Windows). Так как синхронизация идемпотентна, её безопасно гонять периодически:

1. «Планировщик заданий» → **Создать задачу**.
2. Триггер: например, каждые 15 минут, или «при входе в систему».
3. Действие → «Запуск программы»:
   - Программа: `C:\путь\к\run_sync.bat`
   - (в `run_sync.bat` уберите строку `pause`, чтобы окно закрывалось само).
4. Готово — новые прогоны будут подхватываться автоматически.

Быстрый вариант той же задачи из командной строки (подставьте свой путь):

```bat
schtasks /Create /TN "Teable IVF Sync" /TR "C:\path\run_sync.bat" /SC MINUTE /MO 15 /RL LIMITED
```

## Новая клиника = новая база

«Одна база на клинику». Чтобы развернуть идентичную структуру для, например, Lapino:

```bat
python teable_bootstrap.py --space spc4wPGypHLxXSI6JWO --name "Lapino IVF — Digital Twin"
```

Скрипт создаст базу и все 6 таблиц, напечатает новый `base_id`. Сделайте отдельный `teable_config_lapino.json` с этим `base_id` и своей папкой `processed`, и запускайте:

```bat
python teable_sync.py --config teable_config_lapino.json
```

## Проверка

- Базу и таблицы можно открыть: `https://app.teable.ai/base/bseXUoWLEGxBYj1PKIJ`
- После первого реального прогона проверьте: в **Runs** появилась запись, в **Cycles** — пациентки, в **Files** — вложенные CSV/XLSX.

## Заметка о приватности

Teable Cloud — внешний сервер. Вы выбрали облако осознанно; если по требованиям к персональным данным это станет важно, тот же `teable_bootstrap.py` и `teable_sync.py` работают и с локально развёрнутым Teable (Docker) — меняется только `teable_url` в конфиге.
