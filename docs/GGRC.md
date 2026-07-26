# GGRC (Armenia) — правила ingest/normalize

Согласовано с владельцем 2026-07-23. Этот файл — источник истины по *почему*;
код — в `src/embryo/clinics/ggrc.py` и `config/clinics/ggrc/columns.yaml`.
Архитектура: общий `run_analysis.py`/`kpi.py`/`normalize.py`/`report.py`/
роли-агенты не меняются под GGRC — вся специфика живёт в clinic-adapter слое,
тот же принцип, что у Fertimed/Skyfert (`docs/FERTIMED.md`, `docs/SKYFERT.md`).

Полный прогон (Digital Twin, TabPFN, та же DNN, что у Fertimed, KPI, HTML-отчёт)
— **без** MIRI-логов инкубатора и **без** учёта лотов расходников (владелец
подтвердил явно). PGT-А не требует парсинга PDF/скачивания с портала — данные
уже в самой выгрузке.

## Два файла вместо одного

В отличие от всех остальных клиник этого проекта (один входной файл, срезаемый
фильтром `transferred>0 & Preg.notna()` на "eligible"/"widе lab" подмножества),
GGRC поставляет их как ДВА ОТДЕЛЬНЫХ файла:

- `ivf_master_13 (transfer).xlsx` (29 строк) — курируемая владельцем клиники
  когорта: циклы с переносом и известным исходом. Полный пайплайн (DT/TabPFN/
  DNN, KPI, review-list) строится по ней — резолвится `ggrc.prepare()`.
- `ivf_master_13 kpi.xlsx` (лист `IVF Master DB`, 101 строка) — широкая
  lab-популяция (включает freeze-all/без-переноса циклы). Используется
  **только** для KPI-агрегации (таблицы компетенции эмбриолог/врач) и OZEMA —
  резолвится `ggrc.prepare_wide()`. Не несёт исход переноса вообще (в файле
  нет колонки-аналога `Pregnancy`) — не проблема: ни `kpi.KPI_SPECS`, ни
  `ozema.build_ozema_table` не читают `Preg`.

Все 29 `OPU Code` transfer-файла присутствуют в kpi-wide файле с совпадающей
`OPU Date` (проверено на реальной выгрузке) — поэтому ОБЩИЙ (не
Skyfert-специфичный) `_key = patient_id + "#" + date_opu` в
`scripts/run_analysis.py::_add_opu_aliases` уже даёт одинаковый `cycle_uid`
для одного и того же физического цикла в обоих файлах. Кастомный
`build_cycle_key` (как у Skyfert) не понадобился.

Диспетчеризация в `scripts/run_analysis.py`:
`_CLINIC_NORMALIZERS["ggrc"] = ggrc.prepare`,
`_CLINIC_WIDE_LOADERS["ggrc"] = ggrc.prepare_wide` (новый реестр — единственная
клиника, где `lab_cycles` строится из отдельного файла, а не среза того же
`normalized`), `_CLINIC_PGT_MODULES["ggrc"] = ggrc`,
`_CLINIC_PGT_EXTRA_COLUMNS["ggrc"] = ggrc.PGT_RAW_COLUMNS`.

## follicles — оценка, не измерение

Владелец подтвердил (2026-07-23): использовать оценку из черновика
`Draft_for_NN.py` как есть — `follicles = round(Oocytes N × 1.44)`.
Следствие: `OCCR = OCC/follicles` (KPI врача) читается как почти константа
≈0.69 для каждого цикла — это ожидаемо, не баг конвейера, а свойство входных
данных. Интерпретировать OCCR для GGRC с этой оговоркой (нет реального сигнала
о компетенции врача на этапе пункции); MIIR (`MII/OCC`) считается как обычно —
реальный MII в выгрузке есть.

Формулировка причины уточнена 2026-07-26. Ранее здесь было сказано «нет реальной
колонки ни в одном из двух файлов» — для старых выгрузок (`ivf_master_13
(transfer).xlsx`, `ivf_master_13 kpi.xlsx`) это верно, но выгрузка 2025 года
(`предсказания 2025.xlsx`) колонку `number of follicles` / `Количество
фолликулов` СОДЕРЖИТ, заполненную во всех 561 строках. Подключать её не следует:
проверка 2026-07-26 показала, что она сама производна от OCC — 50 уникальных
значений OCC дают ровно 50 уникальных пар (OCC, follicles), отношение
foll/OCC зажато в 1.21–1.23 (формула ≈ `ceil(OCC × 1.21)`), а все строки с
`follicles = 0` — это строки с `OCC = 0`. Независимого подсчёта фолликулов в ней
нет. Подключение заменило бы константу OCCR ≈0.69 на константу ≈0.806, не добавив
сигнала, но при этом 80.6% попадает в венский коридор competency (70–80%) и
метрика начала бы читаться как «врач в норме» — то есть стала бы опаснее
нынешнего явного артефакта. Решение владельца (2026-07-26): оставить оценку 1.44
как есть. В широком KPI-файле (`KPI 2025.xlsx`) колонки фолликулов нет и в
2025-й выгрузке.

## Composite-исход переноса

Колонка `Pregnancy` содержит строки вида `"No"`, `"No, No"`, `"No, Yes"`,
`"1 No, 1 Yes"`, `"Biochemical, No"`, `"No, No, Yes"` — по одному токену на
перенесённый эмбрион/попытку внутри одной строки цикла. Владелец подтвердил
(2026-07-23): `Preg=1`, если хотя бы один токен содержит `"yes"`
(регистронезависимо), иначе `0`. `"Biochemical"` и `"No"` сами по себе не
считаются положительным исходом. Реализовано в `ggrc._classify_outcome` —
собственная функция, не общий `normalize.outcome_to_binary()` (composite-
формат специфичен для этой клиники).

## Эмбриолог/врач — атрибуция KPI

Лабораторный KPI (FR/ClR/BlR/TGBDR/DegR/3pNR) атрибутируется колонке
`Embryologist IVF/ICSI` (эмбриолог, непосредственно проводивший оплодотворение
— владелец подтвердил 2026-07-23, тот же принцип приоритета, что у Fertimed
"Эмбриолог, проводивший оплодотворение"). Заполнена в 100/101 строк kpi-wide
файла — priority-fallback цепочка (как у Fertimed) не понадобилась.

KPI врача (OCCR/MIIR) — единственная колонка `Doctor` (нет разделения
пункция/перенос, как у Fertimed `doctor_opu`/`doctor_et`).

`competency_exclusion_mask` (NO-retrieval/freeze-all токены против
`fertilization_method`) — фактический no-op для GGRC: `Fertilization Method`
здесь означает метод оплодотворения (ICSI/IVF), а не тип цикла, и в выгрузке
нет отдельного треатмент-флага вроде `"Лечение"`/"Криоперенос" у Fertimed.
Действует только критерий "открытый цикл" (`bl_total` не заполнен).

## Отклонения от чернового скрипта (Draft_for_NN.py)

Черновик — отправная точка, не спецификация; две его колонки — вероятно,
placeholder-и, а не согласованные владельцем правила:

- **`frozen` (Заморожено эмбрионов):** черновик присваивает это поле равным
  `Число Bl хор.кач-ва` (число хороших бластоцист) — маловероятно, что это
  замысел, поскольку в выгрузке есть настоящие колонки заморозки:
  `Cryo: Day 5 Embryos` + `Cryo: Day 6 Embryos` + `Cryo: Day 7 Embryos` +
  `Cryo: Other Blastocysts`. `ggrc._frozen_sum` суммирует эти четыре
  (`min_count=1`, тот же паттерн, что у Fertimed `_frozen_sum`).
- **`embryos_d5`:** черновик дублирует туда значение `cleavage_d3`. Поле
  необязательное (не участвует ни в одной формуле KPI/DT), поэтому здесь
  оставлено `NaN`, а не размноженное значение.

## Отсутствующие поля в kpi-wide файле

`ivf_master_13 kpi.xlsx` физически не содержит колонок-аналогов
`inseminated` ("Число инсеминированных") и `cleavage_d3` ("Число дробящихся на
3 день") — только `ivf_master_13 (transfer).xlsx` их несёт. Следствие для
широкой lab-популяции (таблицы KPI эмбриолог/врач, KPI_SPECS считается по
`lab_cycles`):

- `ClR` (Cleavage rate = cleavage_d3/2pN) — `insufficient_data` для каждой
  строки embryologist-таблицы (не крэш, штатное поведение `_numeric_column`
  при отсутствующей колонке целиком).
- `FR`/`DegR` денаминатор не деградирует: `_fr_denominator` берёт MII, если
  он известен и >0 (в kpi-wide файле MII заполнен полностью), к
  `inseminated` обращается только как к fallback.

`bl_total` РЕЗОЛВИТСЯ в обоих файлах, но под разными исходными заголовками:
`"CombinedDays"` в transfer-файле, `"D5-D6-D7"` в kpi-wide файле — одно и то
же поле (сумма Day5+Day6-7 бластоцист), оба алиаса в одном каноне
`config/clinics/ggrc/columns.yaml`.

## ПГТ-А — уже в выгрузке, не парсинг PDF

В отличие от клиники A (`pa.ivf.uz`, скачивание PDF интерактивно) и Fertimed
(`my.f-genetics.com`, тот же принцип), GGRC уже несёт когортные счётчики ПГТ-А
как обычные колонки в обоих файлах: `Biopsy Done`, `Tested Embryos`,
`Not Tested Embryos`, `Euploid`, `Mosaic`, `Failed/NA`, `Pending Result`,
`Aneuploid`. Владелец подтвердил (2026-07-23): "парсить и искать отчеты не
требуется, но в аналитику данные из этих колонок включить стоит".

`ggrc.analyze_reports(cohort, reports_dir, ...)` — та же сигнатура вызова, что
`embryo.pgt.analyze_reports` (дропается в `_CLINIC_PGT_MODULES` без изменений
диспетчера), но `reports_dir` игнорируется (PDF не ищутся) — сводка строится
прямо из `PGT_RAW_COLUMNS` строки `cohort`. Нет данных о конкретном эмбрионе
(кариотип и т.п.), только когортные счётчики — `EMBRYO_COLUMNS = []`, вкладка
"PGT Embryo Events" для GGRC всегда пуста, "PGT Summary" заполнена.
`scripts/run_analysis.py` расширяет `worklist_cols` шага ПГТ на
`ggrc.PGT_RAW_COLUMNS` только для этой клиники (`_CLINIC_PGT_EXTRA_COLUMNS`) —
иначе сырые колонки не попали бы в кадр, который видит `analyze_reports`.

## Прочие поля выгрузки, намеренно не мапятся

`Husband name`, `Oocyte Source` (донор/собственные), `Embryologist
denudation`/`Embryologist OPU`/`Embryologist on Biopsy` (роли, отличные от
`Embryologist IVF/ICSI`), `Transfer: Doctor`/`Transfer: Embryologist
Vitrification/Thawing/Transfer` (отдельные роли на переносе), `Transfer:
Surrogacy Program`/`Transfer: Surrogate Mother`, `description`/`Transfer:
Morphology`/`Transfer: Notes after Thawing` (текст морфологии эмбрионов),
`Cryo: Date of OPU`/`Cryo: Vitrification Dates` — не используются этим
отчётом, попадают в unknown-предупреждение резолвера. Это ожидаемо, не
ошибка.

## Данные

- Копии обеих реальных выгрузок лежат в gitignored `data/inbox/ggrc/transfer/`
  и `data/inbox/ggrc/kpi_wide/` (раздельные подпапки, чтобы автоопределение
  входного файла по `inbox_dir` не подхватило оба разом).
- `предсказания_метамодель.xlsx` (сторонняя модель владельца) НЕ копируется в
  проект и нигде не используется пайплайном — явное указание владельца.

## Запуск

```
python scripts/run_analysis.py --clinic ggrc
```

Через skill: `/run-analysis --ggrc` (см. `.claude/skills/run-analysis/SKILL.md`).
Результаты — та же структура `data/processed/ggrc/`, что и у Fertimed/клиники
A: `DT_Analytics_Report.html`/`.xlsx`, `dt_predictions.csv`,
`OPU_table_enriched.xlsx`, `review_list.csv`, `pgt_summary.csv`,
`ozema_cycles.csv`, `run_manifest.json`, `agent_packets/analysis_bundle.json`.
Вкладки MIRI 6391 и "Партии расходников" покажут "нет данных" — тот же
механизм, которым уже пользуется Skyfert для отсутствующих у неё разделов, не
новый код.

## Открытые задачи

- `external_tools.tabpfn_regression.enabled: false` в
  `config/clinics/ggrc/settings.yaml` — выключено для первого прогона
  (Skyfert-прецедент: модель не проверялась на профиле данных GGRC, очень
  маленькая когорта n=29). Включить: `enabled: true`, разовая правка.
- `amh_impute`/`bmi_impute` унаследованы от клиники A/Fertimed как отправная
  точка — GGRC не заявляла собственную формулу (AMH/BMI отсутствуют в
  выгрузке целиком, импутируются по умолчанию).
