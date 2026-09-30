---
name: lot-forensics
description: Airtable consumable-lot schema (расходные материалы + Barcode Changes History), timeline construction, and cultivation-window correlation logic. Preload into agents that analyze consumable lot changes against embryo cultivation windows.
---
# Форензик смены лотов расходников (Airtable)

Источник: Airtable-база лаборатории (ID базы и таблиц — `consumable_lots.airtable` в
`config/settings.local.yaml`), две таблицы.

- «расходные материалы» (`items_table_id`): позиция (Наименование, Производитель,
  Артикул, атрибут-категория среда/пластик/микроинструменты/материалы, текущий
  Barcode, Start/End date).
- «Barcode Changes History» (`changes_table_id`): одна строка на каждую смену
  лота/скан по конкретной позиции — `Change Date`, значение лота (`All Barcodes`),
  ссылка на позицию.

`src/embryo/lots.py` строит из «Barcode Changes History» таймлайн лотов на позицию:
периоды `period_start .. period_end` (конец = дата следующей смены, `NaT` — лот ещё
активен). Позиции без единой строки смены в таймлайн не попадают — сверять нечего.

Окно культивирования: `Date OPU .. +N дней` (`settings.thresholds.cultivation_window_days`,
по умолчанию 6), то же окно, что уже используется для MIRI-форензика.

Семантика на позицию за окно:
- ровно один пересекающийся период → `event_type=stable` (лот не менялся).
- больше одного пересекающегося периода → одна строка `event_type=lot_change` на
  каждую смену, дата которой попала внутрь окна (`lot_before -> lot_after`,
  `change_date`).

Сводка на цикл (`cycle_uid`): `n_items_tracked`, `n_items_changed`, `changed_items`,
`lot_change_flag`. Флаг называется `lot_change_flag`, не `warning` — это не то же
понятие, что MIRI severity warning/critical, семантика правил MIRI на него не
распространяется.

Ограничение: сама по себе смена лота внутри окна культивирования — кандидат-
confounder, не причина и не оценка качества конкретного лота или производителя.
Формулировать связь с исходом переноса только как ГИПОТЕЗУ.
