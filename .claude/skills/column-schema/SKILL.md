---
name: column-schema
description: Canonical column schema, alias resolution rules, and transfer-outcome binarization logic for IVF lab data exports. Preload into agents that reconcile unknown columns or outcome strings.
---
# Каноническая схема и правила исхода

Источник соответствий — config/columns.yaml. Резолвер приводит любую входную колонку
к канону; неизвестная -> предупреждение (не угадывать молча). Коллизии (2+ колонки в
один канон) -> оставляется первая, выдаётся warning.

Важные различия:
- embryos_d5 («Число эмбрионов 5 дня») ≠ bl_total («Число Bl»): первое — общее число
  эмбрионов на D5, второе — именно бластоцисты.
- afc («Antral follicule count») ≠ follicles («N folicules OPU»): АФК — базовый, идёт в
  импутацию AMH; follicles — фолликулы на пункции, идёт в KPIScore.

Исход -> бинарь (normalize.py): Preg=1 = клиническая беременность ИЛИ утвердительное
да/yes; всё прочее заполненное -> 0; пусто -> NaN (строка отсеивается фильтром).
Новые варианты колонок/исходов добавлять в config/columns.yaml (секции canonical/outcome),
арифметику KPI не трогать.
