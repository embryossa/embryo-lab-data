#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""teable_schema.py — declarative Teable schema for one clinic base.

One place defines every table and field; teable_bootstrap.py applies it (create
a new base or upgrade an existing one, never destructive) and teable_sync.py
writes against exactly these names. Adding a model output = adding a line here
plus a line in the sync's field builder.

Tables
    Runs                  one record per /run-analysis run (snapshot anchor)
    Cycles                cycle master, upserted by the cycle join key
    Predictions           every model output for one cycle in one run
    Prediction vs Reality funnel prediction-vs-fact with ±30% flags
    Regression QC         TabPFN-3 quantitative funnel forecast (P10/P50/P90)
    Review                the analytics review-list (priority / reason)
    OZEMA                 per-cycle oocyte-to-embryo arrest flags
    Calibration           per-run × model discriminative metrics
    Files                 the processed CSV/XLSX files as attachments

Naming note: table display names are "<short key> — <русское название>"; the
short key before the em dash is the stable handle used in code.
"""

from __future__ import annotations

# Cycle funnel stages shared with the report's "Прогноз беременности" journal
# (report._JOURNAL_FUNNEL_STAGES / _JOURNAL_REAL_COLUMN) so the Teable base and
# the HTML report describe the same funnel in the same order.
FUNNEL_STAGES = ("Фолликулы", "ООК", "MII", "2PN", "Бласт", "Хор.бласт")
FUNNEL_REAL_COLUMN = {
    "Фолликулы": "N folicules OPU", "ООК": "OCC", "MII": "MII",
    "2PN": "2pN", "Бласт": "Bl", "Хор.бласт": "Good Bl",
}
# TabPFN-3 regression stages -> candidate column prefixes in regression_qc.csv.
# Mirrors report._JOURNAL_REGRESSION_STAGE, plus the frozen stage which the
# regression emits but the journal tab does not show.
REGRESSION_STAGES = {
    "MII": ("mii",),
    "2PN": ("pn2",),
    "Бласт": ("blasts_after_fert", "blasts_early"),
    "Хор.бласт": ("good",),
    "Витрифицировано": ("frozen",),
}

CLUSTER_CHOICES = [{"name": "C0 Standard", "color": "blueLight1"},
                   {"name": "C1 Poor", "color": "redLight1"},
                   {"name": "C2 High", "color": "greenLight1"}]
CLUSTER_LABELS = {"0": "C0 Standard", "1": "C1 Poor", "2": "C2 High",
                  "c0": "C0 Standard", "c1": "C1 Poor", "c2": "C2 High"}

OUTCOME_CHOICES = [{"name": "Беременность", "color": "greenLight1"},
                   {"name": "Нет", "color": "redLight1"},
                   {"name": "Неизвестно", "color": "grayLight1"}]

FLAG30 = [{"name": "✅ в норме", "color": "greenLight1"},
          {"name": "⚠️ вне ±30%", "color": "redLight1"},
          {"name": "— н/д", "color": "grayLight1"}]

INTERVAL_CHOICES = [{"name": "✅ в P10–P90", "color": "greenLight1"},
                    {"name": "⚠️ вне P10–P90", "color": "redLight1"},
                    {"name": "— н/д", "color": "grayLight1"}]

PRIORITY_CHOICES = [{"name": "1 Критично", "color": "redLight1"},
                    {"name": "2 Рекомендовано", "color": "yellowLight1"},
                    {"name": "3 В норме", "color": "greenLight1"}]

# OZEMA per-stage verdicts. "н/д" (never "n/a" — collides with pandas na_values)
# is the n<4 result mandated by the project rules.
OZEMA_RESULT_CHOICES = [{"name": "flag", "color": "redLight1"},
                        {"name": "ok", "color": "greenLight1"},
                        {"name": "н/д", "color": "grayLight1"}]
OZEMA_FLAG_CHOICES = [{"name": n, "color": "redLight1"} for n in
                      ("OMA", "FF", "ZA", "EEA", "OMA+FF", "FF+EEA", "ZA+EEA",
                       "OMA+EEA", "OMA+ZA", "FF+ZA")] + \
                     [{"name": "—", "color": "greenLight1"},
                      {"name": "н/д", "color": "grayLight1"}]

MODEL_CHOICES = [{"name": m} for m in
                 ["MC", "Bayes", "KAT", "NVSA", "CSDI", "GNN", "DT", "TabPFN", "DNN", "PRAI", "Laya"]]

FILE_TYPE_CHOICES = [{"name": n, "color": c} for n, c in [
    ("dt_predictions", "blueLight1"), ("tabpfn_predictions", "cyanLight1"),
    ("OPU_table_enriched", "greenLight1"), ("regression_qc", "tealLight1"),
    ("review_list", "yellowLight1"), ("ozema_cycles", "orangeLight1"),
    ("kpi_computed", "purpleLight1"), ("DT_Analytics_Report", "pinkLight1"),
    ("other", "grayLight1")]]


# ---------------------------------------------------------------------------
# Field constructors
# ---------------------------------------------------------------------------

def txt():
    return {"type": "singleLineText"}


def url():
    return {"type": "singleLineText", "options": {"showAs": {"type": "url"}}}


def longtxt():
    return {"type": "longText"}


def num(precision):
    return {"type": "number",
            "options": {"formatting": {"type": "decimal", "precision": precision}}}


def date(withtime=False, tz="Asia/Tashkent"):
    return {"type": "date", "options": {"formatting": {
        "date": "YYYY-MM-DD", "time": "HH:mm" if withtime else "None", "timeZone": tz}}}


def sel(choices):
    return {"type": "singleSelect", "options": {"choices": list(choices)}}


def check():
    return {"type": "checkbox"}


def link(table_key):
    """Many-to-one link to another table in the same base (resolved at apply time)."""
    return {"type": "link", "_link_to": table_key,
            "options": {"relationship": "manyOne"}}


def attach():
    return {"type": "attachment"}


def f(name, spec):
    return {"name": name, **spec}


# ---------------------------------------------------------------------------
# The schema
# ---------------------------------------------------------------------------

def _regression_fields():
    out = []
    for stage in REGRESSION_STAGES:
        out += [f(f"{stage} факт", num(0)), f(f"{stage} P10", num(1)),
                f(f"{stage} P50", num(1)), f(f"{stage} P90", num(1)),
                f(f"{stage} интервал", sel(INTERVAL_CHOICES))]
    return out


def schema(tz="Asia/Tashkent"):
    """Ordered list of table definitions. Order matters: link targets first."""
    d = lambda withtime=False: date(withtime, tz)  # noqa: E731 - local alias
    return [
        {"key": "Runs", "name": "Runs — Прогоны анализа", "fields": [
            f("Run ID", txt()), f("Дата прогона", d(True)), f("Клиника", txt()),
            f("Входной файл", txt()), f("Пациентов всего", num(0)),
            f("С исходом", num(0)), f("Положительных исходов", num(0)),
            f("В ревью-листе", num(0)), f("Флагов OZEMA", num(0)),
            f("Версия DT", txt()), f("Карточки (вьюер)", url()),
            f("Заметки", longtxt())]},

        {"key": "Cycles", "name": "Cycles — Пациентки / циклы", "fields": [
            f("ID пациента", txt()), f("Имя", txt()), f("cycle_uid", txt()),
            f("Клиника", txt()), f("Карточка", url()),
            f("Возраст", num(0)), f("AMH", num(2)), f("AFC", num(0)), f("BMI", num(1)),
            f("Попытка", num(0)), f("Дата OPU", d()), f("Дата ET", d()),
            f("N фолликулов", num(0)), f("OCC", num(0)), f("MII", num(0)),
            f("Инсеминировано", num(0)), f("2pN", num(0)), f("Cleavage", num(0)),
            f("Bl", num(0)), f("Good Bl", num(0)), f("Cryo", num(0)),
            f("ET", num(0)), f("День ET", num(0)),
            f("IVF/ICSI", txt()), f("Эмбриолог", txt()), f("Врач", txt()),
            f("Исход", sel(OUTCOME_CHOICES)), f("Кластер", sel(CLUSTER_CHOICES)),
            # Snapshot of the latest run, denormalised onto the cycle master so
            # the "Поиск пациенток" view is self-sufficient: one searchable tab
            # showing who the patient is, what the models said and whether the
            # cycle needs attention — without opening any linked table.
            f("PRAI (посл.)", num(3)), f("DIGITAL TWIN (посл.)", num(3)),
            # Laya (laya_kan_ft) — внешний компаратор DT; пусто, если цикл для Laya непригоден.
            f("Laya (посл.)", num(3)),
            # Why the snapshot looks the way it does: empty for the report cohort,
            # filled for cycles scored by the off-cohort layer (scripts/
            # run_off_cohort.py) — e.g. "Без переноса, 2PN=0: только воронка DT".
            f("Охват прогноза", txt()),
            f("Ширина CI, пп", num(1)), f("Приоритет ревью", sel(PRIORITY_CHOICES)),
            f("Флаг OZEMA", sel(OZEMA_FLAG_CHOICES)), f("Есть флаг OZEMA", check()),
            f("Стадий вне интервала", num(0)),
            f("Прогонов", num(0)), f("Последний прогон", txt())]},

        {"key": "Predictions", "name": "Predictions — Прогнозы моделей", "fields": [
            f("Ключ", txt()), f("Пациентка", link("Cycles")), f("Прогон", link("Runs")),
            f("Timestamp", d(True)),
            # quantitative funnel medians (Digital Twin MC pipeline)
            f("med OCC", num(1)), f("med MII", num(1)), f("med 2PN", num(1)),
            f("med Bl", num(1)), f("med Good", num(1)), f("med Euploid", num(1)),
            # pregnancy probability, model by model
            f("PRAI (TabPFN)", num(3)), f("PRAI CI низ", num(3)), f("PRAI CI верх", num(3)),
            f("Ширина CI, пп", num(1)), f("Широкий CI", check()),
            f("DIGITAL TWIN", num(3)), f("P per transfer", num(3)),
            f("P overall cycle", num(3)), f("P cum if viable", num(3)),
            f("Bayes mean", num(3)), f("Bayes CI низ", num(3)), f("Bayes CI верх", num(3)),
            f("KAT", num(3)), f("NVSA", num(3)), f("CSDI", num(3)), f("GNN", num(3)),
            f("DNN", num(3)), f("Статус DNN", txt()),
            # Laya (laya_kan_ft): внешний компаратор DT, не член ансамбля L1–L7.
            # «Laya» — с калибровкой по центру (RBMO, сдвиг интерсепта), «Laya сырая» — без неё.
            # Для непригодного цикла (нет обязательного входа или перенос невозможен) обе пусты.
            # «Laya−DT, пп» и «Расхождение высокое» — обе модели после калибровки по центру;
            # порог |logit| 0.354 (верхняя треть RBMO, где Brier DT выше) — повод перепроверить, не вердикт.
            f("Laya", num(3)), f("Laya сырая", num(3)), f("Laya пригодна", check()),
            f("Laya−DT, пп", num(1)), f("Расхождение высокое", check()),
            f("Флаги Laya", longtxt()),
            # Availability flags straight from the Digital Twin run: without them
            # an empty GNN/CSDI column is indistinguishable from "the model said
            # nothing for this cycle", and the model silently not loading at all
            # looks the same as missing data.
            f("GNN доступен", check()), f("CSDI доступен", check()),
            # cycle-level risk / structure
            f("Кластер", sel(CLUSTER_CHOICES)), f("C0 prob", num(3)),
            f("C1 prob", num(3)), f("C2 prob", num(3)),
            f("P viable", num(3)), f("P cancel", num(3)),
            f("OHSS умеренный", num(3)), f("OHSS тяжёлый", num(3)),
            f("P без бластоцист", num(3)), f("P без хор. бластоцист", num(3)),
            f("Banking P на MII", num(3)), f("Banking ожид. эуплоидных", num(2)),
            f("Источник NN", longtxt())]},

        {"key": "Prediction vs Reality", "name": "Prediction vs Reality — Прогноз vs Факт",
         "fields": [
            f("Ключ", txt()), f("Пациентка", link("Cycles")), f("Прогон", link("Runs")),
            f("OCC факт", num(0)), f("OCC прогноз", num(1)), f("OCC ±30%", sel(FLAG30)),
            f("Bl факт", num(0)), f("Bl прогноз", num(1)), f("Bl ±30%", sel(FLAG30)),
            f("Good Bl факт", num(0)), f("Good Bl прогноз", num(1)),
            f("Good ±30%", sel(FLAG30)),
            f("KAT−PRAI (пп)", num(1)), f("Исход", sel(OUTCOME_CHOICES)),
            f("Приоритет ревью", sel(PRIORITY_CHOICES)), f("Основание", longtxt())]},

        {"key": "Regression QC", "name": "Regression QC — TabPFN-3 воронка", "fields": [
            f("Ключ", txt()), f("Пациентка", link("Cycles")), f("Прогон", link("Runs")),
            *_regression_fields(),
            f("Стадий вне интервала", num(0))]},

        {"key": "Review", "name": "Review — Ревью-лист", "fields": [
            f("Ключ", txt()), f("Пациентка", link("Cycles")), f("Прогон", link("Runs")),
            f("Приоритет", sel(PRIORITY_CHOICES)), f("Основание", longtxt()),
            f("PRAI", num(3)), f("CI низ", num(3)), f("CI верх", num(3)),
            f("Ширина CI, пп", num(1)), f("Широкий CI", check()),
            f("DIGITAL TWIN", num(3)), f("Исход", sel(OUTCOME_CHOICES)),
            f("Дата OPU", d()), f("Дата ET", d()), f("Карточка", url())]},

        {"key": "OZEMA", "name": "OZEMA — Арест-паттерны цикла", "fields": [
            f("Ключ", txt()), f("Пациентка", link("Cycles")), f("Прогон", link("Runs")),
            f("Дата OPU", d()), f("Возраст", num(0)),
            f("COC", num(0)), f("MII", num(0)), f("2PN", num(0)),
            f("Cleav", num(0)), f("Bl", num(0)),
            f("Флаг", sel(OZEMA_FLAG_CHOICES)), f("Есть флаг", check()),
            f("OMA", sel(OZEMA_RESULT_CHOICES)), f("FF", sel(OZEMA_RESULT_CHOICES)),
            f("ZA", sel(OZEMA_RESULT_CHOICES)), f("EEA", sel(OZEMA_RESULT_CHOICES)),
            f("Флагов у пациентки", num(0)),
            f("Кандидат на генетическое консультирование", check()),
            f("Примечание", longtxt()), f("Эмбриолог", txt()), f("Врач", txt())]},

        {"key": "Calibration", "name": "Calibration — Калибровка моделей", "fields": [
            f("Ключ", txt()), f("Прогон", link("Runs")), f("Модель", sel(MODEL_CHOICES)),
            f("Brier", num(4)), f("AUC-ROC", num(4)), f("Acc@0.5", num(4)),
            f("Mean pred", num(4)), f("Real rate", num(4)), f("N", num(0))]},

        {"key": "Files", "name": "Files — Файлы прогонов", "fields": [
            f("Имя файла", txt()), f("Тип", sel(FILE_TYPE_CHOICES)),
            {"name": "Файл", "type": "attachment"}, f("Прогон", link("Runs")),
            f("Размер, КБ", num(1)), f("Строк", num(0)),
            f("Путь (локальный)", txt()), f("SHA-256", txt()), f("Загружено", d(True))]},
    ]


def table_keys():
    return [t["key"] for t in schema()]


# ---------------------------------------------------------------------------
# Views — the tabs the owner actually works in
# ---------------------------------------------------------------------------

# The columns of the patient-search tab, in display order. The first entry is
# the primary field and must stay visible. Everything not listed is hidden in
# that view (the data is still in the table, just out of the way).
_SEARCH_COLUMNS = [
    "ID пациента", "Имя", "Дата OPU", "Возраст", "AMH", "Исход",
    "PRAI (посл.)", "DIGITAL TWIN (посл.)", "Laya (посл.)", "Охват прогноза", "Приоритет ревью", "Флаг OZEMA",
    "OCC", "2pN", "Bl", "Good Bl", "Кластер", "Карточка",
    "Эмбриолог", "Врач", "Дата ET", "Прогонов", "cycle_uid",
]


def views():
    """Table key -> list of view definitions.

    Filters are (field name, operator, value) triples resolved to field ids at
    apply time; `columns` lists the fields left visible, in order.
    """
    return {
        "Cycles": [
            {"name": "🔎 Поиск пациенток", "type": "grid",
             "columns": _SEARCH_COLUMNS,
             "sort": [("Дата OPU", "desc")]},
            {"name": "⚠️ Ревью: приоритет 1–2", "type": "grid",
             "columns": _SEARCH_COLUMNS,
             "sort": [("Приоритет ревью", "asc"), ("Дата OPU", "desc")],
             "filter": [("Приоритет ревью", "isAnyOf",
                         ["1 Критично", "2 Рекомендовано"])]},
            {"name": "🧬 OZEMA: флаги", "type": "grid",
             "columns": _SEARCH_COLUMNS,
             "sort": [("Дата OPU", "desc")],
             "filter": [("Есть флаг OZEMA", "is", True)]},
            {"name": "❔ Без известного исхода", "type": "grid",
             "columns": _SEARCH_COLUMNS,
             "sort": [("Дата OPU", "desc")],
             "filter": [("Исход", "is", "Неизвестно")]},
            {"name": "📈 Широкий CI TabPFN", "type": "grid",
             "columns": _SEARCH_COLUMNS,
             "sort": [("Ширина CI, пп", "desc")],
             "filter": [("Ширина CI, пп", "isGreater", 20)]},
        ],
        "Predictions": [
            {"name": "🕓 История прогнозов", "type": "grid",
             "sort": [("Timestamp", "desc")]},
        ],
        "Review": [
            {"name": "⚠️ Приоритет 1", "type": "grid",
             "sort": [("Дата OPU", "desc")],
             "filter": [("Приоритет", "is", "1 Критично")]},
        ],
    }
