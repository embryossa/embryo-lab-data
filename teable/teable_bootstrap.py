#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
teable_bootstrap.py — create the IVF Digital Twin 6-table schema in a Teable base.

Use this to stand up an identical base for EACH clinic ("one base per clinic").
It either creates a brand-new base inside a space, or fills an existing empty base.

Examples:
    # create a new base named for the clinic inside a space, then build tables:
    python teable_bootstrap.py --space spc4wPGypHLxXSI6JWO --name "Lapino IVF — Digital Twin"

    # build the schema inside an existing (empty) base:
    python teable_bootstrap.py --base bseXXXXXXXXXXXXXXX

Auth: set the token in teable_config.json (same file the sync uses) or pass --token.
Dependency:  pip install requests
"""

import argparse
import json
import os
import sys
import requests

TZ = "Asia/Tashkent"  # change per clinic if needed


def num(p):
    return {"type": "number", "options": {"formatting": {"type": "decimal", "precision": p}}}


def date(withtime=False):
    return {"type": "date", "options": {"formatting": {
        "date": "YYYY-MM-DD", "time": "HH:mm" if withtime else "None", "timeZone": TZ}}}


def sel(choices):
    return {"type": "singleSelect", "options": {"choices": choices}}


def txt():
    return {"type": "singleLineText"}


def longtxt():
    return {"type": "longText"}


FLAG = [{"name": "✅ в норме", "color": "greenLight1"},
        {"name": "⚠️ вне ±30%", "color": "redLight1"},
        {"name": "— н/д", "color": "grayLight1"}]


def f(name, spec):
    return {"name": name, **spec}


def build(session, url, base_id):
    def create_table(body):
        r = session.post(f"{url}/api/base/{base_id}/table", json=body, timeout=60)
        r.raise_for_status()
        j = r.json()
        print(f"  + {j['name']}  ({j['id']}, {len(j['fields'])} fields)")
        return j["id"]

    print("Building tables...")
    runs = create_table({"name": "Runs — Прогоны анализа", "fields": [
        f("Run ID", txt()), f("Дата прогона", date(True)), f("Входной файл", txt()),
        f("Пациентов всего", num(0)), f("С исходом", num(0)),
        f("Положительных исходов", num(0)), f("Версия DT", txt()), f("Заметки", longtxt())]})

    cyc = create_table({"name": "Cycles — Пациентки / циклы", "fields": [
        f("ID пациента", txt()), f("Имя", txt()), f("Возраст", num(0)), f("AMH", num(2)),
        f("AFC", num(0)), f("BMI", num(1)), f("Попытка", num(0)),
        f("Дата OPU", date()), f("Дата ET", date()), f("N фолликулов", num(0)),
        f("OCC", num(0)), f("MII", num(0)), f("Инсеминировано", num(0)), f("2pN", num(0)),
        f("Cleavage", num(0)), f("Bl", num(0)), f("Good Bl", num(0)), f("Cryo", num(0)),
        f("ET", num(0)), f("День ET", num(0)),
        f("Исход", sel([{"name": "Беременность", "color": "greenLight1"},
                        {"name": "Нет", "color": "redLight1"},
                        {"name": "Неизвестно", "color": "grayLight1"}])),
        f("Кластер", sel([{"name": "C0 Standard", "color": "blueLight1"},
                          {"name": "C1 Poor", "color": "redLight1"},
                          {"name": "C2 High", "color": "greenLight1"}]))]})

    def link(tid):
        return {"type": "link", "options": {"relationship": "manyOne", "foreignTableId": tid}}

    create_table({"name": "Predictions — Прогнозы DT", "fields": [
        f("Ключ", txt()), f("Пациентка", link(cyc)), f("Прогон", link(runs)),
        f("med OCC", num(1)), f("med MII", num(1)), f("med 2PN", num(1)),
        f("med Bl", num(1)), f("med Good", num(1)), f("P per transfer", num(3)),
        f("P overall cycle", num(3)), f("Bayes mean", num(3)), f("KAT", num(3)),
        f("NVSA", num(3)), f("CSDI", num(3)), f("DIGITAL TWIN", num(3)),
        f("OHSS risk", num(3)), f("P cancel", num(3)), f("Banking", txt()),
        f("Timestamp", date(True))]})

    create_table({"name": "Prediction vs Reality — Прогноз vs Факт", "fields": [
        f("Ключ", txt()), f("Пациентка", link(cyc)), f("Прогон", link(runs)),
        f("OCC факт", num(0)), f("OCC прогноз", num(1)), f("OCC ±30%", sel(FLAG)),
        f("Bl факт", num(0)), f("Bl прогноз", num(1)), f("Bl ±30%", sel(FLAG)),
        f("Good Bl факт", num(0)), f("Good Bl прогноз", num(1)), f("Good ±30%", sel(FLAG)),
        f("KAT−PRAI (пп)", num(1)),
        f("Приоритет ревью", sel([{"name": "1 Критично", "color": "redLight1"},
                                  {"name": "2 Рекомендовано", "color": "yellowLight1"},
                                  {"name": "3 В норме", "color": "greenLight1"}])),
        f("Основание", longtxt())]})

    create_table({"name": "Calibration — Калибровка моделей", "fields": [
        f("Ключ", txt()), f("Прогон", link(runs)),
        f("Модель", sel([{"name": m} for m in
                         ["MC", "Bayes", "KAT", "NVSA", "CSDI", "DT", "PRAI"]])),
        f("Brier", num(4)), f("AUC-ROC", num(4)), f("Acc@0.5", num(4)),
        f("Mean pred", num(4)), f("Real rate", num(4)), f("N", num(0))]})

    create_table({"name": "Files — Файлы прогонов", "fields": [
        f("Имя файла", txt()),
        f("Тип", sel([{"name": "dt_predictions", "color": "blueLight1"},
                      {"name": "OPU_table", "color": "greenLight1"},
                      {"name": "dt_vs_real", "color": "yellowLight1"},
                      {"name": "DT_Analytics_Report", "color": "purpleLight1"},
                      {"name": "input_report", "color": "grayLight1"},
                      {"name": "other", "color": "grayLight1"}])),
        {"name": "Файл", "type": "attachment"}, f("Прогон", link(runs)),
        f("Размер, КБ", num(1)), f("Строк", num(0)), f("Путь (локальный)", txt()),
        f("SHA-256", txt()), f("Загружено", date(True))]})

    print(f"\nDone. Base {base_id} now has the full IVF Digital Twin schema.")


def main():
    global TZ
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--config", default=os.path.join(here, "teable_config.json"))
    ap.add_argument("--token")
    ap.add_argument("--url", default="https://app.teable.ai")
    ap.add_argument("--space", help="space id — create a new base inside it")
    ap.add_argument("--name", default="IVF — Digital Twin", help="name for the new base")
    ap.add_argument("--base", help="existing (empty) base id to fill instead")
    ap.add_argument("--tz", default=TZ)
    args = ap.parse_args()
    TZ = args.tz

    token = args.token
    url = args.url
    if not token and os.path.exists(args.config):
        with open(args.config, encoding="utf-8") as fh:
            cfg = json.load(fh)
        token = cfg.get("token")
        url = cfg.get("teable_url", url)
    if not token:
        sys.exit("No token. Pass --token or set it in teable_config.json.")

    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/json"})

    base_id = args.base
    if not base_id:
        if not args.space:
            sys.exit("Pass --base <id> or --space <id> to create a new base.")
        r = s.post(f"{url}/api/base", json={"spaceId": args.space, "name": args.name}, timeout=60)
        r.raise_for_status()
        base_id = r.json()["id"]
        print(f"Created base '{args.name}' -> {base_id}")

    build(s, url, base_id)
    print(f"\nSet base_id={base_id} in a copy of teable_config.json for this clinic.")


if __name__ == "__main__":
    main()
