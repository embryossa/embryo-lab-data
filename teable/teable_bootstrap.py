#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""teable_bootstrap.py — create or upgrade the per-clinic Teable base schema.

"One base per clinic". This script applies teable_schema.schema() to a base:

  * a clinic whose registry entry has no base_id gets a brand-new base created
    in the configured space, and the new id is written back into the config;
  * a clinic that already has a base is UPGRADED — missing tables are created,
    missing fields are added to existing tables. Nothing is renamed, retyped or
    deleted, so existing records and their data are never touched.

Examples:
    python teable_bootstrap.py --all                # every clinic in the registry
    python teable_bootstrap.py --clinic ggrc        # one clinic
    python teable_bootstrap.py --all --dry-run      # show what would change

Dependency:  pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import sys

import teable_common as tc
import teable_schema as ts


def _resolve_link(field: dict, table_ids: dict) -> dict:
    """Turn a schema link placeholder into a concrete foreignTableId payload."""
    body = {k: v for k, v in field.items() if k != "_link_to"}
    target = field["_link_to"]
    if target not in table_ids:
        raise SystemExit(f"link target table '{target}' does not exist yet "
                         f"— check the order of teable_schema.schema()")
    body["options"] = dict(body.get("options") or {})
    body["options"]["foreignTableId"] = table_ids[target]
    return body


def apply_schema(tb: tc.Teable, base_id: str, tz: str, dry_run: bool = False) -> dict:
    """Create missing tables / fields. Returns {'tables': n, 'fields': n}."""
    table_ids = tc.resolve_tables(tb, base_id)
    made_tables = made_fields = 0

    for table in ts.schema(tz):
        key, name, fields = table["key"], table["name"], table["fields"]

        if key not in table_ids:
            # records:[] — without it Teable seeds every new table with three
            # blank rows, which would then show up as phantom cycles/runs.
            body = {"name": name, "records": [],
                    "fields": [_resolve_link(fl, table_ids) if "_link_to" in fl else fl
                               for fl in fields]}
            if dry_run:
                print(f"  [dry] + table {name}  ({len(fields)} fields)")
                # A placeholder id keeps later link targets resolvable in dry-run.
                table_ids[key] = f"DRYRUN::{key}"
                made_tables += 1
                continue
            created = tb.create_table(base_id, body)
            table_ids[key] = created["id"]
            made_tables += 1
            print(f"  + table {name}  ({created['id']}, {len(created['fields'])} fields)")
            continue

        table_id = table_ids[key]
        if str(table_id).startswith("DRYRUN::"):
            continue
        have = {fl["name"] for fl in tb.fields(table_id)}
        missing = [fl for fl in fields if fl["name"] not in have]
        for fl in missing:
            body = _resolve_link(fl, table_ids) if "_link_to" in fl else dict(fl)
            if dry_run:
                print(f"  [dry] + field {name} :: {fl['name']} ({fl['type']})")
            else:
                try:
                    tb.create_field(table_id, body)
                except Exception as exc:  # noqa: BLE001 - report and keep going
                    # A URL-formatted text field is the one spec a older Teable
                    # build may reject; fall back to plain text rather than
                    # aborting the whole upgrade.
                    if body.get("options", {}).get("showAs"):
                        tb.create_field(table_id, {"name": fl["name"],
                                                   "type": "singleLineText"})
                        print(f"  ~ field {name} :: {fl['name']} (plain text fallback)")
                        made_fields += 1
                        continue
                    print(f"  [!] field {name} :: {fl['name']} failed: {exc}")
                    continue
                print(f"  + field {name} :: {fl['name']}")
            made_fields += 1

    return {"tables": made_tables, "fields": made_fields}


def apply_views(tb: tc.Teable, base_id: str, dry_run: bool = False) -> int:
    """Create the working views (tabs) that do not exist yet. Never edits an
    existing view — the owner's own layout tweaks in Teable are left alone."""
    table_ids = tc.resolve_tables(tb, base_id)
    made = 0
    for table_key, defs in ts.views().items():
        table_id = table_ids.get(table_key)
        if not table_id:
            continue
        fields = {f["name"]: f["id"] for f in tb.fields(table_id)}
        have = {v["name"] for v in tb.views(table_id)}
        for view in defs:
            if view["name"] in have:
                continue
            body = {"name": view["name"], "type": view.get("type", "grid")}

            sort_objs = [{"fieldId": fields[n], "order": o}
                         for n, o in view.get("sort", []) if n in fields]
            if sort_objs:
                body["sort"] = {"sortObjs": sort_objs, "manualSort": False}

            filter_set = [{"fieldId": fields[n], "operator": op, "value": val}
                          for n, op, val in view.get("filter", []) if n in fields]
            if filter_set:
                body["filter"] = {"conjunction": "and", "filterSet": filter_set}

            columns = [c for c in view.get("columns", []) if c in fields]
            if columns:
                # Listed columns come first in order; everything else is hidden
                # rather than deleted, so no data ever disappears from the table.
                meta = {fields[c]: {"order": i, "hidden": False}
                        for i, c in enumerate(columns)}
                for i, (name, fid) in enumerate(fields.items()):
                    if name not in columns:
                        meta[fid] = {"order": 1000 + i, "hidden": True}
                body["columnMeta"] = meta

            if dry_run:
                print(f"  [dry] + view {table_key} :: {view['name']}")
            else:
                try:
                    tb.create_view(table_id, body)
                except Exception as exc:  # noqa: BLE001 - a view is not worth aborting for
                    print(f"  [!] view {table_key} :: {view['name']} failed: {exc}")
                    continue
                print(f"  + view {table_key} :: {view['name']}")
            made += 1
    return made


def apply_layout(tb: tc.Teable, base_id: str, dry_run: bool = False) -> int:
    """Make every clinic's base look identical: same table names, same table
    order, same tab order, same column order.

    A base that was upgraded field-by-field ends up different from one created
    from scratch — later tables land at the end of the sidebar, later fields at
    the end of every view, and a table renamed in the schema keeps its old
    title. This pass reconciles all three. It only ever changes names, order and
    position: hidden/visible flags the owner set by hand are preserved, and the
    curated column sets of the managed tabs are left exactly as declared.
    """
    moved = 0

    tables = tb.tables(base_id)
    current = [t["name"].split(" —")[0].strip() for t in tables]
    by_key = dict(zip(current, (t["id"] for t in tables)))

    # -- table display names ------------------------------------------------
    wanted_names = {t["key"]: t["name"] for t in ts.schema()}
    for table, key in zip(tables, current):
        target_name = wanted_names.get(key)
        if not target_name or table["name"] == target_name:
            continue
        if dry_run:
            print(f"  [dry] ~ таблица '{table['name']}' -> '{target_name}'")
        else:
            tb.rename_table(base_id, table["id"], target_name)
            print(f"  ~ таблица переименована: '{table['name']}' -> '{target_name}'")
        moved += 1
    wanted = [k for k in ts.table_keys() if k in by_key]
    # Anything we do not manage keeps its relative order, at the end.
    target = wanted + [k for k in current if k not in wanted]
    if current != target:
        if dry_run:
            print(f"  [dry] ~ порядок таблиц -> {' · '.join(target)}")
            moved += 1
        else:
            for i in range(1, len(target)):
                tb.move_table(base_id, by_key[target[i]], by_key[target[i - 1]])
            print(f"  ~ порядок таблиц выровнен: {' · '.join(target)}")
            moved += 1

    # Views: the managed tabs follow the order they are declared in, right after
    # whatever views already existed (Teable's own default "Grid view").
    view_defs = ts.views()
    for table_key, defs in view_defs.items():
        table_id = by_key.get(table_key)
        if not table_id:
            continue
        views = tb.views(table_id)
        names = [v["name"] for v in views]
        ids = dict(zip(names, (v["id"] for v in views)))
        managed = [d["name"] for d in defs if d["name"] in ids]
        target_v = [n for n in names if n not in managed] + managed
        if names == target_v:
            continue
        if dry_run:
            print(f"  [dry] ~ порядок вкладок {table_key} -> {' · '.join(target_v)}")
        else:
            for i in range(1, len(target_v)):
                tb.move_view(table_id, ids[target_v[i]], ids[target_v[i - 1]])
            print(f"  ~ порядок вкладок {table_key} выровнен")
        moved += 1

    # -- symmetric link fields ---------------------------------------------
    # Teable auto-creates a back-link field on the far side of every link and
    # names it after the linked table as it was named back then. A base that was
    # upgraded therefore keeps the old title (e.g. "Predictions — Прогнозы DT"),
    # so the fields are renamed to whatever the linked table is called now.
    table_titles = {t["id"]: t["name"] for t in tb.tables(base_id)}
    for table in ts.schema():
        table_id = by_key.get(table["key"])
        if not table_id:
            continue
        declared = {f["name"] for f in table["fields"]}
        for field in tb.fields(table_id):
            if field["type"] != "link" or field["name"] in declared:
                continue
            target = table_titles.get((field.get("options") or {}).get("foreignTableId"))
            if not target or field["name"] == target:
                continue
            if dry_run:
                print(f"  [dry] ~ поле-обратная связь '{field['name']}' -> '{target}'")
            else:
                tb.rename_field(table_id, field["id"], target)
                print(f"  ~ поле-обратная связь: '{field['name']}' -> '{target}'")
            moved += 1

    # -- columns inside views ----------------------------------------------
    # The full column state of every view is written explicitly, because Teable
    # simply omits a field from columnMeta until something touches it — and an
    # omitted field is visible. That is why a fresh base showed the back-link
    # columns while an upgraded one hid them. Filters and sorts are never
    # touched here; only column order and visibility.
    curated = {(k, d["name"]): d["columns"] for k, defs in view_defs.items()
               for d in defs if d.get("columns")}
    for table in ts.schema():
        table_id = by_key.get(table["key"])
        if not table_id:
            continue
        schema_order = [f["name"] for f in table["fields"]]
        declared = set(schema_order)
        fields = tb.fields(table_id)
        by_name = {f["name"]: f["id"] for f in fields}
        by_id = {f["id"]: f for f in fields}
        # Declared fields keep schema order; everything else (back-links, fields
        # left over from an older schema) trails behind sorted by name — by name
        # and not by creation order, so the tail is identical in a base built
        # from scratch and in one grown field by field.
        schema_ids = sorted(
            (f["id"] for f in fields),
            key=lambda fid: (0, schema_order.index(by_id[fid]["name"]), "")
            if by_id[fid]["name"] in declared else (1, 0, by_id[fid]["name"]))

        for view in tb.views(table_id):
            meta = view.get("columnMeta") or {}
            columns = curated.get((table["key"], view["name"]))
            if columns:
                # A curated tab pins its own visible set and order; the rest of
                # the fields trail behind it hidden, in schema order.
                head = [by_name[n] for n in columns if n in by_name]
                order_ids = head + [fid for fid in schema_ids if fid not in head]
                visible = set(head)
                state = [(fid, fid not in visible) for fid in order_ids]
            else:
                # Everything else (Teable's default "Grid view" included): schema
                # order, back-link columns hidden, the owner's own hide/show of
                # real fields preserved.
                state = [(fid, (by_id[fid]["name"] not in declared)
                          or bool(meta.get(fid, {}).get("hidden")))
                         for fid in schema_ids]

            actual = [(fid, bool(meta[fid].get("hidden")))
                      for fid in sorted(meta, key=lambda f: meta[f].get("order", 0))
                      if fid in by_id]
            if actual == state:
                continue
            entries = [{"fieldId": fid,
                        "columnMeta": {"order": i, **({"hidden": True} if hid else {})}}
                       for i, (fid, hid) in enumerate(state)]
            if dry_run:
                print(f"  [dry] ~ колонки {table['key']} :: {view['name']}")
            else:
                try:
                    tb.set_column_meta(table_id, view["id"], entries)
                except Exception as exc:  # noqa: BLE001 - cosmetic, never fatal
                    print(f"  [!] колонки {table['key']} :: {view['name']}: {exc}")
                    continue
                print(f"  ~ колонки выровнены: {table['key']} :: {view['name']}")
            moved += 1

    return moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=tc.DEFAULT_CONFIG)
    ap.add_argument("--clinic", help="clinic code from the registry (default: --all)")
    ap.add_argument("--all", action="store_true", help="every clinic in the registry")
    ap.add_argument("--space", help="space id for newly created bases "
                                    "(default: config['space_id'])")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.clinic and not args.all:
        sys.exit("Pass --clinic <code> or --all.")

    cfg = tc.load_config(args.config)
    tb = tc.connect(cfg, dry_run=args.dry_run)
    space_id = args.space or cfg.get("space_id")

    for clinic in tc.clinics(cfg, args.clinic):
        print(f"\n=== {clinic.name} ({clinic.code}) ===")
        base_id = clinic.base_id
        if not base_id:
            if not space_id:
                sys.exit("No base_id for this clinic and no space_id in the config "
                         "— pass --space <id>.")
            title = f"{clinic.name} — Digital Twin"
            if args.dry_run:
                print(f"  [dry] create base '{title}' in {space_id}")
                base_id = "DRYRUN"
            else:
                base_id = tb.create_base(space_id, title)
                tc.save_base_id(cfg, clinic.code, base_id)
                print(f"  created base '{title}' -> {base_id} (written to config)")
        else:
            print(f"  base {base_id} — upgrading in place")

        if base_id == "DRYRUN":
            print("  [dry] full schema would be created")
            continue
        stats = apply_schema(tb, base_id, clinic.timezone, dry_run=args.dry_run)
        n_views = apply_views(tb, base_id, dry_run=args.dry_run)
        n_order = apply_layout(tb, base_id, dry_run=args.dry_run)
        if not stats["tables"] and not stats["fields"] and not n_views and not n_order:
            print("  schema already up to date")
        else:
            print(f"  applied: +{stats['tables']} tables, +{stats['fields']} fields, "
                  f"+{n_views} views, {n_order} reorders")

    print("\nDone.")


if __name__ == "__main__":
    main()
