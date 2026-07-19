"""Clinic-specific ingest/normalize adapters. Each module here produces the same
canonical shape src/embryo/normalize.py produces for the default (clinic A)
pipeline, so the shared kpi/report/contracts/agent layers never need to know
which clinic a run belongs to.
"""
