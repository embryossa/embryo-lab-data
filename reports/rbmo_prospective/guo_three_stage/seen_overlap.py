#!/usr/bin/env python3
"""Which cohort cycles did the Digital Twin (GAT layer: training_file.xlsx, 1,172 protocols) and
TabPFN (context set all_df_with_KPI.xlsx) already see? Exact match on 7 cycle features (age,
follicles, OCC, inseminated, 2PN, blastocysts, good blastocysts; blank blastocysts read as 0),
outcome not used for matching. Null check: same match after shifting age by +3 years.
Writes work/seen_overlap.csv (ref_id + flags only) and seen_overlap_summary.csv."""
import hashlib, sys
from pathlib import Path
import pandas as pd
OUT = Path(__file__).resolve().parent
REF = OUT.parents[2] / 'data' / 'processed' / 'reference' / 'rbmo_prospective_2026'
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _local_paths import local_path  # noqa: E402
SOURCES = {'dt_gat_training': local_path('dt_gat_training'),
           'tabpfn_context': local_path('tabpfn_context')}
X = ['age', 'fol', 'occ', 'ins', 'pn2', 'bl', 'gb']
c = pd.read_csv(REF / 'rbmo_prospective_cohort.csv', encoding='utf-8-sig')
cc = pd.DataFrame({'age': c.age.round(), 'fol': c.follicles_tvp, 'occ': c.OCC, 'ins': c.Inseminated, 'pn2': c['2pN'],
                   'bl': c.Bl.fillna(0), 'gb': c['Good Bl'].fillna(0)}).astype(float)
flags = pd.DataFrame({'ref_id': c.ref_id, 'centre': c.centre})
summ = []
for name, p in SOURCES.items():
    t = pd.read_excel(p)
    tt = pd.DataFrame({'age': pd.to_numeric(t['Возраст'], errors='coerce'), 'fol': t['Количество фолликулов'],
                       'occ': t['Число ОКК'], 'ins': t['Число инсеминированных'], 'pn2': t['2 pN'], 'bl': t['Число Bl'],
                       'gb': t['Число Bl хор.кач-ва']}).apply(pd.to_numeric, errors='coerce')
    keys = set(map(tuple, tt[X].dropna().values))
    hit = cc[X].apply(tuple, axis=1).isin(keys)
    null = cc[X].assign(age=cc.age + 3).apply(tuple, axis=1).isin(keys)
    flags[f'seen_{name}'] = hit
    for cen in flags.centre.unique():
        m = flags.centre == cen
        summ.append(dict(source=name, file=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()[:16], rows=len(t),
                         centre=cen, n=int(m.sum()), matched=int(hit[m].sum()), null_matched_age_plus3=int(null[m].sum())))
(OUT / 'work').mkdir(exist_ok=True)
flags.to_csv(OUT / 'work' / 'seen_overlap.csv', index=False)
S = pd.DataFrame(summ); S.to_csv(OUT / 'seen_overlap_summary.csv', index=False)
print(S.to_string(index=False))
