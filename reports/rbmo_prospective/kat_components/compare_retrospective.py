#!/usr/bin/env python3
"""Prospective (frozen RBMO cohort, 1,026 cycles) minus retrospective development metrics
for the KAT components. Retrospective values are the owner's published development
metrics (supplied 2026-09-27). The two samples are independent, so the SE of the AUC
difference = sqrt(SE_pro^2 + SE_retro^2), with each SE taken from its 95% CI / 3.92.
No CI is available for the retrospective Brier, so its difference carries only the
prospective uncertainty (a lower bound on the true interval width)."""
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(__file__).resolve().parent
RETRO = {'KAN (KAT component)': dict(auc=.677, lo=.651, hi=.703, brier=.207),
         'FT-Transformer (KAT component)': dict(auc=.820, lo=.798, hi=.841, brier=.159)}

m = pd.read_csv(OUT / 'kat_component_metrics.csv')
m = m[m['mode'].str.startswith('prospective') & (m.subset == 'all cycles') & m.model.isin(RETRO)]
rows = []
for _, r in m.iterrows():
    if r.population not in ('Pooled', 'Within-centre weighted'):
        continue
    t = RETRO[r.model]
    se = np.hypot((r.auc_hi - r.auc_lo) / 3.92, (t['hi'] - t['lo']) / 3.92)
    d = r.auc - t['auc']
    row = dict(model=r.model, prospective_population=r.population, n=r.n,
               auc_retro=t['auc'], auc_pro=r.auc, d_auc=d, d_auc_lo=d - 1.96 * se, d_auc_hi=d + 1.96 * se,
               brier_retro=t['brier'], brier_pro=r.brier)
    if pd.notna(r.brier):
        row.update(d_brier=r.brier - t['brier'], d_brier_lo=r.brier_lo - t['brier'], d_brier_hi=r.brier_hi - t['brier'])
    rows.append(row)
c = pd.DataFrame(rows)
c.to_csv(OUT / 'kat_component_retro_vs_prospective.csv', index=False, float_format='%.4f')
pd.set_option('display.width', 250)
print(c.round(3).to_string(index=False))
