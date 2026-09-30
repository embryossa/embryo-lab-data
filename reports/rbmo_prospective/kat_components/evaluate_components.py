#!/usr/bin/env python3
"""AUC (95% CI) and Brier (95% CI) for the KAT components on the frozen RBMO cohort.

Reads kat_component_predictions.csv (run_components.py). Per centre, within-centre
n-weighted AUC (as in the RBMO paper, Table 6) and pooled. AUC CI = DeLong; Brier CI and the
CI of the weighted AUC = 2,000 bootstrap resamples stratified by centre. Paired DeLong
KAN vs FT-Transformer per centre. Sensitivity: cycles whose 7 features exactly match a row of
the KAT training file (all_df_with_KPI.xlsx; guo_three_stage/work/seen_overlap.csv) removed.
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260927)
B = 2000
CENTRES = ['Fertimed', 'GGRC', 'SkyFert', 'Two-Line']
LABEL = {'Fertimed': 'Centre A', 'GGRC': 'Centre B', 'SkyFert': 'Centre C', 'Two-Line': '4th centre'}
MODELS = {'ft': 'FT-Transformer (KAT component)', 'kan': 'KAN (KAT component)',
          'ens_raw': 'KAT ensemble, before isotonic', 'ens_iso': 'KAT ensemble = p_kat_raw'}


def delong_var(y, s):
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    psi = (pos[:, None] > neg[None, :]) + .5 * (pos[:, None] == neg[None, :])
    auc = psi.mean()
    v10, v01 = psi.mean(1), psi.mean(0)
    return auc, v10, v01, m, n


def delong_ci(y, s):
    auc, v10, v01, m, n = delong_var(y, s)
    se = np.sqrt(np.var(v10, ddof=1) / m + np.var(v01, ddof=1) / n)
    return auc, max(0, auc - 1.96 * se), min(1, auc + 1.96 * se)


def delong_paired(y, a, b):
    A, B_ = delong_var(y, a), delong_var(y, b)
    m, n = A[3], A[4]
    s10, s01 = np.cov(np.vstack([A[1], B_[1]])), np.cov(np.vstack([A[2], B_[2]]))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    z = (A[0] - B_[0]) / np.sqrt(var)
    return A[0] - B_[0], 2 * stats.norm.sf(abs(z))


def weighted_auc(df, col):
    g = [(len(x), roc_auc_score(x.Preg, x[col])) for _, x in df.groupby('centre')]
    return sum(n * a for n, a in g) / sum(n for n, _ in g)


def boot_idx(df):
    """indices of one bootstrap sample stratified by centre."""
    return np.concatenate([RNG.choice(ix, len(ix)) for ix in df.groupby('centre').indices.values()])


def evaluate(df, cols, mode, subset):
    rows = []
    idx = [boot_idx(df) for _ in range(B)]
    for c in cols:
        key = c.split('_', 1)[1]
        base = dict(mode=mode, subset=subset, model=MODELS[key])
        for centre in CENTRES + ['Pooled']:
            x = df if centre == 'Pooled' else df[df.centre == centre]
            if len(x) < 10 or x.Preg.nunique() < 2:
                continue
            y, p = x.Preg.values, x[c].values
            auc, lo, hi = delong_ci(y, p)
            bs = [brier_score_loss(y[i], p[i]) for i in (RNG.integers(0, len(y), len(y)) for _ in range(B))]
            rows.append(dict(base, population=LABEL.get(centre, centre), n=len(x), events=int(y.sum()),
                             auc=auc, auc_lo=lo, auc_hi=hi, auc_ci='DeLong',
                             brier=brier_score_loss(y, p), brier_lo=np.percentile(bs, 2.5),
                             brier_hi=np.percentile(bs, 97.5), mean_pred=p.mean(), observed=y.mean()))
        # within-centre n-weighted AUC (paper's headline) with stratified bootstrap CI
        ok = df.groupby('centre').Preg.transform('nunique') == 2
        w = df[ok]
        wb = []
        for i in idx:
            s = df.iloc[i]
            s = s[s.groupby('centre').Preg.transform('nunique') == 2]
            wb.append(weighted_auc(s, c))
        rows.append(dict(base, population='Within-centre weighted', n=len(w), events=int(w.Preg.sum()),
                         auc=weighted_auc(w, c), auc_lo=np.percentile(wb, 2.5), auc_hi=np.percentile(wb, 97.5),
                         auc_ci='stratified bootstrap'))
    return rows


p = pd.read_csv(OUT / 'kat_component_predictions.csv')
seen = pd.read_csv(OUT.parent / 'guo_three_stage' / 'work' / 'seen_overlap.csv')[['ref_id', 'seen_tabpfn_context']]
p = p.merge(seen, on='ref_id', how='left', validate='1:1')
# Centres A/4th reproduce p_kat_raw to 4 dp (batch seed 42+row); B/C came through the lab
# adapter with other MC seeds, so they agree up to Monte-Carlo noise.
err = (p.pro_ens_iso - p.p_kat_raw).abs()
print('reproduction of p_kat_raw |diff| by centre:'); print(err.groupby(p.centre).agg(['mean', 'max']).round(4))
assert len(p) == 1026 and err.max() < .05 and np.corrcoef(p.pro_ens_iso, p.p_kat_raw)[0, 1] > .99, 'prospective KAT not reproduced'

PRO = ['pro_ft', 'pro_kan', 'pro_ens_raw', 'pro_ens_iso']
RET = ['retro_ft', 'retro_kan', 'retro_ens_raw', 'retro_ens_iso']
retro = p[p.retro_complete]
unseen = ~p.seen_tabpfn_context.fillna(False).astype(bool)

rows = (evaluate(p, PRO, 'prospective (as DT ran: OCC, MII known)', 'all cycles')
        + evaluate(retro, RET, 'retrospective (observed 2PN, Bl, good Bl)', '2PN/Bl/good Bl recorded')
        + evaluate(p[unseen], PRO, 'prospective (as DT ran: OCC, MII known)', 'not in KAT training file')
        + evaluate(retro[unseen[p.retro_complete]], RET, 'retrospective (observed 2PN, Bl, good Bl)',
                   '2PN/Bl/good Bl recorded, not in KAT training file'))
m = pd.DataFrame(rows)
m.to_csv(OUT / 'kat_component_metrics.csv', index=False, float_format='%.4f')

tests = []
for mode, df, a, b in [('prospective', p, 'pro_kan', 'pro_ft'), ('retrospective', retro, 'retro_kan', 'retro_ft')]:
    for centre in CENTRES + ['Pooled']:
        x = df if centre == 'Pooled' else df[df.centre == centre]
        if x.Preg.nunique() < 2:
            continue
        dauc, pv = delong_paired(x.Preg.values, x[a].values, x[b].values)
        tests.append(dict(mode=mode, population=LABEL.get(centre, centre), n=len(x),
                          auc_kan_minus_ft=dauc, delong_p=pv, spearman_kan_ft=stats.spearmanr(x[a], x[b])[0]))
pd.DataFrame(tests).to_csv(OUT / 'kat_component_kan_vs_ft.csv', index=False, float_format='%.4f')

f = lambda r: f"{r.auc:.3f} ({r.auc_lo:.3f}–{r.auc_hi:.3f})"
g = lambda r: '' if pd.isna(r.brier) else f"{r.brier:.3f} ({r.brier_lo:.3f}–{r.brier_hi:.3f})"
m['AUC (95% CI)'] = m.apply(f, axis=1)
m['Brier (95% CI)'] = m.apply(g, axis=1)
pd.set_option('display.width', 250, 'display.max_rows', 200, 'display.max_colwidth', 60)
print(m[['mode', 'subset', 'model', 'population', 'n', 'events', 'AUC (95% CI)', 'Brier (95% CI)', 'mean_pred', 'observed']])
print(pd.DataFrame(tests))
