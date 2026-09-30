#!/usr/bin/env python3
"""Metrics, tests and figures for guo_three_stage.py: every Guo model and the prospective
Digital Twin / TabPFN probabilities scored on identical test cycles.
AUC-ROC, PR-AUC, Brier (1,000 bootstrap 95% CI), calibration intercept/slope, accuracy,
sensitivity, specificity, PPV, NPV, F1 at 0.5; DeLong for paired AUCs; category-free NRI
and IDI; decision-curve net benefit; SHAP (TreeExplainer) for the Stage 3 internal split."""
import pickle, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd, matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from sklearn.linear_model import LogisticRegression

OUT = Path(__file__).resolve().parent
REF = OUT.parents[2] / 'data' / 'processed' / 'reference' / 'rbmo_prospective_2026'
RNG = np.random.default_rng(20260922)
LABEL = {'Fertimed': 'Centre A', 'GGRC': 'Centre B', 'SkyFert': 'Centre C', 'Two-Line': '4th centre'}
SPLIT_NAME = {'E:Fertimed': 'External: Centre A held out', 'E:GGRC': 'External: Centre B held out',
              'E:SkyFert': 'External: Centre C held out', 'E:Two-Line': 'External: 4th centre held out',
              'I': 'Internal test (15%)'}
EXTERNAL = ['Fertimed', 'GGRC', 'SkyFert', 'Two-Line']
MIN_SUBSET = 20   # metrics on the DT-unseen subset only when n >= 20 and both outcomes present
mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'pdf.fonttype': 42})
COL = {'S1': '#9aa5b1', 'S2': '#5b8def', 'S2+': '#7a4fb0', 'S3': '#d97757', 'Digital Twin': '#117733', 'TabPFN': '#a67c00'}

R = {'res': [r for f in sorted((OUT / 'work').glob('results_*.pkl')) for r in pickle.load(open(f, 'rb'))['res']]}
assert len({(r['split'], r['stage']) for r in R['res']}) == len(R['res']) == 20, 'expected 5 splits x 4 stages'
seen = pd.read_csv(OUT / 'work' / 'seen_overlap.csv').set_index('ref_id')
path = REF / 'rbmo_prospective_cohort.csv'
assert hashlib.sha256(path.read_bytes()).hexdigest() == json.loads((REF / 'manifest.json').read_text('utf-8'))['cohort_sha256']
d = pd.read_csv(path, encoding='utf-8-sig').set_index('ref_id')


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def point(y, p):
    q = (p >= .5).astype(int)
    tp, tn = ((q == 1) & (y == 1)).sum(), ((q == 0) & (y == 0)).sum()
    fp, fn = ((q == 1) & (y == 0)).sum(), ((q == 0) & (y == 1)).sum()
    div = lambda a, b: a / b if b else np.nan
    slope = LogisticRegression(C=1e6, max_iter=5000).fit(logit(p)[:, None], y).coef_[0, 0]
    # calibration-in-the-large: intercept with the logit of p as offset (slope fixed at 1)
    import statsmodels.api as sm
    cil = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=logit(p)).fit().params[0]
    return dict(auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p), brier=brier_score_loss(y, p),
                cal_intercept=cil, cal_slope=slope, mean_pred=p.mean(), observed=y.mean(),
                accuracy=(tp + tn) / len(y), sensitivity=div(tp, tp + fn), specificity=div(tn, tn + fp),
                ppv=div(tp, tp + fp), npv=div(tn, tn + fn), f1=div(2 * tp, 2 * tp + fp + fn))


def boot(y, p, B=1000):
    out = []
    for _ in range(B):
        i = RNG.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        out.append((roc_auc_score(y[i], p[i]), average_precision_score(y[i], p[i]), brier_score_loss(y[i], p[i])))
    lo, hi = np.percentile(out, [2.5, 97.5], axis=0)
    return dict(auc_lo=lo[0], auc_hi=hi[0], pr_auc_lo=lo[1], pr_auc_hi=hi[1], brier_lo=lo[2], brier_hi=hi[2])


def delong(y, a, b):
    """Paired DeLong test (Sun & Xu 2014 fast algorithm). Returns (AUC_a - AUC_b, p)."""
    def mid(x):
        J = np.argsort(x); Z = x[J]; n = len(x); T = np.zeros(n); i = 0
        while i < n:
            j = i
            while j < n and Z[j] == Z[i]:
                j += 1
            T[i:j] = .5 * (i + j - 1) + 1; i = j
        T2 = np.empty(n); T2[J] = T; return T2
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    V10, V01, aucs = [], [], []
    for s in (a, b):
        tx, ty, tz = mid(s[pos]), mid(s[neg]), mid(np.r_[s[pos], s[neg]])
        auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        V10.append((tz[:m] - tx) / n); V01.append(1 - (tz[m:] - ty) / m); aucs.append(auc)
    S = np.cov(np.array(V10)) / m + np.cov(np.array(V01)) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0
    return aucs[0] - aucs[1], 2 * stats.norm.sf(abs(z))


def nri_idi(y, new, old):
    up, dn = new > old, new < old
    e, ne = y == 1, y == 0
    nri = (up[e].mean() - dn[e].mean()) + (dn[ne].mean() - up[ne].mean())
    idi = (new[e].mean() - old[e].mean()) - (new[ne].mean() - old[ne].mean())
    bs = []
    for _ in range(1000):
        i = RNG.integers(0, len(y), len(y)); yy, nn_, oo = y[i], new[i], old[i]
        if yy.min() == yy.max():
            continue
        bs.append((nn_[yy == 1] - oo[yy == 1]).mean() - (nn_[yy == 0] - oo[yy == 0]).mean())
    p_idi = 2 * min((np.array(bs) <= 0).mean(), (np.array(bs) >= 0).mean())
    return nri, idi, p_idi


def net_benefit(y, p, thr):
    n = len(y)
    return np.array([((p >= t) & (y == 1)).sum() / n - ((p >= t) & (y == 0)).sum() / n * t / (1 - t) for t in thr])


# ---------------- assemble test predictions ----------------
rows, preds = [], {}
for r in R['res']:
    y = d.loc[r['test_ids'], 'Preg'].to_numpy(int)
    for m, p in r['test_pred'].items():
        preds[(r['split'], r['stage'], m)] = (np.asarray(p), y)
    preds[(r['split'], r['stage'], 'SELECTED')] = (np.asarray(r['test_pred'][r['chosen']]), y)
    for name, col in (('Digital Twin', 'DIGITAL TWIN'), ('TabPFN', 'PRAI')):
        preds[(r['split'], 'ref', name)] = (d.loc[r['test_ids'], col].to_numpy(float), y)
    preds[(r['split'], 'ids', '')] = (np.array(r['test_ids']), y)

metrics = []
for (split, stage, m), (p, y) in preds.items():
    if stage == 'ids':
        continue
    row = dict(split=split, test_set=SPLIT_NAME[split], stage=stage, model=m, n=len(y), events=int(y.sum()))
    if stage != 'ref' and m != 'SELECTED':
        rr = next(x for x in R['res'] if x['split'] == split and x['stage'] == stage)
        row.update(internal_auc=rr['int_auc'][m], cv_auc=rr['cv_auc'].get(m), selected=(m == rr['chosen']))
    if stage != 'ref' and m == 'SELECTED':
        row['model'] = 'SELECTED=' + next(x for x in R['res'] if x['split'] == split and x['stage'] == stage)['chosen']
    row.update(point(y, p))
    if m == 'SELECTED' or stage == 'ref':
        row.update(boot(y, p))
    metrics.append(row)
# same test cycles restricted to those absent from the Digital Twin GAT training file
for split in SPLIT_NAME:
    ids, y = preds[(split, 'ids', '')]
    keep = ~seen.loc[ids, 'seen_dt_gat_training'].to_numpy(bool)
    if keep.sum() < MIN_SUBSET or y[keep].min() == y[keep].max():
        metrics.append(dict(split=split, test_set=SPLIT_NAME[split], subset='dt_unseen', stage='-', model='not evaluable',
                            n=int(keep.sum()), events=int(y[keep].sum())))
        continue
    for stage, m in [('S1', 'SELECTED'), ('S2', 'SELECTED'), ('S2+', 'SELECTED'), ('S3', 'SELECTED'),
                     ('ref', 'Digital Twin'), ('ref', 'TabPFN')]:
        p = preds[(split, stage, m)][0][keep]
        name = m if stage == 'ref' else 'SELECTED=' + next(x for x in R['res'] if x['split'] == split and x['stage'] == stage)['chosen']
        row = dict(split=split, test_set=SPLIT_NAME[split], subset='dt_unseen', stage=stage, model=name,
                   n=int(keep.sum()), events=int(y[keep].sum()))
        row.update(point(y[keep], p)); row.update(boot(y[keep], p)); metrics.append(row)
M = pd.DataFrame(metrics)
M['subset'] = M.get('subset').fillna('all')
M.to_csv(OUT / 'metrics_all_models.csv', index=False, float_format='%.4f')
main = M[(M.model.str.startswith('SELECTED')) | (M.stage == 'ref') | (M.model == 'not evaluable')].drop_duplicates(['split', 'subset', 'stage', 'model'])
order = {'S1': 0, 'S2': 1, 'S2+': 2, 'S3': 3, 'ref': 4}
order['-'] = 5
main = main.sort_values(['split', 'subset', 'stage'], key=lambda s: s.map(order) if s.name == 'stage' else s)
main.to_csv(OUT / 'metrics_main.csv', index=False, float_format='%.4f')

# ---------------- pooled external (LOCO) + within-centre weighted ----------------
pool = []
for key in ['S1', 'S2', 'S2+', 'S3', 'Digital Twin', 'TabPFN']:
    ps, ys, cs, per = [], [], [], []
    for held in EXTERNAL:
        s = f'E:{held}'
        p, y = preds[(s, 'ref', key)] if key in ('Digital Twin', 'TabPFN') else preds[(s, key, 'SELECTED')]
        ps.append(p); ys.append(y); per.append((roc_auc_score(y, p), len(y)))
    P, Y = np.concatenate(ps), np.concatenate(ys)
    pool.append(dict(model=key, n=len(Y), pooled_auc=roc_auc_score(Y, P),
                     within_centre_weighted_auc=sum(a * k for a, k in per) / sum(k for _, k in per),
                     pooled_brier=brier_score_loss(Y, P), pooled_pr_auc=average_precision_score(Y, P),
                     **{f'auc_{LABEL[h]}': a for h, (a, _) in zip(EXTERNAL, per)}))
pd.DataFrame(pool).to_csv(OUT / 'metrics_external_pooled.csv', index=False, float_format='%.4f')

# ---------------- paired tests ----------------
tests = []
pairs = [('S2', 'S1'), ('S2+', 'S2'), ('S3', 'S2+'), ('S3', 'S2'), ('S3', 'Digital Twin'), ('S3', 'TabPFN'),
         ('S2+', 'Digital Twin'), ('S2+', 'TabPFN'), ('Digital Twin', 'TabPFN')]
get = lambda s, k: preds[(s, 'ref', k)] if k in ('Digital Twin', 'TabPFN') else preds[(s, k, 'SELECTED')]
for split in SPLIT_NAME:
    for a, b in pairs:
        (pa, y), (pb, _) = get(split, a), get(split, b)
        dauc, p = delong(y, pa, pb)
        nri, idi, pidi = nri_idi(y, pa, pb)
        tests.append(dict(split=split, test_set=SPLIT_NAME[split], new=a, reference=b, n=len(y), delta_auc=dauc,
                          delong_p=p, nri_continuous=nri, idi=idi, idi_boot_p=pidi))
pd.DataFrame(tests).to_csv(OUT / 'paired_tests.csv', index=False, float_format='%.4f')

# ---------------- selection record ----------------
sel = [dict(split=r['split'], stage=r['stage'], chosen=r['chosen'], internal_auc_chosen=r['int_auc'][r['chosen']],
            features=';'.join(r['features']), vif_kept=';'.join(r['vif_kept']), ses_subset=';'.join(r['ses']),
            votes=json.dumps(r['votes'])) for r in R['res']]
pd.DataFrame(sel).to_csv(OUT / 'model_selection.csv', index=False, float_format='%.4f')

# ---------------- test predictions (pseudonymous ref_id only) ----------------
tp = []
for split in SPLIT_NAME:
    ids, y = preds[(split, 'ids', '')]
    row = pd.DataFrame({'split': split, 'ref_id': ids, 'centre': d.loc[ids, 'centre'].values, 'Preg': y,
                        'seen_by_dt_gat_training': seen.loc[ids, 'seen_dt_gat_training'].values})
    for k in ['S1', 'S2', 'S2+', 'S3', 'Digital Twin', 'TabPFN']:
        row[k] = get(split, k)[0]
    tp.append(row)
pd.concat(tp).to_csv(OUT / 'test_predictions.csv', index=False, float_format='%.4f')

# ---------------- figures ----------------
KEYS = ['S1', 'S2', 'S2+', 'S3', 'Digital Twin', 'TabPFN']
LBL = {'S1': 'Stage 1', 'S2': 'Stage 2', 'S2+': 'Stage 2+ (DT inputs)', 'S3': 'Stage 3',
       'Digital Twin': 'Digital Twin', 'TabPFN': 'TabPFN'}
fig, ax = plt.subplots(3, 5, figsize=(225 / 25.4, 140 / 25.4))
thr = np.linspace(.01, .99, 99)
for j, split in enumerate(SPLIT_NAME):
    for k in KEYS:
        p, y = get(split, k)
        fpr, tpr, _ = roc_curve(y, p)
        ls = '--' if k in ('Digital Twin', 'TabPFN') else '-'
        ax[0, j].plot(fpr, tpr, ls, color=COL[k], lw=1, label=f'{LBL[k]} {roc_auc_score(y, p):.2f}')
        q = pd.qcut(p, 5, duplicates='drop')
        g = pd.DataFrame({'p': p, 'y': y, 'q': q}).groupby('q', observed=True).mean()
        ax[1, j].plot(g.p, g.y, 'o' + ls, color=COL[k], ms=2.5, lw=.8)
        ax[2, j].plot(thr, net_benefit(y, p, thr), ls, color=COL[k], lw=.9)
    y = get(split, 'S1')[1]
    ax[2, j].plot(thr, net_benefit(y, np.ones_like(y, float), thr), color='k', lw=.6, label='treat all')
    ax[2, j].axhline(0, color='grey', lw=.5)
    ax[2, j].set_ylim(-.05, max(.05, y.mean() + .02)); ax[2, j].set_xlim(0, 1)
    ax[0, j].plot([0, 1], [0, 1], ':', color='grey', lw=.5); ax[1, j].plot([0, 1], [0, 1], ':', color='grey', lw=.5)
    ax[0, j].set_title(f'{SPLIT_NAME[split]}\n(n={len(y)}, pregnant {y.mean():.0%})', fontsize=7)
    ax[0, j].legend(fontsize=5, loc='lower right', frameon=False)
    ax[1, j].set_xlim(0, 1); ax[1, j].set_ylim(0, 1)
ax[0, 0].set_ylabel('ROC: sensitivity'); ax[1, 0].set_ylabel('Calibration: observed'); ax[2, 0].set_ylabel('Net benefit')
for j in range(5):
    ax[0, j].set_xlabel('1 − specificity', fontsize=6); ax[1, j].set_xlabel('predicted (quintiles)', fontsize=6)
    ax[2, j].set_xlabel('threshold probability', fontsize=6)
fig.tight_layout()
fig.savefig(OUT / 'Figure_roc_calibration_dca.png'); fig.savefig(OUT / 'Figure_roc_calibration_dca.pdf')

# SHAP: best tree learner of the internal Stage 3 model set (TreeExplainer)
try:
    import shap
    r = next(x for x in R['res'] if x['split'] == 'I' and x['stage'] == 'S3')
    models, ses, Xtr, Xte = r['shap_payload']
    trees = [m for m in ['XGBoost', 'LightGBM', 'CatBoost', 'RF'] if m in models]
    best = max(trees, key=lambda m: r['int_auc'][m])
    sv = shap.TreeExplainer(models[best]).shap_values(Xte)
    sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if np.ndim(sv) == 3 else sv)
    imp = pd.Series(np.abs(sv).mean(0), index=Xte.columns).sort_values()
    imp.rename('mean_abs_shap').to_csv(OUT / f'shap_stage3_internal_{best}.csv')
    fig, a = plt.subplots(figsize=(90 / 25.4, 60 / 25.4))
    a.barh(imp.index, imp.values, color=COL['S3'])
    a.set_xlabel(f'mean |SHAP| (log-odds), {best}, Stage 3, internal test')
    fig.savefig(OUT / 'Figure_shap_stage3.png'); fig.savefig(OUT / 'Figure_shap_stage3.pdf')
except Exception as e:
    print('SHAP skipped:', e)
print(main[['split', 'subset', 'stage', 'model', 'n', 'auc', 'auc_lo', 'auc_hi', 'pr_auc', 'brier', 'cal_intercept', 'cal_slope']]
      .to_string(index=False, float_format='%.3f'))
print(pd.DataFrame(pool).to_string(index=False, float_format='%.3f'))
