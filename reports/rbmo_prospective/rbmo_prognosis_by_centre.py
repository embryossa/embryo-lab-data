#!/usr/bin/env python3
"""Same predicted prognosis, different outcome by centre: observed clinical-pregnancy rate
(Wilson 95% CI) per centre within bands of the models' predicted probability (mean of Digital
Twin and TabPFN), against each cell's mean predicted probability. RBMO prospective cohort.

Tests: chi-square across centres within each band; likelihood-ratio test of centre in a
logistic model of pregnancy on the logit of the predicted probability (does centre explain
outcome beyond the model's forecast?). Cells with n < 10 are not plotted.
Outputs 300-dpi PNG + vector PDF + statistics table. Reads the frozen cohort (build_cohort.py)."""
import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.linear_model import LogisticRegression

mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8.5,
    'axes.titleweight': 'bold', 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 6.8, 'axes.linewidth': .7, 'axes.spines.top': False,
    'axes.spines.right': False, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'pdf.fonttype': 42})
MM = 1 / 25.4
C_INK, C_GREY = '#222222', '#888888'
# centre -> (RBMO label, colour, marker); palette validated for CVD separation, shapes as backup
CENTRES = {'Fertimed': ('Centre A', '#5A4FB0', 'o'), 'GGRC': ('Centre B', '#117733', 's'),
           'SkyFert': ('Centre C', '#CC6677', 'D'), 'Two-Line': ('4th centre', '#A67C00', '^')}
BANDS = [(0, 40, '<40%'), (40, 60, '40–60%'), (60, 100, '≥60%')]
MIN_N = 10

OUT = Path(__file__).resolve().parent
REF = OUT.parents[1] / 'data' / 'processed' / 'reference' / 'rbmo_prospective_2026'


def load():
    path = REF / 'rbmo_prospective_cohort.csv'
    manifest = json.loads((REF / 'manifest.json').read_text(encoding='utf-8'))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == manifest['cohort_sha256'], 'frozen cohort differs from its manifest'
    d = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
    df = pd.DataFrame({'centre': d.centre, 'p_dt': d['DIGITAL TWIN'], 'p_tp': d['PRAI'], 'y': d['Preg']})
    assert not df.isna().any(axis=None) and len(df) == manifest['n_cycles']
    df['pred'] = (df.p_dt + df.p_tp) / 2
    df['band'] = pd.cut(df.pred * 100, [b[0] for b in BANDS] + [BANDS[-1][1]],
                        labels=[b[2] for b in BANDS], right=False)
    return df


def wilson(k, n):
    ci = stats.binomtest(int(k), int(n)).proportion_ci(method='wilson')
    return ci.low, ci.high


def loglik(X, y):
    m = LogisticRegression(penalty=None, max_iter=5000).fit(X, y)
    p = np.clip(m.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


df = load()
y = df.y.to_numpy(int)

# ---------------- statistics ----------------
cells = []
for (c, b), x in df.groupby(['centre', 'band'], observed=True):
    lo, hi = wilson(x.y.sum(), len(x))
    cells.append({'centre': c, 'rbmo_label': CENTRES[c][0], 'band': b, 'n': len(x),
                  'pregnancies': int(x.y.sum()), 'observed_%': x.y.mean() * 100,
                  'ci_low_%': lo * 100, 'ci_high_%': hi * 100,
                  'mean_pred_%': x.pred.mean() * 100, 'mean_DT_%': x.p_dt.mean() * 100,
                  'mean_TabPFN_%': x.p_tp.mean() * 100, 'plotted': len(x) >= MIN_N})
C = pd.DataFrame(cells)

band_tests = {}
for _, _, b in BANDS:
    sub = C[(C.band == b) & (C.n >= MIN_N)]
    tab = np.column_stack([sub.pregnancies, sub.n - sub.pregnancies])
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    band_tests[b] = dict(chi2=chi2, dof=dof, p=p, centres=len(sub))

lp = np.log(df.pred / (1 - df.pred)).to_numpy()[:, None]
D = pd.get_dummies(df.centre, drop_first=True).to_numpy(float)
ll0, ll1 = loglik(lp, y), loglik(np.column_stack([lp, D]), y)
lr = 2 * (ll1 - ll0)
lr_p = stats.chi2.sf(lr, D.shape[1])

# ---------------- figure ----------------
fig, ax = plt.subplots(figsize=(150 * MM, 88 * MM))
offs = dict(zip(CENTRES, np.linspace(-.27, .27, len(CENTRES))))
for i, (_, _, b) in enumerate(BANDS):
    if i % 2 == 1:
        ax.axvspan(i - .5, i + .5, color='#f2f2f0', lw=0, zorder=0)
    p = band_tests[b]['p']
    ptxt = '< 0.001' if p < .001 else f'= {p:.3f}'
    ax.text(i, 101, f'centres differ: p {ptxt}', ha='center', va='bottom', fontsize=6.5, color=C_INK)
for _, r in C[C.plotted].iterrows():
    lab, col, mk = CENTRES[r.centre]
    x = [b[2] for b in BANDS].index(r.band) + offs[r.centre]
    ax.plot([x - .045, x + .045], [r['mean_pred_%']] * 2, color=C_GREY, lw=1.6,
            solid_capstyle='butt', zorder=2)
    ax.errorbar(x, r['observed_%'], yerr=[[r['observed_%'] - r['ci_low_%']], [r['ci_high_%'] - r['observed_%']]],
                fmt=mk, ms=5.5 if mk == '^' else 4.5, color=col, mec='white', mew=.5, elinewidth=1, capsize=2, zorder=3)
    ax.text(x, 2, f'{r.n}', ha='center', va='bottom', fontsize=5.8, color='#52514e')
ax.set_xticks(range(len(BANDS)))
ax.set_xticklabels([f'{b}\n(n = {int((df.band == b).sum())})' for _, _, b in BANDS])
ax.set_xlim(-.5, len(BANDS) - .5)
ax.set_ylim(0, 106)
ax.set_yticks(range(0, 101, 20))
ax.set_xlabel('Predicted probability of pregnancy (mean of Digital Twin and TabPFN)')
ax.set_ylabel('Observed clinical pregnancy rate (%)')
ax.text(-.47, 2, 'n:', ha='left', va='bottom', fontsize=5.8, color='#52514e')
h = [Line2D([], [], marker=m, ls='', color=c, mec='white', ms=5, label=f'{name} ({lab})')
     for name, (lab, c, m) in CENTRES.items()]
h += [Line2D([], [], color=C_GREY, lw=1.6, label='Mean predicted probability\nof the same cycles')]
ax.legend(handles=h, loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False, handlelength=1.4,
          labelspacing=.8)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'Figure_prognosis_by_centre.{ext}')
plt.close(fig)

# ---------------- table ----------------
C = C.round(1)
tests = pd.DataFrame([{'band': b, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items()}}
                      for b, t in band_tests.items()])
C.to_csv(OUT / 'prognosis_by_centre.csv', index=False, encoding='utf-8-sig')
tests.assign(lr_centre_adjusted_for_prediction=f'LR chi2 = {lr:.1f}, df = {D.shape[1]}, p = {lr_p:.2g}') \
     .to_csv(OUT / 'prognosis_by_centre_tests.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 220)
print(C.to_string(index=False)); print(tests.to_string(index=False))
print(f'Centre beyond predicted probability: LR chi2 = {lr:.1f}, df = {D.shape[1]}, p = {lr_p:.2g}')
