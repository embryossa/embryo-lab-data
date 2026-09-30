#!/usr/bin/env python3
"""Bland-Altman agreement of predicted clinical-pregnancy probability, Digital Twin
(`DIGITAL TWIN`) vs TabPFN (`PRAI`), in the prospective centres of the RBMO paper.

One panel per centre: pregnancy rates differ 26-70% between centres, so a pooled bias would
mix centre offsets into the agreement estimate (same reason the paper's primary AUC is
within-centre). The pooled summary is written to the table only, as a sensitivity analysis.
Bland-Altman quantifies agreement between the two models, not accuracy against the outcome.
Outputs 300-dpi PNG + vector PDF + statistics table; no identifiers leave the local layer."""
import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8.5,
    'axes.titleweight': 'bold', 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 6.8, 'axes.linewidth': .7, 'axes.spines.top': False,
    'axes.spines.right': False, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'pdf.fonttype': 42})
MM = 1 / 25.4
C_PREG, C_NOPREG, C_INK, C_GREY = '#117733', '#AA4499', '#222222', '#888888'
MINUS = lambda s: s.replace('-', '−')

OUT = Path(__file__).resolve().parent
REF = OUT.parents[1] / 'data' / 'processed' / 'reference' / 'rbmo_prospective_2026'
# figure label order and RBMO label; data come from the frozen cohort (build_cohort.py)
CENTRES = {'Fertimed': 'Centre A', 'GGRC': 'Centre B', 'SkyFert': 'Centre C', 'Two-Line': '4th centre'}


def load():
    path = REF / 'rbmo_prospective_cohort.csv'
    manifest = json.loads((REF / 'manifest.json').read_text(encoding='utf-8'))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == manifest['cohort_sha256'], 'frozen cohort differs from its manifest'
    d = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
    df = pd.DataFrame({'centre': d.centre, 'p_dt': d['DIGITAL TWIN'], 'p_tp': d['PRAI'], 'y': d['Preg']})
    assert not df.isna().any(axis=None)
    return df


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def ba(a, b):
    """Bias, LoA and their 95% CIs (Bland & Altman 1999: SE(LoA) = sqrt(3 s^2 / n)),
    plus proportional-bias regression of difference on mean."""
    m, diff = (a + b) / 2, a - b
    n, bias, sd = len(diff), diff.mean(), diff.std(ddof=1)
    t = stats.t.ppf(.975, n - 1)
    se_b, se_l = sd / np.sqrt(n), np.sqrt(3 * sd ** 2 / n)
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
    reg = stats.linregress(m, diff)
    return dict(n=n, mean=m, diff=diff, bias=bias, sd=sd,
                bias_ci=(bias - t * se_b, bias + t * se_b),
                loa=(lo, hi), lo_ci=(lo - t * se_l, lo + t * se_l),
                hi_ci=(hi - t * se_l, hi + t * se_l),
                slope=reg.slope, intercept=reg.intercept,
                slope_ci=(reg.slope - t * reg.stderr, reg.slope + t * reg.stderr),
                slope_p=reg.pvalue,
                within_loa=np.mean((diff >= lo) & (diff <= hi)) * 100,
                shapiro_p=stats.shapiro(diff).pvalue)


def row(scale, subset, g, r):
    ci = lambda v, c: f'{v:.3f} [{c[0]:.3f}; {c[1]:.3f}]'
    return {'Scale': scale, 'Centre': subset, 'n': r['n'], 'Pregnancies': int(g.y.sum()),
            'Observed rate %': round(g.y.mean() * 100, 1),
            'Mean DT %': round(g.p_dt.mean() * 100, 1), 'Mean TabPFN %': round(g.p_tp.mean() * 100, 1),
            'Pearson r': round(np.corrcoef(g.p_dt, g.p_tp)[0, 1], 3),
            'Bias [95% CI]': ci(r['bias'], r['bias_ci']), 'SD diff': round(r['sd'], 3),
            'Lower LoA [95% CI]': ci(r['loa'][0], r['lo_ci']),
            'Upper LoA [95% CI]': ci(r['loa'][1], r['hi_ci']),
            '% within LoA': round(r['within_loa'], 1),
            'Prop. slope [95% CI]': ci(r['slope'], r['slope_ci']),
            'Prop. slope p': f'{r["slope_p"]:.2g}', 'Shapiro p (diff)': f'{r["shapiro_p"]:.2g}'}


df = load()
SCALES = {'Probability (pp)': lambda g: (g.p_dt.to_numpy() * 100, g.p_tp.to_numpy() * 100),
          'Logit (log-OR)': lambda g: (logit(g.p_dt.to_numpy()), logit(g.p_tp.to_numpy()))}

rows = []
for scale, f in SCALES.items():
    for name in CENTRES:
        g = df[df.centre == name]
        rows.append(row(scale, name, g, ba(*f(g))))
    rows.append(row(scale, 'Pooled (sensitivity)', df, ba(*f(df))))
    # does disagreement depend on the outcome, within centre? (stratified OLS: diff ~ y + centre)
    a, b = f(df)
    X = np.column_stack([df.y.to_numpy(), pd.get_dummies(df.centre).to_numpy(float)])
    beta, *_ = np.linalg.lstsq(X, a - b, rcond=None)
    res = (a - b) - X @ beta
    s2 = res @ res / (len(res) - X.shape[1])
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])
    p = 2 * stats.t.sf(abs(beta[0] / se), len(res) - X.shape[1])
    rows.append({'Scale': scale, 'Centre': 'Pregnant - not pregnant, within centre',
                 'n': len(df), 'Bias [95% CI]': f'{beta[0]:+.3f} (SE {se:.3f}, p={p:.2g})'})

# ---------------- figure: probability scale, one panel per centre ----------------
fig, axes = plt.subplots(2, 2, figsize=(180 * MM, 140 * MM), sharex=True, sharey=True)
allr = [ba(*SCALES['Probability (pp)'](df[df.centre == c])) for c in CENTRES]
xall = np.concatenate([r['mean'] for r in allr]); dall = np.concatenate([r['diff'] for r in allr])
x0, x1 = xall.min() - 3, xall.max() + 3
ylo, yhi = dall.min(), dall.max()
for ax, name, r, tag in zip(axes.flat, CENTRES, allr, 'ABCD'):
    g = df[df.centre == name]
    y = g.y.to_numpy(int)
    ax.axhspan(*r['bias_ci'], color=C_INK, alpha=.10, lw=0, zorder=1)
    for c in (r['lo_ci'], r['hi_ci']):
        ax.axhspan(*c, color=C_GREY, alpha=.12, lw=0, zorder=1)
    ax.axhline(0, color=C_GREY, lw=.8, ls=(0, (4, 3)), zorder=2)
    for yy, mk, col, lab in [(0, 'o', C_NOPREG, 'Not pregnant'), (1, '^', C_PREG, 'Pregnant')]:
        m = y == yy
        ax.scatter(r['mean'][m], r['diff'][m], s=10, marker=mk, color=col, alpha=.6,
                   linewidths=.3, edgecolors='white', zorder=3, label=lab)
    ax.axhline(r['bias'], color=C_INK, lw=1.2, zorder=4)
    for L in r['loa']:
        ax.axhline(L, color=C_INK, lw=.8, ls=(0, (1.5, 2)), zorder=4)
    xs = np.array([r['mean'].min(), r['mean'].max()])
    ax.plot(xs, r['intercept'] + r['slope'] * xs, color=C_INK, lw=.9, ls=(0, (6, 2, 1, 2)), zorder=4)
    for yv, txt in [(r['bias'], f'bias {r["bias"]:+.1f}'), (r['loa'][1], f'{r["loa"][1]:+.1f}'),
                    (r['loa'][0], f'{r["loa"][0]:+.1f}')]:
        ax.text(x1, yv, MINUS(txt), ha='right', va='bottom', fontsize=6.3, color=C_INK,
                bbox=dict(fc='white', ec='none', alpha=.7, pad=.6), zorder=5)
    ax.text(.02, .02, MINUS(f'n = {r["n"]}, pregnancy {y.mean() * 100:.0f}%   '
                            f'slope {r["slope"]:+.2f} [{r["slope_ci"][0]:+.2f}; {r["slope_ci"][1]:+.2f}]'),
            transform=ax.transAxes, ha='left', va='bottom', fontsize=6.3, color=C_INK,
            bbox=dict(fc='white', ec='none', alpha=.7, pad=.6), zorder=5)
    ax.set_title(f'{name} ({CENTRES[name]})')
    ax.text(-.12, 1.04, tag, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom')
    ax.tick_params(labelbottom=True, labelleft=True)
axes[0, 0].set_xlim(x0, x1)
axes[0, 0].set_ylim(ylo - (yhi - ylo) * .13, yhi + (yhi - ylo) * .04)
for ax in axes[1]:
    ax.set_xlabel('Mean of Digital Twin and TabPFN (%)')
for ax in axes[:, 0]:
    ax.set_ylabel('Digital Twin − TabPFN (pp)')

h = [Line2D([], [], color=C_INK, lw=1.2, label='Mean bias (band: 95% CI)'),
     Line2D([], [], color=C_INK, lw=.8, ls=(0, (1.5, 2)), label='95% limits of agreement (band: 95% CI)'),
     Line2D([], [], color=C_INK, lw=.9, ls=(0, (6, 2, 1, 2)), label='Regression of difference on mean'),
     Line2D([], [], color=C_GREY, lw=.8, ls=(0, (4, 3)), label='Zero difference')]
h += axes[0, 0].get_legend_handles_labels()[0]
fig.legend(handles=h, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, -.07),
           handlelength=2.4, columnspacing=1.4)
fig.tight_layout(w_pad=2.0, h_pad=1.6)
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'Figure_bland_altman_pregnancy_prospective.{ext}')
plt.close(fig)

T = pd.DataFrame(rows)
T.to_csv(OUT / 'bland_altman_pregnancy_prospective.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 30)
print(T.to_string(index=False))
