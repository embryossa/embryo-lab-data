#!/usr/bin/env python3
"""Bland-Altman agreement of predicted clinical-pregnancy probability, Digital Twin
(`DIGITAL TWIN`) vs TabPFN (`PRAI`), whole RBMO prospective cohort (1,026 cycles) in one plot.

Panel A: probability scale (percentage points). Panel B: logit scale, where the difference is
the log odds ratio between the models and variance is not compressed near 0 and 1.
Bland-Altman quantifies agreement between the two models, not accuracy against the outcome.
Supporting checks written to the table: centre-adjusted SD and slope (how much of the spread
and trend is between-centre), percentile limits (differences are not normal), outcome effect.
Outputs 300-dpi PNG + vector PDF + statistics table. Reads the frozen cohort (build_cohort.py)."""
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


def load():
    path = REF / 'rbmo_prospective_cohort.csv'
    manifest = json.loads((REF / 'manifest.json').read_text(encoding='utf-8'))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == manifest['cohort_sha256'], 'frozen cohort differs from its manifest'
    d = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
    df = pd.DataFrame({'centre': d.centre, 'p_dt': d['DIGITAL TWIN'], 'p_tp': d['PRAI'], 'y': d['Preg']})
    assert not df.isna().any(axis=None) and len(df) == manifest['n_cycles']
    return df


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def ols(y, X):
    """Coefficients, SEs and residual SD of y ~ X (X includes its own intercept/dummies)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = res @ res / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return beta, se, np.sqrt(s2), dof


def ba(a, b, centre, y):
    m, diff = (a + b) / 2, a - b
    n, bias, sd = len(diff), diff.mean(), diff.std(ddof=1)
    t = stats.t.ppf(.975, n - 1)
    se_b, se_l = sd / np.sqrt(n), np.sqrt(3 * sd ** 2 / n)          # Bland & Altman 1999
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
    reg = stats.linregress(m, diff)
    D = pd.get_dummies(centre).to_numpy(float)
    # within-centre: SD of differences after removing centre means; slope of diff on mean
    _, _, sd_within, _ = ols(diff, D)
    bw, sew, _, dofw = ols(diff, np.column_stack([m, D]))
    tw = stats.t.ppf(.975, dofw)
    # outcome effect, unadjusted and within centre
    wt = stats.ttest_ind(diff[y == 1], diff[y == 0], equal_var=False)
    by, sey, _, dofy = ols(diff, np.column_stack([y, D]))
    return dict(n=n, mean=m, diff=diff, bias=bias, sd=sd,
                bias_ci=(bias - t * se_b, bias + t * se_b),
                loa=(lo, hi), lo_ci=(lo - t * se_l, lo + t * se_l), hi_ci=(hi - t * se_l, hi + t * se_l),
                within_loa=np.mean((diff >= lo) & (diff <= hi)) * 100,
                pct_loa=tuple(np.percentile(diff, [2.5, 97.5])),
                slope=reg.slope, intercept=reg.intercept,
                slope_ci=(reg.slope - t * reg.stderr, reg.slope + t * reg.stderr), slope_p=reg.pvalue,
                slope_w=bw[0], slope_w_ci=(bw[0] - tw * sew[0], bw[0] + tw * sew[0]),
                slope_w_p=2 * stats.t.sf(abs(bw[0] / sew[0]), dofw),
                sd_within=sd_within,
                shapiro_p=stats.shapiro(diff).pvalue, skew=stats.skew(diff),
                hetero=stats.spearmanr(m, np.abs(diff - bias)),
                preg_diff=diff[y == 1].mean() - diff[y == 0].mean(), preg_p=wt.pvalue,
                preg_diff_w=by[0], preg_p_w=2 * stats.t.sf(abs(by[0] / sey[0]), dofy),
                sd_a=a.std(ddof=1), sd_b=b.std(ddof=1), r=np.corrcoef(a, b)[0, 1])


df = load()
y = df.y.to_numpy(int)
SCALES = [dict(k='prob', a=df.p_dt.to_numpy() * 100, b=df.p_tp.to_numpy() * 100, unit='pp',
               xl='Mean of Digital Twin and TabPFN (%)',
               yl='Digital Twin − TabPFN (percentage points)', title='Probability scale', fmt='{:+.1f}'),
          dict(k='logit', a=logit(df.p_dt.to_numpy()), b=logit(df.p_tp.to_numpy()), unit='log-OR',
               xl='Mean of Digital Twin and TabPFN (logit)',
               yl='Digital Twin − TabPFN (log odds ratio)', title='Logit scale', fmt='{:+.2f}')]

rows = []
fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 80 * MM))
for ax, s, tag in zip(axes, SCALES, 'AB'):
    r = ba(s['a'], s['b'], df.centre, y)
    xpad = np.ptp(r['mean']) * .04
    x0, x1 = r['mean'].min() - xpad, r['mean'].max() + xpad
    ax.axhspan(*r['bias_ci'], color=C_INK, alpha=.10, lw=0, zorder=1)
    for ci in (r['lo_ci'], r['hi_ci']):
        ax.axhspan(*ci, color=C_GREY, alpha=.12, lw=0, zorder=1)
    ax.axhline(0, color=C_GREY, lw=.8, ls=(0, (4, 3)), zorder=2)
    for yy, mk, col, lab in [(0, 'o', C_NOPREG, 'Not pregnant'), (1, '^', C_PREG, 'Pregnant')]:
        m = y == yy
        ax.scatter(r['mean'][m], r['diff'][m], s=7, marker=mk, color=col, alpha=.45,
                   linewidths=.25, edgecolors='white', zorder=3, label=f'{lab} (n={m.sum()})')
    ax.axhline(r['bias'], color=C_INK, lw=1.2, zorder=4)
    for L in r['loa']:
        ax.axhline(L, color=C_INK, lw=.8, ls=(0, (1.5, 2)), zorder=4)
    xs = np.array([x0, x1])
    ax.plot(xs, r['intercept'] + r['slope'] * xs, color=C_INK, lw=.9, ls=(0, (6, 2, 1, 2)), zorder=4)
    f = lambda v: MINUS(s['fmt'].format(v))
    for yv, txt in [(r['bias'], f'bias {f(r["bias"])}'), (r['loa'][1], f'+1.96 SD {f(r["loa"][1])}'),
                    (r['loa'][0], f'−1.96 SD {f(r["loa"][0])}')]:
        ax.text(x1, yv, txt, ha='right', va='bottom', fontsize=6.3, color=C_INK,
                bbox=dict(fc='white', ec='none', alpha=.7, pad=.6), zorder=5)
    ax.text(.02, .02, MINUS(f'n = {r["n"]}   proportional bias: slope {r["slope"]:+.2f} '
                            f'[{r["slope_ci"][0]:+.2f}; {r["slope_ci"][1]:+.2f}]'),
            transform=ax.transAxes, ha='left', va='bottom', fontsize=6.3, color=C_INK,
            bbox=dict(fc='white', ec='none', alpha=.7, pad=.6), zorder=5)
    ax.set_xlim(x0, x1)
    ylo, yhi = r['diff'].min(), r['diff'].max()
    ax.set_ylim(ylo - (yhi - ylo) * .13, yhi + (yhi - ylo) * .04)
    ax.set_xlabel(s['xl']); ax.set_ylabel(s['yl']); ax.set_title(s['title'])
    ax.text(-.16, 1.06, tag, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom')

    ci = lambda v, c: f'{v:.4f} [{c[0]:.4f}; {c[1]:.4f}]'
    rows.append({'Scale': s['title'], 'Unit': s['unit'], 'n': r['n'],
                 'SD DT': round(r['sd_a'], 4), 'SD TabPFN': round(r['sd_b'], 4), 'Pearson r': round(r['r'], 3),
                 'Bias [95% CI]': ci(r['bias'], r['bias_ci']), 'SD diff': round(r['sd'], 4),
                 'SD diff within centre': round(r['sd_within'], 4),
                 'Lower LoA [95% CI]': ci(r['loa'][0], r['lo_ci']),
                 'Upper LoA [95% CI]': ci(r['loa'][1], r['hi_ci']),
                 '% within LoA': round(r['within_loa'], 1),
                 'Percentile LoA 2.5-97.5': f'{r["pct_loa"][0]:.4f}; {r["pct_loa"][1]:.4f}',
                 'Prop. slope [95% CI]': ci(r['slope'], r['slope_ci']), 'Prop. slope p': f'{r["slope_p"]:.2g}',
                 'Prop. slope within centre [95% CI]': ci(r['slope_w'], r['slope_w_ci']),
                 'Prop. slope within centre p': f'{r["slope_w_p"]:.2g}',
                 'Shapiro p': f'{r["shapiro_p"]:.2g}', 'Skewness': round(r['skew'], 2),
                 'Heterosc. rho(|d-bias|, mean)': round(r['hetero'].statistic, 3),
                 'Heterosc. p': f'{r["hetero"].pvalue:.2g}',
                 'Pregnant - not pregnant (Welch)': f'{r["preg_diff"]:+.4f}, p={r["preg_p"]:.2g}',
                 'Pregnant - not pregnant (within centre)': f'{r["preg_diff_w"]:+.4f}, p={r["preg_p_w"]:.2g}'})

h = [Line2D([], [], color=C_INK, lw=1.2, label='Mean bias (band: 95% CI)'),
     Line2D([], [], color=C_INK, lw=.8, ls=(0, (1.5, 2)), label='95% limits of agreement (band: 95% CI)'),
     Line2D([], [], color=C_INK, lw=.9, ls=(0, (6, 2, 1, 2)), label='Regression of difference on mean'),
     Line2D([], [], color=C_GREY, lw=.8, ls=(0, (4, 3)), label='Zero difference')]
h += axes[0].get_legend_handles_labels()[0]
fig.legend(handles=h, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, -.13),
           handlelength=2.4, columnspacing=1.4)
fig.tight_layout(w_pad=2.2)
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'Figure_bland_altman_pregnancy_pooled.{ext}')
plt.close(fig)

T = pd.DataFrame(rows)
T.to_csv(OUT / 'bland_altman_pregnancy_pooled.csv', index=False, encoding='utf-8-sig')
print(T.T.to_string(header=False))
print(f'\nObserved {y.mean() * 100:.1f}% | mean DT {df.p_dt.mean() * 100:.1f}% | '
      f'mean TabPFN {df.p_tp.mean() * 100:.1f}%')
