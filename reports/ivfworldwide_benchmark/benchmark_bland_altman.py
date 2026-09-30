#!/usr/bin/env python3
"""Bland-Altman agreement of predicted clinical-pregnancy probability:
Digital Twin (ensemble, `DIGITAL TWIN`) vs TabPFN (`PRAI`), public benchmark cohort.

Bland-Altman compares two methods measuring the same quantity, so it is applied to the
two models' probabilities, not to model vs binary outcome. Panel A is on the probability
scale (percentage points); panel B on the logit scale, where the difference is the log
odds ratio between the models and the variance is not squeezed near 0 and 1.
Outputs 300-dpi PNG + vector PDF + statistics table."""
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

OUT = Path(__file__).resolve().parent
B = OUT.parents[1] / 'data' / 'processed' / 'benchmark'
d = pd.read_csv(B / 'dt_predictions.csv')
d['grp'] = d.patient_id.str.extract(r'^(FRESH|FET_FC)')[0]
p_dt = pd.to_numeric(d['DIGITAL TWIN'], errors='coerce').to_numpy(float)
p_tp = pd.to_numeric(d['PRAI'], errors='coerce').to_numpy(float)
y = d.Preg.to_numpy(int)
ok = np.isfinite(p_dt) & np.isfinite(p_tp)
assert ok.all(), f'{(~ok).sum()} rows without both probabilities'


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
                shapiro_p=stats.shapiro(diff).pvalue,
                hetero_rho=stats.spearmanr(m, np.abs(diff - bias)).statistic,
                hetero_p=stats.spearmanr(m, np.abs(diff - bias)).pvalue)


SCALES = [dict(k='prob', a=p_dt * 100, b=p_tp * 100, unit='pp',
               xl='Mean of Digital Twin and TabPFN (%)',
               yl='Digital Twin − TabPFN (percentage points)', title='Probability scale', fmt='{:+.1f}'),
          dict(k='logit', a=logit(p_dt), b=logit(p_tp), unit='log-OR',
               xl='Mean of Digital Twin and TabPFN (logit)',
               yl='Digital Twin − TabPFN (log odds ratio)', title='Logit scale', fmt='{:+.2f}')]

rows = []
fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 78 * MM))
for ax, s, tag in zip(axes, SCALES, 'AB'):
    r = ba(s['a'], s['b'])
    xpad = np.ptp(r['mean']) * .04
    x0, x1 = r['mean'].min() - xpad, r['mean'].max() + xpad
    # CI bands first so points and lines sit on top
    ax.axhspan(*r['bias_ci'], color=C_INK, alpha=.10, lw=0, zorder=1)
    for ci in (r['lo_ci'], r['hi_ci']):
        ax.axhspan(*ci, color=C_GREY, alpha=.12, lw=0, zorder=1)
    ax.axhline(0, color=C_GREY, lw=.8, ls=(0, (4, 3)), zorder=2)
    for yy, mk, col, lab in [(0, 'o', C_NOPREG, 'Not pregnant'), (1, '^', C_PREG, 'Pregnant')]:
        m = y == yy
        ax.scatter(r['mean'][m], r['diff'][m], s=9, marker=mk, color=col, alpha=.55,
                   linewidths=.3, edgecolors='white', zorder=3, label=f'{lab} (n={m.sum()})')
    ax.axhline(r['bias'], color=C_INK, lw=1.2, zorder=4)
    for L in r['loa']:
        ax.axhline(L, color=C_INK, lw=.8, ls=(0, (1.5, 2)), zorder=4)
    xs = np.array([x0, x1])
    ax.plot(xs, r['intercept'] + r['slope'] * xs, color=C_INK, lw=.9, ls=(0, (6, 2, 1, 2)),
            zorder=4)
    f = lambda v: s['fmt'].format(v).replace('-', '−')
    for yv, txt in [(r['bias'], f'bias {f(r["bias"])}'),
                    (r['loa'][1], f'+1.96 SD {f(r["loa"][1])}'),
                    (r['loa'][0], f'−1.96 SD {f(r["loa"][0])}')]:
        ax.text(x1, yv, txt, ha='right', va='bottom', fontsize=6.3, color=C_INK,
                bbox=dict(fc='white', ec='none', alpha=.7, pad=.6), zorder=5)
    ax.text(.02, .02,
            (f'proportional bias: slope {r["slope"]:+.2f} '
             f'[{r["slope_ci"][0]:+.2f}; {r["slope_ci"][1]:+.2f}]').replace('-', '−'),
            transform=ax.transAxes, ha='left', va='bottom', fontsize=6.3, color=C_INK,
            bbox=dict(fc='white', ec='none', alpha=.7, pad=.6), zorder=5)
    ax.set_xlim(x0, x1)
    ylo, yhi = r['diff'].min(), r['diff'].max()
    ax.set_ylim(ylo - (yhi - ylo) * .13, yhi + (yhi - ylo) * .04)
    ax.set_xlabel(s['xl']); ax.set_ylabel(s['yl']); ax.set_title(s['title'])
    ax.text(-.16, 1.06, tag, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom')

    def fmt_ci(v, ci): return f'{v:.4f} [{ci[0]:.4f}; {ci[1]:.4f}]'
    rows.append({'Scale': s['title'], 'Unit': s['unit'], 'Subset': 'All', 'n': r['n'],
                 'Bias [95% CI]': fmt_ci(r['bias'], r['bias_ci']), 'SD diff': round(r['sd'], 4),
                 'Lower LoA [95% CI]': fmt_ci(r['loa'][0], r['lo_ci']),
                 'Upper LoA [95% CI]': fmt_ci(r['loa'][1], r['hi_ci']),
                 '% within LoA': round(r['within_loa'], 1),
                 'Prop. slope [95% CI]': fmt_ci(r['slope'], r['slope_ci']),
                 'Prop. slope p': f'{r["slope_p"]:.2g}',
                 'Shapiro p (diff)': f'{r["shapiro_p"]:.2g}',
                 'Heterosc. rho(|d-bias|,mean)': round(r['hetero_rho'], 3),
                 'Heterosc. p': f'{r["hetero_p"]:.2g}'})
    # subgroups: bias only (LoA of small subgroups are not interpreted)
    for name, msk in [('FRESH', d.grp.eq('FRESH').to_numpy()), ('FET_FC', d.grp.eq('FET_FC').to_numpy()),
                      ('Pregnant', y == 1), ('Not pregnant', y == 0)]:
        rs = ba(s['a'][msk], s['b'][msk])
        rows.append({'Scale': s['title'], 'Unit': s['unit'], 'Subset': name, 'n': rs['n'],
                     'Bias [95% CI]': fmt_ci(rs['bias'], rs['bias_ci']),
                     'SD diff': round(rs['sd'], 4),
                     'Lower LoA [95% CI]': fmt_ci(rs['loa'][0], rs['lo_ci']),
                     'Upper LoA [95% CI]': fmt_ci(rs['loa'][1], rs['hi_ci'])})
    wt = stats.ttest_ind(r['diff'][y == 1], r['diff'][y == 0], equal_var=False)
    rows.append({'Scale': s['title'], 'Unit': s['unit'], 'Subset': 'Pregnant - Not pregnant (Welch)',
                 'Bias [95% CI]': f'Δbias {r["diff"][y == 1].mean() - r["diff"][y == 0].mean():+.4f}, '
                                  f'p={wt.pvalue:.2g}'})

h = [Line2D([], [], color=C_INK, lw=1.2, label='Mean bias (band: 95% CI)'),
     Line2D([], [], color=C_INK, lw=.8, ls=(0, (1.5, 2)), label='95% limits of agreement (band: 95% CI)'),
     Line2D([], [], color=C_INK, lw=.9, ls=(0, (6, 2, 1, 2)), label='Regression of difference on mean'),
     Line2D([], [], color=C_GREY, lw=.8, ls=(0, (4, 3)), label='Zero difference')]
h += axes[0].get_legend_handles_labels()[0]
fig.legend(handles=h, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, -.13),
           handlelength=2.4, columnspacing=1.4)
fig.tight_layout(w_pad=2.2)
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'FigureB2_bland_altman_pregnancy.{ext}')
plt.close(fig)

T = pd.DataFrame(rows)
T.to_csv(OUT / 'bench_B4_bland_altman.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(T.to_string(index=False))
print(f'\nObserved pregnancy rate {y.mean() * 100:.1f}% | mean DT {p_dt.mean() * 100:.1f}% | '
      f'mean TabPFN {p_tp.mean() * 100:.1f}% | SD DT {p_dt.std(ddof=1) * 100:.1f} / '
      f'TabPFN {p_tp.std(ddof=1) * 100:.1f} pp | Pearson r {np.corrcoef(p_dt, p_tp)[0, 1]:.3f}')
