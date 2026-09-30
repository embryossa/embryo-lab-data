#!/usr/bin/env python3
"""Publication figures: CSDI Digital Twin vs TabPFN-3 on IVF laboratory funnel stages.
Outputs 300-dpi PNG + vector PDF. Colour-blind-safe palette (Okabe-Ito)."""
import numpy as np, pandas as pd, matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

mpl.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8,
    'axes.labelsize': 8, 'axes.titlesize': 8.5, 'axes.titleweight': 'bold',
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': .7, 'xtick.major.width': .7, 'ytick.major.width': .7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
C_CSDI, C_TAB, C_GREY = '#0072B2', '#D55E00', '#666666'
MM = 1 / 25.4

# ---------------- data ----------------
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[2]
os.chdir(Path(__file__).resolve().parent)  # outputs are written next to this script
CSV = ROOT / 'data/processed/fertimed/model_comparison_tabpfn3_vs_digitaltwin.csv'
XLS = ROOT / 'data/processed/fertimed/local_private/model_input.xlsx'
c = pd.read_csv(CSV, encoding='utf-8-sig')
m = pd.read_excel(XLS, sheet_name='model_input').set_index('cycle_uid').loc[c.cycle_uid]
df = c.copy()
df['bl_true'] = m['bl_total'].fillna(0).values
df['good_bl_true'] = m['Число Bl хор.кач-ва'].fillna(0).values
df['frozen_true'] = m['frozen'].fillna(0).values

STAGES = [
    dict(k='2PN', lab='2PN zygotes', y='pn2_actual', dt='dt_pn2_pred',
         tp='tabpfn3_pn2_p50', pre='tabpfn3_pn2', lo=None, hi=None),
    dict(k='BL', lab='Blastocysts', y='bl_true', dt='dt_bl_pred',
         tp='tabpfn3_blasts_after_fert_p50', pre='tabpfn3_blasts_after_fert',
         lo='dt_bl_p025', hi='dt_bl_p975'),
    dict(k='GBL', lab='Good-quality blastocysts', y='good_bl_true', dt='dt_good_bl_pred',
         tp='tabpfn3_good_p50', pre='tabpfn3_good', lo='dt_good_bl_p025', hi='dt_good_bl_p975'),
    dict(k='FRZ', lab='Vitrified embryos', y='frozen_true', dt=None,
         tp='tabpfn3_frozen_p50', pre='tabpfn3_frozen', lo=None, hi=None),
]

def panel_tag(ax, s):
    ax.text(-0.20, 1.06, s, transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='bottom', ha='left')

# =============== FIGURE 1: calibration curves ===============
def fig1():
    ss = STAGES[:3]
    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 72 * MM))
    for ax, s, tag in zip(axes, ss, 'ABC'):
        y = df[s['y']].to_numpy(float)
        series = [('CSDI Digital Twin', df[s['dt']].to_numpy(float), C_CSDI, 'o', '-'),
                  ('TabPFN-3 (P50)', df[s['tp']].to_numpy(float), C_TAB, 's', '-')]
        hi = max(y.max(), max(p.max() for _, p, *_ in series))
        ax.plot([0, hi], [0, hi], color=C_GREY, lw=.8, ls=(0, (4, 3)), zorder=1)
        for name, p, col, mk, ls in series:
            # bin by predicted value into deciles of prediction
            q = np.unique(np.quantile(p, np.linspace(0, 1, 7)))
            if len(q) < 3:
                q = np.unique(np.quantile(p, np.linspace(0, 1, 4)))
            b = np.clip(np.digitize(p, q[1:-1]), 0, len(q) - 2)
            xs, ys, es = [], [], []
            for g in np.unique(b):
                msk = b == g
                if msk.sum() < 5:
                    continue
                xs.append(p[msk].mean()); ys.append(y[msk].mean())
                es.append(y[msk].std(ddof=1) / np.sqrt(msk.sum()))
            ax.errorbar(xs, ys, yerr=es, color=col, marker=mk, ms=3.5, lw=1.2,
                        ls=ls, capsize=1.8, elinewidth=.8, mec='white', mew=.4,
                        label=name, zorder=3)
        ax.set_xlabel('Predicted count (bin mean)')
        if tag == 'A':
            ax.set_ylabel('Observed count (bin mean)')
        ax.set_title(s['lab'])
        lim = hi * 1.05
        ax.set_xlim(-lim * .03, lim); ax.set_ylim(-lim * .03, lim)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='upper left', frameon=False, handlelength=1.6)
        panel_tag(ax, tag)
    axes[0].text(.98, .04, 'dashed = perfect calibration', transform=axes[0].transAxes,
                 ha='right', va='bottom', fontsize=6.5, color=C_GREY, style='italic')
    fig.tight_layout(w_pad=1.8)
    for ext in ('png', 'pdf'):
        fig.savefig(f'Figure1_calibration_curves.{ext}')
    plt.close(fig)

# =============== FIGURE 2: Bland-Altman ===============
def fig2():
    ss = STAGES[:3]
    fig, axes = plt.subplots(2, 3, figsize=(180 * MM, 115 * MM), sharex='col', sharey='col')
    for j, s in enumerate(ss):
        y = df[s['y']].to_numpy(float)
        for i, (name, pcol, col) in enumerate(
                [('CSDI Digital Twin', s['dt'], C_CSDI), ('TabPFN-3 (P50)', s['tp'], C_TAB)]):
            ax = axes[i, j]
            p = df[pcol].to_numpy(float)
            mean_, diff = (p + y) / 2, p - y
            bias, sd = diff.mean(), diff.std(ddof=1)
            loa = (bias - 1.96 * sd, bias + 1.96 * sd)
            jit = (np.random.default_rng(7).random(len(y)) - .5) * .22
            ax.scatter(mean_ + jit, diff + jit, s=7, color=col, alpha=.42,
                       linewidths=0, zorder=2)
            ax.axhline(0, color=C_GREY, lw=.8, ls=(0, (4, 3)), zorder=1)
            ax.axhline(bias, color=col, lw=1.1, zorder=3)
            for L in loa:
                ax.axhline(L, color=col, lw=.8, ls=(0, (1.5, 2)), zorder=3)
            ax.text(.985, .96,
                    f'bias {bias:+.2f}\n95% LoA {loa[0]:+.1f} to {loa[1]:+.1f}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=6.3,
                    linespacing=1.35,
                    bbox=dict(fc='white', ec='none', alpha=.75, pad=1.4))
            if i == 0:
                ax.set_title(s['lab'])
            ax.set_ylabel(f'{name}\nPredicted − observed' if j == 0 else '', fontsize=7.5)
            ax.text(.02, .96, name.split(' ')[0], transform=ax.transAxes, fontsize=6.5,
                    color=col, fontweight='bold', va='top')
            if i == 1:
                ax.set_xlabel('Mean of predicted and observed')
            panel_tag(ax, 'ABCDEF'[i * 3 + j])
    fig.tight_layout(w_pad=1.6, h_pad=1.4)
    for ext in ('png', 'pdf'):
        fig.savefig(f'Figure2_bland_altman.{ext}')
    plt.close(fig)

# =============== FIGURE 3: interval coverage ===============
def fig3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(180 * MM, 65 * MM),
                                   gridspec_kw={'width_ratios': [1.15, 1]})
    labs, tcov, tci, ccov, cci = [], [], [], [], []
    for s in STAGES:
        y = df[s['y']].to_numpy(float)
        lo = df[s['pre'] + '_p10'].to_numpy(float); hi = df[s['pre'] + '_p90'].to_numpy(float)
        cov = np.mean((y >= lo) & (y <= hi))
        ci = stats.binomtest(int(round(cov * len(y))), len(y)).proportion_ci(.95)
        labs.append(s['lab']); tcov.append(cov * 100); tci.append((ci.low * 100, ci.high * 100))
        if s['lo']:
            l = df[s['lo']].to_numpy(float); h = df[s['hi']].to_numpy(float)
            cv = np.mean((y >= l) & (y <= h))
            c2 = stats.binomtest(int(round(cv * len(y))), len(y)).proportion_ci(.95)
            ccov.append(cv * 100); cci.append((c2.low * 100, c2.high * 100))
        else:
            ccov.append(np.nan); cci.append((np.nan, np.nan))
    ypos = np.arange(len(labs))[::-1]
    ax1.axvline(80, color=C_TAB, lw=.9, ls=(0, (4, 3)), zorder=1)
    ax1.axvline(95, color=C_CSDI, lw=.9, ls=(0, (4, 3)), zorder=1)
    for i, yp in enumerate(ypos):
        ax1.plot([tci[i][0], tci[i][1]], [yp + .16] * 2, color=C_TAB, lw=1.1, zorder=2)
        ax1.plot(tcov[i], yp + .16, 'o', color=C_TAB, ms=5, mec='white', mew=.5, zorder=3)
        if not np.isnan(ccov[i]):
            ax1.plot([cci[i][0], cci[i][1]], [yp - .16] * 2, color=C_CSDI, lw=1.1, zorder=2)
            ax1.plot(ccov[i], yp - .16, 's', color=C_CSDI, ms=5, mec='white', mew=.5, zorder=3)
    ax1.set_yticks(ypos); ax1.set_yticklabels(labs)
    ax1.set_xlabel('Empirical coverage (%)'); ax1.set_xlim(45, 102)
    ax1.set_ylim(-.6, len(labs) - .4)
    ax1.set_title('Prediction-interval coverage')
    ax1.legend(handles=[
        Line2D([], [], color=C_TAB, marker='o', ls='-', ms=5, label='TabPFN-3  P10–P90 (nominal 80%)'),
        Line2D([], [], color=C_CSDI, marker='s', ls='-', ms=5, label='CSDI  P2.5–P97.5 (nominal 95%)')],
        loc='upper center', bbox_to_anchor=(.5, -.22), frameon=False, ncol=1)
    panel_tag(ax1, 'A')

    # pinball loss by quantile
    qs = [.1, .25, .5, .75, .9]
    mk = ['o', 's', '^', 'D']
    stage_cols = ['#332288', '#117733', '#88CCEE', '#AA4499']
    for s, mkr, scol in zip(STAGES, mk, stage_cols):
        y = df[s['y']].to_numpy(float); vals = []
        for q, suf in zip(qs, ['p10', 'p25', 'p50', 'p75', 'p90']):
            qq = df[f"{s['pre']}_{suf}"].to_numpy(float)
            vals.append(np.mean(np.maximum(q * (y - qq), (q - 1) * (y - qq))))
        ax2.plot([int(q * 100) for q in qs], vals, marker=mkr, ms=3.8, lw=1.1,
                 color=scol, mec='white', mew=.4, label=s['lab'])
    ax2.set_xlabel('Quantile level (%)'); ax2.set_ylabel('Pinball loss (embryos)')
    ax2.set_title('TabPFN-3 quantile loss'); ax2.set_xticks([10, 25, 50, 75, 90])
    ax2.legend(frameon=False, loc='upper left')
    panel_tag(ax2, 'B')
    fig.tight_layout(w_pad=2.2)
    for ext in ('png', 'pdf'):
        fig.savefig(f'Figure3_interval_calibration.{ext}')
    plt.close(fig)

# =============== FIGURE 4: recalibration vs baseline ===============
def fig4():
    rows = []
    for s in STAGES[:3]:
        y = df[s['y']].to_numpy(float)
        def loo(pred):
            out = np.empty(len(y))
            for i in range(len(y)):
                k = np.ones(len(y), bool); k[i] = False
                out[i] = np.polyval(np.polyfit(pred[k], y[k], 1), pred[i])
            return np.clip(out, 0, None)
        dt, tp = df[s['dt']].to_numpy(float), df[s['tp']].to_numpy(float)
        rows.append(dict(lab=s['lab'], const=np.abs(y - np.median(y)).mean(),
                         c_raw=np.abs(dt - y).mean(), c_rec=np.abs(loo(dt) - y).mean(),
                         t_raw=np.abs(tp - y).mean(), t_rec=np.abs(loo(tp) - y).mean()))
    R = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(150 * MM, 72 * MM))
    x = np.arange(len(R)); w = .19
    bars = [('CSDI, raw', 'c_raw', C_CSDI, .45), ('CSDI, recalibrated', 'c_rec', C_CSDI, 1.0),
            ('TabPFN-3, raw', 't_raw', C_TAB, .45), ('TabPFN-3, recalibrated', 't_rec', C_TAB, 1.0)]
    for i, (name, col, colr, alpha) in enumerate(bars):
        ax.bar(x + (i - 1.5) * w, R[col], w, label=name, color=colr, alpha=alpha,
               edgecolor=colr, linewidth=.6, zorder=2)
    for i, v in enumerate(R['const']):
        ax.plot([x[i] - 2.3 * w, x[i] + 2.3 * w], [v, v], color='black', lw=1.3, zorder=4)
    ax.plot([], [], color='black', lw=1.3, label='Constant baseline (median)')
    ax.set_xticks(x); ax.set_xticklabels(R.lab)
    ax.set_ylabel('Mean absolute error (embryos)')
    ax.set_title('Point-forecast error before and after linear recalibration', pad=26)
    ax.set_ylim(0, max(R.c_raw.max(), R.t_raw.max(), R['const'].max()) * 1.30)
    ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(.5, 1.20),
              columnspacing=1.1, handlelength=1.5)
    ax.grid(axis='y', lw=.4, color='#DDDDDD', zorder=0)
    ax.set_axisbelow(True)
    panel_tag(ax, '')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'Figure4_recalibration.{ext}')
    plt.close(fig)

for f in (fig1, fig2, fig3, fig4):
    f(); print('done', f.__name__)
