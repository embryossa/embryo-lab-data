#!/usr/bin/env python3
"""Benchmark classification figure: ROC, PR, reliability. Digital Twin vs TabPFN vs CSDI."""
import numpy as np, pandas as pd, matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, brier_score_loss
mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8.5,
    'axes.titleweight': 'bold', 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 6.8, 'axes.linewidth': .7, 'axes.spines.top': False,
    'axes.spines.right': False, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'pdf.fonttype': 42})
MM = 1 / 25.4
B = '/sessions/nice-festive-hawking/mnt/embryo-lab-data/data/processed/benchmark/'
d = pd.read_csv(B + 'dt_predictions.csv'); y = d.Preg.to_numpy(int); prev = y.mean()
MODELS = [('Digital Twin', 'DIGITAL TWIN', '#0072B2', '-'),
          ('TabPFN', 'PRAI', '#D55E00', '-'),
          ('CSDI', 'p_csdi', '#009E73', '--')]

def tag(ax, s):
    ax.text(-.16, 1.06, s, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom')

fig, ax = plt.subplots(1, 3, figsize=(180 * MM, 62 * MM))
# ROC
ax[0].plot([0, 1], [0, 1], color='#999', lw=.8, ls=(0, (4, 3)))
for name, col, c, ls in MODELS:
    p = pd.to_numeric(d[col], errors='coerce').to_numpy(float)
    fpr, tpr, _ = roc_curve(y, p)
    ax[0].plot(fpr, tpr, color=c, ls=ls, lw=1.3, label=f'{name} (AUC {roc_auc_score(y,p):.3f})')
ax[0].set_xlabel('1 − specificity'); ax[0].set_ylabel('Sensitivity')
ax[0].set_title('ROC'); ax[0].legend(loc='lower right', frameon=False, handlelength=1.4, borderpad=.2); tag(ax[0], 'A')
ax[0].set_aspect('equal')
# PR
ax[1].axhline(prev, color='#999', lw=.8, ls=(0, (4, 3)))
ax[1].text(.98, prev + .01, f'prevalence {prev:.2f}', ha='right', fontsize=6, color='#555')
for name, col, c, ls in MODELS:
    p = pd.to_numeric(d[col], errors='coerce').to_numpy(float)
    pr, rc, _ = precision_recall_curve(y, p)
    ax[1].plot(rc, pr, color=c, ls=ls, lw=1.3, label=f'{name} (AP {average_precision_score(y,p):.3f})')
ax[1].set_xlabel('Recall'); ax[1].set_ylabel('Precision')
ax[1].set_title('Precision–Recall'); ax[1].legend(loc='upper right', frameon=False, handlelength=1.4, borderpad=.2); tag(ax[1], 'B')
ax[1].set_ylim(0, 1.02); ax[1].set_xlim(0,1)
# reliability
ax[2].plot([0, 1], [0, 1], color='#999', lw=.8, ls=(0, (4, 3)))
bins = np.linspace(0, 1, 8)
for name, col, c, ls in MODELS:
    p = pd.to_numeric(d[col], errors='coerce').to_numpy(float)
    idx = np.digitize(p, bins[1:-1])
    xs, ys = [], []
    for b in np.unique(idx):
        m = idx == b
        if m.sum() < 10:
            continue
        xs.append(p[m].mean()); ys.append(y[m].mean())
    ax[2].plot(xs, ys, color=c, ls=ls, marker='o', ms=3.5, lw=1.2, mec='white', mew=.4,
               label=f'{name} (Brier {brier_score_loss(y,p):.3f})')
ax[2].set_xlabel('Mean predicted probability'); ax[2].set_ylabel('Observed frequency')
ax[2].set_title('Reliability'); ax[2].legend(loc='lower right', frameon=False, handlelength=1.4, borderpad=.2); tag(ax[2], 'C')
ax[2].set_xlim(0, 1); ax[2].set_ylim(0, 1); ax[2].set_aspect('equal')
fig.tight_layout(w_pad=1.6)
for ext in ('png', 'pdf'):
    fig.savefig(f'FigureB_classification.{ext}')
print('saved FigureB_classification')
