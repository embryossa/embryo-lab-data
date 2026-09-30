#!/usr/bin/env python3
"""Benchmark (public FRESH + FET_FC cohorts): CSDI/Digital Twin vs TabPFN.
PART A: funnel-stage regression (MII/2PN/Bl/Good Bl/Frozen), DT point vs TabPFN-3 P50.
PART B: pregnancy classification, Digital Twin vs TabPFN vs CSDI component."""
import numpy as np, pandas as pd
from scipy import stats

rng = np.random.default_rng(20260720)
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[2]
os.chdir(Path(__file__).resolve().parent)  # outputs are written next to this script
B = str(ROOT / 'data/processed/benchmark') + '/'
dt = pd.read_csv(B + 'dt_predictions.csv')
qc = pd.read_csv(B + 'regression_qc.csv')
dt['grp'] = dt.patient_id.str.extract(r'^(FRESH|FET_FC)')[0]

# ================= metric helpers =================
def ccc(y, p):
    my, mp = y.mean(), p.mean()
    return 2 * ((y - my) * (p - mp)).mean() / (y.var(0) + p.var(0) + (my - mp) ** 2)

def reg_metrics(y, p):
    e = p - y
    return dict(n=len(y), MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()), ME=e.mean(),
                within1=np.mean(np.abs(e) <= 1) * 100,
                Spearman=stats.spearmanr(y, p).statistic, CCC=ccc(y, p),
                R2=1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum())

def boot_delta(y, pa, pb, stat='MAE', Bn=10000):
    f = (lambda yy, pp: np.abs(pp - yy).mean()) if stat == 'MAE' else \
        (lambda yy, pp: np.sqrt(((pp - yy) ** 2).mean()))
    idx = rng.integers(0, len(y), size=(Bn, len(y)))
    d = np.array([f(y[i], pa[i]) - f(y[i], pb[i]) for i in idx])
    return f(y, pa) - f(y, pb), np.percentile(d, 2.5), np.percentile(d, 97.5)

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(run, 1.0)
    return adj

# =====================================================================
# PART A — REGRESSION
# =====================================================================
print('#' * 86)
print('# PART A. FUNNEL-STAGE REGRESSION  (public benchmark, n=489 DT / 492 TabPFN-3)')
print('#' * 86)
print('Datasets:', dt.grp.value_counts().to_dict(),
      '| DT skipped 3 rows with missing Age -> join on cycle_uid, n=489')

# align DT med_* to qc actuals via cycle_uid
j = qc.merge(dt[['cycle_uid', 'med_mii', 'med_pn2', 'med_bl', 'med_goodbl', 'grp']],
             on='cycle_uid', how='inner')
print(f'joined rows: {len(j)}')

# MII leakage check
leak = np.isclose(pd.to_numeric(dt.med_mii, errors='coerce'), pd.to_numeric(dt.MII, errors='coerce')).mean()
print(f'DT med_mii == observed MII: {leak*100:.0f}%  -> MII excluded from DT comparison (input leakage)\n')

RSTAGES = [
    dict(k='2PN', lab='2PN zygotes', act='pn2_actual', dt='med_pn2', p50='TPFN3_pn2_p50',
         pre='TPFN3_pn2', leak=False),
    dict(k='BL', lab='Blastocysts', act='blasts_after_fert_actual', dt='med_bl',
         p50='TPFN3_blasts_after_fert_p50', pre='TPFN3_blasts_after_fert', leak=False),
    dict(k='GBL', lab='Good-quality blastocysts', act='good_actual', dt='med_goodbl',
         p50='TPFN3_good_p50', pre='TPFN3_good', leak=False),
    dict(k='FRZ', lab='Vitrified embryos', act='frozen_actual', dt=None,
         p50='TPFN3_frozen_p50', pre='TPFN3_frozen', leak=False),
    dict(k='MII', lab='MII (leaked in DT)', act='mii_actual', dt='med_mii',
         p50='TPFN3_mii_p50', pre='TPFN3_mii', leak=True),
]

print('--- A1. Point accuracy (DT median vs TabPFN-3 P50) ---')
acc, tests = [], []
for s in RSTAGES:
    y = j[s['act']].to_numpy(float)
    tp = j[s['p50']].to_numpy(float)
    if s['dt']:
        dtp = j[s['dt']].to_numpy(float)
        acc.append({**reg_metrics(y, dtp), 'Стадия': s['lab'],
                    'Модель': 'CSDI/DT' + (' [утечка]' if s['leak'] else '')})
    acc.append({**reg_metrics(y, tp), 'Стадия': s['lab'], 'Модель': 'TabPFN-3'})
    if s['dt'] and not s['leak']:
        a, b = np.abs(dtp - y), np.abs(tp - y)
        w = stats.wilcoxon(a, b).pvalue if (a != b).any() else np.nan
        o, lo, hi = boot_delta(y, dtp, tp, 'MAE')
        tests.append(dict(Стадия=s['lab'], n=len(y), dMAE=o, lo=lo, hi=hi,
                          winT=(a > b).mean() * 100, winD=(a < b).mean() * 100, p=w))
A = pd.DataFrame(acc)[['Стадия', 'Модель', 'n', 'MAE', 'RMSE', 'ME', 'within1', 'Spearman', 'CCC', 'R2']]
print(A.round(3).to_string(index=False))

print('\n--- A2. Paired tests (Wilcoxon + bootstrap 10k, Holm on 3 primary stages) ---')
T = pd.DataFrame(tests)
T['p_holm'] = holm(T.p.values)
T['ΔMAE(DT-TabPFN)'] = T.dMAE.round(3)
T['95% CI'] = T.apply(lambda r: f'[{r.lo:+.3f}; {r.hi:+.3f}]', axis=1)
T['%win TabPFN'] = T.winT.round(1)
T['p Holm'] = T.p_holm.map(lambda v: f'{v:.2e}' if v < 1e-3 else f'{v:.4f}')
T['Winner'] = np.where(T.p_holm >= .05, 'no diff', np.where(T.dMAE > 0, 'TabPFN-3', 'CSDI/DT'))
print(T[['Стадия', 'n', 'ΔMAE(DT-TabPFN)', '95% CI', '%win TabPFN', 'p Holm', 'Winner']].to_string(index=False))
print('  ΔMAE>0 => TabPFN-3 more accurate')

print('\n--- A3. Interval calibration ---')
cal = []
for s in RSTAGES:
    y = j[s['act']].to_numpy(float)
    lo = j[s['pre'] + '_p10'].to_numpy(float); hi = j[s['pre'] + '_p90'].to_numpy(float)
    cov = np.mean((y >= lo) & (y <= hi))
    ci = stats.binomtest(int(round(cov * len(y))), len(y)).proportion_ci(.95)
    cal.append(dict(Стадия=s['lab'], n=len(y), Модель='TabPFN-3 P10-P90 (nom 80%)',
                    Coverage=round(cov * 100, 1), CI=f'[{ci.low*100:.1f};{ci.high*100:.1f}]',
                    Dev=round(cov * 100 - 80, 1), Width=round(np.mean(hi - lo), 2)))
print(pd.DataFrame(cal).to_string(index=False))
print('  (DT CI columns not present for funnel stages in benchmark export)')

print('\n--- A4. Recalibration vs constant baseline (LOO linear) ---')
rec = []
for s in RSTAGES:
    if not s['dt'] or s['leak']:
        continue
    y = j[s['act']].to_numpy(float)
    def loo(pred):
        out = np.empty(len(y))
        for i in range(len(y)):
            k = np.ones(len(y), bool); k[i] = False
            out[i] = np.polyval(np.polyfit(pred[k], y[k], 1), pred[i])
        return np.clip(out, 0, None)
    dtp, tp = j[s['dt']].to_numpy(float), j[s['p50']].to_numpy(float)
    rec.append(dict(Стадия=s['lab'], MAE_const=round(np.abs(y - np.median(y)).mean(), 3),
                    MAE_DT=round(np.abs(dtp - y).mean(), 3), MAE_DT_recal=round(np.abs(loo(dtp) - y).mean(), 3),
                    MAE_TabPFN=round(np.abs(tp - y).mean(), 3), MAE_TabPFN_recal=round(np.abs(loo(tp) - y).mean(), 3),
                    beats_const=('yes' if min(np.abs(loo(dtp) - y).mean(), np.abs(loo(tp) - y).mean())
                                 < np.abs(y - np.median(y)).mean() else 'NO')))
R = pd.DataFrame(rec)
print(R.to_string(index=False))

A.to_csv('bench_A1_accuracy.csv', index=False)
T.to_csv('bench_A2_tests.csv', index=False)
pd.DataFrame(cal).to_csv('bench_A3_calibration.csv', index=False)
R.to_csv('bench_A4_recalibration.csv', index=False)
print('\n[Part A tables saved]')

# =====================================================================
# PART B — PREGNANCY CLASSIFICATION
# =====================================================================
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             matthews_corrcoef, f1_score, roc_curve, confusion_matrix, balanced_accuracy_score, log_loss)

print('\n' + '#' * 86)
print('# PART B. PREGNANCY CLASSIFICATION  (ground truth Preg, n=489, positives=142, prev=29.0%)')
print('#' * 86)

y = dt.Preg.to_numpy(int)
prev = y.mean()

def delong_var(y_true, prob):
    """Fast DeLong: returns AUC and structural components for covariance."""
    pos = prob[y_true == 1]; neg = prob[y_true == 0]
    m, n = len(pos), len(neg)
    # midrank
    def midrank(x):
        j = np.argsort(x); z = x[j]; N = len(x); r = np.empty(N)
        i = 0
        while i < N:
            k = i
            while k < N - 1 and z[k + 1] == z[i]:
                k += 1
            r[i:k + 1] = 0.5 * (i + k) + 1
            i = k + 1
        out = np.empty(N); out[j] = r; return out
    tx = midrank(pos); ty = midrank(neg); tz = midrank(np.concatenate([pos, neg]))
    auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
    v01 = (tz[:m] - tx) / n
    v10 = 1 - (tz[m:] - ty) / m
    # Дисперсии компонент здесь не считаются: delong_test ниже вычисляет их сам
    # из возвращаемых v01/v10 (нужны ещё и ковариации между двумя моделями).
    return auc, v01, v10, m, n

def delong_test(y_true, p1, p2):
    a1, v01_1, v10_1, m, n = delong_var(y_true, p1)
    a2, v01_2, v10_2, _, _ = delong_var(y_true, p2)
    cov = np.cov(np.vstack([v01_1, v01_2]))[0, 1] / m + np.cov(np.vstack([v10_1, v10_2]))[0, 1] / n
    var1 = np.var(v01_1, ddof=1) / m + np.var(v10_1, ddof=1) / n
    var2 = np.var(v01_2, ddof=1) / m + np.var(v10_2, ddof=1) / n
    se = np.sqrt(var1 + var2 - 2 * cov)
    z = (a1 - a2) / se if se > 0 else 0.0
    return a1, a2, z, 2 * stats.norm.sf(abs(z))

def auc_ci(y_true, prob, Bn=2000):
    idx = rng.integers(0, len(y_true), size=(Bn, len(y_true)))
    vals = []
    for i in idx:
        if y_true[i].sum() in (0, len(i)):
            continue
        vals.append(roc_auc_score(y_true[i], prob[i]))
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)

def cal_slope_intercept(y_true, prob):
    eps = 1e-6; lp = np.log(np.clip(prob, eps, 1 - eps) / (1 - np.clip(prob, eps, 1 - eps)))
    X = np.column_stack([np.ones_like(lp), lp])
    # newton IRLS logistic
    b = np.zeros(2)
    for _ in range(50):
        eta = X @ b; mu = 1 / (1 + np.exp(-eta)); W = mu * (1 - mu)
        g = X.T @ (y_true - mu); H = X.T @ (X * W[:, None])
        try:
            b += np.linalg.solve(H + 1e-8 * np.eye(2), g)
        except Exception:
            break
    return b[1], b[0]  # slope, intercept

MODELS = [('Digital Twin (ensemble)', 'DIGITAL TWIN'),
          ('TabPFN (PRAI)', 'PRAI'),
          ('CSDI component', 'p_csdi'),
          ('Bayesian ensemble', 'bayes_mean'),
          ('KAT component', 'p_kat_raw'),
          ('NVSA component', 'p_nvsa')]

print('--- B1. Threshold-independent metrics (primary) ---')
rows = []
for name, col in MODELS:
    p = pd.to_numeric(dt[col], errors='coerce').to_numpy(float)
    auc = roc_auc_score(y, p); lo, hi = auc_ci(y, p)
    sl, ic = cal_slope_intercept(y, p)
    rows.append(dict(Model=name, n=len(y), **{
        'ROC-AUC': round(auc, 3), 'AUC 95% CI': f'[{lo:.3f};{hi:.3f}]',
        'PR-AUC': round(average_precision_score(y, p), 3),
        'PR-AUC vs prev': f'{average_precision_score(y, p)/prev:.2f}x',
        'Brier': round(brier_score_loss(y, p), 4),
        'LogLoss': round(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), 4),
        'Cal.slope': round(sl, 2), 'Cal.intercept': round(ic, 2),
        'MeanPred': round(p.mean(), 3)}))
B1 = pd.DataFrame(rows)
print(B1.to_string(index=False))
print(f'  Baseline: PR-AUC of random = prevalence = {prev:.3f}; Brier of always-{prev:.2f} = {prev*(1-prev):.4f}')
print('  Cal.slope<1 = overfit spread; Cal.intercept + MeanPred>>prev = systematic over-prediction')

print('\n--- B2. Threshold metrics at Youden-optimal cut (probs are miscalibrated; 0.5 is uninformative) ---')
rows2 = []
for name, col in MODELS:
    p = pd.to_numeric(dt[col], errors='coerce').to_numpy(float)
    fpr, tpr, thr = roc_curve(y, p)
    yj = thr[np.argmax(tpr - fpr)]
    for tname, t in [('@0.5', 0.5), ('@Youden', yj)]:
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        rows2.append(dict(Model=name, Thr=tname, cut=round(t, 3),
                          MCC=round(matthews_corrcoef(y, pred), 3),
                          F1=round(f1_score(y, pred, zero_division=0), 3),
                          BalAcc=round(balanced_accuracy_score(y, pred), 3),
                          Sens=round(tp / (tp + fn), 3), Spec=round(tn / (tn + fp), 3),
                          TP=tp, FP=fp, FN=fn, TN=tn))
B2 = pd.DataFrame(rows2)
print(B2.to_string(index=False))

print('\n--- B3. DeLong paired AUC comparison vs TabPFN (primary contrasts) ---')
pt = pd.to_numeric(dt['PRAI'], errors='coerce').to_numpy(float)
rows3 = []
for name, col in [('Digital Twin (ensemble)', 'DIGITAL TWIN'), ('CSDI component', 'p_csdi'),
                  ('Bayesian ensemble', 'bayes_mean')]:
    p = pd.to_numeric(dt[col], errors='coerce').to_numpy(float)
    a1, a2, z, pv = delong_test(y, p, pt)
    rows3.append(dict(Contrast=f'{name} vs TabPFN', AUC_A=round(a1, 3), AUC_TabPFN=round(a2, 3),
                      dAUC=round(a1 - a2, 3), z=round(z, 2),
                      p=(f'{pv:.2e}' if pv < 1e-3 else f'{pv:.3f}'),
                      sig=('*' if pv < .05 else 'ns')))
B3 = pd.DataFrame(rows3)
print(B3.to_string(index=False))
print('  H0: equal AUC. All models weak (AUC ~0.57-0.63); differences small.')

print('\n--- B4. Prevalence by subcohort (fresh vs FET base-rate differs) ---')
print(dt.groupby('grp').Preg.agg(['size', 'sum', 'mean']).round(3).to_string())

B1.to_csv('bench_B1_threshold_free.csv', index=False)
B2.to_csv('bench_B2_threshold.csv', index=False)
B3.to_csv('bench_B3_delong.csv', index=False)
print('\n[Part B tables saved]')
