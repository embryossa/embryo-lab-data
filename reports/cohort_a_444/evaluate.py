#!/usr/bin/env python3
"""Metrics for the full DT re-run on the 444-cycle Cohort A (run_dt.py) vs the manuscripts.

Stage agreement: Spearman ρ between the Layer-1 median forecast made before the stage was
observed (stage_*) and the observed count (0 kept); secondary: medians of the full batch run
(med_*, where OCC and inseminated are inputs). 95% CI: 2,000 bootstrap resamples.
Pregnancy: AUC with 95% CI from 2,000 outcome-stratified bootstrap resamples, Brier, mean
prediction vs observed rate (calibration gap = predicted − observed, pp), calibration-in-the-large
and slope. "Before/after recalibration" = Layer 2 FORTUNE intercept 0.40 (code) vs −0.25
(documented), carried through BEFE. For reference also an intercept shift fitted on these same
444 outcomes (in-sample: gap 0 by construction) and 10-fold cross-fitted.
Paired AUC differences: same resamples; two-sided p = min(1, 2·min(P(Δ*≤0), P(Δ*≥0))).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from scipy.special import logit, expit
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

OUT = Path(__file__).resolve().parent
B = 2000
CLAIM_RHO = {'okk': .718, 'mii': .865, 'pn2': .852, 'bl': .761, 'good': .876}
MODELS = [('l7_posterior', 'L7 BEFE combined estimate, intercept 0.40 (as run)'),
          ('l7r_posterior', 'L7 BEFE combined estimate, intercept −0.25'),
          ('l3_p_kat_raw', 'L3 KAT'),
          ('l2_p_per_transfer', 'L2 per-transfer, intercept 0.40'),
          ('l2r_p_per_transfer', 'L2 per-transfer, intercept −0.25'),
          ('l2_p_overall_cycle', 'L1/2 whole-cycle, intercept 0.40'),
          ('l2r_p_overall_cycle', 'L1/2 whole-cycle, intercept −0.25'),
          ('bayes_mean', 'Bayesian posterior mean'),
          ('l5_p_csdi', 'L5 CSDI diffusion'),
          ('l6_p_gnn_ens', 'L6 graph (ensemble)')]
PAIRS = [('l7_posterior', 'l3_p_kat_raw'), ('l7_posterior', 'l2_p_per_transfer'),
         ('l3_p_kat_raw', 'l2_p_per_transfer'), ('l7r_posterior', 'l7_posterior')]


def load():
    shards = sorted((OUT / 'shards').glob('dt_full_444_shard*of*.csv'))
    d = pd.concat([pd.read_csv(p) for p in shards], ignore_index=True).sort_values('row')
    d.to_csv(OUT / 'dt_full_444.csv', index=False)
    return d


def strat_idx(y, rng):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    return np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def boot_aucs(y, P):
    """AUC of every column of P on the same 2,000 stratified resamples."""
    rng = np.random.default_rng(0)
    out = np.empty((B, P.shape[1]))
    for b in range(B):
        i = strat_idx(y, rng)
        out[b] = [roc_auc_score(y[i], P[i, k]) for k in range(P.shape[1])]
    return out


def shift_fit(y, p):
    lp = logit(np.clip(p, 1e-4, 1 - 1e-4))
    return brentq(lambda d: expit(lp + d).mean() - y.mean(), -10, 10), lp


def calib(y, p):
    lp = logit(np.clip(p, 1e-4, 1 - 1e-4))
    # calibration-in-the-large = MLE intercept with offset logit(p): score equation Σ(y − expit(lp + a)) = 0
    d_in, _ = shift_fit(y, p)
    citl = d_in
    slope = LogisticRegression(penalty=None, max_iter=1000).fit(lp.reshape(-1, 1), y).coef_[0, 0]
    p_in = expit(lp + d_in)
    p_cf = np.empty_like(p)
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=0).split(p, y):
        p_cf[te] = expit(lp[te] + shift_fit(y[tr], p[tr])[0])
    return dict(cal_intercept=citl, cal_slope=slope,
                brier_insample_shift=np.mean((p_in - y) ** 2), gap_pp_insample_shift=100 * (p_in.mean() - y.mean()),
                brier_crossfit_shift=np.mean((p_cf - y) ** 2), gap_pp_crossfit_shift=100 * (p_cf.mean() - y.mean()))


def pregnancy(d, subset):
    cols = [c for c, _ in MODELS]
    s = d.dropna(subset=cols)
    y, P = s.preg.to_numpy(int), s[cols].to_numpy(float)
    bs = boot_aucs(y, P)
    rows = []
    for k, (c, label) in enumerate(MODELS):
        p = P[:, k]
        rows.append(dict(subset=subset, model=label, column=c, n=len(y), events=int(y.sum()),
                         auc=roc_auc_score(y, p), auc_lo=np.percentile(bs[:, k], 2.5),
                         auc_hi=np.percentile(bs[:, k], 97.5), brier=np.mean((p - y) ** 2),
                         brier_prevalence_only=y.mean() * (1 - y.mean()), observed=y.mean(),
                         mean_pred=p.mean(), gap_pp=100 * (p.mean() - y.mean()), **calib(y, p)))
    pairs = []
    for a, b in PAIRS:
        ka, kb = cols.index(a), cols.index(b)
        dlt = bs[:, ka] - bs[:, kb]
        pairs.append(dict(subset=subset, a=a, b=b, n=len(y),
                          delta_auc=roc_auc_score(y, P[:, ka]) - roc_auc_score(y, P[:, kb]),
                          lo=np.percentile(dlt, 2.5), hi=np.percentile(dlt, 97.5),
                          p_two_sided=min(1.0, 2 * min((dlt <= 0).mean(), (dlt >= 0).mean())),
                          spearman_predictions=stats.spearmanr(P[:, ka], P[:, kb]).correlation))
    return rows, pairs


def rho(d, pred_prefix, label):
    rng = np.random.default_rng(1)
    rows = []
    for st in ('okk', 'mii', 'pn2', 'bl', 'good'):
        m = d[f'{pred_prefix}{st}'].notna() & d[f'real_{st}'].notna()
        x, z = d.loc[m, f'real_{st}'].to_numpy(), d.loc[m, f'{pred_prefix}{st}'].to_numpy()
        bs = []
        for _ in range(B):
            i = rng.integers(0, len(x), len(x))
            if np.ptp(x[i]) > 0 and np.ptp(z[i]) > 0:
                bs.append(stats.spearmanr(x[i], z[i]).correlation)
        rows.append(dict(forecast=label, stage=st, n=int(m.sum()), rho=stats.spearmanr(x, z).correlation,
                         lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), manuscript=CLAIM_RHO[st]))
    return rows


def characteristics(c):
    q = lambda s: f"{s.median():g} ({s.quantile(.25):g}–{s.quantile(.75):g}); n={s.notna().sum()}"
    ms = lambda s: f"{s.mean():.1f} ± {s.std():.1f}; n={s.notna().sum()}"
    fr = 100 * c.real_pn2 / c.real_insem.where(c.real_insem > 0)
    blr = 100 * c.real_bl / c.real_pn2.where(c.real_pn2 > 0)
    rows = [('Female age, years', ms(c.age), '36.1 ± 5.4'),
            ('Body mass index', 'not in the export (DT used 23.0)', '24.6 ± 4.7'),
            ('AMH, ng/mL', q(c.amh_raw.where(c.amh_raw.between(.01, 30))) + ' (DT used 2.0, batch default)', '2.2 (1.0–4.3)'),
            ('AFC', q(c.afc_raw.where(c.afc_raw.between(1, 60))) + f"; >60 in {int((c.afc_raw > 60).sum())} (DT used 12)", '12 (7–18)'),
            ('Retrieved oocytes', q(c.real_okk), '9 (5–15)'),
            ('Mature oocytes', q(c.real_mii), '7 (4–11)'),
            ('Fertilisation rate, % (2PN/inseminated)', ms(fr), '67.8 ± 18.0'),
            ('Blastocyst rate, % (Bl/2PN)', ms(blr), '42.3 ± 19.1'),
            ('Good-quality blastocysts', q(c.real_good), '2 (0–4)'),
            ('Clinical pregnancy', f"{int(c.preg.sum())}/{len(c)} ({100 * c.preg.mean():.1f}%)", '185/444 (41.7%)'),
            ('Treatment (1C «Лечение»)', '; '.join(f'{k} {v}' for k, v in c.treatment.value_counts().items()), 'ICSI; three centres')]
    return pd.DataFrame(rows, columns=['variable', 'validation.xlsx (re-derived)', 'manuscript Table 3'])


def main():
    d = load()
    assert d.row.is_unique
    ok = d[d.status == 'ok']
    ch = characteristics(d)
    ch.to_csv(OUT / 'cohort_characteristics.csv', index=False)
    rh = pd.DataFrame(rho(ok, 'stage_', 'L1, before the stage was observed') +
                      rho(ok, 'med_', 'full batch run (OCC, inseminated as inputs)'))
    rh.to_csv(OUT / 'stage_spearman.csv', index=False)
    preg, pairs = [], []
    for name, s in {'all cycles': ok, 'mixed outcome record excluded': ok[ok.preg_mixed == 0],
                    'fresh cycles (not «Криоперенос»)': ok[ok.treatment != 'Криоперенос']}.items():
        r, p = pregnancy(s, name)
        preg += r
        pairs += p
    preg, pairs = pd.DataFrame(preg), pd.DataFrame(pairs)
    preg.to_csv(OUT / 'pregnancy_metrics.csv', index=False)
    pairs.to_csv(OUT / 'paired_auc.csv', index=False)

    pd.set_option('display.width', 260, 'display.max_columns', 40, 'display.max_colwidth', 60)
    print('status:', d.status.value_counts().to_dict())
    print(ch.to_string(index=False), '\n')
    print(rh.round(3).to_string(index=False), '\n')
    print(preg[preg.subset == 'all cycles'][['model', 'n', 'events', 'auc', 'auc_lo', 'auc_hi', 'brier', 'mean_pred',
                                             'observed', 'gap_pp', 'cal_intercept', 'cal_slope',
                                             'brier_insample_shift', 'brier_crossfit_shift',
                                             'gap_pp_crossfit_shift']].round(3).to_string(index=False), '\n')
    print(preg[preg.subset != 'all cycles'][['subset', 'model', 'n', 'auc', 'auc_lo', 'auc_hi', 'brier', 'gap_pp']]
          .round(3).to_string(index=False), '\n')
    print(pairs.round(3).to_string(index=False))


if __name__ == '__main__':
    sys.exit(main())
