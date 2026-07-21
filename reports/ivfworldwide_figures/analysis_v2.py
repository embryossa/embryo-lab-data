#!/usr/bin/env python3
"""CSDI Digital Twin vs TabPFN-3: head-to-head, targets corrected from model_input.xlsx.

Correction applied: outcome zeros were dropped (coded NaN) in the comparison export.
They are recoverable from the master table:
  good_bl -> 'Число Bl хор.кач-ва' keeps zeros (53), bl_good drops them (verified 100% match)
  bl      -> bl_total NaN == 0 blastocysts (22 cycles); monotonicity good_bl<=bl<=embryos_d5 holds
  frozen  -> frozen NaN == 0 (215 cycles)
"""
import pandas as pd, numpy as np
from scipy import stats

rng = np.random.default_rng(20260719)
CSV = '/sessions/nice-festive-hawking/mnt/uploads/model_comparison_tabpfn3_vs_digitaltwin.csv'
XLS = '/sessions/nice-festive-hawking/mnt/embryo-lab-data/data/processed/fertimed/local_private/model_input.xlsx'

c = pd.read_csv(CSV, encoding='utf-8-sig')
m = pd.read_excel(XLS, sheet_name='model_input').set_index('cycle_uid')
m = m.loc[c.cycle_uid]  # align, PII columns never used below

df = c.copy()
df['bl_true']      = m['bl_total'].fillna(0).values
df['good_bl_true'] = m['Число Bl хор.кач-ва'].fillna(0).values
df['frozen_true']  = m['frozen'].fillna(0).values
df['mii_true']     = m['mii'].values
df['d5_true']      = m['embryos_d5'].fillna(0).values

assert (df.good_bl_true <= df.bl_true).all() and (df.bl_true <= df.d5_true).all()

def ccc(y, p):
    my, mp = y.mean(), p.mean()
    cov = ((y - my) * (p - mp)).mean()
    return 2 * cov / (y.var(ddof=0) + p.var(ddof=0) + (my - mp) ** 2)

def metrics(y, p):
    e = p - y
    return dict(n=len(y), MAE=np.abs(e).mean(), MdAE=np.median(np.abs(e)),
                RMSE=np.sqrt((e ** 2).mean()), ME=e.mean(),
                within1=np.mean(np.abs(e) <= 1) * 100,
                within2=np.mean(np.abs(e) <= 2) * 100,
                Spearman=stats.spearmanr(y, p).statistic,
                CCC=ccc(y, p),
                R2=1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum())

def boot_delta(y, pa, pb, B=10000, stat='MAE'):
    f = (lambda yy, pp: np.abs(pp - yy).mean()) if stat == 'MAE' else \
        (lambda yy, pp: np.sqrt(((pp - yy) ** 2).mean()))
    obs = f(y, pa) - f(y, pb)
    idx = rng.integers(0, len(y), size=(B, len(y)))
    d = np.array([f(y[i], pa[i]) - f(y[i], pb[i]) for i in idx])
    return obs, np.percentile(d, 2.5), np.percentile(d, 97.5)

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m_ = len(p)
    adj = np.empty(m_); run = 0
    for r, i in enumerate(o):
        run = max(run, (m_ - r) * p[i]); adj[i] = min(run, 1.0)
    return adj

impossible = df['mii_true'] > df['occ_actual']
df['_mii_ok'] = df.mii_true.notna() & ~impossible

STAGES = [
    dict(k='MII',     lab='MII (зрелые ооциты)',            y='mii_true',      dt='dt_mii_pred',
         tp='tabpfn3_mii_p50', pre='tabpfn3_mii', lo=None, hi=None, leak=True,  mask='_mii_ok'),
    dict(k='2PN',     lab='2PN (зиготы)',                   y='pn2_actual',    dt='dt_pn2_pred',
         tp='tabpfn3_pn2_p50', pre='tabpfn3_pn2', lo=None, hi=None, leak=False, mask=None),
    dict(k='BL',      lab='Бластоцисты',                    y='bl_true',       dt='dt_bl_pred',
         tp='tabpfn3_blasts_after_fert_p50', pre='tabpfn3_blasts_after_fert',
         lo='dt_bl_p025', hi='dt_bl_p975', leak=False, mask=None),
    dict(k='BL_e',    lab='Бластоцисты (вариант early)',    y='bl_true',       dt='dt_bl_pred',
         tp='tabpfn3_blasts_early_p50', pre='tabpfn3_blasts_early',
         lo='dt_bl_p025', hi='dt_bl_p975', leak=False, mask=None),
    dict(k='GOOD_BL', lab='Бластоцисты хорошего качества',  y='good_bl_true',  dt='dt_good_bl_pred',
         tp='tabpfn3_good_p50', pre='tabpfn3_good',
         lo='dt_good_bl_p025', hi='dt_good_bl_p975', leak=False, mask=None),
    dict(k='FROZEN',  lab='Витрифицировано',                y='frozen_true',   dt=None,
         tp='tabpfn3_frozen_p50', pre='tabpfn3_frozen', lo=None, hi=None, leak=False, mask=None),
]

def sel(s):
    return df[s['mask']] if s['mask'] else df[s['y']].notna()

print('=' * 84)
print('0. ПОПРАВКА НА ВОССТАНОВЛЕННЫЕ НУЛИ (источник: model_input.xlsx, join по cycle_uid 250/250)')
print('=' * 84)
for lab, old, new in [('Бластоцисты', 'bl_actual', 'bl_true'),
                      ('Хор. бластоцисты', 'good_bl_actual', 'good_bl_true'),
                      ('Витрифицировано', 'frozen_actual', 'frozen_true')]:
    o, n = df[old], df[new]
    print(f'{lab:<18} было n={o.notna().sum():3d} среднее {o.mean():.2f} | '
          f'стало n={len(n):3d} среднее {n.mean():.2f}, нулей {int((n==0).sum())}')
print(f'\nSpearman с 2PN: BL {stats.spearmanr(df.bl_true, df.pn2_actual).statistic:+.3f} '
      f'(было -0.022), GOOD_BL {stats.spearmanr(df.good_bl_true, df.pn2_actual).statistic:+.3f} (было -0.019)')
print('=> после восстановления нулей таргет ведёт себя как настоящее число бластоцист;')
print('   монотонность good_bl <= bl <= embryos_d5 выполняется во всех 250 циклах.')
print(f'\nИмплицитная частота бластуляции (Σbl/Σ2PN): факт '
      f'{df.bl_true.sum()/df.pn2_actual.sum():.3f} | '
      f'CSDI {df.dt_bl_pred.sum()/df.pn2_actual.sum():.3f} | '
      f'TabPFN-3 {df.tabpfn3_blasts_after_fert_p50.sum()/df.pn2_actual.sum():.3f}')

print()
print('=' * 84)
print('1. СВОДНАЯ ТАБЛИЦА ТОЧНОСТИ (CSDI point vs TabPFN-3 P50)')
print('=' * 84)
acc, tests = [], []
for s in STAGES:
    msk = sel(s)
    y = df.loc[msk, s['y']].to_numpy(float)
    tp = df.loc[msk, s['tp']].to_numpy(float)
    if s['dt']:
        dt = df.loc[msk, s['dt']].to_numpy(float)
        acc.append({**metrics(y, dt), 'Стадия': s['lab'],
                    'Модель': 'CSDI' + (' [утечка]' if s['leak'] else '')})
    acc.append({**metrics(y, tp), 'Стадия': s['lab'], 'Модель': 'TabPFN-3'})
    if not s['dt']:
        continue
    ae_dt, ae_tp = np.abs(dt - y), np.abs(tp - y)
    d = ae_dt - ae_tp
    w = stats.wilcoxon(ae_dt, ae_tp, zero_method='wilcox') if (d != 0).any() else None
    o1, l1, h1 = boot_delta(y, dt, tp, stat='MAE')
    o2, l2, h2 = boot_delta(y, dt, tp, stat='RMSE')
    tests.append(dict(Стадия=s['lab'], n=len(y), leak=s['leak'], dMAE=o1, lo=l1, hi=h1,
                      dRMSE=o2, rlo=l2, rhi=h2, ties=int((d == 0).sum()),
                      winT=(d > 0).mean() * 100, winC=(d < 0).mean() * 100,
                      p=(w.pvalue if w else np.nan)))

A = pd.DataFrame(acc)[['Стадия', 'Модель', 'n', 'MAE', 'MdAE', 'RMSE', 'ME',
                       'within1', 'within2', 'Spearman', 'CCC', 'R2']]
print(A.round(3).to_string(index=False))

print()
print('=' * 84)
print('2. ПАРНОЕ СРАВНЕНИЕ (Wilcoxon по |ошибке| + парный bootstrap 10 000, Holm на 4 стадии)')
print('=' * 84)
T = pd.DataFrame(tests)
pr = T[~T.leak].copy(); pr['p_holm'] = holm(pr.p.values)
T = T.merge(pr[['Стадия', 'p_holm']], on='Стадия', how='left')
T['Победитель'] = np.where(T.p_holm.fillna(T.p) >= .05, 'нет различий',
                           np.where(T.dMAE > 0, 'TabPFN-3', 'CSDI'))
T['ΔMAE'] = T.dMAE.round(3)
T['CI ΔMAE'] = T.apply(lambda r: f'[{r.lo:+.3f}; {r.hi:+.3f}]', axis=1)
T['ΔRMSE'] = T.dRMSE.round(3)
T['CI ΔRMSE'] = T.apply(lambda r: f'[{r.rlo:+.3f}; {r.rhi:+.3f}]', axis=1)
T['p'] = T.p.map(lambda v: f'{v:.2e}' if v < 1e-3 else f'{v:.4f}')
T['p Holm'] = T.p_holm.map(lambda v: '— (искл.)' if pd.isna(v) else (f'{v:.2e}' if v < 1e-3 else f'{v:.4f}'))
T['%TabPFN'] = T.winT.round(1); T['%CSDI'] = T.winC.round(1)
print(T[['Стадия', 'n', 'ΔMAE', 'CI ΔMAE', 'ΔRMSE', 'CI ΔRMSE', '%TabPFN', '%CSDI',
         'p', 'p Holm', 'Победитель']].to_string(index=False))
print('\n  ΔMAE = MAE(CSDI) − MAE(TabPFN-3);  >0 => TabPFN-3 точнее')

print()
print('=' * 84)
print('3. КАЛИБРОВКА ИНТЕРВАЛОВ')
print('=' * 84)
cal = []
for s in STAGES:
    msk = sel(s); y = df.loc[msk, s['y']].to_numpy(float)
    lo = df.loc[msk, s['pre'] + '_p10'].to_numpy(float)
    hi = df.loc[msk, s['pre'] + '_p90'].to_numpy(float)
    cov = np.mean((y >= lo) & (y <= hi))
    ci = stats.binomtest(int(round(cov * len(y))), len(y)).proportion_ci(.95)
    cal.append(dict(Стадия=s['lab'], Модель='TabPFN-3 P10–P90 (ном. 80%)', n=len(y),
                    Покрытие=cov * 100, CI=f'[{ci.low*100:.1f}; {ci.high*100:.1f}]',
                    Откл=cov * 100 - 80, Ширина=np.mean(hi - lo)))
    if s['lo']:
        l = df.loc[msk, s['lo']].to_numpy(float); h = df.loc[msk, s['hi']].to_numpy(float)
        cv = np.mean((y >= l) & (y <= h))
        c2 = stats.binomtest(int(round(cv * len(y))), len(y)).proportion_ci(.95)
        cal.append(dict(Стадия=s['lab'], Модель='CSDI P2.5–P97.5 (ном. 95%)', n=len(y),
                        Покрытие=cv * 100, CI=f'[{c2.low*100:.1f}; {c2.high*100:.1f}]',
                        Откл=cv * 100 - 95, Ширина=np.mean(h - l)))
C = pd.DataFrame(cal)
print(C.round(2).to_string(index=False))
print('\n  Откл = фактическое минус номинальное покрытие, пп. <0 = сверхуверенность.')

print()
print('=' * 84)
print('4. PINBALL LOSS (предиктивное распределение TabPFN-3)')
print('=' * 84)
pl = []
for s in STAGES:
    msk = sel(s); y = df.loc[msk, s['y']].to_numpy(float); tot = []
    for q, suf in [(.1, 'p10'), (.25, 'p25'), (.5, 'p50'), (.75, 'p75'), (.9, 'p90')]:
        qq = df.loc[msk, f"{s['pre']}_{suf}"].to_numpy(float)
        tot.append(np.mean(np.maximum(q * (y - qq), (q - 1) * (y - qq))))
    pl.append(dict(Стадия=s['lab'], n=len(y),
                   **{f'@{int(q*100)}': round(v, 3) for q, v in zip([.1, .25, .5, .75, .9], tot)},
                   Средний=round(np.mean(tot), 3)))
print(pd.DataFrame(pl).to_string(index=False))

print()
print('=' * 84)
print('5. ПОДГРУППЫ: ΔMAE (CSDI − TabPFN-3)')
print('=' * 84)
df['_age'] = pd.cut(df.age, [0, 34, 39, 100], labels=['<35', '35-39', '>=40'])
df['_occ'] = pd.cut(df.occ_actual, [0, 5, 10, 100], labels=['OCC 1-5', 'OCC 6-10', 'OCC >10'])
sg = []
for s in STAGES:
    if not s['dt'] or s['leak'] or s['k'] == 'BL_e':
        continue
    for var, lab in [('_age', 'Возраст'), ('_occ', 'Число ОКК'), ('ivf_icsi', 'Метод')]:
        for g, sub in df[sel(s)].groupby(var, observed=True):
            if len(sub) < 8:
                continue
            y = sub[s['y']].to_numpy(float)
            a, b = np.abs(sub[s['dt']].to_numpy(float) - y), np.abs(sub[s['tp']].to_numpy(float) - y)
            p = stats.wilcoxon(a, b).pvalue if (a != b).any() else np.nan
            sg.append(dict(Стадия=s['k'], Срез=lab, Группа=str(g), n=len(y),
                           MAE_CSDI=round(a.mean(), 2), MAE_TabPFN=round(b.mean(), 2),
                           ΔMAE=round(a.mean() - b.mean(), 2),
                           p=(f'{p:.2e}' if p < 1e-3 else f'{p:.4f}')))
S = pd.DataFrame(sg)
print(S.to_string(index=False))

print()
print('=' * 84)
print('6. ДИАГНОСТИКА СИСТЕМАТИЧЕСКОЙ ОШИБКИ CSDI НА БЛАСТОЦИСТАХ')
print('=' * 84)
for s in STAGES:
    if s['k'] not in ('BL', 'GOOD_BL'):
        continue
    msk = sel(s); y = df.loc[msk, s['y']].to_numpy(float)
    dt = df.loc[msk, s['dt']].to_numpy(float); tp = df.loc[msk, s['tp']].to_numpy(float)
    sl_dt = np.polyfit(y, dt, 1); sl_tp = np.polyfit(y, tp, 1)
    print(f"{s['lab']}:")
    print(f"  факт  среднее {y.mean():.2f}; CSDI {dt.mean():.2f} (x{dt.mean()/y.mean():.2f}); "
          f"TabPFN-3 {tp.mean():.2f} (x{tp.mean()/y.mean():.2f})")
    print(f"  регрессия pred~факт: CSDI наклон {sl_dt[0]:+.3f} сдвиг {sl_dt[1]:+.2f} | "
          f"TabPFN-3 наклон {sl_tp[0]:+.3f} сдвиг {sl_tp[1]:+.2f}")
    print(f"  доля циклов с переоценкой: CSDI {np.mean(dt>y)*100:.1f}%, TabPFN-3 {np.mean(tp>y)*100:.1f}%")

print()
print('=' * 84)
print('7. РАЗЛОЖЕНИЕ: КАЛИБРОВКА ШКАЛЫ vs ДИСКРИМИНАЦИЯ + БАЗОВАЯ ЛИНИЯ')
print('=' * 84)
print('LOO-рекалибровка = линейная подгонка pred->факт с исключением текущего наблюдения')
print('(честная оценка того, что осталось бы от модели после снятия систематического сдвига)\n')
rec_rows = []
for k, ycol, dtc, tpc in [('2PN', 'pn2_actual', 'dt_pn2_pred', 'tabpfn3_pn2_p50'),
                          ('BL', 'bl_true', 'dt_bl_pred', 'tabpfn3_blasts_after_fert_p50'),
                          ('GOOD_BL', 'good_bl_true', 'dt_good_bl_pred', 'tabpfn3_good_p50')]:
    y = df[ycol].to_numpy(float)
    def loo(pred):
        out = np.empty(len(y))
        for i in range(len(y)):
            msk = np.ones(len(y), bool); msk[i] = False
            out[i] = np.polyval(np.polyfit(pred[msk], y[msk], 1), pred[i])
        return np.clip(out, 0, None)
    dt, tp = df[dtc].to_numpy(float), df[tpc].to_numpy(float)
    rd, rt = loo(dt), loo(tp)
    rec_rows.append(dict(
        Стадия=k,
        MAE_const=round(np.abs(y - np.median(y)).mean(), 3),
        MAE_CSDI=round(np.abs(dt - y).mean(), 3),
        MAE_CSDI_recal=round(np.abs(rd - y).mean(), 3),
        MAE_TabPFN=round(np.abs(tp - y).mean(), 3),
        MAE_TabPFN_recal=round(np.abs(rt - y).mean(), 3),
        rho_CSDI=round(stats.spearmanr(y, dt).statistic, 3),
        rho_TabPFN=round(stats.spearmanr(y, tp).statistic, 3),
        Лучше_константы=('обе' if min(np.abs(rd - y).mean(), np.abs(rt - y).mean())
                          < np.abs(y - np.median(y)).mean() else 'НИ ОДНА')))
R = pd.DataFrame(rec_rows)
print(R.to_string(index=False))
print('\n  MAE_const = тривиальный прогноз "всегда медиана факта".')
R.to_csv('v2_table5_recalibration.csv', index=False)

A.to_csv('v2_table1_accuracy.csv', index=False)
T.to_csv('v2_table2_tests.csv', index=False)
C.to_csv('v2_table3_calibration.csv', index=False)
S.to_csv('v2_table4_subgroups.csv', index=False)
print('\n[tables saved: v2_table1..4]')
