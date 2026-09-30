#!/usr/bin/env python3
"""Full Digital Twin re-run on the manuscripts' 444-cycle "Cohort A", every layer saved.

Source: PycharmProjects/pythonProject/DNN for Fertimed/pythonProject/validation.xlsx
(Fertimed 1C export, 889 rows); Cohort A = the 444 rows with a recorded transfer outcome
(185 pregnancies, counting the one mixed "одноплодная; не наступила" record as positive).

Configuration = the DT 7.0 batch (batch_analysis.py) used for the 60-cycle / Two-Line and the
RBMO prospective runs: predict_single_patient + compute_l7_posterior, N_SIM=2000, seed 42+row,
AMH/AFC batch defaults 2.0/12, BMI 23, inputs age, attempt, follicles, OCC and inseminated
(as known MII); 2PN, blastocysts and good blastocysts masked (simulated by Layer 1) and fed to
Layers 3-7 from the simulation.

Saved per cycle:
  L1  stage medians of the full run; plus Layer-1-only forecasts made BEFORE each stage is
      observed (stage_*): OCC with nothing known; MII given OCC; 2PN given OCC + inseminated;
      blastocysts given OCC + inseminated + 2PN; good blastocysts given ... + blastocysts.
  L2  per-transfer, cumulative-if-viable, whole-cycle, P(viable) with the FORTUNE intercept in
      the code (0.40) and with the documented recalibrated intercept (-0.25), the latter
      recomputed exactly from the same simulated per-iteration FORTUNE and KPI draws.
  L3  KAT (p_kat_raw) and its interval; NVSA.   L4 cluster probabilities.   L5 CSDI.
  L6  graph (raw, ensemble, weight).            Bayesian posterior mean.
  L7  BEFE posterior, interval, prior/evidence shares, reliability, n_eff - with the L2 source
      at 0.40 (as run) and rebuilt with the -0.25 L2 source.
No identifiers are written (row = position among the 444). Metrics: evaluate.py.
Run with the Digital Twin .venv:  run_dt.py [LIMIT]  |  run_dt.py --shard K N
"""
import copy, os, re, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent))
from _local_paths import local_path  # noqa: E402
SRC = local_path('cohort_a_validation')
DT = local_path('dt_project')
sys.path[:0] = [str(DT / 'src'), str(DT)]
os.chdir(DT)
import ivf_digital_twin as dtm                          # noqa: E402
from ivf_core import predict_single_patient, _BASE_DIR  # noqa: E402
from befe_batch_utils import compute_l7_posterior      # noqa: E402

N_SIM, SEED = 2000, 42
AMH, AFC, BMI = 2.0, 12, 23.0
INTERCEPT_CODE, INTERCEPT_DOC = 0.40, -0.25
SHARD = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 and sys.argv[1] == '--shard' else None
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 and SHARD is None else None

# guard: the code intercept is still 0.40, otherwise the "before" arm is not what we think
_p = dtm.PatientInput(female_age=36.3, amh=2.0, afc=12, bmi=24.0)   # every z-term = 0
assert abs(dtm.fortune_per_transfer_logit(_p) - INTERCEPT_CODE) < 1e-9


def num(v):
    x = pd.to_numeric(str(v).replace(',', '.') if isinstance(v, str) else v, errors='coerce')
    return None if pd.isna(x) else float(x)


def pos_int(v):
    x = num(v)
    return int(round(x)) if x is not None and x > 0 else None


def obs_int(v):
    """Observed count, 0 kept (0 is an observation, not a missing value)."""
    x = num(v)
    return int(round(x)) if x is not None and x >= 0 else None


def age_of(v):
    m = re.search(r'(\d+)', str(v))
    return float(m.group(1)) if m and 10 < float(m.group(1)) < 60 else None


def lg(p):
    return np.log(p / (1 - p))


def l2_recalibrated(res, delta):
    """Stage-7 summaries with the FORTUNE logit shifted by delta, same simulated draws."""
    pf = np.clip(dtm.sigmoid(lg(np.asarray(res['sim_p_fortune'])) + delta), .01, .99)
    pk = np.asarray(res['sim_p_kpi'])
    w = float(res.get('kpi_weight', dtm.KPI_WEIGHT))
    p = np.clip(dtm.sigmoid((1 - w) * lg(pf) + w * lg(pk)), .01, .99)
    n_tx = np.asarray(res['sim_n_tx'])
    viable = n_tx >= 1
    per = float(p.mean())
    if viable.sum() >= 30:
        cum = float(np.mean(1 - (1 - p[viable]) ** n_tx[viable]))
    else:
        cum = float(1 - (1 - per) ** max(int(np.median(n_tx[viable])) if viable.any() else 1, 1))
    cum = max(cum, per)
    return p, dict(p_per_transfer=per, p_cum_if_viable=cum, p_overall_cycle=float(viable.mean()) * cum,
                   p_viable=float(viable.mean()))


def befe_fields(befe, prefix):
    keys = ['posterior', 'ci_low', 'ci_high', 'p_prior', 'p_predictive', 'prior_pull', 'evidence_pull',
            'reliability', 'n_eff', 'diffusion_agreement', 'ood_clinical', 'consensus', 'cluster_label']
    return {f'{prefix}{k}': getattr(befe, k, None) for k in keys} if befe is not None else {}


def l1_only(age, attempt, foll, **known):
    np.random.seed(SEED)
    r = dtm.run_pipeline_extended(dtm.PatientInput(female_age=age, amh=AMH, afc=AFC, bmi=BMI),
                                  known=dtm.KnownValues(**known), attempt_number=attempt,
                                  follicles=foll, nn_model=None, n=N_SIM)
    return r


raw = pd.read_excel(SRC)
coh = raw[raw['Исход переноса'].notna()].reset_index(drop=True)
assert len(coh) == 444, len(coh)
out = coh['Исход переноса'].astype(str)
coh['preg'] = out.str.contains(r'Беременность \+', regex=True).astype(int)
coh['preg_mixed'] = (out.str.contains(r'Беременность \+') & out.str.contains('не наступила')).astype(int)
assert coh.preg.sum() == 185, coh.preg.sum()

rows, t0 = [], time.time()
for i, r in coh.iterrows():
    if LIMIT is not None and i >= LIMIT:
        break
    if SHARD is not None and i % SHARD[1] != SHARD[0]:
        continue
    age, attempt, foll = age_of(r['Возраст']), pos_int(r['№ попытки']) or 1, pos_int(r['Количество фолликулов'])
    occ, insem = pos_int(r['Число ОКК']), pos_int(r['Число инсеминированных'])
    rec = dict(row=i, preg=int(r.preg), preg_mixed=int(r.preg_mixed), treatment=r['Лечение'],
               age=age, attempt=attempt, follicles=foll, amh_raw=num(r['АМГ']), afc_raw=num(r['КАФ']),
               known_okk=occ, known_mii=insem,
               real_okk=obs_int(r['Число ОКК']), real_mii=obs_int(r['Число MII']),
               real_insem=obs_int(r['Число инсеминированных']), real_pn2=obs_int(r['2 pN']),
               real_bl=obs_int(r['Число Bl']), real_good=obs_int(r['Число Bl хор.кач-ва']),
               n_transferred=num(r['Перенесено эмбрионов']))
    if age is None:
        rows.append({**rec, 'status': 'skipped_no_age'})
        continue

    # ── full DT, batch configuration ───────────────────────────────────────
    full = predict_single_patient(age=age, amh=AMH, afc=AFC, bmi=BMI, attempt=attempt, follicles=foll,
                                  known_okk=occ, known_mii=insem, n_sim=N_SIM, seed=SEED + i)
    res = full['res']
    ca = res.get('cluster_analysis', {}) or {}
    probs = ca.get('cluster_probs', {}) or {}
    nn = res.get('nn_prediction', {}) or {}
    gnn = full.get('gnn_result') or {}
    p_l7, befe, _ = compute_l7_posterior(full, age=age, amh=AMH, afc=AFC, bmi=BMI, base_dir=_BASE_DIR)

    sim_p_recal, l2r = l2_recalibrated(res, INTERCEPT_DOC - INTERCEPT_CODE)
    full_r = dict(full)
    full_r['res'] = res_r = copy.copy(res)
    res_r.update(sim_p_combined=sim_p_recal, **l2r)
    p_l7r, befe_r, _ = compute_l7_posterior(full_r, age=age, amh=AMH, afc=AFC, bmi=BMI, base_dir=_BASE_DIR)
    # sanity: recomputing with delta = 0 must reproduce the as-run L2
    _, l2_0 = l2_recalibrated(res, 0.0)

    rec.update(status='ok',
               med_okk=res['okk_med'], med_mii=res['mii_med'], med_pn2=res['pn2_med'],
               med_bl=res['blasts_med'], med_good=res['good_med'],
               l2_p_per_transfer=res.get('p_per_transfer'), l2_p_cum_if_viable=res.get('p_cum_if_viable'),
               l2_p_overall_cycle=res.get('p_overall_cycle'), l2_p_viable=res.get('p_viable'),
               l2_p_per_transfer_fortune=res.get('p_per_transfer_fortune'),
               l2_p_per_transfer_kpi=res.get('p_per_transfer_kpi'),
               l2_recheck_per_transfer=l2_0['p_per_transfer'],
               l2r_p_per_transfer=l2r['p_per_transfer'], l2r_p_cum_if_viable=l2r['p_cum_if_viable'],
               l2r_p_overall_cycle=l2r['p_overall_cycle'],
               bayes_mean=(res.get('posterior') or {}).get('mean'),
               l3_p_kat_raw=full.get('p_kat_raw'), l3_kat_ci_low=(nn.get('base_prob_ci') or (None, None))[0],
               l3_kat_ci_high=(nn.get('base_prob_ci') or (None, None))[1], l3_p_nvsa=full.get('p_nvsa'),
               l4_dominant_cluster=ca.get('dominant_cluster'), l4_c0=probs.get(0), l4_c1=probs.get(1),
               l4_c2=probs.get(2),
               l5_p_csdi=full.get('p_csdi'), l6_p_gnn_raw=full.get('p_gnn_raw'),
               l6_p_gnn_ens=full.get('p_gnn_ens'), l6_w_gnn=gnn.get('w_gnn') if isinstance(gnn, dict) else None,
               l7_value=p_l7, l7r_value=p_l7r,
               nn_available=int(full['nn_available']), csdi_available=int(full['csdi_available']),
               gnn_available=int(full['gnn_available']),
               **befe_fields(befe, 'l7_'), **befe_fields(befe_r, 'l7r_'))

    # ── Layer-1 forecasts made before each stage is observed ──────────────
    s = l1_only(age, attempt, foll)
    rec['stage_okk'] = s['okk_med']
    if occ is not None:
        rec['stage_mii'] = l1_only(age, attempt, foll, okk=occ)['mii_med']
    rec['stage_pn2'] = res['pn2_med'] if (occ is not None or insem is not None) else None
    pn2 = obs_int(r['2 pN'])
    if pn2 is not None and insem is not None:
        rec['stage_bl'] = l1_only(age, attempt, foll, okk=occ, mii=insem, pn2=pn2)['blasts_med']
        bl = obs_int(r['Число Bl'])
        if bl is not None:
            rec['stage_good'] = l1_only(age, attempt, foll, okk=occ, mii=insem, pn2=pn2, blasts=bl)['good_med']
    rows.append(rec)
    if len(rows) % 10 == 0:
        print(f'{len(rows)} rows  {time.time() - t0:.0f}s', flush=True)

name = (f'_smoke_{LIMIT}.csv' if LIMIT is not None else
        f'shards/dt_full_444_shard{SHARD[0]}of{SHARD[1]}.csv' if SHARD else 'dt_full_444.csv')
(OUT / 'shards').mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(OUT / name, index=False)
print('written', OUT / name, f'{time.time() - t0:.0f}s')
