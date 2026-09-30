#!/usr/bin/env python3
"""Score the two KAT (DT layer L3) components separately on the frozen RBMO cohort.

KAT = softmax-weighted KAN + FT-Transformer, then isotonic recalibration (`p_kat_raw`).
The Digital Twin only stores the calibrated ensemble, so each component is re-run here
with the same weights (models/ in IVF Digital Twin, 2026-06-09, before the cohort runs)
and the same 18-feature builder (`build_nn_features`) as DT 7.0:

  prospective   — as DT ran: OCC and MII known, 2PN/Bl/Good Bl Monte-Carlo simulated;
                  component = mean over iterations. Reproduces `p_kat_raw` (check column).
  retrospective — observed 2PN/Bl/Good Bl fed as point values (one deterministic row).

Must be run with the Digital Twin .venv (torch + mambular). Writes predictions only;
metrics are in evaluate_components.py.
"""
import hashlib, json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch

warnings.filterwarnings('ignore')
OUT = Path(__file__).resolve().parent
REF = OUT.parents[2] / 'data' / 'processed' / 'reference' / 'rbmo_prospective_2026'
sys.path.insert(0, str(OUT.parents[1]))
from _local_paths import local_path  # noqa: E402
DT = local_path('dt_project')
sys.path[:0] = [str(DT / 'src'), str(DT)]
import ivf_digital_twin as dt  # noqa: E402

N_SIM, SEED = 2000, 42   # batch_analysis.py: N_SIM=2000, seed 42 + row

path = REF / 'rbmo_prospective_cohort.csv'
assert hashlib.sha256(path.read_bytes()).hexdigest() == json.loads((REF / 'manifest.json').read_text('utf-8'))['cohort_sha256']
d = pd.read_csv(path, encoding='utf-8-sig')

wrap = dt.load_nn_ensemble()
ens = wrap.ensemble_model
w = torch.softmax(ens.raw_weights, 0).detach().numpy()
print(f'weights KAN={w[0]:.3f} FT={w[1]:.3f}; calibrator={wrap.calibrator is not None}')


def components(X: pd.DataFrame):
    x = torch.tensor(X.values.astype(np.float32))
    with torch.no_grad():
        kan = torch.sigmoid(ens.kan_model(x).squeeze(-1)).numpy().astype(float)
    ft = dt._ft_proba(ens.ft_model, X.astype(np.float32)).astype(float)
    raw = w[0] * kan + w[1] * ft
    iso = np.clip(wrap.calibrator.predict(raw), .001, .999) if wrap.calibrator is not None else raw
    return kan, ft, raw, np.clip(iso, .001, .999)


def i(v):
    return None if pd.isna(v) else int(v)


rows = []
for k, r in d.iterrows():
    patient = dt.PatientInput(female_age=float(r.age), amh=float(r.amh), afc=int(r.afc), bmi=float(r.bmi))
    foll = i(r.follicles_tvp)
    rec = dict(ref_id=r.ref_id, centre=r.centre, Preg=int(r.Preg), p_kat_raw=r.p_kat_raw)

    # prospective: OCC + MII (=inseminated, as batch) known, downstream simulated
    np.random.seed(SEED + k)
    res = dt.run_pipeline(patient, known=dt.KnownValues(okk=i(r.known_okk), mii=i(r.known_mii)), n=N_SIM)
    X = dt.build_nn_features(patient, res, int(r.attempt_number), foll)
    for name, v in zip(('kan', 'ft', 'ens_raw', 'ens_iso'), components(X)):
        rec[f'pro_{name}'] = v.mean()

    # retrospective: observed counts, only when 2PN, Bl and Good Bl are all recorded
    obs = [r['2pN'], r['Bl'], r['Good Bl']]
    rec['retro_complete'] = not any(pd.isna(obs))
    if rec['retro_complete']:
        one = lambda v: np.array([float(v)])
        res1 = dict(sim_okk=one(r.known_okk), sim_mii=one(r.known_mii if pd.notna(r.known_mii) else r.known_okk),
                    sim_pn2=one(obs[0]), sim_blasts=one(obs[1]), sim_good=one(obs[2]))
        X1 = dt.build_nn_features(patient, res1, int(r.attempt_number), foll)
        for name, v in zip(('kan', 'ft', 'ens_raw', 'ens_iso'), components(X1)):
            rec[f'retro_{name}'] = float(v[0])
    rows.append(rec)
    if k % 100 == 0:
        print(k, rec['pro_ens_iso'], rec['p_kat_raw'], flush=True)

p = pd.DataFrame(rows)
p.to_csv(OUT / 'kat_component_predictions.csv', index=False)
print('check prospective ens_iso vs stored p_kat_raw: r=%.4f, MAE=%.4f' % (
    np.corrcoef(p.pro_ens_iso, p.p_kat_raw)[0, 1], (p.pro_ens_iso - p.p_kat_raw).abs().mean()))
