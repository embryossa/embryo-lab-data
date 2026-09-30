#!/usr/bin/env python3
"""Freeze the RBMO prospective cohort (1,026 cycles, four centres) as a standalone reference.

The cohort was never stored as one file: two centres live in this repo's runs, two only in
frozen Digital Twin exports whose TabPFN predictions sit in the TabPFN log. Later runs of the
same clinics replace the repo copies, so this script assembles the cohort once, checks it
against the published numbers and writes:

  data/processed/reference/rbmo_prospective_2026/
      rbmo_prospective_cohort.csv   one row per cycle, no identifiers (ref_id only)
      manifest.json                 sources + sha256, per-centre n, verification, column notes
      README.md
  data/processed/local_private/reference/rbmo_prospective_2026_crosswalk.csv
      ref_id -> patient_id / name / _key / record_id / notes, for local re-linking only

The frozen files are made read-only. Re-running refuses to overwrite them unless --force.
"""
import argparse, hashlib, json, os, stat
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
P = REPO / 'data' / 'processed'
AI = REPO.parent                        # sibling projects of embryo-lab-data
DT_RESULTS = AI / 'IVF Digital Twin Pro' / 'IVF Digital Twin' / 'results'
TABPFN_LOG = AI / 'TABPFN' / 'tabpfn_logs' / 'tabpfn_predictions.csv'
OUT = P / 'reference' / 'rbmo_prospective_2026'
PRIV = P / 'local_private' / 'reference'
COHORT = OUT / 'rbmo_prospective_cohort.csv'
CROSSWALK = PRIV / 'rbmo_prospective_2026_crosswalk.csv'

# centre -> (RBMO label, ref_id prefix, source)
CENTRES = {
    'Fertimed': ('Centre A', 'A', ('export', DT_RESULTS / 'dt_analytics_with_outcome_20260804_1910.csv', '2026-08-04')),
    'GGRC': ('Centre B', 'B', ('repo', P / 'ggrc' / 'dt_predictions.csv')),
    'SkyFert': ('Centre C', 'C', ('repo', P / 'skyfert' / 'dt_predictions.csv')),
    'Two-Line': ('4th centre', 'D', ('export', DT_RESULTS / 'dt_analytics_with_outcome_20260812_1829.csv', '2026-08-12')),
}
EXPECTED_N = {'Fertimed': 423, 'GGRC': 412, 'SkyFert': 163, 'Two-Line': 28}
# published values the reconstruction must reproduce (RBMO revision, Tables 4 and 6, abstract)
PUBLISHED = {'dt_auc_centre': {'Fertimed': .532, 'GGRC': .617, 'SkyFert': .505},
             'dt_auc_within_weighted': .561, 'dt_auc_pooled': .640, 'tabpfn_auc_within_weighted': .552}
TOL = .0015

PRIVATE = ['record_id', 'patient_name', 'patient_id', '_key', 'notes']   # -> crosswalk only
SCHEMA = [  # the 92 columns shared by all four sources, minus PRIVATE
    'timestamp', 'clinic_name', 'age', 'amh', 'afc', 'bmi', 'attempt_number', 'sperm_source',
    'follicles_tvp', 'known_okk', 'known_mii', 'known_pn2', 'known_blasts', 'known_good',
    'known_euploid', 'med_okk', 'med_mii', 'med_pn2', 'med_blasts', 'med_good', 'med_euploid',
    'med_warmed', 'p025_okk', 'p975_okk', 'p025_blasts', 'p975_blasts', 'p025_good', 'p975_good',
    'p_per_transfer', 'p_cum_if_viable', 'p_overall_cycle', 'p_viable', 'p_cancel_risk',
    'rate_ci_low', 'rate_ci_high', 'bayes_mean', 'bayes_ci_low', 'bayes_ci_high',
    'bayes_prior_mean', 'bayes_prior_type', 'p_kat_raw', 'p_nvsa', 'ci_kat_low', 'ci_kat_high',
    'ci_nvsa_low', 'ci_nvsa_high', 'p_csdi', 'csdi_ci_low', 'csdi_ci_high', 'p_gnn_raw',
    'p_gnn_ens', 'w_gnn', 'dominant_cluster', 'cluster_c0_prob', 'cluster_c1_prob',
    'cluster_c2_prob', 'ohss_moderate', 'ohss_severe', 'ohss_any', 'p_no_blast',
    'p_no_good_blast', 'banking_p_per_mii', 'banking_expected_euploid', 'nn_source',
    'csdi_available', 'gnn_available', 'real_pn2', 'real_cleav', 'real_bl', 'real_goodbl',
    'real_cryo', 'real_outcome', 'outcome_date', 'PRAI', 'Preg', 'DIGITAL TWIN', 'Age', 'Attempt',
    'N folicules OPU', 'OCC', 'MII', 'Inseminated', 'med_bl', 'med_goodbl', '2pN', 'Bl', 'Good Bl']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def assemble():
    tlog = pd.read_csv(TABPFN_LOG, encoding='utf-8-sig', low_memory=False)
    parts, sources = [], {}
    for name, (label, prefix, src) in CENTRES.items():
        if src[0] == 'repo':
            d = pd.read_csv(src[1], low_memory=False)
            d['tabpfn_ci_low'], d['tabpfn_ci_high'] = d['ci_low'], d['ci_high']
            how = 'PRAI, ci_low, ci_high of the same file'
        else:
            d = pd.read_csv(src[1], encoding='utf-8-sig', low_memory=False)
            day = (tlog[tlog.timestamp.str[:10] == src[2]]
                   .drop_duplicates('patient_id', keep='last')
                   [['patient_id', 'p_tabpfn', 'ci_low', 'ci_high']]
                   .rename(columns={'ci_low': 'tabpfn_ci_low', 'ci_high': 'tabpfn_ci_high'}))
            d = d.merge(day, on='patient_id', how='left', validate='one_to_one')
            d['PRAI'] = d['p_tabpfn']      # export left PRAI empty; fill from the log
            how = f'TabPFN log rows dated {src[2]}, joined on patient_id'
        missing = [c for c in SCHEMA + PRIVATE if c not in d.columns]
        assert not missing, f'{name}: missing columns {missing}'
        d = d.reset_index(drop=True)
        d.insert(0, 'ref_id', [f'{prefix}-{i + 1:04d}' for i in range(len(d))])
        d.insert(1, 'centre', name)
        d.insert(2, 'rbmo_label', label)
        parts.append(d)
        sources[name] = {'rbmo_label': label, 'n': len(d), 'dt_file': str(src[1].relative_to(AI)),
                         'dt_sha256': sha256(src[1]), 'tabpfn_source': how}
    sources['_tabpfn_log'] = {'file': str(TABPFN_LOG.relative_to(AI)), 'sha256': sha256(TABPFN_LOG),
                              'rows_at_freeze': len(tlog)}
    return pd.concat(parts, ignore_index=True), sources


def verify(df):
    for c in ['DIGITAL TWIN', 'PRAI', 'Preg']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    bad = df[['DIGITAL TWIN', 'PRAI', 'Preg']].isna().any(axis=1)
    assert not bad.any(), f'{bad.sum()} cycles without both probabilities and outcome'
    n = df.centre.value_counts().to_dict()
    assert n == EXPECTED_N, f'cohort differs from RBMO: {n}'
    assert df.ref_id.is_unique

    def within(col):
        g = [(roc_auc_score(x.Preg, x[col]), len(x)) for _, x in df.groupby('centre')]
        return sum(a * k for a, k in g) / sum(k for _, k in g)
    got = {'dt_auc_centre': {c: roc_auc_score(x.Preg, x['DIGITAL TWIN'])
                             for c, x in df.groupby('centre') if c in PUBLISHED['dt_auc_centre']},
           'dt_auc_within_weighted': within('DIGITAL TWIN'),
           'dt_auc_pooled': roc_auc_score(df.Preg, df['DIGITAL TWIN']),
           'tabpfn_auc_within_weighted': within('PRAI')}
    checks = []
    for k, pub in PUBLISHED.items():
        items = pub.items() if isinstance(pub, dict) else [(None, pub)]
        for sub, v in items:
            val = got[k][sub] if sub else got[k]
            checks.append({'metric': k + (f'[{sub}]' if sub else ''), 'published': v,
                           'reconstructed': round(float(val), 4), 'ok': bool(abs(val - v) <= TOL)})
    failed = [c for c in checks if not c['ok']]
    assert not failed, f'does not reproduce the paper: {failed}'
    return checks


def write_readonly_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    df.to_csv(tmp, index=False, encoding='utf-8-sig')
    os.replace(tmp, path)
    os.chmod(path, stat.S_IREAD)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--force', action='store_true', help='overwrite an existing frozen reference')
    args = ap.parse_args()
    if COHORT.exists() and not args.force:
        raise SystemExit(f'{COHORT} already frozen; pass --force to rebuild it')
    for p in (COHORT, CROSSWALK, OUT / 'manifest.json', OUT / 'README.md'):
        if p.exists():
            os.chmod(p, stat.S_IREAD | stat.S_IWRITE)

    df, sources = assemble()
    checks = verify(df)
    cols = ['ref_id', 'centre', 'rbmo_label'] + SCHEMA + ['tabpfn_ci_low', 'tabpfn_ci_high']
    cohort = df[cols]
    write_readonly_csv(cohort, COHORT)
    write_readonly_csv(df[['ref_id', 'centre'] + PRIVATE], CROSSWALK)

    ci_w = (pd.to_numeric(df.tabpfn_ci_high, errors='coerce')
            - pd.to_numeric(df.tabpfn_ci_low, errors='coerce'))
    manifest = {
        'name': 'RBMO prospective cohort (Digital Twin vs TabPFN, clinical pregnancy)',
        'frozen_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'builder': str(Path(__file__).relative_to(REPO)),
        'n_cycles': len(cohort), 'n_pregnant': int(df.Preg.sum()),
        'centres': sources,
        'per_centre': {c: {'n': len(x), 'pregnancy_rate': round(x.Preg.mean(), 4)}
                       for c, x in df.groupby('centre', sort=False)},
        'verification_vs_manuscript': checks,
        'cohort_sha256': sha256(COHORT),
        'crosswalk': str(CROSSWALK.relative_to(REPO)),
        'notes': {
            'DIGITAL TWIN': 'combined estimate (RBMO layer 7) = the Digital Twin probability compared in the paper',
            'PRAI': 'TabPFN probability of clinical pregnancy (filled from the TabPFN log for Fertimed and Two-Line)',
            'Preg': 'observed clinical pregnancy, 1/0',
            'tabpfn_ci': f'TabPFN interval; zero width in {int((ci_w == 0).sum())} of {len(ci_w)} cycles, '
                         'so it is not a usable uncertainty estimate',
            'dt_versions': 'Digital Twin 7.0 batch outputs, July-August 2026, fixed before outcomes were analysed',
            'identifiers': 'none in the cohort file; ref_id links to the crosswalk in local_private',
        },
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'README.md').write_text(README.format(n=len(cohort)), encoding='utf-8')
    for p in (OUT / 'manifest.json', OUT / 'README.md'):
        os.chmod(p, stat.S_IREAD)
    print(json.dumps({k: manifest[k] for k in ('n_cycles', 'n_pregnant', 'per_centre')}, indent=1))
    for c in checks:
        print(f"  {c['metric']:32s} published {c['published']:.3f}  reconstructed {c['reconstructed']:.4f}")
    print(f'frozen -> {COHORT.relative_to(REPO)}')


README = """# RBMO prospective cohort — frozen reference

{n} cycles from four centres (Fertimed = Centre A, GGRC = Centre B, SkyFert = Centre C,
Two-Line = 4th centre) with the Digital Twin (`DIGITAL TWIN`) and TabPFN (`PRAI`)
probability of clinical pregnancy and the observed outcome (`Preg`), exactly as analysed in
the RBMO revision (August 2026). The prospective reference for model comparisons as of this
date.

- `rbmo_prospective_cohort.csv`: one row per cycle, no identifiers; `ref_id` is stable.
- `manifest.json`: source files with sha256, per-centre counts, and the check that this file
  reproduces the published AUCs (Tables 4 and 6).
- The ref_id → patient crosswalk is in `data/processed/local_private/reference/` and must not
  be shared.

Files are read-only. Rebuild only deliberately:
`python reports/rbmo_prospective/build_cohort.py --force` (it re-verifies against the paper).
Use the per-centre analysis: pregnancy rates range from 26% to 70% between centres.
"""

if __name__ == '__main__':
    main()
