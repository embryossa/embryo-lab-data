#!/usr/bin/env python3
"""Guo et al. (Front Endocrinol 2026;17:1916182) three-stage ensemble framework, re-run on the
frozen RBMO prospective cohort (1,026 cycles, 4 centres) and compared, on the same test cycles
and with the same metrics, against the prospective Digital Twin (layer 7) and TabPFN
probabilities of clinical pregnancy.

Designs
  E  external: leave-one-centre-out, every centre in turn (A, B, C and the 4th centre, n=28)
     is completely unseen during training, feature selection, tuning and calibration. Development centres split 85/15 (train / internal validation, stratified
     by centre x outcome); the held-out centre is never touched before testing.
  I  internal: one stratified (centre x outcome) random 70/15/15 train / internal / test split
     of all 1,026 cycles, the analogue of Guo's internal test cohort.

Per split and stage, everything is fitted on train only (winsorisation 1/99, MissForest-style
imputation, VIF>10, voting feature selection LASSO/RF/RFE/Boruta >=2 votes + age whitelist,
RobustScaler+Yeo-Johnson for linear/NN/kNN/NB), Optuna TPE 30 trials 5-fold CV AUC, Platt
scaling on the internal set; SES (exhaustive Pareto front of AUC vs mean pairwise
correlation on the internal set, knee point), DEP (15 nearest internal cycles, prune worst
50% by local Brier), stacking (LR meta-learner on 5-fold out-of-fold predictions of the SES
subset). The model with the best internal AUC is the stage's selected model.

Stages available in this cohort (AMH, AFC, BMI and sperm source are constant placeholders in
all 1,026 rows and carry no information):
  S1  age, attempt number
  S2  S1 + follicles at OPU
  S2+ S2 + OCC, MII  (= the information the Digital Twin received: DT-matched)
  S3  S2+ + inseminated, 2PN, blastocysts, good blastocysts (+ missingness indicators)
Outcome: clinical pregnancy (Preg), not live birth as in Guo.
"""
import hashlib, json, itertools, warnings, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import optuna, xgboost as xgb, lightgbm as lgb, catboost as cb

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 20260922
OUT = Path(__file__).resolve().parent
REF = OUT.parents[2] / 'data' / 'processed' / 'reference' / 'rbmo_prospective_2026'
LABEL = {'Fertimed': 'Centre A', 'GGRC': 'Centre B', 'SkyFert': 'Centre C', 'Two-Line': '4th centre'}

S1 = ['age', 'attempt']
S2 = S1 + ['follicles']
S2P = S2 + ['OCC', 'MII']
S3 = S2P + ['inseminated', 'pn2', 'bl', 'good_bl']
STAGES = {'S1': S1, 'S2': S2, 'S2+': S2P, 'S3': S3}
MISS_IND = ['MII', 'inseminated', 'bl', 'good_bl']   # indicator added when missing in train
N_TRIALS = 30


def load():
    path = REF / 'rbmo_prospective_cohort.csv'
    man = json.loads((REF / 'manifest.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == man['cohort_sha256'], 'cohort != manifest'
    d = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
    const = [c for c in ['amh', 'afc', 'bmi', 'sperm_source'] if d[c].nunique() == 1]
    df = pd.DataFrame({'ref_id': d.ref_id, 'centre': d.centre, 'y': d.Preg.astype(int),
                       'age': d.age, 'attempt': d.attempt_number, 'follicles': d.follicles_tvp,
                       'OCC': d.OCC, 'MII': d.MII, 'inseminated': d.Inseminated, 'pn2': d['2pN'],
                       'bl': d.Bl, 'good_bl': d['Good Bl'],
                       'p_dt': d['DIGITAL TWIN'], 'p_tabpfn': d.PRAI})
    assert len(df) == man['n_cycles']
    return df, const


# ---------------- preprocessing (fit on train only) ----------------
class Prep:
    def __init__(self, feats):
        self.feats = feats

    def fit(self, X):
        X = X[self.feats]
        self.lo, self.hi = X.quantile(.01), X.quantile(.99)
        self.ind = [c for c in self.feats if c in MISS_IND and X[c].isna().any()]
        self.imp = None
        if X.isna().any().any():
            self.imp = IterativeImputer(ExtraTreesRegressor(n_estimators=30, random_state=SEED),
                                        max_iter=5, random_state=SEED).fit(X.clip(self.lo, self.hi, axis=1))
        return self

    def transform(self, X):
        X = X[self.feats]
        miss = X[self.ind].isna().astype(float).add_suffix('_missing')
        Xc = X.clip(self.lo, self.hi, axis=1)
        if self.imp is not None:
            Xc = pd.DataFrame(self.imp.transform(Xc), columns=self.feats, index=X.index)
        return pd.concat([Xc, miss], axis=1)


def vif_drop(X, keep=('age',), thr=10.0):
    cols = list(X.columns)
    while len(cols) > 2:
        Z = X[cols].to_numpy(float)
        v = {}
        for i, c in enumerate(cols):
            if X[c].nunique() <= 2 or c in keep:
                continue
            others = np.delete(Z, i, axis=1)
            A = np.column_stack([np.ones(len(Z)), others])
            beta, *_ = np.linalg.lstsq(A, Z[:, i], rcond=None)
            r2 = 1 - np.var(Z[:, i] - A @ beta) / max(np.var(Z[:, i]), 1e-12)
            v[c] = 1 / max(1 - r2, 1e-12)
        if not v or max(v.values()) <= thr:
            break
        cols.remove(max(v, key=v.get))
    return cols


def select_features(X, y, whitelist=('age',)):
    cols = list(X.columns)
    if len(cols) <= 2:
        return cols, {c: 4 for c in cols}
    Xs = make_pipeline(RobustScaler(), PowerTransformer()).fit_transform(X)
    votes = dict.fromkeys(cols, 0)
    las = LogisticRegressionCV(Cs=20, cv=5, penalty='l1', solver='liblinear', scoring='roc_auc',
                               random_state=SEED).fit(Xs, y)
    for c, w in zip(cols, las.coef_[0]):
        votes[c] += abs(w) > 1e-8
    rf = RandomForestClassifier(100, random_state=SEED, class_weight='balanced').fit(X, y)
    for c, w in zip(cols, rf.feature_importances_):
        votes[c] += w >= np.mean(rf.feature_importances_)
    rfe = RFE(LogisticRegression(max_iter=2000), n_features_to_select=max(1, len(cols) // 2)).fit(Xs, y)
    for c, s in zip(cols, rfe.support_):
        votes[c] += bool(s)
    try:
        from boruta import BorutaPy
        np.int, np.float, np.bool = int, float, bool  # boruta uses removed numpy aliases
        br = BorutaPy(RandomForestClassifier(max_depth=7, class_weight='balanced', random_state=SEED),
                      n_estimators='auto', random_state=SEED, max_iter=50).fit(X.to_numpy(float), y)
        for c, s in zip(cols, br.support_ | br.support_weak_):
            votes[c] += bool(s)
    except Exception as e:  # Boruta optional; recorded in the log
        print('boruta skipped:', e, file=sys.stderr)
    sel = [c for c in cols if votes[c] >= 2 or c in whitelist]
    return sel, {c: int(v) for c, v in votes.items()}


# ---------------- base learners with Optuna spaces ----------------
def scaled(m):
    return make_pipeline(RobustScaler(), PowerTransformer(), m)


def make_model(name, t):
    s = dict(random_state=SEED)
    if name == 'LR':
        return scaled(LogisticRegression(C=t.suggest_float('C', 1e-3, 10, log=True), max_iter=5000))
    if name == 'RF':
        return RandomForestClassifier(n_estimators=t.suggest_int('n', 100, 500, step=100),
                                      max_depth=t.suggest_int('d', 2, 10), min_samples_leaf=t.suggest_int('leaf', 1, 30),
                                      class_weight='balanced', n_jobs=1, **s)
    if name == 'XGBoost':
        return xgb.XGBClassifier(n_estimators=t.suggest_int('n', 50, 400, step=50), max_depth=t.suggest_int('d', 2, 6),
                                 learning_rate=t.suggest_float('lr', .01, .3, log=True),
                                 subsample=t.suggest_float('ss', .5, 1), colsample_bytree=t.suggest_float('cs', .5, 1),
                                 min_child_weight=t.suggest_float('mcw', 1, 20, log=True), n_jobs=1, verbosity=0, **s)
    if name == 'LightGBM':
        return lgb.LGBMClassifier(n_estimators=t.suggest_int('n', 50, 400, step=50), num_leaves=t.suggest_int('nl', 4, 31),
                                  learning_rate=t.suggest_float('lr', .01, .3, log=True),
                                  min_child_samples=t.suggest_int('mcs', 5, 60), subsample=t.suggest_float('ss', .5, 1),
                                  subsample_freq=1, class_weight='balanced', n_jobs=1, verbose=-1, **s)
    if name == 'CatBoost':
        return cb.CatBoostClassifier(iterations=t.suggest_int('n', 100, 400, step=100), depth=t.suggest_int('d', 2, 6),
                                     learning_rate=t.suggest_float('lr', .01, .3, log=True),
                                     l2_leaf_reg=t.suggest_float('l2', 1, 10, log=True), auto_class_weights='Balanced',
                                     verbose=0, thread_count=1, random_seed=SEED)
    if name == 'SVM':
        return scaled(SVC(C=t.suggest_float('C', 1e-2, 10, log=True), gamma=t.suggest_float('g', 1e-3, 1, log=True),
                          probability=True, class_weight='balanced', **s))
    if name == 'MLP':
        return scaled(MLPClassifier(hidden_layer_sizes=(t.suggest_int('h', 4, 64),), alpha=t.suggest_float('a', 1e-4, 1, log=True),
                                    learning_rate_init=1e-3, max_iter=1000, early_stopping=True, **s))
    if name == 'NaiveBayes':
        return scaled(GaussianNB(var_smoothing=t.suggest_float('vs', 1e-10, 1e-2, log=True)))
    if name == 'kNN':
        return scaled(KNeighborsClassifier(n_neighbors=t.suggest_int('k', 5, 75), weights=t.suggest_categorical('w', ['uniform', 'distance'])))
    if name == 'DecisionTree':
        return DecisionTreeClassifier(max_depth=t.suggest_int('d', 2, 8), min_samples_leaf=t.suggest_int('leaf', 5, 60),
                                      class_weight='balanced', **s)
    raise KeyError(name)


BASE = ['LR', 'RF', 'XGBoost', 'LightGBM', 'CatBoost', 'SVM', 'MLP', 'NaiveBayes', 'kNN', 'DecisionTree']


def tune(name, X, y):
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    def obj(t):
        m = make_model(name, t)
        aucs = []
        for tr, va in cv.split(X, y):
            m.fit(X.iloc[tr], y[tr])
            aucs.append(roc_auc_score(y[va], m.predict_proba(X.iloc[va])[:, 1]))
        return np.mean(aucs)
    st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=N_TRIALS)
    return make_model(name, optuna.trial.FixedTrial(st.best_params)), st.best_value


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Platt:
    def fit(self, p, y):
        self.m = LogisticRegression(C=1e6, max_iter=5000).fit(logit(p)[:, None], y)
        return self

    def __call__(self, p):
        return self.m.predict_proba(logit(p)[:, None])[:, 1]


def run_stage(split, stage, feats, tr, va, te):
    t0 = time.time()
    ytr, yva = tr.y.to_numpy(), va.y.to_numpy()
    prep = Prep(feats).fit(tr)
    Xtr, Xva, Xte = prep.transform(tr), prep.transform(va), prep.transform(te)
    keep = vif_drop(Xtr)
    sel, votes = select_features(Xtr[keep], ytr)
    Xtr, Xva, Xte = Xtr[sel], Xva[sel], Xte[sel]
    P = {'va': {}, 'te': {}, 'raw_va': {}}
    cv_auc, models, oof = {}, {}, {}
    for b in BASE:
        m, cv_auc[b] = tune(b, Xtr, ytr)
        oof[b] = cross_val_predict(m, Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                   method='predict_proba')[:, 1]
        m.fit(Xtr, ytr)
        models[b] = m
        pv, pt = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
        cal = Platt().fit(pv, yva)
        P['va'][b], P['te'][b] = cal(pv), cal(pt)
    # SES: exhaustive Pareto front (1 - AUC of mean, mean pairwise correlation) on internal set, knee
    V = pd.DataFrame(P['va'])
    corr = V.corr().to_numpy()
    cand = []
    for k in range(2, len(BASE) + 1):
        for sub in itertools.combinations(range(len(BASE)), k):
            idx = list(sub)
            err = 1 - roc_auc_score(yva, V.iloc[:, idx].mean(axis=1))
            c = corr[np.ix_(idx, idx)][np.triu_indices(k, 1)].mean()
            cand.append((err, c, idx))
    front = [a for a in cand if not any((b[0] <= a[0] and b[1] <= a[1]) and (b[0] < a[0] or b[1] < a[1]) for b in cand)]
    E = np.array([[a[0], a[1]] for a in front])
    En = (E - E.min(0)) / np.where(np.ptp(E, 0) > 0, np.ptp(E, 0), 1)
    ses = [BASE[i] for i in front[int(np.argmin(np.hypot(En[:, 0], En[:, 1])))][2]]
    P['va']['SES'] = V[ses].mean(axis=1).to_numpy()
    P['te']['SES'] = pd.DataFrame(P['te'])[ses].mean(axis=1).to_numpy()
    # Stacking on the SES subset: LR meta-learner on out-of-fold train predictions
    meta = LogisticRegressionCV(cv=5, scoring='roc_auc', max_iter=5000).fit(
        logit(np.column_stack([oof[b] for b in ses])), ytr)
    for part, X in (('va', Xva), ('te', Xte)):
        P[part]['Stacking'] = meta.predict_proba(logit(np.column_stack([models[b].predict_proba(X)[:, 1] for b in ses])))[:, 1]
    # DEP: 15 nearest internal cycles, prune worst 50% of base models by local Brier
    sc = make_pipeline(RobustScaler(), PowerTransformer()).fit(Xtr)
    nn = NearestNeighbors(n_neighbors=15).fit(sc.transform(Xva))
    Vv = V[BASE].to_numpy()
    for part, X in (('va', Xva), ('te', Xte)):
        T = pd.DataFrame(P[part])[BASE].to_numpy()
        _, nb = nn.kneighbors(sc.transform(X), n_neighbors=16 if part == 'va' else 15)
        if part == 'va':
            nb = nb[:, 1:]  # leave the cycle itself out
        out = []
        for i in range(len(X)):
            loss = ((Vv[nb[i]] - yva[nb[i], None]) ** 2).mean(0)
            best = np.argsort(loss)[: len(BASE) // 2]
            out.append(T[i, best].mean())
        P[part]['DEP'] = np.array(out)
    int_auc = {m: roc_auc_score(yva, p) for m, p in P['va'].items()}
    chosen = max(int_auc, key=int_auc.get)
    print(f'{split} {stage}: {time.time() - t0:.0f}s, chosen={chosen}', file=sys.stderr, flush=True)
    return dict(split=split, stage=stage, features=sel, votes=votes, vif_kept=keep, ses=ses, chosen=chosen,
                cv_auc=cv_auc, int_auc=int_auc, test_pred=P['te'], test_ids=te.ref_id.tolist(),
                shap_payload=(models, ses, Xtr, Xte) if split == 'I' and stage == 'S3' else None)


def splits(df):
    strat = df.centre + '_' + df.y.astype(str)
    out = []
    for held in ['Fertimed', 'GGRC', 'SkyFert', 'Two-Line']:
        dev, te = df[df.centre != held], df[df.centre == held]
        tr, va = train_test_split(dev, test_size=.15, stratify=strat[dev.index], random_state=SEED)
        out.append((f'E:{held}', tr, va, te))
    tr, rest = train_test_split(df, test_size=.30, stratify=strat, random_state=SEED)
    va, te = train_test_split(rest, test_size=.50, stratify=strat[rest.index], random_state=SEED)
    out.append(('I', tr, va, te))
    return out


if __name__ == '__main__':
    # optional: python guo_three_stage.py E:Two-Line  -> run only the named splits
    only = set(sys.argv[1:])
    df, const = load()
    jobs = [(s, st, f, tr, va, te) for s, tr, va, te in splits(df) if not only or s in only
            for st, f in STAGES.items()]
    res = Parallel(n_jobs=8, verbose=0)(delayed(run_stage)(*j) for j in jobs)
    import pickle
    (OUT / 'work').mkdir(exist_ok=True)
    tag = '_'.join(sorted(only)).replace(':', '-') if only else 'all'
    with open(OUT / 'work' / f'results_{tag}.pkl', 'wb') as fh:
        pickle.dump({'res': res, 'const_dropped': const, 'seed': SEED}, fh)
    print('done', file=sys.stderr)
