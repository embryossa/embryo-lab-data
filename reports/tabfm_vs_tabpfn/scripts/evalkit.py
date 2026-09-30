# -*- coding: utf-8 -*-
"""Metric kit: discrimination, calibration, and paired bootstrap deltas."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             log_loss, accuracy_score)


def ece(y, p, n_bins=10):
    """Equal-width expected calibration error."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def cal_slope_intercept(y, p, eps=1e-6):
    """Cox calibration: logit(y) ~ a + b*logit(p). b=1, a=0 is perfect."""
    from sklearn.linear_model import LogisticRegression
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    y = np.asarray(y, int)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(lp, y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def all_metrics(y, p):
    y = np.asarray(y, int); p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    single = len(np.unique(y)) < 2
    slope, inter = cal_slope_intercept(y, p)
    return {
        "n": int(len(y)),
        "pos_rate": float(y.mean()),
        "mean_pred": float(p.mean()),
        "roc_auc": np.nan if single else float(roc_auc_score(y, p)),
        "pr_auc": np.nan if single else float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": ece(y, p),
        "cal_slope": slope,
        "cal_intercept": inter,
        "acc@0.5": float(accuracy_score(y, (p >= 0.5).astype(int))),
    }


def _stat(y, p, name):
    if name == "roc_auc":
        return roc_auc_score(y, p)
    if name == "pr_auc":
        return average_precision_score(y, p)
    if name == "brier":
        return brier_score_loss(y, p)
    if name == "log_loss":
        return log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])
    if name == "ece":
        return ece(y, p)
    raise ValueError(name)


def boot_ci(y, p, name, n_boot=2000, seed=0, alpha=0.05):
    """Stratified bootstrap CI for one model's metric."""
    y = np.asarray(y, int); p = np.asarray(p, float)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), True),
                              rng.choice(neg, len(neg), True)])
        try:
            vals.append(_stat(y[idx], p[idx], name))
        except ValueError:
            continue
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def paired_boot(y, p_a, p_b, name, n_boot=2000, seed=0, alpha=0.05):
    """Paired stratified bootstrap of (a - b) on the SAME resampled rows."""
    y = np.asarray(y, int); p_a = np.asarray(p_a, float); p_b = np.asarray(p_b, float)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), True),
                              rng.choice(neg, len(neg), True)])
        try:
            d.append(_stat(y[idx], p_a[idx], name) - _stat(y[idx], p_b[idx], name))
        except ValueError:
            continue
    d = np.asarray(d)
    if d.size == 0:
        return dict(delta=np.nan, lo=np.nan, hi=np.nan, p_two_sided=np.nan)
    obs = _stat(y, p_a, name) - _stat(y, p_b, name)
    # two-sided bootstrap p: how often the sign of the resampled delta flips
    pv = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return dict(delta=float(obs),
                lo=float(np.percentile(d, 100 * alpha / 2)),
                hi=float(np.percentile(d, 100 * (1 - alpha / 2))),
                p_two_sided=float(min(1.0, pv)))
