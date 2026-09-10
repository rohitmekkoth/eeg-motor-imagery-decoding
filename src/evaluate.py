"""Evaluation harness. Every experiment in the roadmap is a call to one of these."""

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold


def chance_threshold(n, alpha=0.05):
    """Lowest accuracy beating chance at alpha given n trials. At n=45 this is ~0.62, not 0.50."""
    for k in range(n // 2, n + 1):
        if binomtest(k, n, 0.5, alternative="greater").pvalue < alpha:
            return k / n
    return 1.0


def _run_folds(model_fn, X, y, groups, splitter):
    """Train a fresh model on each training fold and predict its held-out fold."""
    pred = np.full(len(y), -1)
    proba = np.full(len(y), np.nan)

    for train, test in splitter.split(X, y, groups):
        model = model_fn()                    # fresh model per fold — never reuse a fitted one
        model.fit(X[train], y[train])
        pred[test] = model.predict(X[test])
        proba[test] = model.predict_proba(X[test])[:, 1]

    return pred, proba


def _per_subject(y, pred, proba, subjects):
    """Score each subject separately. Per-subject is the unit of analysis, never pooled epochs."""
    rows = []
    for s in np.unique(subjects):
        m = subjects == s
        rows.append({
            "subject": int(s),
            "n": int(m.sum()),
            "acc": balanced_accuracy_score(y[m], pred[m]),
            "auc": roc_auc_score(y[m], proba[m]) if len(np.unique(y[m])) == 2 else np.nan,
            "chance_95": chance_threshold(int(m.sum())),
        })
    return pd.DataFrame(rows)


def tier_a(model_fn, X, y, subjects, n_splits=5, seed=0):
    """Random split over epochs, ignoring who they came from. Expected to be inflated."""
    splitter = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    pred, proba = _run_folds(model_fn, X, y, None, splitter)
    return _per_subject(y, pred, proba, subjects)


def tier_b(model_fn, X, y, subjects, runs):
    """Within each subject, leave one run out. Simulates a user who calibrated the device."""
    frames = []
    for s in np.unique(subjects):
        m = subjects == s
        pred, proba = _run_folds(model_fn, X[m], y[m], runs[m], LeaveOneGroupOut())
        frames.append(_per_subject(y[m], pred, proba, subjects[m]))
    return pd.concat(frames, ignore_index=True)


def tier_c(model_fn, X, y, subjects):
    """Leave one subject out. The headline: does this work on a person never seen in training?"""
    pred, proba = _run_folds(model_fn, X, y, subjects, LeaveOneGroupOut())
    return _per_subject(y, pred, proba, subjects)


def bootstrap_ci(values, n_boot=2000, seed=0):
    """95% CI on the mean, resampling subjects — trials within a subject are not independent."""
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, len(values), replace=True).mean() for _ in range(n_boot)]
    return tuple(np.percentile(means, [2.5, 97.5]))


def summarize(df, label):
    """Collapse per-subject scores into one reportable row."""
    lo, hi = bootstrap_ci(df.acc.values)
    return {
        "tier": label,
        "mean_acc": round(df.acc.mean(), 3),
        "median_acc": round(df.acc.median(), 3),
        "ci_low": round(lo, 3),
        "ci_high": round(hi, 3),
        "min": round(df.acc.min(), 3),
        "max": round(df.acc.max(), 3),
        "mean_auc": round(df.auc.mean(), 3),
        "frac_above_chance": round((df.acc > df.chance_95).mean(), 3),
        "wilcoxon_p": f"{wilcoxon(df.acc - 0.5, alternative='greater').pvalue:.2e}",
    }