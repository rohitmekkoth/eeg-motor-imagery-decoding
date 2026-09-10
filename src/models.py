"""Model definitions. Everything here is fit inside a training fold only."""

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline


def csp_lda(n_components=6):
    """CSP spatial filters into shrinkage LDA. No hyperparameters, so no inner tuning loop."""
    return Pipeline([
        ("csp", CSP(n_components=n_components, log=True)),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])