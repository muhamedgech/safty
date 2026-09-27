"""Score functions plugged into the PACT certificate (paper §3, §4.3).

  topic control  English AdvBench-vs-benign probe (predicts topic, not the model's decision)
  PACT-A0        probe trained on English prompt states with the transported label z
  PACT-A2        A0 after Procrustes-aligning the target language onto English
  PACT-TT        probe trained on target-language states with the same transported z

Every score is s = -P(positive class), so a HIGH score means "looks refusal-worthy".

Speed trick (exact): with L2 regularisation the logistic-regression weights always lie in
the span of the training examples, so we fit in that n-dimensional span instead of the
4096-dimensional hidden space. The fitted model is identical; fitting is ~100x faster.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


@dataclass
class Probe:
    layer: int
    C: float
    mean: np.ndarray        # standardisation, fitted on the fit split only
    std: np.ndarray
    basis: np.ndarray       # (d, r) orthonormal basis of the training span
    clf: LogisticRegression
    cv_auc: float

    def score(self, X: np.ndarray) -> np.ndarray:
        """X: (n, d) states at self.layer. Returns s = -P(y=1)."""
        Z = ((X - self.mean) / self.std) @ self.basis
        return -self.clf.predict_proba(Z)[:, 1]

    def score_layers(self, feats: dict[int, np.ndarray]) -> np.ndarray:
        return self.score(feats[self.layer])


def _span(X: np.ndarray):
    """Standardise X and return (mean, std, basis, coordinates in the training span)."""
    mean = X.mean(0)
    std = X.std(0) + 1e-6
    Xs = (X - mean) / std
    _, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    basis = Vt[S > 1e-8 * S[0]].T
    return mean, std, basis, Xs @ basis


def _cv_auc(Z: np.ndarray, y: np.ndarray, C: float, folds: int, seed: int) -> float:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    aucs = []
    for train, val in splitter.split(Z, y):
        clf = LogisticRegression(C=C, max_iter=5000).fit(Z[train], y[train])
        aucs.append(roc_auc_score(y[val], clf.decision_function(Z[val])))
    return float(np.mean(aucs))


def fit_probe(feats: dict[int, np.ndarray], y: np.ndarray, Cs: list[float], folds: int = 5,
              seed: int = 0, layers: list[int] | None = None) -> Probe:
    """Choose (layer, C) by cross-validated AUC on this fit set, then refit on all of it.

    Selection is redone for every fit set: reusing hyperparameters chosen on a different
    partition leaks information (paper §4.1).
    """
    y = np.asarray(y).astype(int)
    folds = max(2, min(folds, np.bincount(y, minlength=2).min()))
    best = None
    for layer in layers or sorted(feats):
        mean, std, basis, Z = _span(feats[layer])
        for C in Cs:
            auc = _cv_auc(Z, y, C, folds, seed)
            if best is None or auc > best[0]:
                best = (auc, layer, C, mean, std, basis, Z)
    auc, layer, C, mean, std, basis, Z = best
    clf = LogisticRegression(C=C, max_iter=5000).fit(Z, y)
    return Probe(layer, C, mean, std, basis, clf, auc)


@dataclass
class ProcrustesProbe:
    """PACT-A2: project onto English principal components, rotate the target onto English."""
    en_mean: np.ndarray
    tgt_mean: np.ndarray
    components: np.ndarray  # (d, K)
    rotation: np.ndarray    # (K, K)
    probe: Probe            # trained on English coordinates

    def score(self, X_target: np.ndarray) -> np.ndarray:
        return self.probe.score(((X_target - self.tgt_mean) @ self.components) @ self.rotation)


def fit_procrustes_probe(X_en: np.ndarray, X_tgt: np.ndarray, y: np.ndarray, k: int, Cs: list[float],
                         folds: int = 5, seed: int = 0) -> ProcrustesProbe:
    """X_en[i] and X_tgt[i] are the same prompt in English and in the target language.

    R = argmin ||P_tgt R - P_en|| over orthogonal R (solved by SVD), after projecting both
    onto the top-k principal components of the centred English states.
    """
    en_mean, tgt_mean = X_en.mean(0), X_tgt.mean(0)
    _, _, Vt = np.linalg.svd(X_en - en_mean, full_matrices=False)
    components = Vt[:min(k, len(X_en) - 1)].T
    P_en = (X_en - en_mean) @ components
    P_tgt = (X_tgt - tgt_mean) @ components
    U, _, Wt = np.linalg.svd(P_tgt.T @ P_en)
    rotation = U @ Wt
    probe = fit_probe({0: P_en}, y, Cs, folds, seed)
    return ProcrustesProbe(en_mean, tgt_mean, components, rotation, probe)
