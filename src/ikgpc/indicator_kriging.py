"""Simple indicator kriging (IK) of a categorical variable.

Two variants are provided, matching the closure analysis of Sect. 4.3:

* ``shared=True``: one indicator covariance for all classes (the categorical
  analogue of median IK). With class means that sum to one, the raw
  estimates sum to one exactly at every location.
* ``shared=False``: one covariance per class, as in common practice. The
  kriging weights then differ between classes and closure is lost.

Covariance parameters (sill, length-scale, nugget) are fitted by maximum
likelihood on the centred indicators, I_k - m_k. Predictions are returned
raw, before any clipping or renormalization.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

__all__ = ["IndicatorKriging", "one_hot"]


def one_hot(z, C):
    Y = np.zeros((C, len(z)))
    Y[np.asarray(z, dtype=int), np.arange(len(z))] = 1.0
    return Y


class IndicatorKriging:
    """Simple IK with ML-fitted indicator covariance(s).

    Parameters
    ----------
    kernel : template kernel (only its family and anisotropy are used).
    shared : fit one covariance for all classes (True) or one per class.
    means : ``"uniform"`` (1/C, the linearization point of Sect. 4.1),
        ``"proportions"`` (sample class proportions) or an array of length C.
    fit : if False, use ``kernel`` and ``nugget`` as given.
    """

    def __init__(self, kernel, shared=True, means="uniform", nugget=0.1, fit=True):
        self.kernel = kernel
        self.shared = shared
        self.means = means
        self.nugget = nugget
        self.fit_cov = fit

    def _neg_loglik(self, theta, R):
        sill, ls, nug = np.exp(theta)
        k = self.kernel.with_params(variance=sill, lengthscales=(ls,))
        K = k(self.X_) + nug * np.eye(self.X_.shape[0])
        try:
            cf = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10
        logdet = 2.0 * np.sum(np.log(np.diag(cf[0])))
        quad = np.sum(R * cho_solve(cf, R.T).T)
        return 0.5 * quad + 0.5 * R.shape[0] * logdet

    def _fit_one(self, R):
        theta0 = np.log([max(R.var(), 1e-2), 1.0, 0.05])
        bounds = [(np.log(1e-3), np.log(2.0)), (np.log(1e-2), np.log(1e2)), (np.log(1e-4), np.log(1.0))]
        res = minimize(self._neg_loglik, theta0, args=(R,), method="L-BFGS-B", bounds=bounds)
        sill, ls, nug = np.exp(res.x)
        k = self.kernel.with_params(variance=sill, lengthscales=(ls,))
        return k, nug

    def fit(self, X, z, n_classes=None):
        t0 = time.perf_counter()
        self.X_ = np.atleast_2d(np.asarray(X, dtype=float))
        z = np.asarray(z, dtype=int)
        self.C_ = int(n_classes or z.max() + 1)
        self.Y_ = one_hot(z, self.C_)
        if isinstance(self.means, str):
            self.m_ = (np.full(self.C_, 1.0 / self.C_) if self.means == "uniform"
                       else self.Y_.mean(axis=1))
        else:
            self.m_ = np.asarray(self.means, dtype=float)
        R = self.Y_ - self.m_[:, None]
        if not self.fit_cov:
            self.kernels_ = [(self.kernel, self.nugget)] * self.C_
        elif self.shared:
            self.kernels_ = [self._fit_one(R)] * self.C_
        else:
            self.kernels_ = [self._fit_one(R[c:c + 1]) for c in range(self.C_)]
        self.timings = {"fit_s": time.perf_counter() - t0}
        return self

    def weights(self, Xs, c=0):
        k, nug = self.kernels_[c]
        K = k(self.X_) + nug * np.eye(self.X_.shape[0])
        return cho_solve(cho_factor(K, lower=True), k(self.X_, Xs))

    def predict(self, Xs):
        """Raw simple-IK estimates, shape (N, C)."""
        t0 = time.perf_counter()
        Xs = np.atleast_2d(np.asarray(Xs, dtype=float))
        out = np.empty((Xs.shape[0], self.C_))
        cache = {}
        for c in range(self.C_):
            key = id(self.kernels_[c][0]) if self.shared else c
            if key not in cache:
                cache[key] = self.weights(Xs, c)
            W = cache[key]
            out[:, c] = self.m_[c] + (self.Y_[c] - self.m_[c]) @ W
        self.timings["predict_s"] = time.perf_counter() - t0
        return out
