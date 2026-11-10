"""Multi-class Gaussian process classification with a softmax link.

Implements the Laplace approximation of Rasmussen and Williams (2006),
Algorithm 3.3 (posterior mode and approximate log marginal likelihood) and
Algorithm 3.4 (predictive mean, covariance and probabilities), for C latent
fields with independent zero-mean Gaussian process priors that share one
covariance function.

Two properties used in the paper are checked in ``tests/``:

* The negative log posterior is strictly convex in the latent values, so the
  Newton iteration converges to a unique mode.
* With a shared kernel, the mode satisfies sum_k f_k(s_i) = 0 at every data
  site and the predictive latent means also sum to zero, so the symmetric
  identifiability constraint of Sect. 3 holds without being imposed.

Two quantities are written out explicitly because they are easy to get wrong:

* The approximate log marginal likelihood includes -sum(log diag M), from
  det(I + K W) = det(I + K D) det(sum_c E_c).
* The predictive covariance between classes c and c' is
  b_c^T (M M^T)^{-1} b_{c'} with b_c = E_c k_*, from
  Cov = K_** - Q_*^T (I + W K)^{-1} W Q_*.
Both are verified against dense linear algebra in ``tests/test_softmax_gpc.py``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve, cholesky
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

__all__ = ["SoftmaxGPC", "LaplaceState"]


@dataclass
class LaplaceState:
    f: np.ndarray          # (C, n) latent mode
    a: np.ndarray          # (C, n) with f = K a
    pi: np.ndarray         # (C, n) softmax probabilities at the mode
    E: np.ndarray          # (C, n, n)
    L_M: np.ndarray        # (n, n) lower Cholesky factor of sum_c E_c
    log_q: float           # approximate log marginal likelihood
    n_iter: int


def _objective(a, f, Y):
    """Psi(f) = log p(y|f) - 0.5 f^T K^{-1} f, using a = K^{-1} f."""
    return -0.5 * np.sum(a * f) + np.sum(Y * f) - np.sum(logsumexp(f, axis=0))


def _E_matrices(K, pi):
    C, n = pi.shape
    E = np.empty((C, n, n))
    z_sum = 0.0
    eye = np.eye(n)
    for c in range(C):
        sd = np.sqrt(pi[c])
        B = eye + sd[:, None] * K * sd[None, :]
        L = cholesky(B, lower=True)
        E[c] = sd[:, None] * cho_solve((L, True), np.diag(sd))
        z_sum += np.sum(np.log(np.diag(L)))
    L_M = cholesky(E.sum(axis=0), lower=True)
    return E, L_M, z_sum


def laplace_mode(K, Y, a0=None, tol=1e-12, max_iter=200):
    """Newton iteration for the posterior mode (R&W Algorithm 3.3).

    Parameters
    ----------
    K : (n, n) shared prior covariance.
    Y : (C, n) one-hot labels.
    a0 : optional (C, n) warm start, f0 = K a0.
    """
    C, n = Y.shape
    a = np.zeros((C, n)) if a0 is None else a0.copy()
    f = a @ K
    psi = _objective(a, f, Y)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        pi = softmax(f, axis=0)
        E, L_M, _ = _E_matrices(K, pi)
        Wf = pi * f - pi * np.sum(pi * f, axis=0, keepdims=True)
        b = Wf + Y - pi
        cvec = np.einsum("cij,cj->ci", E, b @ K)
        s = cho_solve((L_M, True), cvec.sum(axis=0))
        a_new = b - cvec + np.einsum("cij,j->ci", E, s)
        f_new = a_new @ K
        psi_new = _objective(a_new, f_new, Y)
        step = 1.0
        while psi_new < psi - 1e-12 * max(1.0, abs(psi)) and step > 1e-6:
            step *= 0.5                     # damped Newton (rarely needed)
            a_try = a + step * (a_new - a)
            f_try = f + step * (f_new - f)
            psi_try = _objective(a_try, f_try, Y)
            if psi_try >= psi:
                a_new, f_new, psi_new = a_try, f_try, psi_try
                break
        converged = abs(psi_new - psi) <= tol * max(1.0, abs(psi))
        a, f, psi = a_new, f_new, psi_new
        if converged:
            break
    pi = softmax(f, axis=0)
    E, L_M, z_sum = _E_matrices(K, pi)
    log_q = psi - z_sum - np.sum(np.log(np.diag(L_M)))
    return LaplaceState(f=f, a=a, pi=pi, E=E, L_M=L_M, log_q=float(log_q), n_iter=n_iter)


class SoftmaxGPC:
    """Softmax Gaussian process classifier with a shared kernel.

    Parameters
    ----------
    kernel : a kernel object from :mod:`ikgpc.kernels`.
    optimize : one of ``None``, ``"variance"``, ``"isotropic"`` or
        ``"anisotropic"`` (azimuth fixed at ``kernel.azimuth``; variance,
        minor length-scale and anisotropy ratio >= 1 are fitted).
    """

    def __init__(self, kernel, optimize="isotropic", bounds=None, newton_tol=1e-12):
        self.kernel = kernel
        self.optimize = optimize
        self.bounds = bounds
        self.newton_tol = newton_tol
        self._a_warm = None
        self.timings = {}

    # -- parametrization -------------------------------------------------
    def _theta0(self):
        k = self.kernel
        if self.optimize == "variance":
            return np.log([k.variance])
        if self.optimize == "isotropic":
            return np.log([k.variance, k.lengthscales[0]])
        if self.optimize == "anisotropic":
            ls = tuple(k.lengthscales) * (2 if len(k.lengthscales) == 1 else 1)
            major, minor = max(ls), min(ls)
            return np.log([k.variance, minor, major / minor])
        raise ValueError(self.optimize)

    def _default_bounds(self):
        lv, ll = (np.log(1e-2), np.log(1e3)), (np.log(1e-2), np.log(1e2))
        if self.optimize == "variance":
            return [lv]
        if self.optimize == "isotropic":
            return [lv, ll]
        return [lv, ll, (0.0, np.log(20.0))]

    def _kernel_from_theta(self, theta):
        t = np.exp(theta)
        if self.optimize == "variance":
            return self.kernel.with_params(variance=t[0])
        if self.optimize == "isotropic":
            return self.kernel.with_params(variance=t[0], lengthscales=(t[1],))
        return self.kernel.with_params(variance=t[0], lengthscales=(t[1] * t[2], t[1]))

    # -- fitting ---------------------------------------------------------
    def _laplace(self, kernel):
        K = kernel(self.X_)
        state = laplace_mode(K, self.Y_, a0=self._a_warm, tol=self.newton_tol)
        self._a_warm = state.a
        return K, state

    def log_marginal_likelihood(self, theta):
        _, state = self._laplace(self._kernel_from_theta(np.asarray(theta)))
        return state.log_q

    def fit(self, X, z, n_classes=None):
        t0 = time.perf_counter()
        self.X_ = np.atleast_2d(np.asarray(X, dtype=float))
        if self.X_.shape[0] == 1 and np.ndim(X) == 1:
            self.X_ = self.X_.T
        z = np.asarray(z, dtype=int)
        self.C_ = int(n_classes or z.max() + 1)
        self.Y_ = np.zeros((self.C_, z.size))
        self.Y_[z, np.arange(z.size)] = 1.0
        self.n_lml_evals_ = 0
        if self.optimize:
            def nll(theta):
                self.n_lml_evals_ += 1
                return -self.log_marginal_likelihood(theta)
            res = minimize(nll, self._theta0(), method="L-BFGS-B",
                           bounds=self.bounds or self._default_bounds(),
                           options={"eps": 1e-4, "ftol": 1e-10, "maxiter": 200})
            self.kernel = self._kernel_from_theta(res.x)
            self.opt_result_ = res
        t1 = time.perf_counter()
        self.K_, self.state_ = self._laplace(self.kernel)
        self.timings = {"hyperparameters_s": t1 - t0,
                        "mode_s": time.perf_counter() - t1}
        return self

    # -- prediction (R&W Algorithm 3.4) ----------------------------------
    def predict_latent(self, Xs, batch=1500):
        """Predictive mean (N, C) and covariance (N, C, C) of the latent fields."""
        Xs = np.atleast_2d(np.asarray(Xs, dtype=float))
        st, C = self.state_, self.C_
        R = self.Y_ - st.pi
        means, covs = [], []
        for start in range(0, Xs.shape[0], batch):
            Xb = Xs[start:start + batch]
            Ks = self.kernel(self.X_, Xb)                      # (n, N)
            kss = self.kernel.diag(Xb)                         # (N,)
            mean = (R @ Ks).T                                  # (N, C)
            B = np.einsum("cij,jn->cin", st.E, Ks)             # (C, n, N)
            S = np.stack([cho_solve((st.L_M, True), B[c]) for c in range(C)])
            cov = np.einsum("cin,din->ncd", B, S)              # b_c^T (MM^T)^-1 b_d
            diag = kss[:, None] - np.einsum("in,cin->nc", Ks, B)
            idx = np.arange(C)
            cov[:, idx, idx] += diag
            means.append(mean)
            covs.append(cov)
        return np.concatenate(means), np.concatenate(covs)

    def predict_proba(self, Xs, method="mc", n_samples=2000, seed=0, batch=500):
        """Class probabilities; ``mc`` averages the softmax over the latent
        predictive distribution (R&W Algorithm 3.4), ``plugin`` applies the
        softmax to the predictive mean. Monte Carlo uses common random
        numbers across locations, so probability maps are smooth."""
        mean, cov = self.predict_latent(Xs)
        if method == "plugin":
            return softmax(mean, axis=1)
        C = self.C_
        eps = np.random.default_rng(seed).standard_normal((n_samples, 1, C))
        out = np.empty_like(mean)
        for start in range(0, mean.shape[0], batch):
            m = mean[start:start + batch]
            Lc = np.linalg.cholesky(cov[start:start + batch] + 1e-10 * np.eye(C))
            fs = m[None] + np.einsum("ncd,sxd->snc", Lc, eps)
            out[start:start + batch] = softmax(fs, axis=2).mean(axis=0)
        return out
