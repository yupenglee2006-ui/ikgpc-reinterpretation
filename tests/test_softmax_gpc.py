"""Check the O(Cn^3) Laplace implementation against dense O((Cn)^3) algebra."""
import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.special import softmax

from ikgpc import Matern32, SoftmaxGPC, laplace_mode


@pytest.fixture(scope="module")
def problem():
    rng = np.random.default_rng(1)
    n, C = 14, 3
    X = rng.uniform(0, 4, size=(n, 2))
    z = rng.integers(0, C, size=n)
    Y = np.zeros((C, n)); Y[z, np.arange(n)] = 1
    k = Matern32(variance=2.0, lengthscales=(1.3,))
    K = k(X)
    return X, z, Y, K, k, C, n


def dense_W(pi):
    C, n = pi.shape
    D = np.diag(pi.ravel())
    P = np.vstack([np.diag(pi[c]) for c in range(C)])
    return D - P @ P.T


def test_mode_is_stationary(problem):
    X, z, Y, K, k, C, n = problem
    st = laplace_mode(K, Y)
    # gradient of Psi: -K^{-1} f + y - pi = 0  <=>  f = K (y - pi)
    assert np.allclose(st.f, (Y - st.pi) @ K, atol=1e-8)


def test_mode_is_centred_with_shared_kernel(problem):
    X, z, Y, K, k, C, n = problem
    st = laplace_mode(K, Y)
    assert np.max(np.abs(st.f.sum(axis=0))) < 1e-8


def test_negative_log_posterior_is_strictly_convex(problem):
    X, z, Y, K, k, C, n = problem
    st = laplace_mode(K, Y)
    W = dense_W(st.pi)
    H = np.linalg.inv(block_diag(*[K] * C)) + W
    assert np.min(np.linalg.eigvalsh(0.5 * (H + H.T))) > 0
    assert np.min(np.linalg.eigvalsh(W)) > -1e-12      # W is PSD


def test_log_marginal_likelihood_matches_dense(problem):
    X, z, Y, K, k, C, n = problem
    st = laplace_mode(K, Y)
    Kf = block_diag(*[K] * C)
    W = dense_W(st.pi)
    f = st.f.ravel()
    psi = -0.5 * f @ np.linalg.solve(Kf, f) + np.sum(Y * st.f) \
        - np.sum(np.log(np.exp(st.f).sum(axis=0)))
    _, logdet = np.linalg.slogdet(np.eye(C * n) + Kf @ W)
    assert np.isclose(st.log_q, psi - 0.5 * logdet, rtol=1e-8)


def test_predictive_covariance_matches_dense(problem):
    X, z, Y, K, k, C, n = problem
    gpc = SoftmaxGPC(k, optimize=None).fit(X, z, C)
    Xs = np.array([[1.0, 2.0], [3.5, 0.2], [10.0, 10.0]])
    mean, cov = gpc.predict_latent(Xs)
    st = gpc.state_
    W = dense_W(st.pi)
    Kf = block_diag(*[K] * C)
    for j, xs in enumerate(Xs):
        ks = k(X, xs[None])[:, 0]
        Q = block_diag(*[ks[:, None]] * C)               # (Cn, C)
        A = np.linalg.solve(np.eye(C * n) + W @ Kf, W)
        dense = np.eye(C) * k.variance - Q.T @ A @ Q
        assert np.allclose(cov[j], dense, atol=1e-9)
        assert np.allclose(mean[j], ((Y - st.pi) @ ks), atol=1e-12)
    assert np.allclose(mean.sum(axis=1), 0.0, atol=1e-8)   # centred predictions


def test_probabilities_are_on_the_simplex(problem):
    X, z, Y, K, k, C, n = problem
    gpc = SoftmaxGPC(k, optimize=None).fit(X, z, C)
    P = gpc.predict_proba(np.random.default_rng(0).uniform(-2, 6, (50, 2)), n_samples=200)
    assert np.all((P >= 0) & (P <= 1))
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12)


def test_hyperparameter_fit_increases_evidence(problem):
    X, z, Y, K, k, C, n = problem
    gpc0 = SoftmaxGPC(k, optimize=None).fit(X, z, C)
    gpc1 = SoftmaxGPC(k, optimize="isotropic").fit(X, z, C)
    assert gpc1.state_.log_q >= gpc0.state_.log_q - 1e-8
