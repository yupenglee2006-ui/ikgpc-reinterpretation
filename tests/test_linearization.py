"""Numerical checks of Sect. 4: Eqs. (17)-(20) and the closure result."""
import numpy as np
from scipy.special import softmax

from ikgpc import IndicatorKriging, Matern32, linearized_link, second_order_error, simple_ik_from_latent


def _data(seed=0, n=30, C=4):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 5, size=(n, 2))
    z = rng.integers(0, C, size=n)
    Y = np.zeros((C, n)); Y[z, np.arange(n)] = 1
    return X, z, Y, C


def test_linearized_gpc_equals_simple_ik_with_mean_one_over_C():
    X, z, Y, C = _data()
    k = Matern32(variance=1.0, lengthscales=(1.0,))
    K = k(X) + 1e-9 * np.eye(len(z))
    Xs = np.random.default_rng(1).uniform(-1, 6, size=(40, 2))
    Ks = k(X, Xs)
    p_lin = simple_ik_from_latent(K, Ks, Y)                       # Eq. (17)
    lam = np.linalg.solve(K, Ks)
    p_sk = 1.0 / C + (Y - 1.0 / C) @ lam                           # Eq. (18)
    assert np.allclose(p_lin, p_sk, atol=1e-10)


def test_closure_exact_with_shared_kernel_and_lost_without():
    X, z, Y, C = _data(2)
    Xs = np.random.default_rng(3).uniform(-1, 6, size=(200, 2))
    shared = IndicatorKriging(Matern32(), shared=True, means="proportions").fit(X, z, C)
    assert np.max(np.abs(shared.predict(Xs).sum(axis=1) - 1)) < 1e-12
    per_class = IndicatorKriging(Matern32(), shared=False, means="proportions").fit(X, z, C)
    assert np.max(np.abs(per_class.predict(Xs).sum(axis=1) - 1)) > 1e-4


def test_second_order_error_term():
    rng = np.random.default_rng(4)
    for C in (2, 3, 5):
        g = rng.normal(size=C); g -= g.mean()
        for scale in (1e-1, 1e-2):
            f = scale * g
            exact = softmax(f) - linearized_link(f)
            approx = second_order_error(f)
            # remainder is O(f^3)
            assert np.max(np.abs(exact - approx)) < 10 * scale ** 3


def test_zero_temperature_limit_is_argmax():
    f = np.array([0.3, -1.2, 0.9, 0.0])
    p = softmax(f / 1e-3)
    assert np.argmax(p) == 2 and p[2] > 1 - 1e-12
