"""The softmax link, its first-order expansion and the second-order error.

These functions implement Eqs. (7), (12) and (20) of the paper and are used
to check the linearization result numerically.
"""
from __future__ import annotations

import numpy as np
from scipy.special import softmax as _softmax

__all__ = ["softmax", "linearized_link", "second_order_error", "simple_ik_from_latent"]


def softmax(f, axis=0, temperature=1.0):
    """Softmax over ``axis``; ``temperature`` is tau in Eq. (27)."""
    return _softmax(np.asarray(f) / temperature, axis=axis)


def linearized_link(f, axis=0):
    """First-order expansion about f = 0 under sum_k f_k = 0: p = (1 + f) / C."""
    f = np.asarray(f)
    C = f.shape[axis]
    return (1.0 + f) / C


def second_order_error(f, axis=0):
    """Leading error term of Eq. (20): (1/C)(f_k^2/2 - (1/C) sum_j f_j^2/2)."""
    f = np.asarray(f)
    C = f.shape[axis]
    return (0.5 * f ** 2 - 0.5 * np.mean(f ** 2, axis=axis, keepdims=True)) / C


def simple_ik_from_latent(K, Ks, Y):
    """Linearized GPC predictor of Eqs. (15)-(17): plug f = C*I - 1 into the
    zero-mean GP posterior mean, then map through the linearized link.
    Returns (C, N); equals simple IK with mean 1/C (Eq. 18)."""
    C = Y.shape[0]
    W = np.linalg.solve(K, Ks)
    f_hat = (C * Y - 1.0) @ W
    return (1.0 + f_hat) / C
