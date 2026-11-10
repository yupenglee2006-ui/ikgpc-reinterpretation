"""Diagnostics reported in the paper."""
from __future__ import annotations

import numpy as np

__all__ = ["range_violation", "closure_deviation", "brier", "summarize", "clip_renormalize"]


def range_violation(P):
    """Boolean per row: any estimate outside [0, 1]. ``P`` has shape (N, C)."""
    return np.any((P < 0.0) | (P > 1.0), axis=1)


def closure_deviation(P):
    """|sum_k p_k - 1| per row."""
    return np.abs(P.sum(axis=1) - 1.0)


def brier(P, z):
    """Multi-class Brier score, mean over rows of sum_k (p_k - 1{z=k})^2."""
    Y = np.zeros_like(P)
    Y[np.arange(len(z)), np.asarray(z, dtype=int)] = 1.0
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def summarize(P, tol=0.01):
    dev = closure_deviation(P)
    exc = np.max(np.maximum(P - 1.0, 0.0) + np.maximum(-P, 0.0), axis=1)
    return {
        "range_violation_pct": 100.0 * float(np.mean(range_violation(P))),
        "range_excursion_gt_0.01_pct": 100.0 * float(np.mean(exc > 0.01)),
        "max_excursion": float(exc.max()),
        "closure_violation_pct": 100.0 * float(np.mean(dev > tol)),
        "max_closure_deviation": float(dev.max()),
        "min_estimate": float(P.min()),
        "max_estimate": float(P.max()),
    }


def clip_renormalize(P):
    Q = np.clip(P, 0.0, 1.0)
    s = Q.sum(axis=1, keepdims=True)
    return np.where(s > 0, Q / np.where(s > 0, s, 1.0), 1.0 / P.shape[1])
