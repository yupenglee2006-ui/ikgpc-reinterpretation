"""Stationary covariance kernels used in the paper.

All kernels are shared across classes: the softmax Gaussian process
classifier places independent Gaussian process priors with the *same*
covariance function on the C latent fields, and simple indicator kriging
uses one indicator covariance for all classes (Sect. 4.3 of the paper).

Coordinates follow the geological convention: column 0 is easting,
column 1 is northing, and anisotropy directions are azimuths in degrees
measured clockwise from north.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

__all__ = ["Matern12", "Matern32", "RBF"]


def _pairwise_scaled_distance(X1, X2, lengthscales, azimuth_deg):
    X1 = np.atleast_2d(np.asarray(X1, dtype=float))
    X2 = np.atleast_2d(np.asarray(X2, dtype=float))
    ls = np.atleast_1d(np.asarray(lengthscales, dtype=float))
    if X1.shape[1] == 1:
        D = (X1[:, None, 0] - X2[None, :, 0]) / ls[0]
        return np.abs(D)
    if ls.size == 1:
        diff = X1[:, None, :] - X2[None, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=-1)) / ls[0]
    az = np.deg2rad(azimuth_deg)
    u = np.array([np.sin(az), np.cos(az)])        # major-axis direction
    v = np.array([np.cos(az), -np.sin(az)])       # minor-axis direction
    A = np.stack([u / ls[0], v / ls[1]], axis=1)  # (2, 2) projection
    Y1, Y2 = X1 @ A, X2 @ A
    diff = Y1[:, None, :] - Y2[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


@dataclass(frozen=True)
class Matern32:
    """Matern covariance with smoothness 3/2.

    k(h) = variance * (1 + sqrt(3) r) * exp(-sqrt(3) r),  r = |h| / lengthscale.

    With two length-scales the distance is anisotropic: ``lengthscales[0]``
    applies along ``azimuth`` (major axis) and ``lengthscales[1]`` across it.
    """

    variance: float = 1.0
    lengthscales: tuple = (1.0,)
    azimuth: float = 0.0

    def __call__(self, X1, X2=None):
        X2 = X1 if X2 is None else X2
        r = _pairwise_scaled_distance(X1, X2, self.lengthscales, self.azimuth)
        s = np.sqrt(3.0) * r
        return self.variance * (1.0 + s) * np.exp(-s)

    def diag(self, X):
        return np.full(np.atleast_2d(X).shape[0], self.variance)

    def with_params(self, **kw):
        return replace(self, **kw)


@dataclass(frozen=True)
class RBF:
    """Squared-exponential covariance, k(h) = variance * exp(-r^2 / 2)."""

    variance: float = 1.0
    lengthscales: tuple = (1.0,)
    azimuth: float = 0.0

    def __call__(self, X1, X2=None):
        X2 = X1 if X2 is None else X2
        r = _pairwise_scaled_distance(X1, X2, self.lengthscales, self.azimuth)
        return self.variance * np.exp(-0.5 * r ** 2)

    def diag(self, X):
        return np.full(np.atleast_2d(X).shape[0], self.variance)

    def with_params(self, **kw):
        return replace(self, **kw)


@dataclass(frozen=True)
class Matern12:
    """Exponential covariance (Matern, smoothness 1/2), k(h) = variance * exp(-r).

    Its linear behaviour at the origin matches that of indicator covariances
    of random sets, which makes it the usual choice for indicator kriging.
    """

    variance: float = 1.0
    lengthscales: tuple = (1.0,)
    azimuth: float = 0.0

    def __call__(self, X1, X2=None):
        X2 = X1 if X2 is None else X2
        r = _pairwise_scaled_distance(X1, X2, self.lengthscales, self.azimuth)
        return self.variance * np.exp(-r)

    def diag(self, X):
        return np.full(np.atleast_2d(X).shape[0], self.variance)

    def with_params(self, **kw):
        return replace(self, **kw)
