"""Jura data set helpers: loading, prediction grid and study-area mask."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "jura.csv"

# Class names as used in the manuscript (rock codes 1-5, following the coding
# of Goovaerts 1997 as distributed with the gstat R package).
ROCK_NAMES = ["Argovian", "Kimmeridgian", "Sequanian", "Portlandian", "Quaternary"]


def load_jura(path=DATA):
    """Return coordinates (n, 2) in km and zero-based rock labels (n,)."""
    df = pd.read_csv(path)
    X = df[["X", "Y"]].to_numpy(float)
    z = df["Rock"].to_numpy(int) - 1
    return X, z


def make_grid(X, n=80, pad=0.1):
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = hi - lo
    gx = np.linspace(lo[0] - pad * span[0], hi[0] + pad * span[0], n)
    gy = np.linspace(lo[1] - pad * span[1], hi[1] + pad * span[1], n)
    GX, GY = np.meshgrid(gx, gy)
    return GX, GY, np.c_[GX.ravel(), GY.ravel()]


def hull_mask(X, Xg):
    """True for grid nodes inside the convex hull of the data."""
    return Delaunay(X).find_simplex(Xg) >= 0
