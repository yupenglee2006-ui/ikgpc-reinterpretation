"""Shared figure style (Springer single-column width ~ 12 cm)."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

FIGDIR = Path(__file__).resolve().parents[2] / "results" / "figures"

CLASS_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def setup():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "savefig.dpi": 300,
        "figure.dpi": 110,
    })


def save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "eps", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
