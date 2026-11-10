"""Fig. 1: one-dimensional two-class example.

Layout follows the originally submitted figure: one panel per method.
(a) Simple IK (mean 1/2) of the class-1 indicator; the parts of the curve
    outside [0, 1] are filled and labelled.
(b) Softmax GPC with two kinds of conditioning: its zero-temperature limit
    (hard: the labels are honoured exactly) and the model with latent
    variance 4 (soft: stems join each datum to the GPC probability at the
    same location).

Also reported: the indicator nugget implied by the softmax model, and the
exact evidence as a function of the latent variance (Monte Carlo).

The link to the linearization (|f_1| > 1 wherever IK leaves [0, 1]) is
reported in results/tables/synthetic_1d.json and in the text.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.special import ndtr, log_expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ikgpc import RBF, IndicatorKriging, SoftmaxGPC  # noqa: E402
from ikgpc.plotting import setup, save  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Opaque colours only: the EPS backend drops transparency.
C_IK = "#C44E52"
C_HARD = "#4C72B0"                            # zero-temperature limit (hard)
C_SOFT = "#DD8452"                            # latent variance 4 (soft)
C_BAND = "#F6E6E6"      # region outside [0, 1]
C_EXC = "#E3A3A5"       # IK excursion beyond 0 or 1
YLIM = (-0.3, 1.3)


def runs(x, mask):
    """[start, end] of each contiguous run of True in mask."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return [[float(x[a]), float(x[b - 1])] for a, b in zip(edges[::2], edges[1::2])]


def rbf_corr(a, b, lengthscale):
    return np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / lengthscale ** 2)


def hard_limit_2class(x, I1, xg, lengthscale, n_draws=8_000_000, batch=200_000, seed=0):
    """Zero-temperature limit of two-class softmax GPC.

    As tau -> 0 the label is 1 where d = f_1 - f_2 > 0, so the labels are hard
    inequality constraints on d at the data. The latent scale cancels; only the
    correlation function matters. Latent values at the data are drawn from the
    prior and kept if they agree with every label (exact for this small example;
    use a Gibbs or elliptical slice sampler for larger data sets). The probability
    at x* is the average over kept draws of P(d(x*) > 0 | d at the data).
    Returns (probability, number of kept draws).
    """
    rho = lambda a, b: rbf_corr(a, b, lengthscale)
    K = rho(x, x) + 1e-8 * np.eye(x.size)
    L = np.linalg.cholesky(K)
    A = np.linalg.solve(K, rho(xg, x).T).T                     # simple-kriging weights
    sd = np.sqrt(np.clip(1.0 - np.einsum("ij,ij->i", A, rho(xg, x)), 0, None)) + 1e-12
    rng = np.random.default_rng(seed)
    acc, kept = np.zeros(xg.size), 0
    for _ in range(n_draws // batch):
        D = rng.standard_normal((batch, x.size)) @ L.T
        D = D[np.all((D > 0) == (I1 == 1), axis=1)]
        acc += ndtr((D @ A.T) / sd).sum(0)
        kept += len(D)
    return acc / kept, kept


def implied_nugget_share(variance, C, n=1_000_000, seed=0):
    """Indicator nugget of the softmax model, E[p(1-p)], as a share of the sill,
    for i.i.d. zero-mean latent values with the given variance."""
    F = np.random.default_rng(seed).normal(0, np.sqrt(variance), (n, C))
    P = np.exp(F - F.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    return float((P * (1 - P)).mean() / ((1 / C) * (1 - 1 / C)))


def exact_log_evidence_2class(x, I1, variance, lengthscale, n=4_000_000, batch=200_000, seed=0):
    """log p(labels | latent variance) for two-class softmax GPC, by Monte Carlo over the prior."""
    L = np.linalg.cholesky(rbf_corr(x, x, lengthscale) + 1e-8 * np.eye(x.size))
    sgn = np.where(I1 == 1, 1.0, -1.0)
    rng = np.random.default_rng(seed)
    tot = 0.0
    for _ in range(n // batch):
        D = np.sqrt(2 * variance) * (rng.standard_normal((batch, x.size)) @ L.T)   # d = f1 - f2
        tot += np.exp(log_expit(D * sgn).sum(1)).sum()
    return float(np.log(tot / n))


def frame(ax, title):
    ax.axhspan(1, YLIM[1], color=C_BAND, lw=0, zorder=0)
    ax.axhspan(YLIM[0], 0, color=C_BAND, lw=0, zorder=0)
    for y in (0, 1):
        ax.axhline(y, color="0.55", lw=0.5, zorder=1)
    ax.axvline(5, color="0.6", lw=0.6, ls="--", zorder=1)
    ax.set(xlim=(0, 10), ylim=YLIM, ylabel="$p_1(x)$")
    ax.set_title(title, loc="left")


def main():
    setup()
    rng = np.random.default_rng(42)
    x = np.sort(rng.uniform(0, 10, 15))
    z = (x >= 5.0).astype(int)                 # class 1 (index 0) left of x = 5
    I1 = 1 - z                                 # class-1 indicator data
    X = x[:, None]
    xs = np.linspace(0, 10, 801)[:, None]
    xg = xs[:, 0]

    # Simple IK, mean 1/2, unit sill, nugget 0.1
    kern = RBF(variance=1.0, lengthscales=(2.0,))
    ik = IndicatorKriging(kern, shared=True, means="uniform", nugget=0.1, fit=False).fit(X, z, 2)
    p_ik = ik.predict(xs)[:, 0]

    # Separable data drive the evidence-optimal latent variance upward without
    # bound, so this illustration uses a fixed latent kernel.
    gpc = SoftmaxGPC(RBF(variance=4.0, lengthscales=(2.0,)), optimize=None).fit(X, z, 2)
    mean, _ = gpc.predict_latent(xs)
    p_gpc = gpc.predict_proba(xs, n_samples=4000)[:, 0]
    p_gpc_data = gpc.predict_proba(X, n_samples=4000)[:, 0]
    f1 = mean[:, 0]

    p_hard, n_kept = hard_limit_2class(x, I1, xg, lengthscale=2.0)
    p_hard_data = hard_limit_2class(x, I1, x, lengthscale=2.0)[0]

    above, below = p_ik > 1, p_ik < 0
    viol = above | below

    fig, axes = plt.subplots(2, 1, figsize=(4.75, 4.4), sharex=True, constrained_layout=True)

    # ---- (a) Indicator kriging --------------------------------------------
    ax = axes[0]
    frame(ax, "(a) Indicator kriging")
    ax.fill_between(xg, 1, p_ik, where=above, color=C_EXC, lw=0, zorder=2)
    ax.fill_between(xg, 0, p_ik, where=below, color=C_EXC, lw=0, zorder=2)
    ax.plot(xg, p_ik, color=C_IK, zorder=3)
    ax.scatter(x, I1, s=14, color="k", zorder=5)

    i_max, i_min = np.argmax(p_ik), np.argmin(p_ik)
    arrow = dict(arrowstyle="-|>", color=C_IK, lw=0.6, mutation_scale=6, shrinkA=1, shrinkB=1)
    ax.annotate(f"$>1$, max {p_ik[i_max]:.2f}", xy=(xg[i_max], p_ik[i_max]),
                xytext=(0.15, 1.2), ha="left", va="center", fontsize=7.5,
                color=C_IK, arrowprops=arrow)
    ax.annotate(f"$<0$, min {p_ik[i_min]:.2f}".replace("-", "\u2212"),
                xy=(xg[i_min], p_ik[i_min]), xytext=(9.85, -0.2), ha="right",
                va="center", fontsize=7.5, color=C_IK, arrowprops=arrow)
    ax.text(4.85, 0.12, "class 1", ha="right", fontsize=7, color="0.4", style="italic")
    ax.text(5.15, 0.12, "class 2", ha="left", fontsize=7, color="0.4", style="italic")

    ax.legend(handles=[
        Line2D([], [], color=C_IK, label="IK (simple, mean 1/2)"),
        Line2D([], [], ls="", marker="o", ms=4, color="k", label="Data, $I_1(x_i)$"),
        Patch(color=C_EXC, label="IK outside $[0,1]$"),
    ], loc="lower left", bbox_to_anchor=(0.0, 0.2), frameon=False, fontsize=7)

    # ---- (b) Softmax GPC ---------------------------------------------------
    ax = axes[1]
    frame(ax, "(b) Softmax GPC: hard and soft conditioning")
    ax.plot(xg, p_hard, color=C_HARD, zorder=4)
    ax.vlines(x, I1, p_gpc_data, color=C_SOFT, lw=0.6, zorder=3)
    ax.plot(xg, p_gpc, color=C_SOFT, ls="-.", zorder=4)
    ax.scatter(x, p_gpc_data, s=13, marker="D", facecolor=C_SOFT,
               edgecolor="k", linewidth=0.4, zorder=6)
    ax.scatter(x, I1, s=14, color="k", zorder=5)
    # Under hard conditioning the class boundary lies between the nearest data of the
    # two classes; across that gap the probability falls almost linearly.
    xa, xb = x[I1 == 1].max(), x[I1 == 0].min()
    ax.annotate("", xy=(xa, 1.12), xytext=(xb, 1.12),
                arrowprops=dict(arrowstyle="<->", color=C_HARD, lw=0.6, mutation_scale=6,
                                shrinkA=0, shrinkB=0))
    ax.text((xa + xb) / 2, 1.17, "boundary between nearest data", ha="center", va="bottom",
            fontsize=7, color=C_HARD, bbox=dict(facecolor=C_BAND, edgecolor="none", pad=0.6))
    ax.set_xlabel("$x$")

    ax.legend(handles=[
        Line2D([], [], color=C_HARD, label="Zero-temperature limit: hard"),
        Line2D([], [], color=C_SOFT, ls="-.", label="Latent variance 4: soft"),
        Line2D([], [], color=C_SOFT, lw=0.6, ls="", marker="D", ms=3.5, mec="k", mew=0.4,
               label="Latent variance 4, at data"),
        Line2D([], [], ls="", marker="o", ms=4, color="k", label="Data, $I_1(x_i)$"),
    ], loc="lower left", bbox_to_anchor=(0.0, 0.2), frameon=False, fontsize=7)

    save(fig, "figure1")

    out = {
        "n": int(x.size),
        "gpc_variance": float(gpc.kernel.variance),
        "ik_range_violation_pct_of_domain": 100 * float(viol.mean()),
        "ik_min": float(p_ik.min()), "ik_max": float(p_ik.max()),
        "ik_above_1_intervals": runs(xg, above),
        "ik_below_0_intervals": runs(xg, below),
        "share_of_ik_violations_with_abs_f_gt_1": float((viol & (np.abs(f1) > 1)).sum() / max(viol.sum(), 1)),
        "min_abs_f_where_ik_violates": float(np.abs(f1[viol]).min()) if viol.any() else None,
        "gpc_min": float(p_gpc.min()), "gpc_max": float(p_gpc.max()),
        "gpc_at_class1_data": [float(p_gpc_data[I1 == 1].min()), float(p_gpc_data[I1 == 1].max())],
        "gpc_at_class2_data": [float(p_gpc_data[I1 == 0].min()), float(p_gpc_data[I1 == 0].max())],
        "hard_limit_kept_draws": int(n_kept),
        "hard_limit_max_abs_error_at_data": float(np.abs(p_hard_data - I1).max()),
        "hard_limit_min": float(p_hard.min()), "hard_limit_max": float(p_hard.max()),
        "hard_limit_at_x5": float(p_hard[np.argmin(np.abs(xg - 5.0))]),
        "implied_nugget_share_var4_C2": implied_nugget_share(4.0, 2),
        "implied_nugget_share_jura_var95.8_C5": implied_nugget_share(95.8, 5),   # Jura fit, Sect. 6.2
        "exact_log_evidence_by_latent_variance": {
            str(v): exact_log_evidence_2class(x, I1, v, 2.0) for v in (1, 4, 16, 64, 256, 1024)},
        "exact_log_evidence_zero_temperature": float(np.log(n_kept / 8_000_000)),
    }
    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "tables" / "synthetic_1d.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
