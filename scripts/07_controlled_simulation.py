"""Controlled simulation with a known latent field (Sect. 6.3, Fig. 5).

The latent fields are drawn from a Gaussian process prior, the labels from the
softmax of the field, so the true class probabilities are known at every test
node.  Both estimators are given the family of the true kernel (Matern 3/2),
so that the comparison isolates the link: IK linearizes it, GPC does not.
The signal strength s scales the latent field; as s grows the classes become
locally more dominant.

Metrics, all against the true probabilities at the test nodes:
  * share of nodes with an estimate outside [0, 1] by more than 0.01;
  * mean excursion outside [0, 1];
  * mean absolute error and Brier score against the true probabilities;
  * classification accuracy against the most probable true class.

Outputs: results/tables/controlled_simulation.csv, controlled_simulation.json
and results/figures/figure5.{pdf,eps,png}.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ikgpc import IndicatorKriging, Matern32, SoftmaxGPC, metrics  # noqa: E402
from ikgpc.plotting import save, setup  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"

C, N = 5, 300
DOMAIN = np.array([[0.0, 6.0], [0.0, 5.0]])
SIGNALS = (0.5, 1.0, 1.5, 2.0, 3.0)


def _grid(n=24, margin=0.2):
    gx = np.linspace(DOMAIN[0, 0] + margin, DOMAIN[0, 1] - margin, n)
    gy = np.linspace(DOMAIN[1, 0] + margin, DOMAIN[1, 1] - margin, n)
    GX, GY = np.meshgrid(gx, gy)
    return GX, GY, np.c_[GX.ravel(), GY.ravel()]


def _one(seed, s, Xt, truth_kernel):
    """One replicate: simulate, fit both methods, score against the truth."""
    rng = np.random.default_rng(seed)
    X = np.c_[rng.uniform(*DOMAIN[0], N), rng.uniform(*DOMAIN[1], N)]
    Xall = np.vstack([X, Xt])
    Kall = truth_kernel(Xall) + 1e-8 * np.eye(len(Xall))
    Z = np.linalg.cholesky(Kall) @ rng.standard_normal((len(Xall), C))
    Z -= Z.mean(axis=1, keepdims=True)
    Z /= Z.std()
    f_data, f_test = s * Z[:N], s * Z[N:]
    p_data = softmax(f_data, axis=1)
    u = rng.random(N)
    z = (u[:, None] > np.cumsum(p_data, axis=1)).sum(axis=1).clip(0, C - 1)
    p_true = softmax(f_test, axis=1)

    ik = IndicatorKriging(Matern32(), shared=True, means="uniform").fit(X, z, C)
    P_ik = ik.predict(Xt)
    gpc = SoftmaxGPC(Matern32(variance=4.0, lengthscales=(0.5,)), optimize="isotropic").fit(X, z, C)
    P_gp = gpc.predict_proba(Xt, method="plugin")

    exc = np.max(np.maximum(P_ik - 1.0, 0.0) + np.maximum(-P_ik, 0.0), axis=1)
    lam = ik.weights(Xt)
    return {
        "latent": float(np.mean(np.max(np.abs(f_test), axis=1))),
        "ik_viol": 100.0 * float(np.mean(exc > 0.01)),
        "ik_exc": float(exc.mean()),
        "gpc_viol": 100.0 * float(np.mean(metrics.range_violation(P_gp))),
        "mae_ik": float(np.mean(np.abs(P_ik - p_true))),
        "mae_gpc": float(np.mean(np.abs(P_gp - p_true))),
        "brier_ik": float(np.mean(np.sum((P_ik - p_true) ** 2, axis=1))),
        "brier_gpc": float(np.mean(np.sum((P_gp - p_true) ** 2, axis=1))),
        "acc_ik": 100.0 * float(np.mean(P_ik.argmax(axis=1) == p_true.argmax(axis=1))),
        "acc_gpc": 100.0 * float(np.mean(P_gp.argmax(axis=1) == p_true.argmax(axis=1))),
        "neg_weights": 100.0 * float(np.mean(lam < 0.0)),
        "is_kernel": (float(ik.kernels_[0][0].variance), float(ik.kernels_[0][0].lengthscales[0])),
        "gpc_kernel": (float(gpc.kernel.variance), float(gpc.kernel.lengthscales[0])),
    }


def main(reps=20):
    setup()
    TAB.mkdir(parents=True, exist_ok=True)
    _, _, Xt = _grid()
    truth = Matern32(variance=1.0, lengthscales=(1.0,))
    rows = []
    t0 = time.perf_counter()
    for s in SIGNALS:
        rec = [_one(2026 * 100 + 10 * int(10 * s) + r, s, Xt, truth) for r in range(reps)]
        df = pd.DataFrame(rec)
        row = {"signal": s}
        for key in ("latent", "ik_viol", "ik_exc", "gpc_viol", "mae_ik", "mae_gpc",
                    "brier_ik", "brier_gpc", "acc_ik", "acc_gpc", "neg_weights"):
            row[f"{key}_mean"] = float(df[key].mean())
            row[f"{key}_sd"] = float(df[key].std(ddof=1))
        rows.append(row)
        print(f"s={s}: latent {row['latent_mean']:.2f}, IK outside {row['ik_viol_mean']:.1f}%, "
              f"MAE {row['mae_ik_mean']:.4f} vs {row['mae_gpc_mean']:.4f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "controlled_simulation.csv", index=False)
    meta = {
        "n_classes": C, "n_data": N, "n_test": int(Xt.shape[0]), "replicates": reps,
        "signal": list(SIGNALS),
        "truth_kernel": {"family": "Matern 3/2", "variance": 1.0, "lengthscales_km": [1.0]},
        "domain_km": DOMAIN.tolist(),
        "time_s": time.perf_counter() - t0,
    }
    (TAB / "controlled_simulation.json").write_text(json.dumps(meta, indent=2))

    # ---- Fig. 5 -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(4.75, 2.0), constrained_layout=True)
    ax = axes[0]
    ax.errorbar(out.latent_mean, out.ik_viol_mean, yerr=out.ik_viol_sd, fmt="o-", ms=3,
                lw=1.2, color="#C44E52", capsize=1.5, label="IK")
    ax.axhline(0.0, color="#4C72B0", lw=1.2, ls="--", label="GPC")
    ax.set(xlabel="Latent magnitude", ylabel="Nodes outside $[0,1]$ (%)", ylim=(-2, 100))
    ax.set_title("(a) Coherence", loc="left")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    ax.errorbar(out.latent_mean, out.brier_ik_mean, yerr=out.brier_ik_sd, fmt="o-", ms=3,
                lw=1.2, color="#C44E52", capsize=1.5, label="IK")
    ax.errorbar(out.latent_mean, out.brier_gpc_mean, yerr=out.brier_gpc_sd, fmt="s-", ms=2.6,
                lw=1.2, color="#4C72B0", capsize=1.5, label="GPC")
    ax.set(xlabel="Latent magnitude", ylabel="Brier score against truth")
    ax.set_title("(b) Probability error", loc="left")
    ax.legend(frameon=False, fontsize=7)
    save(fig, "figure5")
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main(reps=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
