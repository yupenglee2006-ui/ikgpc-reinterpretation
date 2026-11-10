"""Fig. 4: tests of the predictions of Sects. 4.3 and 6 on the Jura data."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ikgpc.jura import load_jura  # noqa: E402
from ikgpc.plotting import save, setup  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"


def main():
    setup()
    d = np.load(ROOT / "results" / "jura_grid_predictions.npz")
    ins, GX, GY = d["inside"], d["GX"], d["GY"]
    s_class = d["P_ik2"][ins].sum(axis=1)
    bins = pd.read_csv(TAB / "latent_magnitude_bins.csv")
    prof = pd.read_csv(TAB / "anisotropy_profile.csv")
    an = json.loads((TAB / "anisotropy_summary.json").read_text())
    X, _ = load_jura()

    fig, axes = plt.subplots(2, 2, figsize=(4.75, 4.3), constrained_layout=True)
    ax = axes[0, 0]
    edges = np.linspace(0.85, 1.30, 46)
    ax.hist(s_class, bins=edges, color="#DD8452", label="Per-class kernels")
    ax.axvline(1.0, color="#4C72B0", lw=2.0, label="Shared kernel")
    ax.set(xlabel="Sum of raw IK estimates over classes", ylabel="Grid nodes")
    ax.set_title("(a) Closure", loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=6.5)

    ax = axes[0, 1]
    mag = np.max(np.abs(d["f_mean"]), axis=1)
    P = d["P_ik0"]
    exc = np.max(np.maximum(P - 1, 0) + np.maximum(-P, 0), axis=1)
    im = ax.pcolormesh(GX, GY, np.where(ins, mag, np.nan).reshape(GX.shape), cmap="Greys",
                       shading="auto", rasterized=True)
    sel = ins & (exc > 0.01)
    ax.scatter(d["Xg"][sel, 0], d["Xg"][sel, 1], s=0.6, c="#C44E52", lw=0)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("Latent magnitude", fontsize=7); cb.ax.tick_params(labelsize=6.5)
    ax.set_title("(b) IK excursions > 0.01", loc="left")

    ax = axes[1, 0]
    ax.bar(bins["bin"], bins["ik_excursion_gt_0.01_pct"], color="#C44E52", width=0.75)
    ax.set_xticks(bins["bin"])
    ax.set_xticklabels([f"{v:.1f}" for v in bins["latent_mag_median"]], fontsize=6.5)
    ax.set(xlabel="Latent magnitude (bin median)", ylabel="IK nodes with excursion (%)", ylim=(0, 100))
    ax.set_title("(c) Excursions by latent magnitude", loc="left")

    ax = axes[1, 1]
    ax.plot(prof["azimuth_deg"], prof["log_evidence"], "-o", ms=2.0, color="0.25")
    ax.axhline(an["isotropic_log_evidence"], color="0.5", ls="--", lw=0.8, label="Isotropic")
    b = an["best"]
    ax.plot([b["azimuth_deg"]], [b["log_evidence"]], "o", color="#C44E52", ms=4.5,
            label=f"Maximum, N{b['azimuth_deg']:.0f}$^\\circ$E")
    ax.set(xlabel="Major-axis azimuth (degrees)", ylabel="Log evidence", xlim=(0, 180))
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_title("(d) Anisotropy", loc="left")
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.07), fontsize=6.5)
    save(fig, "figure4")


if __name__ == "__main__":
    main()
