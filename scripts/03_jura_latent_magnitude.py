"""Test of Eq. (20): IK range violations concentrate where the GPC latent
magnitude is large, i.e. where one class dominates.

Uses the grid predictions cached by 02_jura_estimation.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"


def main(n_bins=8):
    d = np.load(ROOT / "results" / "jura_grid_predictions.npz")
    ins = d["inside"]
    f = d["f_mean"][ins]                    # (N, C) GPC latent predictive mean
    P_ik = d["P_ik0"][ins]                  # IK, shared kernel, mean 1/C
    P_gpc = d["P_gpc"][ins]
    mag = np.max(np.abs(f), axis=1)
    exc = np.max(np.maximum(P_ik - 1, 0) + np.maximum(-P_ik, 0), axis=1)
    tv = 0.5 * np.abs(P_ik - P_gpc).sum(axis=1)
    edges = np.quantile(mag, np.linspace(0, 1, n_bins + 1))
    b = np.clip(np.digitize(mag, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for i in range(n_bins):
        s = b == i
        rows.append({"bin": i + 1, "latent_mag_lo": edges[i], "latent_mag_hi": edges[i + 1],
                     "latent_mag_median": float(np.median(mag[s])), "n_nodes": int(s.sum()),
                     "ik_excursion_gt_0.01_pct": 100 * float(np.mean(exc[s] > 0.01)),
                     "ik_mean_excursion": float(exc[s].mean()),
                     "mean_tv_distance_ik_gpc": float(tv[s].mean())})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "latent_magnitude_bins.csv", index=False)
    rho, pval = spearmanr(mag, exc)
    rho_tv, _ = spearmanr(mag, tv)
    out = {"spearman_latent_mag_vs_ik_excursion": float(rho), "p_value": float(pval),
           "spearman_latent_mag_vs_tv_distance": float(rho_tv),
           "share_of_violating_nodes_in_top_half_of_magnitude":
               float(np.mean(mag[exc > 0.01] > np.median(mag)))}
    (TAB / "latent_magnitude_summary.json").write_text(json.dumps(out, indent=2))
    print(df.round(3).to_string(index=False)); print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
