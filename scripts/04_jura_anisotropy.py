"""Anisotropy by Laplace evidence: for each major-axis azimuth (5-degree
steps), fit the latent variance, minor length-scale and anisotropy ratio
(>= 1) of an anisotropic Matern 3/2 kernel and record the approximate log
marginal likelihood."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ikgpc import Matern32, SoftmaxGPC  # noqa: E402
from ikgpc.jura import load_jura  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"


def main(step=5):
    X, z = load_jura()
    C = 5
    iso = SoftmaxGPC(Matern32(variance=4.0, lengthscales=(0.5,)), optimize="isotropic").fit(X, z, C)
    rows, a_warm = [], None
    theta = np.log([iso.kernel.variance, iso.kernel.lengthscales[0], 1.2])
    t0 = time.perf_counter()
    for az in np.arange(0, 180, step):
        k = Matern32(variance=np.exp(theta[0]),
                     lengthscales=(np.exp(theta[1] + theta[2]), np.exp(theta[1])), azimuth=float(az))
        g = SoftmaxGPC(k, optimize="anisotropic")
        g._a_warm = a_warm
        g.fit(X, z, C)
        a_warm = g.state_.a
        theta = g.opt_result_.x
        rows.append({"azimuth_deg": float(az), "log_evidence": g.state_.log_q,
                     "variance": g.kernel.variance, "lengthscale_major_km": g.kernel.lengthscales[0],
                     "lengthscale_minor_km": g.kernel.lengthscales[1],
                     "ratio": g.kernel.lengthscales[0] / g.kernel.lengthscales[1]})
        print(rows[-1], flush=True)
    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "anisotropy_profile.csv", index=False)
    best = df.loc[df.log_evidence.idxmax()].to_dict()
    out = {"best": best, "isotropic_log_evidence": iso.state_.log_q,
           "log_evidence_gain_over_isotropic": best["log_evidence"] - iso.state_.log_q,
           "profile_range": float(df.log_evidence.max() - df.log_evidence.min()),
           "n_azimuths": int(len(df)), "scan_time_s": elapsed}
    (TAB / "anisotropy_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
