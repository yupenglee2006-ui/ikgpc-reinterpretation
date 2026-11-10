"""Ten-fold cross-validation on Jura (replaces the training Brier score of
the original submission). Hyperparameters are refitted in every fold.
IK estimates are clipped to [0, 1] and renormalized before scoring, as in
practice; GPC is scored both as the plug-in softmax and as the Monte Carlo
predictive average."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ikgpc import IndicatorKriging, Matern12, Matern32, SoftmaxGPC, metrics  # noqa: E402
from ikgpc.jura import load_jura  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"


def main(n_folds=10, seed=2026):
    X, z = load_jura()
    C = 5
    folds = np.random.default_rng(seed).permutation(len(z)) % n_folds
    scores = []
    t0 = time.perf_counter()
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        gpc = SoftmaxGPC(Matern32(variance=50.0, lengthscales=(0.5,)), optimize="isotropic").fit(X[tr], z[tr], C)
        preds = {"Softmax GPC, plug-in": gpc.predict_proba(X[te], method="plugin"),
                 "Softmax GPC, predictive average": gpc.predict_proba(X[te], method="mc", n_samples=2000)}
        for name, ik in [("IK, shared kernel, mean 1/C", IndicatorKriging(Matern12(), shared=True, means="uniform")),
                         ("IK, shared kernel, class proportions", IndicatorKriging(Matern12(), shared=True, means="proportions")),
                         ("IK, per-class kernels, class proportions", IndicatorKriging(Matern12(), shared=False, means="proportions"))]:
            raw = ik.fit(X[tr], z[tr], C).predict(X[te])
            preds[name] = metrics.clip_renormalize(raw)
        for name, P in preds.items():
            scores.append({"fold": f, "method": name, "brier": metrics.brier(P, z[te]),
                           "accuracy": float(np.mean(P.argmax(axis=1) == z[te]))})
        print(f"fold {f} done", flush=True)
    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(scores)
    df.to_csv(TAB / "crossvalidation_folds.csv", index=False)
    agg = df.groupby("method", sort=False).agg(brier_mean=("brier", "mean"), brier_sd=("brier", "std"),
                                               accuracy_mean=("accuracy", "mean"),
                                               accuracy_sd=("accuracy", "std")).reset_index()
    agg.to_csv(TAB / "crossvalidation_summary.csv", index=False)
    p0 = np.bincount(z, minlength=C) / len(z)
    out = {"n_folds": n_folds, "seed": seed, "time_s": elapsed,
           "null_model_brier": metrics.brier(np.tile(p0, (len(z), 1)), z)}
    (TAB / "crossvalidation_meta.json").write_text(json.dumps(out, indent=2))
    print(agg.round(4).to_string(index=False)); print(out)


if __name__ == "__main__":
    main()
