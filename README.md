# ikgpc — Indicator kriging as linearized Gaussian process classification

[![tests](https://github.com/yupenglee2006-ui/ikgpc-reinterpretation/actions/workflows/ci.yml/badge.svg)](https://github.com/yupenglee2006-ui/ikgpc-reinterpretation/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24080372.svg)](https://doi.org/10.5281/zenodo.24080372)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Reproduction code for

> Li Y, You J. *Reinterpreting Indicator Kriging from a Machine Learning
> Perspective in the Context of Gaussian Process Classification.*
> Mathematical Geosciences (revised manuscript).

The paper shows that simple indicator kriging (IK) with mean 1/C is the
first-order Taylor approximation of multi-class Gaussian process
classification (GPC) with a softmax link. This repository implements both
methods and regenerates every figure and table of the paper.

## Quick start

```bash
git clone https://github.com/yupenglee2006-ui/ikgpc-reinterpretation.git
cd ikgpc-reinterpretation
# or download the archived release from https://doi.org/10.5281/zenodo.24080372
python -m pip install -e ".[test]"      # numpy, scipy, pandas, matplotlib, pytest
python -m pytest -q                     # 11 tests, < 1 s
python scripts/run_all.py               # all figures and tables, about 20 min on one core
                                        # (about 7 min without the controlled simulation)
```

Generated results are not included in the repository or the archive (they are large and fully
deterministic); they are recreated by `run_all.py`. Outputs are written to `results/figures` (PDF, EPS, PNG) and
`results/tables` (CSV, JSON, LaTeX). `results/run_record.json` records the
software versions, the CPU and the wall-clock time of each step.

## What each script reproduces

|Script|Paper|Content|
|-|-|-|
|`01_synthetic_1d.py`|Fig. 1, Sect. 6.1|One-dimensional two-class example|
|`02_jura_estimation.py`|Table 1, Figs. 2-3, run times|GPC and three IK variants on the Jura grid|
|`03_jura_latent_magnitude.py`|Fig. 4b-c|IK range violations against GPC latent magnitude|
|`04_jura_anisotropy.py`|Fig. 4d|Evidence profile over the azimuth of anisotropy|
|`05_jura_crossvalidation.py`|Table 2|Tenfold cross-validation|
|`06_figure4.py`|Fig. 4|Assembles the four test panels|
|`07_controlled_simulation.py`|Fig. 5, Sect. 6.3|Controlled simulation with a known latent field|

## Main results

|Prediction of the theory|Result on the Jura data|
|-|-|
|With one kernel for all classes, IK estimates sum to one (Eq. 19)|Raw sums equal 1 to within 1.1e-15 at all 2674 nodes|
|With per-class kernels, closure fails|Sums off by more than 0.01 at 52% of nodes (max 0.29)|
|IK leaves [0, 1] where the latent magnitude is large (Eq. 18)|Excursions > 0.01 rise from 31% to 73% of nodes across latent-magnitude octiles|
|Softmax GPC stays on the simplex|No violations; sums equal 1 to machine precision|
|Kernel anisotropy can be fitted by the evidence|Major axis N65°E, ratio 3.0, +23.9 log-evidence over isotropic|
|In a known latent field, violations follow the latent magnitude (Sect. 6.3)|IK outside [0, 1] at 0 → 31% of nodes as the latent magnitude grows from 0.8 to 4.6; GPC never|

In tenfold cross-validation, IK and the GPC plug-in estimate both classify
80% of held-out samples correctly (Brier scores 0.27 and 0.29).

## The package

```
src/ikgpc/
  softmax_gpc.py         Laplace approximation for softmax GPC with a shared kernel
                         (Rasmussen and Williams 2006, Algorithms 3.3-3.4)
  indicator_kriging.py   Simple IK, shared or per-class covariances fitted by ML
  linearization.py       Softmax, its first-order expansion and the error term of Eq. (18)
  kernels.py             Matern 1/2, Matern 3/2 and squared-exponential kernels, anisotropic
  metrics.py             Range, closure and Brier diagnostics
  jura.py, plotting.py   Data loading, grid, figure style
tests/                   Checks against dense linear algebra and the paper's identities
```

Minimal use:

```python
from ikgpc import Matern32, SoftmaxGPC, IndicatorKriging, Matern12
from ikgpc.jura import load_jura, make_grid

X, z = load_jura()
_, _, Xg = make_grid(X)
gpc = SoftmaxGPC(Matern32(variance=4.0, lengthscales=(0.5,)), optimize="isotropic").fit(X, z)
P_gpc = gpc.predict_proba(Xg, method="plugin")          # (N, C), rows on the simplex
ik = IndicatorKriging(Matern12(), shared=True, means="uniform").fit(X, z)
P_ik = ik.predict(Xg)                                   # raw estimates, rows sum to 1
```

## What the tests check

* The Newton iteration reaches a stationary point, f = K(y - pi).
* With a shared kernel the mode and the predictive means are centred over classes (Sect. 3.3).
* The negative log posterior is strictly convex (Sect. 3.3).
* The O(Cn^3) evidence and predictive covariance equal their dense O((Cn)^3) counterparts.
* The linearized GPC predictor equals simple IK with mean 1/C (Eqs. 16-17).
* Closure is exact with a shared kernel and fails with per-class kernels (Eq. 19).
* The remainder after the second-order term of Eq. (18) is O(f^3).
* The zero-temperature softmax is the arg-max rule (Eq. 21).

## Notes on the method

* GPC probabilities are reported as the plug-in softmax of the latent
predictive mean, the quantity linearized in Sect. 4. The Monte Carlo
predictive average is also available (`method="mc"`); with the large latent
variance preferred by the evidence, the Laplace approximation makes it
underconfident (Sect. 6.2).
* Kernel families were chosen per method: Matern 3/2 for the GPC latent fields
by evidence, and the exponential model for IK, the usual choice for
indicator variograms.
* Results are deterministic; reruns reproduce the published numbers exactly.

## Data

`data/jura.csv`: the Jura data set (Atteia et al. 1994; Goovaerts 1997), as
distributed with the `gstat` R package. See [data/README.md](data/README.md)
for provenance and citation.

## Citation

Please cite the article and this software; see [CITATION.cff](CITATION.cff).

> Li Y, You J (2026) ikgpc: reproduction code for "Reinterpreting Indicator
> Kriging from a Machine Learning Perspective in the Context of Gaussian
> Process Classification" (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.24080372

## License

Code: MIT. Data: see `data/README.md`.

