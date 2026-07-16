# Huber Loss Fusion Clustering

`huber-fusion-clustering` is a small Python package for robust subgroup
discovery in longitudinal data. It combines a Huber-loss objective with fusion
penalties and solves the resulting optimization problem with ADMM.

The project is designed for subject-specific trajectories. Each subject has a
coefficient vector, and subjects with similar fitted coefficients are fused
into the same cluster.

## Method

### Longitudinal representation

For subject $i=1,\ldots,n$, let $y_i\in\mathbb{R}^{T_i}$ denote the observed
trajectory, $X_i\in\mathbb{R}^{T_i\times d}$ the design matrix, and
$\beta_i\in\mathbb{R}^d$ a subject-specific coefficient vector. The working
model is $y_i\approx X_i\beta_i$. When only observation times are supplied,
the package constructs a B-spline design matrix; a prebuilt design matrix can
also be passed directly.

Within-subject dependence is represented by a working covariance matrix
$V_i$. Writing $V_i=L_iL_i^{\mathsf T}$, the solver pre-whitens each subject as
$\widetilde y_i=L_i^{-1}y_i$ and $\widetilde X_i=L_i^{-1}X_i$. Covariance
matrices may be supplied by the user. Otherwise, the package constructs a
residual-based AR(1) working covariance estimate.

### Robust fusion objective

The fitted coefficient vectors minimize

```math
\sum_{i=1}^{n}\sum_{t=1}^{T_i}\rho_c\!\left(\widetilde{y}_{it}-\widetilde{x}_{it}^{\mathsf{T}}\beta_i\right)+\sum_{i\lt j}p_\lambda\!\left(\lVert\beta_i-\beta_j\rVert_2\right)
```

The Huber loss is quadratic for central residuals and linear in the tails:
$\rho_c(r)=r^2/2$ when $|r|\le c$, and
$\rho_c(r)=c|r|-c^2/2$ when $|r|\gt c$. Consequently, ordinary residuals retain
their usual quadratic contribution, whereas the influence of a large residual
grows only linearly.

Let $u_{ij}=\lVert\beta_i-\beta_j\rVert_2$. Two pairwise fusion penalties are
available:

- **L2 fusion:** $p_\lambda(u)=\lambda u$, which applies group soft-thresholding
  to coefficient differences.
- **MCP fusion:** $p_{\lambda,\gamma}(u)=\lambda u-u^2/(2\gamma)$ for
  $0\le u\le\gamma\lambda$, and $p_{\lambda,\gamma}(u)=\gamma\lambda^2/2$
  beyond that range. MCP reduces the shrinkage bias on well-separated
  coefficient vectors.

The implementation uses a complete fusion graph, so every pair of subjects is
connected by one penalty term. When a difference is shrunk to zero, the two
subjects share the same fitted trajectory representation. This allows the
number of groups to emerge from the fitted coefficients rather than being
specified in advance.

### ADMM optimization

Let $A$ be the incidence matrix of the complete subject graph and let $B$ stack
the subject-specific coefficient vectors by row. The solver introduces edge
variables $Z=AB$ and scaled dual variables $U$. Each ADMM iteration performs:

1. **Coefficient update.** The Huber term is approximated by iteratively
   reweighted least squares (IRLS). The resulting coupled linear system combines
   the subject-specific weighted normal equations with the graph Laplacian
   $A^{\mathsf T}A$ and is solved by conjugate gradients. Subjects may have
   different numbers of observations.
2. **Fusion update.** Each row of $AB+U$ is passed through either the group L2
   shrinkage operator or the group MCP proximal operator.
3. **Dual update.** The scaled dual variable is updated as
   $U\leftarrow U+AB-Z$.

Iterations stop when both the primal residual $AB-Z$ and the dual residual
$\rho A^{\mathsf T}(Z-Z_{\mathrm{previous}})$ fall below their configured
tolerances, or when the maximum number of ADMM iterations is reached.

### Cluster extraction

After optimization, subjects are connected when the Euclidean distance between
their fitted coefficient vectors is no greater than `tau_cluster`. By default,
connected components of this threshold graph define the cluster labels.
Clusters smaller than `min_cluster_size` are merged into the nearest sufficiently
large cluster in coefficient space.

### Selecting the fusion strength

The package evaluates every value in `lam_grid`. For each candidate $\lambda$,
it refits the model, extracts clusters, and computes a Calinski-Harabasz score
in subject-level coefficient space. Degenerate one-cluster and all-singleton
solutions are excluded from selection, and the candidate with the largest
valid score is returned. Candidate values can be evaluated in parallel through
`ch_n_jobs`.

## Features

- Huber-loss optimization with ADMM
- L2 and MCP fusion penalties
- Support for equal- and unequal-length trajectories
- Threshold-based cluster extraction
- Calinski-Harabasz tuning-parameter selection
- Reproducible synthetic contamination utilities
- High-level estimator and lower-level functional interfaces

## Installation

Clone the repository and install it from the project directory:

```bash
python -m pip install .
```

Python 3.10 or newer is required.

## Quick start

```python
import numpy as np

from huber_fusion_clustering import ADMMClusterConfig, HuberFusionClusterer

rng = np.random.default_rng(42)
t = np.linspace(0.0, 1.0, 10)

group_a = [0.5 + 1.2 * t + rng.normal(0.0, 0.08, t.size) for _ in range(30)]
group_b = [1.8 - 0.8 * t + rng.normal(0.0, 0.08, t.size) for _ in range(30)]
y = np.asarray(group_a + group_b)

config = ADMMClusterConfig(
    lam_grid=np.linspace(0.0001, 1.0, 30).tolist(),
    tau_cluster=0.20,
    min_cluster_size=2,
    max_admm=150,
    ch_n_jobs=1,
    verbose=0,
)

model = HuberFusionClusterer(config)
labels = model.fit_predict(y=y, t=t)

print("Selected lambda:", model.result_.best_lambda)
print("Cluster labels:", labels)
```

A complete runnable version is available in
[`examples/synthetic_demo.py`](examples/synthetic_demo.py).

## Main API

- `HuberFusionClusterer`: high-level `fit` and `fit_predict` interface
- `ADMMClusterConfig`: model and optimization settings
- `fit_admm_cluster`: functional fitting interface
- `admm_huber_fusion`: lower-level ADMM solver
- `cluster_by_threshold`: cluster extraction from coefficient vectors
- `select_lambda_by_ch`: tuning-parameter selection
- `inject_outliers`: reproducible synthetic contamination

## Input formats

Responses may be supplied as either:

- a two-dimensional array with shape `(n_subjects, n_times)`; or
- a list of one-dimensional arrays for unequal-length trajectories.

Time points and covariance matrices may likewise be common across subjects or
provided separately for each subject.

## Data policy

This repository contains source code and a synthetic example only. It does not
include participant-level data, fitted research outputs, or local analysis
documents.

## Limitations

- Complete-graph fusion can become expensive as the number of subjects grows.
- Tuning-parameter selection fits the model repeatedly and may be
  computationally intensive.
- Results can be sensitive to the fusion grid and cluster threshold; these
  settings should be chosen for the scale of the application.

## License

This project is available under the [MIT License](LICENSE).

## Contact

Questions and feedback are welcome at
[yukang.lu@outlook.com](mailto:yukang.lu@outlook.com).
