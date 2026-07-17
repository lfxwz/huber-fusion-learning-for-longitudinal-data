# Huber Fusion Learning for Longitudinal Data

`huber-fusion-clustering` is a small Python package for robust subgroup
discovery in longitudinal data. It combines a Huber-loss objective with fusion
penalties and solves the resulting optimization problem with ADMM.

The project is designed for subject-specific trajectories. Each subject has a
coefficient vector, and subjects with similar fitted coefficients are fused
into the same cluster.

## Method

### Subject-level multivariable model

For subject $i=1,\ldots,n$, let $T_i$ be the number of observations,
$y_i=(y_{i1},\ldots,y_{iT_i})^{\mathsf T}\in\mathbb R^{T_i}$ the univariate
response, and $Z_i\in\mathbb R^{T_i\times p}$ the raw predictor matrix. Its
$(k,j)$ entry is denoted by $x_{ikj}$. Each subject has an individual response
surface, and subgroup discovery is based on similarities among these surfaces.

For continuous predictor $j$, let
$B_j(x)=(B_{j1}(x),\ldots,B_{jq}(x))^{\mathsf T}$ be a reduced B-spline basis.
The package uses common bounds for each predictor across subjects, removes the
duplicated constant direction from every spline block, and retains one explicit
subject-specific intercept $\alpha_i$.

### Hierarchical tensor-product representation

Let $R$ be the maximum interaction order and define

```math
\mathcal S_R=\left\{S\subseteq\{1,\ldots,p\}:1\le |S|\le R\right\}.
```

For a predictor subset $S$, its row-level tensor basis is

```math
\phi_S(x_{ik})=\bigotimes_{j\in S}B_j(x_{ikj}).
```

The complete feature row and subject-specific model are

```math
d_{ik}^{\mathsf T}=\left[1,\left\{\phi_S(x_{ik})^{\mathsf T}:S\in\mathcal S_R\right\}\right],
```

```math
y_{ik}=d_{ik}^{\mathsf T}\theta_i+\varepsilon_{ik}
=\alpha_i+\sum_{S\in\mathcal S_R}\left\langle\Theta_{iS},\phi_S(x_{ik})\right\rangle+\varepsilon_{ik}.
```

Stacking the rows gives $y_i=D_i\theta_i+\varepsilon_i$. With $q$ spline terms
per predictor and an intercept, the coefficient dimension is

```math
d_R=1+\sum_{r=1}^{R}\binom{p}{r}q^r.
```

`max_tensor_order=1` gives an additive model, `max_tensor_order=2` includes all
pairwise tensor interactions, and `max_tensor_order=None` sets $R=p$ and yields
$d_p=(1+q)^p$.

### Working covariance and pre-whitening

Within-subject dependence is represented by a positive-definite working
covariance matrix $V_i$. Writing $V_i=L_iL_i^{\mathsf T}$, define

```math
\widetilde y_i=L_i^{-1}y_i,\qquad \widetilde D_i=L_i^{-1}D_i.
```

Users may supply $V_i$ directly. Otherwise, the package constructs a
residual-based AR(1) working covariance estimate. Supplying identity matrices
gives working independence, as used in the synthetic example.

### Robust fusion objective

The subject-specific coefficients are estimated jointly by minimizing

```math
\min_{\theta_1,\ldots,\theta_n}
\sum_{i=1}^{n}\sum_{k=1}^{T_i}
\rho_c\!\left(\widetilde y_{ik}-\widetilde d_{ik}^{\mathsf T}\theta_i\right)
+\sum_{1\le i<j\le n}p_\lambda\!\left(\lVert\theta_i-\theta_j\rVert_2\right).
```

The Huber loss is

```math
\rho_c(r)=
\begin{cases}
r^2/2, & |r|\le c,\\
c|r|-c^2/2, & |r|>c.
\end{cases}
```

It retains quadratic efficiency for ordinary residuals while limiting the
influence of large residuals. Two fusion penalties are available. Writing
$u=\lVert\theta_i-\theta_j\rVert_2$,

```math
p_\lambda^{\mathrm{L2}}(u)=\lambda u,
```

and

```math
p_{\lambda,\gamma}^{\mathrm{MCP}}(u)=
\begin{cases}
\lambda u-u^2/(2\gamma), & 0\le u\le\gamma\lambda,\\
\gamma\lambda^2/2, & u>\gamma\lambda.
\end{cases}
```

The complete subject graph contains one edge for every pair $(i,j)$. Shrinking
$\theta_i-\theta_j$ to zero makes the two subjects share the same fitted
response surface, so the subgroup structure emerges without specifying the
number of groups in advance.

### ADMM optimization

Let $A\in\mathbb R^{m\times n}$ be the incidence matrix of the complete graph,
$m=\binom n2$, and let $\Theta$ stack the subject coefficients by row. ADMM
introduces $Q=A\Theta$ and a scaled dual variable $U$. Each iteration performs:

1. **Robust coefficient update.** Huber weights are updated by IRLS. The
   resulting system combines subject-level weighted normal equations with the
   graph Laplacian $A^{\mathsf T}A$ and is solved by conjugate gradients.
2. **Fusion update.** Each edge row of $A\Theta+U$ is passed through the group
   L2 or group MCP proximal operator.
3. **Dual update.** $U\leftarrow U+A\Theta-Q$.

Iterations stop when the primal and dual residuals satisfy the configured
tolerances or when `max_admm` is reached.

### Cluster extraction and lambda selection

For a fitted $\widehat\Theta(\lambda)$, subjects are connected when

```math
\lVert\widehat\theta_i(\lambda)-\widehat\theta_j(\lambda)\rVert_2\le\tau.
```

Connected components define the initial clusters; undersized clusters are
merged according to `min_cluster_size`. For every candidate in `lam_grid`, the
model is refitted and labels are extracted. The Calinski-Harabasz score is then
computed in the initial subject-level least-squares coefficient space using
those labels:

```math
\operatorname{CH}(\lambda)=
\frac{\operatorname{between}(\lambda)/(K_\lambda-1)}
{\operatorname{within}(\lambda)/(n-K_\lambda)}.
```

The selected value is the grid candidate with the largest CH score. Candidate
fits can run in parallel through `ch_n_jobs`.

## Features

- Huber-loss optimization with ADMM
- L2 and MCP fusion penalties
- Support for equal- and unequal-length trajectories
- Multiple predictors with a separate B-spline expansion for each predictor
- Adaptive tensor products with a configurable maximum interaction order
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
from sklearn.metrics import normalized_mutual_info_score, rand_score

from huber_fusion_clustering import ADMMClusterConfig, HuberFusionClusterer

rng = np.random.default_rng(20260717)
n_per_group = 30
n_subjects = 3 * n_per_group
n_observations = 120

covariates = rng.uniform(-1.5, 1.5, size=(n_subjects, n_observations, 2))
x1 = covariates[:, :, 0]
x2 = covariates[:, :, 1]

mean_1 = x1 + x1 * x2 + x2**2
mean_2 = np.sin(x1) + np.cos(x1 * x2) + x2
mean_3 = x1**2 - np.sin(x2) - x1 * x2

y = np.vstack(
    [
        mean_1[:n_per_group],
        mean_2[n_per_group : 2 * n_per_group],
        mean_3[2 * n_per_group :],
    ]
)
y += rng.normal(0.0, 0.08, size=y.shape)
true_labels = np.repeat(np.arange(3), n_per_group)

config = ADMMClusterConfig(
    lam_grid=np.linspace(0.0001, 1.0, 30).tolist(),
    df=6,
    degree=3,
    max_tensor_order=2,
    tau_cluster=1.0,
    min_cluster_size=2,
    max_admm=150,
    ch_n_jobs=-1,
    verbose=0,
)

model = HuberFusionClusterer(config)
labels = model.fit_predict(
    y=y,
    covariates=covariates,
    V=np.eye(n_observations),
)

print("Selected lambda:", model.result_.best_lambda)
print("Number of clusters:", model.result_.n_clusters)
print("Cluster sizes:", np.bincount(labels).tolist())
print("Rand index:", rand_score(true_labels, labels))
print("NMI:", normalized_mutual_info_score(true_labels, labels))
```

A complete runnable version is available in
[`examples/synthetic_demo.py`](examples/synthetic_demo.py). It generates
subject-specific predictor values, exercises nonlinear main effects and an
$x_1x_2$ interaction, and reports recovery against the known three-group
partition. On the reference seed, the fitted model recovers 30 subjects in
each group with Rand index and normalized mutual information equal to 1.

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

Raw predictors for the additive-spline interface may be supplied as a
three-dimensional array with shape `(n_subjects, n_observations, n_predictors)`,
a common matrix with shape `(n_observations, n_predictors)`, or a list of
subject-specific matrices with shape `(n_observations_i, n_predictors)`. Pass
these as `covariates=` without `t=`. The existing `t=` interface remains
available for a time-only spline model, and `X=` remains available for an
already constructed design matrix.

Use `max_tensor_order=2` to include every pairwise tensor block regardless of
whether the input contains 2, 10, or 20 predictors. Set it to `None` to include
all orders. Predictor indices in `tensor_coefficient_blocks()` are zero based:
`(0,)` is the first main effect, `(0, 1)` is the first-second interaction, and
`(0, 1, 2)` is the corresponding three-way interaction when enabled.

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
- A tensor design through order $R$ has
  $\sum_{r=0}^{R}\binom{p}{r}q^r$ columns; the unrestricted full-order design
  grows exponentially as $(1+q)^p$.

## License

This project is available under the [MIT License](LICENSE).

## Contact

Questions and feedback are welcome at
[yukang.lu@outlook.com](mailto:yukang.lu@outlook.com).
