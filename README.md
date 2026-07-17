# Huber Fusion Learning for Longitudinal Data

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

For a univariate response with $p$ predictors, the package can construct a
multivariable spline and tensor-product design automatically. Let $B_j(x)$
contain $q$ identifiable B-spline terms for predictor $j$. For observation $k$
from subject $i$, the model is

```math
y_{ik}=\alpha_i+\sum_{j=1}^{p}B_j(x_{ikj})^{\mathsf T}\theta_{ij}+\varepsilon_{ik}.
```

Thus the subject-level design is $[1,\ B_1(x_{i1}),\ldots,B_p(x_{ip})]$, and
each subject has $1+pq$ coefficients. Each predictor uses its own pooled value
range to define common spline knots across subjects. Fusion is applied jointly
to the intercept and all predictor-specific spline coefficients. No time
variable is required for this interface.

The interaction order is controlled by `max_tensor_order`. For every predictor
subset $S\subseteq\{1,\ldots,p\}$ satisfying
$1\le |S|\le R$, where $R$ is the selected maximum order, the design includes
$\bigotimes_{j\in S}B_j(x_j)$. Consequently,

```math
y_{ik}=\alpha_i+\sum_{1\le |S|\le R}\left\langle\Theta_{iS},\bigotimes_{j\in S}B_j(x_{ikj})\right\rangle+\varepsilon_{ik}.
```

`max_tensor_order=1` gives the additive model, `max_tensor_order=2` adds every
pairwise tensor interaction for any number of predictors, and
`max_tensor_order=None` expands through the full $p$-way interaction. With $q$
spline terms per predictor and an intercept, the coefficient dimension is
$\sum_{r=0}^{R}\binom{p}{r}q^r$; the unrestricted case has $(1+q)^p$ columns.

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
n_groups = 3
n_subjects = n_groups * n_per_group
n_observations = 120

covariates = rng.uniform(-1.5, 1.5, size=(n_subjects, n_observations, 2))
x1 = covariates[:, :, 0]
x2 = covariates[:, :, 1]

mean_group_1 = x1 + x1 * x2 + x2**2
mean_group_2 = np.sin(x1) + np.cos(x1 * x2) + x2
mean_group_3 = x1**2 - np.sin(x2) - x1 * x2

responses = np.vstack(
    [
        mean_group_1[:n_per_group],
        mean_group_2[n_per_group : 2 * n_per_group],
        mean_group_3[2 * n_per_group :],
    ]
)
responses += rng.normal(0.0, 0.08, size=responses.shape)
true_labels = np.repeat(np.arange(n_groups), n_per_group)

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
    y=responses,
    covariates=covariates,
    V=np.eye(n_observations),
)

print(f"Selected lambda: {model.result_.best_lambda:.6f}")
print(f"Number of clusters: {model.result_.n_clusters}")
print(f"Cluster sizes: {np.bincount(labels).tolist()}")
print(f"Rand index: {rand_score(true_labels, labels):.6f}")
print(
    "Normalized mutual information: "
    f"{normalized_mutual_info_score(true_labels, labels):.6f}"
)
```

A complete runnable version is available in
[`examples/synthetic_demo.py`](examples/synthetic_demo.py).

### Mathematics of the three-group example

The quick start and runnable example generate $n=90$ subjects, with 30 subjects
in each latent group and $T=120$ observations per subject. Predictor values
vary across both subjects and observations:

```math
x_{ik1},x_{ik2}\overset{\mathrm{iid}}{\sim}\mathrm{Uniform}(-1.5,1.5),
\qquad
\varepsilon_{ik}\overset{\mathrm{iid}}{\sim}\mathcal N(0,0.08^2).
```

If $g_i\in\{1,2,3\}$ is the unknown group of subject $i$, the response is

```math
y_{ik}=f_{g_i}(x_{ik1},x_{ik2})+\varepsilon_{ik},
```

where the three group-specific response surfaces are

```math
\begin{aligned}
f_1(x_1,x_2)&=x_1+x_1x_2+x_2^2,\\
f_2(x_1,x_2)&=\sin(x_1)+\cos(x_1x_2)+x_2,\\
f_3(x_1,x_2)&=x_1^2-\sin(x_2)-x_1x_2.
\end{aligned}
```

The terms $x_1x_2$ and $\cos(x_1x_2)$ make the response genuinely
non-additive: their effect cannot generally be written as a sum
$h_1(x_1)+h_2(x_2)$. With `df=6` and `max_tensor_order=2`, the fitted row-level
design is

```math
d(x_1,x_2)^{\mathsf T}
=\left[1,\ B_1(x_1)^{\mathsf T},\ B_2(x_2)^{\mathsf T},
\{B_1(x_1)\otimes B_2(x_2)\}^{\mathsf T}\right].
```

It therefore has $1+6+6+6^2=49$ columns. The two main-effect blocks represent
smooth changes in one predictor at a time, while the $6\times6$ tensor block
approximates the full pairwise response surface, including both simple and
nonlinear interactions.

Each subject receives its own 49-dimensional coefficient vector. The fusion
penalty pulls together subjects with similar estimated surfaces, and the
resulting fused coefficient vectors determine the clusters. The example scans
30 values of $\lambda$ from $0.0001$ to $1$, selects the candidate with the
largest Calinski-Harabasz score, and extracts clusters using
`tau_cluster=1.0`. With the bundled random seed, it selects
$\lambda=0.413852$ and exactly recovers three clusters of sizes
$(30,30,30)$, with Rand index and normalized mutual information both equal to
1.

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

Raw predictors for the multivariable spline and tensor-product interface may be
supplied as a three-dimensional array with shape
`(n_subjects, n_observations, n_predictors)`, a common matrix with shape
`(n_observations, n_predictors)`, or a list of subject-specific matrices with
shape `(n_observations_i, n_predictors)`. Pass these as `covariates=` without
`t=`. The existing `t=` interface remains available for a time-only spline
model, and `X=` remains available for an already constructed design matrix.

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
