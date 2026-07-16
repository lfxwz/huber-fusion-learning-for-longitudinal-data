# Huber Loss Fusion Clustering

`huber-fusion-clustering` is a small Python package for robust subgroup
discovery in longitudinal data. It combines a Huber-loss objective with fusion
penalties and solves the resulting optimization problem with ADMM.

The project is designed for subject-specific trajectories. Each subject has a
coefficient vector, and subjects with similar fitted coefficients are fused
into the same cluster.

## Method

For subject-specific coefficients $\beta_i$, the package minimizes an objective
of the form

```math
\sum_{i=1}^{n}\sum_{t=1}^{T_i}\rho_c\!\left(\widetilde{y}_{it}-\widetilde{x}_{it}^{\mathsf{T}}\beta_i\right)+\sum_{i\lt j}p_\lambda\!\left(\lVert\beta_i-\beta_j\rVert_2\right)
```

where $\rho_c$ is the Huber loss and $p_\lambda$ is either an L2 or MCP fusion
penalty. Working covariance matrices can be supplied directly or estimated
with an AR(1) structure.

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
