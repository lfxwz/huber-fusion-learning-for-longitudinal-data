"""Run fusion clustering with adaptive pairwise tensor-product splines."""

from __future__ import annotations

import numpy as np

from huber_fusion_clustering import ADMMClusterConfig, HuberFusionClusterer


def make_synthetic_data(
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create two groups with main effects and pairwise interactions."""
    rng = np.random.default_rng(random_state)
    n_per_group = 30
    n_subjects = 2 * n_per_group
    n_observations = 100
    covariates = rng.uniform(-1.0, 1.0, size=(n_subjects, n_observations, 3))
    x1 = covariates[:, :, 0]
    x2 = covariates[:, :, 1]
    x3 = covariates[:, :, 2]

    group_a = (
        0.4
        + 0.9 * x1
        + 0.3 * x1**2
        - 0.6 * x2
        + 0.4 * x2**2
        + 0.5 * x3
        + 0.3 * x3**2
        + 0.7 * x1 * x2
        - 0.5 * x2 * x3
    )
    group_b = (
        1.4
        - 0.8 * x1
        + 0.2 * x1**2
        + 0.7 * x2
        - 0.3 * x2**2
        - 0.5 * x3
        + 0.6 * x3**2
        - 0.6 * x1 * x2
        + 0.6 * x1 * x3
    )
    responses = np.vstack([group_a[:n_per_group], group_b[n_per_group:]])
    responses += rng.normal(0.0, 0.08, size=responses.shape)
    return responses, covariates


def main() -> None:
    responses, covariates = make_synthetic_data()
    config = ADMMClusterConfig(
        lam_grid=np.linspace(0.0001, 1.0, 30).tolist(),
        df=3,
        degree=2,
        max_tensor_order=2,
        tau_cluster=0.20,
        min_cluster_size=2,
        max_admm=150,
        ch_n_jobs=-1,
        verbose=0,
    )
    model = HuberFusionClusterer(config)
    labels = model.fit_predict(
        y=responses,
        covariates=covariates,
        V=np.eye(responses.shape[1]),
    )

    if model.result_ is None:
        raise RuntimeError("The model did not produce a fitted result.")

    counts = np.bincount(labels)
    coefficient_blocks = model.result_.coefficient_blocks()
    tensor_blocks = model.result_.tensor_coefficient_blocks()
    print(f"Covariate shape: {covariates.shape} (subjects, observations, predictors)")
    print(f"Design shape per subject: {model.result_.xlist[0].shape}")
    print(f"Coefficient blocks: {coefficient_blocks.shape} (subjects, functions, basis)")
    print(f"Pairwise block: {tensor_blocks[(0, 1)].shape}")
    print(f"Selected lambda: {model.result_.best_lambda:.3f}")
    print(f"Number of clusters: {model.result_.n_clusters}")
    print(f"Cluster sizes: {counts.tolist()}")


if __name__ == "__main__":
    main()
