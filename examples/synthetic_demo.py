"""Recover three nonlinear subgroups with tensor-product spline effects."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import normalized_mutual_info_score, rand_score

from huber_fusion_clustering import ADMMClusterConfig, HuberFusionClusterer


def make_synthetic_data(
    random_state: int = 20260717,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create three groups with nonlinear main and interaction effects."""
    rng = np.random.default_rng(random_state)
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
    return responses, covariates, true_labels


def main() -> None:
    responses, covariates, true_labels = make_synthetic_data()
    config = ADMMClusterConfig(
        lam_grid=np.linspace(0.0001, 2.0, 30).tolist(),
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
    print(f"Selected lambda: {model.result_.best_lambda:.6f}")
    print(f"Best CH score: {model.result_.best_ch:.6f}")
    print(f"Number of clusters: {model.result_.n_clusters}")
    print(f"Cluster sizes: {counts.tolist()}")
    print(f"Rand index: {rand_score(true_labels, labels):.6f}")
    print(
        "Normalized mutual information: "
        f"{normalized_mutual_info_score(true_labels, labels):.6f}"
    )


if __name__ == "__main__":
    main()
