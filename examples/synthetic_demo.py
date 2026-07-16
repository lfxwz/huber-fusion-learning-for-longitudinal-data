"""Run fusion clustering with a multivariable longitudinal design."""

from __future__ import annotations

import numpy as np

from huber_fusion_clustering import ADMMClusterConfig, HuberFusionClusterer


def make_synthetic_data(random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Create two trajectory groups from four-column subject-level designs."""
    rng = np.random.default_rng(random_state)
    n_per_group = 30
    n_subjects = 2 * n_per_group
    time = np.linspace(0.0, 1.0, 50)
    exposure = rng.normal(0.0, 1.0, size=(n_subjects, time.size))

    design = np.empty((n_subjects, time.size, 4), dtype=float)
    design[:, :, 0] = 1.0
    design[:, :, 1] = time
    design[:, :, 2] = np.sin(2.0 * np.pi * time)
    design[:, :, 3] = exposure

    beta_group_a = np.array([0.5, 1.2, 0.35, 0.45])
    beta_group_b = np.array([1.8, -0.8, -0.25, -0.45])
    coefficients = np.vstack(
        [
            np.tile(beta_group_a, (n_per_group, 1)),
            np.tile(beta_group_b, (n_per_group, 1)),
        ]
    )
    noise = rng.normal(0.0, 0.08, size=(n_subjects, time.size))
    responses = np.einsum("ntd,nd->nt", design, coefficients) + noise
    return responses, design


def main() -> None:
    responses, design = make_synthetic_data()
    config = ADMMClusterConfig(
        lam_grid=np.linspace(0.0001, 1.0, 30).tolist(),
        tau_cluster=0.20,
        min_cluster_size=2,
        max_admm=150,
        ch_n_jobs=-1,
        verbose=0,
    )
    model = HuberFusionClusterer(config)
    labels = model.fit_predict(y=responses, X=design)

    if model.result_ is None:
        raise RuntimeError("The model did not produce a fitted result.")

    counts = np.bincount(labels)
    print(f"Design shape: {design.shape} (subjects, times, predictors)")
    print(f"Selected lambda: {model.result_.best_lambda:.3f}")
    print(f"Number of clusters: {model.result_.n_clusters}")
    print(f"Cluster sizes: {counts.tolist()}")


if __name__ == "__main__":
    main()
