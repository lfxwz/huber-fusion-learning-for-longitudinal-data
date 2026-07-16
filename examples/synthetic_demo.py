"""Run a small fusion-clustering example with synthetic trajectories."""

from __future__ import annotations

import numpy as np

from huber_fusion_clustering import ADMMClusterConfig, HuberFusionClusterer


def make_synthetic_data(random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Create two groups of smooth trajectories with light Gaussian noise."""
    rng = np.random.default_rng(random_state)
    time = np.linspace(0.0, 1.0, 50)
    group_a = [
        0.5 + 1.2 * time + rng.normal(0.0, 0.08, time.size)
        for _ in range(30)
    ]
    group_b = [
        1.8 - 0.8 * time + rng.normal(0.0, 0.08, time.size)
        for _ in range(30)
    ]
    return np.asarray(group_a + group_b), time


def main() -> None:
    responses, time = make_synthetic_data()
    config = ADMMClusterConfig(
        lam_grid=np.linspace(0.0001, 1.0, 30).tolist(),
        tau_cluster=0.20,
        min_cluster_size=2,
        max_admm=150,
        ch_n_jobs=-1,
        verbose=1,
    )
    model = HuberFusionClusterer(config)
    labels = model.fit_predict(y=responses, t=time)

    if model.result_ is None:
        raise RuntimeError("The model did not produce a fitted result.")

    counts = np.bincount(labels)
    print(f"Selected lambda: {model.result_.best_lambda:.3f}")
    print(f"Number of clusters: {model.result_.n_clusters}")
    print(f"Cluster sizes: {counts.tolist()}")


if __name__ == "__main__":
    main()
