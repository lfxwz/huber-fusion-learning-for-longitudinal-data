import numpy as np
import pytest

from huber_fusion_clustering import (
    ADMMClusterConfig,
    HuberFusionClusterer,
    admm_huber_fusion,
    cluster_by_threshold,
    inject_outliers,
    relabel_consecutive,
    select_lambda_by_ch,
)
from huber_fusion_clustering.clustering import _cluster_accuracy
from huber_fusion_clustering.interface import (
    _build_additive_spline_xlist,
    create_bspline_basis_manual,
    estimate_covariance_from_residuals,
)


def test_threshold_clustering_finds_two_groups() -> None:
    coefficients = np.array(
        [
            [0.00, 0.04, 3.00, 3.04],
            [0.00, 0.03, 3.00, 3.02],
        ]
    )

    labels, details = cluster_by_threshold(
        coefficients,
        threshold=0.1,
        min_cluster_size=1,
    )

    np.testing.assert_array_equal(labels, np.array([0, 0, 1, 1]))
    assert details["n_clusters"] == 2


def test_outlier_injection_is_reproducible_and_non_mutating() -> None:
    responses = [np.zeros(8), np.ones(8), np.full(8, 2.0)]

    first, first_info = inject_outliers(
        responses,
        subject_fraction=2 / 3,
        time_fraction=0.25,
        random_state=17,
    )
    second, second_info = inject_outliers(
        responses,
        subject_fraction=2 / 3,
        time_fraction=0.25,
        random_state=17,
    )

    for original, contaminated_first, contaminated_second in zip(
        responses, first, second
    ):
        np.testing.assert_array_equal(contaminated_first, contaminated_second)
        assert not np.shares_memory(original, contaminated_first)
    assert first_info["n_points"] == second_info["n_points"] == 4
    np.testing.assert_array_equal(responses[0], np.zeros(8))


def test_bspline_basis_has_expected_shape_and_partition() -> None:
    time = np.linspace(0.0, 1.0, 9)

    basis = create_bspline_basis_manual(time, df=4, degree=2)

    assert basis.shape == (9, 4)
    np.testing.assert_allclose(basis.sum(axis=1), 1.0)


def test_tensor_design_contains_main_and_pairwise_blocks() -> None:
    values = np.linspace(-1.0, 1.0, 8)
    covariates = np.stack(
        [
            np.column_stack([values, values**2]),
            np.column_stack([values[::-1], values**2]),
        ]
    )
    responses = [np.zeros(8), np.ones(8)]
    config = ADMMClusterConfig(df=3, degree=2, max_tensor_order=2)

    design, info = _build_additive_spline_xlist(
        covariates,
        n=2,
        ylist=responses,
        cfg=config,
    )

    assert design[0].shape == (8, 16)
    assert info["tensor_subsets"] == [[0], [1], [0, 1]]
    assert info["n_columns"] == 16


def test_admm_solver_smoke() -> None:
    time = np.linspace(0.0, 1.0, 8)
    design = np.column_stack([np.ones_like(time), time])
    xlist = [design.copy() for _ in range(4)]
    ylist = [
        0.2 + time,
        0.25 + time,
        2.0 - time,
        2.05 - time,
    ]
    covariance = [np.eye(time.size) for _ in range(4)]

    beta_hat, info = admm_huber_fusion(
        xlist,
        ylist,
        covariance,
        lam=0.1,
        max_admm=5,
        max_irls=1,
        penalty="l2",
    )

    assert beta_hat.shape == (2, 4)
    assert np.all(np.isfinite(beta_hat))
    assert info["iter"] >= 1


# =========================================================================
# End-to-end tests
# =========================================================================


def test_end_to_end_time_spline_two_groups() -> None:
    """Full pipeline: time splines -> cluster -> recover 2 groups."""
    rng = np.random.default_rng(42)
    t = np.linspace(0.0, 1.0, 12)
    n_per_group = 8

    y1 = [np.sin(2 * np.pi * t) + 0.05 * rng.standard_normal(len(t)) for _ in range(n_per_group)]
    y2 = [np.cos(2 * np.pi * t) + 0.05 * rng.standard_normal(len(t)) for _ in range(n_per_group)]
    y = y1 + y2

    config = ADMMClusterConfig(
        df=5,
        penalty="mcp",
        lam_grid=[0.005, 0.01, 0.05, 0.1, 0.3, 0.5],
        tau_cluster=0.5,
        min_cluster_size=2,
        max_admm=80,
        verbose=0,
    )
    model = HuberFusionClusterer(config)
    labels = model.fit_predict(y, t=t)

    assert labels.shape == (2 * n_per_group,)
    assert model.result_ is not None
    assert model.result_.n_clusters == 2
    assert np.bincount(labels).tolist() == [n_per_group, n_per_group]


def test_end_to_end_unequal_length() -> None:
    """Subjects with different observation counts should cluster correctly."""
    rng = np.random.default_rng(123)
    t_lengths = [10, 15, 12, 8, 14, 11]
    y = []
    t_list = []
    for i, T in enumerate(t_lengths):
        ti = np.linspace(0.0, 1.0, T)
        t_list.append(ti)
        base = np.sin(2 * np.pi * ti) if i < 3 else np.cos(2 * np.pi * ti)
        y.append(base + 0.03 * rng.standard_normal(T))

    config = ADMMClusterConfig(
        df=4,
        lam_grid=[0.01, 0.05, 0.1, 0.3],
        tau_cluster=0.5,
        min_cluster_size=2,
        max_admm=60,
    )
    model = HuberFusionClusterer(config)
    labels = model.fit_predict(y, t=t_list)

    assert labels.shape == (6,)
    assert model.result_ is not None
    # First 3 should be one group, last 3 another
    assert len(set(labels[:3])) == 1
    assert len(set(labels[3:])) == 1
    assert labels[0] != labels[3]


def test_reproducibility() -> None:
    """Same seed should produce identical labels across runs."""
    rng = np.random.default_rng(777)
    t = np.linspace(0.0, 1.0, 10)
    y = [np.sin(2 * np.pi * t) + 0.02 * rng.standard_normal(len(t)) for _ in range(6)]

    config = ADMMClusterConfig(df=4, lam_grid=[0.01, 0.05, 0.1], max_admm=40)

    labels_1 = HuberFusionClusterer(config).fit_predict(y, t=t)
    labels_2 = HuberFusionClusterer(config).fit_predict(y, t=t)
    np.testing.assert_array_equal(labels_1, labels_2)


# =========================================================================
# Covariance estimation
# =========================================================================


def test_covariance_estimation_produces_spd() -> None:
    """Estimated covariance matrices must be symmetric positive definite."""
    rng = np.random.default_rng(99)
    t = np.linspace(0.0, 1.0, 8)
    basis = create_bspline_basis_manual(t, df=3, degree=2)
    xlist = [basis.copy() for _ in range(4)]
    ylist = [rng.standard_normal(len(t)) for _ in range(4)]

    vlist, info = estimate_covariance_from_residuals(
        xlist=xlist,
        ylist=ylist,
        tlist=[t] * 4,
        method="ar1",
    )

    assert len(vlist) == 4
    for i, V in enumerate(vlist):
        assert V.shape == (len(t), len(t))
        # Symmetric
        np.testing.assert_allclose(V, V.T, atol=1e-10)
        # Positive definite: all eigenvalues > 0
        eigvals = np.linalg.eigvalsh(V)
        assert np.all(eigvals > 0), f"V[{i}] has non-positive eigenvalue: {eigvals.min()}"
    assert info["source"] == "estimated"
    assert np.isfinite(info["sigma2"]) and info["sigma2"] > 0
    assert -0.99 <= info["rho"] <= 0.99


def test_covariance_estimation_unequal_length() -> None:
    """Covariance estimation works with different trajectory lengths."""
    rng = np.random.default_rng(55)
    lengths = [6, 10, 8]
    xlist = []
    ylist = []
    tlist = []
    for T in lengths:
        ti = np.linspace(0.0, 1.0, T)
        tlist.append(ti)
        xlist.append(create_bspline_basis_manual(ti, df=3, degree=2))
        ylist.append(rng.standard_normal(T))

    vlist, _ = estimate_covariance_from_residuals(
        xlist=xlist, ylist=ylist, tlist=tlist, method="ar1",
    )
    assert len(vlist) == 3
    for i, (V, T) in enumerate(zip(vlist, lengths)):
        assert V.shape == (T, T), f"V[{i}] shape mismatch"


# =========================================================================
# Penalty comparison
# =========================================================================


def test_penalty_l2_vs_mcp_produce_different_results() -> None:
    """L2 and MCP penalties should generally yield different estimates."""
    t = np.linspace(0.0, 1.0, 8)
    design = np.column_stack([np.ones_like(t), t])
    xlist = [design.copy() for _ in range(6)]
    ylist = [
        0.2 + t, 0.25 + t, 0.22 + t,
        2.0 - t, 2.05 - t, 1.98 - t,
    ]
    V = [np.eye(len(t)) for _ in range(6)]

    beta_l2, _ = admm_huber_fusion(
        xlist, ylist, V, lam=0.1, max_admm=50, penalty="l2",
    )
    beta_mcp, _ = admm_huber_fusion(
        xlist, ylist, V, lam=0.1, max_admm=50, penalty="mcp", gamma=3.0,
    )

    # Results should differ (MCP is less biased for large effects)
    diff = np.linalg.norm(beta_l2 - beta_mcp)
    assert diff > 1e-6, f"L2 and MCP produced nearly identical results (diff={diff:.2e})"


# =========================================================================
# CH selection
# =========================================================================


def test_select_lambda_by_ch_finds_reasonable_lambda() -> None:
    """CH selection should pick a lambda that separates two clear groups."""
    rng = np.random.default_rng(33)
    t = np.linspace(0.0, 1.0, 10)
    basis = create_bspline_basis_manual(t, df=4, degree=2)
    xlist = [basis.copy() for _ in range(10)]
    ylist = (
        [np.sin(2 * np.pi * t) + 0.05 * rng.standard_normal(len(t)) for _ in range(5)]
        + [np.cos(2 * np.pi * t) + 0.05 * rng.standard_normal(len(t)) for _ in range(5)]
    )
    V = [np.eye(len(t)) for _ in range(10)]

    result = select_lambda_by_ch(
        xlist, ylist, V,
        lam_grid=[0.001, 0.01, 0.05, 0.1, 0.5],
        tau_cluster=0.5,
        min_cluster_size=2,
        max_admm=60,
        penalty="mcp",
        n_jobs=1,
    )

    assert result.best_lambda > 0
    assert result.best_ch > 0
    assert result.best_beta_hat.shape == (basis.shape[1], 10)
    assert len(result.best_labels) == 10
    assert len(np.unique(result.best_labels)) == 2


# =========================================================================
# Utility functions
# =========================================================================


def test_relabel_consecutive() -> None:
    labels = np.array([5, 5, 10, 5, 10, 20])
    relabeled = relabel_consecutive(labels)
    np.testing.assert_array_equal(relabeled, np.array([0, 0, 1, 0, 1, 2]))


def test_cluster_accuracy_identical() -> None:
    """Perfect prediction should give accuracy = 1."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([2, 2, 0, 0, 1, 1])  # permuted but correct
    acc = _cluster_accuracy(y_true, y_pred)
    assert acc == pytest.approx(1.0)


def test_cluster_accuracy_worst() -> None:
    """Random labels should give low accuracy."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1])
    acc = _cluster_accuracy(y_true, y_pred)
    assert acc < 1.0


# =========================================================================
# Edge cases
# =========================================================================


def test_single_subject() -> None:
    """Single subject should return single cluster without crashing."""
    t = np.linspace(0.0, 1.0, 8)
    y = [np.sin(2 * np.pi * t)]
    config = ADMMClusterConfig(df=3, lam_grid=[0.01], max_admm=20)
    labels = HuberFusionClusterer(config).fit_predict(y, t=t)
    assert labels.shape == (1,)


def test_two_subjects_different_groups() -> None:
    """Two clearly different trajectories should form two clusters."""
    t = np.linspace(0.0, 1.0, 8)
    y = [np.zeros(8), 5.0 * np.ones(8)]
    config = ADMMClusterConfig(
        df=3, lam_grid=[0.001, 0.01, 0.1], tau_cluster=0.1,
        min_cluster_size=1, max_admm=40,
    )
    labels = HuberFusionClusterer(config).fit_predict(y, t=t)
    assert labels.shape == (2,)
    assert labels[0] != labels[1]
