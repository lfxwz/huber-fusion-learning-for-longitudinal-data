import numpy as np

from huber_fusion_clustering import (
    ADMMClusterConfig,
    admm_huber_fusion,
    cluster_by_threshold,
    inject_outliers,
)
from huber_fusion_clustering.interface import (
    _build_additive_spline_xlist,
    create_bspline_basis_manual,
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
