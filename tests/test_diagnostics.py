from __future__ import annotations

import numpy as np
import pytest

from huber_fusion_clustering import (
    ADMMClusterResult,
    HuberFusionClusterer,
    plot_convergence,
    print_convergence_summary,
)
from huber_fusion_clustering.diagnostics import _extract_history


RESIDUAL_HISTORY = [
    (1.0, 0.8, 0.1, 0.1),
    (0.2, 0.1, 0.1, 0.1),
]


def _make_result() -> ADMMClusterResult:
    return ADMMClusterResult(
        labels=np.array([0]),
        beta_hat=np.zeros((1, 1)),
        best_lambda=0.1,
        best_ch=0.0,
        n_clusters=1,
        metrics={},
        history=[{"lambda": 0.1, "info": {"history": RESIDUAL_HISTORY}}],
        info={"history": RESIDUAL_HISTORY},
        xlist=[np.ones((1, 1))],
        ylist=[np.ones(1)],
        vlist=[np.eye(1)],
    )


def test_extract_history_accepts_raw_history_and_solver_info() -> None:
    expected = RESIDUAL_HISTORY

    assert _extract_history(RESIDUAL_HISTORY) == expected
    assert _extract_history({"history": RESIDUAL_HISTORY}) == expected


def test_extract_history_uses_selected_fit_from_cluster_result() -> None:
    result = _make_result()

    assert _extract_history(result) == RESIDUAL_HISTORY


def test_extract_history_accepts_fitted_clusterer() -> None:
    model = HuberFusionClusterer()
    model.result_ = _make_result()

    assert _extract_history(model) == RESIDUAL_HISTORY


def test_extract_history_rejects_lambda_selection_records() -> None:
    assert _extract_history([{"lambda": 0.1}]) == []


def test_plot_convergence_accepts_cluster_result() -> None:
    pyplot = pytest.importorskip("matplotlib.pyplot")

    figure = plot_convergence(_make_result(), show=False)

    assert len(figure.axes) == 2
    pyplot.close(figure)


def test_summary_does_not_claim_convergence_at_iteration_limit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    info = {
        "history": [(0.01, 0.2, 0.1, 0.1)],
        "iter": 1,
        "r_norm": 0.01,
        "s_norm": 0.2,
        "eps_pri": 0.1,
        "eps_dual": 0.1,
    }

    print_convergence_summary(info, last_k=1)

    output = capsys.readouterr().out
    assert "ADMM stopped after 1 iteration" in output
    assert "Status: NOT CONVERGED" in output
