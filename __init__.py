"""Loss-based robust fusion clustering with Huber loss and ADMM."""

from .clustering import cluster_by_threshold, relabel_consecutive
from .contamination import inject_outliers
from .interface import (
    ADMMClusterConfig,
    ADMMClusterResult,
    HuberFusionClusterer,
    fit_admm_cluster,
)
from .model_selection import (
    CHSelectionResult,
    compute_ch_index,
    initial_subject_ols,
    merge_small_clusters,
    select_lambda_by_ch,
)
from .diagnostics import plot_convergence, print_convergence_summary
from .solver import admm_huber_fusion

__version__ = "0.1.0"

__all__ = [
    "ADMMClusterConfig",
    "ADMMClusterResult",
    "CHSelectionResult",
    "HuberFusionClusterer",
    "admm_huber_fusion",
    "cluster_by_threshold",
    "compute_ch_index",
    "fit_admm_cluster",
    "initial_subject_ols",
    "inject_outliers",
    "merge_small_clusters",
    "plot_convergence",
    "print_convergence_summary",
    "relabel_consecutive",
    "select_lambda_by_ch",
]
