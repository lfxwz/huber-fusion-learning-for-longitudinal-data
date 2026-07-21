"""Threshold-based clustering for subject-specific coefficient vectors."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import njit
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist, squareform


@njit
def greedy_merge_numba(distance_matrix: np.ndarray, threshold: float) -> np.ndarray:
    """Merge labels whenever a pairwise distance is below ``threshold``."""
    n_samples = distance_matrix.shape[0]
    labels = np.arange(n_samples)

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if distance_matrix[i, j] < threshold:
                source = labels[j]
                target = labels[i]
                if source != target:
                    for k in range(n_samples):
                        if labels[k] == source:
                            labels[k] = target
    return labels


def _connected_components(adjacency: np.ndarray) -> List[np.ndarray]:
    """Return connected components from a symmetric Boolean adjacency matrix."""
    n_samples = adjacency.shape[0]
    visited = np.zeros(n_samples, dtype=bool)
    components: List[np.ndarray] = []

    for start in range(n_samples):
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True
        component = [start]
        while stack:
            node = stack.pop()
            for neighbor in np.where(adjacency[node])[0]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))
                    component.append(int(neighbor))
        components.append(np.asarray(component, dtype=int))

    return components


def _labels_from_components(components: List[np.ndarray], n_samples: int) -> np.ndarray:
    labels = np.full(n_samples, -1, dtype=int)
    for cluster_id, indices in enumerate(components):
        labels[indices] = cluster_id
    return labels


def _component_clustering(
    coefficients: np.ndarray,
    threshold: float,
    metric: str,
) -> Tuple[np.ndarray, Dict]:
    n_samples = coefficients.shape[1]
    distances = (
        squareform(pdist(coefficients.T, metric=metric))
        if n_samples > 1
        else np.zeros((1, 1))
    )
    adjacency = (distances <= threshold) & (~np.eye(n_samples, dtype=bool))
    components = _connected_components(adjacency)
    labels = _labels_from_components(components, n_samples)
    return labels, {
        "threshold": threshold,
        "clusters": components,
        "distance_matrix": distances,
    }


def _greedy_radius_clustering(
    coefficients: np.ndarray,
    threshold: float,
    metric: str,
) -> Tuple[np.ndarray, Dict]:
    n_samples = coefficients.shape[1]
    distances = (
        squareform(pdist(coefficients.T, metric=metric))
        if n_samples > 1
        else np.zeros((1, 1))
    )
    labels = np.full(n_samples, -1, dtype=int)
    assigned = np.zeros(n_samples, dtype=bool)
    cluster_id = 0

    while not np.all(assigned):
        seed = np.where(~assigned)[0][0]
        members = np.where((distances[seed] <= threshold) & (~assigned))[0]
        labels[members] = cluster_id
        assigned[members] = True
        cluster_id += 1

    clusters = [np.where(labels == label)[0] for label in range(cluster_id)]
    return labels, {
        "threshold": threshold,
        "clusters": clusters,
        "distance_matrix": distances,
    }


def _compact_labels(labels: np.ndarray) -> np.ndarray:
    return relabel_consecutive(labels)


def _cluster_centroids(coefficients: np.ndarray, labels: np.ndarray) -> np.ndarray:
    n_features = coefficients.shape[0]
    unique_labels = np.unique(labels)
    centroids = np.zeros((n_features, len(unique_labels)))

    for column, label in enumerate(unique_labels):
        indices = np.where(labels == label)[0]
        centroids[:, column] = coefficients[:, indices].mean(axis=1)
    return centroids


def _enforce_min_cluster_size(
    coefficients: np.ndarray,
    labels: np.ndarray,
    min_cluster_size: int,
    metric: str,
) -> np.ndarray:
    """Merge undersized clusters into their nearest sufficiently large cluster."""
    labels = labels.copy().astype(int)
    previous_count = -1

    while True:
        unique_labels = np.unique(labels)
        if len(unique_labels) == previous_count:
            break
        previous_count = len(unique_labels)

        sizes = {label: int(np.sum(labels == label)) for label in unique_labels}
        small = [label for label, size in sizes.items() if size < min_cluster_size]
        if not small:
            break

        large = [label for label in unique_labels if label not in small]
        if not large:
            largest = max(sizes, key=sizes.get)
            large = [largest]
            small = [label for label in unique_labels if label != largest]

        centroids = _cluster_centroids(coefficients, labels)
        label_order = np.unique(labels)
        label_to_column = {label: i for i, label in enumerate(label_order)}
        large_columns = np.asarray([label_to_column[label] for label in large], dtype=int)
        large_centroids = centroids[:, large_columns]

        for label in small:
            indices = np.where(labels == label)[0]
            if indices.size == 0:
                continue
            distances = cdist(coefficients[:, indices].T, large_centroids.T, metric=metric)
            nearest = np.argmin(distances, axis=1)
            for index, nearest_index in zip(indices, nearest):
                labels[index] = large[int(nearest_index)]

        labels = _compact_labels(labels)

    return labels


def _cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Permutation-invariant accuracy via Hungarian algorithm."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    true_classes = np.unique(y_true)
    predicted_classes = np.unique(y_pred)
    confusion = np.zeros((len(true_classes), len(predicted_classes)), dtype=int)

    for i, true_class in enumerate(true_classes):
        true_mask = y_true == true_class
        for j, predicted_class in enumerate(predicted_classes):
            confusion[i, j] = np.sum(true_mask & (y_pred == predicted_class))

    rows, columns = linear_sum_assignment(-confusion)
    return float(confusion[rows, columns].sum() / len(y_true))


def relabel_consecutive(labels: np.ndarray) -> np.ndarray:
    """Map arbitrary labels to consecutive integers 0..K-1."""
    labels_arr = np.asarray(labels)
    uniq = np.unique(labels_arr)
    mapping = {old: new for new, old in enumerate(uniq)}
    return np.array([mapping[int(v)] for v in labels_arr], dtype=int)


def cluster_by_threshold(
    coefficients: np.ndarray,
    threshold: float,
    true_labels: Optional[np.ndarray] = None,
    method: str = "components",
    metric: str = "euclidean",
    min_cluster_size: int = 5,
) -> Tuple[np.ndarray, Dict]:
    """Cluster coefficient vectors using a pairwise-distance threshold.

    Parameters
    ----------
    coefficients
        Matrix with shape ``(n_features, n_samples)``. Each column represents
        one sample or subject.
    threshold
        Maximum pairwise distance used to connect or absorb samples.
    true_labels
        Optional labels used only to report permutation-invariant accuracy.
    method
        Either ``"components"`` or ``"greedy-radius"``.
    metric
        Any distance metric accepted by :func:`scipy.spatial.distance.pdist`.
    min_cluster_size
        Clusters smaller than this value are merged into a nearby larger one.
    """
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.ndim != 2:
        raise ValueError("coefficients must have shape (n_features, n_samples).")
    if coefficients.shape[1] == 0:
        raise ValueError("coefficients must contain at least one sample.")
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")
    if min_cluster_size < 1:
        raise ValueError("min_cluster_size must be at least 1.")

    if method == "components":
        labels, details = _component_clustering(coefficients, threshold, metric)
    elif method == "greedy-radius":
        labels, details = _greedy_radius_clustering(coefficients, threshold, metric)
    else:
        raise ValueError("method must be 'components' or 'greedy-radius'.")

    labels = _enforce_min_cluster_size(
        coefficients,
        labels,
        min_cluster_size=min_cluster_size,
        metric=metric,
    )
    labels = _compact_labels(labels)
    clusters = [np.where(labels == label)[0] for label in np.unique(labels)]

    accuracy = np.nan
    if true_labels is not None:
        true_labels = np.asarray(true_labels).ravel()
        if len(true_labels) != coefficients.shape[1]:
            raise ValueError("true_labels must match the number of samples.")
        if len(clusters) == len(np.unique(true_labels)):
            accuracy = _cluster_accuracy(true_labels, labels)

    details.update(
        {
            "method": method,
            "threshold": threshold,
            "n_clusters": len(clusters),
            "accuracy": accuracy,
            "clusters": clusters,
            "min_cluster_size": min_cluster_size,
        }
    )
    return labels, details
