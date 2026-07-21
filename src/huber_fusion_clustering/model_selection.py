"""Tuning-parameter selection for loss-based fusion clustering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import calinski_harabasz_score
from joblib import Parallel, delayed

from .solver import admm_huber_fusion
from .clustering import cluster_by_threshold


ArrayLike = np.ndarray


@dataclass
class CHSelectionResult:
    best_lambda: float
    best_index: int
    best_ch: float
    best_beta_hat: np.ndarray           # shape (d, n)
    best_labels: np.ndarray             # shape (n,)
    best_info: Dict[str, Any]
    history: List[Dict[str, Any]]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _as_1d_float_array(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D array, got shape={arr.shape}.")
    return arr


def initial_subject_ols(xlist: Sequence[ArrayLike], ylist: Sequence[ArrayLike]) -> np.ndarray:
    """
    Compute per-subject OLS coefficients.

    Returns
    -------
    beta_init : np.ndarray, shape (n, d)
        Row i is the OLS coefficient vector for subject i.
    """
    if len(xlist) != len(ylist):
        raise ValueError("xlist and ylist must have the same length.")

    beta_rows: List[np.ndarray] = []
    for i, (x, y) in enumerate(zip(xlist, ylist)):
        x_arr = np.asarray(x, dtype=float)
        y_arr = _as_1d_float_array(y)
        if x_arr.ndim != 2:
            raise ValueError(f"xlist[{i}] must be 2D, got shape={x_arr.shape}.")
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"Row mismatch at subject {i}: X has {x_arr.shape[0]} rows, y has {y_arr.shape[0]} entries."
            )
        coef, *_ = np.linalg.lstsq(x_arr, y_arr, rcond=None)
        beta_rows.append(coef)

    beta_init = np.vstack(beta_rows)
    return beta_init


def relabel_consecutive(labels: Sequence[int]) -> np.ndarray:
    labels_arr = np.asarray(labels)
    uniq = np.unique(labels_arr)
    mapping = {old: new for new, old in enumerate(uniq)}
    return np.array([mapping[int(v)] for v in labels_arr], dtype=int)


def merge_small_clusters(
    beta_subject: np.ndarray,
    labels: Sequence[int],
    min_prop: float = 0.05,
) -> np.ndarray:
    """
    Merge tiny clusters into the nearest large cluster.

    This mirrors the spirit of the R `improve()` function:
    - identify clusters with size <= n * min_prop
    - for each tiny cluster, find the nearest large cluster using coefficient-space distance
    - reassign all subjects in the tiny cluster to that large cluster

    Parameters
    ----------
    beta_subject : np.ndarray, shape (n, d)
        Subject-level coefficient vectors. Each row is one subject.
    labels : array-like, shape (n,)
        Cluster labels.
    min_prop : float
        Small-cluster threshold as a fraction of n.
    """
    beta_subject = np.asarray(beta_subject, dtype=float)
    labels = relabel_consecutive(labels)
    n = beta_subject.shape[0]
    min_size = max(1, int(np.floor(n * min_prop)))

    unique_labels, counts = np.unique(labels, return_counts=True)
    small = unique_labels[counts <= min_size]
    large = unique_labels[counts > min_size]

    if small.size == 0 or large.size == 0:
        return labels

    new_labels = labels.copy()
    for s_lab in small:
        s_idx = np.where(new_labels == s_lab)[0]
        if s_idx.size == 0:
            continue

        s_center = beta_subject[s_idx[0]]
        best_large = None
        best_dist = np.inf
        for l_lab in large:
            l_idx = np.where(new_labels == l_lab)[0]
            if l_idx.size == 0:
                continue
            l_center = beta_subject[l_idx[0]]
            dist = np.linalg.norm(s_center - l_center)
            if dist < best_dist:
                best_dist = dist
                best_large = int(l_lab)

        if best_large is not None:
            new_labels[s_idx] = best_large

    return relabel_consecutive(new_labels)


def compute_ch_index(
    beta_init_subject: np.ndarray,
    labels: Sequence[int],
    band: Optional[float] = None,
    merge_tiny: bool = True,
    min_cluster_prop: float = 0.05,
) -> Tuple[float, np.ndarray]:
    """
    Compute the CH score under the same practical constraints as the R code.

    Returns
    -------
    ch_score : float
    labels_used : np.ndarray
    """
    labels_arr = relabel_consecutive(labels)
    n = beta_init_subject.shape[0]

    if merge_tiny:
        labels_arr = merge_small_clusters(beta_init_subject, labels_arr, min_prop=min_cluster_prop)

    k = np.unique(labels_arr).size
    if band is None:
        band = 5.0 * (n ** (1.0 / 3.0))

    if k <= 1 or k >= n or k > round(float(band), 8):
        return 0.0, labels_arr

    score = float(calinski_harabasz_score(beta_init_subject, labels_arr))
    return score, labels_arr


# -----------------------------------------------------------------------------
# Main selector
# -----------------------------------------------------------------------------

def select_lambda_by_ch(
    xlist: Sequence[ArrayLike],
    ylist: Sequence[ArrayLike],
    Vlist: Sequence[ArrayLike],
    lam_grid: Sequence[float],
    *,
    c: float = 1.345,
    rho: float = 1.0,
    max_admm: int = 500,
    tol_pri: float = 1e-7,
    tol_dual: float = 1e-7,
    max_irls: int = 5,
    cg_tol: float = 1e-8,
    cg_maxit: int = 500,
    verbose: int = 0,
    penalty: str = "mcp",
    gamma: float = 3.0,
    tau_cluster: float = 1e-2,
    min_cluster_size: int = 5,
    cluster_method: str = "components",
    merge_tiny: bool = True,
    min_cluster_prop_for_ch: float = 0.05,
    beta_init_subject: Optional[np.ndarray] = None,
    n_jobs: int = 1,
) -> CHSelectionResult:
    """
    Select lambda by maximizing the Calinski-Harabasz index.

    Parameters
    ----------
    xlist, ylist, Vlist
        Same inputs expected by admm_huber_fusion.
    lam_grid
        Candidate lambda values.
    beta_init_subject
        Optional array of shape (n, d). If None, computed by per-subject OLS.
    Returns
    -------
    CHSelectionResult
    """
    lam_values = [float(v) for v in lam_grid]
    if len(lam_values) == 0:
        raise ValueError("lam_grid must contain at least one lambda value.")

    n = len(xlist)
    if not (len(ylist) == n and len(Vlist) == n):
        raise ValueError("xlist, ylist, and Vlist must have the same length.")

    if beta_init_subject is None:
        beta_init_subject = initial_subject_ols(xlist, ylist)
    else:
        beta_init_subject = np.asarray(beta_init_subject, dtype=float)
        if beta_init_subject.ndim != 2 or beta_init_subject.shape[0] != n:
            raise ValueError(
                "beta_init_subject must have shape (n, d), where n == len(xlist)."
            )

    def evaluate_lambda(idx: int, lam: float) -> Dict[str, Any]:
        admm_kwargs: Dict[str, Any] = dict(
            xlist=xlist,
            ylist=ylist,
            Vlist=Vlist,
            lam=lam,
            c=c,
            rho=rho,
            max_admm=max_admm,
            tol_pri=tol_pri,
            tol_dual=tol_dual,
            max_irls=max_irls,
            cg_tol=cg_tol,
            cg_maxit=cg_maxit,
            verbose=verbose,
            penalty=penalty,
        )
        if penalty.lower() == "mcp":
            admm_kwargs["gamma"] = gamma

        beta_hat, info = admm_huber_fusion(**admm_kwargs)  # expected shape (d, n)
        beta_hat = np.asarray(beta_hat, dtype=float)
        if beta_hat.ndim != 2 or beta_hat.shape[1] != n:
            raise ValueError(
                f"Expected beta_hat shape (d, n) with n={n}, got {beta_hat.shape}."
            )

        labels_raw, cluster_out = cluster_by_threshold(
            beta_hat,
            threshold=tau_cluster,
            true_labels=None,
            method=cluster_method,
            min_cluster_size=min_cluster_size,
        )
        labels_raw = relabel_consecutive(labels_raw)

        ch_value, labels_used = compute_ch_index(
            beta_init_subject=beta_init_subject,
            labels=labels_raw,
            merge_tiny=merge_tiny,
            min_cluster_prop=min_cluster_prop_for_ch,
        )

        k_raw = int(np.unique(labels_raw).size)
        k_used = int(np.unique(labels_used).size)
        record = {
            "index": idx,
            "lambda": lam,
            "ch": ch_value,
            "k_raw": k_raw,
            "k_used": k_used,
            "labels_raw": labels_raw,
            "labels_used": labels_used,
            "beta_hat": beta_hat,
            "info": info,
            "cluster_out": cluster_out,
        }
        return record

    if n_jobs == 1:
        history = [evaluate_lambda(idx, lam) for idx, lam in enumerate(lam_values)]
    else:
        history = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(evaluate_lambda)(idx, lam) for idx, lam in enumerate(lam_values)
        )

    history = sorted(history, key=lambda rec: rec["index"])
    best_idx = -1
    best_score = -np.inf
    best_beta_hat = None
    best_labels = None
    best_info = None

    for record in history:
        idx = int(record["index"])
        score = float(record["ch"])

        if score > best_score or (np.isclose(score, best_score) and idx > best_idx):
            best_idx = idx
            best_score = score
            best_beta_hat = record["beta_hat"]
            best_labels = record["labels_used"]
            best_info = record["info"]

    if best_idx < 0 or best_beta_hat is None or best_labels is None or best_info is None:
        raise RuntimeError("CH selection failed: no valid lambda candidate was evaluated.")

    return CHSelectionResult(
        best_lambda=lam_values[best_idx],
        best_index=best_idx,
        best_ch=best_score,
        best_beta_hat=best_beta_hat,
        best_labels=best_labels,
        best_info=best_info,
        history=history,
    )

__all__ = [
    "CHSelectionResult",
    "initial_subject_ols",
    "merge_small_clusters",
    "compute_ch_index",
    "select_lambda_by_ch",
]
