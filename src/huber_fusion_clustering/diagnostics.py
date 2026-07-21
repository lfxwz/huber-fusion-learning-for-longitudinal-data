"""Convergence diagnostics for ADMM fusion clustering."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np


def plot_convergence(
    history: Sequence[tuple[float, float, float, float]] | Dict[str, Any],
    *,
    ax: Optional[Any] = None,
    figsize: tuple[float, float] = (10, 4),
    title: str = "ADMM Convergence Diagnostics",
    show: bool = True,
) -> Any:
    """Plot primal and dual residual norms over ADMM iterations.

    Parameters
    ----------
    history
        Either the ``history`` list from ``admm_huber_fusion`` (tuples of
        ``(r_norm, s_norm, eps_pri, eps_dual)``), or the ``info`` dict that
        contains it, or an ``ADMMClusterResult``.
    ax
        Optional matplotlib Axes. If None, a new figure is created.
    figsize
        Figure size when creating a new figure.
    title
        Plot title.
    show
        Whether to call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot_convergence(). "
            "Install it with: pip install matplotlib"
        ) from exc

    # Unpack history from various input types
    hist = _extract_history(history)
    if not hist:
        raise ValueError("history is empty; nothing to plot.")

    iters = np.arange(1, len(hist) + 1)
    r_norms = np.array([h[0] for h in hist])
    s_norms = np.array([h[1] for h in hist])
    eps_pris = np.array([h[2] for h in hist])
    eps_duals = np.array([h[3] for h in hist])

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        fig = ax.figure
        axes = [ax, ax.twinx()] if not hasattr(ax, "_convergence_twin") else [ax, ax._convergence_twin]
        axes[0]._convergence_twin = axes[1]

    # Primal residual
    ax0 = axes[0]
    ax0.semilogy(iters, r_norms, "b-", linewidth=1.2, label=r"$\|r\|$ (primal)")
    ax0.semilogy(iters, eps_pris, "b--", linewidth=0.8, alpha=0.6, label=r"$\epsilon_{\mathrm{pri}}$")
    ax0.set_xlabel("Iteration")
    ax0.set_ylabel("Primal residual norm")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.set_title(title, fontsize=10)
    ax0.grid(True, alpha=0.3)

    # Dual residual
    ax1 = axes[1]
    ax1.semilogy(iters, s_norms, "r-", linewidth=1.2, label=r"$\|s\|$ (dual)")
    ax1.semilogy(iters, eps_duals, "r--", linewidth=0.8, alpha=0.6, label=r"$\epsilon_{\mathrm{dual}}$")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Dual residual norm")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title("", fontsize=10)
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()

    if show:
        plt.show()

    return fig


def print_convergence_summary(
    info: Dict[str, Any],
    *,
    last_k: int = 5,
) -> None:
    """Print a text summary of ADMM convergence.

    Parameters
    ----------
    info
        The ``info`` dict returned by ``admm_huber_fusion``.
    last_k
        Number of final iterations to display.
    """
    history = info.get("history", [])
    if not history:
        print("No convergence history available.")
        return

    print(f"ADMM converged in {info['iter']} iterations")
    print(f"Final primal residual: {info['r_norm']:.3e} (tol: {info['eps_pri']:.3e})")
    print(f"Final dual residual:   {info['s_norm']:.3e} (tol: {info['eps_dual']:.3e})")
    converged = info["r_norm"] <= info["eps_pri"] and info["s_norm"] <= info["eps_dual"]
    print(f"Status: {'CONVERGED' if converged else 'NOT CONVERGED (max iter reached)'}")

    if len(history) > last_k:
        print(f"\nLast {last_k} iterations:")
    else:
        print(f"\nAll {len(history)} iterations:")

    start = max(0, len(history) - last_k)
    print(f"  {'iter':>5s}  {'r_norm':>10s}  {'s_norm':>10s}  {'eps_pri':>10s}  {'eps_dual':>10s}")
    for i in range(start, len(history)):
        r, s, ep, ed = history[i]
        print(f"  {i + 1:5d}  {r:10.3e}  {s:10.3e}  {ep:10.3e}  {ed:10.3e}")


def _extract_history(
    history: Sequence[tuple[float, float, float, float]] | Dict[str, Any],
) -> list[tuple[float, float, float, float]]:
    """Extract the raw history list from various input types."""
    # ADMMClusterResult
    if hasattr(history, "history"):
        result = getattr(history, "result_", None)
        if result is not None and hasattr(result, "info"):
            info = result.info
            if isinstance(info, dict) and "history" in info:
                return list(info["history"])
        hist = getattr(history, "history", None)
        if hist is not None:
            return list(hist)

    # info dict
    if isinstance(history, dict):
        if "history" in history:
            return list(history["history"])
        # Maybe it's already a history list stored under a key
        for value in history.values():
            if isinstance(value, list) and value and isinstance(value[0], (tuple, list)):
                if len(value[0]) == 4:
                    return list(value)
        return []

    # Raw list of tuples
    if isinstance(history, (list, tuple)):
        return list(history)

    return []
