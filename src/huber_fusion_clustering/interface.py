from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, rand_score

from .model_selection import select_lambda_by_ch


@dataclass
class ADMMClusterConfig:
    lam_grid: Sequence[float] = field(
        default_factory=lambda: np.linspace(0.0001, 1.0, 30).tolist()
    )
    df: int = 4
    degree: int = 2
    intercept: bool = True
    huber_c: float = 0.75
    admm_rho: float = 1.0
    max_admm: int = 500
    tol_pri: float = 1e-7
    tol_dual: float = 1e-7
    max_irls: int = 5
    cg_tol: float = 1e-8
    cg_maxit: int = 500
    penalty: str = "mcp"
    gamma_mcp: float = 3.0
    tau_cluster: float = 1e-3
    min_cluster_size: int = 2
    cluster_method: str = "components"
    merge_tiny_for_ch: bool = False
    min_cluster_prop_for_ch: float = 0.0
    ch_n_jobs: int = -1
    verbose: int = 0
    covariance_method: str = "ar1"
    covariance_time_scale: float = 6.0
    covariance_ridge: float = 1e-5
    covariance_trim: float = 0.25


@dataclass
class ADMMClusterResult:
    labels: np.ndarray
    beta_hat: np.ndarray
    best_lambda: float
    best_ch: float
    n_clusters: int
    metrics: Dict[str, float]
    history: Sequence[Dict[str, Any]]
    info: Dict[str, Any]
    xlist: Sequence[np.ndarray]
    ylist: Sequence[np.ndarray]
    vlist: Sequence[np.ndarray]

    def fitted_values(self) -> Sequence[np.ndarray] | np.ndarray:
        values = [self.xlist[i] @ self.beta_hat[:, i] for i in range(len(self.xlist))]
        try:
            return np.vstack(values)
        except ValueError:
            return values

    def coefficient_blocks(self) -> np.ndarray:
        """Return additive-spline estimates as ``(subject, predictor, basis)``."""
        design_info = self.info.get("design", {})
        if design_info.get("kind") != "additive_spline":
            raise ValueError(
                "coefficient_blocks() is available only when the model is fitted "
                "with covariates=."
            )

        n_basis = int(design_info["n_basis"])
        n_covariates = int(design_info["n_covariates"])
        offset = int(bool(design_info["has_intercept"]))
        expected_rows = offset + n_basis * n_covariates
        if self.beta_hat.shape[0] != expected_rows:
            raise RuntimeError("Fitted coefficient dimensions do not match design metadata.")
        spline_coefs = self.beta_hat[offset:, :].T
        return spline_coefs.reshape(self.beta_hat.shape[1], n_covariates, n_basis)

    def subject_intercepts(self) -> np.ndarray:
        """Return subject-specific intercepts from an additive-spline fit."""
        design_info = self.info.get("design", {})
        if design_info.get("kind") != "additive_spline":
            raise ValueError(
                "subject_intercepts() is available only when the model is fitted "
                "with covariates=."
            )
        if not design_info.get("has_intercept", False):
            raise ValueError("The fitted additive-spline model has no intercept.")
        return self.beta_hat[0, :].copy()


class HuberFusionClusterer:
    """High-level estimator for Huber-loss fusion clustering."""

    def __init__(self, config: Optional[ADMMClusterConfig] = None) -> None:
        self.config = config or ADMMClusterConfig()
        self.result_: Optional[ADMMClusterResult] = None
        self.labels_: Optional[np.ndarray] = None

    def fit(
        self,
        y: Any,
        t: Optional[Any] = None,
        X: Optional[Any] = None,
        V: Optional[Any] = None,
        covariates: Optional[Any] = None,
    ) -> "HuberFusionClusterer":
        """Fit the model and store the fitted result."""
        self.result_ = fit_admm_cluster(
            y=y,
            t=t,
            X=X,
            V=V,
            covariates=covariates,
            config=self.config,
        )
        self.labels_ = self.result_.labels.copy()
        return self

    def fit_predict(
        self,
        y: Any,
        t: Optional[Any] = None,
        X: Optional[Any] = None,
        V: Optional[Any] = None,
        covariates: Optional[Any] = None,
    ) -> np.ndarray:
        """Fit the model and return cluster labels."""
        self.fit(y=y, t=t, X=X, V=V, covariates=covariates)
        if self.labels_ is None:
            raise RuntimeError("The fitted labels are unavailable.")
        return self.labels_.copy()


def fit_admm_cluster(
    y: Any,
    t: Optional[Any] = None,
    X: Optional[Any] = None,
    V: Optional[Any] = None,
    covariates: Optional[Any] = None,
    *,
    config: Optional[ADMMClusterConfig] = None,
    true_labels: Optional[Sequence[int]] = None,
    beta_init_subject: Optional[np.ndarray] = None,
) -> ADMMClusterResult:
    """
    Fit the ADMM fusion clustering algorithm on subject-level longitudinal data.

    Parameters
    ----------
    y
        Response data. Accepts a list of length n, or an array with shape (n, T).
    t
        Time points used to build the B-spline design matrix. Accepts one
        common 1D array or a list of subject-specific 1D arrays.
    X
        Optional prebuilt design data. When supplied, it overrides t. Accepts
        one common matrix with shape (T, d), a list of n matrices, or an array
        with shape (n, T, d). This is kept for advanced/manual use.
    V
        Optional covariance matrices. If supplied, these matrices are used
        directly. If None, an AR(1) working covariance is estimated from
        initial ridge residuals.
    covariates
        Optional raw predictors for a multivariable additive-spline model.
        Accepts a common matrix with shape (T, p), a list of n matrices with
        shape (T_i, p), or an array with shape (n, T, p). Each predictor is
        expanded separately, producing [1, B_1(x_1), ..., B_p(x_p)].
    true_labels
        Optional labels for evaluation only. The algorithm does not need k.
    """
    cfg = config or ADMMClusterConfig()
    ylist = _coerce_ylist(y)
    if covariates is not None and X is not None:
        raise ValueError("X and covariates cannot be supplied together.")
    if covariates is not None and t is not None:
        raise ValueError("t is not used with covariates; omit t for additive splines.")

    if X is None and covariates is None and _looks_like_x_input(t):
        X = t
        t = None

    if covariates is not None:
        xlist, design_info = _build_additive_spline_xlist(
            covariates,
            n=len(ylist),
            ylist=ylist,
            cfg=cfg,
        )
    elif X is None:
        if t is None:
            raise ValueError("t is required when X is not supplied.")
        xlist = _build_xlist_from_t(t, n=len(ylist), ylist=ylist, cfg=cfg)
        design_info = {
            "kind": "time_spline",
            "n_basis": int(xlist[0].shape[1]),
            "n_covariates": 0,
            "n_coefficient_functions": 1,
        }
    else:
        xlist = _coerce_xlist(X, n=len(ylist), ylist=ylist)
        design_info = {
            "kind": "prebuilt",
            "n_columns": int(xlist[0].shape[1]),
        }
    cov_info: Dict[str, Any] = {"source": "provided" if V is not None else "estimated"}
    if V is None:
        if design_info["kind"] == "time_spline":
            tlist_for_cov = _coerce_tlist(t, n=len(ylist), ylist=ylist)
        else:
            tlist_for_cov = _default_time_from_y(ylist)
        vlist, cov_info = estimate_covariance_from_residuals(
            xlist=xlist,
            ylist=ylist,
            tlist=tlist_for_cov,
            method=cfg.covariance_method,
            time_scale=cfg.covariance_time_scale,
            ridge=cfg.covariance_ridge,
            trim=cfg.covariance_trim,
        )
    else:
        vlist = _coerce_vlist(V, n=len(ylist), ylist=ylist)
    _validate_inputs(xlist, ylist, vlist)

    ch_res = select_lambda_by_ch(
        xlist=xlist,
        ylist=ylist,
        Vlist=vlist,
        lam_grid=cfg.lam_grid,
        c=cfg.huber_c,
        rho=cfg.admm_rho,
        max_admm=cfg.max_admm,
        tol_pri=cfg.tol_pri,
        tol_dual=cfg.tol_dual,
        max_irls=cfg.max_irls,
        cg_tol=cfg.cg_tol,
        cg_maxit=cfg.cg_maxit,
        verbose=cfg.verbose,
        penalty=cfg.penalty,
        gamma=cfg.gamma_mcp,
        tau_cluster=cfg.tau_cluster,
        min_cluster_size=cfg.min_cluster_size,
        cluster_method=cfg.cluster_method,
        merge_tiny=cfg.merge_tiny_for_ch,
        min_cluster_prop_for_ch=cfg.min_cluster_prop_for_ch,
        beta_init_subject=beta_init_subject,
        n_jobs=cfg.ch_n_jobs,
    )

    labels = np.asarray(ch_res.best_labels, dtype=int)
    metrics = _cluster_metrics(true_labels=true_labels, labels=labels)
    return ADMMClusterResult(
        labels=labels,
        beta_hat=np.asarray(ch_res.best_beta_hat, dtype=float),
        best_lambda=float(ch_res.best_lambda),
        best_ch=float(ch_res.best_ch),
        n_clusters=int(np.unique(labels).size),
        metrics=metrics,
        history=ch_res.history,
        info={**ch_res.best_info, "covariance": cov_info, "design": design_info},
        xlist=xlist,
        ylist=ylist,
        vlist=vlist,
    )


def _coerce_ylist(y: Any) -> list[np.ndarray]:
    if isinstance(y, (list, tuple)):
        ylist = [np.asarray(yi, dtype=float).reshape(-1) for yi in y]
    else:
        arr = np.asarray(y, dtype=float)
        if arr.ndim != 2:
            raise ValueError("y must be a list of 1D arrays or a 2D array with shape (n, T).")
        ylist = [arr[i, :].reshape(-1) for i in range(arr.shape[0])]

    if not ylist:
        raise ValueError("y must contain at least one subject.")
    return ylist


def _coerce_xlist(X: Any, *, n: int, ylist: Sequence[np.ndarray]) -> list[np.ndarray]:
    if isinstance(X, (list, tuple)):
        if len(X) != n:
            raise ValueError("When X is a list, len(X) must equal the number of subjects.")
        return [np.asarray(xi, dtype=float) for xi in X]

    arr = np.asarray(X, dtype=float)
    if arr.ndim == 2:
        return [arr.copy() for _ in range(n)]
    if arr.ndim == 3:
        if arr.shape[0] != n:
            raise ValueError("When X is 3D, X.shape[0] must equal the number of subjects.")
        return [arr[i, :, :] for i in range(n)]

    raise ValueError("X must be a 2D common design matrix, a 3D array, or a list of matrices.")


def _looks_like_x_input(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        if not value:
            return False
        return np.asarray(value[0]).ndim == 2
    return np.asarray(value).ndim in (2, 3)


def _build_xlist_from_t(
    t: Any,
    *,
    n: int,
    ylist: Sequence[np.ndarray],
    cfg: ADMMClusterConfig,
) -> list[np.ndarray]:
    tlist = _coerce_tlist(t, n=n, ylist=ylist)
    return [
        create_bspline_basis_manual(ti, df=cfg.df, degree=cfg.degree, intercept=cfg.intercept)
        for ti in tlist
    ]


def _build_additive_spline_xlist(
    covariates: Any,
    *,
    n: int,
    ylist: Sequence[np.ndarray],
    cfg: ADMMClusterConfig,
) -> tuple[list[np.ndarray], Dict[str, Any]]:
    covariate_list = _coerce_covariate_list(covariates, n=n, ylist=ylist)
    n_covariates = int(covariate_list[0].shape[1])

    bounds: list[tuple[float, float]] = []
    for j in range(n_covariates):
        pooled_values = np.concatenate([Zi[:, j] for Zi in covariate_list])
        lower_bound = float(np.min(pooled_values))
        upper_bound = float(np.max(pooled_values))
        if lower_bound == upper_bound:
            raise ValueError(f"covariate x{j + 1} must contain at least two distinct values.")
        bounds.append((lower_bound, upper_bound))

    xlist: list[np.ndarray] = []
    n_basis: Optional[int] = None
    for Zi in covariate_list:
        blocks = [np.ones((Zi.shape[0], 1), dtype=float)] if cfg.intercept else []
        for j, (lower_bound, upper_bound) in enumerate(bounds):
            basis = create_bspline_basis_manual(
                Zi[:, j],
                # Build one extra basis and drop its bias column so each
                # predictor retains exactly cfg.df identifiable spline terms.
                df=cfg.df + 1,
                degree=cfg.degree,
                intercept=False,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
            if n_basis is None:
                n_basis = int(basis.shape[1])
            blocks.append(basis)
        xlist.append(np.concatenate(blocks, axis=1))

    block_order = ["intercept"] if cfg.intercept else []
    block_order.extend(f"x{j + 1}" for j in range(n_covariates))
    return xlist, {
        "kind": "additive_spline",
        "n_basis": int(n_basis or 0),
        "n_covariates": n_covariates,
        "has_intercept": bool(cfg.intercept),
        "block_order": block_order,
        "covariate_bounds": [[lower, upper] for lower, upper in bounds],
    }


def _coerce_covariate_list(
    covariates: Any,
    *,
    n: int,
    ylist: Sequence[np.ndarray],
) -> list[np.ndarray]:
    if isinstance(covariates, (list, tuple)):
        is_subject_list = (
            len(covariates) == n
            and len(covariates) > 0
            and np.asarray(covariates[0]).ndim == 2
        )
        if is_subject_list:
            covariate_list = [np.asarray(Zi, dtype=float) for Zi in covariates]
        else:
            arr = np.asarray(covariates, dtype=float)
            if arr.ndim != 2:
                raise ValueError(
                    "A covariate list must either contain one matrix per subject "
                    "or represent one common 2D matrix."
                )
            covariate_list = [arr.copy() for _ in range(n)]
    else:
        arr = np.asarray(covariates, dtype=float)
        if arr.ndim == 2:
            covariate_list = [arr.copy() for _ in range(n)]
        elif arr.ndim == 3:
            if arr.shape[0] != n:
                raise ValueError(
                    "When covariates is 3D, covariates.shape[0] must equal "
                    "the number of subjects."
                )
            covariate_list = [arr[i, :, :] for i in range(n)]
        else:
            raise ValueError(
                "covariates must be a common 2D matrix, a 3D array with shape "
                "(n, T, p), or a list of subject-specific 2D matrices."
            )

    n_covariates: Optional[int] = None
    for i, (Zi, yi) in enumerate(zip(covariate_list, ylist)):
        if Zi.ndim != 2:
            raise ValueError(f"covariates[{i}] must be 2D.")
        if Zi.shape[0] != yi.shape[0]:
            raise ValueError(f"covariates[{i}] rows must match y[{i}] length.")
        if Zi.shape[1] == 0:
            raise ValueError("covariates must contain at least one predictor.")
        if not np.all(np.isfinite(Zi)):
            raise ValueError(f"covariates[{i}] contains non-finite values.")
        if n_covariates is None:
            n_covariates = int(Zi.shape[1])
        elif Zi.shape[1] != n_covariates:
            raise ValueError("All covariate matrices must have the same number of columns.")
    return covariate_list


def _coerce_tlist(t: Any, *, n: int, ylist: Sequence[np.ndarray]) -> list[np.ndarray]:
    if isinstance(t, (list, tuple)):
        if len(t) != n:
            raise ValueError("When t is a list, len(t) must equal the number of subjects.")
        tlist = [np.asarray(ti, dtype=float).reshape(-1) for ti in t]
    else:
        arr = np.asarray(t, dtype=float)
        if arr.ndim != 1:
            raise ValueError("t must be a 1D common time vector or a list of 1D time vectors.")
        tlist = [arr.copy() for _ in range(n)]

    for i, (ti, yi) in enumerate(zip(tlist, ylist)):
        if ti.shape[0] != yi.shape[0]:
            raise ValueError(f"t[{i}] length must match y[{i}] length.")
    return tlist


def create_bspline_basis_manual(
    t: Any,
    df: int = 4,
    degree: int = 2,
    intercept: bool = True,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
) -> np.ndarray:
    """Build a B-spline design matrix for a one-dimensional time vector."""
    t = np.asarray(t, dtype=float).reshape(-1)
    n = len(t)
    n_knots = df - degree - 1

    if n == 0:
        raise ValueError("t must contain at least one time point.")

    lower = float(t.min()) if lower_bound is None else float(lower_bound)
    upper = float(t.max()) if upper_bound is None else float(upper_bound)
    if upper <= lower:
        raise ValueError("upper_bound must be greater than lower_bound.")
    if np.any(t < lower) or np.any(t > upper):
        raise ValueError("All time points must lie within the spline bounds.")

    if n_knots > 0:
        internal_knots = np.linspace(lower, upper, n_knots + 2)[1:-1]
    else:
        internal_knots = np.array([])

    knots = np.concatenate([
        np.repeat(lower, degree + 1),
        internal_knots,
        np.repeat(upper, degree + 1),
    ])

    k = len(knots)
    n_basis = k - degree - 1
    B = np.zeros((n, n_basis))

    def B_spline(i: int, p: int, t_val: float, knots_arr: np.ndarray) -> float:
        if p == 0:
            return 1.0 if knots_arr[i] <= t_val < knots_arr[i + 1] else 0.0
        denom1 = knots_arr[i + p] - knots_arr[i]
        denom2 = knots_arr[i + p + 1] - knots_arr[i + 1]
        term1 = 0.0
        term2 = 0.0
        if denom1 != 0:
            term1 = (t_val - knots_arr[i]) / denom1 * B_spline(i, p - 1, t_val, knots_arr)
        if denom2 != 0:
            term2 = (knots_arr[i + p + 1] - t_val) / denom2 * B_spline(i + 1, p - 1, t_val, knots_arr)
        return term1 + term2

    for j in range(n):
        for i in range(n_basis):
            if t[j] == upper:
                B[j, i] = 1.0 if i == n_basis - 1 else 0.0
            else:
                B[j, i] = B_spline(i, degree, t[j], knots)

    return B if intercept else B[:, 1:]


def estimate_covariance_from_residuals(
    *,
    xlist: Sequence[np.ndarray],
    ylist: Sequence[np.ndarray],
    tlist: Sequence[np.ndarray],
    method: str = "ar1",
    time_scale: float = 6.0,
    ridge: float = 1e-5,
    trim: float = 0.25,
) -> tuple[list[np.ndarray], Dict[str, Any]]:
    """
    Estimate subject-level working covariance matrices from initial residuals.

    The estimator uses leverage-adjusted ridge residuals, a trimmed variance,
    and adjacent residual products to construct each AR(1) covariance matrix.
    """
    if method != "ar1":
        raise ValueError("Only covariance_method='ar1' is currently supported.")
    if time_scale <= 0:
        raise ValueError("covariance_time_scale must be positive.")

    residuals = _initial_modified_residuals(xlist=xlist, ylist=ylist, ridge=ridge)
    subject_sigma2 = np.array([float(np.mean(ri**2)) for ri in residuals], dtype=float)
    sigma2 = _trimmed_mean(subject_sigma2, trim=trim)
    if not np.isfinite(sigma2) or sigma2 <= 0:
        sigma2 = float(np.mean(subject_sigma2)) if subject_sigma2.size else 1.0
    if not np.isfinite(sigma2) or sigma2 <= 0:
        sigma2 = 1.0

    scaled_tlist = [np.asarray(ti, dtype=float).reshape(-1) / time_scale for ti in tlist]
    adjacent_products: list[np.ndarray] = []
    for ri, ti in zip(residuals, scaled_tlist):
        if len(ri) < 2:
            continue
        prod = (ri[:-1] * ri[1:]) / sigma2
        adjacent = np.isclose(np.diff(ti), 1.0)
        if np.any(adjacent):
            adjacent_products.append(prod[adjacent])

    if adjacent_products:
        rho_values = np.concatenate(adjacent_products)
        all_y = np.concatenate([np.asarray(yi, dtype=float).reshape(-1) for yi in ylist])
        all_t = np.concatenate(scaled_tlist)
        corr = _safe_corr(all_y, all_t)
        if corr > 0:
            rho_values = np.clip(rho_values, 0.0, 1.0)
        else:
            rho_values = np.clip(rho_values, -1.0, 0.0)
        rho = float(np.mean(rho_values))
    else:
        rho = 0.0

    rho = float(np.clip(rho, -0.99, 0.99))
    vlist = [_ensure_spd(sigma2 * _ar1_correlation(ti, rho)) for ti in scaled_tlist]
    info: Dict[str, Any] = {
        "source": "estimated",
        "method": method,
        "sigma2": float(sigma2),
        "rho": rho,
        "time_scale": float(time_scale),
        "adjacent_pairs": int(sum(len(x) for x in adjacent_products)),
    }
    return vlist, info


def _default_time_from_y(ylist: Sequence[np.ndarray]) -> list[np.ndarray]:
    return [np.arange(len(yi), dtype=float) for yi in ylist]


def _initial_modified_residuals(
    *,
    xlist: Sequence[np.ndarray],
    ylist: Sequence[np.ndarray],
    ridge: float,
) -> list[np.ndarray]:
    residuals = []
    for Xi, yi in zip(xlist, ylist):
        Xi = np.asarray(Xi, dtype=float)
        yi = np.asarray(yi, dtype=float).reshape(-1)
        gram = Xi.T @ Xi + ridge * np.eye(Xi.shape[1])
        gram_inv = np.linalg.pinv(gram)
        beta = gram_inv @ Xi.T @ yi
        res = yi - Xi @ beta
        h_diag = np.sum((Xi @ gram_inv) * Xi, axis=1)
        denom = np.sqrt(np.maximum(1.0 - h_diag, 1e-8))
        residuals.append(np.asarray(res / denom, dtype=float))
    return residuals


def _trimmed_mean(values: np.ndarray, *, trim: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    trim = float(np.clip(trim, 0.0, 0.49))
    if trim <= 0:
        return float(np.mean(values))
    values = np.sort(values)
    cut = int(np.floor(values.size * trim))
    if cut == 0 or 2 * cut >= values.size:
        return float(np.mean(values))
    return float(np.mean(values[cut:-cut]))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2 or np.std(a) <= 0 or np.std(b) <= 0:
        return 0.0
    corr = float(np.corrcoef(a, b)[0, 1])
    return corr if np.isfinite(corr) else 0.0


def _ar1_correlation(t: np.ndarray, rho: float) -> np.ndarray:
    t = np.asarray(t, dtype=float).reshape(-1)
    lags = np.abs(t[:, None] - t[None, :])
    if rho >= 0:
        corr = rho ** lags
    else:
        sign = np.sign(rho) ** np.rint(lags)
        corr = sign * (abs(rho) ** lags)
    np.fill_diagonal(corr, 1.0)
    return corr


def _ensure_spd(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    eig_min = float(np.min(np.linalg.eigvalsh(matrix)))
    if eig_min < eps:
        matrix = matrix + (eps - eig_min) * np.eye(matrix.shape[0])
    return matrix


def _coerce_vlist(V: Optional[Any], *, n: int, ylist: Sequence[np.ndarray]) -> list[np.ndarray]:
    if V is None:
        return [np.eye(len(ylist[i]), dtype=float) for i in range(n)]

    if isinstance(V, (list, tuple)):
        if len(V) != n:
            raise ValueError("When V is a list, len(V) must equal the number of subjects.")
        return [np.asarray(vi, dtype=float) for vi in V]

    arr = np.asarray(V, dtype=float)
    if arr.ndim == 2:
        return [arr.copy() for _ in range(n)]
    if arr.ndim == 3:
        if arr.shape[0] != n:
            raise ValueError("When V is 3D, V.shape[0] must equal the number of subjects.")
        return [arr[i, :, :] for i in range(n)]

    raise ValueError("V must be None, a 2D common covariance matrix, a 3D array, or a list of matrices.")


def _validate_inputs(
    xlist: Sequence[np.ndarray],
    ylist: Sequence[np.ndarray],
    vlist: Sequence[np.ndarray],
) -> None:
    n = len(ylist)
    if len(xlist) != n or len(vlist) != n:
        raise ValueError("xlist, ylist, and vlist must have the same length.")

    d0 = None
    for i, (Xi, yi, Vi) in enumerate(zip(xlist, ylist, vlist)):
        if Xi.ndim != 2:
            raise ValueError(f"X[{i}] must be 2D.")
        if yi.ndim != 1:
            raise ValueError(f"y[{i}] must be 1D.")
        if Xi.shape[0] != yi.shape[0]:
            raise ValueError(f"X[{i}] rows must match y[{i}] length.")
        if Vi.shape != (yi.shape[0], yi.shape[0]):
            raise ValueError(f"V[{i}] must have shape ({yi.shape[0]}, {yi.shape[0]}).")
        if d0 is None:
            d0 = Xi.shape[1]
        elif Xi.shape[1] != d0:
            raise ValueError("All X matrices must have the same number of columns.")


def _cluster_metrics(true_labels: Optional[Sequence[int]], labels: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {"k": float(np.unique(labels).size)}
    if true_labels is None:
        return metrics

    y_true = np.asarray(true_labels, dtype=int).reshape(-1)
    if y_true.shape[0] != labels.shape[0]:
        raise ValueError("true_labels length must equal the number of subjects.")

    k_true = int(np.unique(y_true).size)
    k_pred = int(np.unique(labels).size)
    metrics["k_true"] = float(k_true)
    metrics["per_k_true"] = float(k_pred == k_true)
    metrics["ri"] = float(rand_score(y_true, labels))
    metrics["nmi"] = float(normalized_mutual_info_score(y_true, labels, average_method="min"))
    metrics["acc"] = float(_cluster_accuracy(y_true, labels)) if k_pred == k_true else np.nan
    return metrics


def _cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_classes = np.unique(y_true)
    pred_classes = np.unique(y_pred)
    conf = np.zeros((len(true_classes), len(pred_classes)), dtype=int)

    for i, true_class in enumerate(true_classes):
        true_mask = y_true == true_class
        for j, pred_class in enumerate(pred_classes):
            conf[i, j] = int(np.sum(true_mask & (y_pred == pred_class)))

    row_ind, col_ind = linear_sum_assignment(-conf)
    return float(conf[row_ind, col_ind].sum() / y_true.size)
