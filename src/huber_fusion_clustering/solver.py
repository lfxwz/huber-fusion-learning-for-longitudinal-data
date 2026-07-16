"""ADMM solver for Huber-loss fusion clustering."""

import numpy as np
from scipy import linalg
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, cg

from numba import njit


# =========================
# Numba helpers
# =========================
@njit
def huber_weights_numba(r, c=1.345, eps=1e-12):
    """
    w = psi(r)/r for Huber:
      w = 1                    if |r| <= c
      w = c/|r|                if |r| >  c
    """
    w = np.ones_like(r)
    for k in range(r.shape[0]):
        a = abs(r[k])
        if a > c:
            w[k] = c / (a + eps)
    return w


@njit
def group_shrink_numba(v, kappa):
    """
    prox_{kappa*||.||2}(v)  (group lasso / L2 shrink)
    """
    nrm2 = 0.0
    for k in range(v.shape[0]):
        nrm2 += v[k] * v[k]
    nrm = np.sqrt(nrm2)

    if nrm <= kappa:
        return np.zeros_like(v)

    scale = 1.0 - kappa / nrm
    out = np.empty_like(v)
    for k in range(v.shape[0]):
        out[k] = scale * v[k]
    return out


@njit
def group_mcp_numba(v, lam_over_rho, gamma_rho):
    """
    prox of (1/rho)*MCP(||z||2; lam, gamma) at v, group MCP

    Use scaled parameters for ADMM z-step:
      lam_over_rho = lam / rho
      gamma_rho    = gamma * rho   (must be > 1)

    Closed-form (vector) proximal based on r=||v||2:
      if r <= lam_over_rho:
          z = 0
      elif r <= gamma_rho * lam_over_rho:   (== gamma*lam)
          z = ((1 - lam_over_rho/r) / (1 - 1/gamma_rho)) * v
      else:
          z = v
    """
    # r = ||v||2
    r2 = 0.0
    for k in range(v.shape[0]):
        r2 += v[k] * v[k]
    r = np.sqrt(r2)

    lam1 = lam_over_rho
    gam1 = gamma_rho

    if r <= lam1:
        return np.zeros_like(v)

    if r <= gam1 * lam1:
        # scale = (1 - lam1/r) / (1 - 1/gam1)
        denom = 1.0 - 1.0 / gam1
        # denom must be positive; assume gam1>1
        scale = (1.0 - lam1 / r) / denom
        out = np.empty_like(v)
        for k in range(v.shape[0]):
            out[k] = scale * v[k]
        return out

    return v


def compute_Hg_all_variableT(Xt_list, yt_list, Bmat, c):
    """
    Compute per-subject IRLS normal-equation pieces for variable-length data.

    ``Xt_list[i]`` may have shape ``(T_i, d)``, allowing each subject to keep
    their own observation times.
    """
    n = len(Xt_list)
    d = Bmat.shape[1]
    H_all = np.zeros((n, d, d), dtype=np.float64)
    g_all = np.zeros((n, d), dtype=np.float64)

    for i in range(n):
        Xi = Xt_list[i]
        yi = yt_list[i]
        ri = yi - Xi @ Bmat[i]
        wi = huber_weights_numba(ri, c)
        Xw = Xi * wi[:, None]
        H_all[i] = Xi.T @ Xw
        g_all[i] = Xi.T @ (wi * yi)

    return H_all, g_all


# =========================
# Basic utilities
# =========================
def build_complete_graph_edges(n):
    edges = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            edges.append((i, j))
    return edges


def build_incidence_matrix(n, edges):
    """
    A: (m, n) incidence, each row e=(i,j): +1 at i, -1 at j
    """
    m = len(edges)
    row = []
    col = []
    data = []
    for e, (i, j) in enumerate(edges):
        row += [e, e]
        col += [i, j]
        data += [1.0, -1.0]
    A = coo_matrix((data, (row, col)), shape=(m, n)).tocsr()
    return A


def whiten_data(xlist, ylist, Vlist):
    """
    Pre-whiten: Xt = L^{-1}X, yt = L^{-1}y  (L from chol(V))
    """
    n = len(xlist)
    Xt_list, yt_list = [], []
    for i in range(n):
        Xi = xlist[i]
        yi = ylist[i]
        Vi = Vlist[i]
        L = linalg.cholesky(Vi, lower=True)
        Xt = linalg.solve_triangular(L, Xi, lower=True)
        yt = linalg.solve_triangular(L, yi, lower=True)
        Xt_list.append(np.asarray(Xt, dtype=np.float64))
        yt_list.append(np.asarray(yt, dtype=np.float64))
    return Xt_list, yt_list


# =========================
# Main solver
# =========================
def admm_huber_fusion(
    xlist, ylist, Vlist,
    lam=1.0, c=1.345,
    rho=1.0,
    max_admm=200,
    tol_pri=1e-4, tol_dual=1e-4,
    max_irls=5,
    cg_tol=1e-8, cg_maxit=200,
    ridge_H=1e-10,
    irls_stop=0.0,
    penalty="l2",   # "l2" or "mcp"
    gamma=3.0,      # MCP gamma > 1
    verbose=0
):
    """
    Solve:
      min_B  sum_i sum_t huber( (L_i^{-1}(y_i - X_i beta_i))_t )
            + sum_{i<j} P(||beta_i - beta_j||2)
    where:
      P(u)=lam*u for penalty="l2"   (group lasso fusion)
      P(u)=MCP(u; lam, gamma) for penalty="mcp"

    ADMM on edge diffs Z = A B (complete graph).

    Returns:
      B_hat: (d, n)
      info: dict
    """
    n = len(xlist)
    if penalty not in ("l2", "mcp"):
        raise ValueError("penalty must be 'l2' or 'mcp'")
    if penalty == "mcp" and gamma <= 1.0:
        raise ValueError("MCP requires gamma > 1.")

    # 1) whiten. Each subject may have a different number of observations.
    Xt_list, yt_list = whiten_data(xlist, ylist, Vlist)
    _, d = Xt_list[0].shape
    for i in range(n):
        if Xt_list[i].shape[1] != d:
            raise ValueError("All X matrices must have the same number of columns.")

    # 2) graph
    edges = build_complete_graph_edges(n)
    A = build_incidence_matrix(n, edges).tocsr()  # (m,n)
    m = A.shape[0]
    ATA = (A.T @ A).tocsr()                       # (n,n)

    # 3) variables (subject-major)
    Bmat = np.zeros((n, d), dtype=np.float64)     # each row beta_i^T
    Z = np.zeros((m, d), dtype=np.float64)        # edge diffs
    U = np.zeros((m, d), dtype=np.float64)        # scaled dual

    def pack(M):   # (n,d) -> (n*d,)
        return M.reshape(-1)

    def unpack(v): # (n*d,) -> (n,d)
        return v.reshape(n, d)

    # -------- beta-step (IRLS + CG) --------
    def solve_beta(Bmat_init, Z, U):
        bvec = pack(Bmat_init).copy()

        for it_irls in range(max_irls):
            B_now = unpack(bvec)

            # compute H_i and g_i subject by subject; supports variable T_i.
            H_all, g_all = compute_Hg_all_variableT(Xt_list, yt_list, B_now, c)

            # ridge stabilize Hi
            if ridge_H > 0:
                for i in range(n):
                    H_all[i] += ridge_H * np.eye(d)

            # rhs = g + rho*A^T(Z-U)
            V_edges = Z - U
            AtV = A.T @ V_edges            # (n,d)
            rhs_mat = g_all + rho * AtV
            rhs = pack(rhs_mat)

            # linear operator: (blockdiag(Hi) + rho*(ATA ⊗ I_d))
            def matvec(x):
                X = unpack(x)              # (n,d)
                Y = np.zeros_like(X)
                # blockdiag
                for i in range(n):
                    Y[i] = H_all[i] @ X[i]
                # laplacian
                Y += rho * (ATA @ X)
                return pack(Y)

            Lin = LinearOperator((n*d, n*d), matvec=matvec, dtype=np.float64)

            b_new, _ = cg(Lin, rhs, x0=bvec, rtol=cg_tol, atol=0.0, maxiter=cg_maxit)

            if irls_stop > 0:
                denom = max(np.linalg.norm(bvec), 1e-12)
                if np.linalg.norm(b_new - bvec) / denom < irls_stop:
                    bvec = b_new
                    break

            bvec = b_new

        return unpack(bvec)

    # -------- ADMM loop --------
    history = []
    for it in range(max_admm):
        Z_prev = Z.copy()

        # 1) beta-step
        Bmat = solve_beta(Bmat, Z, U)

        # 2) z-step
        Q = (A @ Bmat) + U

        if penalty == "l2":
            kappa = lam / rho
            for e in range(m):
                Z[e] = group_shrink_numba(Q[e], kappa)

        else:  # penalty == "mcp"
            lam1 = lam / rho
            gam1 = gamma * rho  # must be > 1
            if gam1 <= 1.0:
                raise ValueError("gamma*rho must be > 1 for MCP proximal.")
            for e in range(m):
                Z[e] = group_mcp_numba(Q[e], lam1, gam1)

        # 3) u-step
        U = U + (A @ Bmat) - Z

        # 4) diagnostics
        r = (A @ Bmat) - Z
        s = rho * (A.T @ (Z - Z_prev))   # (n,d)

        r_norm = float(np.linalg.norm(r))
        s_norm = float(np.linalg.norm(s))

        eps_pri = float(np.sqrt(m*d) * tol_pri + tol_pri * max(np.linalg.norm(A @ Bmat), np.linalg.norm(Z)))
        eps_dual = float(np.sqrt(n*d) * tol_dual + tol_dual * np.linalg.norm(A.T @ U))

        history.append((r_norm, s_norm, eps_pri, eps_dual))

        if verbose and (it == 0 or (it + 1) % 10 == 0):
            print(
                f"[iter {it + 1:3d}] "
                f"r={r_norm:.3e}  s={s_norm:.3e}  "
                f"eps_pri={eps_pri:.3e}  eps_dual={eps_dual:.3e}"
            )

        if (r_norm <= eps_pri) and (s_norm <= eps_dual):
            break

    B_hat = Bmat.T  # (d,n)
    info = {
        "iter": it + 1,
        "r_norm": r_norm,
        "s_norm": s_norm,
        "eps_pri": eps_pri,
        "eps_dual": eps_dual,
        "history": history,
        "penalty": penalty,
        "gamma": gamma
    }
    return B_hat, info
