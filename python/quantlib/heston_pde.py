"""
Heston model: 2D PDE pricing via Alternating Direction Implicit (ADI).

This module implements the Douglas ADI scheme for the Heston pricing PDE
on a 2D grid in (log-spot, variance), following the operator splitting
of In't Hout & Foulon (2010).

Mathematical setup
------------------
After log-spot transformation X = log(S), time reversal tau = T - t,
and the discount substitution that removes the reaction term, the
Heston PDE for the time-reversed function W(tau, X, v) is

    d_tau W = (r - v/2) d_X W
              + kappa (theta - v) d_v W
              + (v/2) d_XX W
              + rho sigma v d_X_d_v W
              + (sigma^2 / 2) v d_vv W

with initial condition W(0, X, v) = max(exp(X) - K, 0) for a call.

The operator splits as L = L_X + L_v + L_Xv. Douglas treats L_Xv
explicitly and inverts L_X, L_v sequentially, each as a batch of
tridiagonal systems solved by the Thomas algorithm.

See ``theory/phase4/block5_heston_pde.tex`` for the derivation,
boundary condition analysis, and stability discussion.

Public interface
----------------
- ``heston_call_pde``: high-level European call pricer.

References
----------
[InHoutFoulon2010]  In't Hout, K. J.; Foulon, S. (2010). ADI finite
                    difference schemes for option pricing in the
                    Heston model with correlation.
"""

from __future__ import annotations

import numpy as np


# =====================================================================
# Input validation
# =====================================================================

def _validate_heston_params(kappa, theta, sigma, rho, v0):
    if kappa <= 0.0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    if theta <= 0.0:
        raise ValueError(f"theta must be positive, got {theta}")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if not (-1.0 <= rho <= 1.0):
        raise ValueError(f"rho must be in [-1, 1], got {rho}")
    if v0 < 0.0:
        raise ValueError(f"v0 must be non-negative, got {v0}")


def _validate_contract(S0, K, T):
    if S0 <= 0.0:
        raise ValueError(f"S0 must be positive, got {S0}")
    if K <= 0.0:
        raise ValueError(f"K must be positive, got {K}")
    if T <= 0.0:
        raise ValueError(f"T must be positive, got {T}")


def _validate_grid(N_X, N_v, N_tau):
    for name, val in [("N_X", N_X), ("N_v", N_v), ("N_tau", N_tau)]:
        if not isinstance(val, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if val < 4:
            raise ValueError(f"{name} must be at least 4, got {val}")


# =====================================================================
# Vectorised Thomas algorithm
# =====================================================================

def _thomas_batch(a, b, c, d):
    """
    Solve a batch of tridiagonal systems via the Thomas algorithm.

    Each row of (a, b, c, d) is one tridiagonal system. All arrays must
    have the same shape (batch_size, n). Returns x of the same shape.
    The first subdiagonal entry a[:, 0] and the last superdiagonal
    c[:, -1] are unused.
    """
    n = b.shape[-1]
    cp = np.empty_like(c)
    dp = np.empty_like(d)
    cp[..., 0] = c[..., 0] / b[..., 0]
    dp[..., 0] = d[..., 0] / b[..., 0]
    for i in range(1, n):
        denom = b[..., i] - a[..., i] * cp[..., i - 1]
        if i < n - 1:
            cp[..., i] = c[..., i] / denom
        dp[..., i] = (d[..., i] - a[..., i] * dp[..., i - 1]) / denom
    x = np.empty_like(d)
    x[..., -1] = dp[..., -1]
    for i in range(n - 2, -1, -1):
        x[..., i] = dp[..., i] - cp[..., i] * x[..., i + 1]
    return x


# =====================================================================
# Operator coefficients
# =====================================================================
# Coefficients depend on v but not on X. We precompute them once.
#
# L_X applied at (i, j) for interior i:
#   (L_X W)[i, j] = a_X[j] W[i-1, j] + b_X[j] W[i, j] + c_X[j] W[i+1, j]
#
# L_v applied at (i, j) for interior j:
#   (L_v W)[i, j] = a_v[j] W[i, j-1] + b_v[j] W[i, j] + c_v[j] W[i, j+1]

def _build_LX_coefficients(v_grid, dX, r):
    """L_X: (r - v/2) d_X + (v/2) d_XX, centred FD interior coefficients."""
    half_v = 0.5 * v_grid
    drift = r - half_v
    diff  = half_v
    dX2   = dX * dX
    a = diff / dX2 - drift / (2.0 * dX)
    b = -2.0 * diff / dX2
    c = diff / dX2 + drift / (2.0 * dX)
    return a, b, c


def _build_Lv_coefficients(v_grid, dv, kappa, theta, sigma):
    """L_v: kappa (theta - v) d_v + (sigma^2/2) v d_vv, centred FD interior."""
    drift = kappa * (theta - v_grid)
    diff  = 0.5 * sigma * sigma * v_grid
    dv2   = dv * dv
    a = diff / dv2 - drift / (2.0 * dv)
    b = -2.0 * diff / dv2
    c = diff / dv2 + drift / (2.0 * dv)
    return a, b, c


# =====================================================================
# Explicit operator application
# =====================================================================

def _apply_LX(W, a, b, c):
    """
    Apply L_X to W. Output has shape of W.

    Interior i = 1..N_X-1: standard centred stencil.
    Boundaries i = 0, N_X: zero (consistent with d_XX = 0 BC and
    one-sided drift, treated implicitly via the corresponding stencil
    in the implicit solve; for the explicit predictor, zero boundary
    contribution gives a consistent and stable result).
    """
    out = np.zeros_like(W)
    out[1:-1, :] = (a[None, :] * W[:-2, :]
                    + b[None, :] * W[1:-1, :]
                    + c[None, :] * W[2:, :])
    return out


def _apply_Lv(W, a, b, c, kappa, theta, dv, v_grid):
    """
    Apply L_v to W.

    Interior j = 1..N_v-1: standard centred stencil.
    j = 0 (v=0): degenerate PDE in v has only the drift contribution
        kappa theta d_v W; we use forward-difference upwind:
        (L_v W)[i, 0] = kappa theta (W[i, 1] - W[i, 0]) / dv.
    j = N_v (v_max): zero second-derivative; first-derivative drift
        kappa(theta - v_max) is negative (v_max > theta), so we use
        backward-difference upwind:
        (L_v W)[i, -1] = kappa (theta - v_max) (W[i, -1] - W[i, -2]) / dv.
    """
    out = np.zeros_like(W)
    out[:, 1:-1] = (a[None, 1:-1] * W[:, :-2]
                    + b[None, 1:-1] * W[:, 1:-1]
                    + c[None, 1:-1] * W[:, 2:])
    # v = 0 boundary
    out[:, 0] = kappa * theta * (W[:, 1] - W[:, 0]) / dv
    # v = v_max boundary
    v_max = v_grid[-1]
    out[:, -1] = kappa * (theta - v_max) * (W[:, -1] - W[:, -2]) / dv
    return out


def _apply_Lxv(W, dX, dv, rho, sigma, v_grid):
    """
    Apply L_xv = rho sigma v d_X_d_v W with centred 9-point stencil
    on interior (i=1..N_X-1, j=1..N_v-1). Zero on boundary.
    """
    out = np.zeros_like(W)
    coef = rho * sigma * v_grid[1:-1] / (4.0 * dX * dv)
    out[1:-1, 1:-1] = coef[None, :] * (W[2:, 2:] - W[2:, :-2]
                                         - W[:-2, 2:] + W[:-2, :-2])
    return out


# =====================================================================
# Implicit solves
# =====================================================================

def _solve_implicit_LX(RHS, LX_a, LX_b, LX_c, theta_imp, dtau):
    """
    Solve (I - theta_imp dtau L_X) Y = RHS for Y, batched over j.

    System layout: shape (N_v+1, N_X+1) where j is the batch axis.
    Each row j has its own set of (a, b, c) constants from LX_*[j],
    applied uniformly across the i-direction interior.

    Boundaries i = 0, N_X: identity rows (Y = RHS at boundary), which
    gives a stable BC and matches the explicit operator's zero
    contribution there.
    """
    N_X1, N_v1 = RHS.shape
    factor = theta_imp * dtau

    # Build batched tridiagonal arrays of shape (N_v1, N_X1)
    A_sub = np.zeros((N_v1, N_X1))
    A_dia = np.ones((N_v1, N_X1))
    A_sup = np.zeros((N_v1, N_X1))

    # Interior i = 1..N_X-1: each row j has its own a, b, c from LX coefs
    A_sub[:, 1:-1] = (-factor * LX_a)[:, None]   # broadcast a[j] across i
    A_dia[:, 1:-1] = (1.0 - factor * LX_b)[:, None]
    A_sup[:, 1:-1] = (-factor * LX_c)[:, None]
    # Boundaries i=0, N_X: A_sub=0, A_dia=1, A_sup=0 (already set), so Y = RHS

    # RHS layout: transpose to put j as batch axis
    RHS_batched = RHS.T.copy()
    Y_batched = _thomas_batch(A_sub, A_dia, A_sup, RHS_batched)
    return Y_batched.T


def _solve_implicit_Lv(RHS, Lv_a, Lv_b, Lv_c,
                          kappa, theta, dv, v_grid, theta_imp, dtau):
    """
    Solve (I - theta_imp dtau L_v) Y = RHS for Y, batched over i.

    System layout: shape (N_X+1, N_v+1) where i is the batch axis.

    Boundary handling:
    - j = 0: row from the upwind v=0 PDE. The L_v operator at j=0 is
        L_v[0] f = (kappa theta / dv) (f[1] - f[0]).
      So the implicit row reads:
        (1 + factor * kappa theta / dv) Y[0] - (factor * kappa theta / dv) Y[1]
        = RHS[0]
    - j = N_v: row from the upwind v=v_max PDE with backward difference,
        L_v[N] f = kappa (theta - v_max) / dv * (f[N] - f[N-1]).
      Drift kappa (theta - v_max) is negative, so the row is:
        -factor * kappa (theta - v_max) / dv * Y[N-1]
        + (1 + factor * kappa (theta - v_max) / dv) Y[N] = RHS[N].
      But since the drift coefficient is negative, the diagonal entry is
      < 1, which can cause issues with stability. In practice we accept
      this provided dtau is small enough; for typical Heston grids it's
      fine.
    """
    N_X1, N_v1 = RHS.shape
    factor = theta_imp * dtau

    A_sub = np.zeros((N_X1, N_v1))
    A_dia = np.ones((N_X1, N_v1))
    A_sup = np.zeros((N_X1, N_v1))

    # Interior j = 1..N_v-1
    A_sub[:, 1:-1] = -factor * Lv_a[None, 1:-1]
    A_dia[:, 1:-1] = 1.0 - factor * Lv_b[None, 1:-1]
    A_sup[:, 1:-1] = -factor * Lv_c[None, 1:-1]

    # j = 0 (v=0): upwind forward difference, drift = kappa * theta
    # L_v at boundary: (kappa theta / dv) (f[1] - f[0])
    k_th_over_dv = kappa * theta / dv
    A_sub[:, 0] = 0.0
    A_dia[:, 0] = 1.0 + factor * k_th_over_dv
    A_sup[:, 0] = -factor * k_th_over_dv

    # j = N_v (v=v_max): backward difference, drift = kappa(theta - v_max)
    v_max = v_grid[-1]
    drift_max_over_dv = kappa * (theta - v_max) / dv
    # L_v at v_max: drift_max_over_dv * (f[N] - f[N-1])
    A_sub[:, -1] = factor * drift_max_over_dv     # coefficient of f[N-1]
    A_dia[:, -1] = 1.0 - factor * drift_max_over_dv
    A_sup[:, -1] = 0.0

    Y = _thomas_batch(A_sub, A_dia, A_sup, RHS)
    return Y


# =====================================================================
# One Douglas step
# =====================================================================

def _douglas_step(W, dtau, dX, dv, v_grid, kappa, theta, sigma, rho,
                   theta_imp,
                   LX_a, LX_b, LX_c, Lv_a, Lv_b, Lv_c):
    """
    One Douglas ADI step.

    Y0 = W + dtau (L_X + L_v + L_xv) W                  (predictor)
    (I - theta_imp dtau L_X) Y1 = Y0 - theta_imp dtau L_X W
    (I - theta_imp dtau L_v) Y2 = Y1 - theta_imp dtau L_v W
    return Y2.
    """
    LX_W = _apply_LX(W, LX_a, LX_b, LX_c)
    Lv_W = _apply_Lv(W, Lv_a, Lv_b, Lv_c, kappa, theta, dv, v_grid)
    Lxv_W = _apply_Lxv(W, dX, dv, rho, sigma, v_grid)

    Y0 = W + dtau * (LX_W + Lv_W + Lxv_W)

    RHS_X = Y0 - theta_imp * dtau * LX_W
    Y1 = _solve_implicit_LX(RHS_X, LX_a, LX_b, LX_c, theta_imp, dtau)

    RHS_v = Y1 - theta_imp * dtau * Lv_W
    Y2 = _solve_implicit_Lv(RHS_v, Lv_a, Lv_b, Lv_c,
                              kappa, theta, dv, v_grid,
                              theta_imp, dtau)
    return Y2


# =====================================================================
# High-level pricer
# =====================================================================

def heston_call_pde(S0, K, T, kappa, theta, sigma, rho, v0, r,
                     N_X=200, N_v=100, N_tau=100,
                     *,
                     theta_imp=0.5,
                     X_factor=4.0, v_max_factor=5.0):
    """
    Price a European call under Heston via 2D PDE with Douglas ADI.

    Parameters
    ----------
    S0, K, T : float
        Spot, strike, maturity (positive).
    kappa, theta, sigma, rho, v0 : float
        Heston parameters.
    r : float
        Risk-free rate (unconstrained).
    N_X, N_v, N_tau : int
        Grid sizes; default 200, 100, 100. Each must be >= 4.
    theta_imp : float, keyword-only
        Douglas implicitness parameter; default 0.5.
    X_factor, v_max_factor : float, keyword-only
        Truncation factors for the (X, v) domain.

    Returns
    -------
    float
        European call price.

    Notes
    -----
    Error is O(dX^2 + dv^2 + dtau). At default grids the spatial term
    dominates. Halving N_X and N_v should reduce the error by ~4x.
    """
    _validate_heston_params(kappa, theta, sigma, rho, v0)
    _validate_contract(S0, K, T)
    _validate_grid(N_X, N_v, N_tau)

    log_S0 = float(np.log(S0))
    v_max = v_max_factor * theta
    half_width = X_factor * float(np.sqrt(v_max * T))

    X_grid = np.linspace(log_S0 - half_width, log_S0 + half_width, N_X + 1)
    v_grid = np.linspace(0.0, v_max, N_v + 1)

    dX = X_grid[1] - X_grid[0]
    dv = v_grid[1] - v_grid[0]
    dtau = T / N_tau

    LX_a, LX_b, LX_c = _build_LX_coefficients(v_grid, dX, r)
    Lv_a, Lv_b, Lv_c = _build_Lv_coefficients(v_grid, dv, kappa, theta, sigma)

    # Initial condition: W(0, X, v) = max(e^X - K, 0), independent of v
    S_grid = np.exp(X_grid)
    payoff = np.maximum(S_grid - K, 0.0)
    W = np.broadcast_to(payoff[:, None], (N_X + 1, N_v + 1)).copy()

    # Time-stepping
    for _ in range(N_tau):
        W = _douglas_step(W, dtau, dX, dv, v_grid, kappa, theta, sigma, rho,
                            theta_imp,
                            LX_a, LX_b, LX_c, Lv_a, Lv_b, Lv_c)

    # Bilinear interpolation at (log_S0, v0)
    i0 = int(np.searchsorted(X_grid, log_S0) - 1)
    i0 = max(0, min(N_X - 1, i0))
    j0 = int(np.searchsorted(v_grid, v0) - 1)
    j0 = max(0, min(N_v - 1, j0))

    x_frac = (log_S0 - X_grid[i0]) / (X_grid[i0 + 1] - X_grid[i0])
    v_frac = (v0 - v_grid[j0]) / (v_grid[j0 + 1] - v_grid[j0])

    W_interp = (W[i0, j0]         * (1 - x_frac) * (1 - v_frac)
                 + W[i0 + 1, j0]     * x_frac       * (1 - v_frac)
                 + W[i0, j0 + 1]     * (1 - x_frac) * v_frac
                 + W[i0 + 1, j0 + 1] * x_frac       * v_frac)

    # Undo the discount substitution: W = e^{r tau} U, V = e^{-r T} W
    return float(np.exp(-r * T) * W_interp)


# =====================================================================
# Smoke test
# =====================================================================

if __name__ == "__main__":
    S0, K, T = 100.0, 100.0, 0.5
    kappa, theta, sigma, rho = 1.5, 0.04, 0.3, -0.7
    v0, r = 0.04, 0.05

    print("Heston PDE: Douglas ADI smoke test")
    print(f"S0={S0}, K={K}, T={T}, r={r}")
    print(f"kappa={kappa}, theta={theta}, sigma={sigma}, "
          f"rho={rho}, v0={v0}\n")

    import time
    for (N_X, N_v, N_tau) in [(50, 25, 50), (100, 50, 100), (200, 100, 200)]:
        t0 = time.time()
        price = heston_call_pde(S0, K, T, kappa, theta, sigma, rho, v0, r,
                                  N_X, N_v, N_tau)
        elapsed = time.time() - t0
        print(f"N_X={N_X:>3}, N_v={N_v:>3}, N_tau={N_tau:>3}: "
              f"PDE = {price:.6f}  ({elapsed:.2f}s)")

    try:
        from quantlib.heston_fourier import heston_call_lewis
        C_ref = heston_call_lewis(K, T, S0, v0, r, kappa, theta, sigma, rho)
        print(f"\nFourier reference: {C_ref:.6f}")
    except ImportError:
        pass
