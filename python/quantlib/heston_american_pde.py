"""
Heston model: American put pricing via 2D PDE with operator-splitting projection.

This module extends the European 2D PDE pricer of Block 5 to handle
the early-exercise constraint of American puts. The numerical strategy
is operator splitting: after each Douglas ADI time step (which solves
the European PDE), we project onto the constraint
V(t, S, v) >= max(K - S, 0). This is simpler than running PSOR
iterations within each implicit sweep and gives accurate prices for
typical Heston parameter regimes.

The projection in (X = log S, tau = T - t, W = e^{r tau} U) coordinates
becomes:
    W(tau_{n+1}, X, v) = max(W_provisional, e^{r tau_{n+1}} * payoff(X))
where payoff(X) = max(K - exp(X), 0).

The American constraint applies for puts because they have positive
intrinsic value when S < K. American calls without dividends have the
same value as European calls (no early-exercise premium); they are
covered by Block 5's European pricer.

See ``theory/phase4/block6_heston_calibration_exotics.tex`` Section 4.4
for the rationale of choosing PSOR-on-PDE over Longstaff-Schwartz MC.

The kernel functions (Thomas batched solver, operator coefficients,
operator application, implicit solves, Douglas step) are reproduced
from ``heston_pde.py`` for self-containment. A future refactor may
extract these into a shared private kernel module.

Public interface
----------------
- ``heston_american_put_pde``: American put pricer.

References
----------
[AchdouPironneau2005]  Achdou, Y.; Pironneau, O. (2005). Computational
                        Methods for Option Pricing. SIAM. Section 6
                        on free-boundary problems.
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
# Vectorised Thomas batched solver
# =====================================================================

def _thomas_batch(a, b, c, d):
    """Solve a batch of tridiagonal systems via Thomas; see heston_pde.py."""
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

def _build_LX_coefficients(v_grid, dX, r):
    half_v = 0.5 * v_grid
    drift = r - half_v
    diff  = half_v
    dX2   = dX * dX
    a = diff / dX2 - drift / (2.0 * dX)
    b = -2.0 * diff / dX2
    c = diff / dX2 + drift / (2.0 * dX)
    return a, b, c


def _build_Lv_coefficients(v_grid, dv, kappa, theta, sigma):
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
    out = np.zeros_like(W)
    out[1:-1, :] = (a[None, :] * W[:-2, :]
                    + b[None, :] * W[1:-1, :]
                    + c[None, :] * W[2:, :])
    return out


def _apply_Lv(W, a, b, c, kappa, theta, dv, v_grid):
    out = np.zeros_like(W)
    out[:, 1:-1] = (a[None, 1:-1] * W[:, :-2]
                    + b[None, 1:-1] * W[:, 1:-1]
                    + c[None, 1:-1] * W[:, 2:])
    out[:, 0] = kappa * theta * (W[:, 1] - W[:, 0]) / dv
    v_max = v_grid[-1]
    out[:, -1] = kappa * (theta - v_max) * (W[:, -1] - W[:, -2]) / dv
    return out


def _apply_Lxv(W, dX, dv, rho, sigma, v_grid):
    out = np.zeros_like(W)
    coef = rho * sigma * v_grid[1:-1] / (4.0 * dX * dv)
    out[1:-1, 1:-1] = coef[None, :] * (W[2:, 2:] - W[2:, :-2]
                                         - W[:-2, 2:] + W[:-2, :-2])
    return out


# =====================================================================
# Implicit solves
# =====================================================================

def _solve_implicit_LX(RHS, LX_a, LX_b, LX_c, theta_imp, dtau):
    N_X1, N_v1 = RHS.shape
    factor = theta_imp * dtau
    A_sub = np.zeros((N_v1, N_X1))
    A_dia = np.ones((N_v1, N_X1))
    A_sup = np.zeros((N_v1, N_X1))
    A_sub[:, 1:-1] = (-factor * LX_a)[:, None]
    A_dia[:, 1:-1] = (1.0 - factor * LX_b)[:, None]
    A_sup[:, 1:-1] = (-factor * LX_c)[:, None]
    Y_batched = _thomas_batch(A_sub, A_dia, A_sup, RHS.T.copy())
    return Y_batched.T


def _solve_implicit_Lv(RHS, Lv_a, Lv_b, Lv_c,
                          kappa, theta, dv, v_grid, theta_imp, dtau):
    N_X1, N_v1 = RHS.shape
    factor = theta_imp * dtau
    A_sub = np.zeros((N_X1, N_v1))
    A_dia = np.ones((N_X1, N_v1))
    A_sup = np.zeros((N_X1, N_v1))
    A_sub[:, 1:-1] = -factor * Lv_a[None, 1:-1]
    A_dia[:, 1:-1] = 1.0 - factor * Lv_b[None, 1:-1]
    A_sup[:, 1:-1] = -factor * Lv_c[None, 1:-1]
    k_th_over_dv = kappa * theta / dv
    A_sub[:, 0] = 0.0
    A_dia[:, 0] = 1.0 + factor * k_th_over_dv
    A_sup[:, 0] = -factor * k_th_over_dv
    v_max = v_grid[-1]
    drift_max_over_dv = kappa * (theta - v_max) / dv
    A_sub[:, -1] = factor * drift_max_over_dv
    A_dia[:, -1] = 1.0 - factor * drift_max_over_dv
    A_sup[:, -1] = 0.0
    return _thomas_batch(A_sub, A_dia, A_sup, RHS)


# =====================================================================
# One Douglas step
# =====================================================================

def _douglas_step(W, dtau, dX, dv, v_grid, kappa, theta, sigma, rho,
                   theta_imp,
                   LX_a, LX_b, LX_c, Lv_a, Lv_b, Lv_c):
    LX_W = _apply_LX(W, LX_a, LX_b, LX_c)
    Lv_W = _apply_Lv(W, Lv_a, Lv_b, Lv_c, kappa, theta, dv, v_grid)
    Lxv_W = _apply_Lxv(W, dX, dv, rho, sigma, v_grid)
    Y0 = W + dtau * (LX_W + Lv_W + Lxv_W)
    RHS_X = Y0 - theta_imp * dtau * LX_W
    Y1 = _solve_implicit_LX(RHS_X, LX_a, LX_b, LX_c, theta_imp, dtau)
    RHS_v = Y1 - theta_imp * dtau * Lv_W
    return _solve_implicit_Lv(RHS_v, Lv_a, Lv_b, Lv_c,
                                kappa, theta, dv, v_grid, theta_imp, dtau)


# =====================================================================
# American put pricer
# =====================================================================

def heston_american_put_pde(S0, K, T, kappa, theta, sigma, rho, v0, r,
                              N_X=200, N_v=100, N_tau=100,
                              *,
                              theta_imp=0.5,
                              X_factor=4.0, v_max_factor=5.0):
    """
    Price an American put under Heston via 2D PDE with operator-splitting projection.

    The numerical scheme is: at each time step, advance with one Douglas
    ADI sweep (solving the European Heston PDE), then project onto the
    early-exercise constraint
        V(t, S, v) >= max(K - S, 0),
    which in (X = log S, tau = T - t, W = e^{r tau} U) coordinates becomes
        W(tau_{n+1}, X, v) >= e^{r tau_{n+1}} max(K - exp(X), 0).

    The projection is applied at every time step including the first and
    last; this is the operator-splitting approximation to the
    full PSOR-within-sweep approach. For typical Heston parameter
    regimes, the operator-splitting result is accurate to within the
    grid's discretisation error.

    Parameters
    ----------
    S0, K, T : float
        Spot, strike, maturity (positive).
    kappa, theta, sigma, rho, v0 : float
        Heston parameters.
    r : float
        Risk-free rate (unconstrained).
    N_X, N_v, N_tau : int
        Grid sizes; default (200, 100, 100). Each must be >= 4. American
        options often need finer N_tau than European because of the
        early-exercise boundary's regularity; consider doubling N_tau
        if precision is critical.
    theta_imp : float, keyword-only
        Douglas implicitness; default 0.5.
    X_factor, v_max_factor : float, keyword-only
        Domain truncation factors; defaults (4.0, 5.0).

    Returns
    -------
    float
        American put price.

    Notes
    -----
    The American put price under Heston has no closed-form analogue.
    Sanity-check against the European put (which can be computed via
    put-call parity from the Fourier call): American put >= European
    put. The "early exercise premium" is the difference, typically
    0-5% of the option's value depending on moneyness and maturity.

    Algorithm classification: this is the "explicit projection" or
    "operator-splitting" variant of American option PDE pricing. See
    Achdou & Pironneau (2005) Section 6 for the full PSOR analysis;
    the operator-splitting version trades a small amount of accuracy
    near the exercise boundary for substantial implementation
    simplicity.
    """
    _validate_heston_params(kappa, theta, sigma, rho, v0)
    _validate_contract(S0, K, T)
    _validate_grid(N_X, N_v, N_tau)

    log_S0 = float(np.log(S0))
    v_max  = v_max_factor * theta
    half_width = X_factor * float(np.sqrt(v_max * T))

    X_grid = np.linspace(log_S0 - half_width, log_S0 + half_width, N_X + 1)
    v_grid = np.linspace(0.0, v_max, N_v + 1)

    dX = X_grid[1] - X_grid[0]
    dv = v_grid[1] - v_grid[0]
    dtau = T / N_tau

    LX_a, LX_b, LX_c = _build_LX_coefficients(v_grid, dX, r)
    Lv_a, Lv_b, Lv_c = _build_Lv_coefficients(v_grid, dv, kappa, theta, sigma)

    # Payoff (independent of v): max(K - exp(X), 0)
    S_grid = np.exp(X_grid)
    payoff = np.maximum(K - S_grid, 0.0)

    # Initial condition: W(0, X, v) = payoff(X), broadcast over v
    W = np.broadcast_to(payoff[:, None], (N_X + 1, N_v + 1)).copy()

    # Time-stepping with projection.
    # At step n -> n+1, tau goes from n*dtau to (n+1)*dtau.
    # Constraint at tau_{n+1}: W >= exp(r * tau_{n+1}) * payoff.
    for n in range(N_tau):
        tau_next = (n + 1) * dtau
        W = _douglas_step(W, dtau, dX, dv, v_grid, kappa, theta, sigma, rho,
                            theta_imp,
                            LX_a, LX_b, LX_c, Lv_a, Lv_b, Lv_c)
        # Projection onto the early-exercise constraint
        constraint = np.exp(r * tau_next) * payoff
        W = np.maximum(W, constraint[:, None])

    # Bilinear interpolation at (log_S0, v0)
    i0 = int(np.searchsorted(X_grid, log_S0) - 1)
    i0 = max(0, min(N_X - 1, i0))
    j0 = int(np.searchsorted(v_grid, v0) - 1)
    j0 = max(0, min(N_v - 1, j0))

    x_frac = (log_S0 - X_grid[i0]) / (X_grid[i0 + 1] - X_grid[i0])
    v_frac = (v0 - v_grid[j0]) / (v_grid[j0 + 1] - v_grid[j0])

    W_interp = (W[i0,     j0]     * (1 - x_frac) * (1 - v_frac)
                 + W[i0 + 1, j0]     * x_frac       * (1 - v_frac)
                 + W[i0,     j0 + 1] * (1 - x_frac) * v_frac
                 + W[i0 + 1, j0 + 1] * x_frac       * v_frac)

    # Undo the discount substitution: V = exp(-r T) W
    return float(np.exp(-r * T) * W_interp)


# =====================================================================
# Smoke test entry point
# =====================================================================

if __name__ == "__main__":
    S0, K, T = 100.0, 100.0, 0.5
    kappa, theta, sigma, rho = 1.5, 0.04, 0.3, -0.7
    v0, r = 0.04, 0.05

    print("Heston American put: PDE smoke test")
    print(f"S0={S0}, K={K}, T={T}, r={r}")
    print(f"kappa={kappa}, theta={theta}, sigma={sigma}, "
          f"rho={rho}, v0={v0}\n")

    # Compute European put via put-call parity from Fourier call
    try:
        from quantlib.heston_fourier import heston_call_lewis
        C_eur = heston_call_lewis(K, T, S0, v0, r,
                                   kappa, theta, sigma, rho)
        # Put-call parity: C - P = S - K * exp(-rT)
        P_eur = C_eur - S0 + K * np.exp(-r * T)
        print(f"European put (via parity from Fourier call): {P_eur:.6f}")
    except ImportError:
        P_eur = None

    print()
    import time
    print(f"{'Grid':>20}  {'Am put price':>14}  {'EEP':>10}  {'time(s)':>8}")
    for (N_X, N_v, N_tau) in [(50, 25, 50), (100, 50, 100), (200, 100, 200)]:
        t0 = time.time()
        price = heston_american_put_pde(S0, K, T, kappa, theta, sigma, rho,
                                          v0, r, N_X, N_v, N_tau)
        elapsed = time.time() - t0
        eep = price - P_eur if P_eur is not None else float('nan')
        print(f"  {N_X:>3} x {N_v:>3} x {N_tau:>3}    "
              f"{price:>14.6f}  {eep:>10.4f}  {elapsed:>8.2f}")

    # Sanity: deep OTM put should be close to European (no exercise advantage)
    print()
    print("Sanity check: deep OTM put (K=80, far from exercise region):")
    K_otm = 80.0
    P_am_otm = heston_american_put_pde(
        S0, K_otm, T, kappa, theta, sigma, rho, v0, r, 100, 50, 100)
    if P_eur is not None:
        C_eur_otm = heston_call_lewis(K_otm, T, S0, v0, r,
                                        kappa, theta, sigma, rho)
        P_eur_otm = C_eur_otm - S0 + K_otm * np.exp(-r * T)
        print(f"  American put K={K_otm}: {P_am_otm:.6f}")
        print(f"  European put K={K_otm}: {P_eur_otm:.6f}")
        print(f"  EEP (should be small): {P_am_otm - P_eur_otm:.6f}")
