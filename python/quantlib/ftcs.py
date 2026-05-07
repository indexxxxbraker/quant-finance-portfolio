"""
Explicit Forward-Time Centred-Space (FTCS) finite-difference pricer for
European calls and puts.

Phase 3 Block 1. Built on top of the grid infrastructure of Block 0
(quantlib.pde). The pricer marches the discrete Black-Scholes PDE
forward in tau = T - t from the payoff at tau = 0 to the price at
tau = T, then interpolates at the user-supplied spot S to recover
V(S, 0).

The transformed BS PDE solved by all schemes of Phase 3 is

    V_tau = (sigma^2 / 2) V_xx + mu V_x - r V,    mu = r - sigma^2 / 2,

with x = ln(S / K). FTCS evaluates every spatial stencil at the old
time level n, giving the explicit update

    V[j]^{n+1}  =  a_minus * V[j-1]^n  +  a_zero * V[j]^n  +  a_plus * V[j+1]^n,

with constant coefficients

    a_minus = alpha - mu * dtau / (2 * dx)
    a_zero  = 1 - 2 * alpha - r * dtau
    a_plus  = alpha + mu * dtau / (2 * dx)

where alpha = (sigma^2 / 2) * dtau / dx^2 is the Fourier number.

Stability: alpha <= 1/2 is required (CFL). The pricer enforces this
strictly and raises ValueError otherwise. See
theory/phase3/block1_ftcs.tex for the full analysis.
"""

import numpy as np

from quantlib.pde import (
    build_grid,
    call_initial_condition,
    put_initial_condition,
    call_boundary_lower,
    call_boundary_upper,
    put_boundary_lower,
    put_boundary_upper,
    fourier_number,
)


# ---------------------------------------------------------------------------
# Internal time-marching kernel
# ---------------------------------------------------------------------------
def _ftcs_march(grid, V0, bc_lower, bc_upper, *, validate_cfl=True):
    """Time-march V0 from tau = 0 to tau = T using the FTCS scheme.

    This is the lowest-level routine: it does no parameter validation
    beyond an optional CFL check, expects pre-computed Dirichlet
    arrays, and returns the full V array at tau = T.

    Parameters
    ----------
    grid : Grid
        From quantlib.pde.build_grid. Provides dx, dtau, sigma, r, mu,
        and the spatial / temporal nodes.
    V0 : ndarray of shape (N+1,)
        Initial-condition vector at tau = 0.
    bc_lower, bc_upper : ndarray of shape (M+1,)
        Dirichlet values at x_min and x_max for every time level.
    validate_cfl : bool, default True
        Enforce alpha <= 1/2 by raising ValueError. The CFL-violating
        diagnostic script disables this to reproduce the explosion;
        no other caller should pass False.

    Returns
    -------
    ndarray of shape (N+1,)
        Solution at tau = T = grid.taus[-1].

    Raises
    ------
    ValueError
        If validate_cfl is True and alpha > 1/2.
    """
    N, M = grid.N, grid.M
    dx, dtau = grid.dx, grid.dtau
    sigma, r, mu = grid.sigma, grid.r, grid.mu

    alpha = fourier_number(sigma, dtau, dx)
    if validate_cfl and alpha > 0.5:
        raise ValueError(
            f"CFL violated: alpha = {alpha:.6f} > 0.5. "
            f"FTCS will diverge. Increase M (more time steps) or "
            f"decrease N (coarser space) to satisfy "
            f"M / N^2 >= sigma^2 * T / (2 * (x_max - x_min)^2)."
        )

    # Stencil coefficients. Constant in (j, n) thanks to the log-transform.
    nu_signed = mu * dtau / (2.0 * dx)
    a_minus = alpha - nu_signed
    a_zero  = 1.0 - 2.0 * alpha - r * dtau
    a_plus  = alpha + nu_signed

    # Double buffer: one array for the current level, one for the next.
    V = V0.astype(np.float64, copy=True)
    V_new = np.empty_like(V)

    # March forward in tau. After step n, V holds the values at level n+1.
    for n in range(M):
        V_new[0] = bc_lower[n + 1]
        V_new[N] = bc_upper[n + 1]
        # Vectorised stencil over interior nodes j = 1, ..., N-1.
        V_new[1:N] = (a_minus * V[0:N-1]
                    + a_zero  * V[1:N]
                    + a_plus  * V[2:N+1])
        # Swap buffers without copying.
        V, V_new = V_new, V

    return V


# ---------------------------------------------------------------------------
# High-level pricers
# ---------------------------------------------------------------------------
def ftcs_european_call(S, K, r, sigma, T, *, N, M, n_sigma=4.0):
    """Price a European call by the FTCS explicit scheme.

    Parameters
    ----------
    S : float
        Spot price; must be > 0.
    K : float
        Strike; must be > 0.
    r : float
        Risk-free rate (any sign).
    sigma : float
        Volatility; must be > 0.
    T : float
        Time to maturity; must be > 0.
    N : int (keyword-only)
        Number of spatial intervals; must be >= 2.
    M : int (keyword-only)
        Number of time intervals; must be large enough for CFL,
        i.e. alpha = (sigma^2 / 2) * (T / M) * (N / (x_max - x_min))^2
        <= 1/2. The pricer will raise ValueError if M is too small.
    n_sigma : float, default 4.0
        Half-width of the truncated x-domain in units of sigma * sqrt(T).

    Returns
    -------
    float
        Approximation to the Black-Scholes call price.

    Raises
    ------
    ValueError
        If the inputs are invalid (delegated to build_grid) or if the
        CFL condition is violated.
    """
    if S <= 0:
        raise ValueError(f"S must be positive, got {S}")
    grid = build_grid(N=N, M=M, T=T, sigma=sigma, r=r, K=K, n_sigma=n_sigma)
    V0 = call_initial_condition(grid.xs, grid.K)
    bc_lo = call_boundary_lower(grid)
    bc_hi = call_boundary_upper(grid)
    V_final = _ftcs_march(grid, V0, bc_lo, bc_hi)
    return _interpolate_at_spot(grid, V_final, S)


def ftcs_european_put(S, K, r, sigma, T, *, N, M, n_sigma=4.0):
    """Price a European put by the FTCS explicit scheme. See
    ftcs_european_call for parameter documentation.
    """
    if S <= 0:
        raise ValueError(f"S must be positive, got {S}")
    grid = build_grid(N=N, M=M, T=T, sigma=sigma, r=r, K=K, n_sigma=n_sigma)
    V0 = put_initial_condition(grid.xs, grid.K)
    bc_lo = put_boundary_lower(grid)
    bc_hi = put_boundary_upper(grid)
    V_final = _ftcs_march(grid, V0, bc_lo, bc_hi)
    return _interpolate_at_spot(grid, V_final, S)


# ---------------------------------------------------------------------------
# Spot recovery
# ---------------------------------------------------------------------------
def _interpolate_at_spot(grid, V_final, S):
    """Linearly interpolate V_final at x_0 = ln(S / K) to recover the
    price at the user-supplied spot.

    Linear interpolation contributes an O(dx^2) error consistent with
    the spatial discretisation order of FTCS; higher-order interpolants
    would not improve the asymptotic rate.

    Raises ValueError if x_0 falls outside the truncated domain.
    """
    x_0 = np.log(S / grid.K)
    if x_0 < grid.x_min or x_0 > grid.x_max:
        raise ValueError(
            f"Spot S = {S} lies outside the truncated domain "
            f"[{grid.K * np.exp(grid.x_min):.4f}, "
            f"{grid.K * np.exp(grid.x_max):.4f}]. "
            f"Increase n_sigma or pick a closer-to-K spot."
        )
    # np.interp does linear interpolation on a uniformly-spaced array.
    return float(np.interp(x_0, grid.xs, V_final))


# ---------------------------------------------------------------------------
# Helper: minimum M for CFL at a given (N, T, sigma, n_sigma)
# ---------------------------------------------------------------------------
def ftcs_min_M_for_cfl(*, N, T, sigma, n_sigma=4.0, target_alpha=0.4):
    """Smallest M such that alpha = (sigma^2 / 2) * (T / M) / dx^2 <=
    target_alpha. Useful for picking M when prototyping.

    With target_alpha < 0.5 the result has a safety margin: a few
    extra time steps relative to the strict CFL boundary.
    """
    if not (0.0 < target_alpha <= 0.5):
        raise ValueError(
            f"target_alpha must be in (0, 0.5], got {target_alpha}"
        )
    half_width = n_sigma * sigma * np.sqrt(T)
    dx = 2.0 * half_width / N
    dtau_max = 2.0 * target_alpha * dx ** 2 / sigma ** 2
    return int(np.ceil(T / dtau_max))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Hull example 15.6: S = 42, K = 40, r = 0.10, sigma = 0.20, T = 0.5.
    # Closed-form BS price: call = 4.7594, put = 0.8086.
    S, K, r, sigma, T = 42.0, 40.0, 0.10, 0.20, 0.5

    N = 200
    M = ftcs_min_M_for_cfl(N=N, T=T, sigma=sigma, target_alpha=0.4)
    print(f"Hull example 15.6: S={S}, K={K}, r={r}, sigma={sigma}, T={T}")
    print(f"Grid: N={N}, M={M} (target_alpha=0.4)\n")

    c = ftcs_european_call(S, K, r, sigma, T, N=N, M=M)
    p = ftcs_european_put (S, K, r, sigma, T, N=N, M=M)

    print(f"FTCS call price : {c:.6f}  (BS reference 4.7594)")
    print(f"FTCS put  price : {p:.6f}  (BS reference 0.8086)")

    # Put-call parity check: C - P = S - K * exp(-r * T).
    parity_lhs = c - p
    parity_rhs = S - K * np.exp(-r * T)
    print(f"\nPut-call parity:")
    print(f"  C - P              = {parity_lhs:.6f}")
    print(f"  S - K * exp(-r*T)  = {parity_rhs:.6f}")
    print(f"  difference         = {parity_lhs - parity_rhs:.2e}")
