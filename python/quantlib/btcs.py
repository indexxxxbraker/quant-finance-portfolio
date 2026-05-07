"""
Implicit Backward-Time Centred-Space (BTCS) finite-difference pricer for
European calls and puts.

Phase 3 Block 2. Builds on the grid infrastructure of Block 0
(quantlib.pde) and the tridiagonal solver of Block 2
(quantlib.thomas). The pricer is unconditionally stable: any
combination (N, M) with N >= 2 and M >= 1 produces a finite, convergent
solution. There is no CFL constraint to enforce.

The transformed BS PDE is the same as for FTCS:

    V_tau = (sigma^2 / 2) V_xx + mu V_x - r V,    mu = r - sigma^2 / 2.

BTCS evaluates every spatial stencil at the new time level n+1, giving
the implicit relation

    b_minus * V[j-1]^{n+1}  +  b_zero * V[j]^{n+1}  +  b_plus * V[j+1]^{n+1}
        = V[j]^n,

with constant coefficients

    b_minus = -alpha + mu * dtau / (2 * dx)
    b_zero  =  1 + 2 * alpha + r * dtau
    b_plus  = -alpha - mu * dtau / (2 * dx).

The matrix A formed from these coefficients is constant in time, so
its Thomas factorisation is computed once at setup and applied at
each of the M time steps.

See theory/phase3/block2_btcs.tex for the full analysis.
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
)
from quantlib.thomas import thomas_factor, thomas_solve_factored


# ---------------------------------------------------------------------------
# Internal kernel
# ---------------------------------------------------------------------------
def _btcs_march(grid, V0, bc_lower, bc_upper):
    """Time-march V0 from tau = 0 to tau = T using BTCS.

    Builds the tridiagonal matrix from the BTCS coefficients,
    pre-factors it once, and runs M solves with right-hand sides
    determined by V at the previous step plus the boundary
    contributions. There is no CFL check: BTCS is unconditionally
    stable.

    Parameters
    ----------
    grid : Grid
        From quantlib.pde.build_grid.
    V0 : ndarray, shape (N+1,)
        Initial condition at tau = 0.
    bc_lower, bc_upper : ndarray, shape (M+1,)
        Dirichlet values at x_min and x_max for every time level.

    Returns
    -------
    ndarray, shape (N+1,)
        Solution at tau = T = grid.taus[-1].
    """
    N, M = grid.N, grid.M
    dx, dtau = grid.dx, grid.dtau
    sigma, r, mu = grid.sigma, grid.r, grid.mu

    # Stencil coefficients.
    alpha = 0.5 * sigma * sigma * dtau / (dx * dx)
    nu_signed = mu * dtau / (2.0 * dx)
    b_minus = -alpha + nu_signed
    b_zero  =  1.0 + 2.0 * alpha + r * dtau
    b_plus  = -alpha - nu_signed

    # Tridiagonal matrix on the (N-1) interior nodes j = 1, ..., N-1.
    n_int = N - 1
    sub  = np.full(n_int - 1, b_minus)
    diag = np.full(n_int,     b_zero)
    sup  = np.full(n_int - 1, b_plus)

    # Pre-factor once: A is constant across all M time steps.
    factor = thomas_factor(sub, diag, sup)

    V = V0.astype(np.float64, copy=True)

    for n in range(M):
        # Right-hand side d for the new time level.
        # Interior: d[i] = V[i+1]^n for i = 0, ..., N-2 (one shift).
        d = V[1:N].copy()
        # Boundary contributions: the b_minus * V_0^{n+1} and
        # b_plus * V_N^{n+1} terms are moved to the right-hand side.
        d[0]    -= b_minus * bc_lower[n + 1]
        d[-1]   -= b_plus  * bc_upper[n + 1]

        # Solve. Returns the new interior values.
        V[1:N] = thomas_solve_factored(factor, d)
        # New boundary values are known from the BC arrays.
        V[0] = bc_lower[n + 1]
        V[N] = bc_upper[n + 1]

    return V


# ---------------------------------------------------------------------------
# High-level pricers
# ---------------------------------------------------------------------------
def btcs_european_call(S, K, r, sigma, T, *, N, M, n_sigma=4.0):
    """Price a European call by the implicit BTCS scheme.

    Unlike FTCS, BTCS imposes no CFL condition: any M >= 1 produces a
    finite, convergent solution. For O(dx^2) precision use M ~ N^2;
    for O(dx) precision M ~ N is enough.

    Parameters
    ----------
    S, K, r, sigma, T : float
        Standard Black-Scholes parameters; S, K, sigma, T must be
        positive.
    N : int (keyword-only)
        Number of spatial intervals; must be >= 2.
    M : int (keyword-only)
        Number of time intervals; any M >= 1 is valid.
    n_sigma : float, default 4.0
        Half-width of the truncated x-domain in units of sigma * sqrt(T).

    Returns
    -------
    float
        Approximation to the Black-Scholes call price.
    """
    if S <= 0:
        raise ValueError(f"S must be positive, got {S}")
    grid = build_grid(N=N, M=M, T=T, sigma=sigma, r=r, K=K, n_sigma=n_sigma)
    V0 = call_initial_condition(grid.xs, grid.K)
    bc_lo = call_boundary_lower(grid)
    bc_hi = call_boundary_upper(grid)
    V_final = _btcs_march(grid, V0, bc_lo, bc_hi)
    return _interpolate_at_spot(grid, V_final, S)


def btcs_european_put(S, K, r, sigma, T, *, N, M, n_sigma=4.0):
    """Price a European put by the implicit BTCS scheme. See
    btcs_european_call for parameter documentation.
    """
    if S <= 0:
        raise ValueError(f"S must be positive, got {S}")
    grid = build_grid(N=N, M=M, T=T, sigma=sigma, r=r, K=K, n_sigma=n_sigma)
    V0 = put_initial_condition(grid.xs, grid.K)
    bc_lo = put_boundary_lower(grid)
    bc_hi = put_boundary_upper(grid)
    V_final = _btcs_march(grid, V0, bc_lo, bc_hi)
    return _interpolate_at_spot(grid, V_final, S)


# ---------------------------------------------------------------------------
# Spot recovery
# ---------------------------------------------------------------------------
def _interpolate_at_spot(grid, V_final, S):
    """Linearly interpolate V_final at x_0 = ln(S / K)."""
    x_0 = np.log(S / grid.K)
    if x_0 < grid.x_min or x_0 > grid.x_max:
        raise ValueError(
            f"Spot S = {S} lies outside the truncated domain "
            f"[{grid.K * np.exp(grid.x_min):.4f}, "
            f"{grid.K * np.exp(grid.x_max):.4f}]. "
            f"Increase n_sigma or pick a closer-to-K spot."
        )
    return float(np.interp(x_0, grid.xs, V_final))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Hull example 15.6: S = 42, K = 40, r = 0.10, sigma = 0.20, T = 0.5.
    # Closed-form prices: call = 4.7594, put = 0.8086.
    S, K, r, sigma, T = 42.0, 40.0, 0.10, 0.20, 0.5

    print(f"Hull example 15.6: S={S}, K={K}, r={r}, sigma={sigma}, T={T}\n")

    # First, conventional grid (N=200, M=200), satisfies CFL with margin.
    print("Conventional grid (N=200, M=200):")
    print(f"  BTCS call : {btcs_european_call(S, K, r, sigma, T, N=200, M=200):.6f}  (BS 4.7594)")
    print(f"  BTCS put  : {btcs_european_put (S, K, r, sigma, T, N=200, M=200):.6f}  (BS 0.8086)")

    # Same N, very few time steps. FTCS would refuse this configuration
    # (CFL would be violated). BTCS produces a finite, sensible answer.
    print("\nLarge-dtau test (N=200, M=10): would explode in FTCS")
    print(f"  BTCS call : {btcs_european_call(S, K, r, sigma, T, N=200, M=10):.6f}  (less precise but stable)")
    print(f"  BTCS put  : {btcs_european_put (S, K, r, sigma, T, N=200, M=10):.6f}")
