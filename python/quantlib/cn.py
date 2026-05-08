"""
Crank-Nicolson finite-difference pricer for European options, with
Rannacher time-stepping to suppress strike-induced oscillations.

Phase 3 Block 3. Built on the theta-scheme infrastructure of
quantlib.theta_scheme. The pricer is unconditionally stable and
second-order in time and space, achieving O(dx^2 + dtau^2)
convergence at O(N * M) cost.

Default Rannacher protocol: replace the first 2 CN steps by 4
half-time-step BTCS sub-steps. This damps the high-frequency
oscillations introduced by the kink in the call/put payoff at the
strike, restoring full O(dtau^2) convergence (Giles & Carter 2005).

Set rannacher_steps=0 to disable smoothing and observe the kink-
induced oscillations directly. This is exposed as a diagnostic, not
recommended for production.
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
from quantlib.theta_scheme import theta_march


# ---------------------------------------------------------------------------
# Internal kernel
# ---------------------------------------------------------------------------
def _cn_march(grid, V0, bc_lower, bc_upper, *, rannacher_steps=2):
    """Time-march V0 from tau=0 to tau=T using Crank-Nicolson with
    optional Rannacher warm-up.

    Parameters
    ----------
    grid : Grid
    V0 : ndarray
    bc_lower, bc_upper : ndarray of shape (M+1,)
    rannacher_steps : int, default 2
        Number of full-CN-time-step warm-ups, each of which is
        replaced by 2 half-time-step BTCS sub-steps. With
        rannacher_steps=2, the protocol takes 4 BTCS half-steps
        followed by (M - 2) CN steps. Setting 0 disables smoothing.

    Returns
    -------
    ndarray, shape (N+1,)
        Solution at tau = T.

    Raises
    ------
    ValueError
        If rannacher_steps < 0 or > M (cannot warm up more steps than
        we have time for).
    """
    M = grid.M
    if rannacher_steps < 0:
        raise ValueError(
            f"rannacher_steps must be non-negative, got {rannacher_steps}"
        )
    if rannacher_steps > M:
        raise ValueError(
            f"rannacher_steps={rannacher_steps} exceeds M={M}; "
            f"cannot warm up more time steps than the grid has"
        )

    # If no Rannacher requested: pure CN over all M steps.
    if rannacher_steps == 0:
        return theta_march(grid, V0, 0.5, bc_lower, bc_upper)

    # ---- Rannacher warm-up: 2 * rannacher_steps BTCS half-steps ----
    n_half = 2 * rannacher_steps
    half_dtau = 0.5 * grid.dtau

    # Build half-step BC arrays. We need the boundary values at
    # tau = 0, half_dtau, 2*half_dtau, ..., n_half * half_dtau,
    # which equals tau = 0, 0.5*dtau, ..., rannacher_steps * dtau.
    # The original bc_lower[n] corresponds to tau = n * dtau, so
    # we need to interpolate at the half-step grid.
    half_taus = np.arange(n_half + 1) * half_dtau
    full_taus = grid.taus[:rannacher_steps + 1]   # tau_0, ..., tau_{rannacher_steps}
    bc_lower_half = np.interp(half_taus, full_taus,
                              bc_lower[:rannacher_steps + 1])
    bc_upper_half = np.interp(half_taus, full_taus,
                              bc_upper[:rannacher_steps + 1])

    V_warmed = theta_march(
        grid, V0, 1.0, bc_lower_half, bc_upper_half,
        dtau_override=half_dtau, num_steps=n_half,
    )

    # ---- Remaining CN steps ----
    remaining = M - rannacher_steps
    if remaining == 0:
        return V_warmed

    # The CN portion needs bc arrays starting from tau = rannacher_steps * dtau.
    bc_lower_cn = bc_lower[rannacher_steps:]
    bc_upper_cn = bc_upper[rannacher_steps:]

    return theta_march(
        grid, V_warmed, 0.5, bc_lower_cn, bc_upper_cn,
        num_steps=remaining,
    )


# ---------------------------------------------------------------------------
# High-level pricers
# ---------------------------------------------------------------------------
def cn_european_call(S, K, r, sigma, T, *, N, M, n_sigma=4.0,
                     rannacher_steps=2):
    """Price a European call by Crank-Nicolson with Rannacher smoothing.

    Parameters
    ----------
    S, K, r, sigma, T : float
        Standard Black-Scholes parameters.
    N : int (keyword-only)
        Number of spatial intervals.
    M : int (keyword-only)
        Number of time intervals; any M >= rannacher_steps is valid.
    n_sigma : float, default 4.0
        Half-width of the truncated x-domain.
    rannacher_steps : int, default 2
        Number of CN time steps replaced by half-time-step BTCS
        warm-up. Default 2 follows Giles & Carter (2005). Set to 0
        to observe the kink-induced oscillations.

    Returns
    -------
    float
        CN approximation to the Black-Scholes call price.
    """
    if S <= 0:
        raise ValueError(f"S must be positive, got {S}")
    grid = build_grid(N=N, M=M, T=T, sigma=sigma, r=r, K=K, n_sigma=n_sigma)
    V0 = call_initial_condition(grid.xs, grid.K)
    bc_lo = call_boundary_lower(grid)
    bc_hi = call_boundary_upper(grid)
    V_final = _cn_march(grid, V0, bc_lo, bc_hi,
                        rannacher_steps=rannacher_steps)
    return _interpolate_at_spot(grid, V_final, S)


def cn_european_put(S, K, r, sigma, T, *, N, M, n_sigma=4.0,
                    rannacher_steps=2):
    """Price a European put by Crank-Nicolson. See cn_european_call."""
    if S <= 0:
        raise ValueError(f"S must be positive, got {S}")
    grid = build_grid(N=N, M=M, T=T, sigma=sigma, r=r, K=K, n_sigma=n_sigma)
    V0 = put_initial_condition(grid.xs, grid.K)
    bc_lo = put_boundary_lower(grid)
    bc_hi = put_boundary_upper(grid)
    V_final = _cn_march(grid, V0, bc_lo, bc_hi,
                        rannacher_steps=rannacher_steps)
    return _interpolate_at_spot(grid, V_final, S)


# ---------------------------------------------------------------------------
# Spot recovery
# ---------------------------------------------------------------------------
def _interpolate_at_spot(grid, V_final, S):
    """Linear interpolation at x_0 = ln(S / K)."""
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
    S, K, r, sigma, T = 42.0, 40.0, 0.10, 0.20, 0.5
    print(f"Hull 15.6 (S={S}, K={K}, T={T}): BS reference 4.7594\n")

    for label, R in [("CN with Rannacher (default)", 2),
                     ("CN without Rannacher",       0)]:
        c = cn_european_call(S, K, r, sigma, T, N=200, M=200,
                              rannacher_steps=R)
        p = cn_european_put (S, K, r, sigma, T, N=200, M=200,
                              rannacher_steps=R)
        print(f"  {label}:")
        print(f"    call = {c:.6f}    put = {p:.6f}")
