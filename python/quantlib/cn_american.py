"""
American put option pricing by Crank-Nicolson with PSOR.

Phase 3 Block 4. The pricer is built on the theta-scheme infrastructure
of Block 3 (quantlib.theta_scheme) but replaces the per-step Thomas
solve with a PSOR solve of the discrete LCP

    A V^{n+1} >= B V^n + b_bnd
    V^{n+1}   >= g (the payoff)
    componentwise complementarity

at each time step. This handles the early-exercise feature
automatically: at every node, every step, the projection step in
PSOR enforces V >= g, and the complementarity condition
distinguishes the continuation region (where V > g) from the
exercise region (where V = g).

No Rannacher smoothing: the kink in the payoff that motivated
Rannacher in the European case is absorbed by the projection step
in the American case, since nodes near the strike fall in the
exercise region where V = g exactly.

The boundary conditions for the American put differ from the European:

    V(x_min, tau) = K - K * exp(x_min)    (deep ITM: exercise immediately)
    V(x_max, tau) = 0                      (deep OTM: worthless)

The lower BC has no time decay because exercise is immediate
regardless of tau, in contrast to the European
K * exp(-r*tau) - K * exp(x_min).
"""

import numpy as np

from quantlib.pde import build_grid, put_initial_condition
from quantlib.theta_scheme import theta_coeffs
from quantlib.psor import psor_solve


# ---------------------------------------------------------------------------
# American-specific boundary conditions
# ---------------------------------------------------------------------------
def american_put_boundary_lower(grid):
    """Dirichlet boundary for the American put at x = x_min.

    Deep in-the-money: exercise immediately and receive K - S =
    K - K * exp(x_min). No time decay, since the holder always
    chooses to exercise rather than wait.
    """
    return np.full(grid.M + 1, grid.K - grid.K * np.exp(grid.x_min))


def american_put_boundary_upper(grid):
    """Dirichlet boundary for the American put at x = x_max.

    Deep out-of-the-money: option is essentially worthless, V = 0.
    """
    return np.zeros(grid.M + 1)


# ---------------------------------------------------------------------------
# Time-marching kernel
# ---------------------------------------------------------------------------
def _cn_american_march(grid, V0, bc_lower, bc_upper, *,
                       omega, tol_abs, tol_rel, max_iter):
    """Time-march V0 using CN, solving the LCP at each step by PSOR.

    Returns the solution at tau = T plus diagnostic statistics on
    the PSOR iteration counts.
    """
    N, M = grid.N, grid.M
    dx, dtau = grid.dx, grid.dtau
    sigma, r, mu = grid.sigma, grid.r, grid.mu
    K = grid.K

    # CN coefficients: theta = 0.5.
    c = theta_coeffs(0.5, sigma, r, mu, dtau, dx)

    # Pre-compute the LHS tridiagonal entries once.
    n_int = N - 1
    sub  = np.full(n_int - 1, c.beta_minus)
    diag = np.full(n_int,     c.beta_zero)
    sup  = np.full(n_int - 1, c.beta_plus)

    # The obstacle on interior nodes is the put payoff sampled there.
    obstacle = np.maximum(K - K * np.exp(grid.xs[1:N]), 0.0)

    V = V0.astype(np.float64, copy=True)

    # Diagnostic: track iteration counts per time step.
    iter_counts = np.empty(M, dtype=np.int64)

    for n in range(M):
        # Build RHS: same as in Block 3's theta_march.
        rhs = (c.gamma_minus * V[0:N-1]
             + c.gamma_zero  * V[1:N]
             + c.gamma_plus  * V[2:N+1])
        rhs[0]  -= c.beta_minus * bc_lower[n + 1]
        rhs[-1] -= c.beta_plus  * bc_upper[n + 1]

        # Warm start from V^n on the interior. The previous solution
        # is feasible (>= obstacle by construction), so this is valid.
        x0 = V[1:N].copy()

        # Solve the LCP.
        V_int_new, n_iter = psor_solve(
            sub, diag, sup, rhs, obstacle,
            omega=omega, tol_abs=tol_abs, tol_rel=tol_rel,
            max_iter=max_iter, x0=x0,
        )

        V[1:N] = V_int_new
        V[0]   = bc_lower[n + 1]
        V[N]   = bc_upper[n + 1]

        iter_counts[n] = n_iter

    return V, iter_counts


# ---------------------------------------------------------------------------
# High-level pricer
# ---------------------------------------------------------------------------
def cn_american_put(S, K, r, sigma, T, *, N, M, n_sigma=4.0,
                    omega=1.2, tol_abs=1e-8, tol_rel=1e-7,
                    max_iter=10000, return_diagnostics=False):
    """Price an American put by Crank-Nicolson with PSOR.

    Parameters
    ----------
    S, K, r, sigma, T : float
        Standard Black-Scholes parameters.
    N : int (keyword-only)
        Number of spatial intervals.
    M : int (keyword-only)
        Number of time intervals.
    n_sigma : float, default 4.0
        Half-width of the truncated x-domain.
    omega : float, default 1.2
        PSOR relaxation parameter; must be in (0, 2).
    tol_abs, tol_rel : float
        PSOR convergence tolerances.
    max_iter : int, default 10000
        PSOR iteration limit per time step.
    return_diagnostics : bool, default False
        If True, return (price, dict) where dict contains the
        per-time-step PSOR iteration counts. Useful for studying
        the omega-sweep.

    Returns
    -------
    float (or (float, dict) if return_diagnostics)
        The American put price, plus optionally a diagnostics dict.
    """
    if S <= 0:
        raise ValueError(f"S must be positive, got {S}")

    grid = build_grid(N=N, M=M, T=T, sigma=sigma, r=r, K=K, n_sigma=n_sigma)
    V0    = put_initial_condition(grid.xs, grid.K)
    bc_lo = american_put_boundary_lower(grid)
    bc_hi = american_put_boundary_upper(grid)

    V_final, iter_counts = _cn_american_march(
        grid, V0, bc_lo, bc_hi,
        omega=omega, tol_abs=tol_abs, tol_rel=tol_rel,
        max_iter=max_iter,
    )

    price = _interpolate_at_spot(grid, V_final, S)

    if return_diagnostics:
        return price, {
            "iter_counts": iter_counts,
            "total_iterations": int(np.sum(iter_counts)),
            "mean_iter_per_step": float(np.mean(iter_counts)),
            "max_iter_per_step": int(np.max(iter_counts)),
            "final_solution": V_final,
            "grid": grid,
        }
    return price


# ---------------------------------------------------------------------------
# Spot recovery (same as European pricers)
# ---------------------------------------------------------------------------
def _interpolate_at_spot(grid, V_final, S):
    """Linear interpolation at x_0 = ln(S / K)."""
    x_0 = np.log(S / grid.K)
    if x_0 < grid.x_min or x_0 > grid.x_max:
        raise ValueError(
            f"Spot S = {S} lies outside the truncated domain "
            f"[{grid.K * np.exp(grid.x_min):.4f}, "
            f"{grid.K * np.exp(grid.x_max):.4f}]."
        )
    return float(np.interp(x_0, grid.xs, V_final))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Standard ATM American put. Reference value from a high-N CRR
    # binomial: roughly 5.92 (we will cross-validate in the suite).
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    print(f"American put: S={S}, K={K}, r={r}, sigma={sigma}, T={T}")

    price, diag = cn_american_put(
        S, K, r, sigma, T, N=400, M=200,
        return_diagnostics=True,
    )
    print(f"  CN-PSOR price:           {price:.6f}")
    print(f"  Total PSOR iterations:   {diag['total_iterations']}")
    print(f"  Mean iter / time step:   {diag['mean_iter_per_step']:.1f}")
    print(f"  Max iter / time step:    {diag['max_iter_per_step']}")

    # Cross-check against binomial.
    from quantlib.american import binomial_american_put
    p_crr = binomial_american_put(S, K, r, sigma, T, n_steps=2000)
    print(f"  CRR n=2000 reference:    {p_crr:.6f}")
    print(f"  |CN-PSOR - CRR|:         {abs(price - p_crr):.4e}")
