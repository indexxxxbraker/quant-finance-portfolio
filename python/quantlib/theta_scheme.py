"""
Generic theta-scheme finite-difference stepper for the transformed
Black-Scholes PDE.

Phase 3 Block 3. The theta-scheme parametrises a one-parameter family
of FD methods for

    V_tau = (sigma^2 / 2) V_xx + mu V_x - r V

via

    (V^{n+1} - V^n) / dtau
        = theta * L_h V^{n+1} + (1 - theta) * L_h V^n,

where L_h is the centred-difference spatial operator. Special cases:

    theta = 0     FTCS  (explicit, conditional stability)
    theta = 1     BTCS  (implicit, unconditional stability)
    theta = 0.5   Crank-Nicolson (implicit, O(dtau^2) in time)

The stencil at each interior node j = 1, ..., N-1 reads

    beta_- V[j-1]^{n+1} + beta_0 V[j]^{n+1} + beta_+ V[j+1]^{n+1}
        = gamma_- V[j-1]^n + gamma_0 V[j]^n + gamma_+ V[j+1]^n,

with constant coefficients (alpha = sigma^2/2 * dtau / dx^2,
nu = mu * dtau / (2 * dx)):

    beta_-  = -theta     * (alpha - nu)
    beta_0  = 1 + theta * (2 alpha + r dtau)
    beta_+  = -theta     * (alpha + nu)
    gamma_- = +(1-theta) * (alpha - nu)
    gamma_0 = 1 - (1-theta) * (2 alpha + r dtau)
    gamma_+ = +(1-theta) * (alpha + nu)

At each step, V^{n+1} on interior nodes is obtained by tridiagonal
solve of A V_int^{n+1} = (B V_int^n) + boundary terms, where the
boundary terms incorporate Dirichlet values from levels n and n+1.

Used downstream by quantlib.cn (Block 3) and quantlib.psor (Block 4).
"""

from dataclasses import dataclass
import numpy as np

from quantlib.thomas import thomas_factor, thomas_solve_factored


# ---------------------------------------------------------------------------
# Stencil coefficients
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ThetaCoeffs:
    """Pre-computed stencil coefficients for the theta-scheme."""
    beta_minus:  float
    beta_zero:   float
    beta_plus:   float
    gamma_minus: float
    gamma_zero:  float
    gamma_plus:  float


def theta_coeffs(theta, sigma, r, mu, dtau, dx):
    """Compute stencil coefficients of the theta-scheme.

    See module docstring for formulae.
    """
    if not (0.0 <= theta <= 1.0):
        raise ValueError(f"theta must be in [0, 1], got {theta}")

    alpha = 0.5 * sigma * sigma * dtau / (dx * dx)
    nu    = mu * dtau / (2.0 * dx)
    one_minus_theta = 1.0 - theta

    return ThetaCoeffs(
        beta_minus  = -theta           * (alpha - nu),
        beta_zero   = 1.0 + theta      * (2.0 * alpha + r * dtau),
        beta_plus   = -theta           * (alpha + nu),
        gamma_minus = +one_minus_theta * (alpha - nu),
        gamma_zero  = 1.0 - one_minus_theta * (2.0 * alpha + r * dtau),
        gamma_plus  = +one_minus_theta * (alpha + nu),
    )


# ---------------------------------------------------------------------------
# Time-marching kernel
# ---------------------------------------------------------------------------
def theta_march(grid, V0, theta, bc_lower, bc_upper, *,
                dtau_override=None, num_steps=None):
    """Time-march V0 forward using the theta-scheme.

    Parameters
    ----------
    grid : Grid
        From quantlib.pde.build_grid.
    V0 : ndarray, shape (N+1,)
        Initial-condition vector.
    theta : float
        Scheme parameter; must lie in [0, 1].
    bc_lower, bc_upper : ndarray, shape (num_steps + 1,)
        Dirichlet values at x_min and x_max for time levels
        relative to the start of marching: bc[0] is for tau = 0
        of this march, bc[num_steps] for tau = num_steps * dtau.
    dtau_override : float or None
        If provided, use this dtau instead of grid.dtau. Used by
        Rannacher to take half-time-step BTCS warm-up steps.
    num_steps : int or None
        If provided, take this many steps. Else take grid.M steps.
        Used by Rannacher to take a fixed small number of warm-up
        steps regardless of grid.M.

    Returns
    -------
    ndarray, shape (N+1,)
        Solution after num_steps (or grid.M) time steps.

    Notes
    -----
    The LHS matrix A is constant in time: factored once at setup,
    re-applied at every step. Cost per step: O(N). Total: O(N * num_steps).

    For theta = 0 (FTCS) the LHS is the identity and no factorisation
    is needed; the routine takes the explicit-update shortcut.
    """
    N = grid.N
    M = num_steps if num_steps is not None else grid.M
    dx = grid.dx
    dtau = dtau_override if dtau_override is not None else grid.dtau
    sigma, r, mu = grid.sigma, grid.r, grid.mu

    expected = M + 1
    if bc_lower.size != expected or bc_upper.size != expected:
        raise ValueError(
            f"theta_march: bc arrays must have length num_steps + 1 = "
            f"{expected}, got bc_lower={bc_lower.size}, "
            f"bc_upper={bc_upper.size}"
        )

    c = theta_coeffs(theta, sigma, r, mu, dtau, dx)

    is_explicit = (theta == 0.0)
    n_int = N - 1
    factor = None
    if not is_explicit:
        sub  = np.full(n_int - 1, c.beta_minus)
        diag = np.full(n_int,     c.beta_zero)
        sup  = np.full(n_int - 1, c.beta_plus)
        factor = thomas_factor(sub, diag, sup)

    V = V0.astype(np.float64, copy=True)
    rhs = np.empty(n_int, dtype=np.float64)

    for n in range(M):
        # RHS = B * V_int^n + boundary corrections.
        #
        # B is tridiagonal with (gamma_-, gamma_0, gamma_+). Restricted
        # to the (N-1) interior rows, the j-th row of (B V) for the
        # full vector V[0..N] uses V[j-1], V[j], V[j+1]. We can
        # vectorise that exactly the same way as in FTCS, since V here
        # holds V^n at all nodes including boundaries.
        rhs[:] = (c.gamma_minus * V[0:N-1]
                + c.gamma_zero  * V[1:N]
                + c.gamma_plus  * V[2:N+1])

        # Move the implicit boundary terms to the RHS:
        #   beta_- V_0^{n+1}  appears on the LHS for j=1; subtract
        #   beta_+ V_N^{n+1}  appears on the LHS for j=N-1; subtract
        rhs[0]  -= c.beta_minus * bc_lower[n + 1]
        rhs[-1] -= c.beta_plus  * bc_upper[n + 1]

        # Solve the system or take the explicit shortcut.
        if is_explicit:
            V_int_new = rhs   # alias; rhs is fresh each iteration
        else:
            V_int_new = thomas_solve_factored(factor, rhs)

        V[1:N] = V_int_new
        V[0]   = bc_lower[n + 1]
        V[N]   = bc_upper[n + 1]

    return V


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Reproduce BTCS via theta_march with theta=1, and CN via theta=0.5.
    # Hull example 15.6.
    from quantlib.pde import (
        build_grid, call_initial_condition,
        call_boundary_lower, call_boundary_upper,
    )

    S, K, r, sigma, T = 42.0, 40.0, 0.10, 0.20, 0.5
    N, M = 200, 200
    g = build_grid(N=N, M=M, T=T, sigma=sigma, r=r, K=K)
    V0 = call_initial_condition(g.xs, g.K)
    bc_lo = call_boundary_lower(g)
    bc_hi = call_boundary_upper(g)

    V_btcs_via_theta = theta_march(g, V0, 1.0, bc_lo, bc_hi)
    V_cn             = theta_march(g, V0, 0.5, bc_lo, bc_hi)

    x_0 = np.log(S / K)
    c_btcs = float(np.interp(x_0, g.xs, V_btcs_via_theta))
    c_cn   = float(np.interp(x_0, g.xs, V_cn))

    print(f"Hull 15.6 (S={S}, K={K}, T={T}): BS reference 4.7594")
    print(f"  theta=1   (BTCS via theta-stepper): {c_btcs:.6f}")
    print(f"  theta=0.5 (CN, no Rannacher):       {c_cn:.6f}")
