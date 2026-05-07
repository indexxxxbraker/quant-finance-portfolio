"""
Finite-difference grid infrastructure for the Black-Scholes PDE.

Phase 3 Block 0. This module is the scaffolding on which the pricers
of Blocks 1 (FTCS), 2 (BTCS + Thomas), 3 (Crank-Nicolson) and 4 (PSOR)
are built. It contains no pricer; it constructs the (x, tau) grid,
provides Dirichlet boundary conditions and initial-condition vectors,
and exposes stability-number utilities used by the explicit scheme of
Block 1.

The transformed BS PDE solved by all schemes of Phase 3 is

    V_tau = (sigma^2 / 2) V_xx + mu V_x - r V,    mu = r - sigma^2/2,

with x = ln(S/K) and tau = T - t. Derivation in
theory/phase3/block0_pde_foundations.tex.

Conventions
-----------
* Grid is uniform in both x and tau.
* xs has length N+1: xs[0] = x_min, xs[N] = x_max.
* taus has length M+1: taus[0] = 0 (initial condition, i.e. t = T),
  taus[M] = T (terminal answer, i.e. t = 0).
* Boundary functions return arrays of length M+1 indexed by the time
  level n: result[n] is the Dirichlet value at tau = taus[n].
* The strike K is part of the Grid because the spatial coordinate
  x = ln(S/K) is anchored at K.
"""

from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------------
# Grid container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Grid:
    """Immutable container for the uniform (x, tau) discretisation grid.

    Attributes
    ----------
    N : int
        Number of intervals in space; (N+1) spatial nodes.
    M : int
        Number of intervals in time; (M+1) time levels.
    T : float
        Time to maturity (years).
    sigma : float
        Volatility (constant).
    r : float
        Risk-free rate (continuously compounded).
    K : float
        Strike price (anchors x = ln(S/K)).
    x_min, x_max : float
        Truncated spatial bounds (in log-moneyness).
    dx, dtau : float
        Step sizes.
    xs : ndarray of shape (N+1,)
        Spatial nodes x_j = x_min + j*dx.
    taus : ndarray of shape (M+1,)
        Time levels tau_n = n*dtau.

    Notes
    -----
    The drift in log-space mu = r - sigma^2/2 is exposed as a property
    rather than stored, to keep the dataclass canonical (single source of
    truth for sigma and r).
    """
    N: int
    M: int
    T: float
    sigma: float
    r: float
    K: float
    x_min: float
    x_max: float
    dx: float
    dtau: float
    xs: np.ndarray
    taus: np.ndarray

    @property
    def mu(self) -> float:
        """Drift in log-space under the risk-neutral measure: r - sigma^2/2."""
        return self.r - 0.5 * self.sigma * self.sigma


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------
def build_grid(*, N, M, T, sigma, r, K, n_sigma=4.0):
    """Construct a uniform FD grid for the transformed Black-Scholes PDE.

    Truncates the spatial domain to
        x in [-n_sigma * sigma * sqrt(T), +n_sigma * sigma * sqrt(T)],
    where x = ln(S/K). With n_sigma = 4 the lognormal-tail mass outside
    the domain is O(exp(-n_sigma^2 / 2)) ~ 6e-5; with n_sigma = 5 it is
    ~3e-7. For n_sigma >= 4 the truncation error is dominated by the
    discretisation error at any practical (N, M).

    Parameters
    ----------
    N : int (keyword-only)
        Number of spatial intervals; must be >= 2.
    M : int (keyword-only)
        Number of time intervals; must be >= 1.
    T : float
        Time to maturity in years; must be > 0.
    sigma : float
        Volatility; must be > 0.
    r : float
        Risk-free rate (any sign).
    K : float
        Strike price; must be > 0.
    n_sigma : float, default 4.0
        Half-width of the truncated spatial domain measured in
        sigma * sqrt(T); must be > 0.

    Returns
    -------
    Grid
        An immutable Grid instance fully describing the discretisation.

    Raises
    ------
    ValueError
        On any invalid input.
    """
    if N < 2:
        raise ValueError(f"N must be >= 2, got {N}")
    if M < 1:
        raise ValueError(f"M must be >= 1, got {M}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    if n_sigma <= 0:
        raise ValueError(f"n_sigma must be positive, got {n_sigma}")

    half_width = n_sigma * sigma * np.sqrt(T)
    x_min = -half_width
    x_max = +half_width

    dx = (x_max - x_min) / N
    dtau = T / M

    xs = np.linspace(x_min, x_max, N + 1)
    taus = np.linspace(0.0, T, M + 1)

    return Grid(
        N=N, M=M, T=T, sigma=sigma, r=r, K=K,
        x_min=x_min, x_max=x_max,
        dx=dx, dtau=dtau,
        xs=xs, taus=taus,
    )


# ---------------------------------------------------------------------------
# Initial conditions (payoff at tau = 0, i.e. t = T)
# ---------------------------------------------------------------------------
def call_initial_condition(xs, K):
    """Initial condition for a call: K * (e^x - 1)^+, equivalent to (S - K)^+
    written in log-moneyness.

    Parameters
    ----------
    xs : ndarray
        Spatial nodes (log-moneyness).
    K : float
        Strike price.

    Returns
    -------
    ndarray, same shape as xs
        Payoff K * max(e^x - 1, 0).
    """
    return np.maximum(K * (np.exp(xs) - 1.0), 0.0)


def put_initial_condition(xs, K):
    """Initial condition for a put: K * (1 - e^x)^+."""
    return np.maximum(K * (1.0 - np.exp(xs)), 0.0)


# ---------------------------------------------------------------------------
# Dirichlet boundary conditions
# ---------------------------------------------------------------------------
# All functions return arrays of length grid.M + 1 indexed by the time
# level n, so result[n] is the boundary value at tau = grid.taus[n].
def call_boundary_lower(grid):
    """Dirichlet lower boundary for European call: V(x_min, tau) = 0.

    The call has no value when the underlying is essentially zero. The
    truncation x_min = -n_sigma * sigma * sqrt(T) places the lower
    boundary deep enough that the Dirichlet approximation is exact to
    lognormal-tail accuracy.
    """
    return np.zeros(grid.M + 1)


def call_boundary_upper(grid):
    """Dirichlet upper boundary for European call:
        V(x_max, tau) = K * (e^{x_max} - e^{-r tau}).
    Asymptotic limit C(S, t) ~ S - K * e^{-r(T - t)} as S -> infinity,
    written in log-moneyness.
    """
    return grid.K * (np.exp(grid.x_max) - np.exp(-grid.r * grid.taus))


def put_boundary_lower(grid):
    """Dirichlet lower boundary for European put:
        V(x_min, tau) = K * (e^{-r tau} - e^{x_min}).
    Asymptotic limit P(S, t) -> K * e^{-r(T - t)} - S as S -> 0; the
    e^{x_min} term is exponentially small and dropping it would only
    introduce an O(e^{-n_sigma * sigma * sqrt(T)}) inconsistency at the
    boundary.
    """
    return grid.K * (np.exp(-grid.r * grid.taus) - np.exp(grid.x_min))


def put_boundary_upper(grid):
    """Dirichlet upper boundary for European put: V(x_max, tau) = 0.

    The put has no value when the underlying is far above the strike.
    """
    return np.zeros(grid.M + 1)


# ---------------------------------------------------------------------------
# Stability numbers (used by the explicit pricer in Block 1)
# ---------------------------------------------------------------------------
def fourier_number(sigma, dtau, dx):
    """Diffusive Fourier number alpha = (sigma^2 / 2) * dtau / dx^2.

    The FTCS scheme (Block 1) is von Neumann stable iff alpha <= 1/2.
    BTCS (Block 2) and Crank-Nicolson (Block 3) are unconditionally
    stable in alpha.
    """
    return 0.5 * sigma * sigma * dtau / (dx * dx)


def courant_number(mu, dtau, dx):
    """Advective Courant number nu = |mu| * dtau / (2 * dx).

    With centred space differences for the convective term, the FTCS
    von Neumann factor for the full BS PDE is
        g(theta) = 1 - 4*alpha*sin^2(theta/2) + i*nu*sin(theta).
    For typical Phase 3 parameters the diffusive bound alpha <= 1/2
    dominates over the convective constraint.
    """
    return 0.5 * abs(mu) * dtau / dx


def is_explicit_stable(alpha):
    """Return True iff the FTCS scheme is stable for this Fourier number,
    i.e. iff alpha <= 0.5. Diagnostic helper for Block 1.
    """
    return alpha <= 0.5


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Standard smoke test: build a moderate grid for the canonical
    # parameters of Phase 1 / 2 and print the relevant summary.
    g = build_grid(N=200, M=400, T=1.0, sigma=0.20, r=0.05, K=100.0)

    print(f"Grid: N={g.N}, M={g.M}")
    print(f"x   in [{g.x_min:+.4f}, {g.x_max:+.4f}], dx   = {g.dx:.6f}")
    print(f"tau in [{g.taus[0]:.4f}, {g.taus[-1]:.4f}], dtau = {g.dtau:.6f}")
    print(f"mu  = r - sigma^2/2 = {g.mu:+.6f}")
    print()

    alpha = fourier_number(g.sigma, g.dtau, g.dx)
    nu    = courant_number (g.mu,    g.dtau, g.dx)
    print(f"Fourier number alpha = {alpha:.6f}  (FTCS stable: {is_explicit_stable(alpha)})")
    print(f"Courant number nu    = {nu:.6f}")
    print()

    ic_call = call_initial_condition(g.xs, g.K)
    bc_call_hi = call_boundary_upper(g)
    print(f"Call IC at S = K (x = 0): V = {ic_call[g.N // 2]:.6f}  (expected 0)")
    print(f"Call IC at S = K * e (x = 1, j approx 3N/4): V = "
          f"{ic_call[int(0.75 * g.N)]:.4f}")
    print(f"Call upper BC at tau = 0: {bc_call_hi[0]:.4f}  "
          f"(equals IC at j=N: {ic_call[g.N]:.4f})")
    print(f"Call upper BC at tau = T: {bc_call_hi[-1]:.4f}")
