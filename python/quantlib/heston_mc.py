"""
Heston model: Monte Carlo simulation via full-truncation Euler.

This module implements the full-truncation Euler discretisation of
Lord, Koekkoek, van Dijk (2010) for the Heston stochastic-volatility
model under the risk-neutral measure. See
``theory/phase4/block3_heston_mc_basic.tex`` for the derivation,
convergence theory, and discussion of the truncation choice.

Public interface
----------------
- ``simulate_heston_paths``: full-grid path simulator returning
  (log_S, v) on the time mesh of shape (n_steps + 1, n_paths) each.
  Optional antithetic variates available.
- ``simulate_terminal_heston``: convenience wrapper that returns only
  the terminal log-spot column.
- ``mc_european_call_heston``: high-level pricer orchestrating the
  simulator with the discounted call payoff and the model-agnostic
  ``mc_estimator`` from ``quantlib.monte_carlo``.

Notation
--------
The Heston dynamics under Q are

    dS_t = r S_t dt + sqrt(v_t) S_t dW1_t
    dv_t = kappa (theta - v_t) dt + sigma sqrt(v_t) dW2_t
    d<W1, W2>_t = rho dt.

Working in log-spot X_t = log(S_t), Ito's lemma gives

    dX_t = (r - 0.5 v_t) dt + sqrt(v_t) dW1_t.

The full-truncation Euler scheme uses v_n^+ := max(v_n, 0) for the
*coefficients* in both drift and diffusion, while propagating v_n
itself unconstrained. This sidesteps the non-negativity issue of the
square-root coefficient at v = 0 with the smallest weak bias among
the standard fix-up schemes.

References
----------
[Lord2010]      Lord, Koekkoek, van Dijk (2010). A comparison of biased
                simulation schemes for stochastic volatility models.
                Quantitative Finance 10(2).
[Glasserman]    Glasserman, P. (2004). Monte Carlo Methods in Financial
                Engineering. Springer.
"""

from __future__ import annotations

import numpy as np

from quantlib.monte_carlo import (
    MCResult,
    mc_estimator,
    _resolve_rng,
)
from quantlib.gbm import _standard_normals

# =====================================================================
# Input validation
# =====================================================================

def _validate_heston_params(kappa, theta, sigma, rho, v0):
    """Validate Heston model parameters. Raises ValueError on failure."""
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


def _validate_contract_spec(S0, T):
    if S0 <= 0.0:
        raise ValueError(f"S0 must be positive, got {S0}")
    if T <= 0.0:
        raise ValueError(f"T must be positive, got {T}")


def _validate_strike(K):
    if K <= 0.0:
        raise ValueError(f"K must be positive, got {K}")


def _validate_n_paths(n_paths):
    if not isinstance(n_paths, (int, np.integer)):
        raise TypeError(
            f"n_paths must be an integer, got {type(n_paths).__name__}"
        )
    if n_paths < 2:
        raise ValueError(
            f"n_paths must be at least 2 (for sample variance with "
            f"Bessel's correction), got {n_paths}"
        )


def _validate_n_steps(n_steps):
    if not isinstance(n_steps, (int, np.integer)):
        raise TypeError(
            f"n_steps must be an integer, got {type(n_steps).__name__}"
        )
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}")


# =====================================================================
# Path simulator: full-truncation Euler
# =====================================================================

def simulate_heston_paths(S0, v0, r, kappa, theta, sigma, rho, T,
                           n_steps, n_paths,
                           *,
                           seed=None, rng=None,
                           antithetic=False):
    """
    Simulate full paths of the Heston (log_S, v) state via
    full-truncation Euler.

    The scheme is, with v_n^+ := max(v_n, 0):

        dW1_n = sqrt(dt) * Z1_n
        dW2_n = sqrt(dt) * (rho Z1_n + sqrt(1 - rho^2) Z2_n)
        v_{n+1}     = v_n + kappa (theta - v_n^+) dt
                          + sigma sqrt(v_n^+) dW2_n
        log S_{n+1} = log S_n + (r - 0.5 v_n^+) dt
                          + sqrt(v_n^+) dW1_n

    with (Z1_n, Z2_n) i.i.d. standard normals at each step. The
    Cholesky construction of correlated Brownian increments uses
    independent standard normals so that (a) at most two coordinates
    of any QMC sequence are consumed per step, and (b) the same Z1_n
    appears in both dW1_n and dW2_n, preserving the joint distribution
    exactly.

    Parameters
    ----------
    S0 : float
        Initial spot, must be positive.
    v0 : float
        Initial variance, must be non-negative.
    r : float
        Risk-free rate (unconstrained).
    kappa, theta, sigma, rho : float
        Heston parameters; see module docstring for ranges.
    T : float
        Time horizon, must be positive.
    n_steps : int
        Number of discretisation steps; n_steps >= 1.
    n_paths : int
        Number of independent paths; n_paths >= 2 (and even when
        ``antithetic=True``).
    seed : int or None, keyword-only
        Seed for the internally-constructed generator. Pass exactly
        one of ``seed`` and ``rng``.
    rng : numpy.random.Generator or None, keyword-only
        Random generator. Pass exactly one of ``seed`` and ``rng``.
    antithetic : bool, keyword-only, optional
        If True, paths are generated in antithetic pairs: half use
        normal increments (Z1, Z2), half use the negation (-Z1, -Z2).
        Requires n_paths to be even. Default False.

    Returns
    -------
    log_S : ndarray of shape (n_steps + 1, n_paths)
        Log-spot trajectories. Row 0 is log S0 broadcast to all paths.
    v : ndarray of shape (n_steps + 1, n_paths)
        Variance trajectories. Row 0 is v0 broadcast to all paths.
        These are the *unconstrained* v_n (which may be negative);
        the truncated v_n^+ = max(v_n, 0) is recoverable as
        ``np.maximum(v, 0.0)`` if needed.

    Raises
    ------
    ValueError
        On any parameter or contract validation failure, or if
        ``antithetic=True`` and ``n_paths`` is odd.
    """
    _validate_heston_params(kappa, theta, sigma, rho, v0)
    _validate_contract_spec(S0, T)
    _validate_n_steps(n_steps)
    _validate_n_paths(n_paths)
    rng = _resolve_rng(seed, rng)

    if antithetic and (n_paths % 2 != 0):
        raise ValueError(
            f"antithetic=True requires n_paths to be even, got {n_paths}"
        )

    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    sqrt_one_minus_rho2 = np.sqrt(1.0 - rho * rho)

    # Allocate trajectory arrays. Shape (n_steps + 1, n_paths) keeps the
    # time axis as axis 0, matching standard MC conventions; columns are
    # contiguous in memory under default C order, so per-step updates
    # are vectorised across paths efficiently.
    log_S = np.empty((n_steps + 1, n_paths), dtype=np.float64)
    v     = np.empty((n_steps + 1, n_paths), dtype=np.float64)
    log_S[0, :] = np.log(S0)
    v[0, :] = v0

    # Antithetic indices: the first half uses (Z1, Z2) increments, the
    # second half uses (-Z1, -Z2). We store the half size once.
    half = n_paths // 2 if antithetic else n_paths

    # Time-stepping loop. The loop over n is unavoidable (the SDE is
    # sequential), but each step is fully vectorised across paths.
    # Generating the standard normals one step at a time keeps memory
    # bounded at O(n_paths) rather than O(n_steps * n_paths).
    for n in range(n_steps):
        v_pos = np.maximum(v[n, :], 0.0)
        sqrt_v_pos = np.sqrt(v_pos)

        # Generate fresh normals for this step. With antithetic, generate
        # only half and concatenate the negation.
        Z_step = _standard_normals(2 * half, rng).reshape(2, half)
        if antithetic:
            Z1 = np.concatenate([Z_step[0, :], -Z_step[0, :]])
            Z2 = np.concatenate([Z_step[1, :], -Z_step[1, :]])
        else:
            Z1 = Z_step[0, :]
            Z2 = Z_step[1, :]

        dW1 = sqrt_dt * Z1
        dW2 = sqrt_dt * (rho * Z1 + sqrt_one_minus_rho2 * Z2)

        v[n + 1, :]     = (v[n, :]
                           + kappa * (theta - v_pos) * dt
                           + sigma * sqrt_v_pos * dW2)
        log_S[n + 1, :] = (log_S[n, :]
                           + (r - 0.5 * v_pos) * dt
                           + sqrt_v_pos * dW1)

    return log_S, v


def simulate_terminal_heston(S0, v0, r, kappa, theta, sigma, rho, T,
                              n_steps, n_paths,
                              *,
                              seed=None, rng=None,
                              antithetic=False):
    """
    Simulate only the terminal log-spot of the Heston model.

    Memory-efficient variant: only the current values of (log_S, v) are
    held during the iteration, not the full trajectories. This makes
    memory cost O(n_paths) rather than O(n_steps * n_paths), enabling
    large convergence studies.

    For applications that need the full path (path-dependent payoffs,
    visualisation, debugging), use ``simulate_heston_paths`` instead;
    that variant retains the trajectory but pays O(n_steps * n_paths)
    memory.

    Parameters
    ----------
    See ``simulate_heston_paths``.

    Returns
    -------
    S_T : ndarray of shape (n_paths,)
        Terminal spot values.
    """
    _validate_heston_params(kappa, theta, sigma, rho, v0)
    _validate_contract_spec(S0, T)
    _validate_n_steps(n_steps)
    _validate_n_paths(n_paths)
    rng = _resolve_rng(seed, rng)

    if antithetic and (n_paths % 2 != 0):
        raise ValueError(
            f"antithetic=True requires n_paths to be even, got {n_paths}"
        )

    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    sqrt_one_minus_rho2 = np.sqrt(1.0 - rho * rho)

    log_S = np.full(n_paths, np.log(S0), dtype=np.float64)
    v = np.full(n_paths, v0, dtype=np.float64)

    half = n_paths // 2 if antithetic else n_paths

    for _ in range(n_steps):
        v_pos = np.maximum(v, 0.0)
        sqrt_v_pos = np.sqrt(v_pos)

        Z_step = _standard_normals(2 * half, rng).reshape(2, half)
        if antithetic:
            Z1 = np.concatenate([Z_step[0, :], -Z_step[0, :]])
            Z2 = np.concatenate([Z_step[1, :], -Z_step[1, :]])
        else:
            Z1 = Z_step[0, :]
            Z2 = Z_step[1, :]

        dW1 = sqrt_dt * Z1
        dW2 = sqrt_dt * (rho * Z1 + sqrt_one_minus_rho2 * Z2)

        # In-place updates to keep memory bounded. Note: must compute
        # the new log_S before overwriting v, since both depend on the
        # current v_pos.
        log_S += (r - 0.5 * v_pos) * dt + sqrt_v_pos * dW1
        v += kappa * (theta - v_pos) * dt + sigma * sqrt_v_pos * dW2

    return np.exp(log_S)


# =====================================================================
# High-level pricer: European call
# =====================================================================

def mc_european_call_heston(S0, K, v0, r, kappa, theta, sigma, rho, T,
                              n_steps, n_paths,
                              *,
                              seed=None, rng=None,
                              antithetic=False,
                              confidence_level=0.95):
    """
    Price a European call under Heston by full-truncation Euler MC.

    Pipeline: simulate ``n_paths`` of S_T using ``n_steps`` Euler steps,
    evaluate the discounted payoff ``e^{-rT} (S_T - K)^+`` on each
    path, and reduce to a Monte Carlo result via ``mc_estimator``.

    When ``antithetic=True``, paths are generated in pairs (m, m + M/2)
    sharing the same magnitude of normal increments with opposite sign.
    The two payoffs are AVERAGED into a single antithetic sample
    Y^anti_m = (Y^+_m + Y^-_m) / 2 BEFORE being passed to the estimator.
    The estimator then sees M/2 i.i.d. samples (one per pair), and the
    variance of each is at most Var(Y) -- often much less when payoffs
    are monotone in the noise. This is the only way to actually realise
    the variance reduction of the antithetic technique; treating the M
    paths as if independent would discard the negative correlation
    between pairs entirely.

    Parameters
    ----------
    S0, K, v0, r, kappa, theta, sigma, rho, T : float
        Heston parameters and contract spec; see module docstring.
    n_steps, n_paths : int
        Discretisation grid size and number of MC paths. With
        ``antithetic=True``, ``n_paths`` is the total number of paths
        (M, twice the number of antithetic pairs M/2); must be even.
    seed, rng, antithetic, confidence_level : keyword-only
        See ``simulate_heston_paths``. ``confidence_level`` is the
        confidence level of the asymptotic Gaussian interval.

    Returns
    -------
    MCResult
        Named tuple ``(estimate, half_width, sample_variance, n_paths)``.
        Note: when ``antithetic=True``, ``n_paths`` in the result is the
        number of antithetic pairs (M/2), not the total number of paths.
        The ``sample_variance`` is the variance of the paired-average
        Y^anti_m, not of individual payoffs.

    Notes
    -----
    The estimate carries two sources of error: a discretisation bias of
    O(dt) (where dt = T/n_steps) and a statistical error of
    O(n_paths^{-1/2}). The reported half-width covers only the latter.
    For high-precision pricing, increase n_steps and n_paths in
    tandem; see Block 3 writeup, Section 4.4 for the bias-variance
    budget.
    """
    _validate_strike(K)

    S_T = simulate_terminal_heston(
        S0, v0, r, kappa, theta, sigma, rho, T,
        n_steps, n_paths,
        seed=seed, rng=rng, antithetic=antithetic,
    )

    Y = np.exp(-r * T) * np.maximum(S_T - K, 0.0)

    if antithetic:
        # Paths are arranged in two halves: first n_paths/2 use
        # increments (Z1, Z2), second n_paths/2 use (-Z1, -Z2). Pair
        # them and average BEFORE passing to mc_estimator, so that the
        # estimator sees genuinely i.i.d. samples and the half-width
        # correctly reflects the antithetic variance reduction.
        half = n_paths // 2
        Y = 0.5 * (Y[:half] + Y[half:])

    return mc_estimator(Y, confidence_level=confidence_level)


# =====================================================================
# Smoke test entry point
# =====================================================================

if __name__ == "__main__":
    # Standard equity parameter set (matches Block 2 reference).
    S0, K, v0, r = 100.0, 100.0, 0.04, 0.05
    kappa, theta, sigma, rho = 1.5, 0.04, 0.3, -0.7
    T = 0.5

    n_steps = 200
    n_paths = 100_000

    print(f"Heston Monte Carlo: full-truncation Euler smoke test")
    print(f"S0={S0}, K={K}, T={T}, r={r}")
    print(f"kappa={kappa}, theta={theta}, sigma={sigma}, rho={rho}, v0={v0}")
    print(f"n_steps={n_steps}, n_paths={n_paths}")
    print()

    result = mc_european_call_heston(
        S0, K, v0, r, kappa, theta, sigma, rho, T,
        n_steps, n_paths,
        seed=42,
    )
    print(f"MC estimate     : {result.estimate:.6f}")
    print(f"Half-width (95%): {result.half_width:.6f}")
    print(f"Sample variance : {result.sample_variance:.6f}")
    print(f"Sample size     : {result.n_paths}")

    # Compare against Fourier reference (from Block 2).
    try:
        from quantlib.heston_fourier import heston_call_lewis
        C_ref = heston_call_lewis(K, T, S0, v0, r,
                                   kappa, theta, sigma, rho)
        print()
        print(f"Fourier reference: {C_ref:.6f}")
        print(f"MC - Fourier     : {result.estimate - C_ref:+.6f}")
        print(f"In half-width    : "
              f"{abs(result.estimate - C_ref) < result.half_width}")
    except ImportError:
        print("(quantlib.heston_fourier not available for cross-check)")
