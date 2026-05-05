"""Monte Carlo methods for European option pricing under Black-Scholes
geometric Brownian motion.

This module contains:

- ``MCResult``: a named tuple bundling the standard outputs of any
  Monte Carlo run (point estimate, asymptotic confidence half-width,
  sample variance, sample size).
- ``mc_estimator``: the model- and payoff-agnostic statistical reducer
  that turns a vector of i.i.d. payoffs into an ``MCResult``.

Exact GBM samplers (Block 1.1):

- ``simulate_terminal_gbm``: exact sampler of the terminal price
  ``S_T`` under geometric Brownian motion, using the closed-form
  solution of the SDE.
- ``mc_european_call_exact``: high-level pricer for the European call
  using the exact sampler.

Euler-Maruyama discretisation (Block 1.2.1):

- ``simulate_path_euler``: Euler-Maruyama path sampler returning the
  full path (including the initial value).
- ``simulate_terminal_euler``: convenience that returns only the
  terminal value of the Euler-discretised process.
- ``mc_european_call_euler``: high-level pricer using the Euler
  discretisation.

References
----------
Phase 2 Block 0 writeup (Monte Carlo foundations), Block 1.1 writeup
(exact sampler), Block 1.2.0 writeup (SDE discretisation theory),
and Block 1.2.1 writeup (this algorithm). Glasserman, *Monte Carlo
Methods in Financial Engineering*, Chapters 1, 3, and 6.
"""

from typing import NamedTuple, Optional

import numpy as np
from scipy.stats import norm


# =====================================================================
# Public types
# =====================================================================

class MCResult(NamedTuple):
    """Result of a Monte Carlo estimation."""
    estimate: float
    half_width: float
    sample_variance: float
    n_paths: int


# =====================================================================
# Internal helpers
# =====================================================================

def _validate_model_params(S0, r, sigma, T):
    if S0 <= 0:
        raise ValueError(f"S0 must be positive, got {S0}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")


def _validate_strike(K):
    if K <= 0:
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


def _resolve_rng(seed, rng):
    """Return a Generator from exactly one of seed or rng."""
    if (seed is None) == (rng is None):
        raise ValueError(
            "Pass exactly one of `seed` (int) or `rng` (Generator). "
            f"Got seed={seed!r}, rng={rng!r}."
        )
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def _standard_normals(n, rng):
    """Generate ``n`` independent standard normal samples by inversion.

    Inversion is used (rather than the faster Ziggurat method that
    NumPy provides via ``rng.standard_normal``) to maintain
    coordinate-by-coordinate compatibility with low-discrepancy
    sequences, which will be introduced in Phase 2 Block 3 (QMC).
    See the Phase 2 Block 0 writeup, Section 3.3, for the rationale.
    """
    u = rng.uniform(size=n)
    return norm.ppf(u)


def _gbm_exact_from_brownian(S0, r, sigma, T, W_T):
    """Exact GBM terminal price given the terminal Brownian value(s).

    Computes ``S_T = S_0 * exp((r - 0.5 * sigma^2) * T + sigma * W_T)``,
    the closed-form solution of the GBM SDE evaluated at the supplied
    ``W_T``. Used in validation to build an exact reference path that
    shares its Brownian driver with a discretised path, enabling
    pathwise (strong) and CRN-based (weak) error estimation.

    Parameters
    ----------
    S0, r, sigma, T : float
        GBM parameters.
    W_T : float or ndarray
        Terminal Brownian value(s).

    Returns
    -------
    float or ndarray
        The corresponding ``S_T``.

    Notes
    -----
    Private to this module: only the validation script imports it.
    The function is not part of the user-facing API.
    """
    drift = (r - 0.5 * sigma * sigma) * T
    return S0 * np.exp(drift + sigma * W_T)


# =====================================================================
# Generic statistical estimator
# =====================================================================

def mc_estimator(Y, confidence_level=0.95):
    """Reduce a vector of i.i.d. payoff samples to a Monte Carlo result.

    See Phase 2 Block 0 writeup, Section 2.4.
    """
    Y = np.asarray(Y, dtype=np.float64)

    if Y.ndim != 1:
        raise ValueError(f"Y must be 1-D, got shape {Y.shape}")

    n = Y.size
    if n < 2:
        raise ValueError(f"Y must have at least 2 elements, got {n}")

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f"confidence_level must be in (0, 1), got {confidence_level}"
        )

    if not np.all(np.isfinite(Y)):
        raise ValueError("Y contains non-finite values (NaN or +/- inf)")

    estimate = float(np.mean(Y))
    sample_variance = float(np.var(Y, ddof=1))
    z = float(norm.ppf(0.5 * (1.0 + confidence_level)))
    half_width = z * float(np.sqrt(sample_variance / n))

    return MCResult(
        estimate=estimate,
        half_width=half_width,
        sample_variance=sample_variance,
        n_paths=int(n),
    )


# =====================================================================
# Exact GBM sampler (Block 1.1)
# =====================================================================

def simulate_terminal_gbm(S0, r, sigma, T, n_paths, rng):
    """Simulate ``n_paths`` samples of ``S_T`` under geometric Brownian
    motion, exactly.

    See Phase 2 Block 1.1 writeup, Section 3.
    """
    _validate_model_params(S0, r, sigma, T)
    _validate_n_paths(n_paths)

    Z = _standard_normals(n_paths, rng)
    drift = (r - 0.5 * sigma * sigma) * T
    diffusion = sigma * np.sqrt(T)
    return S0 * np.exp(drift + diffusion * Z)


def mc_european_call_exact(S, K, r, sigma, T, n_paths,
                           *,
                           seed=None,
                           rng=None,
                           confidence_level=0.95):
    """Price a European call by Monte Carlo with exact GBM simulation.

    See Phase 2 Block 1.1 writeup. Glasserman, Section 1.1.2.
    """
    _validate_model_params(S, r, sigma, T)
    _validate_strike(K)
    _validate_n_paths(n_paths)
    rng = _resolve_rng(seed, rng)

    S_T = simulate_terminal_gbm(S, r, sigma, T, n_paths, rng)
    Y = np.exp(-r * T) * np.maximum(S_T - K, 0.0)

    return mc_estimator(Y, confidence_level=confidence_level)


# =====================================================================
# Euler-Maruyama scheme (Block 1.2.1)
# =====================================================================

def simulate_path_euler(S0, r, sigma, T, n_steps, n_paths,
                        *,
                        rng=None,
                        delta_W=None):
    """Simulate full Euler-Maruyama paths of GBM.

    Implements the recursion
        S_{n+1} = S_n * (1 + r * h + sigma * dW_n),
    with h = T / n_steps and dW_n ~ N(0, h) independent across n,
    starting from S_0 = S0. Returns the full path including the
    initial value.

    The recursion is a multiplicative one-step process, so it is
    implemented vectorised over paths via ``np.cumprod`` on the
    matrix of factors ``1 + r * h + sigma * delta_W``.

    Parameters
    ----------
    S0 : float
        Initial price. Must be positive.
    r : float
        Risk-free rate.
    sigma : float
        Volatility. Must be positive.
    T : float
        Maturity. Must be positive.
    n_steps : int
        Number of time steps. Must be at least 1.
    n_paths : int
        Number of independent paths. Must be at least 2.
    rng : numpy.random.Generator or None, keyword-only
        Random generator used to draw Brownian increments. Pass
        exactly one of ``rng`` and ``delta_W``.
    delta_W : ndarray of shape (n_paths, n_steps) or None, keyword-only
        Pre-sampled Brownian increments with variance h = T/n_steps
        per element. Use this to drive the simulation by a specific
        Brownian path (essential for common-random-numbers comparisons
        in the validation suite).

    Returns
    -------
    ndarray of shape (n_paths, n_steps + 1)
        ``paths[i, k]`` is the value of path ``i`` at time
        ``k * h``. Column 0 is ``S0``, column ``n_steps`` is the
        terminal value.

    Notes
    -----
    Euler does not preserve positivity in principle: a sufficiently
    negative ``delta_W`` can produce a negative ``S``. For typical
    finance parameters this is astronomically rare and not handled
    here. See Phase 2 Block 1.2.1 writeup, Section 2.2.
    """
    _validate_model_params(S0, r, sigma, T)
    _validate_n_steps(n_steps)
    _validate_n_paths(n_paths)

    if (rng is None) == (delta_W is None):
        raise ValueError(
            "Pass exactly one of `rng` (Generator) or `delta_W` "
            "(pre-sampled increments)."
        )

    h = T / n_steps
    if delta_W is None:
        delta_W = rng.normal(loc=0.0, scale=np.sqrt(h),
                             size=(n_paths, n_steps))
    else:
        delta_W = np.asarray(delta_W, dtype=np.float64)
        if delta_W.shape != (n_paths, n_steps):
            raise ValueError(
                f"delta_W shape must be ({n_paths}, {n_steps}), "
                f"got {delta_W.shape}"
            )

    # Multiplicative recursion: S_{n+1} = S_n * factor_n
    # where factor_n = 1 + r * h + sigma * delta_W_n.
    factors = 1.0 + r * h + sigma * delta_W   # shape (n_paths, n_steps)
    cumulative = np.cumprod(factors, axis=1)  # shape (n_paths, n_steps)

    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = S0
    paths[:, 1:] = S0 * cumulative
    return paths


def simulate_terminal_euler(S0, r, sigma, T, n_steps, n_paths,
                            *,
                            rng=None,
                            delta_W=None):
    """Convenience: Euler-Maruyama terminal value only.

    Returns only the terminal column of ``simulate_path_euler``,
    saving no work in the present implementation (the multiplicative
    recursion is computed via ``cumprod`` regardless), but exposing
    a cleaner signature for the European pricer where the path is
    discarded.

    See ``simulate_path_euler`` for the full parameter description.

    Returns
    -------
    ndarray of shape (n_paths,)
        Independent samples of the Euler-discretised terminal value.
    """
    paths = simulate_path_euler(S0, r, sigma, T, n_steps, n_paths,
                                rng=rng, delta_W=delta_W)
    return paths[:, -1]


def mc_european_call_euler(S, K, r, sigma, T, n_steps, n_paths,
                           *,
                           seed=None,
                           rng=None,
                           confidence_level=0.95):
    """Price a European call by Monte Carlo with Euler-Maruyama paths.

    Pipeline: sample ``n_paths`` Euler-discretised paths with
    ``n_steps`` time steps, evaluate the discounted payoff at the
    terminal column, and reduce via ``mc_estimator``.

    Unlike ``mc_european_call_exact``, this estimator carries a
    discretisation bias of order ``T / n_steps``: as ``n_steps`` is
    increased, the estimate converges to the BS price at weak rate 1.
    See Phase 2 Block 1.2.1 writeup, Section 3.

    Parameters
    ----------
    S, K, r, sigma, T : float
        Standard Black-Scholes inputs.
    n_steps : int
        Number of Euler time steps per path. Must be at least 1.
    n_paths : int
        Number of independent paths. Must be at least 2.
    seed : int or None, keyword-only
        Seed for the internally-constructed random generator.
    rng : numpy.random.Generator or None, keyword-only
        Pre-constructed random generator.
    confidence_level : float, keyword-only, optional
        Confidence level of the asymptotic interval. Default 0.95.

    Returns
    -------
    MCResult
        Named tuple ``(estimate, half_width, sample_variance, n_paths)``.

    Notes
    -----
    For European pricing under GBM, ``mc_european_call_exact`` is
    strictly preferred to this function: it is faster (no inner time
    loop) and unbiased. This pricer exists for two reasons:
    (a) as a controlled benchmark for the empirical validation of the
    Euler convergence orders (see ``validate_mc_european_euler.py``);
    (b) as the structural template for SDE-based Monte Carlo in
    settings where no closed-form solution exists (local volatility,
    stochastic volatility, jump diffusion).
    """
    _validate_model_params(S, r, sigma, T)
    _validate_strike(K)
    _validate_n_steps(n_steps)
    _validate_n_paths(n_paths)
    rng = _resolve_rng(seed, rng)

    S_T = simulate_terminal_euler(S, r, sigma, T, n_steps, n_paths, rng=rng)
    Y = np.exp(-r * T) * np.maximum(S_T - K, 0.0)

    return mc_estimator(Y, confidence_level=confidence_level)


# =====================================================================
# Smoke test entry point (run via `python -m quantlib.monte_carlo`)
# =====================================================================

if __name__ == "__main__":
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.00

    # Exact pricer (Block 1.1).
    exact = mc_european_call_exact(
        S, K, r, sigma, T, n_paths=100_000, seed=42,
    )
    print(f"Exact MC pricer (Block 1.1):")
    print(f"  estimate    : {exact.estimate:.6f}")
    print(f"  half-width  : {exact.half_width:.6f}\n")

    # Euler pricer at moderate n_steps (Block 1.2.1).
    euler = mc_european_call_euler(
        S, K, r, sigma, T, n_steps=100, n_paths=100_000, seed=42,
    )
    print(f"Euler MC pricer (Block 1.2.1, n_steps=100):")
    print(f"  estimate    : {euler.estimate:.6f}")
    print(f"  half-width  : {euler.half_width:.6f}")
