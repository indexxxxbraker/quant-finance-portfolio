"""Monte Carlo methods for European option pricing under Black-Scholes
geometric Brownian motion.

This module contains:

- ``MCResult``: a named tuple bundling the standard outputs of any
  Monte Carlo run (point estimate, asymptotic confidence half-width,
  sample variance, sample size).
- ``mc_estimator``: the model- and payoff-agnostic statistical reducer
  that turns a vector of i.i.d. payoffs into an ``MCResult``.
- ``simulate_terminal_gbm``: exact sampler of the terminal price
  ``S_T`` under geometric Brownian motion, using the closed-form
  solution of the SDE.
- ``mc_european_call_exact``: high-level pricer for the European call
  that orchestrates ``simulate_terminal_gbm`` + payoff +
  ``mc_estimator``.

References
----------
Phase 2 Block 0 writeup (Monte Carlo foundations) and Phase 2
Block 1.1 writeup (this algorithm). Glasserman, *Monte Carlo Methods
in Financial Engineering*, Chapters 1 and 3.
"""

from typing import NamedTuple, Optional

import numpy as np
from scipy.stats import norm


# =====================================================================
# Public types
# =====================================================================

class MCResult(NamedTuple):
    """Result of a Monte Carlo estimation.

    Attributes
    ----------
    estimate : float
        The Monte Carlo point estimate.
    half_width : float
        Asymptotic half-width of the (1 - alpha) confidence interval
        based on the central limit theorem and Slutsky's lemma.
    sample_variance : float
        Sample variance of the i.i.d. payoffs, computed with Bessel's
        correction (denominator n - 1).
    n_paths : int
        Number of i.i.d. samples used.
    """
    estimate: float
    half_width: float
    sample_variance: float
    n_paths: int


# =====================================================================
# Internal helpers
# =====================================================================

def _validate_model_params(S0, r, sigma, T):
    """Validate the model parameters of geometric Brownian motion.

    Note that ``r`` is unconstrained: negative rates are admissible.
    """
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


def _resolve_rng(seed, rng):
    """Return a Generator from exactly one of seed or rng.

    Raises ValueError if both or neither are provided.
    """
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

    The marginal speed cost (~2x slower than Ziggurat) is irrelevant
    at the sample sizes used in practice and is the price paid for
    having a single sampling pipeline for both classical MC and QMC.
    """
    u = rng.uniform(size=n)
    return norm.ppf(u)


# =====================================================================
# Generic statistical estimator
# =====================================================================

def mc_estimator(Y, confidence_level=0.95):
    """Reduce a vector of i.i.d. payoff samples to a Monte Carlo result.

    This function is model- and payoff-agnostic: it computes the
    sample mean, the sample variance (with Bessel's correction), and
    the asymptotic Gaussian confidence interval half-width based on
    the central limit theorem and Slutsky's lemma. It is the universal
    statistical reducer used by every Monte Carlo pricer in the
    project.

    Parameters
    ----------
    Y : array_like
        One-dimensional array of i.i.d. payoff samples. Must have at
        least 2 elements.
    confidence_level : float, optional
        Confidence level of the asymptotic interval, in (0, 1).
        Default 0.95.

    Returns
    -------
    MCResult
        Named tuple ``(estimate, half_width, sample_variance, n_paths)``.

    Raises
    ------
    ValueError
        If ``Y`` has fewer than 2 elements, if ``confidence_level`` is
        not in (0, 1), or if ``Y`` contains non-finite values.

    Notes
    -----
    The half-width is ``z * sqrt(sample_variance / n)`` with
    ``z = norm.ppf(0.5 * (1 + confidence_level))``. This is the
    asymptotic interval; for sample sizes used in practice (n in the
    thousands or more) the difference with the exact ``t``-quantile is
    negligible.

    References
    ----------
    Phase 2 Block 0 writeup, Section 2.4.
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
# Exact GBM sampler
# =====================================================================

def simulate_terminal_gbm(S0, r, sigma, T, n_paths, rng):
    """Simulate ``n_paths`` samples of ``S_T`` under geometric Brownian
    motion, exactly.

    Uses the closed-form solution of the GBM SDE,

        S_T = S_0 * exp((r - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)

    with ``Z ~ N(0, 1)``, so the samples are draws from the *exact*
    distribution of S_T. There is no time-discretisation error.

    Parameters
    ----------
    S0 : float
        Initial price of the underlying. Must be positive.
    r : float
        Risk-free rate. Unconstrained: negative rates are admissible.
    sigma : float
        Volatility. Must be positive.
    T : float
        Time horizon. Must be positive.
    n_paths : int
        Number of independent samples. Must be at least 2.
    rng : numpy.random.Generator
        Random number generator. Construct with
        ``rng = numpy.random.default_rng(seed)`` from a recorded seed.

    Returns
    -------
    ndarray of shape (n_paths,) and dtype float64
        Independent samples of S_T.

    Raises
    ------
    ValueError
        If any of ``S0, sigma, T`` is non-positive, or if
        ``n_paths < 2``.

    Notes
    -----
    Standard normal samples are produced by inversion (see
    ``_standard_normals``) for coordinate-by-coordinate compatibility
    with the quasi-Monte Carlo methods of Phase 2 Block 3.

    References
    ----------
    Phase 2 Block 1.1 writeup, Section 3.
    """
    _validate_model_params(S0, r, sigma, T)
    _validate_n_paths(n_paths)

    Z = _standard_normals(n_paths, rng)
    drift = (r - 0.5 * sigma * sigma) * T
    diffusion = sigma * np.sqrt(T)
    return S0 * np.exp(drift + diffusion * Z)


# =====================================================================
# High-level pricer
# =====================================================================

def mc_european_call_exact(S, K, r, sigma, T, n_paths,
                           *,
                           seed=None,
                           rng=None,
                           confidence_level=0.95):
    """Price a European call by Monte Carlo with exact GBM simulation.

    Pipeline: sample ``n_paths`` of S_T using the closed-form solution
    of the GBM SDE, evaluate the discounted payoff
    ``e^{-rT} * (S_T - K)^+`` on each path, and pass the resulting
    vector of payoffs to ``mc_estimator``.

    This is the canonical baseline of Phase 2: every more elaborate
    Monte Carlo method developed in the phase must reduce, in this
    setting, to a result consistent with this pricer. It is also the
    only algorithm of the phase entirely free of systematic error,
    and therefore the cleanest diagnostic for implementation
    correctness.

    Parameters
    ----------
    S, K, r, sigma, T : float
        Standard Black-Scholes inputs (spot, strike, rate, volatility,
        maturity). ``S, K, sigma, T`` must be positive; ``r`` is
        unconstrained.
    n_paths : int
        Number of independent simulated paths. Must be at least 2.
    seed : int or None, keyword-only
        Seed for the internally-constructed random generator. Pass
        exactly one of ``seed`` and ``rng``.
    rng : numpy.random.Generator or None, keyword-only
        Random generator. Pass exactly one of ``seed`` and ``rng``.
    confidence_level : float, keyword-only, optional
        Confidence level of the asymptotic interval, in (0, 1).
        Default 0.95.

    Returns
    -------
    MCResult
        Named tuple ``(estimate, half_width, sample_variance, n_paths)``.

    Raises
    ------
    ValueError
        If model parameters fail validation, if ``n_paths < 2``, if
        both or neither of ``seed`` and ``rng`` are provided, or if
        ``confidence_level`` is not in (0, 1).

    Notes
    -----
    The closed-form Black-Scholes price (for validation) and the
    closed-form variance of the discounted payoff (for a-priori
    sample-size selection) are available as
    ``quantlib.black_scholes.call_price`` and
    ``quantlib.black_scholes.call_payoff_variance`` respectively.

    References
    ----------
    Phase 2 Block 1.1 writeup. Glasserman, Section 1.1.2.
    """
    _validate_model_params(S, r, sigma, T)
    _validate_strike(K)
    _validate_n_paths(n_paths)
    rng = _resolve_rng(seed, rng)

    S_T = simulate_terminal_gbm(S, r, sigma, T, n_paths, rng)
    Y = np.exp(-r * T) * np.maximum(S_T - K, 0.0)

    return mc_estimator(Y, confidence_level=confidence_level)


# =====================================================================
# Smoke test entry point (run via `python -m quantlib.monte_carlo`)
# =====================================================================

if __name__ == "__main__":
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.00
    result = mc_european_call_exact(
        S, K, r, sigma, T, n_paths=100_000, seed=42,
    )
    print(f"MC estimate     : {result.estimate:.6f}")
    print(f"Half-width (95%): {result.half_width:.6f}")
    print(f"Sample variance : {result.sample_variance:.6f}")
    print(f"Sample size     : {result.n_paths}")
