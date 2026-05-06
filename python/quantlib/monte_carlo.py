"""Statistical reducer and high-level Monte Carlo pricers.

This module contains:

- ``MCResult``: a named tuple bundling the standard outputs of any
  Monte Carlo run (point estimate, asymptotic confidence half-width,
  sample variance, sample size).
- ``mc_estimator``: the model- and payoff-agnostic statistical reducer
  that turns a vector of i.i.d. payoffs into an ``MCResult``.

High-level European call pricers (one per discretisation scheme):

- ``mc_european_call_exact``: exact GBM sampler (Block 1.1).
- ``mc_european_call_euler``: Euler-Maruyama discretisation
  (Block 1.2.1).
- ``mc_european_call_milstein``: Milstein discretisation
  (Block 1.2.2).

The samplers themselves live in ``quantlib.gbm``. This split reflects
the architectural separation between the model-and-scheme-specific
work (sampling paths of a particular SDE under a particular scheme)
and the model-and-payoff-agnostic work (statistical estimation).

References
----------
Phase 2 Block 0 writeup (Monte Carlo foundations); Block 1.1 writeup
(exact pricer); Block 1.2.0 writeup (SDE discretisation theory);
Block 1.2.1 writeup (Euler pricer); Block 1.2.2 writeup (Milstein
pricer). Glasserman, *Monte Carlo Methods in Financial Engineering*,
Chapters 1, 3, and 6.
"""

from typing import NamedTuple

import numpy as np
from scipy.stats import norm

from quantlib.gbm import (
    simulate_terminal_gbm,
    simulate_terminal_euler,
    simulate_terminal_milstein,
    validate_model_params,
    validate_strike,
    validate_n_paths,
    validate_n_steps,
)


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
# Internal helper for pricers: rng resolution
# =====================================================================

def _resolve_rng(seed, rng):
    """Return a Generator from exactly one of seed or rng.

    Used by the high-level pricers, which accept either a seed (for
    one-off calls with reproducibility from a recorded integer) or a
    pre-constructed Generator (for sequences of calls that should
    share a stream).
    """
    if (seed is None) == (rng is None):
        raise ValueError(
            "Pass exactly one of `seed` (int) or `rng` (Generator). "
            f"Got seed={seed!r}, rng={rng!r}."
        )
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


# =====================================================================
# Generic statistical estimator
# =====================================================================

def mc_estimator(Y, confidence_level=0.95):
    """Reduce a vector of i.i.d. payoff samples to a Monte Carlo result.

    This function is model- and payoff-agnostic: it computes the
    sample mean, the sample variance with Bessel's correction, and
    the asymptotic Gaussian confidence interval half-width. It is
    the universal statistical reducer used by every Monte Carlo
    pricer in the project.

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
# High-level pricers
# =====================================================================

def mc_european_call_exact(S, K, r, sigma, T, n_paths,
                           *,
                           seed=None,
                           rng=None,
                           confidence_level=0.95):
    """Price a European call by Monte Carlo with exact GBM simulation.

    Pipeline: sample n_paths of S_T using the closed-form solution of
    the GBM SDE, evaluate the discounted payoff
    ``e^{-rT} * (S_T - K)^+`` on each path, reduce via mc_estimator.

    See Phase 2 Block 1.1 writeup. Glasserman, Section 1.1.2.
    """
    validate_model_params(S, r, sigma, T)
    validate_strike(K)
    validate_n_paths(n_paths)
    rng = _resolve_rng(seed, rng)

    S_T = simulate_terminal_gbm(S, r, sigma, T, n_paths, rng)
    Y = np.exp(-r * T) * np.maximum(S_T - K, 0.0)

    return mc_estimator(Y, confidence_level=confidence_level)


def mc_european_call_euler(S, K, r, sigma, T, n_steps, n_paths,
                           *,
                           seed=None,
                           rng=None,
                           confidence_level=0.95):
    """Price a European call by Monte Carlo with Euler-Maruyama paths.

    Carries a discretisation bias of order ``T / n_steps``: as
    ``n_steps`` is increased, the estimate converges to the BS price
    at weak rate 1.

    See Phase 2 Block 1.2.1 writeup, Section 3.
    """
    validate_model_params(S, r, sigma, T)
    validate_strike(K)
    validate_n_steps(n_steps)
    validate_n_paths(n_paths)
    rng = _resolve_rng(seed, rng)

    S_T = simulate_terminal_euler(S, r, sigma, T, n_steps, n_paths, rng=rng)
    Y = np.exp(-r * T) * np.maximum(S_T - K, 0.0)

    return mc_estimator(Y, confidence_level=confidence_level)


def mc_european_call_milstein(S, K, r, sigma, T, n_steps, n_paths,
                              *,
                              seed=None,
                              rng=None,
                              confidence_level=0.95):
    """Price a European call by Monte Carlo with Milstein paths.

    Both Milstein and Euler have weak order 1 for European pricing,
    so this estimator is statistically indistinguishable from
    ``mc_european_call_euler`` at any practical sample size. The
    Milstein scheme exists for cases where strong order matters
    (pathwise sensitivity, Block 4) and is exposed here for
    completeness and for the empirical convergence study in
    ``validate_mc_european_milstein.py``.

    See Phase 2 Block 1.2.2 writeup.
    """
    validate_model_params(S, r, sigma, T)
    validate_strike(K)
    validate_n_steps(n_steps)
    validate_n_paths(n_paths)
    rng = _resolve_rng(seed, rng)

    S_T = simulate_terminal_milstein(S, r, sigma, T, n_steps, n_paths, rng=rng)
    Y = np.exp(-r * T) * np.maximum(S_T - K, 0.0)

    return mc_estimator(Y, confidence_level=confidence_level)


# =====================================================================
# Smoke test entry point (run via `python -m quantlib.monte_carlo`)
# =====================================================================

if __name__ == "__main__":
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.00

    # Block 1.1 pricer.
    exact = mc_european_call_exact(
        S, K, r, sigma, T, n_paths=100_000, seed=42,
    )
    print(f"Exact MC pricer (Block 1.1):")
    print(f"  estimate    : {exact.estimate:.6f}")
    print(f"  half-width  : {exact.half_width:.6f}\n")

    # Block 1.2.1 pricer.
    euler = mc_european_call_euler(
        S, K, r, sigma, T, n_steps=100, n_paths=100_000, seed=42,
    )
    print(f"Euler MC pricer (Block 1.2.1, n_steps=100):")
    print(f"  estimate    : {euler.estimate:.6f}")
    print(f"  half-width  : {euler.half_width:.6f}\n")

    # Block 1.2.2 pricer.
    milstein = mc_european_call_milstein(
        S, K, r, sigma, T, n_steps=100, n_paths=100_000, seed=42,
    )
    print(f"Milstein MC pricer (Block 1.2.2, n_steps=100):")
    print(f"  estimate    : {milstein.estimate:.6f}")
    print(f"  half-width  : {milstein.half_width:.6f}")
