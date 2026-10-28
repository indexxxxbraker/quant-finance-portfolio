"""Statistical reducer and high-level Monte Carlo pricers.

This module contains:

- ``MCResult``: a named tuple bundling the standard outputs of any
  Monte Carlo run (point estimate, asymptotic confidence half-width,
  sample variance, sample size).
- ``mc_estimator``: the model- and payoff-agnostic statistical reducer
  that turns a vector of i.i.d. payoffs into an ``MCResult``.

High-level European call pricers (one per discretisation scheme):

- ``mc_european_call_exact``: exact GBM sampler.
- ``mc_european_call_euler``: Euler-Maruyama discretisation.
- ``mc_european_call_milstein``: Milstein discretisation.

The samplers themselves live in ``quantlib.gbm``. This split reflects
the architectural separation between the model-and-scheme-specific
work (sampling paths of a particular SDE under a particular scheme)
and the model-and-payoff-agnostic work (statistical estimation).

References
----------
theory/phase2/monte_carlo_foundations.tex (Monte Carlo foundations);
theory/phase2/mc_european_exact.tex (exact pricer);
theory/phase2/sde_discretisation.tex (SDE discretisation theory);
theory/phase2/mc_european_euler.tex (Euler pricer);
theory/phase2/mc_european_milstein.tex (Milstein pricer). Glasserman,
*Monte Carlo Methods in Financial Engineering*, Chapters 1, 3, and 6.
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

    See theory/phase2/monte_carlo_foundations.tex, Section 2.4.
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

    See theory/phase2/mc_european_exact.tex. Glasserman, Section 1.1.2.
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

    See theory/phase2/mc_european_euler.tex, Section 3.
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
    and at practical sample sizes both biases are far below the
    statistical error, so the two estimators agree within noise (for
    the call, Milstein's weak-error constant is in fact three to four
    times larger; see validate_mc_european_milstein.py, Test 2). The
    Milstein scheme exists for cases where strong order matters
    (pathwise sensitivity, quantlib.greeks) and is exposed here for
    completeness and for the empirical convergence study in
    ``validate_mc_european_milstein.py``.

    See theory/phase2/mc_european_milstein.tex.
    """
    validate_model_params(S, r, sigma, T)
    validate_strike(K)
    validate_n_steps(n_steps)
    validate_n_paths(n_paths)
    rng = _resolve_rng(seed, rng)

    S_T = simulate_terminal_milstein(S, r, sigma, T, n_steps, n_paths, rng=rng)
    Y = np.exp(-r * T) * np.maximum(S_T - K, 0.0)

    return mc_estimator(Y, confidence_level=confidence_level)
