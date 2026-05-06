"""Variance-reduced Monte Carlo pricers for the European call.

This module contains pricers that improve over the basic IID
estimators of ``quantlib.monte_carlo`` by exploiting structural
properties of the problem (monotonicity, correlation with tractable
controls). The architectural decision to host these in a separate
module from the basic pricers reflects the conceptual division: the
basic pricers are universal templates, the variance-reduced pricers
are problem-specific optimisations.

Currently implemented:

- ``mc_european_call_exact_av``: antithetic-variates pricer for the
  European call under the exact GBM sampler (Block 2.1).

To be added in Block 2.2:

- ``mc_european_call_exact_cv_underlying``: control variate using the
  discounted underlying ``e^{-rT} S_T``.
- ``mc_european_call_exact_cv_aon``: control variate using the
  asset-or-nothing payoff.

References
----------
Phase 2 Block 2.0 writeup (foundations); Block 2.1 writeup (this
algorithm). Glasserman, *Monte Carlo Methods in Financial
Engineering*, Section 4.2.
"""

import numpy as np

from quantlib.gbm import (
    _standard_normals,
    validate_model_params,
    validate_strike,
    validate_n_paths,
)
from quantlib.monte_carlo import mc_estimator, _resolve_rng


# =====================================================================
# Antithetic-variates pricer (Block 2.1)
# =====================================================================

def mc_european_call_exact_av(S, K, r, sigma, T, n_paths,
                              *,
                              seed=None,
                              rng=None,
                              confidence_level=0.95):
    """Price a European call by Monte Carlo with antithetic variates.

    Implements the antithetic estimator
        Y_i^AV = 0.5 * (f(Z_i) + f(-Z_i))
    where f(z) = e^{-rT} * max(S_0 * exp((r-0.5*sigma^2)*T + sigma*sqrt(T)*z) - K, 0)
    and Z_i ~ N(0, 1) i.i.d. across i.

    Returns an MCResult based on the n_paths paired payoffs Y_i^AV.
    Note that the total number of payoff evaluations is 2 * n_paths
    (one at +Z_i, one at -Z_i), but the i.i.d. unit for variance
    estimation is the pair, hence n_paths is the relevant sample size
    for the central limit theorem.

    Parameters
    ----------
    S, K, r, sigma, T : float
        Standard Black-Scholes inputs.
    n_paths : int
        Number of antithetic *pairs* to simulate. Must be at least 2.
    seed : int or None, keyword-only
        Seed for the internally-constructed random generator. Pass
        exactly one of ``seed`` and ``rng``.
    rng : numpy.random.Generator or None, keyword-only
        Pre-constructed random generator.
    confidence_level : float, keyword-only, optional
        Confidence level of the asymptotic interval. Default 0.95.

    Returns
    -------
    MCResult
        Named tuple ``(estimate, half_width, sample_variance, n_paths)``,
        where ``n_paths`` is the number of pairs (not payoff
        evaluations) and ``sample_variance`` is the variance of the
        paired payoffs.

    Notes
    -----
    The monotonicity of the call payoff in Z (since S_T is monotone
    in Z and (S_T - K)^+ is monotone in S_T) guarantees that
    Cov(f(Z), f(-Z)) <= 0, so this estimator is strictly superior to
    the IID estimator at equal computational budget. The variance
    reduction factor at fixed budget is

        VRF = 1 / (1 + Corr(f(Z), f(-Z))).

    See the Phase 2 Block 2.1 writeup for the derivation and the
    moneyness-dependent prediction of VRF.

    The standard normals Z are sampled by inversion (via
    ``quantlib.gbm._standard_normals``) for QMC compatibility, the
    same convention as in Block 1.1.
    """
    validate_model_params(S, r, sigma, T)
    validate_strike(K)
    validate_n_paths(n_paths)
    rng = _resolve_rng(seed, rng)

    Z = _standard_normals(n_paths, rng)

    drift = (r - 0.5 * sigma * sigma) * T
    diffusion = sigma * np.sqrt(T)
    discount = np.exp(-r * T)

    # Build both halves of the antithetic pair from the same Z.
    S_T_plus  = S * np.exp(drift + diffusion * Z)
    S_T_minus = S * np.exp(drift - diffusion * Z)

    payoff_plus  = discount * np.maximum(S_T_plus  - K, 0.0)
    payoff_minus = discount * np.maximum(S_T_minus - K, 0.0)

    # Y_i^AV: the paired payoff. n_paths i.i.d. samples.
    Y = 0.5 * (payoff_plus + payoff_minus)

    return mc_estimator(Y, confidence_level=confidence_level)


# =====================================================================
# Smoke test entry point (run via `python -m quantlib.variance_reduction`)
# =====================================================================

if __name__ == "__main__":
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.00

    # AV pricer: n_paths = 50_000 pairs = 100_000 payoff evaluations.
    result_av = mc_european_call_exact_av(
        S, K, r, sigma, T, n_paths=50_000, seed=42,
    )
    print(f"AV MC pricer (Block 2.1, n_pairs=50000 = 100000 payoffs):")
    print(f"  estimate    : {result_av.estimate:.6f}")
    print(f"  half-width  : {result_av.half_width:.6f}")
    print(f"  sample var  : {result_av.sample_variance:.6f}")
