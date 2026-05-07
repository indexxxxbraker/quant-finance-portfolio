"""Asian options on the European call by Monte Carlo.

This module implements three pricers for the discretely-monitored
arithmetic Asian call with payoff max(S_avg - K, 0) where
S_avg = (1/N) sum_{k=1}^N S_{t_k}:

  - mc_asian_call_arithmetic_iid: standard MC baseline.
  - mc_asian_call_geometric_iid: same but with geometric average,
    used to validate the closed form.
  - mc_asian_call_arithmetic_cv: arithmetic estimator with the
    geometric Asian as control variate (typical VRF > 1000).

The pricers use exact GBM stepping (not Euler), which is appropriate
for path-dependent options where the bias of Euler would compound
across observation dates.

The closed form for the geometric Asian is provided by
``geometric_asian_call_closed_form`` and uses an effective
volatility sigma_eff and effective rate r_eff; see
``theory/phase2/mc_asian.tex`` for derivation.

References
----------
Phase 2 Block 5 writeup. Kemna and Vorst (1990), J. Banking and
Finance, 14(1):113-129. Glasserman, Section 4.5.
"""

import math

import numpy as np
from scipy.stats import norm

from quantlib.gbm import (
    validate_model_params,
    validate_n_paths,
    validate_strike,
)
from quantlib.monte_carlo import MCResult, _resolve_rng, mc_estimator
from quantlib.variance_reduction import _apply_cv


# =====================================================================
# Closed form for the geometric Asian call
# =====================================================================

def _validate_n_steps(n_steps: int) -> None:
    if not isinstance(n_steps, (int, np.integer)):
        raise TypeError(f"n_steps must be int, got {type(n_steps).__name__}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")


def geometric_asian_call_closed_form(S, K, r, sigma, T, n_steps):
    """Closed-form price for the discretely-monitored geometric Asian
    call with N = n_steps observations on an equispaced grid in (0, T].

    The geometric average S_geom = (prod_k S_{t_k})^{1/N} is lognormal
    under risk-neutral GBM. The price has Black-Scholes-like form with
    an effective volatility sigma_eff and effective rate r_eff:

        sigma_eff = sigma * sqrt((N+1)(2N+1) / (6 N^2))
        r_eff     = (N+1)/(2N) * (r - sigma^2/2) + sigma_eff^2 / 2

    Then C_geom = e^{-rT} * [S * exp(r_eff * T) * Phi(d1) - K * Phi(d2)],
    where d1 = (log(S/K) + (r_eff + sigma_eff^2/2) T) / (sigma_eff sqrt T),
    and d2 = d1 - sigma_eff sqrt T.

    See theory/phase2/mc_asian.tex for full derivation.
    """
    validate_model_params(S, r, sigma, T)
    validate_strike(K)
    _validate_n_steps(n_steps)

    N = n_steps
    fac = (N + 1) * (2 * N + 1) / (6.0 * N * N)
    sigma_eff = sigma * math.sqrt(fac)
    r_eff = ((N + 1) / (2.0 * N)) * (r - 0.5 * sigma * sigma) \
            + 0.5 * sigma_eff * sigma_eff

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r_eff + 0.5 * sigma_eff * sigma_eff) * T) \
         / (sigma_eff * sqrt_T)
    d2 = d1 - sigma_eff * sqrt_T

    return math.exp(-r * T) * (S * math.exp(r_eff * T) * norm.cdf(d1)
                                - K * norm.cdf(d2))


# =====================================================================
# Path simulator
# =====================================================================

def _gbm_paths(S, r, sigma, T, n_paths, n_steps, rng):
    """Generate n_paths exact GBM paths over an equispaced grid of
    n_steps observations in (0, T]. Returns an array of shape
    (n_paths, n_steps), where row i is the trajectory at the
    monitoring dates t_1, ..., t_N.

    Note: S_0 is NOT included in the output; only the random
    monitoring values are.
    """
    h = T / n_steps
    drift = (r - 0.5 * sigma * sigma) * h
    diffusion = sigma * math.sqrt(h)

    Z = rng.standard_normal(size=(n_paths, n_steps))
    increments = np.exp(drift + diffusion * Z)
    # Cumulative product along axis 1 gives S_{t_k}/S_0 at each k.
    factors = np.cumprod(increments, axis=1)
    return S * factors


def _arithmetic_geometric_payoffs(paths, K, r, T):
    """Compute discounted arithmetic-Asian and geometric-Asian payoffs
    for a batch of paths.

    Returns
    -------
    Pi_A : ndarray of shape (n_paths,)
        Discounted arithmetic Asian call payoffs.
    Pi_G : ndarray of shape (n_paths,)
        Discounted geometric Asian call payoffs.
    """
    discount = math.exp(-r * T)

    arithmetic_avg = np.mean(paths, axis=1)
    # Geometric mean via exp(mean(log)). Using log avoids overflow for
    # long paths.
    geometric_avg = np.exp(np.mean(np.log(paths), axis=1))

    Pi_A = discount * np.maximum(arithmetic_avg - K, 0.0)
    Pi_G = discount * np.maximum(geometric_avg - K, 0.0)
    return Pi_A, Pi_G


# =====================================================================
# IID pricers
# =====================================================================

def mc_asian_call_arithmetic_iid(S, K, r, sigma, T, n_paths,
                                 *, n_steps=50,
                                 seed=None, rng=None,
                                 confidence_level=0.95):
    """Arithmetic Asian call by IID Monte Carlo (exact GBM)."""
    validate_model_params(S, r, sigma, T)
    validate_strike(K)
    validate_n_paths(n_paths)
    _validate_n_steps(n_steps)
    rng = _resolve_rng(seed, rng)

    paths = _gbm_paths(S, r, sigma, T, n_paths, n_steps, rng)
    Pi_A, _ = _arithmetic_geometric_payoffs(paths, K, r, T)
    return mc_estimator(Pi_A, confidence_level=confidence_level)


def mc_asian_call_geometric_iid(S, K, r, sigma, T, n_paths,
                                *, n_steps=50,
                                seed=None, rng=None,
                                confidence_level=0.95):
    """Geometric Asian call by IID Monte Carlo. Used to validate the
    closed form, not as a standalone pricer.
    """
    validate_model_params(S, r, sigma, T)
    validate_strike(K)
    validate_n_paths(n_paths)
    _validate_n_steps(n_steps)
    rng = _resolve_rng(seed, rng)

    paths = _gbm_paths(S, r, sigma, T, n_paths, n_steps, rng)
    _, Pi_G = _arithmetic_geometric_payoffs(paths, K, r, T)
    return mc_estimator(Pi_G, confidence_level=confidence_level)


# =====================================================================
# Arithmetic Asian with geometric control variate
# =====================================================================

def mc_asian_call_arithmetic_cv(S, K, r, sigma, T, n_paths,
                                *, n_steps=50,
                                seed=None, rng=None,
                                confidence_level=0.95):
    """Arithmetic Asian call with the geometric Asian as control
    variate.

    Same paths used for both estimators (CRN). The control variate
    correction uses the closed-form geometric Asian price as the
    known mean, and the empirical optimal beta_star = Cov(X, Y) /
    Var(Y) computed from the sample.

    Returns an MCResult whose half-width is computed from the
    variance of (X - beta_star * Y), i.e. (1 - rho^2) Var(X).

    Typical performance: VRF > 1000 vs the IID arithmetic estimator
    at our canonical parameters (rho_emp > 0.9999).
    """
    validate_model_params(S, r, sigma, T)
    validate_strike(K)
    validate_n_paths(n_paths)
    _validate_n_steps(n_steps)
    rng = _resolve_rng(seed, rng)

    paths = _gbm_paths(S, r, sigma, T, n_paths, n_steps, rng)
    Pi_A, Pi_G = _arithmetic_geometric_payoffs(paths, K, r, T)

    # Control variate's known mean: closed-form geometric Asian price.
    cv_mean = geometric_asian_call_closed_form(S, K, r, sigma, T, n_steps)

    return _apply_cv(Pi_A, Pi_G, cv_mean,
                     confidence_level=confidence_level)


# =====================================================================
# Smoke test
# =====================================================================

if __name__ == "__main__":
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    N = 50
    n = 100_000

    cg_closed = geometric_asian_call_closed_form(S, K, r, sigma, T, N)
    print(f"Geometric Asian closed form (N={N}): {cg_closed:.6f}\n")

    res_g = mc_asian_call_geometric_iid(S, K, r, sigma, T, n,
                                         n_steps=N, seed=42)
    err = abs(res_g.estimate - cg_closed) / res_g.half_width
    print(f"Geometric IID (n={n}):")
    print(f"  estimate   = {res_g.estimate:.6f}")
    print(f"  half-width = {res_g.half_width:.6f}")
    print(f"  closed     = {cg_closed:.6f}")
    print(f"  err / hw   = {err:.2f}\n")

    res_a_iid = mc_asian_call_arithmetic_iid(S, K, r, sigma, T, n,
                                              n_steps=N, seed=42)
    print(f"Arithmetic IID (n={n}):")
    print(f"  estimate   = {res_a_iid.estimate:.6f}")
    print(f"  half-width = {res_a_iid.half_width:.6f}\n")

    res_a_cv = mc_asian_call_arithmetic_cv(S, K, r, sigma, T, n,
                                            n_steps=N, seed=42)
    print(f"Arithmetic with geometric CV (n={n}):")
    print(f"  estimate   = {res_a_cv.estimate:.6f}")
    print(f"  half-width = {res_a_cv.half_width:.6f}\n")

    vrf = (res_a_iid.half_width / res_a_cv.half_width) ** 2
    print(f"Variance reduction factor (CV vs IID): {vrf:.0f}x")
    print(f"Half-width ratio (IID / CV)          : {res_a_iid.half_width / res_a_cv.half_width:.1f}x")
