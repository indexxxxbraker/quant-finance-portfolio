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
  European call under exact GBM.

- ``mc_european_call_exact_cv_underlying``: control-variate pricer
  using the discounted underlying e^{-rT} S_T as the control.
  Expectation S_0 by the martingale property.

- ``mc_european_call_exact_cv_aon``: control-variate pricer using the
  discounted asset-or-nothing payoff e^{-rT} S_T * 1_{S_T > K}.
  Expectation S_0 * Phi(d_1) from the BS formula building block.

References
----------
theory/phase2/variance_reduction_foundations.tex (foundations);
theory/phase2/mc_european_av.tex (AV); theory/phase2/mc_european_cv.tex
(CV). Glasserman, *Monte Carlo Methods in Financial Engineering*,
Sections 4.1 and 4.2.
"""

import numpy as np
from scipy.stats import norm

from quantlib.gbm import (
    _standard_normals,
    validate_model_params,
    validate_strike,
    validate_n_paths,
)
from quantlib.monte_carlo import mc_estimator, _resolve_rng


# =====================================================================
# Antithetic-variates pricer
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

    See theory/phase2/mc_european_av.tex for the derivation and the
    moneyness-dependent prediction of VRF.
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
# Control-variates: shared helper
# =====================================================================

def _apply_cv(Y, X, EX, confidence_level):
    """Apply a control-variate adjustment to a sample of payoffs.

    Computes the OLS slope estimator
        c_hat = sum (Y_i - mean Y) * (X_i - mean X)
                / sum (X_i - mean X)^2
    and constructs the adjusted sample
        Y_tilde_i = Y_i - c_hat * (X_i - EX),
    where ``EX`` is the (true, closed-form) expectation of X.

    Returns the MCResult of the adjusted sample.

    Notes
    -----
    Private to this module: callable from the CV pricers below.
    Both ``Y`` and ``X`` must be 1-D arrays of the same length.

    The OLS slope is exactly Cov(Y, X) / Var(X), the optimal
    control-variate coefficient. The adjusted sample inherits the
    correct mean (E[Y]) by unbiasedness of the CV estimator at any
    deterministic c, and inherits the variance Var(Y)*(1 - rho^2)
    when c equals c_hat.

    For the empirical c_hat used here, the estimator picks up an
    O(1/M) bias from the correlation between c_hat and the Y_i;
    this is asymptotically negligible compared to the standard
    error O(1/sqrt(M)). See
    theory/phase2/variance_reduction_foundations.tex, Proposition 4.4.
    """
    Y = np.asarray(Y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    Y_bar = float(np.mean(Y))
    X_bar = float(np.mean(X))

    # OLS slope = Cov(Y, X) / Var(X).
    # Numerator: sum (Y_i - Y_bar)(X_i - X_bar)
    # Denominator: sum (X_i - X_bar)^2
    Y_dev = Y - Y_bar
    X_dev = X - X_bar
    numerator   = float(np.sum(Y_dev * X_dev))
    denominator = float(np.sum(X_dev * X_dev))

    if denominator == 0.0:
        raise ValueError(
            "Control variate has zero sample variance; cannot "
            "estimate slope."
        )

    c_hat = numerator / denominator

    # Adjusted samples. Note: we use EX (true expectation), not X_bar.
    Y_tilde = Y - c_hat * (X - EX)

    return mc_estimator(Y_tilde, confidence_level=confidence_level)


# =====================================================================
# Control-variates: discounted underlying
# =====================================================================

def mc_european_call_exact_cv_underlying(S, K, r, sigma, T, n_paths,
                                         *,
                                         seed=None,
                                         rng=None,
                                         confidence_level=0.95):
    """Price a European call by Monte Carlo with the discounted
    underlying as the control variate.

    Control: X_i = e^{-rT} S_T^{(i)}.
    Closed-form expectation: E[X] = S_0 (martingale property under Q).

    Adjusted estimator: Y_tilde_i = Y_i - c_hat * (X_i - S_0),
    where c_hat is the OLS slope of Y on X.

    At the ATM point rho_1 = Corr(Y, X_1) is about 0.92, which gives
    VRF = 1 / (1 - rho_1^2) of about 6.9. See
    theory/phase2/mc_european_cv.tex, Sections 2 and 4.

    Parameters and return value mirror ``mc_european_call_exact``.
    """
    validate_model_params(S, r, sigma, T)
    validate_strike(K)
    validate_n_paths(n_paths)
    rng = _resolve_rng(seed, rng)

    Z = _standard_normals(n_paths, rng)
    drift = (r - 0.5 * sigma * sigma) * T
    diffusion = sigma * np.sqrt(T)
    discount = np.exp(-r * T)

    S_T = S * np.exp(drift + diffusion * Z)

    # Target: discounted call payoff.
    Y = discount * np.maximum(S_T - K, 0.0)

    # Control: discounted underlying. Expectation: S_0.
    X = discount * S_T
    EX = float(S)   # E[discount * S_T] = S_0

    return _apply_cv(Y, X, EX, confidence_level)


# =====================================================================
# Control-variates: asset-or-nothing payoff
# =====================================================================

def mc_european_call_exact_cv_aon(S, K, r, sigma, T, n_paths,
                                  *,
                                  seed=None,
                                  rng=None,
                                  confidence_level=0.95):
    """Price a European call by Monte Carlo with the asset-or-nothing
    payoff as the control variate.

    Control: X_i = e^{-rT} * S_T^{(i)} * 1_{S_T^{(i)} > K}.
    Closed-form expectation: E[X] = S_0 * Phi(d_1), the standard BS
    asset-or-nothing call value.

    Adjusted estimator: Y_tilde_i = Y_i - c_hat * (X_i - S_0 * Phi(d_1)).

    The control matches the target's shape (zero on {S_T <= K}, linear
    on {S_T > K}), but X_2 - Y = K e^{-rT} 1{S_T > K} adds the variance
    of a digital payoff that the call does not have. At the ATM point
    rho_2 is only about 0.77 (VRF about 2.5), so this control is weaker
    than the underlying. See theory/phase2/mc_european_cv.tex,
    Sections 3 and 4.

    Parameters and return value mirror ``mc_european_call_exact``.
    """
    validate_model_params(S, r, sigma, T)
    validate_strike(K)
    validate_n_paths(n_paths)
    rng = _resolve_rng(seed, rng)

    Z = _standard_normals(n_paths, rng)
    drift = (r - 0.5 * sigma * sigma) * T
    diffusion = sigma * np.sqrt(T)
    discount = np.exp(-r * T)

    S_T = S * np.exp(drift + diffusion * Z)

    # Target: discounted call payoff.
    Y = discount * np.maximum(S_T - K, 0.0)

    # Control: discounted asset-or-nothing. Expectation: S_0 * Phi(d_1).
    X = discount * S_T * (S_T > K).astype(np.float64)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    EX = float(S * norm.cdf(d1))

    return _apply_cv(Y, X, EX, confidence_level)
