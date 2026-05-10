"""
Heston model: calibration to market data.

This module implements the calibration pipeline that, given a
market-observed surface of European call prices at various
strike-maturity pairs, finds Heston parameters
(kappa, theta, sigma, rho, v0) that minimise an aggregate distance
between model and market.

The pricer is the closed-form Fourier method of Block 2, which is
fast enough to evaluate inside the optimisation loop. The optimiser is
Levenberg-Marquardt (via scipy.optimize.least_squares with method='trf'
to respect bound constraints). The objective is the vega-weighted
price residual, which is computationally cheap and approximates
implied-vol RMSE on liquid quotes.

For production-grade calibration of arbitrary surfaces, this LM
optimiser should be wrapped in a global search (e.g.,
scipy.optimize.differential_evolution) to escape local minima before
LM refinement. We provide LM-only here; the wrapping is a few lines
documented in the calibrate_heston docstring.

See ``theory/phase4/block6_heston_calibration_exotics.tex`` for the
mathematical statement of the calibration problem and the discussion
of identifiability.

Public interface
----------------
- ``implied_vol_bs``: BS implied volatility inversion (Newton + Brent fallback).
- ``calibrate_heston``: full calibration pipeline.
- ``CalibrationResult``: namedtuple with parameters, residuals, etc.

References
----------
[Cui2017]  Cui, Y.; del Bano Rollin, S.; Germano, G. (2017). Full and
           fast calibration of the Heston stochastic volatility model.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Optional

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

from quantlib.heston_fourier import heston_call_lewis, black_scholes_call


# =====================================================================
# Result type
# =====================================================================

CalibrationResult = namedtuple(
    "CalibrationResult",
    ["params", "residuals", "rmse", "n_iter", "success", "message"]
)
# params: dict with keys kappa, theta, sigma, rho, v0
# residuals: ndarray of (model - market) prices, in the order of input data
# rmse: scalar root-mean-square of the residuals
# n_iter: number of optimiser iterations
# success: bool
# message: str


# =====================================================================
# Implied volatility inversion
# =====================================================================

def _bs_call_and_vega(sigma, S0, K, T, r):
    """Compute BS call price and vega for a single (sigma, K, T) point."""
    if sigma <= 0.0 or T <= 0.0:
        return float('nan'), 0.0
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    vega  = S0 * sqrt_T * norm.pdf(d1)
    return price, vega


def implied_vol_bs(C_target, K, T, S0, r,
                    *, sigma0=0.2, max_iter=50, tol=1e-8):
    """
    Compute the Black-Scholes implied volatility from a call price.

    Uses Newton-Raphson with Brent fallback. Returns the implied vol
    such that BS(K, T, S0, r, sigma) = C_target, or NaN if the price
    is outside the no-arbitrage bounds.

    Parameters
    ----------
    C_target : float
        The target call price.
    K, T, S0, r : float
        Strike, maturity, spot, risk-free rate.
    sigma0 : float, keyword-only, optional
        Newton initial guess. Default 0.2.
    max_iter : int, keyword-only, optional
        Newton max iterations. Default 50.
    tol : float, keyword-only, optional
        Convergence tolerance on |price residual|. Default 1e-8.

    Returns
    -------
    float
        Implied volatility, or NaN if no admissible IV exists.

    Notes
    -----
    No-arbitrage bounds for a call:
        max(S0 - K*exp(-rT), 0) <= C <= S0.
    Outside these bounds we return NaN.
    """
    # Check no-arbitrage bounds
    intrinsic = max(S0 - K * np.exp(-r * T), 0.0)
    if C_target < intrinsic - 1e-12:
        return float('nan')
    if C_target > S0 + 1e-12:
        return float('nan')

    # Newton-Raphson
    sigma = float(sigma0)
    for _ in range(max_iter):
        price, vega = _bs_call_and_vega(sigma, S0, K, T, r)
        if vega < 1e-12:
            break  # vega too small, fall through to Brent
        diff = price - C_target
        if abs(diff) < tol:
            return sigma
        sigma_new = sigma - diff / vega
        # Bound the step to prevent runaway
        sigma_new = max(0.001, min(5.0, sigma_new))
        if abs(sigma_new - sigma) < 1e-14:
            return sigma_new
        sigma = sigma_new
    else:
        # Newton converged within max_iter (loop exhausted but not via break)
        # The else clause runs when no break occurred. We may still be OK
        # if the last residual was small.
        price, _ = _bs_call_and_vega(sigma, S0, K, T, r)
        if abs(price - C_target) < 100 * tol:
            return sigma

    # Fallback: Brent's method on a wide bracket
    def f(sigma):
        price, _ = _bs_call_and_vega(sigma, S0, K, T, r)
        return price - C_target

    try:
        return brentq(f, 1e-4, 5.0, xtol=tol, maxiter=200)
    except (ValueError, RuntimeError):
        return float('nan')


# =====================================================================
# Pricing pipeline (vectorised over surface)
# =====================================================================

def _heston_prices_on_surface(params, market_data, S0, r):
    """
    Compute Heston model prices on a surface of (K_i, T_i) points.

    Parameters
    ----------
    params : array-like of length 5
        (kappa, theta, sigma, rho, v0) in optimiser order.
    market_data : list of dict
        Each dict has keys 'K', 'T', 'C_market'.
    S0, r : float

    Returns
    -------
    ndarray of shape (n_data,)
        Heston call prices at each (K_i, T_i).

    Notes
    -----
    The Fourier pricer of Block 2 is called per (K, T); a fully
    vectorised version that exploits shared characteristic-function
    evaluations across strikes for a fixed maturity would be
    significantly faster but is left as a future refinement.
    """
    kappa, theta, sigma, rho, v0 = params
    prices = np.empty(len(market_data))
    for i, d in enumerate(market_data):
        prices[i] = heston_call_lewis(
            d["K"], d["T"], S0, v0, r, kappa, theta, sigma, rho)
    return prices


def _compute_vega_weights(market_data, S0, r):
    """
    Compute vega-based weights for the calibration objective.

    For each market quote, the weight is 1 / max(vega, vega_floor),
    where vega is the BS vega at the market-implied volatility. This
    approximately equalises the contribution of each observation in
    IV space, which is what traders actually care about.

    Returns
    -------
    ndarray of shape (n_data,)
        The weights, normalised so the largest weight is 1.
    """
    weights = np.empty(len(market_data))
    vega_floor = 1e-3
    for i, d in enumerate(market_data):
        iv = implied_vol_bs(d["C_market"], d["K"], d["T"], S0, r)
        if np.isnan(iv):
            weights[i] = 1.0  # fallback; option is at boundary of no-arb
            continue
        _, vega = _bs_call_and_vega(iv, S0, d["K"], d["T"], r)
        weights[i] = 1.0 / max(vega, vega_floor)
    # Normalise (largest weight = 1) for numerical conditioning
    weights /= np.max(weights)
    return weights


# =====================================================================
# Calibration entry point
# =====================================================================

# Default bounds. Wide enough to cover most equity calibrations,
# tight enough to keep the optimiser stable.
DEFAULT_BOUNDS = {
    "kappa": (0.01, 20.0),
    "theta": (0.001, 1.0),
    "sigma": (0.01, 3.0),
    "rho":   (-0.999, 0.999),
    "v0":    (0.0001, 1.0),
}


def calibrate_heston(market_data, S0, r,
                       initial_guess=None,
                       *,
                       bounds=None,
                       weighted=True,
                       max_iter=200,
                       verbose=False):
    """
    Calibrate Heston parameters to a market-observed call surface.

    Parameters
    ----------
    market_data : list of dict
        Each dict has keys 'K' (strike), 'T' (maturity), 'C_market'
        (observed call price). Must have at least 5 observations
        (one per parameter); 9-15 typical.
    S0, r : float
        Spot and risk-free rate.
    initial_guess : dict, optional
        Initial parameter guess. Keys: 'kappa', 'theta', 'sigma',
        'rho', 'v0'. If None, a default sensible for equity
        markets is used.
    bounds : dict, keyword-only, optional
        Per-parameter (lower, upper) bounds. If None, DEFAULT_BOUNDS
        is used.
    weighted : bool, keyword-only, optional
        If True (default), use vega-based weights so price residuals
        approximately match IV residuals. If False, use unit weights.
    max_iter : int, keyword-only, optional
        Optimiser maximum iterations. Default 200.
    verbose : bool, keyword-only, optional
        If True, print optimiser progress. Default False.

    Returns
    -------
    CalibrationResult
        Named tuple with calibrated params, residuals, RMSE, etc.

    Notes
    -----
    Uses Levenberg-Marquardt via scipy.optimize.least_squares with
    method='trf' to respect bound constraints. The local LM
    optimiser converges to the nearest local minimum from the initial
    guess; if the surface is complex (e.g., real market data with vol
    smile), wrap this call in differential_evolution for global
    search before LM refinement:

        from scipy.optimize import differential_evolution
        de_result = differential_evolution(...)
        cal_result = calibrate_heston(market_data, S0, r,
                                       initial_guess=de_result.x_dict,
                                       ...)

    See block6_heston_calibration_exotics.tex Section 2.4 for
    discussion of the global vs local optimisation strategy.
    """
    if len(market_data) < 5:
        raise ValueError(
            f"need at least 5 market observations, got {len(market_data)}")

    if bounds is None:
        bounds = DEFAULT_BOUNDS
    if initial_guess is None:
        initial_guess = {"kappa": 2.0, "theta": 0.04, "sigma": 0.5,
                          "rho": -0.5, "v0": 0.04}

    # Optimiser parameter order: (kappa, theta, sigma, rho, v0)
    param_names = ["kappa", "theta", "sigma", "rho", "v0"]
    x0 = np.array([initial_guess[n] for n in param_names])
    lb = np.array([bounds[n][0] for n in param_names])
    ub = np.array([bounds[n][1] for n in param_names])

    # Pre-compute vega weights (depend on market data, not on params)
    if weighted:
        weights = _compute_vega_weights(market_data, S0, r)
    else:
        weights = np.ones(len(market_data))
    sqrt_w = np.sqrt(weights)
    market_prices = np.array([d["C_market"] for d in market_data])

    n_evals = [0]

    def residual_fn(x):
        n_evals[0] += 1
        model_prices = _heston_prices_on_surface(x, market_data, S0, r)
        # Weighted residual: scipy.least_squares minimises 0.5 * sum(r^2).
        # We supply r_i = sqrt(w_i) * (model_i - market_i), so the
        # objective becomes 0.5 * sum(w_i * (model - market)^2).
        return sqrt_w * (model_prices - market_prices)

    if verbose:
        print(f"[calibration] initial: {dict(zip(param_names, x0.round(4)))}")
        print(f"[calibration] bounds : {bounds}")
        print(f"[calibration] {len(market_data)} observations, "
              f"weighted={weighted}")

    result = least_squares(
        residual_fn, x0, bounds=(lb, ub), method='trf',
        max_nfev=max_iter, verbose=0,
    )

    # Compute unweighted residuals for the report
    final_model_prices = _heston_prices_on_surface(
        result.x, market_data, S0, r)
    raw_residuals = final_model_prices - market_prices
    rmse = float(np.sqrt(np.mean(raw_residuals ** 2)))

    if verbose:
        print(f"[calibration] result : "
              f"{dict(zip(param_names, result.x.round(6)))}")
        print(f"[calibration] RMSE   : {rmse:.6f}")
        print(f"[calibration] n_eval : {n_evals[0]}, "
              f"success: {result.success}")

    return CalibrationResult(
        params=dict(zip(param_names, result.x)),
        residuals=raw_residuals,
        rmse=rmse,
        n_iter=n_evals[0],
        success=result.success,
        message=result.message,
    )


# =====================================================================
# Smoke test entry point
# =====================================================================

if __name__ == "__main__":
    print("Heston calibration: smoke test (round-trip on synthetic data)")
    print("=" * 65)

    # Truth parameters
    truth = {"kappa": 1.5, "theta": 0.04, "sigma": 0.3,
              "rho": -0.7, "v0": 0.04}
    S0, r = 100.0, 0.05

    print(f"\nTruth parameters: {truth}")
    print(f"S0 = {S0}, r = {r}")

    # Generate synthetic market data (3 strikes x 3 maturities)
    market_data = []
    for K in [90.0, 100.0, 110.0]:
        for T in [0.25, 0.5, 1.0]:
            C = heston_call_lewis(
                K, T, S0, truth["v0"], r,
                truth["kappa"], truth["theta"],
                truth["sigma"], truth["rho"])
            market_data.append({"K": K, "T": T, "C_market": C})

    print(f"\nGenerated {len(market_data)} synthetic market quotes")

    # Calibrate from a perturbed initial guess
    initial = {"kappa": 1.0, "theta": 0.05, "sigma": 0.5,
                "rho": -0.3, "v0": 0.05}
    print(f"Initial guess  : {initial}")

    import time
    t0 = time.time()
    result = calibrate_heston(market_data, S0, r,
                                initial_guess=initial, verbose=True)
    elapsed = time.time() - t0

    print(f"\nResults after {elapsed:.2f}s:")
    print(f"  Calibrated parameters: {result.params}")
    print(f"  Truth parameters     : {truth}")
    print(f"  Per-parameter error  :")
    for name in ["kappa", "theta", "sigma", "rho", "v0"]:
        err = result.params[name] - truth[name]
        print(f"    {name:>6}: {result.params[name]:>+10.6f} "
              f"vs truth {truth[name]:>+10.6f}, err = {err:>+8.5f}")
    print(f"  RMSE residual: {result.rmse:.6e}")
    print(f"  Success: {result.success}, n_iter: {result.n_iter}")

    print()
    print("=" * 65)
    print("IV inversion smoke test")
    print("=" * 65)
    sigma_truth = 0.25
    K, T = 100.0, 0.5
    C = black_scholes_call(S0, K, T, r, sigma_truth)
    iv = implied_vol_bs(C, K, T, S0, r)
    print(f"BS price at sigma={sigma_truth}: {C:.6f}")
    print(f"Inverted IV from price        : {iv:.6f}")
    print(f"|recovered - truth|           : {abs(iv - sigma_truth):.2e}")
