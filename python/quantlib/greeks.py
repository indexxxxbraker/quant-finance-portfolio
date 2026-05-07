"""Greeks for the European call by Monte Carlo.

Three techniques are implemented for Delta and Vega:
  - Bumping (finite differences with common random numbers).
  - Pathwise sensitivities (differentiation of the payoff).
  - Likelihood ratio (differentiation of the kernel).

For Gamma, only bumping (central second differences) is implemented.
The pathwise method fails because the kink of max(.,0) produces a
Dirac after the second derivative; the LR method has very high
variance for Gamma and is not pursued here.

Each estimator returns an MCResult with the same conventions as the
Block 1.1 pricers (estimate, half_width, sample_variance, n_paths).

References
----------
Phase 2 Block 4 writeup. Broadie and Glasserman, "Estimating security
price derivatives using simulation", Management Science 42(2),
269-285, 1996. Glasserman, *Monte Carlo Methods in Financial
Engineering*, Chapter 7.
"""

import math

import numpy as np
from scipy.stats import norm

from quantlib.gbm import (
    _resolve_increments,
    _standard_normals,
    validate_model_params,
    validate_n_paths,
    validate_strike,
)
from quantlib.monte_carlo import MCResult, _resolve_rng, mc_estimator


# =====================================================================
# Bumping (common-random-numbers central differences)
# =====================================================================

def _terminal_values_with_crn(S0, sigma, T, Z):
    """Evaluate S_T = S0 * exp((r-sigma^2/2)*T + sigma*sqrt(T)*Z) for
    a given S0 and sigma, sharing the same Z across calls.

    Note: r is captured implicitly through the caller via the multiplier
    we apply outside; here we accept S0, sigma, T and pre-built Z.
    The caller passes the appropriate (S0, sigma) to bump.
    """
    drift = -0.5 * sigma * sigma * T
    diffusion = sigma * math.sqrt(T)
    return S0 * np.exp(drift + diffusion * Z)


def _crn_call_payoffs(S0, K, r, sigma, T, Z):
    """Discounted call payoffs at parameters (S0, sigma) using the
    given Z draws (CRN).
    """
    drift = (r - 0.5 * sigma * sigma) * T
    diffusion = sigma * math.sqrt(T)
    discount = math.exp(-r * T)
    S_T = S0 * np.exp(drift + diffusion * Z)
    return discount * np.maximum(S_T - K, 0.0)


def delta_bump(S, K, r, sigma, T, n_paths,
               *, h=1e-2, seed=None, rng=None,
               confidence_level=0.95):
    """Delta by central finite differences with common random numbers.

    Bump is applied multiplicatively: S0 -> S0 * (1 +/- h). This makes
    the bump scale-invariant (as opposed to additive bumping where the
    optimal h would depend on S0).
    """
    validate_model_params(S, r, sigma, T)
    validate_strike(K)
    validate_n_paths(n_paths)
    rng = _resolve_rng(seed, rng)

    Z = _standard_normals(n_paths, rng)

    # CRN: same Z for both up and down evaluations.
    f_up   = _crn_call_payoffs(S * (1 + h), K, r, sigma, T, Z)
    f_down = _crn_call_payoffs(S * (1 - h), K, r, sigma, T, Z)

    # Per-path Delta estimate, then average across paths.
    delta_per_path = (f_up - f_down) / (2.0 * S * h)

    return mc_estimator(delta_per_path, confidence_level=confidence_level)


def vega_bump(S, K, r, sigma, T, n_paths,
              *, h=1e-2, seed=None, rng=None,
              confidence_level=0.95):
    """Vega by central finite differences with common random numbers.

    Bump is applied additively in absolute volatility units: sigma ->
    sigma +/- h. Typical h: 1e-2 (i.e., 1 vol point).
    """
    validate_model_params(S, r, sigma, T)
    validate_strike(K)
    validate_n_paths(n_paths)
    rng = _resolve_rng(seed, rng)

    Z = _standard_normals(n_paths, rng)

    f_up   = _crn_call_payoffs(S, K, r, sigma + h, T, Z)
    f_down = _crn_call_payoffs(S, K, r, sigma - h, T, Z)

    vega_per_path = (f_up - f_down) / (2.0 * h)

    return mc_estimator(vega_per_path, confidence_level=confidence_level)


def gamma_bump(S, K, r, sigma, T, n_paths,
               *, h=1e-2, seed=None, rng=None,
               confidence_level=0.95):
    """Gamma by central second finite differences with common random
    numbers.

    Three pricer evaluations: at S*(1+h), S, S*(1-h). The standard
    second-difference formula gives Gamma = (f+ - 2 f0 + f-) / (S*h)^2.

    Note: Gamma's variance scales like 1/h^2 even with CRN, so this is
    intrinsically a high-variance estimator. The half-width can be
    much wider than for Delta/Vega.
    """
    validate_model_params(S, r, sigma, T)
    validate_strike(K)
    validate_n_paths(n_paths)
    rng = _resolve_rng(seed, rng)

    Z = _standard_normals(n_paths, rng)

    f_up   = _crn_call_payoffs(S * (1 + h), K, r, sigma, T, Z)
    f_mid  = _crn_call_payoffs(S,           K, r, sigma, T, Z)
    f_down = _crn_call_payoffs(S * (1 - h), K, r, sigma, T, Z)

    gamma_per_path = (f_up - 2.0 * f_mid + f_down) / ((S * h) ** 2)

    return mc_estimator(gamma_per_path, confidence_level=confidence_level)


# =====================================================================
# Pathwise sensitivities
# =====================================================================

def delta_pathwise(S, K, r, sigma, T, n_paths,
                   *, seed=None, rng=None,
                   confidence_level=0.95):
    """Delta by pathwise differentiation.

    Estimator: e^{-rT} * 1_{S_T > K} * S_T / S0.

    The discount factor cancels in the ratio S_T/S0; what remains is
    the indicator-times-multiplier. This is the standard pathwise
    Delta for the European call.
    """
    validate_model_params(S, r, sigma, T)
    validate_strike(K)
    validate_n_paths(n_paths)
    rng = _resolve_rng(seed, rng)

    Z = _standard_normals(n_paths, rng)
    drift = (r - 0.5 * sigma * sigma) * T
    diffusion = sigma * math.sqrt(T)
    discount = math.exp(-r * T)

    S_T = S * np.exp(drift + diffusion * Z)
    in_money = (S_T > K).astype(np.float64)

    delta_per_path = discount * in_money * S_T / S

    return mc_estimator(delta_per_path, confidence_level=confidence_level)


def vega_pathwise(S, K, r, sigma, T, n_paths,
                  *, seed=None, rng=None,
                  confidence_level=0.95):
    """Vega by pathwise differentiation.

    Estimator: e^{-rT} * 1_{S_T > K} * S_T * (sqrt(T)*Z - sigma*T).

    Derivation: dS_T/dsigma = S_T * (sqrt(T)*Z - sigma*T).
    """
    validate_model_params(S, r, sigma, T)
    validate_strike(K)
    validate_n_paths(n_paths)
    rng = _resolve_rng(seed, rng)

    Z = _standard_normals(n_paths, rng)
    drift = (r - 0.5 * sigma * sigma) * T
    diffusion = sigma * math.sqrt(T)
    discount = math.exp(-r * T)

    S_T = S * np.exp(drift + diffusion * Z)
    in_money = (S_T > K).astype(np.float64)

    sqrt_T = math.sqrt(T)
    vega_per_path = discount * in_money * S_T * (sqrt_T * Z - sigma * T)

    return mc_estimator(vega_per_path, confidence_level=confidence_level)


# =====================================================================
# Likelihood ratio
# =====================================================================

def delta_lr(S, K, r, sigma, T, n_paths,
             *, seed=None, rng=None,
             confidence_level=0.95):
    """Delta by likelihood ratio.

    Estimator: e^{-rT} * max(S_T - K, 0) * Z / (S0 * sigma * sqrt(T)).

    Score for log S_T with respect to log S0 (which equals log S0):
    d/dS0 log p = (log S_T - mu) / (S0 * sigma^2 * T) = Z / (S0 * sigma * sqrt(T)).
    """
    validate_model_params(S, r, sigma, T)
    validate_strike(K)
    validate_n_paths(n_paths)
    rng = _resolve_rng(seed, rng)

    Z = _standard_normals(n_paths, rng)
    drift = (r - 0.5 * sigma * sigma) * T
    diffusion = sigma * math.sqrt(T)
    discount = math.exp(-r * T)

    S_T = S * np.exp(drift + diffusion * Z)
    payoff = discount * np.maximum(S_T - K, 0.0)

    sqrt_T = math.sqrt(T)
    score = Z / (S * sigma * sqrt_T)
    delta_per_path = payoff * score

    return mc_estimator(delta_per_path, confidence_level=confidence_level)


def vega_lr(S, K, r, sigma, T, n_paths,
            *, seed=None, rng=None,
            confidence_level=0.95):
    """Vega by likelihood ratio.

    Estimator: e^{-rT} * max(S_T - K, 0) * ((Z^2 - 1)/sigma - Z*sqrt(T)).

    Score with respect to sigma:
    d/dsigma log p = -1/sigma + (Z^2 / sigma) - sqrt(T) * Z
                   = (Z^2 - 1) / sigma - sqrt(T) * Z
    where Z = (log S_T - mu) / (sigma sqrt(T)).
    """
    validate_model_params(S, r, sigma, T)
    validate_strike(K)
    validate_n_paths(n_paths)
    rng = _resolve_rng(seed, rng)

    Z = _standard_normals(n_paths, rng)
    drift = (r - 0.5 * sigma * sigma) * T
    diffusion = sigma * math.sqrt(T)
    discount = math.exp(-r * T)

    S_T = S * np.exp(drift + diffusion * Z)
    payoff = discount * np.maximum(S_T - K, 0.0)

    sqrt_T = math.sqrt(T)
    score = (Z * Z - 1.0) / sigma - sqrt_T * Z
    vega_per_path = payoff * score

    return mc_estimator(vega_per_path, confidence_level=confidence_level)


# =====================================================================
# Smoke test
# =====================================================================

if __name__ == "__main__":
    from quantlib.black_scholes import (
        call_delta, call_rho, gamma, vega,
    )

    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    n = 100_000

    # BS reference Greeks.
    bs_delta = call_delta(S, K, r, sigma, T)
    bs_vega  = vega(S, K, r, sigma, T)
    bs_gamma = gamma(S, K, r, sigma, T)
    print(f"BS Delta = {bs_delta:.6f}")
    print(f"BS Vega  = {bs_vega:.6f}")
    print(f"BS Gamma = {bs_gamma:.6f}\n")

    # Delta with three methods.
    print("Delta:")
    for name, fn in [
        ("bump    ", delta_bump),
        ("pathwise", delta_pathwise),
        ("LR      ", delta_lr),
    ]:
        result = fn(S, K, r, sigma, T, n, seed=42)
        print(f"  {name}: est = {result.estimate:.6f}  hw = {result.half_width:.6f}")

    # Vega with three methods.
    print("\nVega:")
    for name, fn in [
        ("bump    ", vega_bump),
        ("pathwise", vega_pathwise),
        ("LR      ", vega_lr),
    ]:
        result = fn(S, K, r, sigma, T, n, seed=42)
        print(f"  {name}: est = {result.estimate:.6f}  hw = {result.half_width:.6f}")

    # Gamma with bump only.
    print("\nGamma:")
    result = gamma_bump(S, K, r, sigma, T, n, seed=42)
    print(f"  bump (only): est = {result.estimate:.6f}  hw = {result.half_width:.6f}")
