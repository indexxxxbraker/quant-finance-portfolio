"""
Black-Scholes pricing and Greeks for European options on non-dividend-paying
assets.

Pricing: see derivation in theory/phase1/black_scholes.tex.

Greeks (call):
    Delta = N(d1)
    Gamma = phi(d1) / (S * sigma * sqrt(T))
    Vega  = S * phi(d1) * sqrt(T)
    Theta = -S * phi(d1) * sigma / (2*sqrt(T)) - r*K*exp(-r*T)*N(d2)
    Rho   = K * T * exp(-r*T) * N(d2)

For the put:
    Delta_put = N(d1) - 1 = -N(-d1)
    Rho_put   = -K * T * exp(-r*T) * N(-d2)
Gamma, Vega, Theta are identical to the call (consequence of put-call
parity: differentiating C - P = S - K*exp(-r*T) kills terms involving
sigma, S^2 derivatives, and the time derivative).

All functions are vectorised: scalar or array inputs of broadcast-compatible
shape are accepted.

Theta convention: this module returns dC/dt (calendar-time derivative).
Some references use -dC/dT instead, which differs by a sign. The chosen
convention matches the Black-Scholes PDE directly:
    Theta + 0.5 * sigma^2 * S^2 * Gamma + r * S * Delta - r * C = 0.
"""

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, r, sigma, T):
    """Compute d1 and d2 in the Black-Scholes formula."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------
def call_price(S, K, r, sigma, T):
    """Black-Scholes price of a European call."""
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def put_price(S, K, r, sigma, T):
    """Black-Scholes price of a European put."""
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------
def call_delta(S, K, r, sigma, T):
    """Delta of a European call: dC/dS."""
    d1, _ = _d1_d2(S, K, r, sigma, T)
    return norm.cdf(d1)


def put_delta(S, K, r, sigma, T):
    """Delta of a European put: dP/dS = N(d1) - 1."""
    d1, _ = _d1_d2(S, K, r, sigma, T)
    return norm.cdf(d1) - 1.0


def gamma(S, K, r, sigma, T):
    """Gamma: d^2 C/dS^2. Identical for calls and puts."""
    d1, _ = _d1_d2(S, K, r, sigma, T)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, r, sigma, T):
    """Vega: dC/dsigma. Identical for calls and puts.

    Returned as the raw partial derivative. In market practice Vega is often
    quoted per 1% change in sigma (i.e. divided by 100); rescaling is left
    to the caller.
    """
    d1, _ = _d1_d2(S, K, r, sigma, T)
    return S * norm.pdf(d1) * np.sqrt(T)


def call_theta(S, K, r, sigma, T):
    """Theta of a European call: dC/dt (calendar time, not time-to-maturity).

    Returned in 'per year' units. Divide by 365 for per-day Theta.
    """
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    return (
        -S * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm.cdf(d2)
    )


def put_theta(S, K, r, sigma, T):
    """Theta of a European put: dP/dt."""
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    return (
        -S * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
        + r * K * np.exp(-r * T) * norm.cdf(-d2)
    )


def call_rho(S, K, r, sigma, T):
    """Rho of a European call: dC/dr."""
    _, d2 = _d1_d2(S, K, r, sigma, T)
    return K * T * np.exp(-r * T) * norm.cdf(d2)


def put_rho(S, K, r, sigma, T):
    """Rho of a European put: dP/dr."""
    _, d2 = _d1_d2(S, K, r, sigma, T)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2)


# ---------------------------------------------------------------------------
# Payoff statistics (Phase 2 Block 1.1)
# ---------------------------------------------------------------------------
def call_payoff_variance(S, K, r, sigma, T):
    """Closed-form variance of the discounted European call payoff
    Y = e^{-rT} (S_T - K)^+ under geometric Brownian motion.

    Parameters
    ----------
    S, K, r, sigma, T : float or array_like
        Standard Black-Scholes inputs (broadcast).

    Returns
    -------
    float or ndarray
        Var(Y). Always non-negative.

    Notes
    -----
    Used in two contexts:

    1. A-priori sample-size selection in vanilla Monte Carlo:
       n_eps ~ (z / eps)^2 * Var(Y) for half-width eps.
    2. As the denominator of the variance reduction factor reported
       throughout Phase 2 Block 2.

    Derivation in theory/phase2/mc_european_exact.tex,
    Proposition 5.1.
    """
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    sqrt_T = np.sqrt(T)

    # E[Y^2] expanded after distributing e^{-2rT}:
    #   term1 = e^{-2rT} * E[S_T^2 * 1_{S_T > K}]
    #         = S^2 * exp(sigma^2 * T) * N(d1 + sigma * sqrt(T))
    #   term2 = e^{-2rT} * (-2K) * E[S_T * 1_{S_T > K}]
    #         = -2 * K * S * exp(-r * T) * N(d1)
    #   term3 = e^{-2rT} * K^2 * P(S_T > K)
    #         = K^2 * exp(-2 * r * T) * N(d2)
    term1 = S * S * np.exp(sigma * sigma * T) * norm.cdf(d1 + sigma * sqrt_T)
    term2 = -2.0 * K * S * np.exp(-r * T) * norm.cdf(d1)
    term3 = K * K * np.exp(-2.0 * r * T) * norm.cdf(d2)
    second_moment_of_Y = term1 + term2 + term3

    bs_call = call_price(S, K, r, sigma, T)

    return second_moment_of_Y - bs_call * bs_call


if __name__ == "__main__":
    # Hull example 15.6 reference values for the price.
    S, K, r, sigma, T = 42.0, 40.0, 0.10, 0.20, 0.5

    print(f"S={S}, K={K}, r={r}, sigma={sigma}, T={T}\n")
    print(f"Call price : {call_price(S, K, r, sigma, T):.6f}")
    print(f"Put price  : {put_price (S, K, r, sigma, T):.6f}\n")

    print(f"Call Delta : {call_delta(S, K, r, sigma, T):.6f}")
    print(f"Put  Delta : {put_delta (S, K, r, sigma, T):.6f}")
    print(f"Gamma      : {gamma     (S, K, r, sigma, T):.6f}")
    print(f"Vega       : {vega      (S, K, r, sigma, T):.6f}")
    print(f"Call Theta : {call_theta(S, K, r, sigma, T):.6f}")
    print(f"Put  Theta : {put_theta (S, K, r, sigma, T):.6f}")
    print(f"Call Rho   : {call_rho  (S, K, r, sigma, T):.6f}")
    print(f"Put  Rho   : {put_rho   (S, K, r, sigma, T):.6f}")
