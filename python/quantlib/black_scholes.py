"""
Black-Scholes pricing for European options on non-dividend-paying assets.

Under the Black-Scholes (1973) model, the underlying price S follows a
geometric Brownian motion. The risk-neutral price of a European call with
strike K and maturity T is

    C(S, K, r, sigma, T) = S * N(d1) - K * exp(-r*T) * N(d2),

with

    d1 = [log(S/K) + (r + sigma**2 / 2) * T] / (sigma * sqrt(T)),
    d2 = d1 - sigma * sqrt(T),

where N is the standard normal CDF. The put price follows from put-call
parity, C - P = S - K * exp(-r*T).

See theory/phase1/black_scholes.tex for the derivation.
"""

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, r, sigma, T):
    """
    Compute d1 and d2 in the Black-Scholes formula.

    All inputs may be scalars or NumPy arrays of broadcast-compatible shape.
    Assumes sigma > 0 and T > 0; the boundary cases (sigma=0, T=0) are not
    handled here and should be addressed at the call site if relevant.
    """
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def call_price(S, K, r, sigma, T):
    """
    Black-Scholes price of a European call option.

    Parameters
    ----------
    S : float or array_like
        Spot price of the underlying.
    K : float or array_like
        Strike price.
    r : float or array_like
        Continuously-compounded risk-free rate.
    sigma : float or array_like
        Volatility (annualised standard deviation of log-returns).
    T : float or array_like
        Time to maturity, in years.

    Returns
    -------
    float or ndarray
        Call price. Shape is the broadcast of the inputs.
    """
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def put_price(S, K, r, sigma, T):
    """
    Black-Scholes price of a European put option.

    See `call_price` for parameter descriptions.
    """
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


if __name__ == "__main__":
    # Hull, Options, Futures, and Other Derivatives (10th ed.), Example 15.6:
    # S = 42, K = 40, r = 10%, sigma = 20%, T = 0.5 years.
    # Reference values: C ~ 4.7594, P ~ 0.8086.
    S, K, r, sigma, T = 42.0, 40.0, 0.10, 0.20, 0.5

    C = call_price(S, K, r, sigma, T)
    P = put_price(S, K, r, sigma, T)

    print(f"Call: {C:.4f}   (Hull: 4.7594)")
    print(f"Put:  {P:.4f}   (Hull: 0.8086)")

    # Sanity check: put-call parity residual.
    parity_residual = abs((C - P) - (S - K * np.exp(-r * T)))
    print(f"Put-call parity residual: {parity_residual:.2e}")
