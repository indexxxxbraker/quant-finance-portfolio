"""
Implied volatility for European call options under the Black-Scholes model.

The implied volatility is the unique sigma > 0 (when it exists) satisfying

    C_BS(S, K, r, sigma, T) = C_market.

By the existence-and-uniqueness theorem (see implied_volatility.tex), this
inversion is well-posed iff the market price respects the no-arbitrage bounds

    max(S - K * exp(-r*T), 0) < C_market < S.

The numerical strategy:

  1. Validate the no-arbitrage bounds; raise ValueError if violated.
  2. Choose an initial guess sigma_0:
       - If the option is near ATM (|log(S/K)| < 0.3), use the
         Brenner-Subrahmanyam approximation, which is tight in that regime.
       - Otherwise, use a neutral default (sigma_0 = 0.3): the BS
         approximation is unreliable away from ATM and can give absurd
         starting values that send Newton to wrong attractors.
  3. Iterate Newton-Raphson with safeguards:
       (a) vanishing Vega -> fallback,
       (b) negative iterate -> fallback,
       (c) too many iterations -> fallback.
  4. Fallback: scipy.optimize.brentq on a conservative bracket.

See theory/phase1/implied_volatility.tex for the mathematical foundations.
"""

import numpy as np
from scipy.optimize import brentq

from quantlib.black_scholes import call_price, vega


# ---------------------------------------------------------------------------
# Default solver parameters
# ---------------------------------------------------------------------------
DEFAULT_TOL = 1e-8
DEFAULT_MAX_ITER = 50
VEGA_FLOOR = 1e-10
SIGMA_BRACKET = (1e-4, 10.0)

# Threshold beyond which Brenner-Subrahmanyam is unreliable. The BS
# expansion is valid for |log(S/K)| small, so we use a moderate cutoff.
LOG_MONEYNESS_ATM_THRESHOLD = 0.3
DEFAULT_SIGMA_GUESS = 0.3  # Neutral starting value when BS not applicable.


def _initial_guess(C_market, S, K, T):
    """
    Choose an initial guess for Newton-Raphson.

    For options near at-the-money (|log(S/K)| < 0.3), use the
    Brenner-Subrahmanyam (1988) approximation, derived from a Taylor
    expansion of C_BS around sigma*sqrt(T) = 0 in the ATM regime:

        sigma_0 = sqrt(2*pi/T) * (C_market - 0.5*(S-K)) / S.

    For options away from the money, this expansion does not hold and
    can produce values orders of magnitude away from the true sigma,
    sending Newton to a wrong attractor. In that regime we fall back
    to a neutral default of 0.3, which is close to typical equity
    volatilities and lies safely inside the basin of attraction for
    most realistic problems.

    The clamp to a positive minimum protects against pathological
    near-bound prices that may yield negative or near-zero guesses.
    """
    if abs(np.log(S / K)) < LOG_MONEYNESS_ATM_THRESHOLD:
        sigma = np.sqrt(2.0 * np.pi / T) * (C_market - 0.5 * (S - K)) / S
        return max(sigma, 1e-3)
    return DEFAULT_SIGMA_GUESS


def implied_volatility(
    C_market,
    S,
    K,
    r,
    T,
    tol=DEFAULT_TOL,
    max_iter=DEFAULT_MAX_ITER,
):
    """
    Compute the Black-Scholes implied volatility of a European call.

    Parameters
    ----------
    C_market : float
        Observed market price of the call.
    S, K, r, T : float
        Spot, strike, risk-free rate, time to maturity (in years).
    tol : float, optional
        Convergence tolerance on the absolute price residual.
    max_iter : int, optional
        Maximum Newton iterations before falling back to Brent.

    Returns
    -------
    float
        The implied volatility sigma > 0.

    Raises
    ------
    ValueError
        If the market price violates the no-arbitrage bounds.
    """
    # 1. Precondition: no-arbitrage bounds.
    intrinsic_fwd = max(S - K * np.exp(-r * T), 0.0)
    if not (intrinsic_fwd < C_market < S):
        raise ValueError(
            f"Market price C={C_market} violates no-arbitrage bounds "
            f"({intrinsic_fwd}, {S}); implied volatility does not exist."
        )

    # 2. Initial guess.
    sigma = _initial_guess(C_market, S, K, T)

    # 3. Newton-Raphson loop with safeguards.
    for _ in range(max_iter):
        price = call_price(S, K, r, sigma, T)
        diff = price - C_market

        if abs(diff) < tol:
            return float(sigma)

        v = vega(S, K, r, sigma, T)
        if v < VEGA_FLOOR:
            return _brent_fallback(C_market, S, K, r, T, tol)

        sigma_new = sigma - diff / v
        if sigma_new <= 0:
            return _brent_fallback(C_market, S, K, r, T, tol)

        sigma = sigma_new

    return _brent_fallback(C_market, S, K, r, T, tol)


def _brent_fallback(C_market, S, K, r, T, tol):
    """
    Bracketing fallback using scipy.optimize.brentq.

    Brent's algorithm combines bisection (guaranteed convergence) with
    inverse quadratic interpolation and the secant method (superlinear
    speed when the function is well-behaved). The bracket [1e-4, 10.0]
    contains any realistic implied volatility.
    """
    objective = lambda sigma: call_price(S, K, r, sigma, T) - C_market
    return float(brentq(objective, *SIGMA_BRACKET, xtol=tol))
