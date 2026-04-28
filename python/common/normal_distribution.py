"""
Standard normal distribution utilities.

This module provides convenience wrappers around the standard normal
distribution functions used throughout the portfolio (e.g. in the
Black-Scholes pricing formula).
"""

import math


def standard_normal_cdf(x: float) -> float:
    """
    Cumulative distribution function of the standard normal N(0, 1).

    Parameters
    ----------
    x : float
        Point at which to evaluate the CDF.

    Returns
    -------
    float
        Phi(x) = P(Z <= x) where Z ~ N(0, 1).
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def standard_normal_pdf(x: float) -> float:
    """
    Probability density function of the standard normal N(0, 1).
    """
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


if __name__ == "__main__":
    # Quick sanity check
    test_points = [-3.0, -1.0, 0.0, 1.0, 3.0]
    print("Standard normal CDF values:")
    for x in test_points:
        print(f"  Phi({x:+.1f}) = {standard_normal_cdf(x):.6f}")