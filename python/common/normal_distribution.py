"""
Standard normal distribution utilities.

This module provides convenience wrappers around the standard normal
distribution functions used throughout the portfolio (e.g. in the
Black-Scholes pricing formula).
"""

from scipy.stats import norm


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
    return norm.cdf(x)


def standard_normal_pdf(x: float) -> float:
    """
    Probability density function of the standard normal N(0, 1).
    """
    return norm.pdf(x)


if __name__ == "__main__":
    # Quick sanity check
    test_points = [-3.0, -1.0, 0.0, 1.0, 3.0]
    print("Standard normal CDF values:")
    for x in test_points:
        print(f"  Phi({x:+.1f}) = {standard_normal_cdf(x):.6f}")