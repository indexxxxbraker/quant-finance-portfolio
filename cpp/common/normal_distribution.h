#pragma once

/**
 * @file normal_distribution.h
 * @brief Standard normal distribution utilities.
 *
 * Provides functions for the standard normal distribution N(0, 1)
 * used throughout the portfolio (e.g. in the Black-Scholes formula).
 */

namespace quant {

/**
 * @brief Cumulative distribution function of N(0, 1).
 * @param x Point at which to evaluate the CDF.
 * @return Phi(x) = P(Z <= x) where Z ~ N(0, 1).
 */
double standard_normal_cdf(double x);

/**
 * @brief Probability density function of N(0, 1).
 */
double standard_normal_pdf(double x);

}  // namespace quant