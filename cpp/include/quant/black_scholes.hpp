// black_scholes.hpp
//
// Black-Scholes pricing for European options on a non-dividend-paying asset.
// Declares the public interface; the implementation lives in black_scholes.cpp.
//
// See theory/phase1/black_scholes.tex for the mathematical derivation.

#pragma once  // Tells the compiler to include this file at most once per
              // translation unit, even if it's referenced multiple times.
              // The traditional alternative is the "include guard" pattern
              // with #ifndef / #define / #endif; #pragma once is shorter,
              // universally supported by modern compilers, and the
              // recommended modern style.

namespace quant {

/**
 * Standard normal cumulative distribution function.
 *
 * Implemented in terms of std::erfc for numerical stability in the tails.
 */
double norm_cdf(double x);

/**
 * Black-Scholes price of a European call option.
 *
 * @param S      Spot price of the underlying.
 * @param K      Strike price.
 * @param r      Continuously-compounded risk-free rate.
 * @param sigma  Volatility (annualised standard deviation of log-returns).
 * @param T      Time to maturity, in years.
 * @return       The call price.
 *
 * Preconditions: S > 0, K > 0, sigma > 0, T > 0. Boundary cases are not
 * handled here.
 */
double call_price(double S, double K, double r, double sigma, double T);

/**
 * Black-Scholes price of a European put option. See call_price for parameters.
 */
double put_price(double S, double K, double r, double sigma, double T);

}  // namespace quant
