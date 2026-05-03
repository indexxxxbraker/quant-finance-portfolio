// implied_volatility.hpp
//
// Implied volatility solver for European call options under the
// Black-Scholes model.
//
// See theory/phase1/implied_volatility.tex for the mathematical foundations.

#pragma once

#include <stdexcept>

namespace quant {

/// Thrown when the market price violates the no-arbitrage bounds.
/// Inherits from std::invalid_argument so callers can catch it via the
/// standard exception hierarchy.
class NoArbitrageBoundsViolation : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

/// Compute the Black-Scholes implied volatility of a European call.
///
/// @param C_market  Observed market price of the call.
/// @param S, K, r, T  Spot, strike, risk-free rate, time to maturity.
/// @param tol       Convergence tolerance on the absolute price residual.
/// @param max_iter  Maximum Newton iterations before falling back to bisection.
/// @return          The implied volatility sigma > 0.
///
/// @throws NoArbitrageBoundsViolation if C_market is outside
///         (max(S - K*exp(-rT), 0), S).
double implied_volatility(double C_market, double S, double K,
                          double r, double T,
                          double tol = 1e-8,
                          int max_iter = 50);

}  // namespace quant
