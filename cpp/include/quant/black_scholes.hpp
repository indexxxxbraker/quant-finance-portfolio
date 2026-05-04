// black_scholes.hpp
//
// Black-Scholes pricing and Greeks for European options on a non-dividend-
// paying asset. Declares the public interface; the implementation lives in
// black_scholes.cpp.
//
// See theory/phase1/black_scholes.tex for the mathematical derivation,
// and theory/phase2/mc_european_exact.tex (Proposition 5.1) for the
// closed-form variance of the discounted call payoff used by the Monte
// Carlo module.

#pragma once

namespace quant {

// ---------------------------------------------------------------------------
// Standard normal density and CDF
// ---------------------------------------------------------------------------

/// Standard normal probability density function: phi(x) = exp(-x^2/2)/sqrt(2*pi).
double norm_pdf(double x);

/// Standard normal cumulative distribution function.
/// Implemented via std::erfc for numerical stability in the tails.
double norm_cdf(double x);


// ---------------------------------------------------------------------------
// Prices
// ---------------------------------------------------------------------------

/// Black-Scholes price of a European call option.
///
/// @param S      Spot price of the underlying.
/// @param K      Strike price.
/// @param r      Continuously-compounded risk-free rate.
/// @param sigma  Volatility (annualised standard deviation of log-returns).
/// @param T      Time to maturity, in years.
///
/// Preconditions: S > 0, K > 0, sigma > 0, T > 0.
double call_price(double S, double K, double r, double sigma, double T);

/// Black-Scholes price of a European put option.
double put_price(double S, double K, double r, double sigma, double T);


// ---------------------------------------------------------------------------
// Greeks
// ---------------------------------------------------------------------------

/// Delta of a European call: dC/dS = N(d1).
double call_delta(double S, double K, double r, double sigma, double T);

/// Delta of a European put: dP/dS = N(d1) - 1.
double put_delta(double S, double K, double r, double sigma, double T);

/// Gamma: d^2 C/dS^2. Identical for calls and puts.
double gamma(double S, double K, double r, double sigma, double T);

/// Vega: dC/dsigma. Identical for calls and puts.
/// Returned as the raw partial derivative (not divided by 100).
double vega(double S, double K, double r, double sigma, double T);

/// Theta of a European call: dC/dt (calendar time, NOT time-to-maturity).
/// Returned in 'per year' units. Divide by 365 for per-day theta.
double call_theta(double S, double K, double r, double sigma, double T);

/// Theta of a European put: dP/dt.
double put_theta(double S, double K, double r, double sigma, double T);

/// Rho of a European call: dC/dr.
double call_rho(double S, double K, double r, double sigma, double T);

/// Rho of a European put: dP/dr.
double put_rho(double S, double K, double r, double sigma, double T);


// ---------------------------------------------------------------------------
// Payoff statistics (Phase 2 Block 1.1)
// ---------------------------------------------------------------------------

/// Closed-form variance of the discounted European call payoff
/// Y = exp(-rT) * max(S_T - K, 0) under geometric Brownian motion.
///
/// Used in two contexts:
///   1. A-priori sample-size selection in vanilla Monte Carlo.
///   2. As the denominator of the variance reduction factor in Block 2.
///
/// Derivation: theory/phase2/mc_european_exact.tex, Proposition 5.1.
double call_payoff_variance(double S, double K, double r, double sigma, double T);

}  // namespace quant
