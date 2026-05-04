// black_scholes.cpp
//
// Implementation of the Black-Scholes pricer, Greeks, and the closed-form
// variance of the discounted call payoff declared in black_scholes.hpp.

#include "black_scholes.hpp"

#include <cmath>

namespace quant {

namespace {

constexpr double SQRT2    = 1.41421356237309504880;  // sqrt(2)
constexpr double SQRT2PI  = 2.50662827463100050241;  // sqrt(2*pi)

/// Compute (d1, d2) and return them through references.
void compute_d1_d2(double S, double K, double r, double sigma, double T,
                   double& d1, double& d2) {
    const double sqrt_T = std::sqrt(T);
    d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T);
    d2 = d1 - sigma * sqrt_T;
}

}  // anonymous namespace


// ---------------------------------------------------------------------------
// Standard normal density and CDF
// ---------------------------------------------------------------------------

double norm_pdf(double x) {
    return std::exp(-0.5 * x * x) / SQRT2PI;
}

double norm_cdf(double x) {
    // N(x) = 0.5 * erfc(-x / sqrt(2)). See header for rationale.
    return 0.5 * std::erfc(-x / SQRT2);
}


// ---------------------------------------------------------------------------
// Prices
// ---------------------------------------------------------------------------

double call_price(double S, double K, double r, double sigma, double T) {
    double d1, d2;
    compute_d1_d2(S, K, r, sigma, T, d1, d2);
    return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

double put_price(double S, double K, double r, double sigma, double T) {
    double d1, d2;
    compute_d1_d2(S, K, r, sigma, T, d1, d2);
    return K * std::exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1);
}


// ---------------------------------------------------------------------------
// Greeks
// ---------------------------------------------------------------------------

double call_delta(double S, double K, double r, double sigma, double T) {
    double d1, d2;
    compute_d1_d2(S, K, r, sigma, T, d1, d2);
    return norm_cdf(d1);
}

double put_delta(double S, double K, double r, double sigma, double T) {
    double d1, d2;
    compute_d1_d2(S, K, r, sigma, T, d1, d2);
    return norm_cdf(d1) - 1.0;
}

double gamma(double S, double K, double r, double sigma, double T) {
    double d1, d2;
    compute_d1_d2(S, K, r, sigma, T, d1, d2);
    return norm_pdf(d1) / (S * sigma * std::sqrt(T));
}

double vega(double S, double K, double r, double sigma, double T) {
    double d1, d2;
    compute_d1_d2(S, K, r, sigma, T, d1, d2);
    return S * norm_pdf(d1) * std::sqrt(T);
}

double call_theta(double S, double K, double r, double sigma, double T) {
    double d1, d2;
    compute_d1_d2(S, K, r, sigma, T, d1, d2);
    return -S * norm_pdf(d1) * sigma / (2.0 * std::sqrt(T))
           - r * K * std::exp(-r * T) * norm_cdf(d2);
}

double put_theta(double S, double K, double r, double sigma, double T) {
    double d1, d2;
    compute_d1_d2(S, K, r, sigma, T, d1, d2);
    return -S * norm_pdf(d1) * sigma / (2.0 * std::sqrt(T))
           + r * K * std::exp(-r * T) * norm_cdf(-d2);
}

double call_rho(double S, double K, double r, double sigma, double T) {
    double d1, d2;
    compute_d1_d2(S, K, r, sigma, T, d1, d2);
    return K * T * std::exp(-r * T) * norm_cdf(d2);
}

double put_rho(double S, double K, double r, double sigma, double T) {
    double d1, d2;
    compute_d1_d2(S, K, r, sigma, T, d1, d2);
    return -K * T * std::exp(-r * T) * norm_cdf(-d2);
}


// ---------------------------------------------------------------------------
// Payoff statistics (Phase 2 Block 1.1)
// ---------------------------------------------------------------------------

double call_payoff_variance(double S, double K, double r, double sigma, double T) {
    // Closed-form variance of the discounted European call payoff
    // Y = exp(-r*T) * max(S_T - K, 0) under geometric Brownian motion.
    // Derivation: theory/phase2/mc_european_exact.tex, Proposition 5.1.
    //
    // Note: d1, d2 are recomputed locally rather than calling compute_d1_d2,
    // to keep this function readable in isolation.
    const double sqrt_T = std::sqrt(T);
    const double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T)
                    / (sigma * sqrt_T);
    const double d2 = d1 - sigma * sqrt_T;

    // Three terms of E[Y^2] after distributing exp(-2rT):
    const double term1 = S * S * std::exp(sigma * sigma * T)
                       * norm_cdf(d1 + sigma * sqrt_T);
    const double term2 = -2.0 * K * S * std::exp(-r * T) * norm_cdf(d1);
    const double term3 = K * K * std::exp(-2.0 * r * T) * norm_cdf(d2);
    const double second_moment_of_Y = term1 + term2 + term3;

    const double bs_call = call_price(S, K, r, sigma, T);

    return second_moment_of_Y - bs_call * bs_call;
}

}  // namespace quant
