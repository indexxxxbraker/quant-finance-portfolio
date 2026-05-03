// implied_volatility.cpp
//
// Implementation of the implied-volatility solver declared in
// implied_volatility.hpp.
//
// Algorithm (mirror of the Python implementation):
//   1. Validate the no-arbitrage bounds; throw if violated.
//   2. Choose initial guess sigma_0:
//      - If |log(S/K)| < 0.3, use Brenner-Subrahmanyam.
//      - Otherwise, use 0.3 (neutral default).
//   3. Iterate Newton-Raphson with safeguards:
//      (a) vanishing Vega -> fallback,
//      (b) negative iterate -> fallback,
//      (c) too many iterations -> fallback.
//   4. Fallback: bisection on the bracket [1e-4, 10.0].

#include "implied_volatility.hpp"
#include "black_scholes.hpp"

#include <cmath>
#include <sstream>

namespace quant {

namespace {

// Algorithmic constants. Match the Python defaults.
constexpr double VEGA_FLOOR     = 1e-10;
constexpr double SIGMA_LOW      = 1e-4;
constexpr double SIGMA_HIGH     = 10.0;
constexpr double LOG_MONEYNESS_ATM_THRESHOLD = 0.3;
constexpr double DEFAULT_SIGMA_GUESS = 0.3;
constexpr double TWO_PI         = 6.28318530717958647692;

/// Choose initial guess. See Python _initial_guess for rationale.
double initial_guess(double C_market, double S, double K, double T) {
    if (std::abs(std::log(S / K)) < LOG_MONEYNESS_ATM_THRESHOLD) {
        const double sigma = std::sqrt(TWO_PI / T) * (C_market - 0.5 * (S - K)) / S;
        return sigma > 1e-3 ? sigma : 1e-3;
    }
    return DEFAULT_SIGMA_GUESS;
}

/// Bisection fallback on [SIGMA_LOW, SIGMA_HIGH].
///
/// Bisection rather than Brent: simpler, no external dependency, and
/// the fallback is rarely invoked. The price for using bisection is
/// linear convergence (versus superlinear for Brent), but this only
/// matters when the fallback fires, which is on edge cases.
double bisection_fallback(double C_market, double S, double K,
                          double r, double T, double tol) {
    double lo = SIGMA_LOW;
    double hi = SIGMA_HIGH;
    double f_lo = call_price(S, K, r, lo, T) - C_market;
    double f_hi = call_price(S, K, r, hi, T) - C_market;

    // The no-arbitrage check upstream guarantees a sign change; this is
    // a defensive check in case of unforeseen numerical pathology.
    if (f_lo * f_hi > 0.0) {
        std::ostringstream oss;
        oss << "bisection_fallback: no sign change in bracket ["
            << lo << ", " << hi << "]; f(lo)=" << f_lo
            << ", f(hi)=" << f_hi;
        throw std::runtime_error(oss.str());
    }

    // Bisect until the residual at the midpoint is below tolerance, or
    // until the bracket itself is too narrow to halve meaningfully.
    constexpr int MAX_BISECT = 100;  // 2^-100 is far below double precision.
    for (int i = 0; i < MAX_BISECT; ++i) {
        const double mid = 0.5 * (lo + hi);
        const double f_mid = call_price(S, K, r, mid, T) - C_market;
        if (std::abs(f_mid) < tol) return mid;
        if (f_mid * f_lo < 0.0) {
            hi = mid;
            f_hi = f_mid;
        } else {
            lo = mid;
            f_lo = f_mid;
        }
    }
    return 0.5 * (lo + hi);
}

}  // anonymous namespace


double implied_volatility(double C_market, double S, double K,
                          double r, double T,
                          double tol, int max_iter) {
    // 1. Precondition: no-arbitrage bounds.
    const double intrinsic_fwd = std::max(S - K * std::exp(-r * T), 0.0);
    if (!(intrinsic_fwd < C_market && C_market < S)) {
        std::ostringstream oss;
        oss << "Market price C=" << C_market
            << " violates no-arbitrage bounds (" << intrinsic_fwd
            << ", " << S << "); implied volatility does not exist.";
        throw NoArbitrageBoundsViolation(oss.str());
    }

    // 2. Initial guess.
    double sigma = initial_guess(C_market, S, K, T);

    // 3. Newton-Raphson with safeguards.
    for (int i = 0; i < max_iter; ++i) {
        const double price = call_price(S, K, r, sigma, T);
        const double diff = price - C_market;

        if (std::abs(diff) < tol) return sigma;

        const double v = vega(S, K, r, sigma, T);
        if (v < VEGA_FLOOR) {
            return bisection_fallback(C_market, S, K, r, T, tol);
        }

        const double sigma_new = sigma - diff / v;
        if (sigma_new <= 0.0) {
            return bisection_fallback(C_market, S, K, r, T, tol);
        }

        sigma = sigma_new;
    }

    return bisection_fallback(C_market, S, K, r, T, tol);
}

}  // namespace quant
