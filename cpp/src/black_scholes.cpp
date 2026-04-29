// black_scholes.cpp
//
// Implementation of the Black-Scholes pricer declared in black_scholes.hpp.

#include "black_scholes.hpp"  // Quotes (not <>) signal a project-local header.

#include <cmath>  // For std::log, std::sqrt, std::exp, std::erfc.

namespace quant {

// Anonymous namespace: anything declared inside is visible only within this
// translation unit (this .cpp file). It's the C++ idiom for "private to this
// file", analogous to a leading underscore in Python.
namespace {

constexpr double SQRT2 = 1.41421356237309504880;  // sqrt(2)

// Helper: compute (d1, d2) and return them through references.
//
// In C++, a function can return only one value directly. To "return" two
// values, the standard idioms are:
//   (a) pack them in a struct or std::pair,
//   (b) pass output parameters by reference, as we do here.
//
// The '&' on the parameters means "pass by reference": the function can
// modify the caller's variables directly, no copy is made. Without '&'
// these would be local copies and the assignments would be lost.
void compute_d1_d2(double S, double K, double r, double sigma, double T,
                   double& d1, double& d2) {
    const double sqrt_T = std::sqrt(T);
    d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T);
    d2 = d1 - sigma * sqrt_T;
}

}  // anonymous namespace

double norm_cdf(double x) {
    // N(x) = 0.5 * erfc(-x / sqrt(2))
    // This formulation is numerically stable for large |x|, where computing
    // N(x) as 0.5 * (1 + erf(x / sqrt(2))) suffers catastrophic cancellation
    // for x << 0.
    return 0.5 * std::erfc(-x / SQRT2);
}

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

}  // namespace quant
