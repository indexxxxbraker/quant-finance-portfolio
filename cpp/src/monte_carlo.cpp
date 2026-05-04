// monte_carlo.cpp — implementation of monte_carlo.hpp.

#include "monte_carlo.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace quant {
namespace {

// =====================================================================
// Inverse standard normal CDF (Acklam's approximation, 2003)
// =====================================================================
//
// Rational approximation of Phi^{-1}(p) for p in (0, 1), with
// relative error below 1.15e-9 over the entire support. This is well
// below the statistical noise of Monte Carlo at the sample sizes used
// in practice, so for our purposes the approximation is exact.
//
// We implement this manually rather than depend on Boost, to keep the
// project self-contained (same decision as for the IV bisection
// fallback in Phase 1).
//
// Reference: P. J. Acklam, "An algorithm for computing the inverse
// normal cumulative distribution function" (technical note, 2003).
double inverse_normal_cdf(double p) {
    if (!(p > 0.0 && p < 1.0)) {
        throw std::invalid_argument(
            "inverse_normal_cdf: p must be in (0, 1)");
    }

    // Coefficients of the rational approximation.
    constexpr double a[6] = {
        -3.969683028665376e+01,  2.209460984245205e+02,
        -2.759285104469687e+02,  1.383577518672690e+02,
        -3.066479806614716e+01,  2.506628277459239e+00
    };
    constexpr double b[5] = {
        -5.447609879822406e+01,  1.615858368580409e+02,
        -1.556989798598866e+02,  6.680131188771972e+01,
        -1.328068155288572e+01
    };
    constexpr double c[6] = {
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00
    };
    constexpr double d[4] = {
         7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00
    };

    // Break-points between tail and central regions.
    constexpr double p_low  = 0.02425;
    constexpr double p_high = 1.0 - p_low;

    if (p < p_low) {
        // Lower tail.
        const double q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
             / ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0);
    }
    if (p <= p_high) {
        // Central region.
        const double q = p - 0.5;
        const double r = q * q;
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
             / (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0);
    }
    // Upper tail. Use symmetry: Phi^{-1}(p) = -Phi^{-1}(1 - p).
    const double q = std::sqrt(-2.0 * std::log(1.0 - p));
    return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
          / ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0);
}


// =====================================================================
// Convenience wrappers
// =====================================================================

// Quantile z = Phi^{-1}((1 + confidence_level) / 2).
double standard_normal_quantile(double confidence_level) {
    return inverse_normal_cdf(0.5 * (1.0 + confidence_level));
}


// =====================================================================
// Input validation
// =====================================================================

void validate_model_params(double S0, double sigma, double T) {
    if (S0 <= 0.0) {
        throw std::invalid_argument("S0 must be positive");
    }
    if (sigma <= 0.0) {
        throw std::invalid_argument("sigma must be positive");
    }
    if (T <= 0.0) {
        throw std::invalid_argument("T must be positive");
    }
}

void validate_strike(double K) {
    if (K <= 0.0) {
        throw std::invalid_argument("K must be positive");
    }
}

void validate_n_paths(std::size_t n_paths) {
    if (n_paths < 2) {
        throw std::invalid_argument(
            "n_paths must be at least 2 (for sample variance with "
            "Bessel's correction)");
    }
}

void validate_confidence_level(double confidence_level) {
    if (!(confidence_level > 0.0 && confidence_level < 1.0)) {
        throw std::invalid_argument(
            "confidence_level must be in (0, 1)");
    }
}

}  // namespace (anonymous)


// =====================================================================
// mc_estimator
// =====================================================================

MCResult mc_estimator(const std::vector<double>& Y,
                      double confidence_level) {
    if (Y.size() < 2) {
        throw std::invalid_argument(
            "Y must have at least 2 elements");
    }
    validate_confidence_level(confidence_level);

    const std::size_t n = Y.size();

    // Sample mean.
    double sum = 0.0;
    for (const double y : Y) {
        sum += y;
    }
    const double mean = sum / static_cast<double>(n);

    // Sample variance (Bessel's correction).
    double sum_sq_dev = 0.0;
    for (const double y : Y) {
        const double dev = y - mean;
        sum_sq_dev += dev * dev;
    }
    const double sample_var = sum_sq_dev / static_cast<double>(n - 1);

    const double z          = standard_normal_quantile(confidence_level);
    const double half_width = z * std::sqrt(sample_var
                                          / static_cast<double>(n));

    return MCResult{ mean, half_width, sample_var, n };
}


// =====================================================================
// simulate_terminal_gbm
// =====================================================================

std::vector<double>
simulate_terminal_gbm(double S0, double r, double sigma, double T,
                      std::size_t n_paths,
                      std::mt19937_64& rng) {
    validate_model_params(S0, sigma, T);
    validate_n_paths(n_paths);

    const double drift     = (r - 0.5 * sigma * sigma) * T;
    const double diffusion = sigma * std::sqrt(T);

    // std::uniform_real_distribution returns [0, 1). The probability of
    // drawing exactly 0 (which would make inverse_normal_cdf throw) is
    // below 2^{-53} per draw; for the sample sizes used here it is
    // astronomically small and we ignore it. If it ever happens, the
    // exception propagates out and aborts the simulation cleanly.
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    std::vector<double> S_T(n_paths);
    for (std::size_t i = 0; i < n_paths; ++i) {
        const double u = uniform(rng);
        const double z = inverse_normal_cdf(u);
        S_T[i] = S0 * std::exp(drift + diffusion * z);
    }
    return S_T;
}


// =====================================================================
// mc_european_call_exact
// =====================================================================

MCResult
mc_european_call_exact(double S, double K, double r, double sigma, double T,
                       std::size_t n_paths,
                       std::mt19937_64& rng,
                       double confidence_level) {
    validate_model_params(S, sigma, T);
    validate_strike(K);
    validate_n_paths(n_paths);
    validate_confidence_level(confidence_level);

    const std::vector<double> S_T =
        simulate_terminal_gbm(S, r, sigma, T, n_paths, rng);

    const double discount = std::exp(-r * T);
    std::vector<double> Y(n_paths);
    for (std::size_t i = 0; i < n_paths; ++i) {
        Y[i] = discount * std::max(S_T[i] - K, 0.0);
    }

    return mc_estimator(Y, confidence_level);
}

}  // namespace quant
