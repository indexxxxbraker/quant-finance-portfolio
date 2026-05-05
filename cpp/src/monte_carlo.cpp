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


// Draw a standard normal from rng by inversion.
//
// Uses a local std::uniform_real_distribution with the half-open
// range [0, 1), then maps via Acklam. The probability of drawing
// exactly 0 (which would crash inverse_normal_cdf) is below 2^{-53}
// per draw and is ignored; if it occurs the exception propagates
// out of the simulation.
//
// The distribution is constructed on each call to preserve byte-exact
// behaviour with the inline implementation used in the Block 1.1
// version of simulate_terminal_gbm: a per-call local distribution is
// guaranteed to be in its initial state, whereas a thread_local one
// could in principle carry residual state.
double standard_normal(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const double u = uniform(rng);
    return inverse_normal_cdf(u);
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

void validate_n_steps(std::size_t n_steps) {
    if (n_steps < 1) {
        throw std::invalid_argument("n_steps must be at least 1");
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
// simulate_terminal_gbm  (Block 1.1)
// =====================================================================

std::vector<double>
simulate_terminal_gbm(double S0, double r, double sigma, double T,
                      std::size_t n_paths,
                      std::mt19937_64& rng) {
    validate_model_params(S0, sigma, T);
    validate_n_paths(n_paths);

    const double drift     = (r - 0.5 * sigma * sigma) * T;
    const double diffusion = sigma * std::sqrt(T);

    std::vector<double> S_T(n_paths);
    for (std::size_t i = 0; i < n_paths; ++i) {
        const double z = standard_normal(rng);
        S_T[i] = S0 * std::exp(drift + diffusion * z);
    }
    return S_T;
}


// =====================================================================
// mc_european_call_exact  (Block 1.1)
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


// =====================================================================
// simulate_path_euler  (Block 1.2.1)
// =====================================================================

std::vector<std::vector<double>>
simulate_path_euler(double S0, double r, double sigma, double T,
                    std::size_t n_steps, std::size_t n_paths,
                    std::mt19937_64& rng) {
    validate_model_params(S0, sigma, T);
    validate_n_steps(n_steps);
    validate_n_paths(n_paths);

    const double h       = T / static_cast<double>(n_steps);
    const double sqrt_h  = std::sqrt(h);
    const double rh      = r * h;

    // Allocate the (n_paths x (n_steps + 1)) matrix.
    std::vector<std::vector<double>> paths(
        n_paths, std::vector<double>(n_steps + 1));

    for (std::size_t i = 0; i < n_paths; ++i) {
        paths[i][0] = S0;
        for (std::size_t k = 0; k < n_steps; ++k) {
            // dW ~ N(0, h), generated as sqrt(h) * Z with Z ~ N(0, 1).
            const double z  = standard_normal(rng);
            const double dW = sqrt_h * z;
            paths[i][k + 1] = paths[i][k] * (1.0 + rh + sigma * dW);
        }
    }
    return paths;
}


// =====================================================================
// simulate_terminal_euler  (Block 1.2.1)
// =====================================================================

std::vector<double>
simulate_terminal_euler(double S0, double r, double sigma, double T,
                        std::size_t n_steps, std::size_t n_paths,
                        std::mt19937_64& rng) {
    validate_model_params(S0, sigma, T);
    validate_n_steps(n_steps);
    validate_n_paths(n_paths);

    const double h       = T / static_cast<double>(n_steps);
    const double sqrt_h  = std::sqrt(h);
    const double rh      = r * h;

    // O(n_paths) memory: the only state kept across steps is the
    // current price vector.
    std::vector<double> S(n_paths, S0);
    for (std::size_t i = 0; i < n_paths; ++i) {
        for (std::size_t k = 0; k < n_steps; ++k) {
            const double z  = standard_normal(rng);
            const double dW = sqrt_h * z;
            S[i] *= (1.0 + rh + sigma * dW);
        }
    }
    return S;
}


// =====================================================================
// mc_european_call_euler  (Block 1.2.1)
// =====================================================================

MCResult
mc_european_call_euler(double S, double K, double r, double sigma, double T,
                       std::size_t n_steps, std::size_t n_paths,
                       std::mt19937_64& rng,
                       double confidence_level) {
    validate_model_params(S, sigma, T);
    validate_strike(K);
    validate_n_steps(n_steps);
    validate_n_paths(n_paths);
    validate_confidence_level(confidence_level);

    const std::vector<double> S_T =
        simulate_terminal_euler(S, r, sigma, T, n_steps, n_paths, rng);

    const double discount = std::exp(-r * T);
    std::vector<double> Y(n_paths);
    for (std::size_t i = 0; i < n_paths; ++i) {
        Y[i] = discount * std::max(S_T[i] - K, 0.0);
    }

    return mc_estimator(Y, confidence_level);
}

}  // namespace quant
