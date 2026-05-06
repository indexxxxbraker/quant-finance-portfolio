// gbm.cpp -- implementation of gbm.hpp.

#include "gbm.hpp"

#include <cmath>
#include <stdexcept>

namespace quant {

// =====================================================================
// Anonymous-namespace helpers (file-local)
// =====================================================================

namespace {

// Draw a standard normal from rng by inversion.
//
// The distribution is constructed as a local object on each call to
// preserve byte-exact reproducibility with the inline implementation
// of Block 1.1's simulate_terminal_gbm: a per-call local distribution
// is guaranteed to be in its initial state, whereas a thread_local
// one could in principle carry residual state between calls.
double standard_normal(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const double u = uniform(rng);
    return inverse_normal_cdf(u);
}

}  // anonymous namespace


// =====================================================================
// Public: inverse_normal_cdf (Acklam's approximation)
// =====================================================================

double inverse_normal_cdf(double p) {
    if (!(p > 0.0 && p < 1.0)) {
        throw std::invalid_argument(
            "inverse_normal_cdf: p must be in (0, 1)");
    }

    // Coefficients of the rational approximation. Relative error
    // below 1.15e-9 over (0, 1). Reference: P. J. Acklam (2003).
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

    constexpr double p_low  = 0.02425;
    constexpr double p_high = 1.0 - p_low;

    if (p < p_low) {
        const double q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
             / ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0);
    }
    if (p <= p_high) {
        const double q = p - 0.5;
        const double r = q * q;
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
             / (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0);
    }
    const double q = std::sqrt(-2.0 * std::log(1.0 - p));
    return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
          / ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0);
}


// =====================================================================
// Public validators
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
// Euler-Maruyama scheme (Block 1.2.1)
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

    std::vector<std::vector<double>> paths(
        n_paths, std::vector<double>(n_steps + 1));

    for (std::size_t i = 0; i < n_paths; ++i) {
        paths[i][0] = S0;
        for (std::size_t k = 0; k < n_steps; ++k) {
            const double z  = standard_normal(rng);
            const double dW = sqrt_h * z;
            paths[i][k + 1] = paths[i][k] * (1.0 + rh + sigma * dW);
        }
    }
    return paths;
}


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
// Milstein scheme (Block 1.2.2)
// =====================================================================

std::vector<std::vector<double>>
simulate_path_milstein(double S0, double r, double sigma, double T,
                       std::size_t n_steps, std::size_t n_paths,
                       std::mt19937_64& rng) {
    validate_model_params(S0, sigma, T);
    validate_n_steps(n_steps);
    validate_n_paths(n_paths);

    const double h        = T / static_cast<double>(n_steps);
    const double sqrt_h   = std::sqrt(h);
    const double rh       = r * h;
    const double half_s2  = 0.5 * sigma * sigma;

    std::vector<std::vector<double>> paths(
        n_paths, std::vector<double>(n_steps + 1));

    for (std::size_t i = 0; i < n_paths; ++i) {
        paths[i][0] = S0;
        for (std::size_t k = 0; k < n_steps; ++k) {
            const double z  = standard_normal(rng);
            const double dW = sqrt_h * z;
            // factor = 1 + r*h + sigma*dW + 0.5*sigma^2*(dW^2 - h)
            const double factor = 1.0 + rh + sigma * dW
                                + half_s2 * (dW * dW - h);
            paths[i][k + 1] = paths[i][k] * factor;
        }
    }
    return paths;
}


std::vector<double>
simulate_terminal_milstein(double S0, double r, double sigma, double T,
                           std::size_t n_steps, std::size_t n_paths,
                           std::mt19937_64& rng) {
    validate_model_params(S0, sigma, T);
    validate_n_steps(n_steps);
    validate_n_paths(n_paths);

    const double h        = T / static_cast<double>(n_steps);
    const double sqrt_h   = std::sqrt(h);
    const double rh       = r * h;
    const double half_s2  = 0.5 * sigma * sigma;

    std::vector<double> S(n_paths, S0);
    for (std::size_t i = 0; i < n_paths; ++i) {
        for (std::size_t k = 0; k < n_steps; ++k) {
            const double z  = standard_normal(rng);
            const double dW = sqrt_h * z;
            const double factor = 1.0 + rh + sigma * dW
                                + half_s2 * (dW * dW - h);
            S[i] *= factor;
        }
    }
    return S;
}

}  // namespace quant
