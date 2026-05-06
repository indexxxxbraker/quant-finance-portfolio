// monte_carlo.cpp -- implementation of monte_carlo.hpp.

#include "monte_carlo.hpp"

#include "gbm.hpp"  // samplers and validators

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace quant {

// =====================================================================
// File-local helpers
// =====================================================================

namespace {

// Quantile z = Phi^{-1}((1 + confidence_level) / 2).
// Uses the same Acklam approximation as the samplers, exposed via
// gbm.hpp, so that all standard normal arithmetic in the project
// goes through a single implementation. Crucial for byte-exact
// reproducibility of mc_estimator output between Block 1.1 and the
// post-refactor versions.
double standard_normal_quantile(double confidence_level) {
    return inverse_normal_cdf(0.5 * (1.0 + confidence_level));
}

}  // anonymous namespace


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

    // Sample variance (Bessel's correction). Two-pass formula for
    // numerical stability: avoids the catastrophic cancellation that
    // afflicts the naive sum-of-squares-minus-square-of-sums formula.
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


// =====================================================================
// mc_european_call_milstein  (Block 1.2.2)
// =====================================================================

MCResult
mc_european_call_milstein(double S, double K, double r, double sigma, double T,
                          std::size_t n_steps, std::size_t n_paths,
                          std::mt19937_64& rng,
                          double confidence_level) {
    validate_model_params(S, sigma, T);
    validate_strike(K);
    validate_n_steps(n_steps);
    validate_n_paths(n_paths);
    validate_confidence_level(confidence_level);

    const std::vector<double> S_T =
        simulate_terminal_milstein(S, r, sigma, T, n_steps, n_paths, rng);

    const double discount = std::exp(-r * T);
    std::vector<double> Y(n_paths);
    for (std::size_t i = 0; i < n_paths; ++i) {
        Y[i] = discount * std::max(S_T[i] - K, 0.0);
    }

    return mc_estimator(Y, confidence_level);
}

}  // namespace quant
