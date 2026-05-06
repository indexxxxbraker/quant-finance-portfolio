// variance_reduction.cpp -- implementation of variance_reduction.hpp.

#include "variance_reduction.hpp"

#include "black_scholes.hpp"   // norm_cdf for the AON expectation
#include "gbm.hpp"             // standard_normal, validators

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace quant {

// =====================================================================
// Antithetic variates (Block 2.1)
// =====================================================================

MCResult
mc_european_call_exact_av(double S, double K, double r, double sigma, double T,
                          std::size_t n_paths,
                          std::mt19937_64& rng,
                          double confidence_level) {
    validate_model_params(S, sigma, T);
    validate_strike(K);
    validate_n_paths(n_paths);
    validate_confidence_level(confidence_level);

    const double drift     = (r - 0.5 * sigma * sigma) * T;
    const double diffusion = sigma * std::sqrt(T);
    const double discount  = std::exp(-r * T);

    // Build n_paths paired payoffs. Each iteration consumes one
    // standard normal and produces one Y_i^AV.
    std::vector<double> Y(n_paths);
    for (std::size_t i = 0; i < n_paths; ++i) {
        const double z = standard_normal(rng);
        const double S_T_plus  = S * std::exp(drift + diffusion * z);
        const double S_T_minus = S * std::exp(drift - diffusion * z);
        const double payoff_plus  = std::max(S_T_plus  - K, 0.0);
        const double payoff_minus = std::max(S_T_minus - K, 0.0);
        Y[i] = 0.5 * discount * (payoff_plus + payoff_minus);
    }

    return mc_estimator(Y, confidence_level);
}


// =====================================================================
// Control variates: shared helper (Block 2.2)
// =====================================================================

namespace {

// Compute the OLS slope c_hat of Y on X, then build the adjusted
// sample Y_tilde_i = Y_i - c_hat * (X_i - EX) and reduce.
//
// Both vectors must have the same length, at least 2 elements.
// Throws std::invalid_argument if X has zero sample variance.
MCResult apply_cv(const std::vector<double>& Y,
                  const std::vector<double>& X,
                  double EX,
                  double confidence_level) {
    const std::size_t n = Y.size();

    // First pass: sample means.
    double sum_Y = 0.0, sum_X = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum_Y += Y[i];
        sum_X += X[i];
    }
    const double Y_bar = sum_Y / static_cast<double>(n);
    const double X_bar = sum_X / static_cast<double>(n);

    // Second pass: numerator (Cov) and denominator (Var) of OLS slope.
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double y_dev = Y[i] - Y_bar;
        const double x_dev = X[i] - X_bar;
        num += y_dev * x_dev;
        den += x_dev * x_dev;
    }

    if (den == 0.0) {
        throw std::invalid_argument(
            "Control variate has zero sample variance; "
            "cannot estimate slope.");
    }

    const double c_hat = num / den;

    // Third pass: build the adjusted sample.
    std::vector<double> Y_tilde(n);
    for (std::size_t i = 0; i < n; ++i) {
        Y_tilde[i] = Y[i] - c_hat * (X[i] - EX);
    }

    return mc_estimator(Y_tilde, confidence_level);
}

}  // anonymous namespace


// =====================================================================
// Control variates: discounted underlying (Block 2.2)
// =====================================================================

MCResult
mc_european_call_exact_cv_underlying(double S, double K, double r,
                                     double sigma, double T,
                                     std::size_t n_paths,
                                     std::mt19937_64& rng,
                                     double confidence_level) {
    validate_model_params(S, sigma, T);
    validate_strike(K);
    validate_n_paths(n_paths);
    validate_confidence_level(confidence_level);

    const double drift     = (r - 0.5 * sigma * sigma) * T;
    const double diffusion = sigma * std::sqrt(T);
    const double discount  = std::exp(-r * T);

    std::vector<double> Y(n_paths);
    std::vector<double> X(n_paths);

    for (std::size_t i = 0; i < n_paths; ++i) {
        const double z = standard_normal(rng);
        const double S_T = S * std::exp(drift + diffusion * z);
        Y[i] = discount * std::max(S_T - K, 0.0);
        X[i] = discount * S_T;
    }

    // E[discount * S_T] = S under the risk-neutral martingale property.
    const double EX = S;

    return apply_cv(Y, X, EX, confidence_level);
}


// =====================================================================
// Control variates: asset-or-nothing payoff (Block 2.2)
// =====================================================================

MCResult
mc_european_call_exact_cv_aon(double S, double K, double r,
                              double sigma, double T,
                              std::size_t n_paths,
                              std::mt19937_64& rng,
                              double confidence_level) {
    validate_model_params(S, sigma, T);
    validate_strike(K);
    validate_n_paths(n_paths);
    validate_confidence_level(confidence_level);

    const double drift     = (r - 0.5 * sigma * sigma) * T;
    const double diffusion = sigma * std::sqrt(T);
    const double discount  = std::exp(-r * T);

    std::vector<double> Y(n_paths);
    std::vector<double> X(n_paths);

    for (std::size_t i = 0; i < n_paths; ++i) {
        const double z = standard_normal(rng);
        const double S_T = S * std::exp(drift + diffusion * z);
        Y[i] = discount * std::max(S_T - K, 0.0);
        X[i] = (S_T > K) ? discount * S_T : 0.0;
    }

    // E[discount * S_T * 1_{S_T > K}] = S * Phi(d_1).
    const double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T)
                    / (sigma * std::sqrt(T));
    const double EX = S * norm_cdf(d1);

    return apply_cv(Y, X, EX, confidence_level);
}

}  // namespace quant
