// greeks.cpp -- implementation of greeks.hpp.

#include "greeks.hpp"

#include "gbm.hpp"   // standard_normal, validators

#include <algorithm>
#include <cmath>
#include <vector>

namespace quant {

namespace {

// Helper: discounted call payoff at parameters (S, sigma) for a single
// pre-drawn standard normal Z. Same r, K, T as the caller.
inline double payoff_at(double S, double K, double r, double sigma,
                        double T, double Z) {
    const double drift = (r - 0.5 * sigma * sigma) * T;
    const double diffusion = sigma * std::sqrt(T);
    const double S_T = S * std::exp(drift + diffusion * Z);
    return std::exp(-r * T) * std::max(S_T - K, 0.0);
}

}  // anonymous namespace


// =====================================================================
// Bumping (CRN central differences)
// =====================================================================

MCResult
delta_bump(double S, double K, double r, double sigma, double T,
           std::size_t n_paths,
           std::mt19937_64& rng,
           double h, double confidence_level) {
    validate_model_params(S, sigma, T);
    validate_strike(K);
    validate_n_paths(n_paths);
    validate_confidence_level(confidence_level);

    const double S_up = S * (1.0 + h);
    const double S_dn = S * (1.0 - h);
    const double bump_denom = 2.0 * S * h;

    std::vector<double> per_path(n_paths);
    for (std::size_t i = 0; i < n_paths; ++i) {
        const double Z = standard_normal(rng);
        const double f_up = payoff_at(S_up, K, r, sigma, T, Z);
        const double f_dn = payoff_at(S_dn, K, r, sigma, T, Z);
        per_path[i] = (f_up - f_dn) / bump_denom;
    }

    return mc_estimator(per_path, confidence_level);
}


MCResult
vega_bump(double S, double K, double r, double sigma, double T,
          std::size_t n_paths,
          std::mt19937_64& rng,
          double h, double confidence_level) {
    validate_model_params(S, sigma, T);
    validate_strike(K);
    validate_n_paths(n_paths);
    validate_confidence_level(confidence_level);

    const double sigma_up = sigma + h;
    const double sigma_dn = sigma - h;
    const double bump_denom = 2.0 * h;

    std::vector<double> per_path(n_paths);
    for (std::size_t i = 0; i < n_paths; ++i) {
        const double Z = standard_normal(rng);
        const double f_up = payoff_at(S, K, r, sigma_up, T, Z);
        const double f_dn = payoff_at(S, K, r, sigma_dn, T, Z);
        per_path[i] = (f_up - f_dn) / bump_denom;
    }

    return mc_estimator(per_path, confidence_level);
}


MCResult
gamma_bump(double S, double K, double r, double sigma, double T,
           std::size_t n_paths,
           std::mt19937_64& rng,
           double h, double confidence_level) {
    validate_model_params(S, sigma, T);
    validate_strike(K);
    validate_n_paths(n_paths);
    validate_confidence_level(confidence_level);

    const double S_up = S * (1.0 + h);
    const double S_dn = S * (1.0 - h);
    const double bump_denom = (S * h) * (S * h);

    std::vector<double> per_path(n_paths);
    for (std::size_t i = 0; i < n_paths; ++i) {
        const double Z = standard_normal(rng);
        const double f_up  = payoff_at(S_up, K, r, sigma, T, Z);
        const double f_mid = payoff_at(S,    K, r, sigma, T, Z);
        const double f_dn  = payoff_at(S_dn, K, r, sigma, T, Z);
        per_path[i] = (f_up - 2.0 * f_mid + f_dn) / bump_denom;
    }

    return mc_estimator(per_path, confidence_level);
}


// =====================================================================
// Pathwise sensitivities
// =====================================================================

MCResult
delta_pathwise(double S, double K, double r, double sigma, double T,
               std::size_t n_paths,
               std::mt19937_64& rng,
               double confidence_level) {
    validate_model_params(S, sigma, T);
    validate_strike(K);
    validate_n_paths(n_paths);
    validate_confidence_level(confidence_level);

    const double drift = (r - 0.5 * sigma * sigma) * T;
    const double diffusion = sigma * std::sqrt(T);
    const double discount = std::exp(-r * T);

    std::vector<double> per_path(n_paths);
    for (std::size_t i = 0; i < n_paths; ++i) {
        const double Z = standard_normal(rng);
        const double S_T = S * std::exp(drift + diffusion * Z);
        const double indicator = (S_T > K) ? 1.0 : 0.0;
        per_path[i] = discount * indicator * S_T / S;
    }

    return mc_estimator(per_path, confidence_level);
}


MCResult
vega_pathwise(double S, double K, double r, double sigma, double T,
              std::size_t n_paths,
              std::mt19937_64& rng,
              double confidence_level) {
    validate_model_params(S, sigma, T);
    validate_strike(K);
    validate_n_paths(n_paths);
    validate_confidence_level(confidence_level);

    const double drift = (r - 0.5 * sigma * sigma) * T;
    const double diffusion = sigma * std::sqrt(T);
    const double discount = std::exp(-r * T);
    const double sqrt_T = std::sqrt(T);

    std::vector<double> per_path(n_paths);
    for (std::size_t i = 0; i < n_paths; ++i) {
        const double Z = standard_normal(rng);
        const double S_T = S * std::exp(drift + diffusion * Z);
        const double indicator = (S_T > K) ? 1.0 : 0.0;
        per_path[i] = discount * indicator * S_T * (sqrt_T * Z - sigma * T);
    }

    return mc_estimator(per_path, confidence_level);
}


// =====================================================================
// Likelihood ratio
// =====================================================================

MCResult
delta_lr(double S, double K, double r, double sigma, double T,
         std::size_t n_paths,
         std::mt19937_64& rng,
         double confidence_level) {
    validate_model_params(S, sigma, T);
    validate_strike(K);
    validate_n_paths(n_paths);
    validate_confidence_level(confidence_level);

    const double drift = (r - 0.5 * sigma * sigma) * T;
    const double diffusion = sigma * std::sqrt(T);
    const double discount = std::exp(-r * T);
    const double sqrt_T = std::sqrt(T);
    const double score_denom = S * sigma * sqrt_T;

    std::vector<double> per_path(n_paths);
    for (std::size_t i = 0; i < n_paths; ++i) {
        const double Z = standard_normal(rng);
        const double S_T = S * std::exp(drift + diffusion * Z);
        const double payoff = discount * std::max(S_T - K, 0.0);
        per_path[i] = payoff * Z / score_denom;
    }

    return mc_estimator(per_path, confidence_level);
}


MCResult
vega_lr(double S, double K, double r, double sigma, double T,
        std::size_t n_paths,
        std::mt19937_64& rng,
        double confidence_level) {
    validate_model_params(S, sigma, T);
    validate_strike(K);
    validate_n_paths(n_paths);
    validate_confidence_level(confidence_level);

    const double drift = (r - 0.5 * sigma * sigma) * T;
    const double diffusion = sigma * std::sqrt(T);
    const double discount = std::exp(-r * T);
    const double sqrt_T = std::sqrt(T);

    std::vector<double> per_path(n_paths);
    for (std::size_t i = 0; i < n_paths; ++i) {
        const double Z = standard_normal(rng);
        const double S_T = S * std::exp(drift + diffusion * Z);
        const double payoff = discount * std::max(S_T - K, 0.0);
        const double score = (Z * Z - 1.0) / sigma - sqrt_T * Z;
        per_path[i] = payoff * score;
    }

    return mc_estimator(per_path, confidence_level);
}

}  // namespace quant
