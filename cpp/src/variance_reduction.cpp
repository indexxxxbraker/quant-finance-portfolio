// variance_reduction.cpp -- implementation of variance_reduction.hpp.

#include "variance_reduction.hpp"

#include "gbm.hpp"   // standard_normal, validators

#include <algorithm>
#include <cmath>
#include <vector>

namespace quant {

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

}  // namespace quant
