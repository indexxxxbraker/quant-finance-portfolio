// asian.cpp -- implementation of asian.hpp.

#include "asian.hpp"

#include "black_scholes.hpp"   // norm_cdf
#include "gbm.hpp"             // standard_normal, validators

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace quant {

namespace {

// Simulate one GBM path with n_steps observations on the equispaced
// grid t_k = k * T / N (k = 1, ..., N), and return the arithmetic and
// geometric averages of the N monitoring values. Note S_0 itself is
// NOT part of the average.
//
// We accumulate sum_k S_k for the arithmetic mean and sum_k log(S_k)
// for the geometric mean (the latter via exp(mean(log)) is the
// numerically stable form for long paths). One standard normal is
// drawn per step.
struct PathAverages {
    double arithmetic;
    double geometric;
};

inline PathAverages
simulate_one_path_averages(double S0, double drift_step,
                           double diffusion_step,
                           std::size_t n_steps,
                           std::mt19937_64& rng) {
    double S        = S0;
    double sum_S    = 0.0;
    double sum_logS = 0.0;
    for (std::size_t k = 0; k < n_steps; ++k) {
        const double Z = standard_normal(rng);
        S *= std::exp(drift_step + diffusion_step * Z);
        sum_S    += S;
        sum_logS += std::log(S);
    }
    const double inv_N = 1.0 / static_cast<double>(n_steps);
    return PathAverages{ sum_S * inv_N,
                         std::exp(sum_logS * inv_N) };
}


// Compute the OLS slope c_hat of Y on X, then build the adjusted
// sample Y_tilde_i = Y_i - c_hat * (X_i - EX) and reduce. Mirrors the
// helper in variance_reduction.cpp; kept private to this module to
// preserve module locality.
//
// Throws std::invalid_argument if X has zero sample variance.
MCResult apply_cv(const std::vector<double>& Y,
                  const std::vector<double>& X,
                  double EX,
                  double confidence_level) {
    const std::size_t n = Y.size();

    // Sample means.
    double sum_Y = 0.0, sum_X = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum_Y += Y[i];
        sum_X += X[i];
    }
    const double Y_bar = sum_Y / static_cast<double>(n);
    const double X_bar = sum_X / static_cast<double>(n);

    // Cov(Y, X) numerator and Var(X) denominator of the OLS slope.
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

    std::vector<double> Y_tilde(n);
    for (std::size_t i = 0; i < n; ++i) {
        Y_tilde[i] = Y[i] - c_hat * (X[i] - EX);
    }
    return mc_estimator(Y_tilde, confidence_level);
}

}  // anonymous namespace


// =====================================================================
// Closed form for the discretely-monitored geometric Asian call
// =====================================================================

double
geometric_asian_call_closed_form(double S, double K, double r,
                                 double sigma, double T,
                                 std::size_t n_steps) {
    validate_model_params(S, sigma, T);
    validate_strike(K);
    validate_n_steps(n_steps);

    const double N         = static_cast<double>(n_steps);
    const double fac       = (N + 1.0) * (2.0 * N + 1.0) / (6.0 * N * N);
    const double sigma_eff = sigma * std::sqrt(fac);
    const double r_eff     = ((N + 1.0) / (2.0 * N))
                              * (r - 0.5 * sigma * sigma)
                            + 0.5 * sigma_eff * sigma_eff;

    const double sqrt_T = std::sqrt(T);
    const double d1 = (std::log(S / K)
                        + (r_eff + 0.5 * sigma_eff * sigma_eff) * T)
                      / (sigma_eff * sqrt_T);
    const double d2 = d1 - sigma_eff * sqrt_T;

    return std::exp(-r * T) * (S * std::exp(r_eff * T) * norm_cdf(d1)
                                - K * norm_cdf(d2));
}


// =====================================================================
// IID pricers
// =====================================================================

MCResult
mc_asian_call_arithmetic_iid(double S, double K, double r,
                             double sigma, double T,
                             std::size_t n_paths,
                             std::size_t n_steps,
                             std::mt19937_64& rng,
                             double confidence_level) {
    validate_model_params(S, sigma, T);
    validate_strike(K);
    validate_n_paths(n_paths);
    validate_n_steps(n_steps);
    validate_confidence_level(confidence_level);

    const double h              = T / static_cast<double>(n_steps);
    const double drift_step     = (r - 0.5 * sigma * sigma) * h;
    const double diffusion_step = sigma * std::sqrt(h);
    const double discount       = std::exp(-r * T);

    std::vector<double> Pi_A(n_paths);
    for (std::size_t i = 0; i < n_paths; ++i) {
        const auto avg = simulate_one_path_averages(S, drift_step,
                                                     diffusion_step,
                                                     n_steps, rng);
        Pi_A[i] = discount * std::max(avg.arithmetic - K, 0.0);
    }
    return mc_estimator(Pi_A, confidence_level);
}


MCResult
mc_asian_call_geometric_iid(double S, double K, double r,
                            double sigma, double T,
                            std::size_t n_paths,
                            std::size_t n_steps,
                            std::mt19937_64& rng,
                            double confidence_level) {
    validate_model_params(S, sigma, T);
    validate_strike(K);
    validate_n_paths(n_paths);
    validate_n_steps(n_steps);
    validate_confidence_level(confidence_level);

    const double h              = T / static_cast<double>(n_steps);
    const double drift_step     = (r - 0.5 * sigma * sigma) * h;
    const double diffusion_step = sigma * std::sqrt(h);
    const double discount       = std::exp(-r * T);

    std::vector<double> Pi_G(n_paths);
    for (std::size_t i = 0; i < n_paths; ++i) {
        const auto avg = simulate_one_path_averages(S, drift_step,
                                                     diffusion_step,
                                                     n_steps, rng);
        Pi_G[i] = discount * std::max(avg.geometric - K, 0.0);
    }
    return mc_estimator(Pi_G, confidence_level);
}


// =====================================================================
// Arithmetic Asian with geometric control variate
// =====================================================================

MCResult
mc_asian_call_arithmetic_cv(double S, double K, double r,
                            double sigma, double T,
                            std::size_t n_paths,
                            std::size_t n_steps,
                            std::mt19937_64& rng,
                            double confidence_level) {
    validate_model_params(S, sigma, T);
    validate_strike(K);
    validate_n_paths(n_paths);
    validate_n_steps(n_steps);
    validate_confidence_level(confidence_level);

    const double h              = T / static_cast<double>(n_steps);
    const double drift_step     = (r - 0.5 * sigma * sigma) * h;
    const double diffusion_step = sigma * std::sqrt(h);
    const double discount       = std::exp(-r * T);

    std::vector<double> Pi_A(n_paths);
    std::vector<double> Pi_G(n_paths);
    for (std::size_t i = 0; i < n_paths; ++i) {
        const auto avg = simulate_one_path_averages(S, drift_step,
                                                     diffusion_step,
                                                     n_steps, rng);
        Pi_A[i] = discount * std::max(avg.arithmetic - K, 0.0);
        Pi_G[i] = discount * std::max(avg.geometric  - K, 0.0);
    }

    const double EX = geometric_asian_call_closed_form(S, K, r, sigma, T,
                                                        n_steps);
    return apply_cv(Pi_A, Pi_G, EX, confidence_level);
}

}  // namespace quant
