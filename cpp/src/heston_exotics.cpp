// heston_exotics.cpp
//
// See heston_exotics.hpp for design notes and
// theory/phase4/block6_heston_calibration_exotics.tex for the mathematical
// derivations.

#include "heston_exotics.hpp"
#include "gbm.hpp"             // standard_normal, validate_*
#include "monte_carlo.hpp"     // MCResult, mc_estimator

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace quant::heston {

namespace {

// =====================================================================
// QE step (reproduced from heston_qe.cpp for self-containment)
// =====================================================================
// One step of the Andersen QE scheme, applied in place to (log_S, v).
// Identical to the qe_step in heston_qe.cpp; reproduced here so this
// module is robust to refactoring of heston_qe internals.

struct LogSCoefficients { double K0, K1, K2, K3, K4; };

LogSCoefficients
compute_logS_coefficients(double r, const HestonParams& p,
                           double dt, double gamma1, double gamma2) {
    LogSCoefficients K{};
    K.K0 = (r - p.rho * p.kappa * p.theta / p.sigma) * dt;
    K.K1 = gamma1 * dt * (p.kappa * p.rho / p.sigma - 0.5) - p.rho / p.sigma;
    K.K2 = gamma2 * dt * (p.kappa * p.rho / p.sigma - 0.5) + p.rho / p.sigma;
    K.K3 = gamma1 * dt * (1.0 - p.rho * p.rho);
    K.K4 = gamma2 * dt * (1.0 - p.rho * p.rho);
    return K;
}

inline void qe_step(double& log_S, double& v,
                     double U, double Z, double Z_indep,
                     const HestonParams& p, double dt, double psi_c,
                     double K0, double K1, double K2, double K3, double K4) {
    const double exp_neg = std::exp(-p.kappa * dt);
    const double one_minus_exp = 1.0 - exp_neg;
    const double m  = p.theta + (v - p.theta) * exp_neg;
    const double s2 = v * p.sigma * p.sigma * exp_neg * one_minus_exp / p.kappa
                    + p.theta * p.sigma * p.sigma * one_minus_exp * one_minus_exp
                      / (2.0 * p.kappa);
    const double psi = s2 / (m * m);

    double v_next;
    if (psi <= psi_c) {
        const double inv_psi = 1.0 / psi;
        const double b2 = 2.0 * inv_psi - 1.0
                          + std::sqrt(2.0 * inv_psi * (2.0 * inv_psi - 1.0));
        const double a = m / (1.0 + b2);
        const double b = std::sqrt(std::max(b2, 0.0));
        v_next = a * (b + Z) * (b + Z);
    } else {
        const double prob_zero = (psi - 1.0) / (psi + 1.0);
        const double beta      = 2.0 / (m * (psi + 1.0));
        if (U <= prob_zero) {
            v_next = 0.0;
        } else {
            v_next = -std::log((1.0 - U) / (1.0 - prob_zero)) / beta;
        }
    }

    const double var_term = std::max(K3 * v + K4 * v_next, 0.0);
    log_S += K0 + K1 * v + K2 * v_next + std::sqrt(var_term) * Z_indep;
    v = v_next;
}


// Common pre-validation
void check_inputs(double S0, double T,
                   const HestonParams& p,
                   std::size_t n_steps, std::size_t n_paths,
                   double psi_c, double gamma1, double gamma2) {
    p.validate();
    if (S0 <= 0.0) throw std::invalid_argument("S0 must be positive");
    if (T <= 0.0)  throw std::invalid_argument("T must be positive");
    quant::validate_n_steps(n_steps);
    quant::validate_n_paths(n_paths);
    if (!(psi_c > 1.0 && psi_c < 2.0))
        throw std::invalid_argument("psi_c must be in (1, 2)");
    if (gamma1 < 0.0 || gamma2 < 0.0)
        throw std::invalid_argument("gamma1, gamma2 must be non-negative");
    if (std::abs(gamma1 + gamma2 - 1.0) > 1e-12)
        throw std::invalid_argument("gamma1 + gamma2 must equal 1");
}

double sample_uniform(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng);
}

}  // anonymous namespace


// =====================================================================
// Asian call
// =====================================================================

MCResult
mc_asian_call_heston(double S0, double K, double r,
                       const HestonParams& p, double T,
                       std::size_t n_steps, std::size_t n_paths,
                       std::mt19937_64& rng,
                       std::size_t n_avg,
                       double psi_c, double gamma1, double gamma2,
                       double confidence_level) {
    check_inputs(S0, T, p, n_steps, n_paths, psi_c, gamma1, gamma2);
    quant::validate_strike(K);

    if (n_avg == 0) n_avg = n_steps;
    if (n_avg < 1) throw std::invalid_argument("n_avg must be at least 1");

    // Compute averaging step indices: { round(k * n_steps / n_avg) : k = 1..n_avg }
    std::unordered_set<std::size_t> sample_indices;
    sample_indices.reserve(n_avg);
    for (std::size_t k = 1; k <= n_avg; ++k) {
        const std::size_t idx = static_cast<std::size_t>(
            std::llround(static_cast<double>(k) * n_steps
                          / static_cast<double>(n_avg)));
        sample_indices.insert(std::min(idx, n_steps));
    }

    const double dt = T / static_cast<double>(n_steps);
    const double log_S0 = std::log(S0);
    const auto K_coef = compute_logS_coefficients(r, p, dt, gamma1, gamma2);

    std::vector<double> Y(n_paths);
    const double disc = std::exp(-r * T);

    for (std::size_t i = 0; i < n_paths; ++i) {
        double log_S = log_S0;
        double v     = p.v0;
        double sum   = 0.0;
        std::size_t count = 0;
        for (std::size_t k = 1; k <= n_steps; ++k) {
            const double U       = sample_uniform(rng);
            const double Z       = quant::standard_normal(rng);
            const double Z_indep = quant::standard_normal(rng);
            qe_step(log_S, v, U, Z, Z_indep, p, dt, psi_c,
                     K_coef.K0, K_coef.K1, K_coef.K2,
                     K_coef.K3, K_coef.K4);
            if (sample_indices.count(k)) {
                sum += std::exp(log_S);
                ++count;
            }
        }
        const double avg_S = sum / static_cast<double>(count);
        Y[i] = disc * std::max(avg_S - K, 0.0);
    }

    return quant::mc_estimator(Y, confidence_level);
}


// =====================================================================
// Lookback call (floating strike)
// =====================================================================

MCResult
mc_lookback_call_heston(double S0, double r,
                          const HestonParams& p, double T,
                          std::size_t n_steps, std::size_t n_paths,
                          std::mt19937_64& rng,
                          double psi_c, double gamma1, double gamma2,
                          double confidence_level) {
    check_inputs(S0, T, p, n_steps, n_paths, psi_c, gamma1, gamma2);

    const double dt = T / static_cast<double>(n_steps);
    const double log_S0 = std::log(S0);
    const auto K_coef = compute_logS_coefficients(r, p, dt, gamma1, gamma2);

    std::vector<double> Y(n_paths);
    const double disc = std::exp(-r * T);

    for (std::size_t i = 0; i < n_paths; ++i) {
        double log_S = log_S0;
        double v     = p.v0;
        double S_min = S0;  // includes S_0 in the minimum
        for (std::size_t k = 0; k < n_steps; ++k) {
            const double U       = sample_uniform(rng);
            const double Z       = quant::standard_normal(rng);
            const double Z_indep = quant::standard_normal(rng);
            qe_step(log_S, v, U, Z, Z_indep, p, dt, psi_c,
                     K_coef.K0, K_coef.K1, K_coef.K2,
                     K_coef.K3, K_coef.K4);
            const double S_current = std::exp(log_S);
            if (S_current < S_min) S_min = S_current;
        }
        const double S_T = std::exp(log_S);
        Y[i] = disc * (S_T - S_min);
    }

    return quant::mc_estimator(Y, confidence_level);
}


// =====================================================================
// Up-and-out barrier call
// =====================================================================

MCResult
mc_barrier_call_heston(double S0, double K, double H, double r,
                          const HestonParams& p, double T,
                          std::size_t n_steps, std::size_t n_paths,
                          std::mt19937_64& rng,
                          double psi_c, double gamma1, double gamma2,
                          double confidence_level) {
    check_inputs(S0, T, p, n_steps, n_paths, psi_c, gamma1, gamma2);
    quant::validate_strike(K);
    if (H <= S0)
        throw std::invalid_argument("H must be greater than S0");

    const double dt = T / static_cast<double>(n_steps);
    const double log_S0 = std::log(S0);
    const auto K_coef = compute_logS_coefficients(r, p, dt, gamma1, gamma2);

    std::vector<double> Y(n_paths);
    const double disc = std::exp(-r * T);

    for (std::size_t i = 0; i < n_paths; ++i) {
        double log_S = log_S0;
        double v     = p.v0;
        bool knocked_out = false;
        for (std::size_t k = 0; k < n_steps && !knocked_out; ++k) {
            const double U       = sample_uniform(rng);
            const double Z       = quant::standard_normal(rng);
            const double Z_indep = quant::standard_normal(rng);
            qe_step(log_S, v, U, Z, Z_indep, p, dt, psi_c,
                     K_coef.K0, K_coef.K1, K_coef.K2,
                     K_coef.K3, K_coef.K4);
            if (std::exp(log_S) >= H) knocked_out = true;
        }
        if (knocked_out) {
            Y[i] = 0.0;
        } else {
            const double S_T = std::exp(log_S);
            Y[i] = disc * std::max(S_T - K, 0.0);
        }
    }

    return quant::mc_estimator(Y, confidence_level);
}

}  // namespace quant::heston
