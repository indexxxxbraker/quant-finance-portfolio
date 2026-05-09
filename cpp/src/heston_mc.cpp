// heston_mc.cpp
//
// See heston_mc.hpp for design notes and theory/phase4/block3_heston_mc_basic.tex
// for the mathematical derivation.

#include "heston_mc.hpp"
#include "gbm.hpp"             // standard_normal, validate_strike, validate_n_steps, validate_n_paths
#include "monte_carlo.hpp"     // MCResult, mc_estimator

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace quant::heston {

namespace {

// One step of the full-truncation Euler update, applied in place to
// (log_S, v). Used by both simulate_heston_paths and
// simulate_terminal_heston so the recursion is defined in exactly one
// place. Returns nothing; the new values are written back into log_S
// and v.
inline void ft_step(double& log_S,
                     double& v,
                     double dW1,
                     double dW2,
                     const HestonParams& p,
                     double r,
                     double dt) {
    const double v_pos     = std::max(v, 0.0);
    const double sqrt_vpos = std::sqrt(v_pos);

    // Compute the new log_S using the current v_pos before overwriting v.
    log_S += (r - 0.5 * v_pos) * dt + sqrt_vpos * dW1;
    v     += p.kappa * (p.theta - v_pos) * dt + p.sigma * sqrt_vpos * dW2;
}


// Common pre-validation for both path and terminal simulators.
void check_inputs(double S0,
                   double T,
                   const HestonParams& p,
                   std::size_t n_steps,
                   std::size_t n_paths,
                   bool antithetic) {
    p.validate();
    if (S0 <= 0.0) throw std::invalid_argument("S0 must be positive");
    if (T <= 0.0)  throw std::invalid_argument("T must be positive");
    quant::validate_n_steps(n_steps);
    quant::validate_n_paths(n_paths);
    if (antithetic && (n_paths % 2 != 0)) {
        throw std::invalid_argument(
            "antithetic = true requires n_paths to be even");
    }
}

}  // anonymous namespace


// =====================================================================
// Path simulator
// =====================================================================

HestonPaths
simulate_heston_paths(double S0,
                       double r,
                       const HestonParams& p,
                       double T,
                       std::size_t n_steps,
                       std::size_t n_paths,
                       std::mt19937_64& rng,
                       bool antithetic) {
    check_inputs(S0, T, p, n_steps, n_paths, antithetic);

    const double dt = T / static_cast<double>(n_steps);
    const double sqrt_dt = std::sqrt(dt);
    const double sqrt_one_minus_rho2 = std::sqrt(1.0 - p.rho * p.rho);
    const double log_S0 = std::log(S0);

    HestonPaths out;
    out.log_S.assign(n_paths, std::vector<double>(n_steps + 1));
    out.v.assign(n_paths, std::vector<double>(n_steps + 1));

    if (!antithetic) {
        // Path-outer iteration: each path is processed entirely before
        // moving to the next, maximising cache locality on the inner
        // time loop.
        for (std::size_t i = 0; i < n_paths; ++i) {
            out.log_S[i][0] = log_S0;
            out.v[i][0]     = p.v0;
            for (std::size_t k = 0; k < n_steps; ++k) {
                const double Z1 = quant::standard_normal(rng);
                const double Z2 = quant::standard_normal(rng);
                const double dW1 = sqrt_dt * Z1;
                const double dW2 = sqrt_dt * (p.rho * Z1
                                              + sqrt_one_minus_rho2 * Z2);
                double log_S = out.log_S[i][k];
                double v     = out.v[i][k];
                ft_step(log_S, v, dW1, dW2, p, r, dt);
                out.log_S[i][k + 1] = log_S;
                out.v[i][k + 1]     = v;
            }
        }
    } else {
        // Antithetic: process pairs (i, i + half). For each pair, the
        // RNG is consumed once per step and the same magnitude of
        // (Z1, Z2) is used with opposite signs in the two paths. This
        // is the only way to achieve the correlation that lets the
        // antithetic-pair average have lower variance than two
        // independent paths.
        const std::size_t half = n_paths / 2;
        for (std::size_t i = 0; i < half; ++i) {
            const std::size_t j = i + half;
            out.log_S[i][0] = log_S0;
            out.v[i][0]     = p.v0;
            out.log_S[j][0] = log_S0;
            out.v[j][0]     = p.v0;
            for (std::size_t k = 0; k < n_steps; ++k) {
                const double Z1 = quant::standard_normal(rng);
                const double Z2 = quant::standard_normal(rng);
                const double dW1_pos = sqrt_dt * Z1;
                const double dW2_pos = sqrt_dt * (p.rho * Z1
                                                   + sqrt_one_minus_rho2 * Z2);
                // Antithetic: same Z1, Z2 with opposite sign.
                const double dW1_neg = -dW1_pos;
                const double dW2_neg = -dW2_pos;

                double log_S_pos = out.log_S[i][k];
                double v_pos_path = out.v[i][k];
                ft_step(log_S_pos, v_pos_path, dW1_pos, dW2_pos, p, r, dt);
                out.log_S[i][k + 1] = log_S_pos;
                out.v[i][k + 1]     = v_pos_path;

                double log_S_neg = out.log_S[j][k];
                double v_neg_path = out.v[j][k];
                ft_step(log_S_neg, v_neg_path, dW1_neg, dW2_neg, p, r, dt);
                out.log_S[j][k + 1] = log_S_neg;
                out.v[j][k + 1]     = v_neg_path;
            }
        }
    }

    return out;
}


// =====================================================================
// Terminal-only simulator
// =====================================================================

std::vector<double>
simulate_terminal_heston(double S0,
                          double r,
                          const HestonParams& p,
                          double T,
                          std::size_t n_steps,
                          std::size_t n_paths,
                          std::mt19937_64& rng,
                          bool antithetic) {
    check_inputs(S0, T, p, n_steps, n_paths, antithetic);

    const double dt = T / static_cast<double>(n_steps);
    const double sqrt_dt = std::sqrt(dt);
    const double sqrt_one_minus_rho2 = std::sqrt(1.0 - p.rho * p.rho);
    const double log_S0 = std::log(S0);

    std::vector<double> S_T(n_paths);

    if (!antithetic) {
        for (std::size_t i = 0; i < n_paths; ++i) {
            double log_S = log_S0;
            double v     = p.v0;
            for (std::size_t k = 0; k < n_steps; ++k) {
                const double Z1 = quant::standard_normal(rng);
                const double Z2 = quant::standard_normal(rng);
                const double dW1 = sqrt_dt * Z1;
                const double dW2 = sqrt_dt * (p.rho * Z1
                                              + sqrt_one_minus_rho2 * Z2);
                ft_step(log_S, v, dW1, dW2, p, r, dt);
            }
            S_T[i] = std::exp(log_S);
        }
    } else {
        const std::size_t half = n_paths / 2;
        for (std::size_t i = 0; i < half; ++i) {
            double log_S_pos = log_S0;
            double v_pos     = p.v0;
            double log_S_neg = log_S0;
            double v_neg     = p.v0;
            for (std::size_t k = 0; k < n_steps; ++k) {
                const double Z1 = quant::standard_normal(rng);
                const double Z2 = quant::standard_normal(rng);
                const double dW1 = sqrt_dt * Z1;
                const double dW2 = sqrt_dt * (p.rho * Z1
                                              + sqrt_one_minus_rho2 * Z2);
                ft_step(log_S_pos, v_pos,  dW1,  dW2, p, r, dt);
                ft_step(log_S_neg, v_neg, -dW1, -dW2, p, r, dt);
            }
            S_T[i]        = std::exp(log_S_pos);
            S_T[i + half] = std::exp(log_S_neg);
        }
    }

    return S_T;
}


// =====================================================================
// High-level pricer
// =====================================================================

MCResult
mc_european_call_heston(double S0,
                          double K,
                          double r,
                          const HestonParams& p,
                          double T,
                          std::size_t n_steps,
                          std::size_t n_paths,
                          std::mt19937_64& rng,
                          bool antithetic,
                          double confidence_level) {
    quant::validate_strike(K);
    // Other inputs validated downstream by simulate_terminal_heston.

    const std::vector<double> S_T = simulate_terminal_heston(
        S0, r, p, T, n_steps, n_paths, rng, antithetic);

    const double disc = std::exp(-r * T);

    if (!antithetic) {
        std::vector<double> Y(n_paths);
        for (std::size_t i = 0; i < n_paths; ++i) {
            Y[i] = disc * std::max(S_T[i] - K, 0.0);
        }
        return quant::mc_estimator(Y, confidence_level);
    } else {
        // Pair up (+) and (-) paths and average BEFORE passing to
        // mc_estimator, so the estimator sees genuinely i.i.d. samples
        // and the half-width correctly reflects the antithetic variance
        // reduction. See block2_heston_fourier.tex's discussion of
        // similar patterns; the same logic applies on the Python side.
        const std::size_t half = n_paths / 2;
        std::vector<double> Y(half);
        for (std::size_t i = 0; i < half; ++i) {
            const double Y_pos = disc * std::max(S_T[i]        - K, 0.0);
            const double Y_neg = disc * std::max(S_T[i + half] - K, 0.0);
            Y[i] = 0.5 * (Y_pos + Y_neg);
        }
        return quant::mc_estimator(Y, confidence_level);
    }
}

}  // namespace quant::heston
