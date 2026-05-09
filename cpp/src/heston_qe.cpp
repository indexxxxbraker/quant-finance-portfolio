// heston_qe.cpp
//
// See heston_qe.hpp for design notes and theory/phase4/block4_heston_qe.tex
// for the mathematical derivation.

#include "heston_qe.hpp"
#include "gbm.hpp"             // standard_normal, validate_n_steps, validate_n_paths, validate_strike
#include "monte_carlo.hpp"     // MCResult, mc_estimator

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace quant::heston {

namespace {

// QE scheme tuning constants. Held in the anonymous namespace because
// Andersen's recommendation (psi_c = 1.5, gamma1 = gamma2 = 0.5) is
// passed in as defaults from the public interface; if a caller wants
// non-default values, they pass them explicitly.

// One step of the Andersen QE scheme, applied in place to (log_S, v).
// Both the variance update (Q or E regime, branched on psi) and the
// log-spot update (central discretisation) are encapsulated here.
//
// The K_i coefficients of the log-spot are passed in pre-computed,
// since they depend only on (r, p, dt, gamma1, gamma2) and not on the
// path state. Computing them once per simulation and not per step
// saves a noticeable fraction of the inner-loop cost.
//
// U is a Uniform(0,1) draw, Z is a standard normal (used only in the
// Q regime), Z_indep is a standard normal independent of the variance
// (drives the log-spot noise). Three stochastic inputs per step.
inline void qe_step(double& log_S,
                     double& v,
                     double U,
                     double Z,
                     double Z_indep,
                     const HestonParams& p,
                     double dt,
                     double psi_c,
                     double K0, double K1, double K2,
                     double K3, double K4) {
    // Conditional moments of v_{n+1} | v_n.
    const double exp_neg = std::exp(-p.kappa * dt);
    const double one_minus_exp = 1.0 - exp_neg;
    const double m  = p.theta + (v - p.theta) * exp_neg;
    const double s2 = v * p.sigma * p.sigma * exp_neg * one_minus_exp / p.kappa
                    + p.theta * p.sigma * p.sigma * one_minus_exp * one_minus_exp
                      / (2.0 * p.kappa);
    const double psi = s2 / (m * m);

    // Sample v_next via the Q or E regime depending on psi.
    double v_next;
    if (psi <= psi_c) {
        // Quadratic regime: v_next = a (b + Z)^2.
        const double inv_psi = 1.0 / psi;
        const double b2 = 2.0 * inv_psi - 1.0
                          + std::sqrt(2.0 * inv_psi * (2.0 * inv_psi - 1.0));
        const double a = m / (1.0 + b2);
        const double b = std::sqrt(std::max(b2, 0.0));
        v_next = a * (b + Z) * (b + Z);
    } else {
        // Exponential regime: mixture of atom at 0 and Exp(beta).
        const double prob_zero = (psi - 1.0) / (psi + 1.0);
        const double beta      = 2.0 / (m * (psi + 1.0));
        if (U <= prob_zero) {
            v_next = 0.0;
        } else {
            v_next = -std::log((1.0 - U) / (1.0 - prob_zero)) / beta;
        }
    }

    // Log-spot update (central discretisation).
    // var_term = K3 v_n + K4 v_{n+1}; clamped at zero to absorb
    // floating-point underflow when both are zero.
    const double var_term = std::max(K3 * v + K4 * v_next, 0.0);
    log_S += K0 + K1 * v + K2 * v_next + std::sqrt(var_term) * Z_indep;
    v      = v_next;
}


// Common pre-validation for the public API.
void check_inputs(double S0,
                   double T,
                   const HestonParams& p,
                   std::size_t n_steps,
                   std::size_t n_paths,
                   bool antithetic,
                   double psi_c,
                   double gamma1,
                   double gamma2) {
    p.validate();
    if (S0 <= 0.0) throw std::invalid_argument("S0 must be positive");
    if (T <= 0.0)  throw std::invalid_argument("T must be positive");
    quant::validate_n_steps(n_steps);
    quant::validate_n_paths(n_paths);
    if (antithetic && (n_paths % 2 != 0)) {
        throw std::invalid_argument(
            "antithetic = true requires n_paths to be even");
    }
    if (!(psi_c > 1.0 && psi_c < 2.0)) {
        throw std::invalid_argument(
            "psi_c must be in (1, 2); Andersen's recommendation is 1.5");
    }
    if (gamma1 < 0.0 || gamma2 < 0.0) {
        throw std::invalid_argument("gamma1, gamma2 must be non-negative");
    }
    if (std::abs(gamma1 + gamma2 - 1.0) > 1e-12) {
        throw std::invalid_argument(
            "gamma1 + gamma2 must equal 1");
    }
}


// Pre-compute the K_0, ..., K_4 coefficients of the log-spot update.
// These depend only on (r, params, dt, gamma1, gamma2) and not on the
// path state; computing once outside the inner loop avoids redundancy.
struct LogSCoefficients {
    double K0, K1, K2, K3, K4;
};

LogSCoefficients compute_logS_coefficients(double r,
                                              const HestonParams& p,
                                              double dt,
                                              double gamma1,
                                              double gamma2) {
    LogSCoefficients K{};
    K.K0 = (r - p.rho * p.kappa * p.theta / p.sigma) * dt;
    K.K1 = gamma1 * dt * (p.kappa * p.rho / p.sigma - 0.5) - p.rho / p.sigma;
    K.K2 = gamma2 * dt * (p.kappa * p.rho / p.sigma - 0.5) + p.rho / p.sigma;
    K.K3 = gamma1 * dt * (1.0 - p.rho * p.rho);
    K.K4 = gamma2 * dt * (1.0 - p.rho * p.rho);
    return K;
}


// Helper for sampling Uniform(0, 1) by inversion of a single uniform
// draw from rng. Mirrors quant::standard_normal in style.
//
// Note: the open-interval guard against U=0 / U=1 is not strictly
// needed in either regime. In Q regime, U is unused. In E regime, if
// U <= p we go to v=0 (no sampling issue); if U > p, then 1-U is in
// (0, 1-p) and log is well-defined. We pass U through unmodified.
double sample_uniform(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng);
}

}  // anonymous namespace


// =====================================================================
// Terminal simulator
// =====================================================================

std::vector<double>
simulate_terminal_heston_qe(double S0,
                              double r,
                              const HestonParams& p,
                              double T,
                              std::size_t n_steps,
                              std::size_t n_paths,
                              std::mt19937_64& rng,
                              bool antithetic,
                              double psi_c,
                              double gamma1,
                              double gamma2) {
    check_inputs(S0, T, p, n_steps, n_paths, antithetic,
                 psi_c, gamma1, gamma2);

    const double dt = T / static_cast<double>(n_steps);
    const double log_S0 = std::log(S0);
    const auto K = compute_logS_coefficients(r, p, dt, gamma1, gamma2);

    std::vector<double> S_T(n_paths);

    if (!antithetic) {
        // Path-outer iteration: each path runs to completion before
        // the next, maximising cache locality on the inner time loop.
        for (std::size_t i = 0; i < n_paths; ++i) {
            double log_S = log_S0;
            double v     = p.v0;
            for (std::size_t k = 0; k < n_steps; ++k) {
                const double U       = sample_uniform(rng);
                const double Z       = quant::standard_normal(rng);
                const double Z_indep = quant::standard_normal(rng);
                qe_step(log_S, v, U, Z, Z_indep, p, dt, psi_c,
                         K.K0, K.K1, K.K2, K.K3, K.K4);
            }
            S_T[i] = std::exp(log_S);
        }
    } else {
        // Antithetic: pair up (i, i + half), sharing the magnitude of
        // the stochastic inputs. The "+" path uses (U, Z, Z_indep);
        // the "-" path uses (1-U, -Z, -Z_indep). Both paths advance
        // step by step in lockstep.
        const std::size_t half = n_paths / 2;
        for (std::size_t i = 0; i < half; ++i) {
            double log_S_pos = log_S0;
            double v_pos     = p.v0;
            double log_S_neg = log_S0;
            double v_neg     = p.v0;
            for (std::size_t k = 0; k < n_steps; ++k) {
                const double U       = sample_uniform(rng);
                const double Z       = quant::standard_normal(rng);
                const double Z_indep = quant::standard_normal(rng);
                qe_step(log_S_pos, v_pos,
                         U, Z, Z_indep,
                         p, dt, psi_c,
                         K.K0, K.K1, K.K2, K.K3, K.K4);
                qe_step(log_S_neg, v_neg,
                         1.0 - U, -Z, -Z_indep,
                         p, dt, psi_c,
                         K.K0, K.K1, K.K2, K.K3, K.K4);
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
mc_european_call_heston_qe(double S0,
                              double K_strike,
                              double r,
                              const HestonParams& p,
                              double T,
                              std::size_t n_steps,
                              std::size_t n_paths,
                              std::mt19937_64& rng,
                              bool antithetic,
                              double psi_c,
                              double gamma1,
                              double gamma2,
                              double confidence_level) {
    quant::validate_strike(K_strike);

    const std::vector<double> S_T = simulate_terminal_heston_qe(
        S0, r, p, T, n_steps, n_paths, rng,
        antithetic, psi_c, gamma1, gamma2);

    const double disc = std::exp(-r * T);

    if (!antithetic) {
        std::vector<double> Y(n_paths);
        for (std::size_t i = 0; i < n_paths; ++i) {
            Y[i] = disc * std::max(S_T[i] - K_strike, 0.0);
        }
        return quant::mc_estimator(Y, confidence_level);
    } else {
        // Pair payoffs and average BEFORE passing to mc_estimator,
        // so the half-width correctly reflects the antithetic
        // variance reduction. Same pattern as Block 3.
        const std::size_t half = n_paths / 2;
        std::vector<double> Y(half);
        for (std::size_t i = 0; i < half; ++i) {
            const double Y_pos = disc * std::max(S_T[i] - K_strike, 0.0);
            const double Y_neg = disc * std::max(S_T[i + half] - K_strike, 0.0);
            Y[i] = 0.5 * (Y_pos + Y_neg);
        }
        return quant::mc_estimator(Y, confidence_level);
    }
}

}  // namespace quant::heston
