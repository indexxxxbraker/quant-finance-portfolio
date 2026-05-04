// monte_carlo.hpp
//
// Monte Carlo methods for European option pricing under Black-Scholes
// geometric Brownian motion.
//
// Public interface:
//
//   - struct MCResult: bundle of the standard outputs of any Monte
//     Carlo run (point estimate, asymptotic confidence half-width,
//     sample variance, sample size).
//
//   - mc_estimator: model- and payoff-agnostic statistical reducer
//     that turns a vector of i.i.d. payoffs into an MCResult.
//
//   - simulate_terminal_gbm: exact sampler of the terminal price S_T
//     under geometric Brownian motion, using the closed-form solution
//     of the SDE.
//
//   - mc_european_call_exact: high-level pricer for the European call
//     that orchestrates simulate_terminal_gbm + payoff + mc_estimator.
//
// References:
//   Phase 2 Block 0 writeup (Monte Carlo foundations) and Phase 2
//   Block 1.1 writeup (this algorithm). Glasserman, Monte Carlo
//   Methods in Financial Engineering, Chapters 1 and 3.

#pragma once

#include <cstddef>
#include <random>
#include <vector>

namespace quant {

// =====================================================================
// Public types
// =====================================================================

struct MCResult {
    double      estimate;          // Monte Carlo point estimate
    double      half_width;        // Asymptotic CI half-width
    double      sample_variance;   // Bessel-corrected sample variance
    std::size_t n_paths;           // Number of i.i.d. samples used
};


// =====================================================================
// Public functions
// =====================================================================

// Reduce a vector of i.i.d. payoff samples to a Monte Carlo result.
//
// Computes the sample mean, the sample variance with Bessel's
// correction, and the asymptotic Gaussian confidence interval
// half-width based on the central limit theorem and Slutsky's lemma.
//
// Throws std::invalid_argument if Y has fewer than 2 elements or if
// confidence_level is not in (0, 1).
MCResult
mc_estimator(const std::vector<double>& Y,
             double confidence_level = 0.95);


// Simulate n_paths samples of S_T under geometric Brownian motion,
// exactly: uses S_T = S0 * exp((r - sigma^2/2) * T + sigma*sqrt(T)*Z),
// with Z ~ N(0, 1) sampled by inversion. No time-discretisation error.
//
// rng must outlive the call. The function consumes n_paths uniforms
// from rng; on return rng has advanced by that amount.
//
// Throws std::invalid_argument if S0, sigma or T is non-positive, or
// if n_paths < 2.
std::vector<double>
simulate_terminal_gbm(double S0, double r, double sigma, double T,
                      std::size_t n_paths,
                      std::mt19937_64& rng);


// Price a European call by Monte Carlo with exact GBM simulation.
//
// Pipeline: sample n_paths of S_T using simulate_terminal_gbm,
// evaluate the discounted payoff exp(-r*T) * max(S_T - K, 0) on each
// path, and pass the resulting vector to mc_estimator.
//
// rng must outlive the call. The function consumes n_paths uniforms
// from rng.
//
// Throws std::invalid_argument on parameter validation failure.
MCResult
mc_european_call_exact(double S, double K, double r, double sigma, double T,
                       std::size_t n_paths,
                       std::mt19937_64& rng,
                       double confidence_level = 0.95);

}  // namespace quant
