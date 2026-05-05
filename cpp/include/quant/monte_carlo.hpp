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
//     using the exact sampler.
//
//   - simulate_path_euler: Euler-Maruyama path sampler returning the
//     full discretised path (Block 1.2.1).
//
//   - simulate_terminal_euler: Euler-Maruyama sampler returning only
//     the terminal value, with O(M) memory (Block 1.2.1).
//
//   - mc_european_call_euler: high-level pricer for the European call
//     using the Euler-Maruyama discretisation (Block 1.2.1).
//
// References:
//   Phase 2 Block 0 writeup (foundations), Block 1.1 writeup (exact
//   sampler), Block 1.2.0 writeup (SDE discretisation theory), and
//   Block 1.2.1 writeup (Euler for European call). Glasserman, Monte
//   Carlo Methods in Financial Engineering, Chapters 1, 3, and 6.

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
// Generic statistical estimator
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


// =====================================================================
// Exact GBM sampler (Block 1.1)
// =====================================================================

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


// =====================================================================
// Euler-Maruyama scheme (Block 1.2.1)
// =====================================================================

// Simulate n_paths Euler-Maruyama paths of GBM with n_steps each.
// Returns a (n_paths x (n_steps + 1)) matrix as a vector of vectors:
// paths[i][k] is the value of path i at time k * h, where h =
// T / n_steps. The first column is S0, the last column is the
// Euler-discretised terminal value.
//
// The recursion implemented is
//   S_{n+1} = S_n * (1 + r * h + sigma * dW_n),
// with dW_n ~ N(0, h) drawn independently per (path, step).
//
// Memory: O(n_paths * n_steps). Use simulate_terminal_euler when the
// full path is not needed.
//
// rng must outlive the call. Throws std::invalid_argument on
// parameter validation failure.
//
// References:
//   Phase 2 Block 1.2.0 writeup (theory), Block 1.2.1 writeup.
std::vector<std::vector<double>>
simulate_path_euler(double S0, double r, double sigma, double T,
                    std::size_t n_steps, std::size_t n_paths,
                    std::mt19937_64& rng);


// Simulate n_paths Euler-Maruyama terminal values, with O(n_paths)
// memory. Equivalent to taking the last column of simulate_path_euler
// but advances state in place rather than storing the full path.
//
// Same parameters and semantics as simulate_path_euler. Returns a
// vector of length n_paths with the Euler-discretised terminal
// values.
std::vector<double>
simulate_terminal_euler(double S0, double r, double sigma, double T,
                        std::size_t n_steps, std::size_t n_paths,
                        std::mt19937_64& rng);


// Price a European call by Monte Carlo with Euler-Maruyama paths.
//
// Pipeline: sample n_paths Euler-discretised terminal values via
// simulate_terminal_euler, evaluate the discounted payoff on each,
// reduce via mc_estimator.
//
// Carries a discretisation bias of order T / n_steps: the estimator
// converges to the BS price at weak rate 1 as n_steps grows.
//
// Throws std::invalid_argument on parameter validation failure.
MCResult
mc_european_call_euler(double S, double K, double r, double sigma, double T,
                       std::size_t n_steps, std::size_t n_paths,
                       std::mt19937_64& rng,
                       double confidence_level = 0.95);

}  // namespace quant
