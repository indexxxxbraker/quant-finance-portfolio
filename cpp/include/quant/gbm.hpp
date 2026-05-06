// gbm.hpp
//
// GBM-specific samplers and shared validators.
//
// This header declares all samplers of the geometric Brownian motion
// state under the risk-neutral measure, plus the input validators
// shared between the samplers themselves and the higher-level pricers
// in monte_carlo.hpp. The split between gbm and monte_carlo mirrors
// the Python side: GBM lives here, the model-agnostic statistical
// reducer and the European pricers live in monte_carlo.hpp.
//
// Public samplers:
//
//   - simulate_terminal_gbm: exact sampler of S_T using the
//     closed-form solution of the GBM SDE (Block 1.1).
//
//   - simulate_path_euler / simulate_terminal_euler: Euler-Maruyama
//     scheme (Block 1.2.1).
//
//   - simulate_path_milstein / simulate_terminal_milstein: Milstein
//     scheme (Block 1.2.2).
//
// Public validators (used by monte_carlo.cpp pricers):
//
//   - validate_model_params, validate_strike, validate_n_paths,
//     validate_n_steps, validate_confidence_level.
//
// References:
//   Phase 2 Block 0 writeup (foundations); Block 1.1 writeup (exact);
//   Block 1.2.0 writeup (SDE discretisation theory); Block 1.2.1
//   writeup (Euler); Block 1.2.2 writeup (Milstein).

#pragma once

#include <cstddef>
#include <random>
#include <vector>

namespace quant {

// =====================================================================
// Public validators
// =====================================================================

// Validate (S0, sigma, T) > 0. r is unconstrained (negative rates OK).
// Throws std::invalid_argument on failure.
void validate_model_params(double S0, double sigma, double T);

// Validate K > 0.
void validate_strike(double K);

// Validate n_paths >= 2 (for sample variance with Bessel's correction).
void validate_n_paths(std::size_t n_paths);

// Validate n_steps >= 1.
void validate_n_steps(std::size_t n_steps);

// Validate confidence_level in (0, 1).
void validate_confidence_level(double confidence_level);


// =====================================================================
// Standard normal utilities (used by both samplers and pricers)
// =====================================================================

// Inverse standard normal CDF, Acklam's rational approximation (2003).
// Relative error below 1.15e-9 over (0, 1). Throws
// std::invalid_argument if p is not in (0, 1).
//
// Exposed publicly because monte_carlo.cpp's mc_estimator needs the
// standard normal quantile for the asymptotic CI half-width.
double inverse_normal_cdf(double p);


// Draw a standard normal from rng by inversion (one uniform draw,
// then inverse_normal_cdf).
//
// Exposed publicly because variance_reduction.cpp needs to generate
// standard normals from the same rng stream as the GBM samplers,
// preserving the inversion-based pipeline that ensures QMC
// compatibility (Phase 2 Block 3).
double standard_normal(std::mt19937_64& rng);


// =====================================================================
// Exact GBM sampler (Block 1.1)
// =====================================================================

// Simulate n_paths samples of S_T under geometric Brownian motion,
// exactly: uses S_T = S0 * exp((r - sigma^2/2) * T + sigma*sqrt(T)*Z),
// with Z ~ N(0, 1) sampled by inversion. No time-discretisation error.
//
// rng must outlive the call. Throws std::invalid_argument if S0,
// sigma or T is non-positive, or if n_paths < 2.
std::vector<double>
simulate_terminal_gbm(double S0, double r, double sigma, double T,
                      std::size_t n_paths,
                      std::mt19937_64& rng);


// =====================================================================
// Euler-Maruyama scheme (Block 1.2.1)
// =====================================================================

// Simulate n_paths Euler-Maruyama paths of GBM with n_steps each.
// Returns a (n_paths x (n_steps + 1)) matrix as a vector of vectors.
//
// Recursion: S_{n+1} = S_n * (1 + r*h + sigma*dW_n),
// with dW_n ~ N(0, h) and h = T / n_steps.
//
// Memory: O(n_paths * n_steps). Use simulate_terminal_euler when only
// the terminal value is needed.
std::vector<std::vector<double>>
simulate_path_euler(double S0, double r, double sigma, double T,
                    std::size_t n_steps, std::size_t n_paths,
                    std::mt19937_64& rng);


// Same as simulate_path_euler but returns only the terminal column.
// Memory: O(n_paths). Advances the state in place rather than
// storing the full path.
std::vector<double>
simulate_terminal_euler(double S0, double r, double sigma, double T,
                        std::size_t n_steps, std::size_t n_paths,
                        std::mt19937_64& rng);


// =====================================================================
// Milstein scheme (Block 1.2.2)
// =====================================================================

// Simulate n_paths Milstein paths of GBM with n_steps each.
// Returns a (n_paths x (n_steps + 1)) matrix as a vector of vectors.
//
// Recursion (multiplicative, quadratic in dW_n):
//   S_{n+1} = S_n * (1 + r*h + sigma*dW_n
//                    + 0.5 * sigma^2 * (dW_n^2 - h)).
// The corrector term has zero mean (E[dW_n^2 - h] = 0) but is
// pathwise nonzero. It lifts the strong order from 1/2 (Euler) to 1;
// the weak order remains 1.
//
// Memory: O(n_paths * n_steps). Use simulate_terminal_milstein when
// only the terminal value is needed.
//
// References: Phase 2 Block 1.2.2 writeup, Section 2.
std::vector<std::vector<double>>
simulate_path_milstein(double S0, double r, double sigma, double T,
                       std::size_t n_steps, std::size_t n_paths,
                       std::mt19937_64& rng);


// Same as simulate_path_milstein but returns only the terminal column.
// Memory: O(n_paths). Advances the state in place.
std::vector<double>
simulate_terminal_milstein(double S0, double r, double sigma, double T,
                           std::size_t n_steps, std::size_t n_paths,
                           std::mt19937_64& rng);

}  // namespace quant
