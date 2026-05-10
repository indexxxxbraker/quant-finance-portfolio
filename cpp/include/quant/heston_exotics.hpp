// heston_exotics.hpp
//
// Heston model: exotic option pricing via Andersen QE Monte Carlo.
//
// Three path-dependent European-style pricers:
//   - mc_asian_call_heston       (arithmetic-average Asian call)
//   - mc_lookback_call_heston    (floating-strike lookback call)
//   - mc_barrier_call_heston     (up-and-out barrier call)
//
// All three share the QE simulation infrastructure of Block 4 and
// differ only in the path-functional accumulated during the time
// loop. See theory/phase4/block6_heston_calibration_exotics.tex for
// the mathematical statement of each payoff and discussion of
// discrete-monitoring biases for lookback and barrier.

#pragma once

#include "heston_fourier.hpp"   // HestonParams
#include "monte_carlo.hpp"      // MCResult

#include <cstddef>
#include <random>

namespace quant::heston {

// =====================================================================
// Asian call (arithmetic average)
// =====================================================================
//
// Payoff: max( mean(S_{t_1}, ..., S_{t_n_avg}) - K, 0 ),
// where averaging dates are uniformly placed across [0, T] at simulation
// step indices { round(k * n_steps / n_avg) for k = 1, ..., n_avg }.
// For n_avg = n_steps, averaging at every step (default).
// For n_avg = 1, single sample at t = T (recovers European call).
//
// Throws std::invalid_argument on parameter validation failure.
MCResult
mc_asian_call_heston(double S0,
                       double K,
                       double r,
                       const HestonParams& p,
                       double T,
                       std::size_t n_steps,
                       std::size_t n_paths,
                       std::mt19937_64& rng,
                       std::size_t n_avg = 0,    // 0 means "n_steps"
                       double psi_c = 1.5,
                       double gamma1 = 0.5,
                       double gamma2 = 0.5,
                       double confidence_level = 0.95);


// =====================================================================
// Lookback call (floating strike)
// =====================================================================
//
// Payoff: S_T - min(S_{t_0}, S_{t_1}, ..., S_{t_n_steps}).
// Discrete-monitoring bias: positive (true continuous min < discrete
// min), so the discrete estimator overestimates the lookback payoff.
// The bias scales as O(sqrt(dt)).
MCResult
mc_lookback_call_heston(double S0,
                          double r,
                          const HestonParams& p,
                          double T,
                          std::size_t n_steps,
                          std::size_t n_paths,
                          std::mt19937_64& rng,
                          double psi_c = 1.5,
                          double gamma1 = 0.5,
                          double gamma2 = 0.5,
                          double confidence_level = 0.95);


// =====================================================================
// Up-and-out barrier call
// =====================================================================
//
// Payoff: max(S_T - K, 0) * I( max(S_{t_0}, ..., S_{t_n_steps}) < H ).
// Requires H > S0 (otherwise immediately knocked out).
// Discrete-monitoring bias: negative (true continuous max > discrete
// max), so paths that "should" knock out are sometimes counted as
// surviving, leading to overestimated barrier prices.
MCResult
mc_barrier_call_heston(double S0,
                         double K,
                         double H,
                         double r,
                         const HestonParams& p,
                         double T,
                         std::size_t n_steps,
                         std::size_t n_paths,
                         std::mt19937_64& rng,
                         double psi_c = 1.5,
                         double gamma1 = 0.5,
                         double gamma2 = 0.5,
                         double confidence_level = 0.95);

}  // namespace quant::heston
