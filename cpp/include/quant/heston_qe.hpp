// heston_qe.hpp
//
// Heston model: Monte Carlo simulation via Andersen's Quadratic-Exponential
// (QE) scheme.
//
// Mirrors quantlib.heston_qe on the Python side. Reuses HestonParams from
// heston_fourier.hpp and MCResult from monte_carlo.hpp, just like the
// full-truncation Euler module of Block 3.
//
// The QE scheme matches the first two conditional moments of v_{t+dt}|v_t
// exactly, with two regimes (quadratic for psi <= psi_c, exponential for
// psi > psi_c). The log-spot uses Andersen's central discretisation, which
// uses both v_n and v_{n+1} for higher-order accuracy than the simple
// Euler of Block 3. See theory/phase4/block4_heston_qe.tex for the full
// derivation.

#pragma once

#include "heston_fourier.hpp"   // HestonParams
#include "monte_carlo.hpp"      // MCResult

#include <cstddef>
#include <random>
#include <vector>

namespace quant::heston {

// =====================================================================
// Public interface
// =====================================================================

// Simulate only the terminal spot S_T = exp(log_S_N) under the Heston
// model via Andersen's QE scheme, with O(n_paths) memory.
//
// The variance update samples from a moment-matching distribution
// (quadratic regime for low psi, exponential mixture with atom at zero
// for high psi). The log-spot uses Andersen's central discretisation
// with weights gamma1, gamma2 on (v_n, v_{n+1}) respectively;
// gamma1 = gamma2 = 0.5 gives the symmetric trapezoidal scheme and is
// the recommended default.
//
// When antithetic = true, paths are generated in pairs (i, i + n_paths/2)
// sharing the magnitude of all stochastic inputs (uniform U for the
// E regime, normals for the Q regime and the log-spot). The pair uses
// (U, Z, Z_indep) for the "+" path and (1-U, -Z, -Z_indep) for the "-"
// path. n_paths must then be even.
//
// Throws std::invalid_argument on parameter validation failure.
std::vector<double>
simulate_terminal_heston_qe(double S0,
                              double r,
                              const HestonParams& p,
                              double T,
                              std::size_t n_steps,
                              std::size_t n_paths,
                              std::mt19937_64& rng,
                              bool antithetic = false,
                              double psi_c = 1.5,
                              double gamma1 = 0.5,
                              double gamma2 = 0.5);


// Price a European call under Heston by Andersen QE Monte Carlo.
//
// Pipeline: simulate n_paths terminal values via simulate_terminal_heston_qe,
// evaluate the discounted payoff exp(-r*T) * max(S_T - K, 0) on each
// path, reduce to MCResult via mc_estimator. With antithetic = true,
// pairs are averaged before the estimator sees them, so that the
// half-width correctly reflects the variance reduction.
//
// QE typically achieves the same precision as full-truncation Euler
// (Block 3) at 4-10x fewer time steps, with the gap widening for low
// Feller parameter. For benign parameters (Feller >> 1) the gap is
// modest; for aggressive parameters (Feller << 1) it can exceed 30x.
//
// Throws std::invalid_argument on parameter validation failure.
MCResult
mc_european_call_heston_qe(double S0,
                              double K,
                              double r,
                              const HestonParams& p,
                              double T,
                              std::size_t n_steps,
                              std::size_t n_paths,
                              std::mt19937_64& rng,
                              bool antithetic = false,
                              double psi_c = 1.5,
                              double gamma1 = 0.5,
                              double gamma2 = 0.5,
                              double confidence_level = 0.95);

}  // namespace quant::heston
