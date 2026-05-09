// heston_mc.hpp
//
// Heston model: Monte Carlo simulation via full-truncation Euler.
//
// Mirrors the design of quantlib.heston_mc on the Python side. The
// scheme of Lord, Koekkoek, van Dijk (2010) discretises the Heston
// system with v_n^+ := max(v_n, 0) used in both drift and diffusion
// coefficients while v_n itself is propagated unconstrained, sidestepping
// the non-Lipschitz singularity at v = 0 with the smallest weak bias
// among the standard fix-up schemes.
//
// See theory/phase4/block3_heston_mc_basic.tex for the derivation,
// convergence theory, and validation strategy.

#pragma once

#include "heston_fourier.hpp"   // HestonParams
#include "monte_carlo.hpp"      // MCResult

#include <cstddef>
#include <random>
#include <vector>

namespace quant::heston {

// =====================================================================
// Public types
// =====================================================================

// Bundle of Heston (log_S, v) trajectories, with paths in the outer
// dimension and time steps in the inner dimension. log_S[i][k] is
// log S at time t_k = k * (T / n_steps), path i; v[i][k] is the
// (unconstrained) variance at the same point. Note that v[i][k] may be
// negative; the truncated value v_n^+ = max(v[i][k], 0.0) can be
// recovered on the fly when needed.
struct HestonPaths {
    std::vector<std::vector<double>> log_S;  // shape (n_paths, n_steps + 1)
    std::vector<std::vector<double>> v;      // shape (n_paths, n_steps + 1)
};


// =====================================================================
// Path simulator: full-truncation Euler
// =====================================================================

// Simulate full paths of (log_S, v) under the Heston model via
// full-truncation Euler.
//
// One step of the scheme reads, with v_n^+ := max(v_n, 0):
//
//   dW1_n = sqrt(dt) Z1_n,   Z1_n ~ N(0, 1)
//   dW2_n = sqrt(dt) (rho Z1_n + sqrt(1 - rho^2) Z2_n),  Z2_n ~ N(0, 1)
//   v_{n+1}     = v_n + kappa (theta - v_n^+) dt
//                     + sigma sqrt(v_n^+) dW2_n
//   log S_{n+1} = log S_n + (r - 0.5 v_n^+) dt
//                     + sqrt(v_n^+) dW1_n
//
// The Cholesky construction uses two independent standard normals at
// each step to produce a pair of correlated Brownian increments,
// matching the convention on the Python side.
//
// When antithetic = true, paths are generated in pairs (i, i + n_paths/2)
// sharing the same magnitude of Z1, Z2 with opposite sign. n_paths
// must then be even.
//
// Memory: O(n_paths * n_steps). Use simulate_terminal_heston when only
// the terminal log-spot is required.
//
// rng must outlive the call. Throws std::invalid_argument on
// parameter or contract validation failure.
HestonPaths
simulate_heston_paths(double S0,
                       double r,
                       const HestonParams& p,
                       double T,
                       std::size_t n_steps,
                       std::size_t n_paths,
                       std::mt19937_64& rng,
                       bool antithetic = false);


// Simulate only the terminal spot S_T = exp(log_S_N) under the Heston
// model via full-truncation Euler, with O(n_paths) memory.
//
// Equivalent to taking { exp(log_S[i][n_steps]) for i in 0..n_paths }
// from simulate_heston_paths, but advances state in place rather than
// storing the full trajectories. Used by the European pricer below.
//
// rng must outlive the call.
std::vector<double>
simulate_terminal_heston(double S0,
                          double r,
                          const HestonParams& p,
                          double T,
                          std::size_t n_steps,
                          std::size_t n_paths,
                          std::mt19937_64& rng,
                          bool antithetic = false);


// =====================================================================
// High-level pricer: European call
// =====================================================================

// Price a European call under Heston by full-truncation Euler MC.
//
// Pipeline: simulate n_paths terminal values via simulate_terminal_heston,
// evaluate the discounted payoff exp(-r*T) * max(S_T - K, 0) on each
// path, reduce to MCResult via mc_estimator. With antithetic = true,
// the n_paths discounted payoffs are first paired and averaged into
// n_paths / 2 antithetic samples before being passed to mc_estimator;
// the resulting MCResult therefore has n_paths = n_paths / 2 and a
// half-width that correctly reflects the variance reduction.
//
// The estimate carries two sources of error: a discretisation bias
// O(dt = T / n_steps) and a statistical error O(n_paths^{-1/2}).
// Only the latter is reflected in the half-width.
//
// rng must outlive the call.
MCResult
mc_european_call_heston(double S0,
                          double K,
                          double r,
                          const HestonParams& p,
                          double T,
                          std::size_t n_steps,
                          std::size_t n_paths,
                          std::mt19937_64& rng,
                          bool antithetic = false,
                          double confidence_level = 0.95);

}  // namespace quant::heston
