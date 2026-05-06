// variance_reduction.hpp
//
// Variance-reduced Monte Carlo pricers for the European call.
//
// This module hosts pricers that improve over the basic IID
// estimators of monte_carlo.hpp by exploiting structural properties
// of the problem (monotonicity for AV, correlation with tractable
// controls for CV). Mirrors the Python module layout
// (quantlib.variance_reduction).
//
// Currently implemented:
//
//   - mc_european_call_exact_av: antithetic-variates pricer for the
//     European call under exact GBM (Block 2.1).
//
// To be added in Block 2.2:
//
//   - mc_european_call_exact_cv_underlying
//   - mc_european_call_exact_cv_aon
//
// References:
//   Phase 2 Block 2.0 writeup (foundations); Block 2.1 writeup
//   (this algorithm). Glasserman, Section 4.2.

#pragma once

#include "monte_carlo.hpp"   // for MCResult

#include <cstddef>
#include <random>

namespace quant {

// Price a European call by Monte Carlo with antithetic variates.
//
// Implements
//   Y_i^AV = 0.5 * (f(Z_i) + f(-Z_i))
// where f(z) = e^{-rT} * max(S * exp((r-0.5*sigma^2)*T + sigma*sqrt(T)*z) - K, 0)
// and Z_i ~ N(0, 1) i.i.d. Returns an MCResult based on the n_paths
// paired payoffs Y_i^AV.
//
// Note: total payoff evaluations is 2 * n_paths (one at +Z_i, one
// at -Z_i), but the i.i.d. unit for variance estimation is the pair,
// so n_paths is the relevant sample size for the central limit
// theorem.
//
// rng must outlive the call. Throws std::invalid_argument on input
// validation failure.
MCResult
mc_european_call_exact_av(double S, double K, double r, double sigma, double T,
                          std::size_t n_paths,
                          std::mt19937_64& rng,
                          double confidence_level = 0.95);

}  // namespace quant
