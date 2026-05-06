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
//   - mc_european_call_exact_cv_underlying: control-variate pricer
//     using the discounted underlying as the control (Block 2.2).
//
//   - mc_european_call_exact_cv_aon: control-variate pricer using
//     the discounted asset-or-nothing payoff as the control (Block
//     2.2).
//
// References:
//   Phase 2 Block 2.0 writeup (foundations); Block 2.1 writeup
//   (antithetic); Block 2.2 writeup (control variates).
//   Glasserman, Sections 4.1 and 4.2.

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


// Price a European call by Monte Carlo with the discounted underlying
// as the control variate (Block 2.2).
//
// Control: X_i = e^{-rT} S_T^{(i)}.
// Closed-form expectation: E[X] = S_0 (martingale property under Q).
//
// Adjusted estimator: Y_tilde_i = Y_i - c_hat * (X_i - S_0),
// where c_hat is the OLS slope of Y on X computed from the same paths.
//
// Empirical correlation rho_1 ~ 0.92 at the ATM point gives a typical
// VRF ~ 6.9. See the Phase 2 Block 2.2 writeup, Section 4 for the
// surprising-but-correct analysis (the underlying outperforms the
// asset-or-nothing despite the latter "shape-matching" the target).
MCResult
mc_european_call_exact_cv_underlying(double S, double K, double r,
                                     double sigma, double T,
                                     std::size_t n_paths,
                                     std::mt19937_64& rng,
                                     double confidence_level = 0.95);


// Price a European call by Monte Carlo with the asset-or-nothing
// payoff as the control variate (Block 2.2).
//
// Control: X_i = e^{-rT} S_T^{(i)} * 1_{S_T^{(i)} > K}.
// Closed-form expectation: E[X] = S_0 * Phi(d_1), the standard BS
// asset-or-nothing call value.
//
// Adjusted estimator: Y_tilde_i = Y_i - c_hat * (X_i - S_0 * Phi(d_1)).
//
// Empirical correlation rho_2 ~ 0.77 at the ATM point gives a typical
// VRF ~ 2.5: lower than the underlying control, contrary to a naive
// shape-matching argument. See Block 2.2 writeup for the analysis.
MCResult
mc_european_call_exact_cv_aon(double S, double K, double r,
                              double sigma, double T,
                              std::size_t n_paths,
                              std::mt19937_64& rng,
                              double confidence_level = 0.95);

}  // namespace quant
