// american.hpp
//
// American put option pricers by binomial tree and by
// Longstaff-Schwartz Monte Carlo.
//
// Two functions are exposed:
//
//   - binomial_american_put: Cox-Ross-Rubinstein binomial tree.
//     Deterministic (no randomness); the only error is the lattice
//     discretisation gap between the Bermudan with N exercise dates
//     and the true continuum-exercise American. Used as the ground
//     truth in the limit N -> infty.
//
//   - lsm_american_put: Longstaff-Schwartz regression-based Monte
//     Carlo with Laguerre polynomials of degree 0 through M-1 as
//     the basis (caller chooses M). Restricted to in-the-money
//     paths at each exercise date.
//
// The LSM linear-algebra core is a from-scratch Cholesky solver of
// the M x M normal-equations matrix Psi^T Psi. M is small (4 by
// default), so a direct solve is fast and educationally valuable.
// No external linear-algebra library is required.
//
// References:
//   Phase 2 Block 6 writeup. Longstaff and Schwartz (2001),
//   Review of Financial Studies 14(1):113-147. Cox, Ross and
//   Rubinstein (1979), JFE 7(3):229-263. Glasserman, Chapter 8.

#pragma once

#include "monte_carlo.hpp"   // for MCResult

#include <cstddef>
#include <random>

namespace quant {

// =====================================================================
// Cox-Ross-Rubinstein binomial tree (ground truth)
// =====================================================================

// Price an American put by the CRR binomial tree.
//
// Lattice parameters:
//   u = exp(sigma * sqrt(dt)),  d = 1/u,
//   p = (exp(r*dt) - d) / (u - d).
//
// Backward induction: at each non-terminal node the value is the max
// of the intrinsic (K - S)^+ and the discounted risk-neutral
// expectation of the next-step values.
//
// Throws std::invalid_argument on bad inputs or if the resulting
// risk-neutral probability falls outside (0, 1) (which signals
// either pathological parameters or insufficient n_steps for the
// chosen r and sigma).
double
binomial_american_put(double S, double K, double r, double sigma, double T,
                      std::size_t n_steps);


// =====================================================================
// Longstaff-Schwartz Monte Carlo
// =====================================================================

// Price an American put by Longstaff-Schwartz regression Monte Carlo
// with the first basis_size Laguerre polynomials evaluated at S/K
// as the regression basis.
//
// Algorithm (see Phase 2 Block 6 writeup, Section 3):
//   1. Simulate n_paths exact-GBM paths over n_steps equispaced
//      observation dates.
//   2. Initialise per-path cash flows at maturity.
//   3. Backward induction: at each k = n_steps-1 down to 1, discount
//      cash flows one step, regress the discounted future cash flows
//      onto the Laguerre basis values at S_k/K (in-the-money paths
//      only), and replace cash flows with the intrinsic value where
//      the regression-based continuation estimate is dominated by
//      the intrinsic value.
//   4. Discount once more to t_0; reduce.
//
// Returns an MCResult under the standard MC asymptotic CI. The
// estimator is generally not exactly unbiased (in-sample fitting
// biases the price up; sub-optimal exercise rule biases it down),
// but the two biases approximately cancel for large n_paths and
// reasonable basis_size.
//
// rng must outlive the call. Throws std::invalid_argument on bad
// inputs.
MCResult
lsm_american_put(double S, double K, double r, double sigma, double T,
                 std::size_t n_paths, std::size_t n_steps,
                 std::size_t basis_size,
                 std::mt19937_64& rng,
                 double confidence_level = 0.95);

}  // namespace quant
