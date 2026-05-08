// cn.hpp
//
// Phase 3 Block 3: Crank-Nicolson finite-difference pricer for
// European options under Black-Scholes, with Rannacher time-stepping
// to suppress strike-induced oscillations.
//
// The CN scheme is unconditionally stable and second-order in both
// time and space: O(dx^2 + dtau^2) error at O(N * M) cost. With M
// scaling as N (instead of N^2), CN achieves O(dx^2) precision at
// O(N^2) total cost, breaking the O(N^3) barrier of FTCS/BTCS.
//
// Default Rannacher protocol: replace the first 2 CN steps by 4
// half-time-step BTCS sub-steps. This damps the high-frequency
// modes excited by the kink in the call/put payoff, restoring full
// O(dtau^2) global convergence (Giles & Carter 2005). Set
// rannacher_steps = 0 to observe the kink-induced oscillation
// directly.

#pragma once

#include "pde.hpp"

namespace quant::pde {

/// Price a European call by Crank-Nicolson with Rannacher smoothing.
///
/// @param S, K, r, sigma, T   Standard Black-Scholes parameters.
/// @param N                   Number of spatial intervals.
/// @param M                   Number of time intervals; any M >=
///                            rannacher_steps is valid.
/// @param n_sigma             Half-width of the truncated x-domain;
///                            default 4.
/// @param rannacher_steps     Number of CN time steps replaced by
///                            half-time-step BTCS warm-up. Default 2
///                            follows Giles & Carter (2005). Set to
///                            0 to observe the kink-induced
///                            oscillation phenomenon.
///
/// @throws std::invalid_argument on invalid inputs.
double cn_european_call(double S, double K, double r,
                        double sigma, double T,
                        int N, int M, double n_sigma = 4.0,
                        int rannacher_steps = 2);

/// Price a European put by Crank-Nicolson. See cn_european_call.
double cn_european_put(double S, double K, double r,
                       double sigma, double T,
                       int N, int M, double n_sigma = 4.0,
                       int rannacher_steps = 2);

}  // namespace quant::pde
