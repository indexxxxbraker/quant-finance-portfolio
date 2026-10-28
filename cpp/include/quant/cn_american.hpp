// cn_american.hpp
//
// American put option pricing by Crank-Nicolson with
// PSOR.
//
// The pricer reuses the theta-scheme infrastructure (theta_scheme.hpp) but
// replaces the per-step Thomas solve with a PSOR solve of the
// discrete LCP arising from the early-exercise constraint
// V >= g (the payoff). PSOR's projection step automatically enforces
// the constraint at every grid node, every time step, with the
// complementarity condition distinguishing continuation from exercise
// regions.
//
// No Rannacher smoothing. The projection clips the kink-induced
// oscillation from below but does not remove it; at fixed N the time
// error is first order in dtau (validate_cn_american.py, test [8]).
//
// The lower boundary condition for the American put is
//
//     V(x_min, tau) = K - K * exp(x_min)
//
// with NO time-decay factor, since deep ITM the option is exercised
// immediately regardless of remaining time.

#pragma once

#include "pde.hpp"

namespace quant::pde {

/// Price an American put by CN with PSOR.
///
/// @param S, K, r, sigma, T   Standard Black-Scholes parameters.
/// @param N                   Number of spatial intervals.
/// @param M                   Number of time intervals.
/// @param n_sigma             Truncated-domain half-width.
/// @param omega               PSOR relaxation parameter; in (0, 2).
/// @param tol_abs, tol_rel    PSOR convergence tolerances.
/// @param max_iter            PSOR iteration limit per time step.
///
/// @throws std::invalid_argument on invalid inputs.
/// @throws std::runtime_error    if PSOR fails to converge at any
///                                time step.
double cn_american_put(double S, double K, double r,
                       double sigma, double T,
                       int N, int M,
                       double n_sigma  = 4.0,
                       double omega    = 1.2,
                       double tol_abs  = 1e-10,
                       double tol_rel  = 1e-9,
                       int    max_iter = 10000);

}  // namespace quant::pde
