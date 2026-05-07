// btcs.hpp
//
// Phase 3 Block 2: implicit Backward-Time Centred-Space (BTCS) finite-
// difference pricer for European options under Black-Scholes.
//
// Built on the grid infrastructure of Block 0 (pde.hpp) and the
// Thomas algorithm of Block 2 (thomas.hpp). The pricer is
// unconditionally stable: any combination (N, M) with N >= 2, M >= 1
// produces a finite, convergent solution. There is no CFL constraint
// to enforce.
//
// At each time step the BTCS scheme requires solving the tridiagonal
// system
//
//     [ b_minus  b_zero   b_plus ] V^{n+1} = V^n + (boundary terms)
//
// with constant coefficients
//
//     b_minus = -alpha + mu * dtau / (2 * dx)
//     b_zero  =  1 + 2 * alpha + r * dtau
//     b_plus  = -alpha - mu * dtau / (2 * dx).
//
// The matrix is constant in time, so its Thomas factorisation is
// computed once at setup and reused for all M solves.

#pragma once

#include "pde.hpp"

namespace quant::pde {

/// Price a European call by the implicit BTCS scheme.
///
/// @param S        Spot price; must be > 0.
/// @param K        Strike;     must be > 0.
/// @param r        Risk-free rate.
/// @param sigma    Volatility; must be > 0.
/// @param T        Time to maturity; must be > 0.
/// @param N        Number of spatial intervals; must be >= 2.
/// @param M        Number of time intervals; any M >= 1 is valid
///                 (no CFL constraint, in contrast to FTCS).
/// @param n_sigma  Half-width of the truncated x-domain; default 4.
///
/// @throws std::invalid_argument on invalid inputs (S, K, sigma, T
///         non-positive; N < 2; M < 1).
double btcs_european_call(double S, double K, double r,
                          double sigma, double T,
                          int N, int M, double n_sigma = 4.0);

/// Price a European put by the implicit BTCS scheme.
/// Same semantics as btcs_european_call.
double btcs_european_put(double S, double K, double r,
                         double sigma, double T,
                         int N, int M, double n_sigma = 4.0);

}  // namespace quant::pde
