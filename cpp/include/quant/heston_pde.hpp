// heston_pde.hpp
//
// Heston model: 2D PDE pricing via Alternating Direction Implicit (ADI).
//
// This module mirrors quantlib.heston_pde on the Python side. It uses
// HestonParams from heston_fourier.hpp for parameter passing, and
// implements the Douglas ADI scheme of In't Hout & Foulon (2010).
//
// The 2D grid in (X = log S, v) is stored row-major as a flat
// std::vector<double>: index (i, j) -> i * (N_v + 1) + j, where i
// ranges over X and j over v.
//
// See theory/phase4/block5_heston_pde.tex for the derivation.
//
// Note: the Thomas tridiagonal solver in this module is internal,
// inlined in the anonymous namespace because it operates on batched
// systems with a different memory layout than the 1D quant::thomas
// utility from Phase 3 Block 2. They solve the same algorithm but
// with different APIs adapted to their respective use cases.

#pragma once

#include "heston_fourier.hpp"   // HestonParams

#include <cstddef>

namespace quant::heston {

// =====================================================================
// Public interface
// =====================================================================

// Price a European call under Heston via 2D PDE with Douglas ADI.
//
// The pricer builds uniform grids in (X = log S, v, tau = T - t),
// applies the call payoff as initial condition, and runs N_tau Douglas
// time-steps. The price at (S0, v0) is recovered by bilinear
// interpolation from the final grid, then multiplied by exp(-r*T) to
// undo the discount substitution applied during PDE setup.
//
// Parameters:
//   S0, K, T            spot, strike, maturity (positive)
//   r                   risk-free rate (unconstrained)
//   p                   Heston parameters (kappa, theta, sigma, rho, v0)
//   N_X, N_v, N_tau     grid sizes (each must be >= 4)
//   theta_imp           Douglas implicitness; default 0.5
//   X_factor            log-spot grid extent multiplier; default 4.0
//   v_max_factor        v-grid extent multiplier (* theta); default 5.0
//
// Returns: European call price.
//
// Convergence: error is O(dX^2 + dv^2 + dtau). At default grids
// (N_X = 200, N_v = 100, N_tau = 100) the spatial error dominates;
// halving (N_X, N_v) reduces error by ~4x.
//
// Throws std::invalid_argument on parameter validation failure.
double heston_call_pde(double S0,
                        double K,
                        double T,
                        double r,
                        const HestonParams& p,
                        std::size_t N_X = 200,
                        std::size_t N_v = 100,
                        std::size_t N_tau = 100,
                        double theta_imp = 0.5,
                        double X_factor = 4.0,
                        double v_max_factor = 5.0);

}  // namespace quant::heston
