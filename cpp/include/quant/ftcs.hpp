// ftcs.hpp
//
// Phase 3 Block 1: explicit Forward-Time Centred-Space (FTCS) finite-
// difference pricer for European options under Black-Scholes.
//
// Built on the grid infrastructure of Block 0 (pde.hpp). The pricer
// time-marches the discrete BS PDE forward in tau = T - t from the
// payoff at tau = 0 to the price at tau = T, then linearly interpolates
// at the user-supplied spot S.
//
// Stability: the FTCS scheme is conditionally stable; the CFL bound
// alpha <= 1/2 is enforced strictly by the high-level pricers, with
// std::invalid_argument thrown on violation. See
// theory/phase3/block1_ftcs.tex for the full analysis.

#pragma once

#include <vector>

#include "pde.hpp"

namespace quant::pde {

// ---------------------------------------------------------------------------
// High-level pricers
// ---------------------------------------------------------------------------

/// Price a European call by the explicit FTCS scheme.
///
/// @param S        Spot price; must be > 0.
/// @param K        Strike;     must be > 0.
/// @param r        Risk-free rate (any sign).
/// @param sigma    Volatility; must be > 0.
/// @param T        Time to maturity; must be > 0.
/// @param N        Number of spatial intervals; must be >= 2.
/// @param M        Number of time intervals; large enough to satisfy
///                 the CFL condition alpha = (sigma^2 / 2) * (T / M)
///                 / dx^2 <= 1/2. Otherwise std::invalid_argument is
///                 thrown.
/// @param n_sigma  Half-width of the truncated x-domain in units of
///                 sigma * sqrt(T); default 4.
///
/// @throws std::invalid_argument on invalid inputs or CFL violation.
double ftcs_european_call(double S, double K, double r,
                          double sigma, double T,
                          int N, int M, double n_sigma = 4.0);

/// Price a European put by the explicit FTCS scheme.
/// Same semantics as ftcs_european_call.
double ftcs_european_put(double S, double K, double r,
                         double sigma, double T,
                         int N, int M, double n_sigma = 4.0);

// ---------------------------------------------------------------------------
// Helper: minimum M for CFL at a given (N, T, sigma, n_sigma)
// ---------------------------------------------------------------------------

/// Smallest M such that alpha <= target_alpha. Useful for picking M
/// when prototyping. With target_alpha < 0.5 the result has a safety
/// margin: a few extra time steps relative to the strict CFL boundary.
///
/// @param N             Number of spatial intervals.
/// @param T             Time to maturity.
/// @param sigma         Volatility.
/// @param n_sigma       Half-width of the truncated domain (default 4).
/// @param target_alpha  Target Fourier number; must lie in (0, 0.5].
///
/// @throws std::invalid_argument if target_alpha is out of range.
int ftcs_min_M_for_cfl(int N, double T, double sigma,
                       double n_sigma = 4.0,
                       double target_alpha = 0.4);

// ---------------------------------------------------------------------------
// Low-level time-marching kernel (exposed for diagnostics / tests)
// ---------------------------------------------------------------------------

/// Time-march V0 from tau = 0 to tau = T using the FTCS scheme.
///
/// This is the lowest-level routine: it does no parameter validation
/// beyond an optional CFL check, expects pre-computed Dirichlet
/// arrays, and returns the full V vector at tau = T.
///
/// @param grid          Grid from build_grid().
/// @param V0            Initial-condition vector, size N+1.
/// @param bc_lower      Dirichlet values at x_min, one per time level
///                      (size M+1).
/// @param bc_upper      Dirichlet values at x_max, one per time level
///                      (size M+1).
/// @param validate_cfl  If true, throw std::invalid_argument when
///                      alpha > 1/2. The CFL-violating diagnostic
///                      passes false; no other caller should.
///
/// @throws std::invalid_argument if validate_cfl and alpha > 1/2.
std::vector<double> ftcs_march(const Grid& grid,
                               const std::vector<double>& V0,
                               const std::vector<double>& bc_lower,
                               const std::vector<double>& bc_upper,
                               bool validate_cfl = true);

}  // namespace quant::pde
