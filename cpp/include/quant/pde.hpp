// pde.hpp
//
// Phase 3 Block 0: finite-difference grid infrastructure for the
// Black-Scholes PDE. Declares the public interface; the implementation
// lives in pde.cpp.
//
// This module is the scaffolding for the FTCS, BTCS, Crank-Nicolson and
// PSOR schemes implemented in Phase 3 Blocks 1-4. It contains no pricer:
// it constructs the (x, tau) grid, returns Dirichlet boundary
// conditions and initial-condition vectors, and exposes the
// stability-number utilities used by the explicit scheme.
//
// The transformed Black-Scholes PDE solved by all Phase 3 schemes is
//
//     V_tau = (sigma^2 / 2) V_xx + mu V_x - r V,   mu = r - sigma^2/2,
//
// with x = ln(S/K), tau = T - t. Derivation in
// theory/phase3/block0_pde_foundations.tex.
//
// Conventions:
//   - xs has length N+1: xs[0] = x_min, xs[N] = x_max.
//   - taus has length M+1: taus[0] = 0, taus[M] = T.
//   - Boundary functions return arrays of length M+1 indexed by the
//     time-level n, so result[n] is the Dirichlet value at tau = taus[n].

#pragma once

#include <vector>

namespace quant::pde {

// ---------------------------------------------------------------------------
// Grid container
// ---------------------------------------------------------------------------

/// Immutable description of the uniform (x, tau) discretisation grid.
///
/// Members are public for direct access; the struct is constructed only
/// through build_grid() and is not meant to be mutated afterwards. The
/// drift mu = r - sigma^2/2 is exposed as a member function rather than
/// a stored field, to keep (sigma, r) the single source of truth.
struct Grid {
    int N;                          ///< number of intervals in x; (N+1) nodes
    int M;                          ///< number of intervals in tau; (M+1) levels
    double T;                       ///< time to maturity (years)
    double sigma;                   ///< volatility (constant)
    double r;                       ///< risk-free rate
    double K;                       ///< strike (anchors x = ln(S/K))
    double x_min;                   ///< truncated lower bound in log-moneyness
    double x_max;                   ///< truncated upper bound in log-moneyness
    double dx;                      ///< spatial step (x_max - x_min) / N
    double dtau;                    ///< temporal step T / M
    std::vector<double> xs;         ///< spatial nodes; size N+1
    std::vector<double> taus;       ///< time levels; size M+1

    /// Drift in log-space under the risk-neutral measure: r - sigma^2 / 2.
    double mu() const noexcept { return r - 0.5 * sigma * sigma; }
};

// ---------------------------------------------------------------------------
// Grid construction
// ---------------------------------------------------------------------------

/// Construct a uniform finite-difference grid for the transformed
/// Black-Scholes PDE.
///
/// Truncates the spatial domain to
///     x in [-n_sigma * sigma * sqrt(T), +n_sigma * sigma * sqrt(T)],
/// where x = ln(S/K). With n_sigma = 4 the lognormal-tail mass outside
/// the domain is O(exp(-n_sigma^2 / 2)) ~ 6e-5, which is well below the
/// discretisation error at any practical (N, M).
///
/// @param N        Number of spatial intervals; must be >= 2.
/// @param M        Number of time intervals;    must be >= 1.
/// @param T        Time to maturity;            must be > 0.
/// @param sigma    Volatility;                  must be > 0.
/// @param r        Risk-free rate (any sign).
/// @param K        Strike;                       must be > 0.
/// @param n_sigma  Half-width of the truncated domain in units of
///                 sigma * sqrt(T); must be > 0. Default 4.
///
/// @throws std::invalid_argument on any invalid input.
Grid build_grid(int N, int M, double T,
                double sigma, double r, double K,
                double n_sigma = 4.0);

// ---------------------------------------------------------------------------
// Initial conditions (payoff at tau = 0, i.e. t = T)
// ---------------------------------------------------------------------------

/// Initial condition for a European/American call:
///     V(x, 0) = K * max(e^x - 1, 0).
/// Equivalent to the call payoff (S - K)^+ written in log-moneyness.
std::vector<double> call_initial_condition(const std::vector<double>& xs,
                                           double K);

/// Initial condition for a European/American put:
///     V(x, 0) = K * max(1 - e^x, 0).
std::vector<double> put_initial_condition(const std::vector<double>& xs,
                                          double K);

// ---------------------------------------------------------------------------
// Dirichlet boundary conditions
// ---------------------------------------------------------------------------
// All functions return std::vector<double> of length grid.M + 1
// indexed by the time-level n: result[n] is the boundary value at
// tau = grid.taus[n].

/// Dirichlet lower boundary for the European call: V(x_min, tau) = 0.
std::vector<double> call_boundary_lower(const Grid& g);

/// Dirichlet upper boundary for the European call:
///     V(x_max, tau) = K * (e^{x_max} - e^{-r tau}).
/// Asymptotic limit C(S, t) ~ S - K * e^{-r(T - t)} as S -> infinity,
/// in log-moneyness.
std::vector<double> call_boundary_upper(const Grid& g);

/// Dirichlet lower boundary for the European put:
///     V(x_min, tau) = K * (e^{-r tau} - e^{x_min}).
/// Asymptotic limit P(S, t) -> K * e^{-r(T - t)} - S as S -> 0; the
/// e^{x_min} term is exponentially small.
std::vector<double> put_boundary_lower(const Grid& g);

/// Dirichlet upper boundary for the European put: V(x_max, tau) = 0.
std::vector<double> put_boundary_upper(const Grid& g);

// ---------------------------------------------------------------------------
// Stability numbers
// ---------------------------------------------------------------------------

/// Diffusive Fourier number alpha = (sigma^2 / 2) * dtau / dx^2.
///
/// FTCS (Block 1) is von Neumann stable iff alpha <= 1/2. BTCS
/// (Block 2) and Crank-Nicolson (Block 3) are unconditionally stable
/// in alpha.
double fourier_number(double sigma, double dtau, double dx) noexcept;

/// Advective Courant number nu = |mu| * dtau / (2 * dx).
///
/// With centred space differences for the convective term, the FTCS
/// von Neumann factor for the full BS PDE is
///     g(theta) = 1 - 4*alpha*sin^2(theta/2) + i*nu*sin(theta).
/// For typical Phase 3 parameters the diffusive bound alpha <= 1/2
/// dominates over the convective constraint.
double courant_number(double mu, double dtau, double dx) noexcept;

/// Return true iff the FTCS scheme is stable for this Fourier number,
/// i.e. iff alpha <= 0.5. Diagnostic helper for Block 1.
bool is_explicit_stable(double alpha) noexcept;

}  // namespace quant::pde
