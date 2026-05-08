// theta_scheme.hpp
//
// Phase 3 Block 3: generic theta-scheme finite-difference stepper for
// the transformed Black-Scholes PDE.
//
// The theta-scheme parametrises a one-parameter family:
//
//   (V^{n+1} - V^n) / dtau = theta * L_h V^{n+1}
//                          + (1 - theta) * L_h V^n,
//
//   theta = 0      FTCS (explicit, conditional stability)
//   theta = 1      BTCS (implicit, unconditional stability)
//   theta = 0.5    Crank-Nicolson (implicit, O(dtau^2) in time)
//
// Used by quant::pde::cn_* (Block 3) and will be reused in Block 4
// (PSOR for American options).

#pragma once

#include "pde.hpp"

#include <vector>

namespace quant::pde {

/// Pre-computed stencil coefficients for the theta-scheme.
struct ThetaCoeffs {
    double beta_minus;     ///< sub-diagonal of LHS matrix
    double beta_zero;      ///< diagonal      of LHS matrix
    double beta_plus;      ///< super-diagonal of LHS matrix
    double gamma_minus;    ///< sub-diagonal of RHS stencil
    double gamma_zero;     ///< diagonal      of RHS stencil
    double gamma_plus;     ///< super-diagonal of RHS stencil
};

/// Compute the stencil coefficients for given theta and grid step
/// sizes. See theory/phase3/block3_crank_nicolson.tex for the
/// formulae.
///
/// @throws std::invalid_argument if theta is not in [0, 1].
ThetaCoeffs theta_coeffs(double theta,
                         double sigma, double r, double mu,
                         double dtau, double dx);

/// Time-march V0 forward using the theta-scheme.
///
/// @param grid           Grid from build_grid().
/// @param V0             Initial-condition vector, size N+1.
/// @param theta          Scheme parameter; must lie in [0, 1].
/// @param bc_lower       Dirichlet values at x_min, length num_steps+1.
/// @param bc_upper       Dirichlet values at x_max, length num_steps+1.
/// @param dtau_override  If positive, use this dtau instead of grid.dtau.
///                       Pass 0 (the default) to use grid.dtau.
/// @param num_steps      Number of steps. If <0, take grid.M steps.
///                       Default -1.
///
/// @returns the solution after num_steps time steps.
///
/// @throws std::invalid_argument on parameter or shape errors.
std::vector<double> theta_march(const Grid& grid,
                                const std::vector<double>& V0,
                                double theta,
                                const std::vector<double>& bc_lower,
                                const std::vector<double>& bc_upper,
                                double dtau_override = 0.0,
                                int num_steps = -1);

}  // namespace quant::pde
