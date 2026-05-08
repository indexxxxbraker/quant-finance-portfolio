// psor.hpp
//
// Phase 3 Block 4: Projected Successive Over-Relaxation (PSOR) solver
// for tridiagonal linear complementarity problems.
//
// Solves the LCP
//
//     A x >= rhs           (componentwise)
//     x   >= obstacle      (componentwise)
//     (A x - rhs) (x - obstacle) = 0
//
// where A is tridiagonal, by SOR iteration with a max-projection onto
// the obstacle:
//
//     x_i^{(k+1)} = max{ obstacle_i,
//                        (1 - omega) * x_i^{(k)}
//                        + omega/diag_i * (rhs_i - sub_{i-1} * x_{i-1}^{(k+1)}
//                                                  - sup_i * x_{i+1}^{(k)}) }
//
// Convergence is geometric for omega in (0, 2) when A is symmetric
// positive-definite (Cryer 1971), and for diagonally-dominant
// M-matrices in the non-symmetric case.

#pragma once

#include <utility>
#include <vector>

namespace quant::pde {

/// Result of a PSOR solve: solution and number of iterations actually
/// performed.
struct PSORResult {
    std::vector<double> x;
    int n_iter;
};

/// Solve a tridiagonal LCP by Projected SOR.
///
/// @param sub          Sub-diagonal of A, length n-1.
/// @param diag         Main diagonal of A, length n. Must be non-zero.
/// @param sup          Super-diagonal of A, length n-1.
/// @param rhs          Right-hand side, length n.
/// @param obstacle     Lower bound for the solution, length n.
/// @param omega        Relaxation parameter; must lie in (0, 2).
///                     Default 1.2.
/// @param tol_abs      Absolute tolerance on the iteration increment.
///                     Default 1e-8.
/// @param tol_rel      Relative tolerance. Default 1e-7.
/// @param max_iter     Iteration limit. Default 10000.
/// @param x0           Initial guess (size n) or empty vector for
///                     default (= obstacle). If non-empty, must
///                     satisfy x0 >= obstacle.
///
/// @returns PSORResult with the solution and the iteration count.
///
/// @throws std::invalid_argument on parameter or shape errors.
/// @throws std::runtime_error    if max_iter is reached without
///                                satisfying the tolerance.
PSORResult psor_solve(
    const std::vector<double>& sub,
    const std::vector<double>& diag,
    const std::vector<double>& sup,
    const std::vector<double>& rhs,
    const std::vector<double>& obstacle,
    double omega    = 1.2,
    double tol_abs  = 1e-8,
    double tol_rel  = 1e-7,
    int    max_iter = 10000,
    const std::vector<double>& x0 = {});

}  // namespace quant::pde
