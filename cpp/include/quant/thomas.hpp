// thomas.hpp
//
// Phase 3 Block 2: Thomas algorithm for tridiagonal linear systems.
//
// Solves Ax = rhs in O(n) operations for the system
//
//     diag[0]   sup[0]                                   x[0]   = rhs[0]
//     sub[0]    diag[1]   sup[1]                         x[1]   = rhs[1]
//               sub[1]    diag[2]   sup[2]               x[2]   = rhs[2]
//                         ...                            ...
//                         sub[n-2]  diag[n-1]            x[n-1] = rhs[n-1]
//
// The implementation does Gaussian elimination without pivoting. This
// is provably backward-stable when the matrix is diagonally dominant
// (Higham 2002, Theorem 9.5). The BTCS application of this block
// satisfies diagonal dominance whenever the grid Pe'clet condition
// |mu| * dx / sigma^2 <= 1 holds, which is the case across the
// project's parameter range.
//
// For repeated solves with a fixed matrix and varying right-hand side
// (the BTCS situation: M time steps with the same A), use
//
//     ThomasFactor f = thomas_factor(sub, diag, sup);
//     for each rhs:
//         x = f.solve(rhs);
//
// to avoid recomputing the (n-1) divisions of the forward sweep at
// each solve.

#pragma once

#include <vector>

namespace quant::pde {

/// Pre-computed factorisation of a tridiagonal matrix.
/// Returned by thomas_factor(); used by ThomasFactor::solve().
struct ThomasFactor {
    std::vector<double> sub;       ///< copy of the sub-diagonal
    std::vector<double> c_prime;   ///< modified super-diagonal, length n-1
    std::vector<double> m;         ///< effective pivots, length n

    /// Apply the factorisation to a right-hand side rhs of length n,
    /// returning the solution.
    /// @throws std::invalid_argument on size mismatch.
    std::vector<double> solve(const std::vector<double>& rhs) const;
};


/// One-shot tridiagonal solve.
///
/// @param sub   sub-diagonal entries (length n-1; sub[i] is at row i+1, col i)
/// @param diag  main diagonal      (length n)
/// @param sup   super-diagonal     (length n-1; sup[i] is at row i, col i+1)
/// @param rhs   right-hand side    (length n)
///
/// @returns the solution x (length n)
///
/// @throws std::invalid_argument on size mismatch or empty system.
/// @throws std::runtime_error    if a pivot is zero (matrix singular
///                                or not diagonally dominant).
std::vector<double> thomas_solve(const std::vector<double>& sub,
                                 const std::vector<double>& diag,
                                 const std::vector<double>& sup,
                                 const std::vector<double>& rhs);


/// Compute the factorisation of a tridiagonal matrix for repeated
/// solves with different right-hand sides.
///
/// @param sub, diag, sup   matrix entries; same shapes as in thomas_solve.
///
/// @returns a ThomasFactor instance; pass to ThomasFactor::solve().
///
/// @throws std::invalid_argument on size mismatch.
/// @throws std::runtime_error    if a pivot is zero.
ThomasFactor thomas_factor(const std::vector<double>& sub,
                           const std::vector<double>& diag,
                           const std::vector<double>& sup);

}  // namespace quant::pde
