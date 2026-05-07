// btcs.cpp
//
// Implementation of the BTCS pricer declared in btcs.hpp.

#include "btcs.hpp"
#include "thomas.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace quant::pde {

namespace {

/// Linearly interpolate V_final at x_0 = log(S / K) to recover the
/// price at the user-supplied spot.
double interpolate_at_spot(const Grid& grid,
                           const std::vector<double>& V_final,
                           double S) {
    const double x_0 = std::log(S / grid.K);
    if (x_0 < grid.x_min || x_0 > grid.x_max) {
        throw std::invalid_argument(
            "btcs: spot S = " + std::to_string(S)
            + " lies outside the truncated domain. "
              "Increase n_sigma or pick a closer-to-K spot.");
    }
    const double idx_real = (x_0 - grid.x_min) / grid.dx;
    int j = static_cast<int>(std::floor(idx_real));
    if (j >= grid.N) j = grid.N - 1;
    if (j < 0)       j = 0;
    const double w = idx_real - static_cast<double>(j);
    return (1.0 - w) * V_final[static_cast<std::size_t>(j)]
         +        w  * V_final[static_cast<std::size_t>(j) + 1];
}

/// Time-march V0 from tau=0 to tau=T using BTCS. No CFL check (BTCS
/// is unconditionally stable).
std::vector<double> btcs_march(const Grid& grid,
                               const std::vector<double>& V0,
                               const std::vector<double>& bc_lower,
                               const std::vector<double>& bc_upper) {
    const int N = grid.N;
    const int M = grid.M;
    const double dx = grid.dx;
    const double dtau = grid.dtau;
    const double sigma = grid.sigma;
    const double r = grid.r;
    const double mu = grid.mu();

    // BTCS coefficients.
    const double alpha = 0.5 * sigma * sigma * dtau / (dx * dx);
    const double nu_signed = mu * dtau / (2.0 * dx);
    const double b_minus = -alpha + nu_signed;
    const double b_zero  =  1.0 + 2.0 * alpha + r * dtau;
    const double b_plus  = -alpha - nu_signed;

    // Tridiagonal matrix on the (N-1) interior nodes.
    const int n_int = N - 1;
    const std::vector<double> sub (static_cast<std::size_t>(n_int) - 1,
                                    b_minus);
    const std::vector<double> diag(static_cast<std::size_t>(n_int),
                                    b_zero);
    const std::vector<double> sup (static_cast<std::size_t>(n_int) - 1,
                                    b_plus);

    // Pre-factor once. The matrix is constant in time.
    const ThomasFactor factor = thomas_factor(sub, diag, sup);

    std::vector<double> V = V0;
    std::vector<double> d(static_cast<std::size_t>(n_int));

    for (int n = 0; n < M; ++n) {
        // Build the right-hand side. Interior values from V^n,
        // augmented at the boundaries.
        for (int j = 1; j < N; ++j) {
            d[static_cast<std::size_t>(j - 1)] =
                V[static_cast<std::size_t>(j)];
        }
        d[0]                            -= b_minus * bc_lower[static_cast<std::size_t>(n) + 1];
        d[static_cast<std::size_t>(n_int) - 1] -= b_plus  * bc_upper[static_cast<std::size_t>(n) + 1];

        // Solve.
        const std::vector<double> V_int_new = factor.solve(d);

        // Place new interior values back.
        for (int j = 1; j < N; ++j) {
            V[static_cast<std::size_t>(j)] =
                V_int_new[static_cast<std::size_t>(j - 1)];
        }
        // Set new boundary values.
        V[0] = bc_lower[static_cast<std::size_t>(n) + 1];
        V[static_cast<std::size_t>(N)] =
            bc_upper[static_cast<std::size_t>(n) + 1];
    }
    return V;
}

}  // anonymous namespace


// ---------------------------------------------------------------------------
// High-level pricers
// ---------------------------------------------------------------------------

double btcs_european_call(double S, double K, double r,
                          double sigma, double T,
                          int N, int M, double n_sigma) {
    if (S <= 0.0) {
        throw std::invalid_argument(
            "btcs_european_call: S must be positive, got "
            + std::to_string(S));
    }
    const Grid g = build_grid(N, M, T, sigma, r, K, n_sigma);
    const std::vector<double> V0    = call_initial_condition(g.xs, g.K);
    const std::vector<double> bc_lo = call_boundary_lower(g);
    const std::vector<double> bc_hi = call_boundary_upper(g);
    const std::vector<double> V_final = btcs_march(g, V0, bc_lo, bc_hi);
    return interpolate_at_spot(g, V_final, S);
}

double btcs_european_put(double S, double K, double r,
                         double sigma, double T,
                         int N, int M, double n_sigma) {
    if (S <= 0.0) {
        throw std::invalid_argument(
            "btcs_european_put: S must be positive, got "
            + std::to_string(S));
    }
    const Grid g = build_grid(N, M, T, sigma, r, K, n_sigma);
    const std::vector<double> V0    = put_initial_condition(g.xs, g.K);
    const std::vector<double> bc_lo = put_boundary_lower(g);
    const std::vector<double> bc_hi = put_boundary_upper(g);
    const std::vector<double> V_final = btcs_march(g, V0, bc_lo, bc_hi);
    return interpolate_at_spot(g, V_final, S);
}

}  // namespace quant::pde
