// ftcs.cpp
//
// Implementation of the explicit FTCS pricer declared in ftcs.hpp.
// Phase 3 Block 1.

#include "ftcs.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace quant::pde {

namespace {

/// Linearly interpolate V_final at x_0 = log(S / K) to recover the
/// price at the user-supplied spot. Linear interpolation contributes
/// an O(dx^2) error consistent with the spatial discretisation order.
///
/// Throws std::invalid_argument if x_0 is outside [x_min, x_max].
double interpolate_at_spot(const Grid& grid,
                           const std::vector<double>& V_final,
                           double S) {
    const double x_0 = std::log(S / grid.K);
    if (x_0 < grid.x_min || x_0 > grid.x_max) {
        throw std::invalid_argument(
            "ftcs: spot S = " + std::to_string(S)
            + " lies outside the truncated domain. "
            + "Increase n_sigma or pick a closer-to-K spot.");
    }
    // Locate the cell [xs[j], xs[j+1]] containing x_0. The grid is
    // uniform, so an O(1) formula suffices; we do a defensive clamp
    // in case x_0 == x_max (right endpoint).
    const double idx_real = (x_0 - grid.x_min) / grid.dx;
    int j = static_cast<int>(std::floor(idx_real));
    if (j >= grid.N) j = grid.N - 1;
    if (j < 0)       j = 0;
    const double w = idx_real - static_cast<double>(j);
    return (1.0 - w) * V_final[static_cast<std::size_t>(j)]
         +        w  * V_final[static_cast<std::size_t>(j) + 1];
}

}  // anonymous namespace


// ---------------------------------------------------------------------------
// Time-marching kernel
// ---------------------------------------------------------------------------

std::vector<double> ftcs_march(const Grid& grid,
                               const std::vector<double>& V0,
                               const std::vector<double>& bc_lower,
                               const std::vector<double>& bc_upper,
                               bool validate_cfl) {
    const int N = grid.N;
    const int M = grid.M;
    const double dx = grid.dx;
    const double dtau = grid.dtau;
    const double sigma = grid.sigma;
    const double r = grid.r;
    const double mu = grid.mu();

    const double alpha = 0.5 * sigma * sigma * dtau / (dx * dx);
    if (validate_cfl && alpha > 0.5) {
        throw std::invalid_argument(
            "ftcs: CFL violated, alpha = " + std::to_string(alpha)
            + " > 0.5. FTCS will diverge. "
              "Increase M (more time steps) or decrease N "
              "(coarser space) to restore stability.");
    }

    // Stencil coefficients. Constant in (j, n) thanks to the log-transform.
    const double nu_signed = mu * dtau / (2.0 * dx);
    const double a_minus = alpha - nu_signed;
    const double a_zero  = 1.0 - 2.0 * alpha - r * dtau;
    const double a_plus  = alpha + nu_signed;

    // Double buffer: V holds the current level, V_new the next.
    std::vector<double> V    = V0;
    std::vector<double> V_new(static_cast<std::size_t>(N) + 1);

    // March forward in tau. After step n, V holds level n+1.
    for (int n = 0; n < M; ++n) {
        V_new[0] = bc_lower[static_cast<std::size_t>(n) + 1];
        V_new[static_cast<std::size_t>(N)] =
            bc_upper[static_cast<std::size_t>(n) + 1];

        for (int j = 1; j < N; ++j) {
            const std::size_t jp = static_cast<std::size_t>(j);
            V_new[jp] =
                a_minus * V[jp - 1]
              + a_zero  * V[jp]
              + a_plus  * V[jp + 1];
        }

        // Swap buffers: O(1) on std::vector via std::swap.
        std::swap(V, V_new);
    }
    return V;
}


// ---------------------------------------------------------------------------
// High-level pricers
// ---------------------------------------------------------------------------

double ftcs_european_call(double S, double K, double r,
                          double sigma, double T,
                          int N, int M, double n_sigma) {
    if (S <= 0.0) {
        throw std::invalid_argument(
            "ftcs_european_call: S must be positive, got "
            + std::to_string(S));
    }
    const Grid g = build_grid(N, M, T, sigma, r, K, n_sigma);
    const std::vector<double> V0    = call_initial_condition(g.xs, g.K);
    const std::vector<double> bc_lo = call_boundary_lower(g);
    const std::vector<double> bc_hi = call_boundary_upper(g);
    const std::vector<double> V_final = ftcs_march(g, V0, bc_lo, bc_hi);
    return interpolate_at_spot(g, V_final, S);
}

double ftcs_european_put(double S, double K, double r,
                         double sigma, double T,
                         int N, int M, double n_sigma) {
    if (S <= 0.0) {
        throw std::invalid_argument(
            "ftcs_european_put: S must be positive, got "
            + std::to_string(S));
    }
    const Grid g = build_grid(N, M, T, sigma, r, K, n_sigma);
    const std::vector<double> V0    = put_initial_condition(g.xs, g.K);
    const std::vector<double> bc_lo = put_boundary_lower(g);
    const std::vector<double> bc_hi = put_boundary_upper(g);
    const std::vector<double> V_final = ftcs_march(g, V0, bc_lo, bc_hi);
    return interpolate_at_spot(g, V_final, S);
}


// ---------------------------------------------------------------------------
// Helper: minimum M for CFL
// ---------------------------------------------------------------------------

int ftcs_min_M_for_cfl(int N, double T, double sigma,
                       double n_sigma, double target_alpha) {
    if (target_alpha <= 0.0 || target_alpha > 0.5) {
        throw std::invalid_argument(
            "ftcs_min_M_for_cfl: target_alpha must be in (0, 0.5], got "
            + std::to_string(target_alpha));
    }
    const double half_width = n_sigma * sigma * std::sqrt(T);
    const double dx = 2.0 * half_width / static_cast<double>(N);
    const double dtau_max = 2.0 * target_alpha * dx * dx / (sigma * sigma);
    return static_cast<int>(std::ceil(T / dtau_max));
}

}  // namespace quant::pde
