// cn_american.cpp
//
// Implementation of the American put pricer declared in cn_american.hpp.

#include "cn_american.hpp"
#include "psor.hpp"
#include "theta_scheme.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace quant::pde {

namespace {

/// Lower-boundary array for the American put: deep ITM, no time decay.
std::vector<double> american_put_boundary_lower(const Grid& g) {
    const double v = g.K - g.K * std::exp(g.x_min);
    return std::vector<double>(static_cast<std::size_t>(g.M) + 1, v);
}

/// Upper-boundary array for the American put: deep OTM, V = 0.
std::vector<double> american_put_boundary_upper(const Grid& g) {
    return std::vector<double>(static_cast<std::size_t>(g.M) + 1, 0.0);
}

double interpolate_at_spot(const Grid& grid,
                           const std::vector<double>& V_final,
                           double S) {
    const double x_0 = std::log(S / grid.K);
    if (x_0 < grid.x_min || x_0 > grid.x_max) {
        throw std::invalid_argument(
            "cn_american_put: spot S = " + std::to_string(S)
            + " lies outside the truncated domain.");
    }
    const double idx_real = (x_0 - grid.x_min) / grid.dx;
    int j = static_cast<int>(std::floor(idx_real));
    if (j >= grid.N) j = grid.N - 1;
    if (j < 0)       j = 0;
    const double w = idx_real - static_cast<double>(j);
    return (1.0 - w) * V_final[static_cast<std::size_t>(j)]
         +        w  * V_final[static_cast<std::size_t>(j) + 1];
}

}  // anonymous namespace


double cn_american_put(double S, double K, double r,
                       double sigma, double T,
                       int N, int M,
                       double n_sigma,
                       double omega,
                       double tol_abs,
                       double tol_rel,
                       int    max_iter) {
    if (S <= 0.0) {
        throw std::invalid_argument(
            "cn_american_put: S must be positive, got "
            + std::to_string(S));
    }

    const Grid g = build_grid(N, M, T, sigma, r, K, n_sigma);
    const std::vector<double> V0    = put_initial_condition(g.xs, g.K);
    const std::vector<double> bc_lo = american_put_boundary_lower(g);
    const std::vector<double> bc_hi = american_put_boundary_upper(g);

    // CN coefficients.
    const ThetaCoeffs c = theta_coeffs(
        0.5, g.sigma, g.r, g.mu(), g.dtau, g.dx);

    const int n_int = N - 1;

    // Pre-build the LHS tridiagonal entries (constant in time).
    const std::vector<double> sub (
        static_cast<std::size_t>(n_int) - 1, c.beta_minus);
    const std::vector<double> diag(
        static_cast<std::size_t>(n_int),     c.beta_zero);
    const std::vector<double> sup (
        static_cast<std::size_t>(n_int) - 1, c.beta_plus);

    // Obstacle on interior nodes: put payoff at x_j.
    std::vector<double> obstacle(static_cast<std::size_t>(n_int));
    for (int j = 1; j < N; ++j) {
        const double payoff = K - K * std::exp(g.xs[static_cast<std::size_t>(j)]);
        obstacle[static_cast<std::size_t>(j) - 1] = std::max(payoff, 0.0);
    }

    std::vector<double> V = V0;
    std::vector<double> rhs(static_cast<std::size_t>(n_int));

    for (int n = 0; n < M; ++n) {
        // RHS = B * V_int^n + boundary corrections.
        for (int j = 1; j < N; ++j) {
            const std::size_t jp = static_cast<std::size_t>(j);
            rhs[jp - 1] =
                c.gamma_minus * V[jp - 1]
              + c.gamma_zero  * V[jp]
              + c.gamma_plus  * V[jp + 1];
        }
        rhs[0] -=
            c.beta_minus * bc_lo[static_cast<std::size_t>(n) + 1];
        rhs[static_cast<std::size_t>(n_int) - 1] -=
            c.beta_plus * bc_hi[static_cast<std::size_t>(n) + 1];

        // Warm start: previous interior values.
        std::vector<double> x0(static_cast<std::size_t>(n_int));
        for (int j = 1; j < N; ++j) {
            x0[static_cast<std::size_t>(j) - 1] =
                V[static_cast<std::size_t>(j)];
        }

        // PSOR solve.
        const PSORResult res = psor_solve(
            sub, diag, sup, rhs, obstacle,
            omega, tol_abs, tol_rel, max_iter, x0);

        for (int j = 1; j < N; ++j) {
            V[static_cast<std::size_t>(j)] =
                res.x[static_cast<std::size_t>(j) - 1];
        }
        V[0] = bc_lo[static_cast<std::size_t>(n) + 1];
        V[static_cast<std::size_t>(N)] =
            bc_hi[static_cast<std::size_t>(n) + 1];
    }

    return interpolate_at_spot(g, V, S);
}

}  // namespace quant::pde
