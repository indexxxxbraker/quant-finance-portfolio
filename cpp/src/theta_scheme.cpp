// theta_scheme.cpp
//
// Implementation of the generic theta-scheme stepper declared in
// theta_scheme.hpp.

#include "theta_scheme.hpp"
#include "thomas.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace quant::pde {

ThetaCoeffs theta_coeffs(double theta,
                         double sigma, double r, double mu,
                         double dtau, double dx) {
    if (theta < 0.0 || theta > 1.0) {
        throw std::invalid_argument(
            "theta_coeffs: theta must be in [0, 1], got "
            + std::to_string(theta));
    }
    const double alpha = 0.5 * sigma * sigma * dtau / (dx * dx);
    const double nu    = mu * dtau / (2.0 * dx);
    const double one_minus_theta = 1.0 - theta;

    ThetaCoeffs c;
    c.beta_minus  = -theta            * (alpha - nu);
    c.beta_zero   = 1.0 + theta       * (2.0 * alpha + r * dtau);
    c.beta_plus   = -theta            * (alpha + nu);
    c.gamma_minus = +one_minus_theta  * (alpha - nu);
    c.gamma_zero  = 1.0 - one_minus_theta * (2.0 * alpha + r * dtau);
    c.gamma_plus  = +one_minus_theta  * (alpha + nu);
    return c;
}


std::vector<double> theta_march(const Grid& grid,
                                const std::vector<double>& V0,
                                double theta,
                                const std::vector<double>& bc_lower,
                                const std::vector<double>& bc_upper,
                                double dtau_override,
                                int num_steps) {
    const int N = grid.N;
    const int M = (num_steps >= 0) ? num_steps : grid.M;
    const double dx   = grid.dx;
    const double dtau = (dtau_override > 0.0) ? dtau_override : grid.dtau;

    const std::size_t expected = static_cast<std::size_t>(M) + 1;
    if (bc_lower.size() != expected || bc_upper.size() != expected) {
        throw std::invalid_argument(
            "theta_march: bc arrays must have length num_steps + 1 = "
            + std::to_string(expected) + ", got bc_lower="
            + std::to_string(bc_lower.size())
            + ", bc_upper=" + std::to_string(bc_upper.size()));
    }

    const ThetaCoeffs c = theta_coeffs(
        theta, grid.sigma, grid.r, grid.mu(), dtau, dx);

    const bool is_explicit = (theta == 0.0);
    const int n_int = N - 1;

    // Pre-factor the LHS tridiagonal once.
    ThomasFactor factor;
    if (!is_explicit) {
        const std::vector<double> sub (
            static_cast<std::size_t>(n_int) - 1, c.beta_minus);
        const std::vector<double> diag(
            static_cast<std::size_t>(n_int),     c.beta_zero);
        const std::vector<double> sup (
            static_cast<std::size_t>(n_int) - 1, c.beta_plus);
        factor = thomas_factor(sub, diag, sup);
    }

    std::vector<double> V = V0;
    std::vector<double> rhs(static_cast<std::size_t>(n_int));

    for (int n = 0; n < M; ++n) {
        // RHS = B * V_int^n. Stencil application using V at all nodes
        // (including the boundary positions, which correctly hold V^n).
        for (int j = 1; j < N; ++j) {
            const std::size_t jp = static_cast<std::size_t>(j);
            rhs[jp - 1] =
                c.gamma_minus * V[jp - 1]
              + c.gamma_zero  * V[jp]
              + c.gamma_plus  * V[jp + 1];
        }
        // Boundary corrections: move beta_- V_0^{n+1} and
        // beta_+ V_N^{n+1} from LHS to RHS (with sign flip).
        rhs[0] -=
            c.beta_minus * bc_lower[static_cast<std::size_t>(n) + 1];
        rhs[static_cast<std::size_t>(n_int) - 1] -=
            c.beta_plus * bc_upper[static_cast<std::size_t>(n) + 1];

        std::vector<double> V_int_new;
        if (is_explicit) {
            V_int_new = rhs;            // copy: shape (N-1,)
        } else {
            V_int_new = factor.solve(rhs);
        }

        for (int j = 1; j < N; ++j) {
            V[static_cast<std::size_t>(j)] =
                V_int_new[static_cast<std::size_t>(j - 1)];
        }
        V[0] = bc_lower[static_cast<std::size_t>(n) + 1];
        V[static_cast<std::size_t>(N)] =
            bc_upper[static_cast<std::size_t>(n) + 1];
    }
    return V;
}

}  // namespace quant::pde
