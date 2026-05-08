// cn.cpp
//
// Implementation of the Crank-Nicolson pricer declared in cn.hpp.

#include "cn.hpp"
#include "theta_scheme.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace quant::pde {

namespace {

double interpolate_at_spot(const Grid& grid,
                           const std::vector<double>& V_final,
                           double S) {
    const double x_0 = std::log(S / grid.K);
    if (x_0 < grid.x_min || x_0 > grid.x_max) {
        throw std::invalid_argument(
            "cn: spot S = " + std::to_string(S)
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

/// Linear interpolation of a 1D array sampled at full_taus[0..K]
/// onto half_taus[0..2K]. Both arrays are assumed sorted ascending.
std::vector<double> interp_to_half_steps(
    const std::vector<double>& full_values,
    int rannacher_steps) {
    const int K_steps = rannacher_steps;
    const int n_half = 2 * K_steps;
    std::vector<double> result(static_cast<std::size_t>(n_half) + 1);
    // half_taus[i] = i * dtau/2 for i = 0, ..., 2*K
    // full_taus[k] = k * dtau   for k = 0, ..., K
    // The half-grid hits the full grid at every other point: i = 2k.
    for (int i = 0; i <= n_half; ++i) {
        if (i % 2 == 0) {
            result[static_cast<std::size_t>(i)] =
                full_values[static_cast<std::size_t>(i / 2)];
        } else {
            const int k_lo = i / 2;
            result[static_cast<std::size_t>(i)] = 0.5 *
                (full_values[static_cast<std::size_t>(k_lo)]
               + full_values[static_cast<std::size_t>(k_lo) + 1]);
        }
    }
    return result;
}

std::vector<double> cn_march(const Grid& grid,
                             const std::vector<double>& V0,
                             const std::vector<double>& bc_lower,
                             const std::vector<double>& bc_upper,
                             int rannacher_steps) {
    const int M = grid.M;
    if (rannacher_steps < 0) {
        throw std::invalid_argument(
            "cn: rannacher_steps must be non-negative, got "
            + std::to_string(rannacher_steps));
    }
    if (rannacher_steps > M) {
        throw std::invalid_argument(
            "cn: rannacher_steps=" + std::to_string(rannacher_steps)
            + " exceeds M=" + std::to_string(M));
    }

    if (rannacher_steps == 0) {
        return theta_march(grid, V0, 0.5, bc_lower, bc_upper);
    }

    // ---- Rannacher warm-up: 2 * rannacher_steps BTCS half-steps ----
    const int n_half = 2 * rannacher_steps;
    const double half_dtau = 0.5 * grid.dtau;

    // Truncate bc arrays to the warm-up region and interpolate to half-step.
    std::vector<double> bc_lower_full_warm(
        bc_lower.begin(),
        bc_lower.begin() + rannacher_steps + 1);
    std::vector<double> bc_upper_full_warm(
        bc_upper.begin(),
        bc_upper.begin() + rannacher_steps + 1);
    std::vector<double> bc_lower_half =
        interp_to_half_steps(bc_lower_full_warm, rannacher_steps);
    std::vector<double> bc_upper_half =
        interp_to_half_steps(bc_upper_full_warm, rannacher_steps);

    std::vector<double> V_warmed = theta_march(
        grid, V0, /*theta=*/1.0,
        bc_lower_half, bc_upper_half,
        /*dtau_override=*/half_dtau,
        /*num_steps=*/n_half);

    // ---- Remaining CN steps ----
    const int remaining = M - rannacher_steps;
    if (remaining == 0) return V_warmed;

    std::vector<double> bc_lower_cn(
        bc_lower.begin() + rannacher_steps, bc_lower.end());
    std::vector<double> bc_upper_cn(
        bc_upper.begin() + rannacher_steps, bc_upper.end());

    return theta_march(
        grid, V_warmed, /*theta=*/0.5,
        bc_lower_cn, bc_upper_cn,
        /*dtau_override=*/0.0,
        /*num_steps=*/remaining);
}

}  // anonymous namespace


double cn_european_call(double S, double K, double r,
                        double sigma, double T,
                        int N, int M, double n_sigma,
                        int rannacher_steps) {
    if (S <= 0.0) {
        throw std::invalid_argument(
            "cn_european_call: S must be positive, got "
            + std::to_string(S));
    }
    const Grid g = build_grid(N, M, T, sigma, r, K, n_sigma);
    const std::vector<double> V0    = call_initial_condition(g.xs, g.K);
    const std::vector<double> bc_lo = call_boundary_lower(g);
    const std::vector<double> bc_hi = call_boundary_upper(g);
    const std::vector<double> V_final =
        cn_march(g, V0, bc_lo, bc_hi, rannacher_steps);
    return interpolate_at_spot(g, V_final, S);
}

double cn_european_put(double S, double K, double r,
                       double sigma, double T,
                       int N, int M, double n_sigma,
                       int rannacher_steps) {
    if (S <= 0.0) {
        throw std::invalid_argument(
            "cn_european_put: S must be positive, got "
            + std::to_string(S));
    }
    const Grid g = build_grid(N, M, T, sigma, r, K, n_sigma);
    const std::vector<double> V0    = put_initial_condition(g.xs, g.K);
    const std::vector<double> bc_lo = put_boundary_lower(g);
    const std::vector<double> bc_hi = put_boundary_upper(g);
    const std::vector<double> V_final =
        cn_march(g, V0, bc_lo, bc_hi, rannacher_steps);
    return interpolate_at_spot(g, V_final, S);
}

}  // namespace quant::pde
