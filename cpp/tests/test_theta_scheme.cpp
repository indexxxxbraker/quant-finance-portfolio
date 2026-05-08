// test_theta_scheme.cpp
//
// Catch2 tests for the generic theta-scheme stepper of Phase 3 Block 3.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "theta_scheme.hpp"
#include "btcs.hpp"
#include "ftcs.hpp"
#include "pde.hpp"
#include "black_scholes.hpp"

#include <cmath>
#include <stdexcept>

using Catch::Approx;
using namespace quant::pde;


// ---------------------------------------------------------------------------
// Coefficient sanity checks
// ---------------------------------------------------------------------------

TEST_CASE("theta_coeffs: theta=0 reduces to FTCS shape",
          "[theta_scheme][coeffs]") {
    const ThetaCoeffs c = theta_coeffs(0.0, 0.20, 0.05, 0.03, 0.001, 0.01);
    // With theta=0: all betas should be (0, 1, 0).
    REQUIRE(c.beta_minus == Approx(0.0).margin(1e-14));
    REQUIRE(c.beta_zero  == Approx(1.0).margin(1e-14));
    REQUIRE(c.beta_plus  == Approx(0.0).margin(1e-14));
}

TEST_CASE("theta_coeffs: theta=1 reduces to BTCS shape",
          "[theta_scheme][coeffs]") {
    const ThetaCoeffs c = theta_coeffs(1.0, 0.20, 0.05, 0.03, 0.001, 0.01);
    // With theta=1: all gammas should be (0, 1, 0).
    REQUIRE(c.gamma_minus == Approx(0.0).margin(1e-14));
    REQUIRE(c.gamma_zero  == Approx(1.0).margin(1e-14));
    REQUIRE(c.gamma_plus  == Approx(0.0).margin(1e-14));
}

TEST_CASE("theta_coeffs: invalid theta raises", "[theta_scheme][coeffs]") {
    REQUIRE_THROWS_AS(theta_coeffs(-0.1, 0.20, 0.05, 0.03, 0.001, 0.01),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(theta_coeffs( 1.1, 0.20, 0.05, 0.03, 0.001, 0.01),
                      std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Reproduction of BTCS via theta_march(theta=1)
// ---------------------------------------------------------------------------

TEST_CASE("theta_scheme: theta=1 reproduces BTCS to machine precision",
          "[theta_scheme][btcs-equiv]") {
    const double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const int N = 200, M = 800;

    const Grid g = build_grid(N, M, T, sigma, r, K);
    const auto V0    = call_initial_condition(g.xs, g.K);
    const auto bc_lo = call_boundary_lower(g);
    const auto bc_hi = call_boundary_upper(g);

    const auto V_via_theta = theta_march(g, V0, 1.0, bc_lo, bc_hi);

    const double x_0 = std::log(S / K);
    const int j = static_cast<int>(std::floor((x_0 - g.x_min) / g.dx));
    const double w = (x_0 - g.x_min) / g.dx - static_cast<double>(j);
    const double c_via_theta = (1.0 - w) * V_via_theta[
        static_cast<std::size_t>(j)] + w * V_via_theta[
        static_cast<std::size_t>(j) + 1];

    const double c_btcs = btcs_european_call(S, K, r, sigma, T, N, M);
    REQUIRE(c_via_theta == Approx(c_btcs).margin(1e-12));
}

// ---------------------------------------------------------------------------
// Reproduction of FTCS via theta_march(theta=0)
// ---------------------------------------------------------------------------

TEST_CASE("theta_scheme: theta=0 reproduces FTCS to machine precision",
          "[theta_scheme][ftcs-equiv]") {
    // Use a grid that satisfies CFL.
    const double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const int N = 100;
    const int M = ftcs_min_M_for_cfl(N, T, sigma, 4.0, 0.4);

    const Grid g = build_grid(N, M, T, sigma, r, K);
    const auto V0    = call_initial_condition(g.xs, g.K);
    const auto bc_lo = call_boundary_lower(g);
    const auto bc_hi = call_boundary_upper(g);

    const auto V_via_theta = theta_march(g, V0, 0.0, bc_lo, bc_hi);

    const double x_0 = std::log(S / K);
    const int j = static_cast<int>(std::floor((x_0 - g.x_min) / g.dx));
    const double w = (x_0 - g.x_min) / g.dx - static_cast<double>(j);
    const double c_via_theta = (1.0 - w) * V_via_theta[
        static_cast<std::size_t>(j)] + w * V_via_theta[
        static_cast<std::size_t>(j) + 1];

    const double c_ftcs = ftcs_european_call(S, K, r, sigma, T, N, M);
    REQUIRE(c_via_theta == Approx(c_ftcs).margin(1e-12));
}

// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------

TEST_CASE("theta_march: bc array length mismatch raises",
          "[theta_scheme][validation]") {
    const Grid g = build_grid(50, 20, 1.0, 0.20, 0.05, 100.0);
    const auto V0 = call_initial_condition(g.xs, g.K);
    std::vector<double> bc_lo(g.M + 1, 0.0);
    std::vector<double> bc_hi_short(g.M, 0.0);   // wrong size

    REQUIRE_THROWS_AS(
        theta_march(g, V0, 0.5, bc_lo, bc_hi_short),
        std::invalid_argument);
}
