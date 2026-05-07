// test_pde.cpp
//
// Catch2 tests for the PDE grid infrastructure of Phase 3 Block 0.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pde.hpp"

#include <cmath>
#include <stdexcept>

using Catch::Approx;
using namespace quant::pde;


// ---------------------------------------------------------------------------
// Grid construction
// ---------------------------------------------------------------------------

TEST_CASE("Grid construction: shapes and spacings", "[pde][grid]") {
    const int    N = 200;
    const int    M = 400;
    const double T = 1.0;
    const double sigma = 0.20;
    const double r = 0.05;
    const double K = 100.0;
    const double n_sigma = 4.0;

    const Grid g = build_grid(N, M, T, sigma, r, K, n_sigma);

    REQUIRE(g.N == N);
    REQUIRE(g.M == M);
    REQUIRE(static_cast<int>(g.xs.size())   == N + 1);
    REQUIRE(static_cast<int>(g.taus.size()) == M + 1);

    const double expected_half = n_sigma * sigma * std::sqrt(T);
    REQUIRE(g.x_min == Approx(-expected_half).margin(1e-14));
    REQUIRE(g.x_max == Approx(+expected_half).margin(1e-14));
    REQUIRE(g.dx    == Approx((g.x_max - g.x_min) / N).margin(1e-14));
    REQUIRE(g.dtau  == Approx(T / M).margin(1e-14));

    REQUIRE(g.xs.front()  == Approx(g.x_min).margin(1e-14));
    REQUIRE(g.xs.back()   == Approx(g.x_max).margin(1e-14));
    REQUIRE(g.taus.front()== Approx(0.0   ).margin(1e-14));
    REQUIRE(g.taus.back() == Approx(T     ).margin(1e-14));

    REQUIRE(g.mu() == Approx(r - 0.5 * sigma * sigma).margin(1e-14));
}

TEST_CASE("Grid construction: uniform spacing to machine precision",
          "[pde][grid]") {
    const Grid g = build_grid(200, 400, 1.0, 0.20, 0.05, 100.0);

    for (int j = 0; j < g.N; ++j) {
        const double diff = g.xs[j + 1] - g.xs[j];
        REQUIRE(diff == Approx(g.dx).margin(1e-14));
    }
    for (int n = 0; n < g.M; ++n) {
        const double diff = g.taus[n + 1] - g.taus[n];
        REQUIRE(diff == Approx(g.dtau).margin(1e-14));
    }
}

TEST_CASE("Grid construction: invalid inputs raise", "[pde][grid][validation]") {
    REQUIRE_THROWS_AS(build_grid( 1, 100, 1.0, 0.20, 0.05, 100.0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(build_grid(100,  0,  1.0, 0.20, 0.05, 100.0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(build_grid(100, 100, 0.0, 0.20, 0.05, 100.0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(build_grid(100, 100,-1.0, 0.20, 0.05, 100.0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(build_grid(100, 100, 1.0, 0.00, 0.05, 100.0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(build_grid(100, 100, 1.0,-0.10, 0.05, 100.0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(build_grid(100, 100, 1.0, 0.20, 0.05,   0.0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(build_grid(100, 100, 1.0, 0.20, 0.05,-100.0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(build_grid(100, 100, 1.0, 0.20, 0.05, 100.0, 0.0),
                      std::invalid_argument);
}


// ---------------------------------------------------------------------------
// Initial conditions
// ---------------------------------------------------------------------------

TEST_CASE("Initial conditions: call matches (S - K)^+ at every node",
          "[pde][initial-condition]") {
    const Grid g = build_grid(100, 100, 1.0, 0.20, 0.05, 100.0);
    const auto ic = call_initial_condition(g.xs, g.K);

    REQUIRE(ic.size() == g.xs.size());
    for (std::size_t j = 0; j < g.xs.size(); ++j) {
        const double S        = g.K * std::exp(g.xs[j]);
        const double expected = std::max(S - g.K, 0.0);
        REQUIRE(ic[j] == Approx(expected).margin(1e-12));
    }
}

TEST_CASE("Initial conditions: put matches (K - S)^+ at every node",
          "[pde][initial-condition]") {
    const Grid g = build_grid(100, 100, 1.0, 0.20, 0.05, 100.0);
    const auto ic = put_initial_condition(g.xs, g.K);

    for (std::size_t j = 0; j < g.xs.size(); ++j) {
        const double S        = g.K * std::exp(g.xs[j]);
        const double expected = std::max(g.K - S, 0.0);
        REQUIRE(ic[j] == Approx(expected).margin(1e-12));
    }
}

TEST_CASE("Initial conditions: zero at the at-the-money node",
          "[pde][initial-condition]") {
    // Use even N so the central node x = 0 is exactly representable.
    const Grid g = build_grid(100, 100, 1.0, 0.20, 0.05, 100.0);

    // Locate the node closest to x = 0; with our symmetric domain and
    // even N, the centre node has index N/2 and x exactly 0.
    const std::size_t j_atm = static_cast<std::size_t>(g.N / 2);
    REQUIRE(g.xs[j_atm] == Approx(0.0).margin(1e-14));

    const auto ic_call = call_initial_condition(g.xs, g.K);
    const auto ic_put  = put_initial_condition (g.xs, g.K);
    REQUIRE(ic_call[j_atm] == Approx(0.0).margin(1e-12));
    REQUIRE(ic_put [j_atm] == Approx(0.0).margin(1e-12));
}

TEST_CASE("Initial conditions: monotonicity in j",
          "[pde][initial-condition]") {
    const Grid g = build_grid(200, 100, 1.0, 0.20, 0.05, 100.0);
    const auto ic_call = call_initial_condition(g.xs, g.K);
    const auto ic_put  = put_initial_condition (g.xs, g.K);

    for (std::size_t j = 0; j + 1 < ic_call.size(); ++j) {
        REQUIRE(ic_call[j + 1] >= ic_call[j] - 1e-12);
        REQUIRE(ic_put [j + 1] <= ic_put [j] + 1e-12);
    }
}


// ---------------------------------------------------------------------------
// Boundary conditions
// ---------------------------------------------------------------------------

TEST_CASE("Boundary conditions: call lower and put upper are zero",
          "[pde][boundary]") {
    const Grid g = build_grid(100, 200, 1.0, 0.20, 0.05, 100.0);

    const auto bc_call_lo = call_boundary_lower(g);
    const auto bc_put_hi  = put_boundary_upper (g);

    REQUIRE(bc_call_lo.size() == static_cast<std::size_t>(g.M + 1));
    REQUIRE(bc_put_hi.size()  == static_cast<std::size_t>(g.M + 1));

    for (std::size_t n = 0; n < bc_call_lo.size(); ++n) {
        REQUIRE(bc_call_lo[n] == 0.0);
        REQUIRE(bc_put_hi [n] == 0.0);
    }
}

TEST_CASE("Boundary conditions: call upper matches BS asymptotic",
          "[pde][boundary]") {
    const double K = 100.0;
    const double r = 0.05;
    const double T = 1.0;
    const Grid g = build_grid(100, 200, T, 0.20, r, K);
    const auto bc_call_hi = call_boundary_upper(g);

    // Endpoint values:
    //   At tau = 0: K * (e^{x_max} - 1)
    //   At tau = T: K * (e^{x_max} - e^{-rT})
    const double e_xmax = std::exp(g.x_max);
    REQUIRE(bc_call_hi.front() ==
            Approx(K * (e_xmax - 1.0)).margin(1e-12));
    REQUIRE(bc_call_hi.back() ==
            Approx(K * (e_xmax - std::exp(-r * T))).margin(1e-12));

    // Pointwise formula at every level.
    for (std::size_t n = 0; n < bc_call_hi.size(); ++n) {
        const double expected = K * (e_xmax - std::exp(-r * g.taus[n]));
        REQUIRE(bc_call_hi[n] == Approx(expected).margin(1e-12));
    }
}

TEST_CASE("Boundary conditions: put lower matches BS asymptotic",
          "[pde][boundary]") {
    const double K = 100.0;
    const double r = 0.05;
    const double T = 1.0;
    const Grid g = build_grid(100, 200, T, 0.20, r, K);
    const auto bc_put_lo = put_boundary_lower(g);

    const double e_xmin = std::exp(g.x_min);
    REQUIRE(bc_put_lo.front() ==
            Approx(K * (1.0 - e_xmin)).margin(1e-12));
    REQUIRE(bc_put_lo.back() ==
            Approx(K * (std::exp(-r * T) - e_xmin)).margin(1e-12));

    for (std::size_t n = 0; n < bc_put_lo.size(); ++n) {
        const double expected =
            K * (std::exp(-r * g.taus[n]) - e_xmin);
        REQUIRE(bc_put_lo[n] == Approx(expected).margin(1e-12));
    }
}

TEST_CASE("Boundary conditions: corner consistency BC(tau=0) = IC at endpoint",
          "[pde][boundary][corner]") {
    const Grid g = build_grid(100, 200, 1.0, 0.20, 0.05, 100.0);
    const auto ic_call = call_initial_condition(g.xs, g.K);
    const auto ic_put  = put_initial_condition (g.xs, g.K);

    const auto bc_call_lo = call_boundary_lower(g);
    const auto bc_call_hi = call_boundary_upper(g);
    const auto bc_put_lo  = put_boundary_lower (g);
    const auto bc_put_hi  = put_boundary_upper (g);

    REQUIRE(bc_call_lo.front() == Approx(ic_call.front()).margin(1e-12));
    REQUIRE(bc_call_hi.front() == Approx(ic_call.back ()).margin(1e-12));
    REQUIRE(bc_put_lo .front() == Approx(ic_put .front()).margin(1e-12));
    REQUIRE(bc_put_hi .front() == Approx(ic_put .back ()).margin(1e-12));
}

TEST_CASE("Boundary conditions: monotonicity in tau (positive r)",
          "[pde][boundary]") {
    const Grid g = build_grid(100, 200, 1.0, 0.20, 0.05, 100.0);
    const auto bc_call_hi = call_boundary_upper(g);
    const auto bc_put_lo  = put_boundary_lower (g);

    // Call upper boundary K*(e^{x_max} - e^{-r tau}) is increasing in tau.
    // Put  lower boundary K*(e^{-r tau} - e^{x_min}) is decreasing in tau.
    for (std::size_t n = 0; n + 1 < bc_call_hi.size(); ++n) {
        REQUIRE(bc_call_hi[n + 1] >= bc_call_hi[n] - 1e-14);
        REQUIRE(bc_put_lo [n + 1] <= bc_put_lo [n] + 1e-14);
    }
}


// ---------------------------------------------------------------------------
// Stability numbers
// ---------------------------------------------------------------------------

TEST_CASE("Stability numbers: closed-form check", "[pde][stability]") {
    const double sigma = 0.20;
    const double dx = 0.008;
    const double dtau = 0.0025;
    const double r = 0.05;
    const double mu = r - 0.5 * sigma * sigma;   // = 0.03

    REQUIRE(fourier_number(sigma, dtau, dx) ==
            Approx(0.5 * sigma * sigma * dtau / (dx * dx)).margin(1e-14));
    REQUIRE(courant_number(mu, dtau, dx) ==
            Approx(0.5 * std::abs(mu) * dtau / dx).margin(1e-14));
    REQUIRE(courant_number(-mu, dtau, dx) ==
            courant_number(mu, dtau, dx));   // symmetry in sign of mu
}

TEST_CASE("Stability numbers: explicit stability boundary",
          "[pde][stability][cfl]") {
    REQUIRE(is_explicit_stable(0.5)            == true );
    REQUIRE(is_explicit_stable(0.5 + 1e-12)    == false);
    REQUIRE(is_explicit_stable(0.49)           == true );
    REQUIRE(is_explicit_stable(1.0)            == false);
    REQUIRE(is_explicit_stable(0.0)            == true );
}
