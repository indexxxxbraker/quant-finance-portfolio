// test_btcs.cpp
//
// Catch2 tests for the implicit BTCS pricer of Phase 3 Block 2.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "btcs.hpp"
#include "black_scholes.hpp"

#include <cmath>
#include <stdexcept>

using Catch::Approx;
using namespace quant::pde;


// ---------------------------------------------------------------------------
// Cross-validation against Black-Scholes closed form
// ---------------------------------------------------------------------------

TEST_CASE("BTCS: cross-validation, Hull example 15.6",
          "[btcs][cross-validation]") {
    const double S = 42.0, K = 40.0, r = 0.10, sigma = 0.20, T = 0.5;
    const int N = 200, M = 800;

    const double c_btcs = btcs_european_call(S, K, r, sigma, T, N, M);
    const double p_btcs = btcs_european_put (S, K, r, sigma, T, N, M);
    const double c_bs   = quant::call_price(S, K, r, sigma, T);
    const double p_bs   = quant::put_price (S, K, r, sigma, T);

    REQUIRE(c_btcs == Approx(c_bs).margin(5e-3));
    REQUIRE(p_btcs == Approx(p_bs).margin(5e-3));
}

TEST_CASE("BTCS: cross-validation, ATM and OTM/ITM",
          "[btcs][cross-validation]") {
    const double K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const int N = 200, M = 800;

    for (double S : {90.0, 100.0, 110.0}) {
        const double c_btcs = btcs_european_call(S, K, r, sigma, T, N, M);
        const double p_btcs = btcs_european_put (S, K, r, sigma, T, N, M);
        const double c_bs   = quant::call_price(S, K, r, sigma, T);
        const double p_bs   = quant::put_price (S, K, r, sigma, T);
        REQUIRE(c_btcs == Approx(c_bs).margin(5e-3));
        REQUIRE(p_btcs == Approx(p_bs).margin(5e-3));
    }
}


// ---------------------------------------------------------------------------
// Convergence
// ---------------------------------------------------------------------------

TEST_CASE("BTCS: quadratic convergence under balanced refinement",
          "[btcs][convergence]") {
    const double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double c_bs = quant::call_price(S, K, r, sigma, T);

    struct Level { int N; int M; };
    const Level levels[] = {{100, 200}, {200, 800}, {400, 3200}};

    double errors[3];
    for (std::size_t i = 0; i < 3; ++i) {
        const double c =
            btcs_european_call(S, K, r, sigma, T, levels[i].N, levels[i].M);
        errors[i] = std::abs(c - c_bs);
    }

    REQUIRE(errors[0] / errors[1] >= 3.0);
    REQUIRE(errors[0] / errors[1] <= 5.5);
    REQUIRE(errors[1] / errors[2] >= 3.0);
    REQUIRE(errors[1] / errors[2] <= 5.5);
}

TEST_CASE("BTCS: first-order in time at fixed N",
          "[btcs][convergence]") {
    // Large N so the spatial error is below the temporal error.
    const double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double c_bs = quant::call_price(S, K, r, sigma, T);

    const int N = 800;
    const int Ms[] = {10, 20, 40};
    double errors[3];
    for (std::size_t i = 0; i < 3; ++i) {
        errors[i] = std::abs(
            btcs_european_call(S, K, r, sigma, T, N, Ms[i]) - c_bs);
    }
    REQUIRE(errors[0] / errors[1] >= 1.5);
    REQUIRE(errors[0] / errors[1] <= 2.5);
    REQUIRE(errors[1] / errors[2] >= 1.5);
    REQUIRE(errors[1] / errors[2] <= 2.5);
}


// ---------------------------------------------------------------------------
// Put-call parity
// ---------------------------------------------------------------------------

TEST_CASE("BTCS: put-call parity holds across spots",
          "[btcs][parity]") {
    const double K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const int N = 200, M = 800;

    for (double S : {85.0, 95.0, 100.0, 105.0, 115.0}) {
        const double c = btcs_european_call(S, K, r, sigma, T, N, M);
        const double p = btcs_european_put (S, K, r, sigma, T, N, M);
        const double rhs = S - K * std::exp(-r * T);
        REQUIRE((c - p) == Approx(rhs).margin(1e-2));
    }
}


// ---------------------------------------------------------------------------
// Unconditional stability
// ---------------------------------------------------------------------------

TEST_CASE("BTCS: large-dtau test produces a finite, sensible price",
          "[btcs][stability]") {
    // M=10 gives dtau=0.1, well above the FTCS CFL bound (FTCS would
    // explode catastrophically here). BTCS produces something useful.
    const double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double c_bs = quant::call_price(S, K, r, sigma, T);

    const double c_btcs = btcs_european_call(S, K, r, sigma, T, 200, 10);
    REQUIRE(std::isfinite(c_btcs));
    REQUIRE(std::abs(c_btcs - c_bs) / c_bs < 0.05);
}

TEST_CASE("BTCS: extreme-dtau test does not explode",
          "[btcs][stability]") {
    // M=2 gives dtau=0.5; an absurd choice. The price will be far
    // from accurate but must be finite.
    const double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double c_btcs = btcs_european_call(S, K, r, sigma, T, 200, 2);
    REQUIRE(std::isfinite(c_btcs));
}


// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------

TEST_CASE("BTCS: invalid spot raises", "[btcs][validation]") {
    REQUIRE_THROWS_AS(
        btcs_european_call(0.0, 100.0, 0.05, 0.20, 1.0, 200, 800),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        btcs_european_call(-10.0, 100.0, 0.05, 0.20, 1.0, 200, 800),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        btcs_european_call(1000.0, 100.0, 0.05, 0.20, 1.0, 200, 800),
        std::invalid_argument);
}

TEST_CASE("BTCS: M = 1 is valid (no CFL constraint)",
          "[btcs][validation]") {
    // Single time step. BTCS does not refuse this, in contrast to
    // FTCS which would reject it as CFL-violating.
    const double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    REQUIRE_NOTHROW(btcs_european_call(S, K, r, sigma, T, 200, 1));
}
