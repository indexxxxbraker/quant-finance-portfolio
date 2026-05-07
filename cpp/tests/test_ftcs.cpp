// test_ftcs.cpp
//
// Catch2 tests for the explicit FTCS pricer of Phase 3 Block 1.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "ftcs.hpp"
#include "pde.hpp"
#include "black_scholes.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

using Catch::Approx;
using namespace quant::pde;


// ---------------------------------------------------------------------------
// Cross-validation against Black-Scholes closed form
// ---------------------------------------------------------------------------

TEST_CASE("FTCS: cross-validation, Hull example 15.6",
          "[ftcs][cross-validation]") {
    // S = 42, K = 40, r = 0.10, sigma = 0.20, T = 0.5
    // Closed-form prices: call = 4.7594, put = 0.8086
    const double S = 42.0, K = 40.0, r = 0.10, sigma = 0.20, T = 0.5;
    const int N = 200;
    const int M = ftcs_min_M_for_cfl(N, T, sigma, 4.0, 0.4);

    const double c_ftcs = ftcs_european_call(S, K, r, sigma, T, N, M);
    const double p_ftcs = ftcs_european_put (S, K, r, sigma, T, N, M);
    const double c_bs   = quant::call_price(S, K, r, sigma, T);
    const double p_bs   = quant::put_price (S, K, r, sigma, T);

    REQUIRE(c_ftcs == Approx(c_bs).margin(5e-3));
    REQUIRE(p_ftcs == Approx(p_bs).margin(5e-3));
}

TEST_CASE("FTCS: cross-validation, ATM and OTM/ITM",
          "[ftcs][cross-validation]") {
    const double K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const int N = 200;
    const int M = ftcs_min_M_for_cfl(N, T, sigma, 4.0, 0.4);

    for (double S : {90.0, 100.0, 110.0}) {
        const double c_ftcs = ftcs_european_call(S, K, r, sigma, T, N, M);
        const double p_ftcs = ftcs_european_put (S, K, r, sigma, T, N, M);
        const double c_bs   = quant::call_price(S, K, r, sigma, T);
        const double p_bs   = quant::put_price (S, K, r, sigma, T);

        REQUIRE(c_ftcs == Approx(c_bs).margin(5e-3));
        REQUIRE(p_ftcs == Approx(p_bs).margin(5e-3));
    }
}

TEST_CASE("FTCS: cross-validation, varying volatility",
          "[ftcs][cross-validation]") {
    const double S = 100.0, K = 100.0, r = 0.05, T = 1.0;
    const int N = 200;

    for (double sigma : {0.10, 0.20, 0.30, 0.40}) {
        const int M = ftcs_min_M_for_cfl(N, T, sigma, 4.0, 0.4);
        const double c_ftcs = ftcs_european_call(S, K, r, sigma, T, N, M);
        const double c_bs   = quant::call_price(S, K, r, sigma, T);
        REQUIRE(c_ftcs == Approx(c_bs).margin(5e-3));
    }
}


// ---------------------------------------------------------------------------
// Convergence rate
// ---------------------------------------------------------------------------

TEST_CASE("FTCS: empirical convergence rate is ~ O(dx^2)",
          "[ftcs][convergence]") {
    const double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double c_bs = quant::call_price(S, K, r, sigma, T);

    // Three refinement levels: doubling N and quadrupling M.
    struct Level { int N; int M; };
    const Level levels[] = {
        {100,  200},
        {200,  800},
        {400, 3200},
    };

    double errors[3];
    for (std::size_t i = 0; i < 3; ++i) {
        const double c = ftcs_european_call(S, K, r, sigma, T,
                                            levels[i].N, levels[i].M);
        errors[i] = std::abs(c - c_bs);
    }

    // Each refinement should reduce the error by ~4. Accept ratios in
    // [3.0, 5.5] to absorb constant-prefactor variability.
    const double ratio_01 = errors[0] / errors[1];
    const double ratio_12 = errors[1] / errors[2];
    REQUIRE(ratio_01 >= 3.0);
    REQUIRE(ratio_01 <= 5.5);
    REQUIRE(ratio_12 >= 3.0);
    REQUIRE(ratio_12 <= 5.5);
}


// ---------------------------------------------------------------------------
// Put-call parity
// ---------------------------------------------------------------------------

TEST_CASE("FTCS: put-call parity holds across spots",
          "[ftcs][parity]") {
    const double K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const int N = 200;
    const int M = ftcs_min_M_for_cfl(N, T, sigma, 4.0, 0.4);

    for (double S : {85.0, 95.0, 100.0, 105.0, 115.0}) {
        const double c = ftcs_european_call(S, K, r, sigma, T, N, M);
        const double p = ftcs_european_put (S, K, r, sigma, T, N, M);
        const double rhs = S - K * std::exp(-r * T);
        // Both prices have O(dx^2) error; their difference can have
        // up to twice that, so use a 1e-2 tolerance.
        REQUIRE((c - p) == Approx(rhs).margin(1e-2));
    }
}


// ---------------------------------------------------------------------------
// CFL enforcement
// ---------------------------------------------------------------------------

TEST_CASE("FTCS: CFL violation throws std::invalid_argument",
          "[ftcs][cfl]") {
    // With sigma = 0.2, T = 1, n_sigma = 4, N = 100, dx = 0.016.
    // Stable boundary alpha = 0.5 needs M >= 78. M = 3 violates by far.
    REQUIRE_THROWS_AS(
        ftcs_european_call(100.0, 100.0, 0.05, 0.20, 1.0, 100, 3),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        ftcs_european_put(100.0, 100.0, 0.05, 0.20, 1.0, 100, 3),
        std::invalid_argument);
}

TEST_CASE("FTCS: ftcs_march bypasses CFL with validate_cfl=false",
          "[ftcs][cfl][diagnostic]") {
    // Reproduce the explosion documented in the manual personal
    // (Phase 3 numerical lesson 1). With these parameters alpha ~ 1.56,
    // and the final V vector should exceed any reasonable price ceiling
    // and exhibit the sawtooth signature.
    const double K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const Grid g = build_grid(100, 50, T, sigma, r, K);
    const auto V0 = call_initial_condition(g.xs, g.K);
    const auto bc_lo = call_boundary_lower(g);
    const auto bc_hi = call_boundary_upper(g);

    // With validate_cfl=true this would throw.
    const std::vector<double> V_unstable = ftcs_march(
        g, V0, bc_lo, bc_hi, /*validate_cfl=*/false);

    double max_abs = 0.0;
    int sign_changes = 0;
    for (std::size_t j = 0; j < V_unstable.size(); ++j) {
        max_abs = std::max(max_abs, std::abs(V_unstable[j]));
        if (j > 0
            && (V_unstable[j] >  0.0) != (V_unstable[j - 1] >  0.0)
            &&  V_unstable[j]      != 0.0
            &&  V_unstable[j - 1]  != 0.0) {
            ++sign_changes;
        }
    }
    REQUIRE(max_abs > 1e6);            // BS price is ~10
    REQUIRE(sign_changes > 10);        // sawtooth signature
}


// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------

TEST_CASE("FTCS: invalid spot throws", "[ftcs][validation]") {
    // S = 0
    REQUIRE_THROWS_AS(
        ftcs_european_call(0.0, 100.0, 0.05, 0.20, 1.0, 200, 800),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        ftcs_european_call(-10.0, 100.0, 0.05, 0.20, 1.0, 200, 800),
        std::invalid_argument);
    // Out of truncated domain: x_max = 4 * 0.2 * 1 = 0.8;
    // S_max = 100 * exp(0.8) ~ 222.55. S = 1000 is well above.
    REQUIRE_THROWS_AS(
        ftcs_european_call(1000.0, 100.0, 0.05, 0.20, 1.0, 200, 800),
        std::invalid_argument);
}

TEST_CASE("FTCS: ftcs_min_M_for_cfl rejects bad target_alpha",
          "[ftcs][validation]") {
    REQUIRE_THROWS_AS(ftcs_min_M_for_cfl(100, 1.0, 0.2, 4.0,  0.0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(ftcs_min_M_for_cfl(100, 1.0, 0.2, 4.0, -0.1),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(ftcs_min_M_for_cfl(100, 1.0, 0.2, 4.0,  0.6),
                      std::invalid_argument);
    // Boundary case: 0.5 is allowed.
    REQUIRE_NOTHROW(ftcs_min_M_for_cfl(100, 1.0, 0.2, 4.0, 0.5));
}
