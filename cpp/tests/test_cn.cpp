// test_cn.cpp
//
// Catch2 tests for the Crank-Nicolson pricer of Phase 3 Block 3.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "cn.hpp"
#include "black_scholes.hpp"

#include <cmath>
#include <stdexcept>

using Catch::Approx;
using namespace quant::pde;


// ---------------------------------------------------------------------------
// Cross-validation
// ---------------------------------------------------------------------------

TEST_CASE("CN: Hull example 15.6", "[cn][cross-validation]") {
    const double S = 42.0, K = 40.0, r = 0.10, sigma = 0.20, T = 0.5;
    const int N = 400, M = 200;

    const double c_cn = cn_european_call(S, K, r, sigma, T, N, M);
    const double p_cn = cn_european_put (S, K, r, sigma, T, N, M);
    const double c_bs = quant::call_price(S, K, r, sigma, T);
    const double p_bs = quant::put_price (S, K, r, sigma, T);

    REQUIRE(c_cn == Approx(c_bs).margin(5e-3));
    REQUIRE(p_cn == Approx(p_bs).margin(5e-3));
}

TEST_CASE("CN: ATM and OTM/ITM, varying volatility",
          "[cn][cross-validation]") {
    const double K = 100.0, r = 0.05, T = 1.0;
    const int N = 400, M = 200;

    for (double sigma : {0.10, 0.20, 0.30}) {
        for (double S : {90.0, 100.0, 110.0}) {
            const double c = cn_european_call(S, K, r, sigma, T, N, M);
            const double p = cn_european_put (S, K, r, sigma, T, N, M);
            REQUIRE(c == Approx(quant::call_price(S, K, r, sigma, T))
                          .margin(5e-3));
            REQUIRE(p == Approx(quant::put_price (S, K, r, sigma, T))
                          .margin(5e-3));
        }
    }
}

// ---------------------------------------------------------------------------
// Quadratic-time convergence with Rannacher
// ---------------------------------------------------------------------------

TEST_CASE("CN with Rannacher: quadratic convergence in time",
          "[cn][convergence]") {
    const double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double c_bs = quant::call_price(S, K, r, sigma, T);
    const int N = 800;
    const int Ms[] = {10, 20, 40};

    double errors[3];
    for (std::size_t i = 0; i < 3; ++i) {
        errors[i] = std::abs(
            cn_european_call(S, K, r, sigma, T, N, Ms[i]) - c_bs);
    }
    // Both ratios should be > 3.0, conclusively above BTCS first-order.
    REQUIRE(errors[0] / errors[1] > 3.0);
    REQUIRE(errors[1] / errors[2] > 3.0);
}

// ---------------------------------------------------------------------------
// Kink artefact: vanilla CN has degraded convergence
// ---------------------------------------------------------------------------

TEST_CASE("CN without Rannacher: kink artefact degrades convergence",
          "[cn][kink][diagnostic]") {
    const double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double c_bs = quant::call_price(S, K, r, sigma, T);
    const int N = 800;
    const int Ms[] = {10, 20, 40};

    double errors[3];
    for (std::size_t i = 0; i < 3; ++i) {
        errors[i] = std::abs(
            cn_european_call(S, K, r, sigma, T, N, Ms[i],
                             4.0, /*rannacher_steps=*/0) - c_bs);
    }
    // Without Rannacher the ratios should be < 3.0, distinguishably
    // worse than the Rannacher version above.
    REQUIRE(errors[0] / errors[1] < 3.0);
    REQUIRE(errors[1] / errors[2] < 3.0);
    // Vanilla CN error at M=10 should be much larger than Rannacher's.
    REQUIRE(errors[0] > 5.0 * 8e-3);   // Rannacher gives ~8e-3 at M=10.
}

// ---------------------------------------------------------------------------
// Put-call parity
// ---------------------------------------------------------------------------

TEST_CASE("CN: put-call parity", "[cn][parity]") {
    const double K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const int N = 400, M = 200;
    for (double S : {85.0, 95.0, 100.0, 105.0, 115.0}) {
        const double c = cn_european_call(S, K, r, sigma, T, N, M);
        const double p = cn_european_put (S, K, r, sigma, T, N, M);
        const double rhs = S - K * std::exp(-r * T);
        REQUIRE((c - p) == Approx(rhs).margin(1e-2));
    }
}

// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------

TEST_CASE("CN: invalid inputs raise", "[cn][validation]") {
    REQUIRE_THROWS_AS(
        cn_european_call(0.0, 100.0, 0.05, 0.20, 1.0, 200, 200),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cn_european_call(1000.0, 100.0, 0.05, 0.20, 1.0, 200, 200),
        std::invalid_argument);
    // rannacher_steps < 0
    REQUIRE_THROWS_AS(
        cn_european_call(100.0, 100.0, 0.05, 0.20, 1.0, 200, 200, 4.0, -1),
        std::invalid_argument);
    // rannacher_steps > M
    REQUIRE_THROWS_AS(
        cn_european_call(100.0, 100.0, 0.05, 0.20, 1.0, 200, 2, 4.0, 5),
        std::invalid_argument);
}
