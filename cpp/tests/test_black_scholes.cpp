// test_black_scholes.cpp
//
// Unit tests for the Black-Scholes pricer.
// Mirror of python/validate_black_scholes.py adapted to Catch2 idioms.
//
// Run from the build directory with:
//     ./quant_tests              # run all
//     ./quant_tests "[parity]"   # run only parity tests
//     ./quant_tests --list-tests # show all tests

#include "black_scholes.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <random>

using quant::call_price;
using quant::put_price;

// ---------------------------------------------------------------------------
// Tolerances. Matched to those used in validate_black_scholes.py.
// ---------------------------------------------------------------------------
namespace {
constexpr double TOL_HULL   = 1e-3;     // Hull rounds to 4 decimals.
constexpr double TOL_PARITY = 1e-12;    // Machine-precision identity.
constexpr double TOL_LIMIT  = 1e-6;     // Asymptotic checks.
constexpr double TOL_MONO   = 1e-12;    // Monotonicity FP noise.
}  // namespace


// ---------------------------------------------------------------------------
// Test 1: Hull textbook reference value
// ---------------------------------------------------------------------------
TEST_CASE("Hull example 15.6 reference values", "[bs][reference]") {
    // S=42, K=40, r=10%, sigma=20%, T=0.5 -> C=4.7594, P=0.8086.
    const double S = 42.0, K = 40.0, r = 0.10, sigma = 0.20, T = 0.5;

    const double C = call_price(S, K, r, sigma, T);
    const double P = put_price(S, K, r, sigma, T);

    SECTION("Call matches Hull to 4 decimals") {
        REQUIRE_THAT(C, Catch::Matchers::WithinAbs(4.7594, TOL_HULL));
    }

    SECTION("Put matches Hull to 4 decimals") {
        REQUIRE_THAT(P, Catch::Matchers::WithinAbs(0.8086, TOL_HULL));
    }

    SECTION("Put-call parity holds at this point") {
        const double residual = std::abs((C - P) - (S - K * std::exp(-r * T)));
        REQUIRE(residual < TOL_PARITY);
    }
}


// ---------------------------------------------------------------------------
// Test 2: Put-call parity over a random parameter grid
// ---------------------------------------------------------------------------
TEST_CASE("Put-call parity holds on random grid", "[bs][parity]") {
    // Model-free identity: C - P = S - K * exp(-r*T). Must hold to machine
    // precision regardless of the model. The seed is fixed so the test is
    // deterministic across runs.
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> uS    (50.0, 150.0);
    std::uniform_real_distribution<double> uK    (50.0, 150.0);
    std::uniform_real_distribution<double> ur    (0.01, 0.10);
    std::uniform_real_distribution<double> usigma(0.10, 0.50);
    std::uniform_real_distribution<double> uT    (0.1,  2.0);

    constexpr int N = 10000;
    double max_residual = 0.0;
    for (int i = 0; i < N; ++i) {
        const double S = uS(rng), K = uK(rng), r = ur(rng);
        const double sigma = usigma(rng), T = uT(rng);

        const double C = call_price(S, K, r, sigma, T);
        const double P = put_price(S, K, r, sigma, T);
        const double residual = std::abs((C - P) - (S - K * std::exp(-r * T)));
        if (residual > max_residual) max_residual = residual;
    }

    INFO("Max residual over " << N << " samples: " << max_residual);
    REQUIRE(max_residual < TOL_PARITY);
}


// ---------------------------------------------------------------------------
// Test 3: Limit T -> 0+
// ---------------------------------------------------------------------------
TEST_CASE("Prices converge to intrinsic value as T -> 0", "[bs][limit]") {
    // At T = 1e-8, ITM/OTM options are essentially at intrinsic value.
    // The ATM case is special: the time value decays only as O(sqrt(T)),
    // not exponentially. The leading-order asymptotic is
    //     C_ATM ~ S * sigma * sqrt(T / (2*pi))
    // which we use as the target rather than 0.
    const double T = 1e-8, r = 0.05, sigma = 0.20;

    SECTION("ITM call -> intrinsic value (S - K)") {
        const double price = call_price(110.0, 100.0, r, sigma, T);
        REQUIRE_THAT(price, Catch::Matchers::WithinAbs(10.0, TOL_LIMIT));
    }

    SECTION("OTM call -> 0") {
        const double price = call_price(90.0, 100.0, r, sigma, T);
        REQUIRE_THAT(price, Catch::Matchers::WithinAbs(0.0, TOL_LIMIT));
    }

    SECTION("ATM call -> S * sigma * sqrt(T / (2*pi))") {
        const double price = call_price(100.0, 100.0, r, sigma, T);
        const double target = 100.0 * sigma * std::sqrt(T / (2.0 * M_PI));
        // Looser tolerance: the asymptotic is leading-order only,
        // with O(T) corrections. At T=1e-8, those corrections are tiny but
        // nonzero, so we use a relative tolerance instead of TOL_LIMIT.
        REQUIRE_THAT(price, Catch::Matchers::WithinRel(target, 1e-3));
    }

    SECTION("ITM put -> intrinsic value (K - S)") {
        const double price = put_price(90.0, 100.0, r, sigma, T);
        REQUIRE_THAT(price, Catch::Matchers::WithinAbs(10.0, TOL_LIMIT));
    }

    SECTION("OTM put -> 0") {
        const double price = put_price(110.0, 100.0, r, sigma, T);
        REQUIRE_THAT(price, Catch::Matchers::WithinAbs(0.0, TOL_LIMIT));
    }
}


// ---------------------------------------------------------------------------
// Test 4: Deep ITM and OTM asymptotics
// ---------------------------------------------------------------------------
TEST_CASE("Deep ITM and OTM behaviour", "[bs][asymptotic]") {
    const double K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;

    SECTION("Deep ITM call -> S - K*exp(-rT)") {
        const double S = 1000.0;
        const double price = call_price(S, K, r, sigma, T);
        const double target = S - K * std::exp(-r * T);
        REQUIRE_THAT(price, Catch::Matchers::WithinAbs(target, TOL_LIMIT));
    }

    SECTION("Deep OTM call -> 0") {
        const double S = 10.0;
        const double price = call_price(S, K, r, sigma, T);
        REQUIRE_THAT(price, Catch::Matchers::WithinAbs(0.0, TOL_LIMIT));
    }
}


// ---------------------------------------------------------------------------
// Test 5: Monotonicities
// ---------------------------------------------------------------------------
TEST_CASE("Call price monotonicities", "[bs][monotonicity]") {
    // Each monotonicity probes a different partial derivative of the price.
    // A sign error in any of the formula components would break at least one.
    const double S0 = 100.0, K0 = 100.0, r0 = 0.05, sigma0 = 0.20, T0 = 1.0;
    constexpr int N = 50;

    auto monotone = [](const std::vector<double>& v, bool increasing) {
        for (size_t i = 1; i < v.size(); ++i) {
            const double diff = v[i] - v[i - 1];
            if (increasing && diff < -TOL_MONO) return false;
            if (!increasing && diff > TOL_MONO) return false;
        }
        return true;
    };

    SECTION("Increasing in S") {
        std::vector<double> prices(N);
        for (int i = 0; i < N; ++i) {
            const double S = 50.0 + (200.0 - 50.0) * i / (N - 1);
            prices[i] = call_price(S, K0, r0, sigma0, T0);
        }
        REQUIRE(monotone(prices, true));
    }

    SECTION("Decreasing in K") {
        std::vector<double> prices(N);
        for (int i = 0; i < N; ++i) {
            const double K = 50.0 + (200.0 - 50.0) * i / (N - 1);
            prices[i] = call_price(S0, K, r0, sigma0, T0);
        }
        REQUIRE(monotone(prices, false));
    }

    SECTION("Increasing in sigma (Vega > 0)") {
        std::vector<double> prices(N);
        for (int i = 0; i < N; ++i) {
            const double sigma = 0.05 + (0.80 - 0.05) * i / (N - 1);
            prices[i] = call_price(S0, K0, r0, sigma, T0);
        }
        REQUIRE(monotone(prices, true));
    }

    SECTION("Increasing in T (no dividends)") {
        std::vector<double> prices(N);
        for (int i = 0; i < N; ++i) {
            const double T = 0.01 + (5.0 - 0.01) * i / (N - 1);
            prices[i] = call_price(S0, K0, r0, sigma0, T);
        }
        REQUIRE(monotone(prices, true));
    }
}


// ---------------------------------------------------------------------------
// Test 6: No-arbitrage bounds
// ---------------------------------------------------------------------------
TEST_CASE("No-arbitrage bounds hold on random grid", "[bs][bounds]") {
    std::mt19937_64 rng(123);
    std::uniform_real_distribution<double> uS    (50.0, 150.0);
    std::uniform_real_distribution<double> uK    (50.0, 150.0);
    std::uniform_real_distribution<double> ur    (0.01, 0.10);
    std::uniform_real_distribution<double> usigma(0.10, 0.50);
    std::uniform_real_distribution<double> uT    (0.1,  2.0);

    constexpr int N = 10000;
    bool call_lower = true, call_upper = true;
    bool put_lower = true,  put_upper = true;

    for (int i = 0; i < N; ++i) {
        const double S = uS(rng), K = uK(rng), r = ur(rng);
        const double sigma = usigma(rng), T = uT(rng);
        const double C = call_price(S, K, r, sigma, T);
        const double P = put_price(S, K, r, sigma, T);
        const double pv_K = K * std::exp(-r * T);

        if (C < std::max(S - pv_K, 0.0) - TOL_PARITY) call_lower = false;
        if (C > S + TOL_PARITY)                       call_upper = false;
        if (P < std::max(pv_K - S, 0.0) - TOL_PARITY) put_lower  = false;
        if (P > pv_K + TOL_PARITY)                    put_upper  = false;
    }

    REQUIRE(call_lower);
    REQUIRE(call_upper);
    REQUIRE(put_lower);
    REQUIRE(put_upper);
}
