// test_black_scholes.cpp
//
// Unit tests for the Black-Scholes pricer and Greeks.
//
// Run from the build directory with:
//     ./quant_tests              # run all
//     ./quant_tests "[parity]"   # run only parity tests
//     ./quant_tests "[greek]"    # run only Greeks tests
//     ./quant_tests --list-tests # show all tests

#include "black_scholes.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <random>
#include <vector>

using quant::call_price;
using quant::put_price;
using quant::call_delta;
using quant::put_delta;
using quant::gamma;
using quant::vega;
using quant::call_theta;
using quant::put_theta;
using quant::call_rho;
using quant::put_rho;


// ---------------------------------------------------------------------------
// Tolerances
// ---------------------------------------------------------------------------
namespace {
constexpr double TOL_HULL     = 1e-3;     // Hull rounds to 4 decimals.
constexpr double TOL_PARITY   = 1e-12;    // Machine-precision identity.
constexpr double TOL_LIMIT    = 1e-6;     // Asymptotic checks.
constexpr double TOL_MONO     = 1e-12;    // Monotonicity FP noise.
constexpr double TOL_FD       = 1e-6;     // Finite-difference accuracy.
constexpr double TOL_PDE      = 1e-10;    // BS PDE residual.
constexpr double TOL_IDENTITY = 1e-12;    // Vega-Gamma identity.
}  // namespace


// ===========================================================================
// PRICE TESTS (from algorithm 1)
// ===========================================================================

// ---------------------------------------------------------------------------
// Test: Hull textbook reference value
// ---------------------------------------------------------------------------
TEST_CASE("Hull example 15.6 reference values", "[bs][reference]") {
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
// Test: put-call parity over a random parameter grid
// ---------------------------------------------------------------------------
TEST_CASE("Put-call parity holds on random grid", "[bs][parity]") {
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
// Test: limit T -> 0+
// ---------------------------------------------------------------------------
TEST_CASE("Prices converge to intrinsic value as T -> 0", "[bs][limit]") {
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
// Test: deep ITM and OTM asymptotics
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
// Test: monotonicities of the call price
// ---------------------------------------------------------------------------
TEST_CASE("Call price monotonicities", "[bs][monotonicity]") {
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
// Test: no-arbitrage bounds
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


// ===========================================================================
// GREEKS TESTS (algorithm 2)
// ===========================================================================

// ---------------------------------------------------------------------------
// Test: finite-difference verification of all Greeks
// ---------------------------------------------------------------------------
TEST_CASE("Greeks match central finite differences", "[greek][fd]") {
    const double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double h = 1e-5;

    SECTION("Call Delta = (C(S+h) - C(S-h)) / (2h)") {
        const double analytic = call_delta(S, K, r, sigma, T);
        const double numerical =
            (call_price(S + h, K, r, sigma, T) - call_price(S - h, K, r, sigma, T))
            / (2.0 * h);
        REQUIRE_THAT(analytic, Catch::Matchers::WithinAbs(numerical, TOL_FD));
    }

    SECTION("Put Delta = (P(S+h) - P(S-h)) / (2h)") {
        const double analytic = put_delta(S, K, r, sigma, T);
        const double numerical =
            (put_price(S + h, K, r, sigma, T) - put_price(S - h, K, r, sigma, T))
            / (2.0 * h);
        REQUIRE_THAT(analytic, Catch::Matchers::WithinAbs(numerical, TOL_FD));
    }

    SECTION("Gamma = (C(S+h) - 2*C(S) + C(S-h)) / h^2") {
        const double analytic = gamma(S, K, r, sigma, T);
        const double numerical =
            (call_price(S + h, K, r, sigma, T)
             - 2.0 * call_price(S, K, r, sigma, T)
             + call_price(S - h, K, r, sigma, T))
            / (h * h);
        // Looser tolerance: second-derivative central differences amplify
        // round-off by 1/h^2, giving an irreducible error of order 1e-4.
        REQUIRE_THAT(analytic, Catch::Matchers::WithinAbs(numerical, 1e-4));
    }

    SECTION("Vega = (C(sigma+h) - C(sigma-h)) / (2h)") {
        const double analytic = vega(S, K, r, sigma, T);
        const double numerical =
            (call_price(S, K, r, sigma + h, T) - call_price(S, K, r, sigma - h, T))
            / (2.0 * h);
        REQUIRE_THAT(analytic, Catch::Matchers::WithinAbs(numerical, TOL_FD));
    }

    SECTION("Call Theta = -dC/dT (sign flip)") {
        // Theta = dC/dt and dC/dt = -dC/dT, so we bump T and negate.
        const double analytic = call_theta(S, K, r, sigma, T);
        const double numerical = -(
            call_price(S, K, r, sigma, T + h) - call_price(S, K, r, sigma, T - h))
            / (2.0 * h);
        REQUIRE_THAT(analytic, Catch::Matchers::WithinAbs(numerical, TOL_FD));
    }

    SECTION("Put Theta = -dP/dT") {
        const double analytic = put_theta(S, K, r, sigma, T);
        const double numerical = -(
            put_price(S, K, r, sigma, T + h) - put_price(S, K, r, sigma, T - h))
            / (2.0 * h);
        REQUIRE_THAT(analytic, Catch::Matchers::WithinAbs(numerical, TOL_FD));
    }

    SECTION("Call Rho = (C(r+h) - C(r-h)) / (2h)") {
        const double analytic = call_rho(S, K, r, sigma, T);
        const double numerical =
            (call_price(S, K, r + h, sigma, T) - call_price(S, K, r - h, sigma, T))
            / (2.0 * h);
        REQUIRE_THAT(analytic, Catch::Matchers::WithinAbs(numerical, TOL_FD));
    }

    SECTION("Put Rho = (P(r+h) - P(r-h)) / (2h)") {
        const double analytic = put_rho(S, K, r, sigma, T);
        const double numerical =
            (put_price(S, K, r + h, sigma, T) - put_price(S, K, r - h, sigma, T))
            / (2.0 * h);
        REQUIRE_THAT(analytic, Catch::Matchers::WithinAbs(numerical, TOL_FD));
    }
}


// ---------------------------------------------------------------------------
// Test: the Black-Scholes PDE residual
// ---------------------------------------------------------------------------
TEST_CASE("Greeks satisfy the Black-Scholes PDE", "[greek][pde]") {
    // BS PDE: Theta + 0.5 * sigma^2 * S^2 * Gamma + r * S * Delta - r * C = 0.
    // Must hold to machine precision over the entire parameter space, for both
    // calls and puts (the PDE is the same; it constrains the price function,
    // not the option type).
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> uS    (50.0, 150.0);
    std::uniform_real_distribution<double> uK    (50.0, 150.0);
    std::uniform_real_distribution<double> ur    (0.01, 0.10);
    std::uniform_real_distribution<double> usigma(0.10, 0.50);
    std::uniform_real_distribution<double> uT    (0.1,  2.0);

    constexpr int N = 10000;
    double max_call = 0.0, max_put = 0.0;
    for (int i = 0; i < N; ++i) {
        const double S = uS(rng), K = uK(rng), r = ur(rng);
        const double sigma = usigma(rng), T = uT(rng);

        const double Ga = gamma(S, K, r, sigma, T);

        const double C  = call_price(S, K, r, sigma, T);
        const double Th = call_theta(S, K, r, sigma, T);
        const double De = call_delta(S, K, r, sigma, T);
        const double res_call = std::abs(
            Th + 0.5 * sigma * sigma * S * S * Ga + r * S * De - r * C);
        if (res_call > max_call) max_call = res_call;

        const double P   = put_price(S, K, r, sigma, T);
        const double ThP = put_theta(S, K, r, sigma, T);
        const double DeP = put_delta(S, K, r, sigma, T);
        const double res_put = std::abs(
            ThP + 0.5 * sigma * sigma * S * S * Ga + r * S * DeP - r * P);
        if (res_put > max_put) max_put = res_put;
    }

    INFO("Max PDE residual (call): " << max_call);
    INFO("Max PDE residual (put):  " << max_put);
    REQUIRE(max_call < TOL_PDE);
    REQUIRE(max_put  < TOL_PDE);
}


// ---------------------------------------------------------------------------
// Test: the Vega-Gamma identity
// ---------------------------------------------------------------------------
TEST_CASE("Vega-Gamma identity holds", "[greek][identity]") {
    // Algebraic identity: Vega = S^2 * sigma * T * Gamma. Holds at machine
    // precision for any (S, K, r, sigma, T).
    std::mt19937_64 rng(99);
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

        const double V  = vega (S, K, r, sigma, T);
        const double Ga = gamma(S, K, r, sigma, T);
        const double res = std::abs(V - S * S * sigma * T * Ga);
        if (res > max_residual) max_residual = res;
    }

    INFO("Max identity residual: " << max_residual);
    REQUIRE(max_residual < TOL_IDENTITY);
}
