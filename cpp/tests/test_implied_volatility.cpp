// test_implied_volatility.cpp
//
// Unit tests for the implied-volatility solver.
// Mirror of python/validate_implied_volatility.py adapted to Catch2 idioms.
//
// Run from the build directory:
//     ./quant_tests "[iv]"           # run only IV tests
//     ./quant_tests "[iv][roundtrip]"
//     ./quant_tests "[iv][edge]"

#include "black_scholes.hpp"
#include "implied_volatility.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <random>

using quant::call_price;
using quant::vega;
using quant::implied_volatility;
using quant::NoArbitrageBoundsViolation;


// ---------------------------------------------------------------------------
// Tolerances and constants
// ---------------------------------------------------------------------------
namespace {
constexpr double TOL_ROUNDTRIP   = 1e-5;   // Saturated by double-precision.
constexpr double TOL_EDGE        = 1e-4;   // Looser, for safeguard fallbacks.
constexpr double VEGA_THRESHOLD  = 1e-3;   // Below this, inversion not resolvable.

/// Skip ill-conditioned points where Vega is too small for the inversion
/// to be numerically resolvable. See validate_implied_volatility.py for
/// the derivation: |Delta_sigma| ~ |Delta_C| / Vega.
bool well_conditioned(double S, double K, double r, double sigma, double T) {
    return vega(S, K, r, sigma, T) > VEGA_THRESHOLD;
}
}  // namespace


// ---------------------------------------------------------------------------
// Test 1: Round-trip on a random parameter grid
// ---------------------------------------------------------------------------
TEST_CASE("Implied volatility round-trip on random grid", "[iv][roundtrip]") {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> uS    (80.0, 120.0);
    std::uniform_real_distribution<double> umon  (0.7, 1.3);    // K/S ratio
    std::uniform_real_distribution<double> ur    (0.01, 0.10);
    std::uniform_real_distribution<double> usig  (0.05, 0.80);
    std::uniform_real_distribution<double> uT    (0.1, 2.0);

    constexpr int N = 10000;
    double max_err = 0.0;
    int n_tested = 0;
    int n_failed = 0;

    for (int i = 0; i < N; ++i) {
        const double S      = uS(rng);
        const double K      = S * umon(rng);
        const double r      = ur(rng);
        const double sigma  = usig(rng);
        const double T      = uT(rng);

        if (!well_conditioned(S, K, r, sigma, T)) continue;

        const double C = call_price(S, K, r, sigma, T);
        try {
            const double iv = implied_volatility(C, S, K, r, T);
            const double err = std::abs(iv - sigma);
            if (err > max_err) max_err = err;
            ++n_tested;
        } catch (...) {
            ++n_failed;
        }
    }

    INFO("Tested: " << n_tested << " / " << N);
    INFO("Failures: " << n_failed);
    INFO("Max recovery error: " << max_err);
    REQUIRE(n_failed == 0);
    REQUIRE(max_err < TOL_ROUNDTRIP);
}


// ---------------------------------------------------------------------------
// Test 2: Edge cases (Newton-unsafe regimes)
// ---------------------------------------------------------------------------
TEST_CASE("Implied volatility handles edge cases", "[iv][edge]") {
    // Each section probes a regime where Newton becomes unsafe and the
    // bisection fallback must take over. The algorithm should still return
    // a correct sigma_iv.

    SECTION("Moderate ITM") {
        const double S = 130.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
        const double C = call_price(S, K, r, sigma, T);
        const double iv = implied_volatility(C, S, K, r, T);
        REQUIRE_THAT(iv, Catch::Matchers::WithinAbs(sigma, TOL_EDGE));
    }

    SECTION("Moderate OTM") {
        const double S = 80.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
        const double C = call_price(S, K, r, sigma, T);
        const double iv = implied_volatility(C, S, K, r, T);
        REQUIRE_THAT(iv, Catch::Matchers::WithinAbs(sigma, TOL_EDGE));
    }

    SECTION("Short expiry ATM") {
        const double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 0.001;
        const double C = call_price(S, K, r, sigma, T);
        const double iv = implied_volatility(C, S, K, r, T);
        REQUIRE_THAT(iv, Catch::Matchers::WithinAbs(sigma, TOL_EDGE));
    }

    SECTION("Short expiry OTM") {
        const double S = 100.0, K = 105.0, r = 0.05, sigma = 0.20, T = 0.01;
        const double C = call_price(S, K, r, sigma, T);
        const double iv = implied_volatility(C, S, K, r, T);
        REQUIRE_THAT(iv, Catch::Matchers::WithinAbs(sigma, TOL_EDGE));
    }

    SECTION("Very low sigma") {
        const double S = 100.0, K = 100.0, r = 0.05, sigma = 0.01, T = 1.0;
        const double C = call_price(S, K, r, sigma, T);
        const double iv = implied_volatility(C, S, K, r, T);
        REQUIRE_THAT(iv, Catch::Matchers::WithinAbs(sigma, TOL_EDGE));
    }

    SECTION("Very high sigma") {
        const double S = 100.0, K = 100.0, r = 0.05, sigma = 2.00, T = 1.0;
        const double C = call_price(S, K, r, sigma, T);
        const double iv = implied_volatility(C, S, K, r, T);
        REQUIRE_THAT(iv, Catch::Matchers::WithinAbs(sigma, TOL_EDGE));
    }
}


// ---------------------------------------------------------------------------
// Test 3: Bounds violations
// ---------------------------------------------------------------------------
TEST_CASE("Implied volatility throws on bounds violations", "[iv][bounds]") {
    // Prices outside the no-arbitrage range have no implied volatility
    // (Theorem 2.5 in implied_volatility.tex). The solver must throw
    // NoArbitrageBoundsViolation.
    const double S = 100.0, K = 100.0, r = 0.05, T = 1.0;
    const double intrinsic_fwd = std::max(S - K * std::exp(-r * T), 0.0);

    SECTION("Below lower bound") {
        REQUIRE_THROWS_AS(
            implied_volatility(intrinsic_fwd / 2.0, S, K, r, T),
            NoArbitrageBoundsViolation
        );
    }

    SECTION("At lower bound (strict inequality)") {
        REQUIRE_THROWS_AS(
            implied_volatility(intrinsic_fwd, S, K, r, T),
            NoArbitrageBoundsViolation
        );
    }

    SECTION("Above upper bound") {
        REQUIRE_THROWS_AS(
            implied_volatility(S * 1.1, S, K, r, T),
            NoArbitrageBoundsViolation
        );
    }

    SECTION("At upper bound (strict inequality)") {
        REQUIRE_THROWS_AS(
            implied_volatility(S, S, K, r, T),
            NoArbitrageBoundsViolation
        );
    }

    SECTION("Negative price") {
        REQUIRE_THROWS_AS(
            implied_volatility(-1.0, S, K, r, T),
            NoArbitrageBoundsViolation
        );
    }
}


// ---------------------------------------------------------------------------
// Test 4: Textbook reference
// ---------------------------------------------------------------------------
TEST_CASE("Implied volatility matches Hull Example 19.6", "[iv][reference]") {
    // Hull (10th ed.), Example 19.6: S=21, K=20, r=10%, T=0.25, C=1.875.
    // Reported implied volatility ~0.235.
    const double S = 21.0, K = 20.0, r = 0.10, T = 0.25;
    const double C = 1.875;
    const double iv = implied_volatility(C, S, K, r, T);
    REQUIRE_THAT(iv, Catch::Matchers::WithinAbs(0.235, 1e-2));
}
