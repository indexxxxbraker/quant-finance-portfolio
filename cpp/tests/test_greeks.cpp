// test_greeks.cpp -- Catch2 tests for the Greeks module (Block 4).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "black_scholes.hpp"
#include "greeks.hpp"

#include <cmath>
#include <random>

using Catch::Matchers::WithinAbs;


// =====================================================================
// Block 4: BS coherence for all 7 estimators
// =====================================================================

TEST_CASE("Greeks: Delta bump consistent with BS",
          "[greeks][delta][bump][bs]") {
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double bs_delta = quant::call_delta(S, K, r, sigma, T);

    std::mt19937_64 rng(42);
    const auto result = quant::delta_bump(S, K, r, sigma, T, 100'000, rng);

    INFO("Delta bump est = " << result.estimate << ", BS = " << bs_delta
         << ", hw = " << result.half_width);
    REQUIRE(std::abs(result.estimate - bs_delta) <= 3.0 * result.half_width);
}


TEST_CASE("Greeks: Delta pathwise consistent with BS",
          "[greeks][delta][pathwise][bs]") {
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double bs_delta = quant::call_delta(S, K, r, sigma, T);

    std::mt19937_64 rng(42);
    const auto result = quant::delta_pathwise(S, K, r, sigma, T, 100'000, rng);

    INFO("Delta pathwise est = " << result.estimate << ", BS = " << bs_delta
         << ", hw = " << result.half_width);
    REQUIRE(std::abs(result.estimate - bs_delta) <= 3.0 * result.half_width);
}


TEST_CASE("Greeks: Delta LR consistent with BS",
          "[greeks][delta][lr][bs]") {
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double bs_delta = quant::call_delta(S, K, r, sigma, T);

    std::mt19937_64 rng(42);
    const auto result = quant::delta_lr(S, K, r, sigma, T, 100'000, rng);

    INFO("Delta LR est = " << result.estimate << ", BS = " << bs_delta
         << ", hw = " << result.half_width);
    REQUIRE(std::abs(result.estimate - bs_delta) <= 3.0 * result.half_width);
}


TEST_CASE("Greeks: Vega bump consistent with BS",
          "[greeks][vega][bump][bs]") {
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double bs_vega = quant::vega(S, K, r, sigma, T);

    std::mt19937_64 rng(42);
    const auto result = quant::vega_bump(S, K, r, sigma, T, 100'000, rng);

    INFO("Vega bump est = " << result.estimate << ", BS = " << bs_vega
         << ", hw = " << result.half_width);
    REQUIRE(std::abs(result.estimate - bs_vega) <= 3.0 * result.half_width);
}


TEST_CASE("Greeks: Vega pathwise consistent with BS",
          "[greeks][vega][pathwise][bs]") {
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double bs_vega = quant::vega(S, K, r, sigma, T);

    std::mt19937_64 rng(42);
    const auto result = quant::vega_pathwise(S, K, r, sigma, T, 100'000, rng);

    INFO("Vega pathwise est = " << result.estimate << ", BS = " << bs_vega
         << ", hw = " << result.half_width);
    REQUIRE(std::abs(result.estimate - bs_vega) <= 3.0 * result.half_width);
}


TEST_CASE("Greeks: Vega LR consistent with BS",
          "[greeks][vega][lr][bs]") {
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double bs_vega = quant::vega(S, K, r, sigma, T);

    std::mt19937_64 rng(42);
    const auto result = quant::vega_lr(S, K, r, sigma, T, 100'000, rng);

    INFO("Vega LR est = " << result.estimate << ", BS = " << bs_vega
         << ", hw = " << result.half_width);
    REQUIRE(std::abs(result.estimate - bs_vega) <= 3.0 * result.half_width);
}


TEST_CASE("Greeks: Gamma bump consistent with BS",
          "[greeks][gamma][bump][bs]") {
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double bs_gamma = quant::gamma(S, K, r, sigma, T);

    std::mt19937_64 rng(42);
    const auto result = quant::gamma_bump(S, K, r, sigma, T, 100'000, rng);

    INFO("Gamma bump est = " << result.estimate << ", BS = " << bs_gamma
         << ", hw = " << result.half_width);
    REQUIRE(std::abs(result.estimate - bs_gamma) <= 3.0 * result.half_width);
}


// =====================================================================
// Variance ranking: pathwise <= bump < LR for Delta
// =====================================================================

TEST_CASE("Greeks: Delta variance ranking (pathwise comparable to bump < LR)",
          "[greeks][delta][variance]") {
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    constexpr std::size_t n = 100'000;

    std::mt19937_64 rng_bump(11), rng_pw(11), rng_lr(11);

    const auto res_bump = quant::delta_bump(S, K, r, sigma, T, n, rng_bump);
    const auto res_pw   = quant::delta_pathwise(S, K, r, sigma, T, n, rng_pw);
    const auto res_lr   = quant::delta_lr(S, K, r, sigma, T, n, rng_lr);

    INFO("hw bump = " << res_bump.half_width
         << ", pathwise = " << res_pw.half_width
         << ", LR = " << res_lr.half_width);

    // LR should be at least 1.5x wider than pathwise.
    REQUIRE(res_lr.half_width > 1.5 * res_pw.half_width);

    // Bump should be roughly comparable to pathwise (within 0.7x..1.5x).
    const double ratio = res_bump.half_width / res_pw.half_width;
    REQUIRE(ratio >= 0.7);
    REQUIRE(ratio <= 1.5);
}


// =====================================================================
// Variance ranking: same for Vega
// =====================================================================

TEST_CASE("Greeks: Vega variance ranking (pathwise comparable to bump < LR)",
          "[greeks][vega][variance]") {
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    constexpr std::size_t n = 100'000;

    std::mt19937_64 rng_bump(11), rng_pw(11), rng_lr(11);

    const auto res_bump = quant::vega_bump(S, K, r, sigma, T, n, rng_bump);
    const auto res_pw   = quant::vega_pathwise(S, K, r, sigma, T, n, rng_pw);
    const auto res_lr   = quant::vega_lr(S, K, r, sigma, T, n, rng_lr);

    INFO("hw bump = " << res_bump.half_width
         << ", pathwise = " << res_pw.half_width
         << ", LR = " << res_lr.half_width);

    REQUIRE(res_lr.half_width > 1.5 * res_pw.half_width);

    const double ratio = res_bump.half_width / res_pw.half_width;
    REQUIRE(ratio >= 0.7);
    REQUIRE(ratio <= 1.5);
}


// =====================================================================
// Input validation (one representative pricer per method)
// =====================================================================

TEST_CASE("Greeks: input validation across methods",
          "[greeks][validation]") {
    std::mt19937_64 rng(42);

    SECTION("delta_bump: S must be positive") {
        REQUIRE_THROWS_AS(
            quant::delta_bump(-1.0, 100.0, 0.05, 0.20, 1.0, 1000, rng),
            std::invalid_argument);
    }
    SECTION("delta_bump: K must be positive") {
        REQUIRE_THROWS_AS(
            quant::delta_bump(100.0, 0.0, 0.05, 0.20, 1.0, 1000, rng),
            std::invalid_argument);
    }
    SECTION("delta_bump: sigma must be positive") {
        REQUIRE_THROWS_AS(
            quant::delta_bump(100.0, 100.0, 0.05, -0.10, 1.0, 1000, rng),
            std::invalid_argument);
    }
    SECTION("delta_bump: T must be positive") {
        REQUIRE_THROWS_AS(
            quant::delta_bump(100.0, 100.0, 0.05, 0.20, 0.0, 1000, rng),
            std::invalid_argument);
    }
    SECTION("delta_bump: n_paths must be at least 2") {
        REQUIRE_THROWS_AS(
            quant::delta_bump(100.0, 100.0, 0.05, 0.20, 1.0, 1, rng),
            std::invalid_argument);
    }
    SECTION("delta_pathwise: rejects bad S") {
        REQUIRE_THROWS_AS(
            quant::delta_pathwise(-1.0, 100.0, 0.05, 0.20, 1.0, 1000, rng),
            std::invalid_argument);
    }
    SECTION("delta_lr: rejects bad sigma") {
        REQUIRE_THROWS_AS(
            quant::delta_lr(100.0, 100.0, 0.05, -0.10, 1.0, 1000, rng),
            std::invalid_argument);
    }
    SECTION("gamma_bump: rejects bad K") {
        REQUIRE_THROWS_AS(
            quant::gamma_bump(100.0, 0.0, 0.05, 0.20, 1.0, 1000, rng),
            std::invalid_argument);
    }
    SECTION("negative r is admissible across methods") {
        REQUIRE_NOTHROW(
            quant::delta_bump(100.0, 100.0, -0.02, 0.20, 1.0, 1000, rng));
        REQUIRE_NOTHROW(
            quant::vega_pathwise(100.0, 100.0, -0.02, 0.20, 1.0, 1000, rng));
        REQUIRE_NOTHROW(
            quant::delta_lr(100.0, 100.0, -0.02, 0.20, 1.0, 1000, rng));
    }
}


// =====================================================================
// Determinism under fixed seed
// =====================================================================

TEST_CASE("Greeks: determinism under fixed seed (one pricer per method)",
          "[greeks][determinism]") {
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    constexpr std::size_t n = 1000;
    constexpr std::uint64_t seed = 7;

    SECTION("delta_pathwise") {
        std::mt19937_64 rng_a(seed), rng_b(seed);
        const auto a = quant::delta_pathwise(S, K, r, sigma, T, n, rng_a);
        const auto b = quant::delta_pathwise(S, K, r, sigma, T, n, rng_b);
        REQUIRE(a.estimate == b.estimate);
        REQUIRE(a.half_width == b.half_width);
    }
    SECTION("vega_lr") {
        std::mt19937_64 rng_a(seed), rng_b(seed);
        const auto a = quant::vega_lr(S, K, r, sigma, T, n, rng_a);
        const auto b = quant::vega_lr(S, K, r, sigma, T, n, rng_b);
        REQUIRE(a.estimate == b.estimate);
        REQUIRE(a.half_width == b.half_width);
    }
    SECTION("gamma_bump") {
        std::mt19937_64 rng_a(seed), rng_b(seed);
        const auto a = quant::gamma_bump(S, K, r, sigma, T, n, rng_a);
        const auto b = quant::gamma_bump(S, K, r, sigma, T, n, rng_b);
        REQUIRE(a.estimate == b.estimate);
        REQUIRE(a.half_width == b.half_width);
    }
}
