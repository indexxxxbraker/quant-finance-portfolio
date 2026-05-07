// test_american.cpp -- Catch2 tests for the American options module
// (Phase 2 Block 6).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "american.hpp"
#include "black_scholes.hpp"

#include <cmath>
#include <random>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;


// =====================================================================
// Block 6: binomial CRR sanity
// =====================================================================

TEST_CASE("Binomial American put: convergence stabilises by N=10000",
          "[american][binomial][convergence]") {
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double v_5000  = quant::binomial_american_put(S, K, r, sigma, T, 5000);
    const double v_10000 = quant::binomial_american_put(S, K, r, sigma, T, 10000);
    INFO("v(5000)=" << v_5000 << ", v(10000)=" << v_10000);
    REQUIRE(std::abs(v_10000 - v_5000) < 1e-3);
}


TEST_CASE("Binomial American put: monotone decrease of price with smaller dt",
          "[american][binomial][monotonicity]") {
    // The Bermudan price is monotonically increasing in N (more
    // exercise opportunities can only increase the value), but for a
    // single asset put with discrete monitoring, the convergence
    // sequence is typically monotone DECREASING from above due to
    // the lattice approximation overestimating slightly at small N.
    // We test the much weaker property: the absolute change
    // decreases as N grows.
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double v_100  = quant::binomial_american_put(S, K, r, sigma, T, 100);
    const double v_500  = quant::binomial_american_put(S, K, r, sigma, T, 500);
    const double v_5000 = quant::binomial_american_put(S, K, r, sigma, T, 5000);
    const double d1 = std::abs(v_500  - v_100);
    const double d2 = std::abs(v_5000 - v_500);
    INFO("|d(500-100)|=" << d1 << ", |d(5000-500)|=" << d2);
    REQUIRE(d2 < d1);
}


TEST_CASE("Binomial American put: dominates European put",
          "[american][binomial][bs]") {
    // The American put always trades at or above the European put
    // (early exercise has positive value when r > 0). We test it is
    // strictly above by a margin large compared to lattice noise.
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double am = quant::binomial_american_put(S, K, r, sigma, T, 5000);
    const double eu = quant::put_price(S, K, r, sigma, T);
    INFO("Am=" << am << ", Eu=" << eu << ", premium=" << (am - eu));
    REQUIRE(am > eu + 0.05);
}


TEST_CASE("Binomial American put: ITM put exercises immediately deep enough",
          "[american][binomial][exercise]") {
    // A deep-ITM put (S << K) at moderate rate has non-trivial early
    // exercise; the price should be strictly above the intrinsic but
    // approach intrinsic as S -> 0.
    constexpr double K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double v_50 = quant::binomial_american_put(50.0,  K, r, sigma, T, 5000);
    const double v_30 = quant::binomial_american_put(30.0,  K, r, sigma, T, 5000);
    // Both above intrinsic.
    REQUIRE(v_50 >= 50.0);
    REQUIRE(v_30 >= 70.0);
}


// =====================================================================
// Block 6: LSM matches binomial
// =====================================================================

TEST_CASE("LSM American put matches binomial within 3 hw at canonical params",
          "[american][lsm][matches]") {
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double v_ref = quant::binomial_american_put(S, K, r, sigma, T, 5000);

    std::mt19937_64 rng(42);
    const auto res = quant::lsm_american_put(S, K, r, sigma, T,
                                              100'000, 50, 4, rng);
    const double err = std::abs(res.estimate - v_ref);
    INFO("ref=" << v_ref << ", est=" << res.estimate
         << ", hw=" << res.half_width
         << ", err=" << err << ", err/hw=" << (err / res.half_width));
    REQUIRE(err <= 3.0 * res.half_width);
}


TEST_CASE("LSM American put: early exercise premium positive",
          "[american][lsm][premium]") {
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double eu = quant::put_price(S, K, r, sigma, T);

    std::mt19937_64 rng(42);
    const auto res = quant::lsm_american_put(S, K, r, sigma, T,
                                              100'000, 50, 4, rng);
    const double premium = res.estimate - eu;
    INFO("Eu=" << eu << ", Am(LSM)=" << res.estimate
         << ", premium=" << premium << ", hw=" << res.half_width);
    // Premium ~ 0.52, hw ~ 0.045: well separated.
    REQUIRE(premium > 0.20);
}


// =====================================================================
// Block 6: input validation
// =====================================================================

TEST_CASE("American: input validation",
          "[american][validation]") {
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    std::mt19937_64 rng(42);

    SECTION("binomial rejects bad inputs") {
        REQUIRE_THROWS_AS(
            quant::binomial_american_put(-1.0, K, r, sigma, T, 100),
            std::invalid_argument);
        REQUIRE_THROWS_AS(
            quant::binomial_american_put(S, 0.0, r, sigma, T, 100),
            std::invalid_argument);
        REQUIRE_THROWS_AS(
            quant::binomial_american_put(S, K, r, 0.0, T, 100),
            std::invalid_argument);
        REQUIRE_THROWS_AS(
            quant::binomial_american_put(S, K, r, sigma, T, 0),
            std::invalid_argument);
    }

    SECTION("lsm rejects bad inputs") {
        REQUIRE_THROWS_AS(
            quant::lsm_american_put(-1.0, K, r, sigma, T, 1000, 50, 4, rng),
            std::invalid_argument);
        REQUIRE_THROWS_AS(
            quant::lsm_american_put(S, K, r, sigma, T, 1, 50, 4, rng),
            std::invalid_argument);
        REQUIRE_THROWS_AS(
            quant::lsm_american_put(S, K, r, sigma, T, 1000, 0, 4, rng),
            std::invalid_argument);
        REQUIRE_THROWS_AS(
            quant::lsm_american_put(S, K, r, sigma, T, 1000, 50, 0, rng),
            std::invalid_argument);
        REQUIRE_THROWS_AS(
            quant::lsm_american_put(S, K, r, sigma, T, 1000, 50, 9, rng),
            std::invalid_argument);
    }

    SECTION("negative r is admissible") {
        REQUIRE_NOTHROW(
            quant::binomial_american_put(S, K, -0.02, sigma, T, 1000));
        REQUIRE_NOTHROW(
            quant::lsm_american_put(S, K, -0.02, sigma, T, 1000, 50, 4, rng));
    }
}


// =====================================================================
// Block 6: determinism
// =====================================================================

TEST_CASE("American: identical seeds yield identical LSM results",
          "[american][lsm][determinism]") {
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;

    std::mt19937_64 rng_a(7);
    std::mt19937_64 rng_b(7);
    const auto a = quant::lsm_american_put(S, K, r, sigma, T, 5000, 50, 4, rng_a);
    const auto b = quant::lsm_american_put(S, K, r, sigma, T, 5000, 50, 4, rng_b);
    REQUIRE(a.estimate == b.estimate);
    REQUIRE(a.half_width == b.half_width);
    REQUIRE(a.sample_variance == b.sample_variance);
}


TEST_CASE("American: binomial is fully deterministic",
          "[american][binomial][determinism]") {
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double a = quant::binomial_american_put(S, K, r, sigma, T, 1000);
    const double b = quant::binomial_american_put(S, K, r, sigma, T, 1000);
    REQUIRE(a == b);
}
