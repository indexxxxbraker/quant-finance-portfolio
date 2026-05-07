// test_asian.cpp -- Catch2 tests for the Asian options module (Block 5).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "asian.hpp"
#include "black_scholes.hpp"

#include <cmath>
#include <random>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;


// =====================================================================
// Block 5: closed form sanity checks
// =====================================================================

TEST_CASE("Asian closed form: reduces to vanilla BS at N=1",
          "[asian][closed_form][bs]") {
    // At N=1, the geometric Asian over a single observation at T is
    // S_T itself, so the geometric Asian call must equal the vanilla
    // European call.
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double bs   = quant::call_price(S, K, r, sigma, T);
    const double asia = quant::geometric_asian_call_closed_form(
                            S, K, r, sigma, T, 1);
    REQUIRE_THAT(asia, WithinRel(bs, 1e-12));
}


TEST_CASE("Asian closed form: continuous-monitoring limit recovers sigma/sqrt(3)",
          "[asian][closed_form][limit]") {
    // As N -> infty, sigma_eff -> sigma / sqrt(3). Test convergence
    // by comparing closed forms at large N against the analytical
    // limit (a vanilla BS call with vol sigma/sqrt(3) and the same
    // limit for r_eff).
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;

    const double sigma_lim = sigma / std::sqrt(3.0);
    // r_eff limit = (1/2)(r - sigma^2/2) + sigma_lim^2/2
    const double r_lim     = 0.5 * (r - 0.5 * sigma * sigma)
                              + 0.5 * sigma_lim * sigma_lim;
    const double sqrt_T = std::sqrt(T);
    const double d1_lim = (std::log(S / K)
                            + (r_lim + 0.5 * sigma_lim * sigma_lim) * T)
                          / (sigma_lim * sqrt_T);
    const double d2_lim = d1_lim - sigma_lim * sqrt_T;
    const double C_lim  = std::exp(-r * T)
                            * (S * std::exp(r_lim * T) * quant::norm_cdf(d1_lim)
                               - K * quant::norm_cdf(d2_lim));

    const double C_N1000 = quant::geometric_asian_call_closed_form(
                                S, K, r, sigma, T, 1000);
    INFO("C_lim = " << C_lim << ", C_N=1000 = " << C_N1000);
    // Convergence is O(1/N), so at N=1000 we expect ~3 digits of
    // agreement.
    REQUIRE_THAT(C_N1000, WithinAbs(C_lim, 1e-2));
}


TEST_CASE("Asian closed form: monotonic decrease in N",
          "[asian][closed_form][monotonicity]") {
    // sigma_eff = sigma * sqrt((N+1)(2N+1)/(6N^2)) is monotonically
    // decreasing in N (going from sigma at N=1 to sigma/sqrt(3) at
    // N=infty). Lower vol => lower call price (BS is monotone in vol).
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double C1   = quant::geometric_asian_call_closed_form(S, K, r, sigma, T, 1);
    const double C10  = quant::geometric_asian_call_closed_form(S, K, r, sigma, T, 10);
    const double C50  = quant::geometric_asian_call_closed_form(S, K, r, sigma, T, 50);
    const double C500 = quant::geometric_asian_call_closed_form(S, K, r, sigma, T, 500);

    INFO("C(N=1)=" << C1 << ", C(10)=" << C10
         << ", C(50)=" << C50 << ", C(500)=" << C500);
    REQUIRE(C1   > C10);
    REQUIRE(C10  > C50);
    REQUIRE(C50  > C500);
}


// =====================================================================
// Block 5: IID estimator agreement with the closed form
// =====================================================================

TEST_CASE("Asian: geometric IID matches closed form within 3 hw",
          "[asian][geometric][iid][closed_form]") {
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    constexpr std::size_t N = 50;
    const double closed = quant::geometric_asian_call_closed_form(
                              S, K, r, sigma, T, N);

    std::mt19937_64 rng(42);
    const auto res = quant::mc_asian_call_geometric_iid(
                          S, K, r, sigma, T, 100'000, N, rng);

    INFO("est=" << res.estimate << ", closed=" << closed
         << ", hw=" << res.half_width
         << ", err/hw=" << std::abs(res.estimate - closed) / res.half_width);
    REQUIRE(std::abs(res.estimate - closed) <= 3.0 * res.half_width);
}


// =====================================================================
// Block 5: arithmetic-IID and arithmetic-CV agree (CV is unbiased)
// =====================================================================

TEST_CASE("Asian: arithmetic CV agrees with arithmetic IID",
          "[asian][arithmetic][cv][unbiasedness]") {
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    constexpr std::size_t N = 50;

    std::mt19937_64 rng_a(42);
    const auto res_iid = quant::mc_asian_call_arithmetic_iid(
                              S, K, r, sigma, T, 100'000, N, rng_a);

    std::mt19937_64 rng_b(43);
    const auto res_cv = quant::mc_asian_call_arithmetic_cv(
                              S, K, r, sigma, T, 100'000, N, rng_b);

    // The two estimators are independent (different rng seeds), so
    // their difference's standard deviation is sqrt(hw_iid^2 + hw_cv^2).
    // Allow 4 sigma tolerance.
    const double tol = 4.0 * std::sqrt(res_iid.half_width * res_iid.half_width
                                     + res_cv.half_width  * res_cv.half_width);
    INFO("IID=" << res_iid.estimate << " (hw=" << res_iid.half_width << ")"
         << ", CV=" << res_cv.estimate  << " (hw=" << res_cv.half_width << ")"
         << ", tol=" << tol);
    REQUIRE(std::abs(res_iid.estimate - res_cv.estimate) <= tol);
}


// =====================================================================
// Block 5: variance reduction factor for the CV
// =====================================================================

TEST_CASE("Asian: geometric CV gives huge VRF (>500)",
          "[asian][cv][vrf]") {
    // Geometric and arithmetic averages of the same lognormal path
    // are bound by AM-GM and share all the noise. Empirical
    // correlation > 0.999 at our parameters, predicting VRF > 1000.
    // We require VRF > 500 with margin against MC noise.
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    constexpr std::size_t N = 50;
    constexpr std::size_t n = 50'000;

    std::mt19937_64 rng_a(42);
    const auto res_iid = quant::mc_asian_call_arithmetic_iid(
                              S, K, r, sigma, T, n, N, rng_a);

    std::mt19937_64 rng_b(42);
    const auto res_cv = quant::mc_asian_call_arithmetic_cv(
                              S, K, r, sigma, T, n, N, rng_b);

    const double vrf = (res_iid.half_width / res_cv.half_width)
                        * (res_iid.half_width / res_cv.half_width);
    INFO("hw_iid=" << res_iid.half_width
         << ", hw_cv=" << res_cv.half_width
         << ", VRF=" << vrf);
    REQUIRE(vrf > 500.0);
}


// =====================================================================
// Block 5: input validation
// =====================================================================

TEST_CASE("Asian: input validation",
          "[asian][validation]") {
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    constexpr std::size_t N = 50;
    std::mt19937_64 rng(42);

    SECTION("closed form rejects bad inputs") {
        REQUIRE_THROWS_AS(
            quant::geometric_asian_call_closed_form(-1.0, K, r, sigma, T, N),
            std::invalid_argument);
        REQUIRE_THROWS_AS(
            quant::geometric_asian_call_closed_form(S, 0.0, r, sigma, T, N),
            std::invalid_argument);
        REQUIRE_THROWS_AS(
            quant::geometric_asian_call_closed_form(S, K, r, 0.0, T, N),
            std::invalid_argument);
        REQUIRE_THROWS_AS(
            quant::geometric_asian_call_closed_form(S, K, r, sigma, T, 0),
            std::invalid_argument);
    }

    SECTION("arithmetic IID rejects bad inputs") {
        REQUIRE_THROWS_AS(
            quant::mc_asian_call_arithmetic_iid(-1.0, K, r, sigma, T, 1'000, N, rng),
            std::invalid_argument);
        REQUIRE_THROWS_AS(
            quant::mc_asian_call_arithmetic_iid(S, K, r, sigma, T, 1, N, rng),
            std::invalid_argument);
    }

    SECTION("arithmetic CV rejects bad inputs") {
        REQUIRE_THROWS_AS(
            quant::mc_asian_call_arithmetic_cv(S, K, r, sigma, T, 1'000, 0, rng),
            std::invalid_argument);
    }

    SECTION("negative r is admissible") {
        REQUIRE_NOTHROW(
            quant::geometric_asian_call_closed_form(S, K, -0.02, sigma, T, N));
        REQUIRE_NOTHROW(
            quant::mc_asian_call_arithmetic_iid(S, K, -0.02, sigma, T, 1'000, N, rng));
    }
}


// =====================================================================
// Block 5: determinism
// =====================================================================

TEST_CASE("Asian: identical seeds yield identical results",
          "[asian][determinism]") {
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    constexpr std::size_t N = 50;

    SECTION("arithmetic IID") {
        std::mt19937_64 rng_a(7);
        std::mt19937_64 rng_b(7);
        const auto a = quant::mc_asian_call_arithmetic_iid(S, K, r, sigma, T, 5'000, N, rng_a);
        const auto b = quant::mc_asian_call_arithmetic_iid(S, K, r, sigma, T, 5'000, N, rng_b);
        REQUIRE(a.estimate == b.estimate);
        REQUIRE(a.half_width == b.half_width);
    }

    SECTION("geometric IID") {
        std::mt19937_64 rng_a(7);
        std::mt19937_64 rng_b(7);
        const auto a = quant::mc_asian_call_geometric_iid(S, K, r, sigma, T, 5'000, N, rng_a);
        const auto b = quant::mc_asian_call_geometric_iid(S, K, r, sigma, T, 5'000, N, rng_b);
        REQUIRE(a.estimate == b.estimate);
        REQUIRE(a.half_width == b.half_width);
    }

    SECTION("arithmetic CV") {
        std::mt19937_64 rng_a(7);
        std::mt19937_64 rng_b(7);
        const auto a = quant::mc_asian_call_arithmetic_cv(S, K, r, sigma, T, 5'000, N, rng_a);
        const auto b = quant::mc_asian_call_arithmetic_cv(S, K, r, sigma, T, 5'000, N, rng_b);
        REQUIRE(a.estimate == b.estimate);
        REQUIRE(a.half_width == b.half_width);
    }
}
