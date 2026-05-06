// test_variance_reduction.cpp -- Catch2 tests for variance_reduction.
//
// Block 2.1: antithetic-variates pricer for the European call.
//
// Empirical verification of the variance reduction factor (with the
// VRF and rho measurements) lives in the Python validation script
// validate_mc_european_av.py. The C++ tests focus on structural and
// per-implementation correctness: input validation, agreement with
// BS, and a sanity check that the AV estimator's sample variance is
// smaller than the IID estimator's sample variance.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "black_scholes.hpp"
#include "monte_carlo.hpp"
#include "variance_reduction.hpp"

#include <cmath>
#include <cstdint>
#include <random>

using Catch::Matchers::WithinAbs;


// =====================================================================
// Block 2.1: input validation
// =====================================================================

TEST_CASE("MC AV: input validation",
          "[mc][av][validation]") {
    std::mt19937_64 rng(42);

    SECTION("S must be positive") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_exact_av(
                -1.0, 100.0, 0.05, 0.20, 1.0, 1000, rng),
            std::invalid_argument);
    }
    SECTION("K must be positive") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_exact_av(
                100.0, 0.0, 0.05, 0.20, 1.0, 1000, rng),
            std::invalid_argument);
    }
    SECTION("sigma must be positive") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_exact_av(
                100.0, 100.0, 0.05, -0.10, 1.0, 1000, rng),
            std::invalid_argument);
    }
    SECTION("T must be positive") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_exact_av(
                100.0, 100.0, 0.05, 0.20, 0.0, 1000, rng),
            std::invalid_argument);
    }
    SECTION("n_paths must be at least 2") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_exact_av(
                100.0, 100.0, 0.05, 0.20, 1.0, 1, rng),
            std::invalid_argument);
    }
    SECTION("r is unconstrained (negative rates admissible)") {
        REQUIRE_NOTHROW(
            quant::mc_european_call_exact_av(
                100.0, 100.0, -0.02, 0.20, 1.0, 1000, rng));
    }
}


// =====================================================================
// Block 2.1: AV estimator agrees with BS within a few half-widths
// =====================================================================

TEST_CASE("AV pricer: consistent with BS",
          "[mc][av][bs]") {
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double bs_price = quant::call_price(S, K, r, sigma, T);

    constexpr std::size_t n_paths = 50'000;

    std::mt19937_64 rng(42);
    const auto result = quant::mc_european_call_exact_av(
        S, K, r, sigma, T, n_paths, rng);

    INFO("AV estimate = " << result.estimate
         << ", BS = " << bs_price
         << ", half-width = " << result.half_width);

    REQUIRE(std::abs(result.estimate - bs_price)
            <= 3.0 * result.half_width);
}


// =====================================================================
// Block 2.1: AV variance is smaller than IID variance at equal budget
// =====================================================================

TEST_CASE("AV pricer: half-width below IID at equal payoff budget",
          "[mc][av][vrf]") {
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;

    // AV with n_pairs pairs uses 2*n_pairs payoff evaluations.
    // IID with n_iid = 2*n_pairs paths uses the same number.
    constexpr std::size_t n_pairs = 50'000;
    constexpr std::size_t n_iid   = 2 * n_pairs;

    std::mt19937_64 rng_av(11);
    std::mt19937_64 rng_iid(11);

    const auto result_av  = quant::mc_european_call_exact_av(
        S, K, r, sigma, T, n_pairs, rng_av);
    const auto result_iid = quant::mc_european_call_exact(
        S, K, r, sigma, T, n_iid,   rng_iid);

    // The half-width ratio (IID / AV) should be > 1 by the
    // monotonicity-induced negative correlation. For ATM call with
    // these parameters, the predicted ratio is sqrt(VRF) ~ sqrt(2)
    // ~= 1.41, but for safety we only require > 1.05 to guard
    // against MC noise at this sample size.
    const double ratio = result_iid.half_width / result_av.half_width;

    INFO("Half-width AV = " << result_av.half_width
         << ", IID = " << result_iid.half_width
         << ", ratio = " << ratio);

    REQUIRE(ratio > 1.05);
}


// =====================================================================
// Block 2.1: paired payoff is symmetric in the sign of Z
// =====================================================================

TEST_CASE("AV pricer: identical estimate from rng-equivalent seeds",
          "[mc][av][symmetry]") {
    // The estimator is deterministic given the rng state. Two
    // different rng instances starting from the same seed must
    // produce the same estimate.
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    constexpr std::size_t n_paths = 1000;
    constexpr uint64_t seed = 7;

    std::mt19937_64 rng_a(seed);
    std::mt19937_64 rng_b(seed);

    const auto a = quant::mc_european_call_exact_av(
        S, K, r, sigma, T, n_paths, rng_a);
    const auto b = quant::mc_european_call_exact_av(
        S, K, r, sigma, T, n_paths, rng_b);

    REQUIRE(a.estimate == b.estimate);
    REQUIRE(a.half_width == b.half_width);
    REQUIRE(a.sample_variance == b.sample_variance);
    REQUIRE(a.n_paths == b.n_paths);
}
