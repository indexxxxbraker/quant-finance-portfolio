// test_qmc.cpp -- Catch2 tests for the QMC module (Block 3).
//
// Tests cover:
//   - Halton sequence sanity (first point in low dimension).
//   - Sobol sequence sanity (first point should be 0.5 in all dims).
//   - Input validation for the QMC and RQMC pricers.
//   - BS coherence of RQMC (modulo the Euler bias).
//   - Half-width comparison vs IID at equal payoff budget.
//   - Determinism of RQMC under fixed rng seed.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "black_scholes.hpp"
#include "monte_carlo.hpp"
#include "qmc.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;


// =====================================================================
// Halton: first point in low dimension equals 1/p_j
// =====================================================================

TEST_CASE("Halton: first point is (1/2, 1/3, 1/5, 1/7, 1/11) in d = 5",
          "[qmc][halton]") {
    quant::Halton h(5);
    std::vector<double> buf(5);
    h.generate(1, buf.data());

    REQUIRE_THAT(buf[0], WithinRel(0.5,        1e-12));
    REQUIRE_THAT(buf[1], WithinRel(1.0 / 3.0,  1e-12));
    REQUIRE_THAT(buf[2], WithinRel(0.2,        1e-12));
    REQUIRE_THAT(buf[3], WithinRel(1.0 / 7.0,  1e-12));
    REQUIRE_THAT(buf[4], WithinRel(1.0 / 11.0, 1e-12));
}


TEST_CASE("Halton: dim > 20 is rejected", "[qmc][halton][validation]") {
    REQUIRE_THROWS_AS(quant::Halton(21), std::invalid_argument);
}


// =====================================================================
// Sobol: first point is 0.5 in every dimension
// =====================================================================

TEST_CASE("Sobol: first point is 0.5 in every dimension",
          "[qmc][sobol]") {
    quant::Sobol s(5);
    std::vector<double> buf(5);
    s.generate(1, buf.data());

    for (std::size_t j = 0; j < 5; ++j) {
        REQUIRE_THAT(buf[j], WithinAbs(0.5, 1e-15));
    }
}


TEST_CASE("Sobol: reset() restarts from index 1", "[qmc][sobol]") {
    quant::Sobol s(3);
    std::vector<double> buf1(3 * 4);
    std::vector<double> buf2(3 * 4);

    s.generate(4, buf1.data());
    s.reset();
    s.generate(4, buf2.data());

    for (std::size_t i = 0; i < buf1.size(); ++i) {
        REQUIRE(buf1[i] == buf2[i]);
    }
}


// =====================================================================
// Deterministic QMC pricer: input validation
// =====================================================================

TEST_CASE("QMC pricer: input validation", "[qmc][validation]") {
    SECTION("S must be positive") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_euler_qmc(
                -1.0, 100.0, 0.05, 0.20, 1.0, 1024, 20, "sobol"),
            std::invalid_argument);
    }
    SECTION("K must be positive") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_euler_qmc(
                100.0, 0.0, 0.05, 0.20, 1.0, 1024, 20, "sobol"),
            std::invalid_argument);
    }
    SECTION("sigma must be positive") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_euler_qmc(
                100.0, 100.0, 0.05, -0.10, 1.0, 1024, 20, "sobol"),
            std::invalid_argument);
    }
    SECTION("T must be positive") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_euler_qmc(
                100.0, 100.0, 0.05, 0.20, 0.0, 1024, 20, "sobol"),
            std::invalid_argument);
    }
    SECTION("n_paths must be at least 2") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_euler_qmc(
                100.0, 100.0, 0.05, 0.20, 1.0, 1, 20, "sobol"),
            std::invalid_argument);
    }
    SECTION("unknown sequence string") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_euler_qmc(
                100.0, 100.0, 0.05, 0.20, 1.0, 1024, 20, "lattice"),
            std::invalid_argument);
    }
    SECTION("r is unconstrained") {
        REQUIRE_NOTHROW(
            quant::mc_european_call_euler_qmc(
                100.0, 100.0, -0.02, 0.20, 1.0, 1024, 20, "sobol"));
    }
}


// =====================================================================
// RQMC pricer: input validation
// =====================================================================

TEST_CASE("RQMC pricer: input validation", "[qmc][rqmc][validation]") {
    std::mt19937_64 rng(42);

    SECTION("S must be positive") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_euler_rqmc(
                -1.0, 100.0, 0.05, 0.20, 1.0, 1024, 20, 10, rng),
            std::invalid_argument);
    }
    SECTION("K must be positive") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_euler_rqmc(
                100.0, 0.0, 0.05, 0.20, 1.0, 1024, 20, 10, rng),
            std::invalid_argument);
    }
    SECTION("sigma must be positive") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_euler_rqmc(
                100.0, 100.0, 0.05, -0.10, 1.0, 1024, 20, 10, rng),
            std::invalid_argument);
    }
    SECTION("T must be positive") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_euler_rqmc(
                100.0, 100.0, 0.05, 0.20, 0.0, 1024, 20, 10, rng),
            std::invalid_argument);
    }
    SECTION("n_paths must be at least 2") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_euler_rqmc(
                100.0, 100.0, 0.05, 0.20, 1.0, 1, 20, 10, rng),
            std::invalid_argument);
    }
    SECTION("n_replications must be at least 2") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_euler_rqmc(
                100.0, 100.0, 0.05, 0.20, 1.0, 1024, 20, 1, rng),
            std::invalid_argument);
    }
    SECTION("r is unconstrained") {
        REQUIRE_NOTHROW(
            quant::mc_european_call_euler_rqmc(
                100.0, 100.0, -0.02, 0.20, 1.0, 1024, 20, 10, rng));
    }
}


// =====================================================================
// RQMC: BS coherence (modulo Euler bias)
// =====================================================================

TEST_CASE("RQMC pricer: consistent with BS modulo Euler bias",
          "[qmc][rqmc][bs]") {
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double bs_price = quant::call_price(S, K, r, sigma, T);

    // Use n_paths * R = 4096 * 20 = 81920 payoffs.
    constexpr std::size_t n_paths = 4096;
    constexpr std::size_t n_steps = 20;
    constexpr std::size_t n_reps  = 20;

    std::mt19937_64 rng(42);
    const auto result = quant::mc_european_call_euler_rqmc(
        S, K, r, sigma, T, n_paths, n_steps, n_reps, rng);

    // We expect the RQMC estimate to fall within ~5*hw of BS, given:
    // (a) Euler with N=20 has bias ~ 0.02 (smaller than the half-width
    //     at this budget),
    // (b) RQMC half-width is ~ 0.015 at this budget.
    INFO("RQMC est = " << result.estimate
         << ", BS = " << bs_price
         << ", hw = " << result.half_width);
    REQUIRE(std::abs(result.estimate - bs_price)
            <= 5.0 * result.half_width);
}


// =====================================================================
// RQMC: half-width below IID at equal payoff budget
// =====================================================================

TEST_CASE("RQMC pricer: half-width below IID at equal payoff budget",
          "[qmc][rqmc][vrf]") {
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    constexpr std::size_t n_steps = 20;

    // Common total budget: 100k payoffs.
    constexpr std::size_t n_iid    = 100'000;
    constexpr std::size_t n_per_r  = 5000;
    constexpr std::size_t n_reps   = 20;

    std::mt19937_64 rng_iid(11);
    std::mt19937_64 rng_rqmc(11);

    const auto result_iid = quant::mc_european_call_euler(
        S, K, r, sigma, T, n_steps, n_iid, rng_iid);
    const auto result_rqmc = quant::mc_european_call_euler_rqmc(
        S, K, r, sigma, T, n_per_r, n_steps, n_reps, rng_rqmc);

    const double ratio = result_iid.half_width / result_rqmc.half_width;

    INFO("IID hw = " << result_iid.half_width
         << ", RQMC hw = " << result_rqmc.half_width
         << ", ratio = " << ratio);

    // We require RQMC to win significantly. Predicted ratio at d=20
    // is about 7-8x; we set a conservative lower bound of 3x to
    // guard against MC noise at this single-seed test.
    REQUIRE(ratio > 3.0);
}


// =====================================================================
// RQMC: determinism under seed
// =====================================================================

TEST_CASE("RQMC pricer: identical estimate from rng-equivalent seeds",
          "[qmc][rqmc][determinism]") {
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    constexpr std::size_t n_paths = 1024;
    constexpr std::size_t n_steps = 10;
    constexpr std::size_t n_reps  = 8;
    constexpr std::uint64_t seed = 7;

    std::mt19937_64 rng_a(seed);
    std::mt19937_64 rng_b(seed);

    const auto a = quant::mc_european_call_euler_rqmc(
        S, K, r, sigma, T, n_paths, n_steps, n_reps, rng_a);
    const auto b = quant::mc_european_call_euler_rqmc(
        S, K, r, sigma, T, n_paths, n_steps, n_reps, rng_b);

    REQUIRE(a.estimate == b.estimate);
    REQUIRE(a.half_width == b.half_width);
    REQUIRE(a.sample_variance == b.sample_variance);
}


// =====================================================================
// Deterministic QMC: Halton and Sobol agree with BS to within Euler bias
// =====================================================================

TEST_CASE("QMC-Sobol pricer: consistent with BS at moderate budget",
          "[qmc][sobol][bs]") {
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double bs_price = quant::call_price(S, K, r, sigma, T);

    const double est = quant::mc_european_call_euler_qmc(
        S, K, r, sigma, T, 16384, 20, "sobol");

    // Without a half-width, we set a heuristic tolerance: 0.5 = 5 * 0.1
    // (where 0.1 is roughly the IID half-width at 16k paths). Sobol
    // should beat that easily.
    INFO("Sobol estimate = " << est << ", BS = " << bs_price
         << ", err = " << std::abs(est - bs_price));
    REQUIRE(std::abs(est - bs_price) < 0.5);
}
