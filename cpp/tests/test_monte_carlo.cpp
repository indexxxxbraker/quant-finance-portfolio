// test_monte_carlo.cpp — Catch2 tests for the MC vanilla European
// call pricer with exact GBM simulation.
//
// Mirrors the four triangulation tests of the Phase 2 Block 1.1
// writeup (Section 6), plus a set of input-validation tests. The
// validation tests are virtually free; the statistical tests run for
// a few seconds in Release mode.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "black_scholes.hpp"
#include "monte_carlo.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using Catch::Matchers::WithinAbs;


// =====================================================================
// Input validation
// =====================================================================

TEST_CASE("MC vanilla: input validation", "[mc][validation]") {
    std::mt19937_64 rng(42);

    SECTION("S must be positive") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_exact(
                -1.0, 100.0, 0.05, 0.20, 1.0, 1000, rng),
            std::invalid_argument);
    }
    SECTION("K must be positive") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_exact(
                100.0, 0.0, 0.05, 0.20, 1.0, 1000, rng),
            std::invalid_argument);
    }
    SECTION("sigma must be positive") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_exact(
                100.0, 100.0, 0.05, -0.10, 1.0, 1000, rng),
            std::invalid_argument);
    }
    SECTION("T must be positive") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_exact(
                100.0, 100.0, 0.05, 0.20, 0.0, 1000, rng),
            std::invalid_argument);
    }
    SECTION("n_paths must be at least 2") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_exact(
                100.0, 100.0, 0.05, 0.20, 1.0, 1, rng),
            std::invalid_argument);
    }
    SECTION("confidence_level must be in (0, 1)") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_exact(
                100.0, 100.0, 0.05, 0.20, 1.0, 1000, rng, 1.5),
            std::invalid_argument);
        REQUIRE_THROWS_AS(
            quant::mc_european_call_exact(
                100.0, 100.0, 0.05, 0.20, 1.0, 1000, rng, 0.0),
            std::invalid_argument);
    }
    SECTION("r is unconstrained (negative rates admissible)") {
        REQUIRE_NOTHROW(
            quant::mc_european_call_exact(
                100.0, 100.0, -0.02, 0.20, 1.0, 1000, rng));
    }
}


// =====================================================================
// Test 1: Containment frequency of the BS price by the MC CI
// =====================================================================

TEST_CASE("MC vanilla: containment frequency of BS price",
          "[mc][containment]") {
    struct Case { const char* label; double S, K, r, sigma, T; };
    const std::vector<Case> grid = {
        {"ATM",          100.0, 100.0, 0.05, 0.20, 1.0},
        {"ITM",          110.0, 100.0, 0.05, 0.20, 1.0},
        {"OTM",           90.0, 100.0, 0.05, 0.20, 1.0},
        {"LongMaturity", 100.0, 100.0, 0.05, 0.30, 5.0},
    };

    constexpr int          n_seeds          = 200;
    constexpr std::size_t  n_paths          = 10'000;
    constexpr double       confidence_level = 0.95;
    // Binomial 95% CI for p = 0.95 with n = 200 trials: ~[0.92, 0.98].
    constexpr double       lower_acceptance = 0.92;
    constexpr double       upper_acceptance = 0.98;

    for (const auto& c : grid) {
        const double bs_price = quant::call_price(
            c.S, c.K, c.r, c.sigma, c.T);

        int contained = 0;
        for (int seed = 0; seed < n_seeds; ++seed) {
            std::mt19937_64 rng(static_cast<uint64_t>(seed));
            const auto result = quant::mc_european_call_exact(
                c.S, c.K, c.r, c.sigma, c.T, n_paths, rng,
                confidence_level);
            if (std::abs(result.estimate - bs_price) <= result.half_width) {
                ++contained;
            }
        }
        const double rate = static_cast<double>(contained) / n_seeds;

        INFO("Case [" << c.label << "] containment rate = " << rate);
        REQUIRE(rate >= lower_acceptance);
        REQUIRE(rate <= upper_acceptance);
    }
}


// =====================================================================
// Test 2: Empirical convergence rate (slope -1/2 in log-log)
// =====================================================================

TEST_CASE("MC vanilla: empirical convergence rate",
          "[mc][convergence]") {
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double bs_price = quant::call_price(S, K, r, sigma, T);

    const std::vector<std::size_t> n_values = {
        100, 316, 1'000, 3'162, 10'000, 31'623, 100'000
    };
    constexpr int n_seeds_per_n = 50;

    std::vector<double> log_n;
    std::vector<double> log_rmse;

    for (const auto n : n_values) {
        double sum_sq = 0.0;
        for (int k = 0; k < n_seeds_per_n; ++k) {
            // Distinct seeds per (n, k) so no two runs share noise.
            const uint64_t seed =
                static_cast<uint64_t>(1'000'000ULL) * n
                + static_cast<uint64_t>(k);
            std::mt19937_64 rng(seed);
            const auto result = quant::mc_european_call_exact(
                S, K, r, sigma, T, n, rng);
            const double err = result.estimate - bs_price;
            sum_sq += err * err;
        }
        const double rmse = std::sqrt(sum_sq / n_seeds_per_n);
        log_n.push_back(std::log(static_cast<double>(n)));
        log_rmse.push_back(std::log(rmse));
    }

    // Ordinary-least-squares linear fit:
    //   log_rmse_i ~ slope * log_n_i + intercept
    const std::size_t K_pts = log_n.size();
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    for (std::size_t i = 0; i < K_pts; ++i) {
        sum_x  += log_n[i];
        sum_y  += log_rmse[i];
        sum_xy += log_n[i] * log_rmse[i];
        sum_xx += log_n[i] * log_n[i];
    }
    const double m = static_cast<double>(K_pts);
    const double slope =
        (m * sum_xy - sum_x * sum_y) / (m * sum_xx - sum_x * sum_x);

    INFO("Fitted slope = " << slope << " (expected -0.5)");
    REQUIRE_THAT(slope, WithinAbs(-0.5, 0.10));
}


// =====================================================================
// Test 3: Sample variance vs closed-form variance
// =====================================================================

TEST_CASE("MC vanilla: sample variance matches closed form",
          "[mc][variance]") {
    struct Case { const char* label; double S, K, r, sigma, T; };
    const std::vector<Case> grid = {
        {"ATM",     100.0, 100.0, 0.05, 0.20, 1.0},
        {"ITM",     110.0, 100.0, 0.05, 0.20, 1.0},
        {"OTM",      90.0, 100.0, 0.05, 0.20, 1.0},
        {"HighVol", 100.0, 100.0, 0.05, 0.50, 1.0},
    };

    constexpr std::size_t n_paths = 1'000'000;
    constexpr double      rel_tol = 0.05;

    for (const auto& c : grid) {
        std::mt19937_64 rng(0);
        const auto result = quant::mc_european_call_exact(
            c.S, c.K, c.r, c.sigma, c.T, n_paths, rng);
        const double var_closed = quant::call_payoff_variance(
            c.S, c.K, c.r, c.sigma, c.T);
        const double rel_err =
            std::abs(result.sample_variance - var_closed) / var_closed;

        INFO("Case [" << c.label
             << "] var_closed = " << var_closed
             << "  var_sample = " << result.sample_variance
             << "  rel_err = " << rel_err);
        REQUIRE(rel_err < rel_tol);
    }
}


// =====================================================================
// Test 4: Monotonicities (CRN) and asymptotic limits
// =====================================================================

TEST_CASE("MC vanilla: monotonicities and asymptotic limits",
          "[mc][limits]") {
    constexpr std::size_t n_paths = 10'000;
    constexpr uint64_t    seed    = 42;

    SECTION("Monotone in S (pathwise under CRN)") {
        std::mt19937_64 rng_low(seed), rng_high(seed);
        const auto low = quant::mc_european_call_exact(
            90.0, 100.0, 0.05, 0.20, 1.0, n_paths, rng_low);
        const auto high = quant::mc_european_call_exact(
            110.0, 100.0, 0.05, 0.20, 1.0, n_paths, rng_high);
        REQUIRE(high.estimate >= low.estimate);
    }

    SECTION("Antimonotone in K (pathwise under CRN)") {
        std::mt19937_64 rng_low(seed), rng_high(seed);
        const auto low_K = quant::mc_european_call_exact(
            100.0, 90.0, 0.05, 0.20, 1.0, n_paths, rng_low);
        const auto high_K = quant::mc_european_call_exact(
            100.0, 110.0, 0.05, 0.20, 1.0, n_paths, rng_high);
        REQUIRE(high_K.estimate <= low_K.estimate);
    }

    SECTION("Monotone in sigma (in expectation, sharpened by CRN)") {
        std::mt19937_64 rng_low(seed), rng_high(seed);
        const auto low_sig = quant::mc_european_call_exact(
            100.0, 100.0, 0.05, 0.20, 1.0, n_paths, rng_low);
        const auto high_sig = quant::mc_european_call_exact(
            100.0, 100.0, 0.05, 0.40, 1.0, n_paths, rng_high);
        REQUIRE(high_sig.estimate >= low_sig.estimate);
    }

    SECTION("Monotone in T (in expectation, sharpened by CRN)") {
        std::mt19937_64 rng_low(seed), rng_high(seed);
        const auto low_T = quant::mc_european_call_exact(
            100.0, 100.0, 0.05, 0.20, 1.0, n_paths, rng_low);
        const auto high_T = quant::mc_european_call_exact(
            100.0, 100.0, 0.05, 0.20, 2.0, n_paths, rng_high);
        REQUIRE(high_T.estimate >= low_T.estimate);
    }

    SECTION("S -> infty implies C / S -> 1") {
        // Tolerance of 1e-2 covers the deterministic finite-S
        // component (~1e-3) and the MC noise on C/S (~4e-3).
        // Same calibration as the Python validation, see comment
        // in validate_mc_european_exact.py.
        std::mt19937_64 rng(seed);
        constexpr double huge_S = 100'000.0;
        const auto result = quant::mc_european_call_exact(
            huge_S, 100.0, 0.05, 0.20, 1.0, n_paths, rng);
        const double ratio = result.estimate / huge_S;
        REQUIRE_THAT(ratio, WithinAbs(1.0, 1e-2));
    }

    SECTION("S -> 0 implies C -> 0") {
        std::mt19937_64 rng(seed);
        constexpr double tiny_S = 1e-6;
        const auto result = quant::mc_european_call_exact(
            tiny_S, 100.0, 0.05, 0.20, 1.0, n_paths, rng);
        REQUIRE(result.estimate < 1e-10);
    }
}


// =====================================================================
// Cross-check: call_payoff_variance against MC estimate at very large n
// =====================================================================

TEST_CASE("BS: call_payoff_variance matches MC at large n",
          "[bs][variance]") {
    // This test cross-checks the closed-form variance against an
    // independent MC estimate at very large n, confirming that the
    // formula in Phase 2 Block 1.1 writeup, Proposition 5.1, is
    // implemented correctly.
    struct Case { const char* label; double S, K, r, sigma, T; };
    const std::vector<Case> grid = {
        {"ATM",     100.0, 100.0, 0.05, 0.20, 1.0},
        {"OTM",      90.0, 100.0, 0.05, 0.20, 1.0},
    };

    constexpr std::size_t n_paths = 2'000'000;

    for (const auto& c : grid) {
        std::mt19937_64 rng(123);
        const auto result = quant::mc_european_call_exact(
            c.S, c.K, c.r, c.sigma, c.T, n_paths, rng);
        const double var_closed = quant::call_payoff_variance(
            c.S, c.K, c.r, c.sigma, c.T);

        INFO("Case [" << c.label
             << "] var_closed = " << var_closed
             << "  var_mc = " << result.sample_variance);
        // 3% relative tolerance at n = 2e6 is generous.
        REQUIRE_THAT(result.sample_variance / var_closed,
                     WithinAbs(1.0, 0.03));
    }
}
