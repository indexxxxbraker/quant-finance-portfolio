// test_monte_carlo.cpp -- Catch2 tests for the Monte Carlo module.
//
// Covers Block 1.1 (exact GBM sampler), Block 1.2.1 (Euler-Maruyama),
// and Block 1.2.2 (Milstein).
//
// The Block 1.1 tests mirror the four triangulation tests of the
// Block 1.1 writeup plus input-validation cases.
//
// The Block 1.2.x tests verify:
//   - Input validation of the new pricers and samplers.
//   - simulate_path_* returns the correct shape and starts at S0.
//   - simulate_terminal_* agrees with simulate_path_*'s last column
//     when both consume the same RNG state.
//   - The pricer at large n_steps produces an estimate consistent
//     with the BS price within a few half-widths.
//
// Empirical convergence-order verification (slope tests) lives in
// the Python validation scripts.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "black_scholes.hpp"
#include "monte_carlo.hpp"
#include "gbm.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using Catch::Matchers::WithinAbs;


// =====================================================================
// Block 1.1: input validation (exact pricer)
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
// Block 1.1: containment frequency
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
// Block 1.1: empirical convergence rate
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
// Block 1.1: variance agreement with closed form
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
// Block 1.1: monotonicities and asymptotic limits
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
// Block 1.1: closed-form variance cross-check
// =====================================================================

TEST_CASE("BS: call_payoff_variance matches MC at large n",
          "[bs][variance]") {
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
        REQUIRE_THAT(result.sample_variance / var_closed,
                     WithinAbs(1.0, 0.03));
    }
}


// =====================================================================
// Block 1.2.1: input validation (Euler pricer)
// =====================================================================

TEST_CASE("MC Euler: input validation", "[mc][euler][validation]") {
    std::mt19937_64 rng(42);

    SECTION("S must be positive") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_euler(
                -1.0, 100.0, 0.05, 0.20, 1.0, 50, 1000, rng),
            std::invalid_argument);
    }
    SECTION("K must be positive") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_euler(
                100.0, 0.0, 0.05, 0.20, 1.0, 50, 1000, rng),
            std::invalid_argument);
    }
    SECTION("sigma must be positive") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_euler(
                100.0, 100.0, 0.05, -0.10, 1.0, 50, 1000, rng),
            std::invalid_argument);
    }
    SECTION("T must be positive") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_euler(
                100.0, 100.0, 0.05, 0.20, 0.0, 50, 1000, rng),
            std::invalid_argument);
    }
    SECTION("n_steps must be at least 1") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_euler(
                100.0, 100.0, 0.05, 0.20, 1.0, 0, 1000, rng),
            std::invalid_argument);
    }
    SECTION("n_paths must be at least 2") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_euler(
                100.0, 100.0, 0.05, 0.20, 1.0, 50, 1, rng),
            std::invalid_argument);
    }
    SECTION("r is unconstrained (negative rates admissible)") {
        REQUIRE_NOTHROW(
            quant::mc_european_call_euler(
                100.0, 100.0, -0.02, 0.20, 1.0, 50, 1000, rng));
    }
}


// =====================================================================
// Block 1.2.1: simulate_path_euler shape and initial value
// =====================================================================

TEST_CASE("Euler path: shape and initial column",
          "[mc][euler][path]") {
    constexpr double S0 = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    constexpr std::size_t n_steps = 50;
    constexpr std::size_t n_paths = 100;

    std::mt19937_64 rng(123);
    const auto paths = quant::simulate_path_euler(
        S0, r, sigma, T, n_steps, n_paths, rng);

    REQUIRE(paths.size() == n_paths);
    for (const auto& path : paths) {
        REQUIRE(path.size() == n_steps + 1);
        REQUIRE(path.front() == S0);
    }
}


// =====================================================================
// Block 1.2.1: terminal sampler agrees with path sampler under same seed
// =====================================================================

TEST_CASE("Euler: terminal sampler agrees with path's last column",
          "[mc][euler][equivalence]") {
    constexpr double S0 = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    constexpr std::size_t n_steps = 30;
    constexpr std::size_t n_paths = 50;
    constexpr uint64_t seed = 7;

    std::mt19937_64 rng_path(seed), rng_terminal(seed);

    const auto paths = quant::simulate_path_euler(
        S0, r, sigma, T, n_steps, n_paths, rng_path);
    const auto terminal = quant::simulate_terminal_euler(
        S0, r, sigma, T, n_steps, n_paths, rng_terminal);

    REQUIRE(terminal.size() == n_paths);
    for (std::size_t i = 0; i < n_paths; ++i) {
        REQUIRE(terminal[i] == paths[i].back());
    }
}


// =====================================================================
// Block 1.2.1: sanity check vs BS at large n_steps
// =====================================================================

TEST_CASE("Euler pricer: consistent with BS at large n_steps",
          "[mc][euler][bs]") {
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double bs_price = quant::call_price(S, K, r, sigma, T);

    constexpr std::size_t n_steps = 200;
    constexpr std::size_t n_paths = 50'000;

    std::mt19937_64 rng(42);
    const auto result = quant::mc_european_call_euler(
        S, K, r, sigma, T, n_steps, n_paths, rng);

    INFO("Euler estimate = " << result.estimate
         << ", BS = " << bs_price
         << ", half-width = " << result.half_width);

    REQUIRE(std::abs(result.estimate - bs_price)
            <= 3.0 * result.half_width);
}


// =====================================================================
// Block 1.2.2: input validation (Milstein pricer)
// =====================================================================

TEST_CASE("MC Milstein: input validation",
          "[mc][milstein][validation]") {
    std::mt19937_64 rng(42);

    SECTION("S must be positive") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_milstein(
                -1.0, 100.0, 0.05, 0.20, 1.0, 50, 1000, rng),
            std::invalid_argument);
    }
    SECTION("K must be positive") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_milstein(
                100.0, 0.0, 0.05, 0.20, 1.0, 50, 1000, rng),
            std::invalid_argument);
    }
    SECTION("sigma must be positive") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_milstein(
                100.0, 100.0, 0.05, -0.10, 1.0, 50, 1000, rng),
            std::invalid_argument);
    }
    SECTION("T must be positive") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_milstein(
                100.0, 100.0, 0.05, 0.20, 0.0, 50, 1000, rng),
            std::invalid_argument);
    }
    SECTION("n_steps must be at least 1") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_milstein(
                100.0, 100.0, 0.05, 0.20, 1.0, 0, 1000, rng),
            std::invalid_argument);
    }
    SECTION("n_paths must be at least 2") {
        REQUIRE_THROWS_AS(
            quant::mc_european_call_milstein(
                100.0, 100.0, 0.05, 0.20, 1.0, 50, 1, rng),
            std::invalid_argument);
    }
    SECTION("r is unconstrained (negative rates admissible)") {
        REQUIRE_NOTHROW(
            quant::mc_european_call_milstein(
                100.0, 100.0, -0.02, 0.20, 1.0, 50, 1000, rng));
    }
}


// =====================================================================
// Block 1.2.2: simulate_path_milstein shape and initial value
// =====================================================================

TEST_CASE("Milstein path: shape and initial column",
          "[mc][milstein][path]") {
    constexpr double S0 = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    constexpr std::size_t n_steps = 50;
    constexpr std::size_t n_paths = 100;

    std::mt19937_64 rng(123);
    const auto paths = quant::simulate_path_milstein(
        S0, r, sigma, T, n_steps, n_paths, rng);

    REQUIRE(paths.size() == n_paths);
    for (const auto& path : paths) {
        REQUIRE(path.size() == n_steps + 1);
        REQUIRE(path.front() == S0);
    }
}


// =====================================================================
// Block 1.2.2: terminal sampler agrees with path sampler under same seed
// =====================================================================

TEST_CASE("Milstein: terminal sampler agrees with path's last column",
          "[mc][milstein][equivalence]") {
    constexpr double S0 = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    constexpr std::size_t n_steps = 30;
    constexpr std::size_t n_paths = 50;
    constexpr uint64_t seed = 7;

    std::mt19937_64 rng_path(seed), rng_terminal(seed);

    const auto paths = quant::simulate_path_milstein(
        S0, r, sigma, T, n_steps, n_paths, rng_path);
    const auto terminal = quant::simulate_terminal_milstein(
        S0, r, sigma, T, n_steps, n_paths, rng_terminal);

    REQUIRE(terminal.size() == n_paths);
    for (std::size_t i = 0; i < n_paths; ++i) {
        // Bit-exact: same RNG state, same arithmetic, same order.
        REQUIRE(terminal[i] == paths[i].back());
    }
}


// =====================================================================
// Block 1.2.2: sanity check vs BS at large n_steps
// =====================================================================

TEST_CASE("Milstein pricer: consistent with BS at large n_steps",
          "[mc][milstein][bs]") {
    constexpr double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double bs_price = quant::call_price(S, K, r, sigma, T);

    constexpr std::size_t n_steps = 200;
    constexpr std::size_t n_paths = 50'000;

    std::mt19937_64 rng(42);
    const auto result = quant::mc_european_call_milstein(
        S, K, r, sigma, T, n_steps, n_paths, rng);

    INFO("Milstein estimate = " << result.estimate
         << ", BS = " << bs_price
         << ", half-width = " << result.half_width);

    REQUIRE(std::abs(result.estimate - bs_price)
            <= 3.0 * result.half_width);
}
