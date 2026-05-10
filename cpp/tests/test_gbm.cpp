// test_gbm.cpp -- Catch2 tests for the gbm module (validators, RNG
// utilities, exact / Euler / Milstein samplers).
//
// Coverage organised in five groups:
//
//   1. Validators rejecting bad inputs in isolation.
//   2. inverse_normal_cdf hitting known textbook values, symmetry
//      identity, and throwing on out-of-domain inputs.
//   3. standard_normal empirical moments matching N(0, 1) within
//      five standard errors at n = 1e5.
//   4. simulate_terminal_gbm reproducing closed-form moments of S_T
//      and of log(S_T / S_0) within five standard errors at
//      n = 2e5; simulate_terminal_euler and simulate_terminal_milstein
//      converging to the same closed-form mean of S_T at fine grids.
//   5. Path samplers producing the expected shape (n_paths rows,
//      n_steps + 1 columns, starting at S_0).
//   6. Determinism: identical seeds yield bit-exact identical
//      sample sequences and vectors for every sampler.
//
// The statistical tests use 5-sigma tolerances. Under the (correct)
// null, P(|X - mu| > 5 sigma) ~ 5.7e-7, well below any practical
// CI flake threshold.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "gbm.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;


// =====================================================================
// Group 1: validators in isolation
// =====================================================================

TEST_CASE("gbm validators reject bad inputs in isolation",
          "[gbm][validators]") {
    SECTION("validate_model_params") {
        REQUIRE_THROWS_AS(quant::validate_model_params(-1.0, 0.20, 1.0),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(quant::validate_model_params(0.0, 0.20, 1.0),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(quant::validate_model_params(100.0, -0.20, 1.0),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(quant::validate_model_params(100.0, 0.0, 1.0),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(quant::validate_model_params(100.0, 0.20, -1.0),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(quant::validate_model_params(100.0, 0.20, 0.0),
                          std::invalid_argument);
        REQUIRE_NOTHROW(quant::validate_model_params(100.0, 0.20, 1.0));
    }

    SECTION("validate_strike") {
        REQUIRE_THROWS_AS(quant::validate_strike(-100.0),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(quant::validate_strike(0.0),
                          std::invalid_argument);
        REQUIRE_NOTHROW(quant::validate_strike(100.0));
    }

    SECTION("validate_n_paths") {
        REQUIRE_THROWS_AS(quant::validate_n_paths(0),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(quant::validate_n_paths(1),
                          std::invalid_argument);
        REQUIRE_NOTHROW(quant::validate_n_paths(2));
        REQUIRE_NOTHROW(quant::validate_n_paths(100'000));
    }

    SECTION("validate_n_steps") {
        REQUIRE_THROWS_AS(quant::validate_n_steps(0),
                          std::invalid_argument);
        REQUIRE_NOTHROW(quant::validate_n_steps(1));
        REQUIRE_NOTHROW(quant::validate_n_steps(1000));
    }

    SECTION("validate_confidence_level") {
        REQUIRE_THROWS_AS(quant::validate_confidence_level(0.0),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(quant::validate_confidence_level(1.0),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(quant::validate_confidence_level(-0.10),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(quant::validate_confidence_level(1.10),
                          std::invalid_argument);
        REQUIRE_NOTHROW(quant::validate_confidence_level(0.95));
        REQUIRE_NOTHROW(quant::validate_confidence_level(0.5));
    }

    SECTION("negative r is admissible everywhere it appears") {
        // r is unconstrained; verify by indirect call through samplers
        // (validate_model_params does not take r).
        std::mt19937_64 rng(42);
        REQUIRE_NOTHROW(quant::simulate_terminal_gbm(
                            100.0, -0.02, 0.20, 1.0, 1000, rng));
    }
}


// =====================================================================
// Group 2: inverse_normal_cdf reference values, symmetry, validation
// =====================================================================

TEST_CASE("inverse_normal_cdf: reference quantile values",
          "[gbm][inverse_normal_cdf]") {
    // Acklam claims relative error < 1.15e-9. Tolerance 1e-8 is safe.
    constexpr double TOL = 1e-8;
    REQUIRE_THAT(quant::inverse_normal_cdf(0.5),
                 WithinAbs(0.0, TOL));
    REQUIRE_THAT(quant::inverse_normal_cdf(0.975),
                 WithinAbs( 1.959963984540054, TOL));
    REQUIRE_THAT(quant::inverse_normal_cdf(0.025),
                 WithinAbs(-1.959963984540054, TOL));
    REQUIRE_THAT(quant::inverse_normal_cdf(0.99),
                 WithinAbs( 2.326347874040841, TOL));
    REQUIRE_THAT(quant::inverse_normal_cdf(0.01),
                 WithinAbs(-2.326347874040841, TOL));
    REQUIRE_THAT(quant::inverse_normal_cdf(0.999),
                 WithinAbs( 3.090232306167814, TOL));
    REQUIRE_THAT(quant::inverse_normal_cdf(0.001),
                 WithinAbs(-3.090232306167814, TOL));
}


TEST_CASE("inverse_normal_cdf: symmetry inv(p) = -inv(1 - p)",
          "[gbm][inverse_normal_cdf][symmetry]") {
    constexpr double TOL = 1e-9;
    for (double p : {0.001, 0.05, 0.10, 0.30, 0.45}) {
        const double a = quant::inverse_normal_cdf(p);
        const double b = quant::inverse_normal_cdf(1.0 - p);
        REQUIRE_THAT(a, WithinAbs(-b, TOL));
    }
}


TEST_CASE("inverse_normal_cdf: throws on boundary and out-of-domain",
          "[gbm][inverse_normal_cdf][validation]") {
    REQUIRE_THROWS_AS(quant::inverse_normal_cdf( 0.0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(quant::inverse_normal_cdf( 1.0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(quant::inverse_normal_cdf(-0.1),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(quant::inverse_normal_cdf( 1.1),
                      std::invalid_argument);
}


// =====================================================================
// Group 3: standard_normal distributional checks
// =====================================================================

TEST_CASE("standard_normal: empirical mean and variance match N(0, 1)",
          "[gbm][standard_normal][stats]") {
    // SE(sample mean) = 1 / sqrt(n).
    // SE(sample variance) for normal ~ sqrt(2 / n) (asymptotic).
    constexpr std::size_t n = 100'000;
    std::mt19937_64 rng(42);

    double sum = 0.0, sum_sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double z = quant::standard_normal(rng);
        sum    += z;
        sum_sq += z * z;
    }
    const double mean = sum / static_cast<double>(n);
    const double var  = sum_sq / static_cast<double>(n) - mean * mean;

    const double se_mean = 1.0 / std::sqrt(static_cast<double>(n));
    const double se_var  = std::sqrt(2.0 / static_cast<double>(n));

    INFO("mean = " << mean << " (5 SE = " << 5.0 * se_mean << ")");
    INFO("var  = " << var  << " (5 SE = " << 5.0 * se_var  << ")");
    REQUIRE(std::abs(mean)        < 5.0 * se_mean);
    REQUIRE(std::abs(var - 1.0)   < 5.0 * se_var);
}


// =====================================================================
// Group 4: sampler moment-matching and convergence
// =====================================================================

namespace {
inline double mean_of(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}
}  // namespace


TEST_CASE("simulate_terminal_gbm: sample mean matches S0 exp(rT) within 5 SE",
          "[gbm][exact][stats]") {
    constexpr double S0 = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    constexpr std::size_t n = 200'000;
    std::mt19937_64 rng(42);

    const auto S_T = quant::simulate_terminal_gbm(S0, r, sigma, T, n, rng);
    REQUIRE(S_T.size() == n);

    // Theoretical mean and standard deviation of S_T under risk-neutral GBM.
    const double mean_theo = S0 * std::exp(r * T);
    const double sd_theo   = S0 * std::exp(r * T)
                              * std::sqrt(std::exp(sigma * sigma * T) - 1.0);
    const double se        = sd_theo / std::sqrt(static_cast<double>(n));

    const double mean_emp  = mean_of(S_T);

    INFO("mean_emp = "  << mean_emp
         << ", mean_theo = " << mean_theo
         << ", 5 SE = " << 5.0 * se);
    REQUIRE(std::abs(mean_emp - mean_theo) < 5.0 * se);
}


TEST_CASE("simulate_terminal_gbm: log-returns are N((r - sigma^2/2)T, sigma^2 T)",
          "[gbm][exact][log_returns]") {
    constexpr double S0 = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    constexpr std::size_t n = 200'000;
    std::mt19937_64 rng(42);

    const auto S_T = quant::simulate_terminal_gbm(S0, r, sigma, T, n, rng);

    double sum = 0.0, sum_sq = 0.0;
    for (double x : S_T) {
        const double lx = std::log(x / S0);
        sum    += lx;
        sum_sq += lx * lx;
    }
    const double mean_emp = sum / static_cast<double>(n);
    const double var_emp  = sum_sq / static_cast<double>(n)
                              - mean_emp * mean_emp;

    const double mean_theo = (r - 0.5 * sigma * sigma) * T;
    const double var_theo  = sigma * sigma * T;

    // log-returns are exactly Gaussian here, so sample variance has
    // SE = var_theo * sqrt(2 / n) and sample mean has SE = sqrt(var_theo / n).
    const double se_mean = std::sqrt(var_theo / static_cast<double>(n));
    const double se_var  = var_theo
                              * std::sqrt(2.0 / static_cast<double>(n));

    INFO("mean: emp=" << mean_emp << " theo=" << mean_theo
         << " 5 SE=" << 5.0 * se_mean);
    INFO("var:  emp=" << var_emp  << " theo=" << var_theo
         << " 5 SE=" << 5.0 * se_var);
    REQUIRE(std::abs(mean_emp - mean_theo) < 5.0 * se_mean);
    REQUIRE(std::abs(var_emp  - var_theo)  < 5.0 * se_var);
}


TEST_CASE("simulate_terminal_euler: converges to exact mean at fine grid",
          "[gbm][euler][convergence]") {
    constexpr double S0 = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    constexpr std::size_t n = 200'000;
    constexpr std::size_t n_steps = 500;
    std::mt19937_64 rng(43);

    const auto S_T = quant::simulate_terminal_euler(
                          S0, r, sigma, T, n_steps, n, rng);
    REQUIRE(S_T.size() == n);

    const double mean_theo = S0 * std::exp(r * T);
    const double sd_theo   = S0 * std::exp(r * T)
                              * std::sqrt(std::exp(sigma * sigma * T) - 1.0);
    const double se        = sd_theo / std::sqrt(static_cast<double>(n));

    const double mean_emp  = mean_of(S_T);

    INFO("Euler mean: emp=" << mean_emp << " theo=" << mean_theo
         << " 5 SE=" << 5.0 * se);
    REQUIRE(std::abs(mean_emp - mean_theo) < 5.0 * se);
}


TEST_CASE("simulate_terminal_milstein: converges to exact mean at fine grid",
          "[gbm][milstein][convergence]") {
    constexpr double S0 = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    constexpr std::size_t n = 200'000;
    constexpr std::size_t n_steps = 500;
    std::mt19937_64 rng(44);

    const auto S_T = quant::simulate_terminal_milstein(
                          S0, r, sigma, T, n_steps, n, rng);
    REQUIRE(S_T.size() == n);

    const double mean_theo = S0 * std::exp(r * T);
    const double sd_theo   = S0 * std::exp(r * T)
                              * std::sqrt(std::exp(sigma * sigma * T) - 1.0);
    const double se        = sd_theo / std::sqrt(static_cast<double>(n));

    const double mean_emp  = mean_of(S_T);

    INFO("Milstein mean: emp=" << mean_emp << " theo=" << mean_theo
         << " 5 SE=" << 5.0 * se);
    REQUIRE(std::abs(mean_emp - mean_theo) < 5.0 * se);
}


// =====================================================================
// Group 5: path samplers shape and starting value
// =====================================================================

TEST_CASE("path samplers produce (n_paths x n_steps+1) shape starting at S0",
          "[gbm][path][shape]") {
    constexpr double S0 = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    constexpr std::size_t n = 100;
    constexpr std::size_t n_steps = 50;

    SECTION("simulate_path_euler") {
        std::mt19937_64 rng(42);
        const auto paths = quant::simulate_path_euler(
                              S0, r, sigma, T, n_steps, n, rng);
        REQUIRE(paths.size() == n);
        REQUIRE(paths.front().size() == n_steps + 1);
        REQUIRE(paths.back().size()  == n_steps + 1);
        REQUIRE(paths[0][0] == S0);
        REQUIRE(paths[n - 1][0] == S0);
    }

    SECTION("simulate_path_milstein") {
        std::mt19937_64 rng(42);
        const auto paths = quant::simulate_path_milstein(
                              S0, r, sigma, T, n_steps, n, rng);
        REQUIRE(paths.size() == n);
        REQUIRE(paths.front().size() == n_steps + 1);
        REQUIRE(paths.back().size()  == n_steps + 1);
        REQUIRE(paths[0][0] == S0);
        REQUIRE(paths[n - 1][0] == S0);
    }
}


// =====================================================================
// Group 6: determinism (identical seed -> identical output)
// =====================================================================

TEST_CASE("samplers are deterministic with respect to seed",
          "[gbm][determinism]") {
    constexpr double S0 = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    constexpr std::size_t n = 5'000;
    constexpr std::size_t n_steps = 50;

    SECTION("standard_normal: identical sequences") {
        std::mt19937_64 rng_a(7), rng_b(7);
        for (int i = 0; i < 100; ++i) {
            REQUIRE(quant::standard_normal(rng_a)
                    == quant::standard_normal(rng_b));
        }
    }

    SECTION("simulate_terminal_gbm: identical vectors") {
        std::mt19937_64 rng_a(7), rng_b(7);
        const auto a = quant::simulate_terminal_gbm(
                            S0, r, sigma, T, n, rng_a);
        const auto b = quant::simulate_terminal_gbm(
                            S0, r, sigma, T, n, rng_b);
        REQUIRE(a == b);
    }

    SECTION("simulate_terminal_euler: identical vectors") {
        std::mt19937_64 rng_a(7), rng_b(7);
        const auto a = quant::simulate_terminal_euler(
                            S0, r, sigma, T, n_steps, n, rng_a);
        const auto b = quant::simulate_terminal_euler(
                            S0, r, sigma, T, n_steps, n, rng_b);
        REQUIRE(a == b);
    }

    SECTION("simulate_terminal_milstein: identical vectors") {
        std::mt19937_64 rng_a(7), rng_b(7);
        const auto a = quant::simulate_terminal_milstein(
                            S0, r, sigma, T, n_steps, n, rng_a);
        const auto b = quant::simulate_terminal_milstein(
                            S0, r, sigma, T, n_steps, n, rng_b);
        REQUIRE(a == b);
    }

    SECTION("simulate_path_euler: identical matrices") {
        std::mt19937_64 rng_a(7), rng_b(7);
        const auto a = quant::simulate_path_euler(
                            S0, r, sigma, T, n_steps, 200, rng_a);
        const auto b = quant::simulate_path_euler(
                            S0, r, sigma, T, n_steps, 200, rng_b);
        REQUIRE(a == b);
    }

    SECTION("simulate_path_milstein: identical matrices") {
        std::mt19937_64 rng_a(7), rng_b(7);
        const auto a = quant::simulate_path_milstein(
                            S0, r, sigma, T, n_steps, 200, rng_a);
        const auto b = quant::simulate_path_milstein(
                            S0, r, sigma, T, n_steps, 200, rng_b);
        REQUIRE(a == b);
    }
}
