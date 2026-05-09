// test_heston_mc.cpp
//
// Catch2 tests for the Heston Monte Carlo pricer (full-truncation Euler)
// of Phase 4 Block 3.
//
// Tests are statistical: we cannot compare a Monte Carlo estimate
// bit-for-bit against any other implementation (different RNGs and
// different sampling orders). Instead, we assert that the estimate
// falls within a small multiple of its half-width from a bias-free
// Fourier reference. With a fixed seed and parameter set, this is a
// deterministic check; with the half-widths the suite reports, the
// false-positive rate is well below 1%.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "heston_fourier.hpp"
#include "heston_mc.hpp"
#include "monte_carlo.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

using Catch::Approx;
using namespace quant::heston;

namespace {

// Standard equity parameter set; matches Python validation suite.
// Feller parameter nu = 2 kappa theta / sigma^2 = 1.333.
const HestonParams REF_PARAMS = {
    /*kappa*/ 1.5,
    /*theta*/ 0.04,
    /*sigma*/ 0.3,
    /*rho*/  -0.7,
    /*v0*/    0.04,
};
const double REF_S0 = 100.0;
const double REF_R  = 0.05;

// Aggressive parameter set; Feller nu = 0.04, near-singular boundary.
const HestonParams AGGRESSIVE_PARAMS = {
    /*kappa*/ 0.5,
    /*theta*/ 0.04,
    /*sigma*/ 1.0,
    /*rho*/  -0.9,
    /*v0*/    0.04,
};

// Returns true iff |estimate - reference| < n_sigma * half_width.
bool within(const quant::MCResult& res, double reference, double n_sigma) {
    return std::abs(res.estimate - reference) < n_sigma * res.half_width;
}

}  // anonymous namespace


TEST_CASE("Heston MC: BS limit (sigma -> 0)",
          "[heston_mc][bs_limit]") {
    // As sigma -> 0 with v0 = theta, Heston reduces to BS at constant
    // vol sqrt(v0). At sigma = 0.01, the Heston-BS gap is dominated by
    // the linear-in-sigma leverage correction; at the MC half-widths
    // used here, agreement within ~3 half-widths is the right standard.
    const double T = 0.5;
    const double K = 100.0;
    const double C_bs = black_scholes_call(REF_S0, K, T, REF_R,
                                             std::sqrt(REF_PARAMS.v0));

    HestonParams p = REF_PARAMS;
    p.sigma = 0.01;
    std::mt19937_64 rng(42);
    const auto res = mc_european_call_heston(
        REF_S0, K, REF_R, p, T, /*n_steps=*/200, /*n_paths=*/200000, rng);

    REQUIRE(within(res, C_bs, /*n_sigma=*/3.0));
}


TEST_CASE("Heston MC: agreement with Fourier (standard parameters)",
          "[heston_mc][cross_method]") {
    // Snapshot Fourier prices computed by Python heston_call_lewis
    // (ground truth, bias < 1e-6). The C++ MC must agree within
    // 3 half-widths.
    struct Snapshot { double K; double T; double C_fourier; };
    const Snapshot snapshots[] = {
        { 90.0, 0.25, 11.8920828594},
        { 90.0, 0.50, 13.8369446176},
        { 90.0, 1.00, 17.1069368611},
        {100.0, 0.25,  4.5842062306},
        {100.0, 0.50,  6.8257341234},
        {100.0, 1.00, 10.3618690210},
        {110.0, 0.25,  0.8483767762},
        {110.0, 0.50,  2.3340705240},
        {110.0, 1.00,  5.3179531129},
    };

    for (const auto& s : snapshots) {
        std::mt19937_64 rng(42);
        const auto res = mc_european_call_heston(
            REF_S0, s.K, REF_R, REF_PARAMS, s.T,
            /*n_steps=*/200, /*n_paths=*/200000, rng);

        // 3 half-widths gives ~99.7% asymptotic coverage; we add a
        // small absolute slack (1e-3) to absorb the residual
        // discretisation bias on top of the statistical noise.
        const double tol = 3.0 * res.half_width + 1e-3;
        const double err = std::abs(res.estimate - s.C_fourier);
        INFO("K=" << s.K << " T=" << s.T
             << ", MC=" << res.estimate
             << ", Fourier=" << s.C_fourier
             << ", err=" << err
             << ", tol=" << tol);
        REQUIRE(err < tol);
    }
}


TEST_CASE("Heston MC: agreement with Fourier (aggressive parameters)",
          "[heston_mc][cross_method][aggressive]") {
    // Aggressive parameter set: low Feller parameter (0.04) means the
    // variance process spends measurable time near zero, where the
    // full-truncation scheme genuinely fires. Discretisation bias is
    // larger here, so we relax the tolerance to 4 half-widths.
    struct Snapshot { double T; double C_fourier; };
    const Snapshot snapshots[] = {
        {0.5, 5.5663688555},
        {1.0, 8.3248140286},
    };

    for (const auto& s : snapshots) {
        std::mt19937_64 rng(42);
        const auto res = mc_european_call_heston(
            REF_S0, /*K=*/100.0, REF_R, AGGRESSIVE_PARAMS, s.T,
            /*n_steps=*/200, /*n_paths=*/200000, rng);

        const double tol = 4.0 * res.half_width + 5e-3;
        const double err = std::abs(res.estimate - s.C_fourier);
        INFO("T=" << s.T
             << ", MC=" << res.estimate
             << ", Fourier=" << s.C_fourier
             << ", err=" << err);
        REQUIRE(err < tol);
    }
}


TEST_CASE("Heston MC: half_width scales as 1/sqrt(n_paths)",
          "[heston_mc][stat_convergence]") {
    // The half-width should decrease as 1/sqrt(M); equivalently,
    // half_width * sqrt(M) should be approximately constant across
    // sample sizes. We allow 5% deviation, which is well within the
    // typical Monte Carlo asymptotic regime at the sizes tested.
    const double T = 0.5;
    const double K = 100.0;

    double hw_x_sqrtM_min = 1e30, hw_x_sqrtM_max = 0.0;
    for (std::size_t M : {std::size_t(10000),
                            std::size_t(40000),
                            std::size_t(160000)}) {
        std::mt19937_64 rng(42);
        const auto res = mc_european_call_heston(
            REF_S0, K, REF_R, REF_PARAMS, T,
            /*n_steps=*/100, M, rng);
        const double v = res.half_width * std::sqrt(static_cast<double>(M));
        if (v < hw_x_sqrtM_min) hw_x_sqrtM_min = v;
        if (v > hw_x_sqrtM_max) hw_x_sqrtM_max = v;
    }
    const double rel_spread = (hw_x_sqrtM_max - hw_x_sqrtM_min)
                              / hw_x_sqrtM_min;
    INFO("HW * sqrt(M) ranged in ["
         << hw_x_sqrtM_min << ", " << hw_x_sqrtM_max << "], "
         << "relative spread " << rel_spread);
    REQUIRE(rel_spread < 0.05);
}


TEST_CASE("Heston MC: antithetic reduces variance",
          "[heston_mc][antithetic]") {
    const double T = 0.5;
    const double K = 100.0;
    const std::size_t n_paths = 100000;
    const std::size_t n_steps = 200;

    std::mt19937_64 rng_plain(42);
    const auto plain = mc_european_call_heston(
        REF_S0, K, REF_R, REF_PARAMS, T, n_steps, n_paths,
        rng_plain, /*antithetic=*/false);

    std::mt19937_64 rng_anti(42);
    const auto anti = mc_european_call_heston(
        REF_S0, K, REF_R, REF_PARAMS, T, n_steps, n_paths,
        rng_anti, /*antithetic=*/true);

    SECTION("variance reduction at least 2x") {
        const double ratio = plain.sample_variance / anti.sample_variance;
        INFO("variance reduction: " << ratio << "x");
        REQUIRE(ratio > 2.0);
    }

    SECTION("plain and antithetic estimates statistically consistent") {
        const double err = std::abs(plain.estimate - anti.estimate);
        const double combined_hw = plain.half_width + anti.half_width;
        INFO("|plain - anti| = " << err
             << ", combined HW = " << combined_hw);
        REQUIRE(err < combined_hw);
    }

    SECTION("both estimates agree with Fourier reference") {
        const double C_ref = heston_call_lewis(K, T, REF_S0, REF_R,
                                                 REF_PARAMS);
        REQUIRE(within(plain, C_ref, 3.0));
        REQUIRE(within(anti, C_ref, 3.0));
    }

    SECTION("antithetic returns half the path count") {
        // With antithetic the M paths are paired, so the estimator
        // sees M/2 i.i.d. samples; this is what the MCResult reports.
        REQUIRE(plain.n_paths == n_paths);
        REQUIRE(anti.n_paths == n_paths / 2);
    }
}


TEST_CASE("Heston MC: simulate_heston_paths returns correct shape",
          "[heston_mc][shape]") {
    const double T = 0.5;
    const std::size_t n_steps = 50;
    const std::size_t n_paths = 100;

    std::mt19937_64 rng(42);
    const auto paths = simulate_heston_paths(
        REF_S0, REF_R, REF_PARAMS, T, n_steps, n_paths, rng);

    REQUIRE(paths.log_S.size() == n_paths);
    REQUIRE(paths.v.size() == n_paths);
    REQUIRE(paths.log_S[0].size() == n_steps + 1);
    REQUIRE(paths.v[0].size() == n_steps + 1);

    // Initial conditions correctly set on every path.
    for (std::size_t i = 0; i < n_paths; ++i) {
        REQUIRE(paths.log_S[i][0] == Approx(std::log(REF_S0)).margin(1e-12));
        REQUIRE(paths.v[i][0] == Approx(REF_PARAMS.v0).margin(1e-12));
    }

    // All log_S values are finite.
    for (std::size_t i = 0; i < n_paths; ++i) {
        for (std::size_t k = 0; k <= n_steps; ++k) {
            REQUIRE(std::isfinite(paths.log_S[i][k]));
            REQUIRE(std::isfinite(paths.v[i][k]));
        }
    }
}


TEST_CASE("Heston MC: simulate_terminal_heston matches simulate_heston_paths",
          "[heston_mc][consistency]") {
    // With the same seed, the terminal-only simulator must produce
    // the same S_T as the full path simulator's last column. Both use
    // the same RNG sequence (Z1, Z2, Z1, Z2, ...) per path, so they
    // are bit-for-bit equivalent. This test catches accidental
    // divergence between the two code paths during refactors.
    const double T = 0.5;
    const std::size_t n_steps = 50;
    const std::size_t n_paths = 100;

    std::mt19937_64 rng_path(42);
    const auto paths = simulate_heston_paths(
        REF_S0, REF_R, REF_PARAMS, T, n_steps, n_paths, rng_path);

    std::mt19937_64 rng_term(42);
    const auto S_T = simulate_terminal_heston(
        REF_S0, REF_R, REF_PARAMS, T, n_steps, n_paths, rng_term);

    REQUIRE(S_T.size() == n_paths);
    for (std::size_t i = 0; i < n_paths; ++i) {
        const double S_T_from_path = std::exp(paths.log_S[i][n_steps]);
        REQUIRE(S_T[i] == Approx(S_T_from_path).margin(1e-12));
    }
}


TEST_CASE("Heston MC: validation errors",
          "[heston_mc][validation]") {
    std::mt19937_64 rng(42);

    SECTION("invalid HestonParams") {
        HestonParams p_bad = REF_PARAMS;
        p_bad.kappa = -1.0;
        REQUIRE_THROWS_AS(
            simulate_terminal_heston(REF_S0, REF_R, p_bad, 0.5, 100, 1000, rng),
            std::invalid_argument);
    }

    SECTION("non-positive S0") {
        REQUIRE_THROWS_AS(
            simulate_terminal_heston(-100.0, REF_R, REF_PARAMS, 0.5, 100, 1000, rng),
            std::invalid_argument);
    }

    SECTION("non-positive T") {
        REQUIRE_THROWS_AS(
            simulate_terminal_heston(REF_S0, REF_R, REF_PARAMS, -0.1, 100, 1000, rng),
            std::invalid_argument);
    }

    SECTION("antithetic with odd n_paths") {
        REQUIRE_THROWS_AS(
            simulate_terminal_heston(REF_S0, REF_R, REF_PARAMS, 0.5,
                                      100, 1001, rng, /*antithetic=*/true),
            std::invalid_argument);
    }

    SECTION("non-positive K") {
        REQUIRE_THROWS_AS(
            mc_european_call_heston(REF_S0, -100.0, REF_R, REF_PARAMS, 0.5,
                                      100, 1000, rng),
            std::invalid_argument);
    }
}
