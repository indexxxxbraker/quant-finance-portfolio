// test_heston_qe.cpp
//
// Catch2 tests for the Heston Andersen QE pricer of Phase 4 Block 4.
//
// Tests are statistical (different RNGs Python<->C++ make bit-for-bit
// comparison impossible). Tolerances are multiples of the half-width;
// reference values come from the bias-free Fourier pricer of Block 2.
//
// The fifth test case asserts the central empirical claim of Block 4:
// QE has substantially smaller bias than full-truncation Euler at the
// same (n_steps, n_paths), most dramatically in the low-Feller regime.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "heston_fourier.hpp"
#include "heston_mc.hpp"          // mc_european_call_heston (FT-Euler baseline)
#include "heston_qe.hpp"
#include "monte_carlo.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

using Catch::Approx;
using namespace quant::heston;

namespace {

// Standard equity parameter set; Feller nu = 1.333.
const HestonParams REF_PARAMS = {
    /*kappa*/ 1.5,
    /*theta*/ 0.04,
    /*sigma*/ 0.3,
    /*rho*/  -0.7,
    /*v0*/    0.04,
};
const double REF_S0 = 100.0;
const double REF_R  = 0.05;

// Aggressive equity parameter set; Feller nu = 0.04.
// This is the regime where QE's structural advantage over FT-Euler is
// largest (atom-at-zero handling).
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


TEST_CASE("Heston QE: BS limit (sigma -> 0)",
          "[heston_qe][bs_limit]") {
    const double T = 0.5;
    const double K = 100.0;
    const double C_bs = black_scholes_call(REF_S0, K, T, REF_R,
                                             std::sqrt(REF_PARAMS.v0));

    HestonParams p = REF_PARAMS;
    p.sigma = 0.01;
    std::mt19937_64 rng(42);
    const auto res = mc_european_call_heston_qe(
        REF_S0, K, REF_R, p, T, /*n_steps=*/100, /*n_paths=*/200000, rng);

    REQUIRE(within(res, C_bs, /*n_sigma=*/3.0));
}


TEST_CASE("Heston QE: agreement with Fourier (standard parameters)",
          "[heston_qe][cross_method]") {
    // Snapshots computed by Python heston_call_lewis (Block 2 ground
    // truth, bias < 1e-6). QE at n_steps=50 must agree within
    // 3 half-widths -- a stricter test than FT-Euler at the same
    // n_steps because QE has lower bias.
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
        const auto res = mc_european_call_heston_qe(
            REF_S0, s.K, REF_R, REF_PARAMS, s.T,
            /*n_steps=*/50, /*n_paths=*/200000, rng);

        const double tol = 3.0 * res.half_width + 1e-3;
        const double err = std::abs(res.estimate - s.C_fourier);
        INFO("K=" << s.K << " T=" << s.T
             << ", QE=" << res.estimate
             << ", Fourier=" << s.C_fourier
             << ", err=" << err
             << ", tol=" << tol);
        REQUIRE(err < tol);
    }
}


TEST_CASE("Heston QE: agreement with Fourier (aggressive parameters)",
          "[heston_qe][cross_method][aggressive]") {
    // Aggressive parameter set: low Feller (0.04) is the regime where
    // QE shines structurally. QE at n_steps=50 should already be very
    // close to Fourier; tolerance same 3 half-widths.
    struct Snapshot { double T; double C_fourier; };
    const Snapshot snapshots[] = {
        {0.25, 3.9515039164},
        {0.50, 5.5663688555},
        {1.00, 8.3248140286},
    };

    for (const auto& s : snapshots) {
        std::mt19937_64 rng(42);
        const auto res = mc_european_call_heston_qe(
            REF_S0, /*K=*/100.0, REF_R, AGGRESSIVE_PARAMS, s.T,
            /*n_steps=*/50, /*n_paths=*/200000, rng);

        const double tol = 3.0 * res.half_width + 1e-3;
        const double err = std::abs(res.estimate - s.C_fourier);
        INFO("T=" << s.T
             << ", QE=" << res.estimate
             << ", Fourier=" << s.C_fourier
             << ", err=" << err);
        REQUIRE(err < tol);
    }
}


TEST_CASE("Heston QE: half_width scales as 1/sqrt(n_paths)",
          "[heston_qe][stat_convergence]") {
    const double T = 0.5;
    const double K = 100.0;

    double hw_x_sqrtM_min = 1e30, hw_x_sqrtM_max = 0.0;
    for (std::size_t M : {std::size_t(10000),
                            std::size_t(40000),
                            std::size_t(160000)}) {
        std::mt19937_64 rng(42);
        const auto res = mc_european_call_heston_qe(
            REF_S0, K, REF_R, REF_PARAMS, T,
            /*n_steps=*/50, M, rng);
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


TEST_CASE("Heston QE: antithetic reduces variance",
          "[heston_qe][antithetic]") {
    const double T = 0.5;
    const double K = 100.0;
    const std::size_t n_paths = 100000;
    const std::size_t n_steps = 50;

    std::mt19937_64 rng_plain(42);
    const auto plain = mc_european_call_heston_qe(
        REF_S0, K, REF_R, REF_PARAMS, T, n_steps, n_paths,
        rng_plain, /*antithetic=*/false);

    std::mt19937_64 rng_anti(42);
    const auto anti = mc_european_call_heston_qe(
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
        REQUIRE(plain.n_paths == n_paths);
        REQUIRE(anti.n_paths == n_paths / 2);
    }
}


TEST_CASE("Heston QE: bias < FT-Euler bias (aggressive regime)",
          "[heston_qe][bias_comparison]") {
    // The central empirical claim of Block 4: QE has substantially
    // smaller bias than full-truncation Euler at the same (n_steps,
    // n_paths), with the gap most dramatic in the low-Feller regime.
    //
    // To resolve the bias from statistical noise we average over
    // multiple seeds. The expected ratio |bias_FTE / bias_QE| is in
    // the range 5-50x; we assert >= 5x as a conservative lower bound.
    const double T = 0.5;
    const double K = 100.0;
    const std::size_t n_steps = 50;
    const std::size_t n_paths = 200000;
    const int n_seeds = 5;

    const double C_ref = heston_call_lewis(K, T, REF_S0, REF_R,
                                             AGGRESSIVE_PARAMS);

    double qe_sum = 0.0, fte_sum = 0.0;
    for (int seed = 0; seed < n_seeds; ++seed) {
        std::mt19937_64 rng_qe(static_cast<unsigned>(seed));
        const auto qe = mc_european_call_heston_qe(
            REF_S0, K, REF_R, AGGRESSIVE_PARAMS, T, n_steps, n_paths, rng_qe);
        qe_sum += qe.estimate;

        std::mt19937_64 rng_fte(static_cast<unsigned>(seed));
        const auto fte = mc_european_call_heston(
            REF_S0, K, REF_R, AGGRESSIVE_PARAMS, T, n_steps, n_paths, rng_fte);
        fte_sum += fte.estimate;
    }
    const double qe_avg  = qe_sum / n_seeds;
    const double fte_avg = fte_sum / n_seeds;
    const double qe_bias  = std::abs(qe_avg - C_ref);
    const double fte_bias = std::abs(fte_avg - C_ref);
    const double ratio    = fte_bias / std::max(qe_bias, 1e-10);

    INFO("Fourier=" << C_ref
         << ", QE avg=" << qe_avg << " bias=" << qe_avg - C_ref
         << ", FTE avg=" << fte_avg << " bias=" << fte_avg - C_ref
         << ", |FTE/QE|=" << ratio << "x");

    SECTION("QE bias strictly less than FTE bias") {
        REQUIRE(qe_bias < fte_bias);
    }

    SECTION("QE bias reduction at least 5x") {
        REQUIRE(ratio >= 5.0);
    }
}


TEST_CASE("Heston QE: validation errors",
          "[heston_qe][validation]") {
    std::mt19937_64 rng(42);

    SECTION("invalid HestonParams") {
        HestonParams p_bad = REF_PARAMS;
        p_bad.kappa = -1.0;
        REQUIRE_THROWS_AS(
            simulate_terminal_heston_qe(REF_S0, REF_R, p_bad, 0.5,
                                          100, 1000, rng),
            std::invalid_argument);
    }

    SECTION("non-positive S0") {
        REQUIRE_THROWS_AS(
            simulate_terminal_heston_qe(-100.0, REF_R, REF_PARAMS, 0.5,
                                          100, 1000, rng),
            std::invalid_argument);
    }

    SECTION("non-positive T") {
        REQUIRE_THROWS_AS(
            simulate_terminal_heston_qe(REF_S0, REF_R, REF_PARAMS, -0.1,
                                          100, 1000, rng),
            std::invalid_argument);
    }

    SECTION("antithetic with odd n_paths") {
        REQUIRE_THROWS_AS(
            simulate_terminal_heston_qe(REF_S0, REF_R, REF_PARAMS, 0.5,
                                          100, 1001, rng,
                                          /*antithetic=*/true),
            std::invalid_argument);
    }

    SECTION("psi_c outside (1, 2)") {
        REQUIRE_THROWS_AS(
            simulate_terminal_heston_qe(REF_S0, REF_R, REF_PARAMS, 0.5,
                                          100, 1000, rng,
                                          /*antithetic=*/false,
                                          /*psi_c=*/0.5),
            std::invalid_argument);
        REQUIRE_THROWS_AS(
            simulate_terminal_heston_qe(REF_S0, REF_R, REF_PARAMS, 0.5,
                                          100, 1000, rng,
                                          /*antithetic=*/false,
                                          /*psi_c=*/2.5),
            std::invalid_argument);
    }

    SECTION("gamma1 + gamma2 != 1") {
        REQUIRE_THROWS_AS(
            simulate_terminal_heston_qe(REF_S0, REF_R, REF_PARAMS, 0.5,
                                          100, 1000, rng,
                                          /*antithetic=*/false,
                                          /*psi_c=*/1.5,
                                          /*gamma1=*/0.3,
                                          /*gamma2=*/0.3),
            std::invalid_argument);
    }

    SECTION("non-positive K") {
        REQUIRE_THROWS_AS(
            mc_european_call_heston_qe(REF_S0, -100.0, REF_R, REF_PARAMS, 0.5,
                                          100, 1000, rng),
            std::invalid_argument);
    }
}
