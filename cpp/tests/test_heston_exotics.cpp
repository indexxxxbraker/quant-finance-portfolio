// test_heston_exotics.cpp
//
// Catch2 tests for the Heston exotic Monte Carlo pricers (Asian,
// Lookback, Barrier) of Phase 4 Block 6.
//
// Structure mirrors validate_heston_exotics.py: limit cases against
// Fourier ground truth where applicable, structural bounds (Asian <
// European, etc.), statistical 1/sqrt(M) scaling, and discrete
// monitoring bias for lookback. Tests are statistical, with
// tolerances based on multiples of the half-width.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "heston_exotics.hpp"
#include "heston_fourier.hpp"
#include "monte_carlo.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

using namespace quant::heston;

namespace {

const HestonParams REF_PARAMS = {
    /*kappa*/ 1.5, /*theta*/ 0.04, /*sigma*/ 0.3,
    /*rho*/  -0.7, /*v0*/    0.04,
};
const double REF_S0 = 100.0;
const double REF_K  = 100.0;
const double REF_R  = 0.05;
const double REF_T  = 0.5;

bool within(const quant::MCResult& res, double reference, double n_sigma) {
    return std::abs(res.estimate - reference) < n_sigma * res.half_width;
}

}  // anonymous namespace


TEST_CASE("Asian call: limit n_avg=1 matches European",
          "[heston_exotics][asian][limit]") {
    // With n_avg=1, the single sample is at t=T, and the Asian payoff
    // reduces to max(S_T - K, 0), the European call. MC vs Fourier.
    std::mt19937_64 rng(42);
    const auto asian = mc_asian_call_heston(
        REF_S0, REF_K, REF_R, REF_PARAMS, REF_T,
        /*n_steps=*/100, /*n_paths=*/200000, rng, /*n_avg=*/1);

    const double C_eur = heston_call_lewis(REF_K, REF_T, REF_S0, REF_R,
                                             REF_PARAMS);
    INFO("Asian = " << asian.estimate << " +/- " << asian.half_width
         << ", European = " << C_eur);
    REQUIRE(within(asian, C_eur, 3.0));
}


TEST_CASE("Barrier call: limit H -> infty matches European",
          "[heston_exotics][barrier][limit]") {
    // With H very large, knockout never occurs and the payoff equals
    // the European call.
    std::mt19937_64 rng(42);
    const auto barrier = mc_barrier_call_heston(
        REF_S0, REF_K, /*H=*/1e6, REF_R, REF_PARAMS, REF_T,
        /*n_steps=*/100, /*n_paths=*/200000, rng);

    const double C_eur = heston_call_lewis(REF_K, REF_T, REF_S0, REF_R,
                                             REF_PARAMS);
    INFO("Barrier = " << barrier.estimate << " +/- " << barrier.half_width
         << ", European = " << C_eur);
    REQUIRE(within(barrier, C_eur, 3.0));
}


TEST_CASE("Asian call: less than European call (averaging reduces vol)",
          "[heston_exotics][asian][bound]") {
    std::mt19937_64 rng(42);
    const auto asian = mc_asian_call_heston(
        REF_S0, REF_K, REF_R, REF_PARAMS, REF_T,
        /*n_steps=*/100, /*n_paths=*/200000, rng, /*n_avg=*/50);

    const double C_eur = heston_call_lewis(REF_K, REF_T, REF_S0, REF_R,
                                             REF_PARAMS);
    INFO("Asian = " << asian.estimate << ", European = " << C_eur);
    // Strict: Asian < European by more than 3 half-widths
    REQUIRE(C_eur - asian.estimate > 3.0 * asian.half_width);
}


TEST_CASE("Barrier call: with finite H, less than European",
          "[heston_exotics][barrier][bound]") {
    std::mt19937_64 rng(42);
    const auto barrier = mc_barrier_call_heston(
        REF_S0, REF_K, /*H=*/130.0, REF_R, REF_PARAMS, REF_T,
        /*n_steps=*/100, /*n_paths=*/200000, rng);

    const double C_eur = heston_call_lewis(REF_K, REF_T, REF_S0, REF_R,
                                             REF_PARAMS);
    INFO("Barrier(H=130) = " << barrier.estimate
         << ", European = " << C_eur);
    REQUIRE(C_eur - barrier.estimate > 3.0 * barrier.half_width);
}


TEST_CASE("Statistical scaling 1/sqrt(M) for all three exotics",
          "[heston_exotics][stat_convergence]") {
    SECTION("Asian") {
        double hw_min = 1e30, hw_max = 0.0;
        for (std::size_t M : {std::size_t(10000),
                                std::size_t(40000),
                                std::size_t(160000)}) {
            std::mt19937_64 rng(42);
            const auto r = mc_asian_call_heston(
                REF_S0, REF_K, REF_R, REF_PARAMS, REF_T,
                /*n_steps=*/50, M, rng);
            const double v = r.half_width
                             * std::sqrt(static_cast<double>(M));
            if (v < hw_min) hw_min = v;
            if (v > hw_max) hw_max = v;
        }
        REQUIRE((hw_max - hw_min) / hw_min < 0.05);
    }

    SECTION("Lookback") {
        double hw_min = 1e30, hw_max = 0.0;
        for (std::size_t M : {std::size_t(10000),
                                std::size_t(40000),
                                std::size_t(160000)}) {
            std::mt19937_64 rng(42);
            const auto r = mc_lookback_call_heston(
                REF_S0, REF_R, REF_PARAMS, REF_T,
                /*n_steps=*/50, M, rng);
            const double v = r.half_width
                             * std::sqrt(static_cast<double>(M));
            if (v < hw_min) hw_min = v;
            if (v > hw_max) hw_max = v;
        }
        REQUIRE((hw_max - hw_min) / hw_min < 0.05);
    }

    SECTION("Barrier") {
        double hw_min = 1e30, hw_max = 0.0;
        for (std::size_t M : {std::size_t(10000),
                                std::size_t(40000),
                                std::size_t(160000)}) {
            std::mt19937_64 rng(42);
            const auto r = mc_barrier_call_heston(
                REF_S0, REF_K, /*H=*/130.0, REF_R, REF_PARAMS, REF_T,
                /*n_steps=*/50, M, rng);
            const double v = r.half_width
                             * std::sqrt(static_cast<double>(M));
            if (v < hw_min) hw_min = v;
            if (v > hw_max) hw_max = v;
        }
        REQUIRE((hw_max - hw_min) / hw_min < 0.05);
    }
}


TEST_CASE("Lookback discrete monitoring bias is positive",
          "[heston_exotics][lookback][bias]") {
    // Lookback price increases monotonically with n_steps because
    // the discrete minimum overestimates the continuous minimum.
    std::mt19937_64 rng_25(42);
    const auto r_25 = mc_lookback_call_heston(
        REF_S0, REF_R, REF_PARAMS, REF_T,
        /*n_steps=*/25, /*n_paths=*/100000, rng_25);

    std::mt19937_64 rng_200(42);
    const auto r_200 = mc_lookback_call_heston(
        REF_S0, REF_R, REF_PARAMS, REF_T,
        /*n_steps=*/200, /*n_paths=*/100000, rng_200);

    INFO("Lookback @ n=25: " << r_25.estimate
         << ", @ n=200: " << r_200.estimate);
    // Price should grow by at least 0.1 over this range
    REQUIRE(r_200.estimate - r_25.estimate > 0.1);
}


TEST_CASE("Heston exotics: validation errors",
          "[heston_exotics][validation]") {
    std::mt19937_64 rng(42);

    SECTION("Asian: invalid HestonParams") {
        HestonParams p_bad = REF_PARAMS;
        p_bad.kappa = -1.0;
        REQUIRE_THROWS_AS(
            mc_asian_call_heston(REF_S0, REF_K, REF_R, p_bad, REF_T,
                                   100, 1000, rng),
            std::invalid_argument);
    }
    SECTION("Lookback: non-positive S0") {
        REQUIRE_THROWS_AS(
            mc_lookback_call_heston(-100.0, REF_R, REF_PARAMS, REF_T,
                                       100, 1000, rng),
            std::invalid_argument);
    }
    SECTION("Barrier: H <= S0") {
        REQUIRE_THROWS_AS(
            mc_barrier_call_heston(REF_S0, REF_K, /*H=*/100.0, REF_R,
                                       REF_PARAMS, REF_T,
                                       100, 1000, rng),
            std::invalid_argument);
        REQUIRE_THROWS_AS(
            mc_barrier_call_heston(REF_S0, REF_K, /*H=*/50.0, REF_R,
                                       REF_PARAMS, REF_T,
                                       100, 1000, rng),
            std::invalid_argument);
    }
    SECTION("Asian: non-positive K") {
        REQUIRE_THROWS_AS(
            mc_asian_call_heston(REF_S0, -100.0, REF_R, REF_PARAMS, REF_T,
                                   100, 1000, rng),
            std::invalid_argument);
    }
}
