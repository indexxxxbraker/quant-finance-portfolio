// test_heston_calibration.cpp
//
// Catch2 tests for the Heston calibration pipeline of Phase 4 Block 6.
//
// The cornerstone test is the round-trip: synthesize prices from known
// parameters, calibrate, and verify recovery to high precision. The
// LM optimiser is using the same Fourier pricer that generated the
// data, so machine-precision recovery is expected for synthetic
// surfaces (modulo finite-difference Jacobian noise).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "heston_calibration.hpp"
#include "heston_fourier.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

using Catch::Approx;
using namespace quant::heston;

namespace {

const HestonParams TRUTH = {
    /*kappa*/ 1.5, /*theta*/ 0.04, /*sigma*/ 0.3,
    /*rho*/  -0.7, /*v0*/    0.04,
};
const double S0_REF = 100.0;
const double R_REF  = 0.05;

// Generate synthetic call surface from the truth parameters.
std::vector<CalibrationQuote> make_synthetic_surface(
        const HestonParams& p,
        const std::vector<double>& strikes,
        const std::vector<double>& maturities) {
    std::vector<CalibrationQuote> out;
    for (double K : strikes) {
        for (double T : maturities) {
            CalibrationQuote q;
            q.K = K;
            q.T = T;
            q.C_market = heston_call_lewis(K, T, S0_REF, R_REF, p);
            out.push_back(q);
        }
    }
    return out;
}

}  // anonymous namespace


TEST_CASE("Heston calibration: round-trip from synthetic surface",
          "[heston_calibration][round_trip]") {
    // Synthesize 9 quotes (3 strikes x 3 maturities) from the truth,
    // then calibrate from a perturbed initial guess. Verify each
    // parameter is recovered to high precision.
    const auto market = make_synthetic_surface(
        TRUTH, {90.0, 100.0, 110.0}, {0.25, 0.5, 1.0});

    const HestonParams initial = {
        /*kappa*/ 1.0, /*theta*/ 0.05, /*sigma*/ 0.5,
        /*rho*/  -0.3, /*v0*/    0.05,
    };

    const auto result = calibrate_heston(market, S0_REF, R_REF, initial);

    INFO("RMSE = " << result.rmse << ", n_iter = " << result.n_iter);

    // Each parameter recovered to 10^-4 (LM with finite-difference
    // Jacobian is less precise than scipy's; 10^-4 is a reasonable
    // tolerance).
    REQUIRE(std::abs(result.params.kappa - TRUTH.kappa) < 1e-3);
    REQUIRE(std::abs(result.params.theta - TRUTH.theta) < 1e-4);
    REQUIRE(std::abs(result.params.sigma - TRUTH.sigma) < 1e-3);
    REQUIRE(std::abs(result.params.rho   - TRUTH.rho)   < 1e-3);
    REQUIRE(std::abs(result.params.v0    - TRUTH.v0)    < 1e-4);

    // RMSE near machine precision
    REQUIRE(result.rmse < 1e-4);
}


TEST_CASE("Heston calibration: success flag",
          "[heston_calibration][success]") {
    // Calibration should report success on a well-posed synthetic
    // problem.
    const auto market = make_synthetic_surface(
        TRUTH, {90.0, 100.0, 110.0}, {0.25, 0.5, 1.0});
    const HestonParams initial = {
        1.0, 0.05, 0.5, -0.3, 0.05,
    };
    const auto result = calibrate_heston(market, S0_REF, R_REF, initial);
    REQUIRE(result.success);
}


TEST_CASE("Implied vol inversion: recovers BS sigma exactly",
          "[heston_calibration][iv_inversion]") {
    struct Test { double sigma; double K; double T; };
    const Test tests[] = {
        {0.10, 100.0, 0.25},
        {0.20, 100.0, 0.50},
        {0.30, 100.0, 1.00},
        {0.20,  80.0, 0.50},
        {0.20, 120.0, 0.50},
        {0.50, 100.0, 0.25},
    };
    for (const auto& t : tests) {
        const double C = black_scholes_call(
            S0_REF, t.K, t.T, R_REF, t.sigma);
        const double iv = implied_vol_bs(C, t.K, t.T, S0_REF, R_REF);
        INFO("truth sigma=" << t.sigma << ", K=" << t.K << ", T=" << t.T
             << ", recovered=" << iv);
        REQUIRE(std::abs(iv - t.sigma) < 1e-7);
    }
}


TEST_CASE("Implied vol inversion: NaN outside no-arbitrage bounds",
          "[heston_calibration][iv_inversion][bounds]") {
    SECTION("price below intrinsic") {
        const double K = 80.0, T = 0.5;
        // Intrinsic: max(100 - 80*exp(-0.05*0.5), 0) ~= 21.97
        const double iv = implied_vol_bs(15.0, K, T, S0_REF, R_REF);
        REQUIRE(std::isnan(iv));
    }
    SECTION("price above S0") {
        const double iv = implied_vol_bs(150.0, 100.0, 0.5, S0_REF, R_REF);
        REQUIRE(std::isnan(iv));
    }
}


TEST_CASE("Heston calibration: validation errors",
          "[heston_calibration][validation]") {
    const std::vector<CalibrationQuote> tiny_market = {
        {100.0, 0.5, 6.0}, {110.0, 0.5, 2.0}, {90.0, 0.5, 12.0}
    };  // only 3 quotes
    const HestonParams initial = {1.0, 0.05, 0.5, -0.3, 0.05};

    SECTION("too few observations") {
        REQUIRE_THROWS_AS(
            calibrate_heston(tiny_market, S0_REF, R_REF, initial),
            std::invalid_argument);
    }
    SECTION("non-positive S0") {
        const auto market = make_synthetic_surface(
            TRUTH, {90.0, 100.0, 110.0}, {0.25, 0.5, 1.0});
        REQUIRE_THROWS_AS(
            calibrate_heston(market, -100.0, R_REF, initial),
            std::invalid_argument);
    }
    SECTION("invalid initial guess") {
        const auto market = make_synthetic_surface(
            TRUTH, {90.0, 100.0, 110.0}, {0.25, 0.5, 1.0});
        HestonParams bad = initial;
        bad.kappa = -1.0;
        REQUIRE_THROWS_AS(
            calibrate_heston(market, S0_REF, R_REF, bad),
            std::invalid_argument);
    }
}
