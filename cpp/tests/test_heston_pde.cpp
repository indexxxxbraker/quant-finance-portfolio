// test_heston_pde.cpp
//
// Catch2 tests for the Heston 2D PDE pricer (Douglas ADI) of Phase 4
// Block 5.
//
// Structure mirrors validate_heston_pde.py: BS limit, cross-method vs
// Fourier, spatial convergence with halving, and sanity vs QE Monte
// Carlo. Tests are deterministic (PDE is deterministic) so tolerances
// are absolute, not multiples of half-width.
//
// Grid sizes are smaller than in the Python validator to keep test
// runtime bounded (~5-10 seconds total). The asymptotic O(h^2)
// convergence is still clearly observable at these sizes.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "heston_fourier.hpp"
#include "heston_mc.hpp"            // for FT-Euler comparison (sanity)
#include "heston_pde.hpp"
#include "heston_qe.hpp"            // QE Monte Carlo for cross-method sanity
#include "monte_carlo.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

using Catch::Approx;
using namespace quant::heston;

namespace {

// Standard equity parameter set; Feller nu = 1.333 (well-posed regime).
const HestonParams REF_PARAMS = {
    /*kappa*/ 1.5,
    /*theta*/ 0.04,
    /*sigma*/ 0.3,
    /*rho*/  -0.7,
    /*v0*/    0.04,
};
const double REF_S0 = 100.0;
const double REF_R  = 0.05;

}  // anonymous namespace


TEST_CASE("Heston PDE: BS limit (sigma -> 0)",
          "[heston_pde][bs_limit]") {
    // As sigma -> 0 with v0 = theta, Heston PDE -> BS at constant vol
    // sqrt(v0). Tolerance is absolute (PDE has deterministic
    // discretisation error of order ~0.005 at these grids).
    const double T = 0.5;
    const double K = 100.0;
    const double C_bs = black_scholes_call(REF_S0, K, T, REF_R,
                                             std::sqrt(REF_PARAMS.v0));

    HestonParams p = REF_PARAMS;
    p.sigma = 0.01;
    const double price = heston_call_pde(REF_S0, K, T, REF_R, p,
                                            /*N_X=*/100, /*N_v=*/50,
                                            /*N_tau=*/100);
    const double err = std::abs(price - C_bs);
    INFO("PDE = " << price << ", BS = " << C_bs << ", err = " << err);
    REQUIRE(err < 0.05);
}


TEST_CASE("Heston PDE: agreement with Fourier (panel of K, T)",
          "[heston_pde][cross_method]") {
    // Snapshot Fourier prices computed by the Python heston_call_lewis
    // (Block 2 ground truth, bias < 1e-6).
    //
    // At grid (100, 50, 100) the PDE error is roughly 0.025. We use
    // a tolerance of 0.05 to accommodate this comfortably while still
    // catching gross regressions.
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
        const double price = heston_call_pde(REF_S0, s.K, s.T, REF_R,
                                                REF_PARAMS,
                                                /*N_X=*/100, /*N_v=*/50,
                                                /*N_tau=*/100);
        const double err = std::abs(price - s.C_fourier);
        INFO("K=" << s.K << " T=" << s.T
             << ", PDE=" << price
             << ", Fourier=" << s.C_fourier
             << ", err=" << err);
        REQUIRE(err < 0.05);
    }
}


TEST_CASE("Heston PDE: spatial convergence with halving",
          "[heston_pde][convergence]") {
    // The central theoretical claim of Block 5: the spatial
    // discretisation is O(dX^2 + dv^2). Halving (dX, dv) by doubling
    // (N_X, N_v) should reduce the error by approximately 4.
    //
    // We use a generous N_tau for all grids so the temporal error
    // (O(dtau)) doesn't pollute the spatial study.
    const double T = 0.5;
    const double K = 100.0;
    const std::size_t N_tau_fixed = 200;
    const double C_ref = heston_call_lewis(K, T, REF_S0, REF_R,
                                             REF_PARAMS);

    const double price_50  = heston_call_pde(REF_S0, K, T, REF_R, REF_PARAMS,
                                               50, 25, N_tau_fixed);
    const double price_100 = heston_call_pde(REF_S0, K, T, REF_R, REF_PARAMS,
                                               100, 50, N_tau_fixed);
    const double price_200 = heston_call_pde(REF_S0, K, T, REF_R, REF_PARAMS,
                                               200, 100, N_tau_fixed);

    const double err_50  = std::abs(price_50  - C_ref);
    const double err_100 = std::abs(price_100 - C_ref);
    const double err_200 = std::abs(price_200 - C_ref);

    INFO("Errors: 50=" << err_50
         << ", 100=" << err_100
         << ", 200=" << err_200);

    SECTION("error decreases monotonically with grid refinement") {
        REQUIRE(err_100 < err_50);
        REQUIRE(err_200 < err_100);
    }

    SECTION("ratio close to 4 (consistent with O(h^2))") {
        // Asymptotic regime: medium-to-fine ratio should be closer to
        // 4 than coarse-to-medium (which may have pre-asymptotic effects).
        const double ratio_fine = err_100 / err_200;
        INFO("ratio (100->200) = " << ratio_fine);
        REQUIRE(ratio_fine > 2.5);
        REQUIRE(ratio_fine < 6.0);
    }
}


TEST_CASE("Heston PDE: sanity vs QE Monte Carlo",
          "[heston_pde][cross_method_mc]") {
    // The third independent method: QE Monte Carlo from Block 4. PDE
    // and QE should agree within 3*QE_HW + PDE_error.
    //
    // This catches systematic biases that Fourier might not catch
    // (e.g., PDE BC error in a regime where Fourier is also subtly
    // misbehaving for some unknown reason).
    const double T = 0.5;
    const double K = 100.0;

    const double pde_price = heston_call_pde(REF_S0, K, T, REF_R, REF_PARAMS,
                                                /*N_X=*/100, /*N_v=*/50,
                                                /*N_tau=*/100);

    std::mt19937_64 rng(42);
    const auto qe = mc_european_call_heston_qe(
        REF_S0, K, REF_R, REF_PARAMS, T,
        /*n_steps=*/50, /*n_paths=*/200000, rng);

    const double err = std::abs(pde_price - qe.estimate);
    const double tol = 3.0 * qe.half_width + 0.05;
    INFO("PDE = " << pde_price
         << ", QE = " << qe.estimate << " +/- " << qe.half_width
         << ", err = " << err << ", tol = " << tol);
    REQUIRE(err < tol);
}


TEST_CASE("Heston PDE: validation errors",
          "[heston_pde][validation]") {
    // Verify that bad inputs raise std::invalid_argument cleanly.
    SECTION("invalid HestonParams (kappa <= 0)") {
        HestonParams p_bad = REF_PARAMS;
        p_bad.kappa = -1.0;
        REQUIRE_THROWS_AS(
            heston_call_pde(REF_S0, 100.0, 0.5, REF_R, p_bad, 50, 25, 50),
            std::invalid_argument);
    }
    SECTION("non-positive S0") {
        REQUIRE_THROWS_AS(
            heston_call_pde(-100.0, 100.0, 0.5, REF_R, REF_PARAMS, 50, 25, 50),
            std::invalid_argument);
    }
    SECTION("non-positive K") {
        REQUIRE_THROWS_AS(
            heston_call_pde(REF_S0, -100.0, 0.5, REF_R, REF_PARAMS, 50, 25, 50),
            std::invalid_argument);
    }
    SECTION("non-positive T") {
        REQUIRE_THROWS_AS(
            heston_call_pde(REF_S0, 100.0, -0.1, REF_R, REF_PARAMS, 50, 25, 50),
            std::invalid_argument);
    }
    SECTION("N_X too small") {
        REQUIRE_THROWS_AS(
            heston_call_pde(REF_S0, 100.0, 0.5, REF_R, REF_PARAMS, 2, 25, 50),
            std::invalid_argument);
    }
    SECTION("N_v too small") {
        REQUIRE_THROWS_AS(
            heston_call_pde(REF_S0, 100.0, 0.5, REF_R, REF_PARAMS, 50, 2, 50),
            std::invalid_argument);
    }
    SECTION("theta_imp out of [0, 1]") {
        REQUIRE_THROWS_AS(
            heston_call_pde(REF_S0, 100.0, 0.5, REF_R, REF_PARAMS,
                              50, 25, 50, /*theta_imp=*/-0.1),
            std::invalid_argument);
        REQUIRE_THROWS_AS(
            heston_call_pde(REF_S0, 100.0, 0.5, REF_R, REF_PARAMS,
                              50, 25, 50, /*theta_imp=*/1.1),
            std::invalid_argument);
    }
    SECTION("v_max_factor <= 1") {
        REQUIRE_THROWS_AS(
            heston_call_pde(REF_S0, 100.0, 0.5, REF_R, REF_PARAMS,
                              50, 25, 50, /*theta_imp=*/0.5,
                              /*X_factor=*/4.0, /*v_max_factor=*/0.9),
            std::invalid_argument);
    }
}


TEST_CASE("Heston PDE: returns finite, positive prices",
          "[heston_pde][sanity]") {
    // Sanity check: across a range of strikes from deep ITM to deep
    // OTM, the PDE should return finite, non-negative prices.
    const double T = 0.5;
    const std::size_t N_X = 50, N_v = 25, N_tau = 50;

    for (double K : {50.0, 80.0, 100.0, 120.0, 150.0}) {
        const double price = heston_call_pde(REF_S0, K, T, REF_R, REF_PARAMS,
                                                N_X, N_v, N_tau);
        INFO("K=" << K << ", PDE=" << price);
        REQUIRE(std::isfinite(price));
        // Non-negativity up to discretisation error: when the true
        // price is small (deep OTM), the PDE error can produce
        // slightly negative values. We allow a small absolute slack.
        REQUIRE(price > -0.01);
        // Lower bound: call price >= max(S0 - K*exp(-rT), 0)
        const double intrinsic_disc = std::max(
            REF_S0 - K * std::exp(-REF_R * T), 0.0);
        REQUIRE(price >= intrinsic_disc - 0.5);  // generous slack
        // Upper bound: call price <= S0
        REQUIRE(price <= REF_S0 + 0.1);  // slack for boundary error
    }
}
