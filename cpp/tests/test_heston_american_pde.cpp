// test_heston_american_pde.cpp
//
// Catch2 tests for the Heston American put PDE pricer of Phase 4 Block 6.
//
// Without a Fourier ground truth for American options, the tests focus
// on structural bounds and limit cases:
//   - American put >= European put (positive EEP)
//   - Deep ITM put = intrinsic K - S0 (immediate exercise optimal)
//   - Deep OTM put close to European put (no exercise advantage)
//   - Spatial convergence: error decreases with grid refinement
//   - Validation of inputs

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "heston_american_pde.hpp"
#include "heston_fourier.hpp"

#include <cmath>
#include <stdexcept>

using Catch::Approx;
using namespace quant::heston;

namespace {

const HestonParams REF_PARAMS = {
    /*kappa*/ 1.5, /*theta*/ 0.04, /*sigma*/ 0.3,
    /*rho*/  -0.7, /*v0*/    0.04,
};
const double REF_S0 = 100.0;
const double REF_R  = 0.05;
const double REF_T  = 0.5;

// European put via put-call parity from Fourier call.
double european_put(double K, double T) {
    const double C = heston_call_lewis(K, T, REF_S0, REF_R, REF_PARAMS);
    return C - REF_S0 + K * std::exp(-REF_R * T);
}

}  // anonymous namespace


TEST_CASE("American put: at least European put (positive EEP)",
          "[heston_american][bound]") {
    // Sanity bound across moneyness range. EEP must be non-negative
    // up to floating-point precision.
    for (double K : {80.0, 100.0, 120.0}) {
        const double am = heston_american_put_pde(
            REF_S0, K, REF_T, REF_R, REF_PARAMS,
            /*N_X=*/100, /*N_v=*/50, /*N_tau=*/100);
        const double eur = european_put(K, REF_T);
        const double eep = am - eur;
        INFO("K=" << K << ", American=" << am
             << ", European=" << eur << ", EEP=" << eep);
        REQUIRE(eep > -1e-6);
    }
}


TEST_CASE("American put: deep ITM equals intrinsic value",
          "[heston_american][limit]") {
    // For deep ITM (K=120, S0=100), immediate exercise is optimal:
    // the option is worth exactly its intrinsic value K - S0 = 20.
    const double K = 120.0;
    const double am = heston_american_put_pde(
        REF_S0, K, REF_T, REF_R, REF_PARAMS,
        /*N_X=*/200, /*N_v=*/100, /*N_tau=*/200);
    const double intrinsic = K - REF_S0;
    INFO("K=" << K << ", American=" << am
         << ", intrinsic=" << intrinsic);
    // The PDE should match intrinsic to ~0.01 at this grid
    REQUIRE(std::abs(am - intrinsic) < 0.01);
}


TEST_CASE("American put: deep OTM close to European",
          "[heston_american][limit]") {
    // For deep OTM (K=70, S0=100), early exercise is suboptimal:
    // EEP should be small in absolute terms.
    const double K = 70.0;
    const double am = heston_american_put_pde(
        REF_S0, K, REF_T, REF_R, REF_PARAMS,
        /*N_X=*/200, /*N_v=*/100, /*N_tau=*/200);
    const double eur = european_put(K, REF_T);
    const double eep = am - eur;
    const double rel_eep = eep / eur;
    INFO("K=" << K << ", American=" << am
         << ", European=" << eur
         << ", EEP=" << eep << " (" << 100*rel_eep << "%)");
    // Deep OTM EEP should be < 5% of European value
    REQUIRE(eep >= -1e-6);
    REQUIRE(rel_eep < 0.05);
}


TEST_CASE("American put: spatial convergence with halving",
          "[heston_american][convergence]") {
    // The same halving pattern as Block 5: doubling N_X and N_v
    // should reduce error in successive estimates by a factor close
    // to 4. We do not have a Fourier truth for the American put, so
    // we use the finest grid as a pseudo-truth and check that
    // coarser grids approach it.
    const double K = 100.0;
    const std::size_t N_tau_fixed = 200;

    const double price_50  = heston_american_put_pde(
        REF_S0, K, REF_T, REF_R, REF_PARAMS, 50, 25, N_tau_fixed);
    const double price_100 = heston_american_put_pde(
        REF_S0, K, REF_T, REF_R, REF_PARAMS, 100, 50, N_tau_fixed);
    const double price_200 = heston_american_put_pde(
        REF_S0, K, REF_T, REF_R, REF_PARAMS, 200, 100, N_tau_fixed);

    // Treat price_200 as the closest approximation to truth
    const double err_50  = std::abs(price_50  - price_200);
    const double err_100 = std::abs(price_100 - price_200);

    INFO("price_50=" << price_50
         << ", price_100=" << price_100
         << ", price_200=" << price_200
         << ", err_50=" << err_50
         << ", err_100=" << err_100);

    SECTION("error decreases monotonically with grid refinement") {
        REQUIRE(err_100 < err_50);
    }
    SECTION("ratio close to 4 (consistent with O(h^2) on smooth region)") {
        // The American projection introduces non-smoothness near the
        // exercise boundary, so the ratio may be < 4. Allow a wider
        // window than Block 5.
        const double ratio = err_50 / err_100;
        INFO("err_50 / err_100 = " << ratio);
        REQUIRE(ratio > 2.0);
    }
}


TEST_CASE("American put: returns finite, positive prices",
          "[heston_american][sanity]") {
    // Basic sanity across a range of strikes.
    for (double K : {50.0, 80.0, 100.0, 120.0, 150.0}) {
        const double am = heston_american_put_pde(
            REF_S0, K, REF_T, REF_R, REF_PARAMS, 50, 25, 50);
        INFO("K=" << K << ", American put = " << am);
        REQUIRE(std::isfinite(am));
        // American put price >= 0 (with small slack for FP error in
        // deep OTM regime where price is close to zero)
        REQUIRE(am > -0.01);
        // American put price <= K (cannot exceed strike payoff in
        // worst case)
        REQUIRE(am <= K + 0.5);
    }
}


TEST_CASE("American put: validation errors",
          "[heston_american][validation]") {
    SECTION("invalid HestonParams") {
        HestonParams p_bad = REF_PARAMS;
        p_bad.kappa = -1.0;
        REQUIRE_THROWS_AS(
            heston_american_put_pde(REF_S0, 100.0, REF_T, REF_R, p_bad,
                                       50, 25, 50),
            std::invalid_argument);
    }
    SECTION("non-positive S0") {
        REQUIRE_THROWS_AS(
            heston_american_put_pde(-100.0, 100.0, REF_T, REF_R,
                                       REF_PARAMS, 50, 25, 50),
            std::invalid_argument);
    }
    SECTION("non-positive K") {
        REQUIRE_THROWS_AS(
            heston_american_put_pde(REF_S0, -100.0, REF_T, REF_R,
                                       REF_PARAMS, 50, 25, 50),
            std::invalid_argument);
    }
    SECTION("non-positive T") {
        REQUIRE_THROWS_AS(
            heston_american_put_pde(REF_S0, 100.0, -0.1, REF_R,
                                       REF_PARAMS, 50, 25, 50),
            std::invalid_argument);
    }
    SECTION("grid size too small") {
        REQUIRE_THROWS_AS(
            heston_american_put_pde(REF_S0, 100.0, REF_T, REF_R,
                                       REF_PARAMS, 2, 25, 50),
            std::invalid_argument);
    }
    SECTION("theta_imp out of [0, 1]") {
        REQUIRE_THROWS_AS(
            heston_american_put_pde(REF_S0, 100.0, REF_T, REF_R,
                                       REF_PARAMS, 50, 25, 50,
                                       /*theta_imp=*/-0.1),
            std::invalid_argument);
    }
}
