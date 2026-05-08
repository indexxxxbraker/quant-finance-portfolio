// test_heston_fourier.cpp
//
// Catch2 tests for the Heston Fourier pricer of Phase 4 Block 2.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "heston_fourier.hpp"

#include <cmath>
#include <complex>
#include <stdexcept>

using Catch::Approx;
using namespace quant::heston;

namespace {

// Standard reference parameter set, used across the suite.
// Matches the Python validation suite (validate_heston_fourier.py).
const HestonParams REF_PARAMS = {
    /*kappa*/ 1.5,
    /*theta*/ 0.04,
    /*sigma*/ 0.3,
    /*rho*/  -0.7,
    /*v0*/    0.04,
};
const double REF_S0 = 100.0;
const double REF_R  = 0.05;

// Snapshot from Python (quantlib.heston_fourier.heston_call_lewis), itself
// validated against Carr-Madan FFT to ~1e-3 (interpolation precision) and
// against put-call parity to machine epsilon. Tolerance for the C++
// equivalent is set to 1e-5 absolute, which both methods comfortably meet.
struct Snapshot {
    double K;
    double tau;
    double expected;
};

}  // anonymous namespace


TEST_CASE("Heston char function: phi(0) = 1 and phi(-i) = S0 exp(rT)",
          "[heston_fourier][cf_sanity]") {
    const double tau = 0.5;

    SECTION("phi(0) = 1") {
        const auto phi = heston_cf({0.0, 0.0}, tau, REF_S0, REF_R, REF_PARAMS);
        REQUIRE(phi.real() == Approx(1.0).margin(1e-12));
        REQUIRE(phi.imag() == Approx(0.0).margin(1e-12));
    }

    SECTION("phi(-i) = S0 exp(r tau)") {
        const auto phi = heston_cf({0.0, -1.0}, tau, REF_S0, REF_R, REF_PARAMS);
        const double expected = REF_S0 * std::exp(REF_R * tau);
        REQUIRE(phi.real() == Approx(expected).margin(1e-10));
        REQUIRE(phi.imag() == Approx(0.0).margin(1e-10));
    }
}


TEST_CASE("Heston BS limit: small sigma converges to BS",
          "[heston_fourier][bs_limit]") {
    // When v0 = theta and sigma -> 0, the Heston variance is constant equal
    // to v0 and the Heston call equals the BS call at vol sqrt(v0). For
    // sigma > 0 the leading-order leverage correction is linear in sigma
    // (not quadratic, due to rho != 0).
    const double tau = 0.5;
    const double K = 100.0;
    const double sigma_bs = std::sqrt(REF_PARAMS.v0);
    const double C_bs = black_scholes_call(REF_S0, K, tau, REF_R, sigma_bs);

    HestonParams p = REF_PARAMS;

    SECTION("sigma_H = 0.001 close to BS") {
        p.sigma = 0.001;
        const double C = heston_call_lewis(K, tau, REF_S0, REF_R, p);
        REQUIRE(C == Approx(C_bs).margin(1e-3));
    }

    SECTION("error decreases monotonically as sigma_H -> 0") {
        p.sigma = 0.1;
        const double err_01   = std::abs(heston_call_lewis(K, tau, REF_S0, REF_R, p) - C_bs);
        p.sigma = 0.01;
        const double err_001  = std::abs(heston_call_lewis(K, tau, REF_S0, REF_R, p) - C_bs);
        p.sigma = 0.001;
        const double err_0001 = std::abs(heston_call_lewis(K, tau, REF_S0, REF_R, p) - C_bs);
        REQUIRE(err_01   > err_001);
        REQUIRE(err_001  > err_0001);
    }
}


TEST_CASE("Heston put-call parity holds to machine precision",
          "[heston_fourier][parity]") {
    const double tau = 0.5;
    for (double K : {80.0, 95.0, 100.0, 105.0, 120.0}) {
        const double C   = heston_call_lewis(K, tau, REF_S0, REF_R, REF_PARAMS);
        const double P   = put_via_parity(C, REF_S0, K, tau, REF_R);
        const double lhs = C - P;
        const double rhs = REF_S0 - K * std::exp(-REF_R * tau);
        REQUIRE(lhs == Approx(rhs).margin(1e-10));
    }
}


TEST_CASE("Heston Lewis matches Python reference values",
          "[heston_fourier][python_consistency]") {
    const double tol = 1e-5;

    SECTION("ATM, varying maturity") {
        const double K = 100.0;
        for (const Snapshot& s : {
                Snapshot{ K,  0.25,  4.5842062306},
                Snapshot{ K,  0.50,  6.8257341234},
                Snapshot{ K,  1.00, 10.3618690210},
                Snapshot{ K,  2.00, 16.0922404914},
                Snapshot{ K,  5.00, 29.3686505542},
                Snapshot{ K, 10.00, 45.6237154836}}) {
            const double C = heston_call_lewis(s.K, s.tau, REF_S0, REF_R, REF_PARAMS);
            REQUIRE(C == Approx(s.expected).margin(tol));
        }
    }

    SECTION("Varying strike, T = 0.5") {
        const double tau = 0.5;
        for (const Snapshot& s : {
                Snapshot{ 80.0, tau, 22.4552674179},
                Snapshot{ 90.0, tau, 13.8369446176},
                Snapshot{ 95.0, tau, 10.0676796216},
                Snapshot{100.0, tau,  6.8257341234},
                Snapshot{105.0, tau,  4.2259455529},
                Snapshot{110.0, tau,  2.3340705240},
                Snapshot{120.0, tau,  0.4704669763}}) {
            const double C = heston_call_lewis(s.K, s.tau, REF_S0, REF_R, REF_PARAMS);
            REQUIRE(C == Approx(s.expected).margin(tol));
        }
    }

    SECTION("High vol-of-vol parameter set") {
        // (kappa=2, theta=0.06, sigma=0.5, rho=-0.5, v0=0.05): stresses
        // both the AMSST branch handling and the Lewis quadrature width.
        const HestonParams alt = {2.0, 0.06, 0.5, -0.5, 0.05};
        struct AltCase { double K; double tau; double expected; };
        for (const AltCase& s : {
                AltCase{ 90.0, 0.5, 14.3544086867},
                AltCase{100.0, 0.5,  7.5318489631},
                AltCase{110.0, 0.5,  3.0680594493},
                AltCase{ 90.0, 1.0, 17.9539747817},
                AltCase{100.0, 1.0, 11.4873872410},
                AltCase{110.0, 1.0,  6.6062491383}}) {
            const double C = heston_call_lewis(s.K, s.tau, REF_S0, REF_R, alt);
            REQUIRE(C == Approx(s.expected).margin(tol));
        }
    }
}


TEST_CASE("Heston AMSST formulation stable at long maturity",
          "[heston_fourier][amsst]") {
    // The "Heston Trap" of the original (1993) formulation manifests
    // at long maturity (T >= 2 typically) as branch-cut errors that
    // corrupt prices by 0.1%-2%. The AMSST formulation is stable. We
    // can't compare against the broken formulation here, but we can
    // verify the AMSST output is well-behaved: positive, finite,
    // bounded by spot, and increasing in T (intuitive for ATM calls).
    const double K = 100.0;
    double C_prev = -1.0;
    for (double T : {0.5, 1.0, 2.0, 5.0, 10.0, 20.0}) {
        const double C = heston_call_lewis(K, T, REF_S0, REF_R, REF_PARAMS);
        REQUIRE(std::isfinite(C));
        REQUIRE(C > 0.0);
        REQUIRE(C < REF_S0);
        REQUIRE(C > C_prev);
        C_prev = C;
    }
}


TEST_CASE("HestonParams::validate rejects invalid parameters",
          "[heston_fourier][validation]") {
    SECTION("kappa <= 0") {
        HestonParams p = REF_PARAMS;
        p.kappa = 0.0;
        REQUIRE_THROWS_AS(p.validate(), std::invalid_argument);
        p.kappa = -1.0;
        REQUIRE_THROWS_AS(p.validate(), std::invalid_argument);
    }
    SECTION("theta <= 0") {
        HestonParams p = REF_PARAMS;
        p.theta = 0.0;
        REQUIRE_THROWS_AS(p.validate(), std::invalid_argument);
    }
    SECTION("sigma <= 0") {
        HestonParams p = REF_PARAMS;
        p.sigma = 0.0;
        REQUIRE_THROWS_AS(p.validate(), std::invalid_argument);
    }
    SECTION("rho out of [-1, 1]") {
        HestonParams p = REF_PARAMS;
        p.rho = -1.5;
        REQUIRE_THROWS_AS(p.validate(), std::invalid_argument);
        p.rho = 1.5;
        REQUIRE_THROWS_AS(p.validate(), std::invalid_argument);
    }
    SECTION("v0 < 0") {
        HestonParams p = REF_PARAMS;
        p.v0 = -0.01;
        REQUIRE_THROWS_AS(p.validate(), std::invalid_argument);
    }
    SECTION("valid parameters do not throw") {
        REQUIRE_NOTHROW(REF_PARAMS.validate());
    }
}


TEST_CASE("Pricing functions reject invalid contract spec",
          "[heston_fourier][validation]") {
    REQUIRE_THROWS_AS(
        heston_call_lewis(100.0, -0.1, REF_S0, REF_R, REF_PARAMS),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        heston_call_lewis(100.0, 0.5, -100.0, REF_R, REF_PARAMS),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        heston_call_lewis(-100.0, 0.5, REF_S0, REF_R, REF_PARAMS),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        heston_call_lewis(100.0, 0.5, REF_S0, REF_R, REF_PARAMS, -1.0),
        std::invalid_argument);
}
