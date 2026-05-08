// test_cn_american.cpp
//
// Catch2 tests for the CN-American put pricer of Phase 3 Block 4.
//
// Cross-validation is performed against a self-contained CRR binomial
// reference embedded locally in this file. Keeps the test independent
// of the Phase 2 american module.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "cn_american.hpp"
#include "cn.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

using Catch::Approx;
using namespace quant::pde;

namespace {

/// Self-contained CRR American put. Independent reference; the price
/// converges to the true American put value at order 1/n_steps, so
/// n_steps = 2000 gives roughly 4 correct decimals.
double crr_american_put(double S, double K, double r,
                        double sigma, double T, int n_steps) {
    const double dt = T / static_cast<double>(n_steps);
    const double u = std::exp(sigma * std::sqrt(dt));
    const double d = 1.0 / u;
    const double p = (std::exp(r * dt) - d) / (u - d);
    const double disc = std::exp(-r * dt);

    std::vector<double> V(static_cast<std::size_t>(n_steps) + 1);
    for (int i = 0; i <= n_steps; ++i) {
        const double S_T = S * std::pow(u, n_steps - i) * std::pow(d, i);
        V[static_cast<std::size_t>(i)] = std::max(K - S_T, 0.0);
    }
    for (int step = n_steps - 1; step >= 0; --step) {
        for (int i = 0; i <= step; ++i) {
            const double S_node = S * std::pow(u, step - i) * std::pow(d, i);
            const double cont = disc * (
                p * V[static_cast<std::size_t>(i)]
              + (1.0 - p) * V[static_cast<std::size_t>(i) + 1]);
            const double exer = std::max(K - S_node, 0.0);
            V[static_cast<std::size_t>(i)] = std::max(cont, exer);
        }
    }
    return V[0];
}

}  // anonymous namespace


TEST_CASE("CN-American: cross-validation against CRR binomial",
          "[cn_american][cross-validation]") {
    const int N = 200, M = 100;
    const int n_crr = 2000;

    struct Case {
        double S, K, r, sigma, T;
    };
    const Case cases[] = {
        {100.0, 100.0, 0.05, 0.20, 1.00},
        { 90.0, 100.0, 0.05, 0.20, 1.00},
        {110.0, 100.0, 0.05, 0.20, 1.00},
        {100.0, 100.0, 0.05, 0.30, 1.00},
        {100.0, 100.0, 0.05, 0.20, 0.25},
        { 42.0,  40.0, 0.10, 0.20, 0.50},
    };
    for (const auto& c : cases) {
        const double p_psor = cn_american_put(
            c.S, c.K, c.r, c.sigma, c.T, N, M);
        const double p_crr = crr_american_put(
            c.S, c.K, c.r, c.sigma, c.T, n_crr);
        REQUIRE(p_psor == Approx(p_crr).margin(5e-3));
    }
}

TEST_CASE("CN-American: American put dominates European put",
          "[cn_american][dominance]") {
    const double K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const int N = 200, M = 100;
    for (double S : {80.0, 90.0, 100.0, 110.0, 120.0}) {
        const double p_amer = cn_american_put(S, K, r, sigma, T, N, M);
        const double p_eur  = cn_european_put (S, K, r, sigma, T, N, M);
        REQUIRE(p_amer >= p_eur - 1e-6);
    }
    const double pa_80 = cn_american_put(80.0, K, r, sigma, T, N, M);
    const double pe_80 = cn_european_put (80.0, K, r, sigma, T, N, M);
    REQUIRE(pa_80 - pe_80 > 0.1);
}

TEST_CASE("CN-American: spatial refinement reduces the error",
          "[cn_american][convergence]") {
    const double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double p_ref = crr_american_put(S, K, r, sigma, T, 5000);

    const double p_100 = cn_american_put(S, K, r, sigma, T, 100,  50);
    const double p_200 = cn_american_put(S, K, r, sigma, T, 200, 100);
    const double p_400 = cn_american_put(S, K, r, sigma, T, 400, 200);

    REQUIRE(std::abs(p_200 - p_ref) < std::abs(p_100 - p_ref));
    REQUIRE(std::abs(p_400 - p_ref) < 5e-3);
}

TEST_CASE("CN-American: invalid inputs raise",
          "[cn_american][validation]") {
    const double K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    REQUIRE_THROWS_AS(
        cn_american_put(0.0, K, r, sigma, T, 200, 100),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cn_american_put(1000.0, K, r, sigma, T, 200, 100),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cn_american_put(100.0, K, r, sigma, T, 100, 50, 4.0, 0.0),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cn_american_put(100.0, K, r, sigma, T, 100, 50, 4.0, 2.0),
        std::invalid_argument);
}
