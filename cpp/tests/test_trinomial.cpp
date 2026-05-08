// test_trinomial.cpp - Catch2 tests for the Kamrad-Ritchken trinomial.
//
// Mirrors validate_trinomial.py: cross-validation against BS (European)
// and an embedded CRR binomial reference (American), first-order
// convergence in n_steps, lambda sweep, and input validation.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>

#include "trinomial.hpp"

using Catch::Approx;
using quant::pde::trinomial_european_call;
using quant::pde::trinomial_european_put;
using quant::pde::trinomial_american_put;

// --------------------------------------------------------------------------
// Local references for testing. Self-contained so that the test does not
// depend on the project's binomial.hpp / black_scholes.hpp interfaces.
// --------------------------------------------------------------------------
namespace {

double normal_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double bs_call(double S, double K, double r, double sigma, double T) {
    const double sqrtT = std::sqrt(T);
    const double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) /
                      (sigma * sqrtT);
    const double d2 = d1 - sigma * sqrtT;
    return S * normal_cdf(d1) - K * std::exp(-r * T) * normal_cdf(d2);
}

double bs_put(double S, double K, double r, double sigma, double T) {
    const double sqrtT = std::sqrt(T);
    const double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) /
                      (sigma * sqrtT);
    const double d2 = d1 - sigma * sqrtT;
    return K * std::exp(-r * T) * normal_cdf(-d2) - S * normal_cdf(-d1);
}

// CRR binomial American put. Self-contained for the test, mirroring
// the pattern in test_cn_american.cpp from Block 4.
double crr_american_put(double S, double K, double r, double sigma,
                         double T, int n_steps) {
    const double dt = T / static_cast<double>(n_steps);
    const double u = std::exp(sigma * std::sqrt(dt));
    const double d = 1.0 / u;
    const double p = (std::exp(r * dt) - d) / (u - d);
    const double disc = std::exp(-r * dt);

    std::vector<double> V(n_steps + 1);
    for (int i = 0; i <= n_steps; ++i) {
        const double S_T =
            S * std::pow(u, n_steps - i) * std::pow(d, i);
        V[i] = std::max(K - S_T, 0.0);
    }
    for (int step = n_steps - 1; step >= 0; --step) {
        for (int i = 0; i <= step; ++i) {
            const double S_node =
                S * std::pow(u, step - i) * std::pow(d, i);
            const double cont = disc * (p * V[i] + (1.0 - p) * V[i + 1]);
            const double exer = std::max(K - S_node, 0.0);
            V[i] = std::max(cont, exer);
        }
    }
    return V[0];
}

}  // namespace

// --------------------------------------------------------------------------
TEST_CASE("trinomial European cross-validation against Black-Scholes",
          "[trinomial]") {
    struct Case {
        double S, K, r, sigma, T;
    };
    std::vector<Case> cases = {
        {100.0, 100.0, 0.05, 0.20, 1.00},
        { 90.0, 100.0, 0.05, 0.20, 1.00},
        {110.0, 100.0, 0.05, 0.20, 1.00},
        {100.0, 100.0, 0.05, 0.30, 1.00},
        {100.0, 100.0, 0.05, 0.20, 0.25},
        { 42.0,  40.0, 0.10, 0.20, 0.50},
    };
    const int n_steps = 2000;
    const double tol = 5e-3;

    for (const auto& tc : cases) {
        const double c_tri = trinomial_european_call(
            tc.S, tc.K, tc.r, tc.sigma, tc.T, n_steps);
        const double p_tri = trinomial_european_put(
            tc.S, tc.K, tc.r, tc.sigma, tc.T, n_steps);
        const double c_bs = bs_call(tc.S, tc.K, tc.r, tc.sigma, tc.T);
        const double p_bs = bs_put (tc.S, tc.K, tc.r, tc.sigma, tc.T);

        REQUIRE(std::abs(c_tri - c_bs) < tol);
        REQUIRE(std::abs(p_tri - p_bs) < tol);
    }
}

TEST_CASE("trinomial American cross-validation against CRR binomial",
          "[trinomial]") {
    struct Case {
        double S, K, r, sigma, T;
    };
    std::vector<Case> cases = {
        {100.0, 100.0, 0.05, 0.20, 1.00},
        { 90.0, 100.0, 0.05, 0.20, 1.00},
        {110.0, 100.0, 0.05, 0.20, 1.00},
        {100.0, 100.0, 0.05, 0.30, 1.00},
        { 42.0,  40.0, 0.10, 0.20, 0.50},
    };
    const int n_steps = 2000;
    const double tol = 5e-3;

    for (const auto& tc : cases) {
        const double p_tri = trinomial_american_put(
            tc.S, tc.K, tc.r, tc.sigma, tc.T, n_steps);
        const double p_crr = crr_american_put(
            tc.S, tc.K, tc.r, tc.sigma, tc.T, n_steps);
        REQUIRE(std::abs(p_tri - p_crr) < tol);
    }
}

TEST_CASE("trinomial first-order convergence in n_steps", "[trinomial]") {
    const double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double c_bs = bs_call(S, K, r, sigma, T);

    std::vector<int> n_list = {200, 400, 800, 1600};
    std::vector<double> errors;
    for (int n : n_list) {
        const double c = trinomial_european_call(S, K, r, sigma, T, n);
        errors.push_back(std::abs(c - c_bs));
    }

    // Each ratio should be near 2 (first-order). Wide band [1.2, 4.0] for
    // robustness against the well-known at-the-money lattice oscillation.
    for (std::size_t i = 0; i + 1 < errors.size(); ++i) {
        const double ratio = errors[i] / errors[i + 1];
        REQUIRE(ratio >= 1.2);
        REQUIRE(ratio <= 4.0);
    }
}

TEST_CASE("trinomial lambda sweep", "[trinomial]") {
    const double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double c_bs = bs_call(S, K, r, sigma, T);
    const int n_steps = 1000;

    std::vector<double> lambdas = {1.0, 1.5, 2.0, 3.0, 5.0};
    std::vector<double> prices;
    for (double lam : lambdas) {
        prices.push_back(trinomial_european_call(
            S, K, r, sigma, T, n_steps, lam));
    }

    // All prices within 5e-3 of BS.
    for (double p : prices) {
        REQUIRE(std::abs(p - c_bs) < 5e-3);
    }
    // Spread across lambdas < 1e-2.
    const double pmax = *std::max_element(prices.begin(), prices.end());
    const double pmin = *std::min_element(prices.begin(), prices.end());
    REQUIRE(pmax - pmin < 1e-2);
}

TEST_CASE("trinomial input validation", "[trinomial]") {
    const double K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    const double S = 100.0;
    const int n = 100;

    REQUIRE_THROWS_AS(trinomial_european_call(0.0, K, r, sigma, T, n),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(trinomial_european_call(S, -10.0, r, sigma, T, n),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(trinomial_european_call(S, K, r, 0.0, T, n),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(trinomial_european_call(S, K, r, sigma, T, 0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(trinomial_european_call(S, K, r, sigma, T, n, 0.5),
                      std::invalid_argument);

    // lambda = 1 should NOT throw.
    REQUIRE_NOTHROW(trinomial_european_call(S, K, r, sigma, T, n, 1.0));
}
