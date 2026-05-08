// test_psor.cpp
//
// Catch2 tests for the PSOR LCP solver of Phase 3 Block 4.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "psor.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

using Catch::Approx;
using namespace quant::pde;

namespace {

// Reference: dense Gaussian elimination without pivoting.
std::vector<double> dense_solve(std::vector<std::vector<double>> A,
                                std::vector<double> b) {
    const std::size_t n = A.size();
    for (std::size_t k = 0; k < n; ++k) {
        const double pivot = A[k][k];
        for (std::size_t i = k + 1; i < n; ++i) {
            const double f = A[i][k] / pivot;
            for (std::size_t j = k; j < n; ++j) A[i][j] -= f * A[k][j];
            b[i] -= f * b[k];
        }
    }
    std::vector<double> x(n);
    for (std::size_t i = n; i-- > 0;) {
        double s = b[i];
        for (std::size_t j = i + 1; j < n; ++j) s -= A[i][j] * x[j];
        x[i] = s / A[i][i];
    }
    return x;
}

std::vector<std::vector<double>> assemble_dense(
    const std::vector<double>& sub,
    const std::vector<double>& diag,
    const std::vector<double>& sup) {
    const std::size_t n = diag.size();
    std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) A[i][i] = diag[i];
    for (std::size_t i = 0; i < n - 1; ++i) {
        A[i + 1][i] = sub[i];
        A[i][i + 1] = sup[i];
    }
    return A;
}

}  // anonymous namespace


TEST_CASE("PSOR: unconstrained matches linear solve",
          "[psor][unconstrained]") {
    std::mt19937_64 rng(42);
    std::normal_distribution<double> N01(0.0, 1.0);

    for (int n : {3, 10, 50, 100}) {
        std::vector<double> sub(n - 1), sup(n - 1), diag(n), rhs(n);
        for (int i = 0; i < n - 1; ++i) {
            sub[i] = N01(rng);
            sup[i] = N01(rng);
        }
        for (int i = 0; i < n; ++i) {
            const double off = (i > 0     ? std::abs(sub[i - 1]) : 0.0)
                             + (i < n - 1 ? std::abs(sup[i])     : 0.0);
            diag[i] = 1.0 + 2.0 * off;     // strictly diagonally dominant
            rhs[i]  = N01(rng);
        }
        const std::vector<double> obstacle(n, -1e10);
        const PSORResult res = psor_solve(
            sub, diag, sup, rhs, obstacle, 1.2, 1e-12, 1e-12, 10000);
        const std::vector<double> x_ref = dense_solve(
            assemble_dense(sub, diag, sup), rhs);
        for (int i = 0; i < n; ++i) {
            REQUIRE(res.x[i] == Approx(x_ref[i]).margin(1e-6));
        }
    }
}

TEST_CASE("PSOR: constrained problem respects obstacle",
          "[psor][constrained]") {
    const std::vector<double> sub  = {-1, -1, -1};
    const std::vector<double> diag = { 4,  4,  4,  4};
    const std::vector<double> sup  = {-1, -1, -1};
    const std::vector<double> rhs  = { 1,  2,  3,  4};
    const std::vector<double> obstacle = {1.0, 1.0, 1.0, 1.0};

    const PSORResult res = psor_solve(sub, diag, sup, rhs, obstacle);
    for (std::size_t i = 0; i < 4; ++i) {
        REQUIRE(res.x[i] >= 1.0 - 1e-10);
    }
    // The first component should hit the obstacle (active set).
    REQUIRE(res.x[0] == Approx(1.0).margin(1e-8));
}

TEST_CASE("PSOR: omega sweep has interior minimum",
          "[psor][omega]") {
    const int n = 100;
    const std::vector<double> sub (n - 1, -1.0);
    const std::vector<double> diag(n,      2.5);
    const std::vector<double> sup (n - 1, -1.0);
    std::vector<double> rhs(n, 1.0);
    const std::vector<double> obstacle(n, -10.0);

    int iter_05 = 0, iter_14 = 0, iter_18 = 0;
    {
        const PSORResult r = psor_solve(sub, diag, sup, rhs, obstacle,
                                         0.5, 1e-9, 1e-9, 20000);
        iter_05 = r.n_iter;
    }
    {
        const PSORResult r = psor_solve(sub, diag, sup, rhs, obstacle,
                                         1.4, 1e-9, 1e-9, 20000);
        iter_14 = r.n_iter;
    }
    {
        const PSORResult r = psor_solve(sub, diag, sup, rhs, obstacle,
                                         1.8, 1e-9, 1e-9, 20000);
        iter_18 = r.n_iter;
    }
    REQUIRE(iter_14 < iter_05);
    REQUIRE(iter_14 < iter_18);
}

TEST_CASE("PSOR: invalid omega raises", "[psor][validation]") {
    const std::vector<double> sub  = {-1.0};
    const std::vector<double> diag = { 2.0,  2.0};
    const std::vector<double> sup  = {-1.0};
    const std::vector<double> rhs  = { 1.0,  1.0};
    const std::vector<double> obs  = {-10.0, -10.0};

    REQUIRE_THROWS_AS(
        psor_solve(sub, diag, sup, rhs, obs, 0.0),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        psor_solve(sub, diag, sup, rhs, obs, 2.0),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        psor_solve(sub, diag, sup, rhs, obs, -0.1),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        psor_solve(sub, diag, sup, rhs, obs, 2.1),
        std::invalid_argument);
}

TEST_CASE("PSOR: shape mismatch raises", "[psor][validation]") {
    const std::vector<double> sub2  = {1.0, 2.0};
    const std::vector<double> diag3 = {3.0, 4.0, 5.0};
    const std::vector<double> sup2  = {1.0, 2.0};
    const std::vector<double> rhs3  = {1.0, 2.0, 3.0};
    const std::vector<double> obs3  = {-1.0, -1.0, -1.0};

    REQUIRE_NOTHROW(psor_solve(sub2, diag3, sup2, rhs3, obs3));
    REQUIRE_THROWS_AS(
        psor_solve({1.0}, diag3, sup2, rhs3, obs3),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        psor_solve(sub2, diag3, sup2, std::vector<double>{1.0, 2.0}, obs3),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        psor_solve(sub2, diag3, sup2, rhs3, std::vector<double>{0.0, 0.0}),
        std::invalid_argument);
}

TEST_CASE("PSOR: max_iter exceeded raises runtime_error",
          "[psor][validation]") {
    const int n = 50;
    const std::vector<double> sub (n - 1, -1.0);
    const std::vector<double> diag(n,      2.5);
    const std::vector<double> sup (n - 1, -1.0);
    const std::vector<double> rhs (n,      1.0);
    const std::vector<double> obs (n,   -100.0);

    REQUIRE_THROWS_AS(
        psor_solve(sub, diag, sup, rhs, obs, 1.2, 1e-15, 1e-15, 5),
        std::runtime_error);
}
