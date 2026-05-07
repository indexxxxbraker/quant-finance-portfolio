// test_thomas.cpp
//
// Catch2 tests for the Thomas tridiagonal solver of Phase 3 Block 2.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "thomas.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

using Catch::Approx;
using namespace quant::pde;


// ---------------------------------------------------------------------------
// Reference: dense Gaussian elimination (no pivoting). Used purely
// to cross-validate Thomas. O(n^3) but only invoked at small n.
// ---------------------------------------------------------------------------
namespace {

std::vector<double> dense_solve(std::vector<std::vector<double>> A,
                                std::vector<double> b) {
    const std::size_t n = A.size();
    // Forward elimination.
    for (std::size_t k = 0; k < n; ++k) {
        const double pivot = A[k][k];
        for (std::size_t i = k + 1; i < n; ++i) {
            const double f = A[i][k] / pivot;
            for (std::size_t j = k; j < n; ++j) {
                A[i][j] -= f * A[k][j];
            }
            b[i] -= f * b[k];
        }
    }
    // Backward substitution.
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


// ---------------------------------------------------------------------------
// Hand-checkable systems
// ---------------------------------------------------------------------------

TEST_CASE("Thomas: 4x4 hand-checkable", "[thomas]") {
    // Matrix:
    //   4 -1  0  0       1
    //  -1  4 -1  0   x = 2
    //   0 -1  4 -1       3
    //   0  0 -1  4       4
    const std::vector<double> sub  = {-1, -1, -1};
    const std::vector<double> diag = { 4,  4,  4,  4};
    const std::vector<double> sup  = {-1, -1, -1};
    const std::vector<double> rhs  = { 1,  2,  3,  4};

    const std::vector<double> x = thomas_solve(sub, diag, sup, rhs);

    // Cross-validate against dense solve.
    const std::vector<double> x_ref = dense_solve(
        assemble_dense(sub, diag, sup), rhs);

    REQUIRE(x.size() == 4);
    for (std::size_t i = 0; i < 4; ++i) {
        REQUIRE(x[i] == Approx(x_ref[i]).margin(1e-12));
    }
}

TEST_CASE("Thomas: 1x1 system", "[thomas][edge]") {
    const std::vector<double> sub  = {};
    const std::vector<double> diag = {3.0};
    const std::vector<double> sup  = {};
    const std::vector<double> rhs  = {6.0};

    const std::vector<double> x = thomas_solve(sub, diag, sup, rhs);
    REQUIRE(x.size() == 1);
    REQUIRE(x[0] == Approx(2.0).margin(1e-14));
}

TEST_CASE("Thomas: 2x2 system", "[thomas][edge]") {
    //  2 -1   x1   3
    // -1  2   x2 = 0
    // Solution: x1=2, x2=1.
    const std::vector<double> sub  = {-1.0};
    const std::vector<double> diag = { 2.0,  2.0};
    const std::vector<double> sup  = {-1.0};
    const std::vector<double> rhs  = { 3.0,  0.0};

    const std::vector<double> x = thomas_solve(sub, diag, sup, rhs);
    REQUIRE(x.size() == 2);
    REQUIRE(x[0] == Approx(2.0).margin(1e-14));
    REQUIRE(x[1] == Approx(1.0).margin(1e-14));
}


// ---------------------------------------------------------------------------
// Random diagonally-dominant systems
// ---------------------------------------------------------------------------

TEST_CASE("Thomas: random diagonally-dominant systems agree with dense LU",
          "[thomas][random]") {
    std::mt19937_64 rng(42);
    std::normal_distribution<double> N01(0.0, 1.0);

    for (int n : {3, 5, 10, 50, 100}) {
        std::vector<double> sub(n - 1), sup(n - 1), diag(n), rhs(n);
        for (int i = 0; i < n - 1; ++i) {
            sub[i] = N01(rng);
            sup[i] = N01(rng);
        }
        for (int i = 0; i < n; ++i) {
            const double off = (i > 0     ? std::abs(sub[i - 1]) : 0.0)
                             + (i < n - 1 ? std::abs(sup[i])     : 0.0);
            const int sign = (rng() & 1) ? +1 : -1;
            diag[i] = sign * (1.0 + 2.0 * off);
            rhs[i]  = N01(rng);
        }
        const std::vector<double> x_thomas =
            thomas_solve(sub, diag, sup, rhs);
        const std::vector<double> x_dense  = dense_solve(
            assemble_dense(sub, diag, sup), rhs);
        for (int i = 0; i < n; ++i) {
            REQUIRE(x_thomas[i] == Approx(x_dense[i]).margin(1e-10));
        }
    }
}


// ---------------------------------------------------------------------------
// Pre-factored interface
// ---------------------------------------------------------------------------

TEST_CASE("Thomas: factored solve agrees with one-shot solve",
          "[thomas][factor]") {
    std::mt19937_64 rng(7);
    std::normal_distribution<double> N01(0.0, 1.0);

    const int n = 50;
    std::vector<double> sub(n - 1), sup(n - 1), diag(n);
    for (int i = 0; i < n - 1; ++i) {
        sub[i] = N01(rng);
        sup[i] = N01(rng);
    }
    for (int i = 0; i < n; ++i) {
        const double off = (i > 0     ? std::abs(sub[i - 1]) : 0.0)
                         + (i < n - 1 ? std::abs(sup[i])     : 0.0);
        diag[i] = 1.0 + 2.0 * off;
    }
    const ThomasFactor f = thomas_factor(sub, diag, sup);

    // Try several distinct rhs values.
    for (int trial = 0; trial < 5; ++trial) {
        std::vector<double> rhs(n);
        for (int i = 0; i < n; ++i) rhs[i] = N01(rng);

        const std::vector<double> x_one_shot =
            thomas_solve(sub, diag, sup, rhs);
        const std::vector<double> x_factored = f.solve(rhs);

        for (int i = 0; i < n; ++i) {
            REQUIRE(x_factored[i] == Approx(x_one_shot[i]).margin(1e-14));
        }
    }
}


// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------

TEST_CASE("Thomas: invalid sizes raise std::invalid_argument",
          "[thomas][validation]") {
    const std::vector<double> sub2  = {1.0, 2.0};
    const std::vector<double> diag3 = {3.0, 4.0, 5.0};
    const std::vector<double> sup2  = {6.0, 7.0};
    const std::vector<double> rhs3  = {8.0, 9.0, 10.0};

    // Correct sizes succeed.
    REQUIRE_NOTHROW(thomas_solve(sub2, diag3, sup2, rhs3));

    // Wrong sub size.
    REQUIRE_THROWS_AS(
        thomas_solve(std::vector<double>{1.0}, diag3, sup2, rhs3),
        std::invalid_argument);

    // Wrong sup size.
    REQUIRE_THROWS_AS(
        thomas_solve(sub2, diag3, std::vector<double>{1.0}, rhs3),
        std::invalid_argument);

    // Wrong rhs size.
    REQUIRE_THROWS_AS(
        thomas_solve(sub2, diag3, sup2, std::vector<double>{1.0, 2.0}),
        std::invalid_argument);

    // Empty system.
    REQUIRE_THROWS_AS(
        thomas_solve({}, {}, {}, {}),
        std::invalid_argument);
}

TEST_CASE("Thomas: zero pivot raises std::runtime_error",
          "[thomas][validation]") {
    // diag[0] = 0 -> immediate failure.
    REQUIRE_THROWS_AS(
        thomas_solve({-1.0}, {0.0, 1.0}, {1.0}, {1.0, 1.0}),
        std::runtime_error);
}
