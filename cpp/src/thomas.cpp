// thomas.cpp
//
// Implementation of the Thomas algorithm declared in thomas.hpp.

#include "thomas.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace quant::pde {

namespace {

/// Validate the three matrix arrays have consistent sizes.
/// Returns n = diag.size().
std::size_t check_matrix_sizes(const std::vector<double>& sub,
                               const std::vector<double>& diag,
                               const std::vector<double>& sup,
                               const char* fn_name) {
    const std::size_t n = diag.size();
    if (n == 0) {
        throw std::invalid_argument(std::string(fn_name) + ": empty system");
    }
    if (sub.size() != n - 1) {
        throw std::invalid_argument(
            std::string(fn_name) + ": sub must have length n-1 = "
            + std::to_string(n - 1) + ", got " + std::to_string(sub.size()));
    }
    if (sup.size() != n - 1) {
        throw std::invalid_argument(
            std::string(fn_name) + ": sup must have length n-1 = "
            + std::to_string(n - 1) + ", got " + std::to_string(sup.size()));
    }
    return n;
}

}  // anonymous namespace


// ---------------------------------------------------------------------------
// One-shot solver
// ---------------------------------------------------------------------------

std::vector<double> thomas_solve(const std::vector<double>& sub,
                                 const std::vector<double>& diag,
                                 const std::vector<double>& sup,
                                 const std::vector<double>& rhs) {
    const std::size_t n = check_matrix_sizes(sub, diag, sup, "thomas_solve");
    if (rhs.size() != n) {
        throw std::invalid_argument(
            "thomas_solve: rhs has length " + std::to_string(rhs.size())
            + ", expected " + std::to_string(n));
    }

    // Working buffers.
    std::vector<double> c_prime(n > 0 ? n - 1 : 0);
    std::vector<double> d_prime(n);

    // Forward sweep.
    if (diag[0] == 0.0) {
        throw std::runtime_error("thomas_solve: zero pivot at row 0");
    }
    if (n > 1) {
        c_prime[0] = sup[0] / diag[0];
    }
    d_prime[0] = rhs[0] / diag[0];

    for (std::size_t i = 1; i < n; ++i) {
        const double m_i = diag[i] - sub[i - 1] * (i - 1 < c_prime.size()
                                                   ? c_prime[i - 1]
                                                   : 0.0);
        if (m_i == 0.0) {
            throw std::runtime_error(
                "thomas_solve: zero effective pivot at row "
                + std::to_string(i)
                + "; the matrix is singular or not diagonally dominant");
        }
        if (i < n - 1) {
            c_prime[i] = sup[i] / m_i;
        }
        d_prime[i] = (rhs[i] - sub[i - 1] * d_prime[i - 1]) / m_i;
    }

    // Backward substitution.
    std::vector<double> x(n);
    x[n - 1] = d_prime[n - 1];
    for (std::size_t i = n - 1; i > 0; --i) {
        x[i - 1] = d_prime[i - 1] - c_prime[i - 1] * x[i];
    }
    return x;
}


// ---------------------------------------------------------------------------
// Pre-factored interface
// ---------------------------------------------------------------------------

ThomasFactor thomas_factor(const std::vector<double>& sub,
                           const std::vector<double>& diag,
                           const std::vector<double>& sup) {
    const std::size_t n = check_matrix_sizes(sub, diag, sup, "thomas_factor");

    ThomasFactor f;
    f.sub = sub;                   // copy: the d-recursion needs it later
    f.c_prime.resize(n > 0 ? n - 1 : 0);
    f.m.resize(n);

    if (diag[0] == 0.0) {
        throw std::runtime_error("thomas_factor: zero pivot at row 0");
    }
    f.m[0] = diag[0];
    if (n > 1) {
        f.c_prime[0] = sup[0] / f.m[0];
    }

    for (std::size_t i = 1; i < n; ++i) {
        f.m[i] = diag[i] - sub[i - 1] * (i - 1 < f.c_prime.size()
                                         ? f.c_prime[i - 1]
                                         : 0.0);
        if (f.m[i] == 0.0) {
            throw std::runtime_error(
                "thomas_factor: zero effective pivot at row "
                + std::to_string(i));
        }
        if (i < n - 1) {
            f.c_prime[i] = sup[i] / f.m[i];
        }
    }
    return f;
}


std::vector<double> ThomasFactor::solve(const std::vector<double>& rhs) const {
    const std::size_t n = m.size();
    if (rhs.size() != n) {
        throw std::invalid_argument(
            "ThomasFactor::solve: rhs has length " + std::to_string(rhs.size())
            + ", expected " + std::to_string(n));
    }

    // Forward d-recursion. m and c_prime are already known.
    std::vector<double> d_prime(n);
    d_prime[0] = rhs[0] / m[0];
    for (std::size_t i = 1; i < n; ++i) {
        d_prime[i] = (rhs[i] - sub[i - 1] * d_prime[i - 1]) / m[i];
    }

    // Backward substitution.
    std::vector<double> x(n);
    x[n - 1] = d_prime[n - 1];
    for (std::size_t i = n - 1; i > 0; --i) {
        x[i - 1] = d_prime[i - 1] - c_prime[i - 1] * x[i];
    }
    return x;
}

}  // namespace quant::pde
