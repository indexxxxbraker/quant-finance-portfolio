// psor.cpp
//
// Implementation of the PSOR solver declared in psor.hpp.

#include "psor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace quant::pde {

PSORResult psor_solve(
    const std::vector<double>& sub,
    const std::vector<double>& diag,
    const std::vector<double>& sup,
    const std::vector<double>& rhs,
    const std::vector<double>& obstacle,
    double omega,
    double tol_abs,
    double tol_rel,
    int    max_iter,
    const std::vector<double>& x0) {

    const std::size_t n = diag.size();
    if (n == 0) {
        throw std::invalid_argument("psor_solve: empty system");
    }
    if (sub.size() != n - 1) {
        throw std::invalid_argument(
            "psor_solve: sub must have length n-1 = "
            + std::to_string(n - 1) + ", got "
            + std::to_string(sub.size()));
    }
    if (sup.size() != n - 1) {
        throw std::invalid_argument(
            "psor_solve: sup must have length n-1 = "
            + std::to_string(n - 1) + ", got "
            + std::to_string(sup.size()));
    }
    if (rhs.size() != n) {
        throw std::invalid_argument(
            "psor_solve: rhs must have length n = "
            + std::to_string(n) + ", got "
            + std::to_string(rhs.size()));
    }
    if (obstacle.size() != n) {
        throw std::invalid_argument(
            "psor_solve: obstacle must have length n = "
            + std::to_string(n) + ", got "
            + std::to_string(obstacle.size()));
    }
    if (!(0.0 < omega && omega < 2.0)) {
        throw std::invalid_argument(
            "psor_solve: omega must lie strictly in (0, 2), got "
            + std::to_string(omega));
    }
    if (max_iter <= 0) {
        throw std::invalid_argument(
            "psor_solve: max_iter must be positive, got "
            + std::to_string(max_iter));
    }

    // Initial guess.
    std::vector<double> x;
    if (x0.empty()) {
        x = obstacle;
    } else {
        if (x0.size() != n) {
            throw std::invalid_argument(
                "psor_solve: x0 has length " + std::to_string(x0.size())
                + ", expected " + std::to_string(n));
        }
        x = x0;
        for (std::size_t i = 0; i < n; ++i) {
            if (x[i] < obstacle[i] - 1e-14) {
                throw std::invalid_argument(
                    "psor_solve: x0 must satisfy x0 >= obstacle");
            }
        }
    }

    // Pre-compute inverse diagonals.
    std::vector<double> inv_diag(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (diag[i] == 0.0 || !std::isfinite(diag[i])) {
            throw std::invalid_argument(
                "psor_solve: diag[" + std::to_string(i)
                + "] is zero or non-finite");
        }
        inv_diag[i] = 1.0 / diag[i];
    }

    // PSOR iteration.
    for (int k = 0; k < max_iter; ++k) {
        double max_change = 0.0;

        for (std::size_t i = 0; i < n; ++i) {
            double gs_update;
            if (i == 0) {
                gs_update = (rhs[0] - sup[0] * x[1]) * inv_diag[0];
            } else if (i == n - 1) {
                gs_update = (rhs[n - 1] - sub[n - 2] * x[n - 2]) * inv_diag[n - 1];
            } else {
                gs_update = (rhs[i]
                             - sub[i - 1] * x[i - 1]
                             - sup[i]     * x[i + 1]) * inv_diag[i];
            }
            double new_val = (1.0 - omega) * x[i] + omega * gs_update;

            // Project onto the obstacle.
            if (new_val < obstacle[i]) new_val = obstacle[i];

            const double change = std::abs(new_val - x[i]);
            if (change > max_change) max_change = change;

            x[i] = new_val;
        }

        // Stopping criterion.
        double max_x = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double a = std::abs(x[i]);
            if (a > max_x) max_x = a;
        }
        const double threshold = tol_abs + tol_rel * max_x;
        if (max_change < threshold) {
            return {std::move(x), k + 1};
        }
    }

    throw std::runtime_error(
        "psor_solve: did not converge in " + std::to_string(max_iter)
        + " iterations. Try a different omega (current "
        + std::to_string(omega) + ") or check matrix conditioning.");
}

}  // namespace quant::pde
