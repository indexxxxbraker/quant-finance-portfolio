// pde.cpp
//
// Implementation of the PDE grid infrastructure declared in pde.hpp.
// Phase 3 Block 0.

#include "pde.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace quant::pde {

// ---------------------------------------------------------------------------
// Grid construction
// ---------------------------------------------------------------------------

Grid build_grid(int N, int M, double T,
                double sigma, double r, double K,
                double n_sigma) {
    // Argument validation. Each branch raises std::invalid_argument with
    // a short message identifying which constraint failed.
    if (N < 2) {
        throw std::invalid_argument(
            "build_grid: N must be >= 2, got " + std::to_string(N));
    }
    if (M < 1) {
        throw std::invalid_argument(
            "build_grid: M must be >= 1, got " + std::to_string(M));
    }
    if (T <= 0.0) {
        throw std::invalid_argument(
            "build_grid: T must be positive, got " + std::to_string(T));
    }
    if (sigma <= 0.0) {
        throw std::invalid_argument(
            "build_grid: sigma must be positive, got " + std::to_string(sigma));
    }
    if (K <= 0.0) {
        throw std::invalid_argument(
            "build_grid: K must be positive, got " + std::to_string(K));
    }
    if (n_sigma <= 0.0) {
        throw std::invalid_argument(
            "build_grid: n_sigma must be positive, got " + std::to_string(n_sigma));
    }

    const double half_width = n_sigma * sigma * std::sqrt(T);
    const double x_min = -half_width;
    const double x_max = +half_width;
    const double dx   = (x_max - x_min) / static_cast<double>(N);
    const double dtau = T / static_cast<double>(M);

    Grid g;
    g.N     = N;
    g.M     = M;
    g.T     = T;
    g.sigma = sigma;
    g.r     = r;
    g.K     = K;
    g.x_min = x_min;
    g.x_max = x_max;
    g.dx    = dx;
    g.dtau  = dtau;

    g.xs.resize(static_cast<std::size_t>(N) + 1);
    g.taus.resize(static_cast<std::size_t>(M) + 1);

    // Generate xs by direct formula xs[j] = x_min + j * dx, rather than
    // by repeated accumulation, to keep round-off independent of j and
    // ensure xs[N] equals x_max exactly (or to within one ULP). The
    // same is done for taus.
    for (int j = 0; j <= N; ++j) {
        g.xs[static_cast<std::size_t>(j)] =
            x_min + static_cast<double>(j) * dx;
    }
    for (int n = 0; n <= M; ++n) {
        g.taus[static_cast<std::size_t>(n)] =
            static_cast<double>(n) * dtau;
    }

    return g;
}


// ---------------------------------------------------------------------------
// Initial conditions
// ---------------------------------------------------------------------------

std::vector<double> call_initial_condition(const std::vector<double>& xs,
                                           double K) {
    std::vector<double> ic(xs.size());
    for (std::size_t j = 0; j < xs.size(); ++j) {
        const double payoff = K * (std::exp(xs[j]) - 1.0);
        ic[j] = payoff > 0.0 ? payoff : 0.0;
    }
    return ic;
}

std::vector<double> put_initial_condition(const std::vector<double>& xs,
                                          double K) {
    std::vector<double> ic(xs.size());
    for (std::size_t j = 0; j < xs.size(); ++j) {
        const double payoff = K * (1.0 - std::exp(xs[j]));
        ic[j] = payoff > 0.0 ? payoff : 0.0;
    }
    return ic;
}

// ---------------------------------------------------------------------------
// Boundary conditions
// ---------------------------------------------------------------------------

std::vector<double> call_boundary_lower(const Grid& g) {
    return std::vector<double>(static_cast<std::size_t>(g.M) + 1, 0.0);
}

std::vector<double> call_boundary_upper(const Grid& g) {
    std::vector<double> bc(static_cast<std::size_t>(g.M) + 1);
    const double e_xmax = std::exp(g.x_max);
    for (std::size_t n = 0; n < bc.size(); ++n) {
        bc[n] = g.K * (e_xmax - std::exp(-g.r * g.taus[n]));
    }
    return bc;
}

std::vector<double> put_boundary_lower(const Grid& g) {
    std::vector<double> bc(static_cast<std::size_t>(g.M) + 1);
    const double e_xmin = std::exp(g.x_min);
    for (std::size_t n = 0; n < bc.size(); ++n) {
        bc[n] = g.K * (std::exp(-g.r * g.taus[n]) - e_xmin);
    }
    return bc;
}

std::vector<double> put_boundary_upper(const Grid& g) {
    return std::vector<double>(static_cast<std::size_t>(g.M) + 1, 0.0);
}

// ---------------------------------------------------------------------------
// Stability numbers
// ---------------------------------------------------------------------------

double fourier_number(double sigma, double dtau, double dx) noexcept {
    return 0.5 * sigma * sigma * dtau / (dx * dx);
}

double courant_number(double mu, double dtau, double dx) noexcept {
    return 0.5 * std::abs(mu) * dtau / dx;
}

bool is_explicit_stable(double alpha) noexcept {
    return alpha <= 0.5;
}

}  // namespace quant::pde
