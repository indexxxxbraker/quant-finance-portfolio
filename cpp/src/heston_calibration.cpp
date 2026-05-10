// heston_calibration.cpp
//
// See heston_calibration.hpp for design notes and
// theory/phase4/block6_heston_calibration_exotics.tex for the mathematical
// formulation.

#include "heston_calibration.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

namespace quant::heston {

namespace {

// =====================================================================
// Standard normal helpers
// =====================================================================

inline double phi(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

inline double Phi(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}


// =====================================================================
// Black-Scholes call and vega
// =====================================================================

void bs_call_and_vega(double sigma, double S0, double K, double T, double r,
                       double& price, double& vega) {
    if (sigma <= 0.0 || T <= 0.0) { price = std::nan(""); vega = 0.0; return; }
    const double sqrt_T = std::sqrt(T);
    const double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T)
                      / (sigma * sqrt_T);
    const double d2 = d1 - sigma * sqrt_T;
    price = S0 * Phi(d1) - K * std::exp(-r * T) * Phi(d2);
    vega  = S0 * sqrt_T * phi(d1);
}


// =====================================================================
// Cholesky solve for 5x5 symmetric positive-definite system
// =====================================================================
// Solves A x = b with A symmetric positive definite, in place.
// Returns false if the matrix is not positive definite (e.g., during
// pathological LM iterations); the caller should handle this by
// increasing the damping parameter.

bool cholesky_solve_5(double A[5][5], const double b[5], double x[5]) {
    double L[5][5] = {{0.0}};
    for (int j = 0; j < 5; ++j) {
        double sum = A[j][j];
        for (int k = 0; k < j; ++k) sum -= L[j][k] * L[j][k];
        if (sum <= 0.0) return false;  // not positive-definite
        L[j][j] = std::sqrt(sum);
        for (int i = j + 1; i < 5; ++i) {
            double s2 = A[i][j];
            for (int k = 0; k < j; ++k) s2 -= L[i][k] * L[j][k];
            L[i][j] = s2 / L[j][j];
        }
    }
    // Forward solve: L y = b
    double y[5];
    for (int i = 0; i < 5; ++i) {
        double s = b[i];
        for (int k = 0; k < i; ++k) s -= L[i][k] * y[k];
        y[i] = s / L[i][i];
    }
    // Backward solve: L^T x = y
    for (int i = 4; i >= 0; --i) {
        double s = y[i];
        for (int k = i + 1; k < 5; ++k) s -= L[k][i] * x[k];
        x[i] = s / L[i][i];
    }
    return true;
}


// =====================================================================
// Parameter packing / unpacking
// =====================================================================
// Internal coordinates z = (log kappa, log theta, log sigma, atanh rho,
// log v0) live in R^5 unconstrained. The transformation guarantees that
// any z gives a valid HestonParams; no bounds enforcement needed in LM.

HestonParams unpack_params(const double z[5]) {
    HestonParams p;
    p.kappa = std::exp(z[0]);
    p.theta = std::exp(z[1]);
    p.sigma = std::exp(z[2]);
    p.rho   = std::tanh(z[3]);
    p.v0    = std::exp(z[4]);
    return p;
}

void pack_params(const HestonParams& p, double z[5]) {
    z[0] = std::log(p.kappa);
    z[1] = std::log(p.theta);
    z[2] = std::log(p.sigma);
    // Clamp rho slightly inside (-1, 1) for numerical safety on atanh
    const double rho_safe = std::max(std::min(p.rho, 0.999999), -0.999999);
    z[3] = std::atanh(rho_safe);
    z[4] = std::log(p.v0);
}


// =====================================================================
// Residuals and Jacobian
// =====================================================================
// Residual r_i = sqrt(w_i) * (model_i - market_i).
// Jacobian J_{ij} = d r_i / d z_j computed by central finite differences.

void compute_residuals(const std::vector<CalibrationQuote>& market,
                        double S0, double r,
                        const HestonParams& p,
                        const std::vector<double>& sqrt_w,
                        std::vector<double>& residuals_out) {
    const std::size_t N = market.size();
    for (std::size_t i = 0; i < N; ++i) {
        const double model = heston_call_lewis(
            market[i].K, market[i].T, S0, r, p);
        residuals_out[i] = sqrt_w[i] * (model - market[i].C_market);
    }
}

void compute_jacobian(const std::vector<CalibrationQuote>& market,
                        double S0, double r,
                        const double z[5],
                        const std::vector<double>& sqrt_w,
                        std::vector<std::vector<double>>& J_out) {
    const std::size_t N = market.size();
    const double eps = 1e-5;
    std::vector<double> r_plus(N), r_minus(N);
    for (int j = 0; j < 5; ++j) {
        double z_p[5], z_m[5];
        std::copy(z, z + 5, z_p);
        std::copy(z, z + 5, z_m);
        z_p[j] = z[j] + eps;
        z_m[j] = z[j] - eps;
        compute_residuals(market, S0, r, unpack_params(z_p), sqrt_w, r_plus);
        compute_residuals(market, S0, r, unpack_params(z_m), sqrt_w, r_minus);
        for (std::size_t i = 0; i < N; ++i) {
            J_out[i][j] = (r_plus[i] - r_minus[i]) / (2.0 * eps);
        }
    }
}


// =====================================================================
// Vega weights
// =====================================================================

std::vector<double> compute_vega_weights(
        const std::vector<CalibrationQuote>& market, double S0, double r) {
    const std::size_t N = market.size();
    std::vector<double> weights(N);
    const double vega_floor = 1e-3;
    for (std::size_t i = 0; i < N; ++i) {
        const double iv = implied_vol_bs(market[i].C_market,
                                          market[i].K, market[i].T,
                                          S0, r);
        if (std::isnan(iv)) {
            weights[i] = 1.0;
            continue;
        }
        double price_unused, vega;
        bs_call_and_vega(iv, S0, market[i].K, market[i].T, r,
                          price_unused, vega);
        weights[i] = 1.0 / std::max(vega, vega_floor);
    }
    const double max_w = *std::max_element(weights.begin(), weights.end());
    if (max_w > 0.0) {
        for (auto& w : weights) w /= max_w;
    }
    return weights;
}

}  // anonymous namespace


// =====================================================================
// Public: implied vol inversion
// =====================================================================

double implied_vol_bs(double C_target, double K, double T,
                        double S0, double r,
                        double sigma0, int max_iter, double tol) {
    // No-arbitrage bounds
    const double intrinsic = std::max(S0 - K * std::exp(-r * T), 0.0);
    if (C_target < intrinsic - 1e-12) return std::nan("");
    if (C_target > S0 + 1e-12)        return std::nan("");

    // Newton-Raphson
    double sigma = sigma0;
    for (int iter = 0; iter < max_iter; ++iter) {
        double price, vega;
        bs_call_and_vega(sigma, S0, K, T, r, price, vega);
        const double diff = price - C_target;
        if (std::abs(diff) < tol) return sigma;
        if (vega < 1e-12) break;
        double sigma_new = sigma - diff / vega;
        sigma_new = std::max(0.001, std::min(5.0, sigma_new));
        if (std::abs(sigma_new - sigma) < 1e-14) return sigma_new;
        sigma = sigma_new;
    }

    // Bisection fallback
    auto f = [&](double s) {
        double price, vega;
        bs_call_and_vega(s, S0, K, T, r, price, vega);
        return price - C_target;
    };
    double lo = 1e-4, hi = 5.0;
    double f_lo = f(lo), f_hi = f(hi);
    if (f_lo * f_hi > 0.0) return std::nan("");
    for (int iter = 0; iter < 200; ++iter) {
        const double mid = 0.5 * (lo + hi);
        const double f_mid = f(mid);
        if (std::abs(f_mid) < tol || (hi - lo) < 1e-10) return mid;
        if (f_lo * f_mid < 0.0) { hi = mid; f_hi = f_mid; }
        else                     { lo = mid; f_lo = f_mid; }
    }
    return 0.5 * (lo + hi);
}


// =====================================================================
// Public: calibration via Levenberg-Marquardt
// =====================================================================

CalibrationResult
calibrate_heston(const std::vector<CalibrationQuote>& market,
                   double S0, double r,
                   const HestonParams& initial_guess,
                   bool weighted, int max_iter) {
    if (market.size() < 5)
        throw std::invalid_argument("need at least 5 market observations");
    if (S0 <= 0.0)
        throw std::invalid_argument("S0 must be positive");
    initial_guess.validate();

    const std::size_t N = market.size();

    // Vega weights
    std::vector<double> sqrt_w(N, 1.0);
    if (weighted) {
        const auto w = compute_vega_weights(market, S0, r);
        for (std::size_t i = 0; i < N; ++i) sqrt_w[i] = std::sqrt(w[i]);
    }

    // Initial point in z-coordinates
    double z[5];
    pack_params(initial_guess, z);

    // Buffers
    std::vector<double> residuals(N);
    std::vector<std::vector<double>> J(N, std::vector<double>(5));

    // Initial loss
    compute_residuals(market, S0, r, unpack_params(z), sqrt_w, residuals);
    double current_loss = 0.0;
    for (auto v : residuals) current_loss += v * v;

    // Levenberg-Marquardt main loop
    double lambda = 1e-3;
    int iter_count = 0;
    bool converged = false;
    const double conv_tol = 1e-10;

    for (int iter = 0; iter < max_iter; ++iter) {
        iter_count = iter + 1;

        compute_jacobian(market, S0, r, z, sqrt_w, J);

        // Build normal equations: A = J^T J + lambda * diag(J^T J), g = J^T r
        // Using Marquardt's variant (lambda * diag(J^T J)) is more robust
        // than Levenberg's plain (lambda * I).
        double A[5][5] = {{0.0}};
        double g[5]    = {0.0};
        for (int a = 0; a < 5; ++a) {
            for (int b_ = 0; b_ < 5; ++b_) {
                double s = 0.0;
                for (std::size_t i = 0; i < N; ++i) s += J[i][a] * J[i][b_];
                A[a][b_] = s;
            }
            double s = 0.0;
            for (std::size_t i = 0; i < N; ++i) s += J[i][a] * residuals[i];
            g[a] = -s;  // RHS for delta = (J^T J)^-1 (-J^T r)
        }
        for (int a = 0; a < 5; ++a) A[a][a] *= (1.0 + lambda);

        // Solve A delta = g
        double delta[5];
        if (!cholesky_solve_5(A, g, delta)) {
            // Not positive-definite: increase damping and retry
            lambda *= 10.0;
            continue;
        }

        // Try the step
        double z_new[5];
        for (int a = 0; a < 5; ++a) z_new[a] = z[a] + delta[a];

        std::vector<double> new_residuals(N);
        compute_residuals(market, S0, r, unpack_params(z_new),
                           sqrt_w, new_residuals);
        double new_loss = 0.0;
        for (auto v : new_residuals) new_loss += v * v;

        if (new_loss < current_loss) {
            // Accept step
            std::copy(z_new, z_new + 5, z);
            residuals = new_residuals;
            current_loss = new_loss;
            lambda = std::max(lambda * 0.5, 1e-12);

            // Convergence check: max absolute step
            double max_step = 0.0;
            for (int a = 0; a < 5; ++a)
                max_step = std::max(max_step, std::abs(delta[a]));
            if (max_step < conv_tol) {
                converged = true;
                break;
            }
        } else {
            // Reject step, increase damping
            lambda *= 10.0;
            if (lambda > 1e12) break;
        }
    }

    // Build result
    CalibrationResult result;
    result.params = unpack_params(z);
    // Final raw residuals (unweighted)
    result.residuals.resize(N);
    for (std::size_t i = 0; i < N; ++i) {
        const double model = heston_call_lewis(
            market[i].K, market[i].T, S0, r, result.params);
        result.residuals[i] = model - market[i].C_market;
    }
    double sse = 0.0;
    for (auto v : result.residuals) sse += v * v;
    result.rmse = std::sqrt(sse / static_cast<double>(N));
    result.n_iter = iter_count;
    result.success = converged;
    return result;
}

}  // namespace quant::heston
