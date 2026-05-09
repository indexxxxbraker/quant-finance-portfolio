// heston_pde.cpp
//
// See heston_pde.hpp for the public interface and
// theory/phase4/block5_heston_pde.tex for the mathematical derivation.

#include "heston_pde.hpp"
#include "gbm.hpp"  // validate_strike helpers (reused; not strictly all needed)

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace quant::heston {

namespace {

// =====================================================================
// 2D grid storage: row-major flat vector
// =====================================================================
// Index (i, j) where i in [0, N_X], j in [0, N_v] maps to
// i * stride + j, with stride = N_v + 1. We define small wrappers for
// readability; the compiler inlines them with -O2.

struct Grid2D {
    std::vector<double> data;
    std::size_t N_X;
    std::size_t N_v;

    Grid2D(std::size_t Nx, std::size_t Nv)
        : data((Nx + 1) * (Nv + 1), 0.0), N_X(Nx), N_v(Nv) {}

    inline std::size_t stride() const { return N_v + 1; }

    inline double& at(std::size_t i, std::size_t j) {
        return data[i * stride() + j];
    }
    inline double at(std::size_t i, std::size_t j) const {
        return data[i * stride() + j];
    }
};


// =====================================================================
// Input validation
// =====================================================================

void check_inputs(double S0, double K, double T,
                   const HestonParams& p,
                   std::size_t N_X, std::size_t N_v, std::size_t N_tau,
                   double theta_imp,
                   double X_factor, double v_max_factor) {
    p.validate();
    if (S0 <= 0.0) throw std::invalid_argument("S0 must be positive");
    if (K  <= 0.0) throw std::invalid_argument("K must be positive");
    if (T  <= 0.0) throw std::invalid_argument("T must be positive");
    if (N_X < 4) throw std::invalid_argument("N_X must be at least 4");
    if (N_v < 4) throw std::invalid_argument("N_v must be at least 4");
    if (N_tau < 4) throw std::invalid_argument("N_tau must be at least 4");
    if (theta_imp < 0.0 || theta_imp > 1.0)
        throw std::invalid_argument("theta_imp must be in [0, 1]");
    if (X_factor <= 0.0)
        throw std::invalid_argument("X_factor must be positive");
    if (v_max_factor <= 1.0)
        throw std::invalid_argument(
            "v_max_factor must be > 1 (so v_max > theta)");
}


// =====================================================================
// Operator coefficients (depend on v but not on X)
// =====================================================================

// L_X coefficients per v-level: (r - v/2) d_X + (v/2) d_XX, centred FD.
// At interior i:
//   (L_X W)[i, j] = a[j] W[i-1, j] + b[j] W[i, j] + c[j] W[i+1, j]
struct LXCoefficients {
    std::vector<double> a, b, c;  // size N_v + 1 each
};

LXCoefficients build_LX(const std::vector<double>& v_grid,
                          double dX, double r) {
    const std::size_t N_v1 = v_grid.size();
    LXCoefficients k;
    k.a.resize(N_v1);
    k.b.resize(N_v1);
    k.c.resize(N_v1);
    const double dX2 = dX * dX;
    for (std::size_t j = 0; j < N_v1; ++j) {
        const double half_v = 0.5 * v_grid[j];
        const double drift  = r - half_v;
        const double diff   = half_v;
        k.a[j] = diff / dX2 - drift / (2.0 * dX);
        k.b[j] = -2.0 * diff / dX2;
        k.c[j] = diff / dX2 + drift / (2.0 * dX);
    }
    return k;
}


// L_v coefficients per v-level (interior j only): kappa(theta - v) d_v
// + (sigma^2 / 2) v d_vv. At interior j:
//   (L_v W)[i, j] = a[j] W[i, j-1] + b[j] W[i, j] + c[j] W[i, j+1]
struct LvCoefficients {
    std::vector<double> a, b, c;  // size N_v + 1 each (boundaries 0/N_v unused)
};

LvCoefficients build_Lv(const std::vector<double>& v_grid,
                          double dv, double kappa, double theta, double sigma) {
    const std::size_t N_v1 = v_grid.size();
    LvCoefficients k;
    k.a.resize(N_v1);
    k.b.resize(N_v1);
    k.c.resize(N_v1);
    const double dv2 = dv * dv;
    for (std::size_t j = 0; j < N_v1; ++j) {
        const double drift = kappa * (theta - v_grid[j]);
        const double diff  = 0.5 * sigma * sigma * v_grid[j];
        k.a[j] = diff / dv2 - drift / (2.0 * dv);
        k.b[j] = -2.0 * diff / dv2;
        k.c[j] = diff / dv2 + drift / (2.0 * dv);
    }
    return k;
}


// =====================================================================
// Explicit operator application: out = L_op * W
// =====================================================================

void apply_LX(const Grid2D& W, const LXCoefficients& k, Grid2D& out) {
    const std::size_t N_X = W.N_X, N_v = W.N_v;
    // Initialise out to zero (boundaries i=0, N_X stay zero)
    std::fill(out.data.begin(), out.data.end(), 0.0);
    // Interior i = 1..N_X-1
    for (std::size_t i = 1; i < N_X; ++i) {
        for (std::size_t j = 0; j <= N_v; ++j) {
            out.at(i, j) = k.a[j] * W.at(i - 1, j)
                            + k.b[j] * W.at(i,     j)
                            + k.c[j] * W.at(i + 1, j);
        }
    }
}


void apply_Lv(const Grid2D& W, const LvCoefficients& k,
                 double kappa, double theta, double dv,
                 const std::vector<double>& v_grid,
                 Grid2D& out) {
    const std::size_t N_X = W.N_X, N_v = W.N_v;
    std::fill(out.data.begin(), out.data.end(), 0.0);
    // Interior j = 1..N_v-1
    for (std::size_t i = 0; i <= N_X; ++i) {
        for (std::size_t j = 1; j < N_v; ++j) {
            out.at(i, j) = k.a[j] * W.at(i, j - 1)
                            + k.b[j] * W.at(i, j)
                            + k.c[j] * W.at(i, j + 1);
        }
    }
    // j = 0 (v=0 boundary): forward upwind, drift = kappa * theta > 0
    const double k_th_over_dv = kappa * theta / dv;
    for (std::size_t i = 0; i <= N_X; ++i) {
        out.at(i, 0) = k_th_over_dv * (W.at(i, 1) - W.at(i, 0));
    }
    // j = N_v (v=v_max boundary): backward upwind, drift = kappa(theta - v_max) < 0
    const double v_max = v_grid.back();
    const double drift_max_over_dv = kappa * (theta - v_max) / dv;
    for (std::size_t i = 0; i <= N_X; ++i) {
        out.at(i, N_v) = drift_max_over_dv
                         * (W.at(i, N_v) - W.at(i, N_v - 1));
    }
}


void apply_Lxv(const Grid2D& W,
                 double dX, double dv, double rho, double sigma,
                 const std::vector<double>& v_grid,
                 Grid2D& out) {
    const std::size_t N_X = W.N_X, N_v = W.N_v;
    std::fill(out.data.begin(), out.data.end(), 0.0);
    const double prefactor = rho * sigma / (4.0 * dX * dv);
    // Interior (i, j) only; boundaries stay zero.
    for (std::size_t i = 1; i < N_X; ++i) {
        for (std::size_t j = 1; j < N_v; ++j) {
            const double coef = prefactor * v_grid[j];
            out.at(i, j) = coef * (W.at(i + 1, j + 1)
                                    - W.at(i + 1, j - 1)
                                    - W.at(i - 1, j + 1)
                                    + W.at(i - 1, j - 1));
        }
    }
}


// =====================================================================
// Thomas tridiagonal solver (in-place, on a single system)
// =====================================================================
// Given (sub, diag, sup, rhs) of size n, solves the tridiagonal system
// in place, overwriting rhs with the solution. The arrays sub, diag,
// sup are also overwritten as workspace.
//
// Preconditions: diag[0] != 0 and pivots remain non-zero throughout.
// For Heston ADI with reasonable dtau, this is the case.
//
// The 1D thomas_solve in quant::thomas (Phase 3 Block 2) does the
// same algorithm, but takes std::vector<double> arguments and returns
// a new vector. Our use case is line-by-line on a 2D grid where we
// want zero-allocation per line; hence the in-place buffer-based API.

void thomas_solve_inplace(double* sub, double* diag, double* sup,
                            double* rhs, std::size_t n) {
    // Forward sweep
    sup[0] /= diag[0];
    rhs[0] /= diag[0];
    for (std::size_t i = 1; i < n; ++i) {
        const double denom = diag[i] - sub[i] * sup[i - 1];
        if (i < n - 1) sup[i] /= denom;
        rhs[i] = (rhs[i] - sub[i] * rhs[i - 1]) / denom;
    }
    // Back substitution: rhs is the solution
    for (std::size_t i = n - 1; i-- > 0; ) {
        rhs[i] -= sup[i] * rhs[i + 1];
    }
}


// =====================================================================
// Implicit solves: (I - factor L_op) Y = RHS
// =====================================================================

// Solve (I - factor L_X) Y = RHS in place on RHS.
// For each row j (variance level), we solve a tridiagonal system in i.
// Boundaries i = 0, N_X: identity (Y = RHS).
//
// For each j we copy the L_X coefficients (which are scalars for that
// j) into per-line tridiagonal buffers, then call thomas_solve_inplace.
void solve_implicit_LX(Grid2D& RHS,
                          const LXCoefficients& k,
                          double theta_imp, double dtau) {
    const std::size_t N_X = RHS.N_X, N_v = RHS.N_v;
    const double factor = theta_imp * dtau;
    const std::size_t n = N_X + 1;

    // Per-line buffers (allocated once, reused for each j)
    std::vector<double> sub(n), diag(n), sup(n), rhs(n);

    for (std::size_t j = 0; j <= N_v; ++j) {
        // Build the tridiagonal for line j.
        // Boundaries: identity rows (Y = RHS)
        sub[0]   = 0.0;
        diag[0]  = 1.0;
        sup[0]   = 0.0;
        sub[n-1] = 0.0;
        diag[n-1] = 1.0;
        sup[n-1] = 0.0;
        // Interior i = 1..N_X-1: same coefficients k.a[j], k.b[j], k.c[j]
        const double ai = -factor * k.a[j];
        const double bi = 1.0 - factor * k.b[j];
        const double ci = -factor * k.c[j];
        for (std::size_t i = 1; i < N_X; ++i) {
            sub[i]  = ai;
            diag[i] = bi;
            sup[i]  = ci;
        }
        // Pull out RHS line j into rhs buffer
        for (std::size_t i = 0; i <= N_X; ++i) rhs[i] = RHS.at(i, j);
        // Solve in place
        thomas_solve_inplace(sub.data(), diag.data(), sup.data(),
                              rhs.data(), n);
        // Write back
        for (std::size_t i = 0; i <= N_X; ++i) RHS.at(i, j) = rhs[i];
    }
}


// Solve (I - factor L_v) Y = RHS in place on RHS.
// For each column i, we solve a tridiagonal system in j.
// Boundary j = 0: upwind v=0 PDE (forward difference).
// Boundary j = N_v: upwind v=v_max PDE (backward difference).
void solve_implicit_Lv(Grid2D& RHS,
                          const LvCoefficients& k,
                          double kappa, double theta, double dv,
                          const std::vector<double>& v_grid,
                          double theta_imp, double dtau) {
    const std::size_t N_X = RHS.N_X, N_v = RHS.N_v;
    const double factor = theta_imp * dtau;
    const std::size_t n = N_v + 1;
    const double v_max = v_grid.back();
    const double k_th_over_dv = kappa * theta / dv;
    const double drift_max_over_dv = kappa * (theta - v_max) / dv;

    std::vector<double> sub(n), diag(n), sup(n), rhs(n);

    for (std::size_t i = 0; i <= N_X; ++i) {
        // Boundary j = 0: from upwind L_v at v=0
        // Row: (1 + factor * k_th_over_dv) Y[0] - factor * k_th_over_dv Y[1] = RHS[0]
        sub[0]  = 0.0;
        diag[0] = 1.0 + factor * k_th_over_dv;
        sup[0]  = -factor * k_th_over_dv;
        // Boundary j = N_v: from upwind L_v at v_max with backward diff
        // Row: (factor * drift_max_over_dv) Y[N-1]
        //      + (1 - factor * drift_max_over_dv) Y[N] = RHS[N]
        // (drift_max_over_dv is negative)
        sub[n-1]  = factor * drift_max_over_dv;
        diag[n-1] = 1.0 - factor * drift_max_over_dv;
        sup[n-1]  = 0.0;
        // Interior j = 1..N_v-1: standard centred stencil
        for (std::size_t j = 1; j < N_v; ++j) {
            sub[j]  = -factor * k.a[j];
            diag[j] = 1.0 - factor * k.b[j];
            sup[j]  = -factor * k.c[j];
        }
        // Pull out RHS column i
        for (std::size_t j = 0; j <= N_v; ++j) rhs[j] = RHS.at(i, j);
        thomas_solve_inplace(sub.data(), diag.data(), sup.data(),
                              rhs.data(), n);
        for (std::size_t j = 0; j <= N_v; ++j) RHS.at(i, j) = rhs[j];
    }
}


// =====================================================================
// One Douglas step
// =====================================================================

void douglas_step(Grid2D& W,
                    Grid2D& tmp_LX, Grid2D& tmp_Lv, Grid2D& tmp_Lxv,
                    Grid2D& tmp_RHS,
                    double dtau, double dX, double dv,
                    const std::vector<double>& v_grid,
                    double kappa, double theta, double sigma, double rho,
                    double theta_imp,
                    const LXCoefficients& kX, const LvCoefficients& kv) {
    // Compute explicit operator applications on current W
    apply_LX (W, kX, tmp_LX);
    apply_Lv (W, kv, kappa, theta, dv, v_grid, tmp_Lv);
    apply_Lxv(W, dX, dv, rho, sigma, v_grid, tmp_Lxv);

    // Predictor: Y0 = W + dtau (LX_W + Lv_W + Lxv_W)
    // We store Y0 in tmp_RHS to free up W for later overwriting.
    const std::size_t total = W.data.size();
    for (std::size_t k = 0; k < total; ++k) {
        tmp_RHS.data[k] = W.data[k]
                          + dtau * (tmp_LX.data[k] + tmp_Lv.data[k] + tmp_Lxv.data[k]);
    }
    // Now tmp_RHS = Y0.

    // Implicit X: solve (I - theta_imp dtau L_X) Y1 = Y0 - theta_imp dtau LX_W
    // RHS_X stored in tmp_RHS (overwriting Y0; we don't need Y0 after this)
    const double factor = theta_imp * dtau;
    for (std::size_t k = 0; k < total; ++k) {
        tmp_RHS.data[k] -= factor * tmp_LX.data[k];
    }
    solve_implicit_LX(tmp_RHS, kX, theta_imp, dtau);
    // Now tmp_RHS = Y1.

    // Implicit v: solve (I - theta_imp dtau L_v) Y2 = Y1 - theta_imp dtau Lv_W
    for (std::size_t k = 0; k < total; ++k) {
        tmp_RHS.data[k] -= factor * tmp_Lv.data[k];
    }
    solve_implicit_Lv(tmp_RHS, kv, kappa, theta, dv, v_grid,
                        theta_imp, dtau);
    // Now tmp_RHS = Y2.

    // Copy result back to W
    std::swap(W.data, tmp_RHS.data);
}

}  // anonymous namespace


// =====================================================================
// Public pricer
// =====================================================================

double heston_call_pde(double S0, double K, double T, double r,
                         const HestonParams& p,
                         std::size_t N_X, std::size_t N_v, std::size_t N_tau,
                         double theta_imp,
                         double X_factor, double v_max_factor) {
    check_inputs(S0, K, T, p, N_X, N_v, N_tau, theta_imp,
                  X_factor, v_max_factor);

    // Build grids
    const double log_S0 = std::log(S0);
    const double v_max  = v_max_factor * p.theta;
    const double half_width = X_factor * std::sqrt(v_max * T);

    std::vector<double> X_grid(N_X + 1), v_grid(N_v + 1);
    const double dX = (2.0 * half_width) / static_cast<double>(N_X);
    const double dv = v_max / static_cast<double>(N_v);
    const double dtau = T / static_cast<double>(N_tau);

    for (std::size_t i = 0; i <= N_X; ++i)
        X_grid[i] = log_S0 - half_width + static_cast<double>(i) * dX;
    for (std::size_t j = 0; j <= N_v; ++j)
        v_grid[j] = static_cast<double>(j) * dv;

    // Pre-compute operator coefficients (they don't change with tau)
    const auto kX = build_LX(v_grid, dX, r);
    const auto kv = build_Lv(v_grid, dv, p.kappa, p.theta, p.sigma);

    // Allocate the working grid and temporary buffers
    Grid2D W      (N_X, N_v);
    Grid2D tmp_LX (N_X, N_v);
    Grid2D tmp_Lv (N_X, N_v);
    Grid2D tmp_Lxv(N_X, N_v);
    Grid2D tmp_RHS(N_X, N_v);

    // Initial condition: W(0, X, v) = max(exp(X) - K, 0), independent of v
    for (std::size_t i = 0; i <= N_X; ++i) {
        const double payoff = std::max(std::exp(X_grid[i]) - K, 0.0);
        for (std::size_t j = 0; j <= N_v; ++j) {
            W.at(i, j) = payoff;
        }
    }

    // Time-stepping
    for (std::size_t n = 0; n < N_tau; ++n) {
        douglas_step(W, tmp_LX, tmp_Lv, tmp_Lxv, tmp_RHS,
                       dtau, dX, dv, v_grid,
                       p.kappa, p.theta, p.sigma, p.rho,
                       theta_imp, kX, kv);
    }

    // Bilinear interpolation at (log_S0, p.v0)
    auto find_index = [](const std::vector<double>& g, double x,
                          std::size_t n_minus_1) -> std::size_t {
        // Linear search; small grids so this is fine
        for (std::size_t k = 0; k < n_minus_1; ++k) {
            if (g[k + 1] >= x) return k;
        }
        return n_minus_1 - 1;
    };
    const std::size_t i0 = find_index(X_grid, log_S0, N_X);
    const std::size_t j0 = find_index(v_grid, p.v0, N_v);

    const double x_frac = (log_S0 - X_grid[i0]) / (X_grid[i0 + 1] - X_grid[i0]);
    const double v_frac = (p.v0 - v_grid[j0])  / (v_grid[j0 + 1] - v_grid[j0]);

    const double W_interp = W.at(i0,     j0)     * (1.0 - x_frac) * (1.0 - v_frac)
                            + W.at(i0 + 1, j0)     * x_frac         * (1.0 - v_frac)
                            + W.at(i0,     j0 + 1) * (1.0 - x_frac) * v_frac
                            + W.at(i0 + 1, j0 + 1) * x_frac         * v_frac;

    // Undo the discount substitution: W = exp(r tau) U, so V = exp(-r T) W
    return std::exp(-r * T) * W_interp;
}

}  // namespace quant::heston
