// american.cpp -- implementation of american.hpp.

#include "american.hpp"

#include "gbm.hpp"   // standard_normal, validators

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace quant {

namespace {

// Validate basis_size in [1, 8]. Numerical conditioning of the
// Laguerre normal-equations matrix degrades rapidly above M = 6 or
// so on the [0, 1] put domain; M = 8 is a defensive upper bound.
void validate_basis_size(std::size_t M) {
    if (M < 1) {
        throw std::invalid_argument("basis_size must be >= 1");
    }
    if (M > 8) {
        throw std::invalid_argument(
            "basis_size must be <= 8 (numerical conditioning)");
    }
}


// Evaluate the first M Laguerre polynomials L_0(x), ..., L_{M-1}(x)
// at scalar x via Bonnet's recurrence:
//   L_0 = 1,  L_1 = 1 - x,
//   (n + 1) L_{n+1} = (2n + 1 - x) L_n - n L_{n-1}.
// Writes M values into out[0..M).
inline void laguerre_basis(double x, std::size_t M, double* out) {
    if (M >= 1) out[0] = 1.0;
    if (M >= 2) out[1] = 1.0 - x;
    for (std::size_t k = 1; k + 1 < M; ++k) {
        out[k + 1] = ((2.0 * static_cast<double>(k) + 1.0 - x) * out[k]
                       - static_cast<double>(k) * out[k - 1])
                     / static_cast<double>(k + 1);
    }
}


// In-place Cholesky factorisation A = L L^T where A is M x M
// positive-definite and stored in row-major order. After the call,
// the lower triangle of A contains L; the upper triangle is left
// alone (and is not consulted by subsequent routines).
//
// Throws std::runtime_error if a diagonal element falls below
// epsilon, which signals A is not positive-definite to within
// working precision (typically because the design matrix Psi is
// rank-deficient at this exercise date).
void cholesky_factorize(double* A, std::size_t M) {
    constexpr double EPS = 1e-12;
    for (std::size_t j = 0; j < M; ++j) {
        double s = A[j * M + j];
        for (std::size_t k = 0; k < j; ++k) {
            s -= A[j * M + k] * A[j * M + k];
        }
        if (s <= EPS) {
            throw std::runtime_error(
                "cholesky_factorize: matrix not positive definite "
                "(diagonal below threshold)");
        }
        A[j * M + j] = std::sqrt(s);
        const double inv_diag = 1.0 / A[j * M + j];
        for (std::size_t i = j + 1; i < M; ++i) {
            double dot = A[i * M + j];
            for (std::size_t k = 0; k < j; ++k) {
                dot -= A[i * M + k] * A[j * M + k];
            }
            A[i * M + j] = dot * inv_diag;
        }
    }
}


// Solve L L^T x = b in place, where L is the lower-triangular
// Cholesky factor stored in the lower triangle of L_storage (M x M
// row-major). Forward substitution L y = b followed by back
// substitution L^T x = y; both phases write into b_then_x.
void cholesky_solve_in_place(const double* L, std::size_t M,
                              double* b_then_x) {
    // Forward substitution: L y = b.
    for (std::size_t i = 0; i < M; ++i) {
        double s = b_then_x[i];
        for (std::size_t k = 0; k < i; ++k) {
            s -= L[i * M + k] * b_then_x[k];
        }
        b_then_x[i] = s / L[i * M + i];
    }
    // Back substitution: L^T x = y.
    for (std::size_t i_plus = M; i_plus > 0; --i_plus) {
        const std::size_t i = i_plus - 1;
        double s = b_then_x[i];
        for (std::size_t k = i + 1; k < M; ++k) {
            s -= L[k * M + i] * b_then_x[k];
        }
        b_then_x[i] = s / L[i * M + i];
    }
}

}  // anonymous namespace


// =====================================================================
// Cox-Ross-Rubinstein binomial tree
// =====================================================================

double
binomial_american_put(double S, double K, double r, double sigma, double T,
                      std::size_t n_steps) {
    validate_model_params(S, sigma, T);
    validate_strike(K);
    validate_n_steps(n_steps);

    const double dt   = T / static_cast<double>(n_steps);
    const double u    = std::exp(sigma * std::sqrt(dt));
    const double d    = 1.0 / u;
    const double disc = std::exp(-r * dt);
    const double p    = (std::exp(r * dt) - d) / (u - d);

    if (p <= 0.0 || p >= 1.0) {
        throw std::invalid_argument(
            "Risk-neutral probability not in (0, 1); "
            "increase n_steps or check r, sigma.");
    }

    // Terminal-node asset prices and option values.
    // S_N(j) = S * u^(2j - N), j = 0 .. N. Use the recursion
    // S_N(j+1) = S_N(j) * u^2 to avoid n_steps calls to std::pow.
    const double u_sq = u * u;
    std::vector<double> V(n_steps + 1);

    {
        double Sj = S * std::pow(d, static_cast<double>(n_steps));
        for (std::size_t j = 0; j <= n_steps; ++j) {
            V[j] = std::max(K - Sj, 0.0);
            Sj *= u_sq;
        }
    }

    // Backward induction. At step k, V holds the option values at
    // the k+1 nodes j = 0..k. We rewrite V in place because each
    // V_new[j] depends only on V[j] and V[j+1] (computed left-to-right).
    for (std::size_t k_plus = n_steps; k_plus > 0; --k_plus) {
        const std::size_t k = k_plus - 1;
        double Sj = S * std::pow(d, static_cast<double>(k));
        for (std::size_t j = 0; j <= k; ++j) {
            const double cont = disc * (p * V[j + 1] + (1.0 - p) * V[j]);
            // Intrinsic is non-negative whenever Sj <= K. If Sj > K,
            // intrinsic is 0; cont is also >= 0; max gives cont.
            V[j] = std::max(K - Sj, cont);
            Sj *= u_sq;
        }
    }

    return V[0];
}


// =====================================================================
// Longstaff-Schwartz Monte Carlo
// =====================================================================

MCResult
lsm_american_put(double S, double K, double r, double sigma, double T,
                 std::size_t n_paths, std::size_t n_steps,
                 std::size_t basis_size,
                 std::mt19937_64& rng,
                 double confidence_level) {
    validate_model_params(S, sigma, T);
    validate_strike(K);
    validate_n_paths(n_paths);
    validate_n_steps(n_steps);
    validate_basis_size(basis_size);
    validate_confidence_level(confidence_level);

    const double dt        = T / static_cast<double>(n_steps);
    const double drift     = (r - 0.5 * sigma * sigma) * dt;
    const double diffusion = sigma * std::sqrt(dt);
    const double disc_step = std::exp(-r * dt);
    const std::size_t M    = basis_size;
    const std::size_t cols = n_steps + 1;

    // ---- Step 1: simulate paths ----
    // paths[i * cols + k] = S_{t_k}^{(i)}.  paths[i * cols + 0] = S.
    std::vector<double> paths(n_paths * cols);
    for (std::size_t i = 0; i < n_paths; ++i) {
        double Sx = S;
        paths[i * cols + 0] = Sx;
        for (std::size_t k = 0; k < n_steps; ++k) {
            const double Z = standard_normal(rng);
            Sx *= std::exp(drift + diffusion * Z);
            paths[i * cols + (k + 1)] = Sx;
        }
    }

    // ---- Step 2: cash flows initialised at maturity ----
    std::vector<double> Y(n_paths);
    for (std::size_t i = 0; i < n_paths; ++i) {
        Y[i] = std::max(K - paths[i * cols + n_steps], 0.0);
    }

    // ---- Step 3: backward induction ----
    // Allocate buffers once; reuse across loop iterations.
    std::vector<std::size_t> itm_idx;
    itm_idx.reserve(n_paths);
    std::vector<double> Psi(n_paths * M);   // row-major (n_itm x M used)
    std::vector<double> A(M * M);
    std::vector<double> b(M);

    for (std::size_t k_plus = n_steps; k_plus > 1; --k_plus) {
        const std::size_t k = k_plus - 1;   // k = n_steps-1, ..., 1

        // Discount cash flows one step.
        for (std::size_t i = 0; i < n_paths; ++i) {
            Y[i] *= disc_step;
        }

        // Identify ITM paths: K - S_k > 0.
        itm_idx.clear();
        for (std::size_t i = 0; i < n_paths; ++i) {
            if (paths[i * cols + k] < K) {
                itm_idx.push_back(i);
            }
        }
        const std::size_t n_itm = itm_idx.size();
        if (n_itm < M) {
            // Cannot identify M coefficients; skip exercise this step.
            continue;
        }

        // Build design matrix: Psi[row, m] = L_m(S_k^(i) / K).
        for (std::size_t row = 0; row < n_itm; ++row) {
            const std::size_t i = itm_idx[row];
            const double x = paths[i * cols + k] / K;
            laguerre_basis(x, M, &Psi[row * M]);
        }

        // Build normal-equations matrix A = Psi^T Psi (M x M)
        // and right-hand side b = Psi^T Y_itm (M).
        for (std::size_t a = 0; a < M; ++a) {
            for (std::size_t bcol = 0; bcol < M; ++bcol) {
                double s = 0.0;
                for (std::size_t row = 0; row < n_itm; ++row) {
                    s += Psi[row * M + a] * Psi[row * M + bcol];
                }
                A[a * M + bcol] = s;
            }
            double s = 0.0;
            for (std::size_t row = 0; row < n_itm; ++row) {
                s += Psi[row * M + a] * Y[itm_idx[row]];
            }
            b[a] = s;
        }

        // Solve A beta = b via Cholesky. If A is rank-deficient
        // (rare; would mean basis values collinear on this set of
        // ITM paths), skip exercise this step.
        try {
            cholesky_factorize(A.data(), M);
        } catch (const std::runtime_error&) {
            continue;
        }
        cholesky_solve_in_place(A.data(), M, b.data());
        // beta is now in b.

        // Apply the exercise rule: if intrinsic >= continuation
        // estimate, set Y_i to intrinsic.
        for (std::size_t row = 0; row < n_itm; ++row) {
            double C_hat = 0.0;
            for (std::size_t m = 0; m < M; ++m) {
                C_hat += Psi[row * M + m] * b[m];
            }
            const std::size_t i = itm_idx[row];
            const double intrinsic = K - paths[i * cols + k];
            if (intrinsic >= C_hat) {
                Y[i] = intrinsic;
            }
        }
    }

    // ---- Step 4: final discount to t_0 ----
    for (std::size_t i = 0; i < n_paths; ++i) {
        Y[i] *= disc_step;
    }

    // ---- Step 5: t_0 exercise comparison ----
    const auto cont_result = mc_estimator(Y, confidence_level);
    const double intrinsic_0 = std::max(K - S, 0.0);
    if (intrinsic_0 > cont_result.estimate) {
        return MCResult{intrinsic_0, 0.0, 0.0, n_paths};
    }
    return cont_result;
}

}  // namespace quant
