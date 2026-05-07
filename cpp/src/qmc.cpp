// qmc.cpp -- implementation of qmc.hpp.

#include "qmc.hpp"

#include "black_scholes.hpp"   // norm_cdf if needed (not used directly here)
#include "gbm.hpp"              // inverse_normal_cdf, validators

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace quant {

namespace {

// First 20 prime numbers, used as Halton bases.
const int FIRST_PRIMES[20] = {
    2,  3,  5,  7, 11, 13, 17, 19, 23, 29,
   31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
};

// Compute the base-b radical inverse of i.
// Iterative implementation; bounded floating-point accumulator.
double radical_inverse(std::uint64_t i, int base) {
    double result = 0.0;
    double f = 1.0 / static_cast<double>(base);
    while (i > 0) {
        result += f * static_cast<double>(i % static_cast<std::uint64_t>(base));
        i /= static_cast<std::uint64_t>(base);
        f /= static_cast<double>(base);
    }
    return result;
}

// Find the rightmost zero bit of i (1-based index).
// Used for the Sobol Gray-code recursion: u_{i+1} = u_i XOR V[c],
// where c is the position of the rightmost zero bit of i.
//
// For i = 0 (binary ...0), rightmost zero bit is position 1.
// For i = 1 (binary ...01), rightmost zero bit is position 2.
// For i = 2 (binary ...10), rightmost zero bit is position 1.
// For i = 3 (binary ...11), rightmost zero bit is position 3.
// Etc.
inline int rightmost_zero_bit(std::uint64_t i) {
    // Equivalent to: position of lowest 1 bit of (~i), 1-based.
    // For Sobol Gray code, this returns 1 + ctz(~i).
    int pos = 1;
    while (i & 1u) {
        i >>= 1;
        ++pos;
    }
    return pos;
}

// Convert a Sobol-state integer to a double in [0, 1).
// State integers are stored left-shifted to occupy the high bits;
// dividing by 2^64 (as a floating value) gives a value in [0, 1).
inline double state_to_double(std::uint64_t x) {
    // 2^64 as a double (exact since 2^64 is representable in float64).
    constexpr double TWO_64 = 18446744073709551616.0;
    return static_cast<double>(x) / TWO_64;
}

}  // anonymous namespace


// =====================================================================
// Halton implementation
// =====================================================================

Halton::Halton(std::size_t d) : d_(d), index_(1) {
    if (d == 0) {
        throw std::invalid_argument("Halton: d must be >= 1");
    }
    if (d > 20) {
        throw std::invalid_argument(
            "Halton: d > 20 not supported (would need primes beyond 71)");
    }
    primes_.assign(FIRST_PRIMES, FIRST_PRIMES + d);
}

void Halton::generate(std::size_t n, double* out) {
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < d_; ++j) {
            out[i * d_ + j] = radical_inverse(index_ + i, primes_[j]);
        }
    }
    index_ += n;
}


// =====================================================================
// Sobol implementation
// =====================================================================

Sobol::Sobol(std::size_t d, const std::string& data_file)
    : d_(d), index_(1) {
    if (d == 0) {
        throw std::invalid_argument("Sobol: d must be >= 1");
    }
    if (d > MAX_DIM) {
        throw std::invalid_argument(
            "Sobol: d > " + std::to_string(MAX_DIM) +
            " not supported in this build");
    }

    // Number of bits in the Sobol state. We use 32: enough for
    // 2^32 ~ 4 billion points, more than ample for our budgets.
    constexpr int BITS = 32;

    V_.assign(d_, std::vector<std::uint64_t>(BITS + 1, 0));

    // Dimension 1 is special: m_k = 1 for all k, V_k = 2^(BITS - k).
    // (This corresponds to the trivial polynomial "x + 0", i.e. no
    // primitive polynomial information needed.)
    for (int k = 1; k <= BITS; ++k) {
        V_[0][k] = static_cast<std::uint64_t>(1) << (BITS - k);
        // Shift to high bits of uint64_t for state storage.
        V_[0][k] <<= (64 - BITS);
    }

    if (d_ > 1) {
        load_direction_numbers(data_file);
    }

    x_.assign(d_, 0);
}

void Sobol::load_direction_numbers(const std::string& data_file) {
    std::ifstream in(data_file);
    if (!in) {
        throw std::runtime_error(
            "Sobol: cannot open direction-numbers file: " + data_file);
    }

    constexpr int BITS = 32;

    // Skip the header line.
    std::string header;
    std::getline(in, header);

    // Read dimensions 2, 3, ..., d_.
    for (std::size_t j = 1; j < d_; ++j) {
        std::string line;
        if (!std::getline(in, line)) {
            throw std::runtime_error(
                "Sobol: direction-numbers file ended at dimension " +
                std::to_string(j + 1));
        }
        std::istringstream iss(line);

        std::size_t d_index;
        int s;          // degree of primitive polynomial
        std::uint64_t a;  // polynomial coefficient encoding
        iss >> d_index >> s >> a;
        if (!iss) {
            throw std::runtime_error(
                "Sobol: malformed line for dimension " +
                std::to_string(j + 1));
        }

        // Read the s initial direction numbers m_1, ..., m_s.
        std::vector<std::uint64_t> m(s + 1, 0);
        for (int k = 1; k <= s; ++k) {
            iss >> m[k];
        }

        // Extend the m sequence using the recurrence:
        //   m_k = 2 a_1 m_{k-1} XOR 2^2 a_2 m_{k-2} XOR ... XOR
        //         2^{s-1} a_{s-1} m_{k-s+1} XOR (2^s m_{k-s} XOR m_{k-s})
        // for k > s.
        //
        // The polynomial coefficients a_1, ..., a_{s-1} are encoded
        // in `a` (s-1 bits, with a_1 being the MSB).
        for (int k = s + 1; k <= BITS; ++k) {
            std::uint64_t m_k = m[k - s] ^ (m[k - s] << s);
            for (int i = 1; i <= s - 1; ++i) {
                // Extract a_i from the encoding `a`.
                int a_i = static_cast<int>((a >> (s - 1 - i)) & 1);
                if (a_i) {
                    m_k ^= m[k - i] << i;
                }
            }
            m.push_back(m_k);
        }

        // Convert m_k to direction numbers V_k by shifting:
        //   V_k = m_k * 2^{BITS - k}
        // and store left-shifted into the high bits of uint64_t.
        for (int k = 1; k <= BITS; ++k) {
            V_[j][k] = m[k] << (BITS - k);
            V_[j][k] <<= (64 - BITS);
        }
    }
}

void Sobol::reset() {
    index_ = 1;
    std::fill(x_.begin(), x_.end(), 0);
}

void Sobol::generate(std::size_t n, double* out) {
    // We use the Gray-code recursion. State x_[j] holds the current
    // Sobol point in dimension j (as a high-bit-shifted uint64_t).
    // To advance from index i to index i+1: x XOR V_[j][c], where c
    // is the rightmost-zero-bit position of i.
    //
    // Index 0 (the all-zero point) is degenerate under Phi^{-1}, so
    // we skip it: index_ starts at 1 in the constructor, meaning the
    // first call to generate produces points 1, 2, 3, ...
    //
    // Building the i-th point from scratch via XOR over the binary
    // expansion of i requires up to log2(i) operations. The
    // Gray-code shortcut needs only one XOR per dimension per step.
    // To get to index_ = N from a fresh state, we need to advance
    // sequentially from 0; reset() does this implicitly.

    for (std::size_t i = 0; i < n; ++i) {
        // Update state from index_-1 to index_. Use Gray code: c is
        // the position of the rightmost zero bit of (index_ - 1).
        std::uint64_t prev = index_ - 1;
        int c = rightmost_zero_bit(prev);

        for (std::size_t j = 0; j < d_; ++j) {
            x_[j] ^= V_[j][c];
            out[i * d_ + j] = state_to_double(x_[j]);
        }
        ++index_;
    }
}


// =====================================================================
// Digital shift
// =====================================================================

void digital_shift(std::size_t n, std::size_t d,
                   const double* shift,
                   double* points) {
    // Convert to 53-bit unsigned integers, XOR, convert back.
    constexpr double SCALE = 9007199254740992.0;  // 2^53
    constexpr double INV_SCALE = 1.0 / SCALE;

    // Pre-convert shift.
    std::vector<std::uint64_t> shift_int(d);
    for (std::size_t j = 0; j < d; ++j) {
        shift_int[j] = static_cast<std::uint64_t>(shift[j] * SCALE);
    }

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < d; ++j) {
            const std::uint64_t u_int = static_cast<std::uint64_t>(
                points[i * d + j] * SCALE);
            const std::uint64_t shifted = u_int ^ shift_int[j];
            points[i * d + j] = static_cast<double>(shifted) * INV_SCALE;
        }
    }
}


// =====================================================================
// Helper: convert (n_paths, n_steps) uniforms into discounted Euler
// payoffs for the European call.
// =====================================================================

namespace {

// Generates n_paths discounted European-call payoffs from an n_paths-by-n_steps
// row-major matrix of uniform points u in [0, 1)^{n_steps}. Inverts each u_{i,k}
// to a standard normal Z via inverse_normal_cdf, runs the Euler step over
// k = 0, ..., n_steps - 1, and returns the discounted payoff per path.
//
// Clipping: avoid u in {0, 1} which would map to +/- infinity under Phi^{-1}.
void euler_payoffs(const double* u, std::size_t n_paths, std::size_t n_steps,
                   double S, double K, double r, double sigma, double T,
                   std::vector<double>& out) {
    const double h = T / static_cast<double>(n_steps);
    const double sqrt_h = std::sqrt(h);
    const double discount = std::exp(-r * T);
    const double EPS = 1e-15;

    out.resize(n_paths);

    for (std::size_t i = 0; i < n_paths; ++i) {
        double S_path = S;
        for (std::size_t k = 0; k < n_steps; ++k) {
            double u_ik = u[i * n_steps + k];
            // Clip away from 0 and 1.
            if (u_ik < EPS) u_ik = EPS;
            if (u_ik > 1.0 - EPS) u_ik = 1.0 - EPS;

            const double Z = inverse_normal_cdf(u_ik);
            S_path *= 1.0 + r * h + sigma * sqrt_h * Z;
        }
        out[i] = discount * std::max(S_path - K, 0.0);
    }
}

}  // anonymous namespace


// =====================================================================
// Deterministic QMC pricer
// =====================================================================

double mc_european_call_euler_qmc(double S, double K, double r,
                                  double sigma, double T,
                                  std::size_t n_paths,
                                  std::size_t n_steps,
                                  const std::string& sequence) {
    validate_model_params(S, sigma, T);
    validate_strike(K);
    validate_n_paths(n_paths);
    validate_n_steps(n_steps);

    std::vector<double> u(n_paths * n_steps);

    if (sequence == "halton") {
        Halton h(n_steps);
        h.generate(n_paths, u.data());
    } else if (sequence == "sobol") {
        Sobol s(n_steps);
        s.generate(n_paths, u.data());
    } else {
        throw std::invalid_argument(
            "mc_european_call_euler_qmc: sequence must be 'halton' or "
            "'sobol', got '" + sequence + "'");
    }

    std::vector<double> payoffs;
    euler_payoffs(u.data(), n_paths, n_steps, S, K, r, sigma, T, payoffs);

    double sum = 0.0;
    for (double p : payoffs) sum += p;
    return sum / static_cast<double>(n_paths);
}


// =====================================================================
// RQMC pricer with digital shift
// =====================================================================

MCResult
mc_european_call_euler_rqmc(double S, double K, double r,
                            double sigma, double T,
                            std::size_t n_paths,
                            std::size_t n_steps,
                            std::size_t n_replications,
                            std::mt19937_64& rng,
                            double confidence_level) {
    validate_model_params(S, sigma, T);
    validate_strike(K);
    validate_n_paths(n_paths);
    validate_n_steps(n_steps);
    validate_confidence_level(confidence_level);

    if (n_replications < 2) {
        throw std::invalid_argument(
            "mc_european_call_euler_rqmc: n_replications must be >= 2 "
            "for a meaningful sample variance");
    }

    // Generate the base Sobol point set ONCE.
    Sobol sob(n_steps);
    std::vector<double> u_base(n_paths * n_steps);
    sob.generate(n_paths, u_base.data());

    // Build R replication estimates.
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::vector<double> replication_estimates(n_replications);
    std::vector<double> u_shifted(n_paths * n_steps);
    std::vector<double> shift(n_steps);
    std::vector<double> payoffs;

    for (std::size_t r_idx = 0; r_idx < n_replications; ++r_idx) {
        // Draw a fresh shift.
        for (std::size_t j = 0; j < n_steps; ++j) {
            shift[j] = uniform(rng);
        }

        // Copy and shift.
        std::memcpy(u_shifted.data(), u_base.data(),
                    sizeof(double) * u_base.size());
        digital_shift(n_paths, n_steps, shift.data(), u_shifted.data());

        // Compute Euler payoffs.
        euler_payoffs(u_shifted.data(), n_paths, n_steps,
                      S, K, r, sigma, T, payoffs);

        // Mean of payoffs = this replication's estimate.
        double sum = 0.0;
        for (double p : payoffs) sum += p;
        replication_estimates[r_idx] = sum / static_cast<double>(n_paths);
    }

    // Reduce: estimate, sample variance, half-width over the R
    // replications. The CLT applies to these R i.i.d. estimates.
    return mc_estimator(replication_estimates, confidence_level);
}

}  // namespace quant
