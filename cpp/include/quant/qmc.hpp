// qmc.hpp
//
// Quasi-Monte Carlo and Randomized QMC pricers for the European call.
//
// This module implements the Block 3 part of the variance-reduction
// toolkit: deterministic QMC sequences (Halton, Sobol) and a randomized
// QMC pricer that uses digital shifting to recover a half-width.
//
// The Sobol implementation is from-scratch with Joe-Kuo direction
// numbers (file new-joe-kuo-6.21201, distributed with the project under
// BSD-style licence and located at SOBOL_DATA_FILE, defined as a
// compile-time macro in CMakeLists.txt).
//
// The pricers exposed here use the Euler discretisation of Block 1.2.1
// with N steps; the integration problem is N-dimensional. Validation
// is performed at d = 20 in Block 3.1.
//
// References:
//   Phase 2 Block 3.0 writeup (foundations); Block 3.1 writeup
//   (this implementation). Glasserman, Chapter 5.
//   Joe & Kuo, SIAM J. Sci. Comput. 30:2635-2654, 2008.

#pragma once

#include "monte_carlo.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace quant {

// =====================================================================
// Halton sequence (from-scratch implementation)
// =====================================================================

// d-dimensional Halton sequence. Each coordinate j uses the prime
// base p_{j+1} (coord 0 uses base 2, coord 1 uses base 3, ...).
// Supports up to d = 20 dimensions (uses primes up to 71).
//
// The sequence index starts at 1 (index 0 produces the all-zero
// point, which is degenerate under the Phi^{-1} transformation).
class Halton {
public:
    // Construct a Halton generator in dimension d.
    // Throws std::invalid_argument if d > 20.
    explicit Halton(std::size_t d);

    // Fill an n-by-d row-major array `out` with the next n Halton
    // points. Each row is a d-dimensional point in [0, 1)^d.
    // Advances internal state by n.
    void generate(std::size_t n, double* out);

    // Reset state to start over.
    void reset() { index_ = 1; }

    std::size_t dimension() const { return d_; }

private:
    std::size_t d_;
    std::size_t index_;             // next index to produce (1-based)
    std::vector<int> primes_;       // primes p_1, ..., p_d
};


// =====================================================================
// Sobol sequence (from-scratch implementation, Joe-Kuo direction numbers)
// =====================================================================

// d-dimensional Sobol sequence using Joe-Kuo direction numbers.
//
// Implementation details:
//
//   - The constructor loads the direction-number file at the path
//     given by `data_file` (or, by default, the path baked into the
//     binary at compile time via SOBOL_DATA_FILE).
//   - Direction numbers are stored as 64-bit integers shifted left
//     by (64 - bit), so points can be recovered as int / 2^64.
//   - We use the Gray-code optimisation: u_{i+1} differs from u_i
//     by XOR with the direction number corresponding to the rightmost
//     zero bit of i.
//
// The implementation supports up to d = 20 dimensions (we cap here
// since (a) higher dimensions are beyond the Block 3 scope, and (b)
// keeps memory bounded). The cap can be lifted by changing the
// internal MAX_DIM constant.
class Sobol {
public:
    // Construct a Sobol generator in dimension d.
    // Loads direction numbers from the file at `data_file` (default:
    // the SOBOL_DATA_FILE macro, set in CMakeLists.txt).
    // Throws std::runtime_error if the file cannot be opened or if d
    // exceeds the file's available dimensions.
    // Throws std::invalid_argument if d == 0 or d > 20.
    explicit Sobol(std::size_t d,
                   const std::string& data_file = SOBOL_DATA_FILE);

    // Fill an n-by-d row-major array `out` with the next n Sobol
    // points. Index 0 is the all-zero point; we skip it (start at
    // index 1) for compatibility with Phi^{-1}.
    void generate(std::size_t n, double* out);

    // Reset state to start over (next call to generate starts at index 1).
    void reset();

    std::size_t dimension() const { return d_; }

private:
    static constexpr std::size_t MAX_DIM = 20;

    void load_direction_numbers(const std::string& data_file);

    std::size_t d_;
    std::size_t index_;             // next index to produce (1-based)
    std::vector<std::uint64_t> x_;  // current state, one per dimension
    // Direction numbers: V_[j][k] is the k-th direction number for
    // dimension j, scaled to occupy the high 64 bits.
    std::vector<std::vector<std::uint64_t>> V_;
};


// =====================================================================
// Digital shift (for RQMC)
// =====================================================================

// Apply a digital shift to a QMC point set in place.
//
// For each entry u_{i,j} and shift component xi_j (both in [0, 1)),
// convert to 53-bit integers (the mantissa precision of float64),
// XOR, and convert back. This is the standard digital shift in
// floating-point arithmetic.
//
// Both arrays are row-major n-by-d. shift has length d.
void digital_shift(std::size_t n, std::size_t d,
                   const double* shift,
                   double* points);


// =====================================================================
// Deterministic QMC pricer
// =====================================================================

// Deterministic QMC pricer for the European call using Euler
// discretisation. Generates a single QMC point set in dimension
// n_steps, inverts to normals via Phi^{-1}, runs the Euler scheme,
// returns the average discounted payoff.
//
// Returns a single double. We deliberately do NOT return MCResult
// because the deterministic estimator has no half-width: the
// Koksma-Hlawka bound requires V(f), which is intractable for option
// payoffs. The user is forced to acknowledge the lack of an error
// bar; for proper error estimation, use the RQMC pricer below.
//
// `sequence` is one of "halton" or "sobol".
//
// Throws std::invalid_argument on bad inputs (S/K/sigma/T <= 0,
// n_paths < 2, n_steps < 1, unknown sequence string).
double mc_european_call_euler_qmc(double S, double K, double r,
                                  double sigma, double T,
                                  std::size_t n_paths,
                                  std::size_t n_steps,
                                  const std::string& sequence);


// =====================================================================
// RQMC pricer with digital shift
// =====================================================================

// Randomized QMC pricer for the European call using Sobol with R
// independent digital shifts.
//
// Procedure:
//   1. Generate a single deterministic Sobol point set of size n_paths.
//   2. For each of n_replications:
//      a. Draw a uniform shift in [0, 1)^{n_steps}.
//      b. XOR the Sobol points with the shift.
//      c. Run the Euler pricer on the shifted points.
//   3. The R replication estimates are i.i.d.; their sample mean and
//      sample variance give the RQMC estimate and half-width.
//
// Returns an MCResult where:
//   estimate         = mean of R replication estimates
//   half_width       = z * sigma_rep / sqrt(R)
//   sample_variance  = sample variance of R replication estimates
//   n_paths          = R (the number of i.i.d. units)
//
// Note: the n_paths field stores R, not n_paths * n_replications.
// This convention matches the AV pricer of Block 2.1, where n_paths
// is the number of i.i.d. units (pairs in AV, replications in RQMC).
//
// rng must outlive the call. Throws std::invalid_argument on bad inputs.
MCResult
mc_european_call_euler_rqmc(double S, double K, double r,
                            double sigma, double T,
                            std::size_t n_paths,
                            std::size_t n_steps,
                            std::size_t n_replications,
                            std::mt19937_64& rng,
                            double confidence_level = 0.95);

}  // namespace quant
