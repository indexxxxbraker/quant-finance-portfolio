// monte_carlo.hpp
//
// Statistical reducer and high-level Monte Carlo pricers.
//
// Public interface:
//
//   - struct MCResult: bundle of the standard outputs of any Monte
//     Carlo run (point estimate, asymptotic CI half-width, sample
//     variance, sample size).
//
//   - mc_estimator: model- and payoff-agnostic statistical reducer
//     that turns a vector of i.i.d. payoffs into an MCResult.
//
//   - mc_european_call_exact: pricer using the exact GBM sampler
//     (Block 1.1).
//
//   - mc_european_call_euler: pricer using Euler-Maruyama
//     discretisation (Block 1.2.1).
//
//   - mc_european_call_milstein: pricer using Milstein
//     discretisation (Block 1.2.2).
//
// The samplers themselves live in gbm.hpp. This split mirrors the
// Python module layout (quantlib.gbm + quantlib.monte_carlo) and
// reflects the architectural separation between
// model-and-scheme-specific work (sampling) and model-agnostic work
// (statistical reduction, payoff orchestration).
//
// References:
//   Phase 2 Block 0 writeup (foundations); Block 1.1 writeup (exact);
//   Block 1.2.0 writeup (SDE discretisation theory); Block 1.2.1
//   writeup (Euler); Block 1.2.2 writeup (Milstein).

#pragma once

#include <cstddef>
#include <random>
#include <vector>

namespace quant {

// =====================================================================
// Public types
// =====================================================================

struct MCResult {
    double      estimate;
    double      half_width;
    double      sample_variance;
    std::size_t n_paths;
};


// =====================================================================
// Generic statistical estimator
// =====================================================================

// Reduce a vector of i.i.d. payoff samples to a Monte Carlo result.
//
// Computes the sample mean, the sample variance with Bessel's
// correction, and the asymptotic Gaussian confidence interval
// half-width based on the central limit theorem and Slutsky's lemma.
//
// Throws std::invalid_argument if Y has fewer than 2 elements or if
// confidence_level is not in (0, 1).
MCResult
mc_estimator(const std::vector<double>& Y,
             double confidence_level = 0.95);


// =====================================================================
// High-level pricers
// =====================================================================

// Price a European call by Monte Carlo with exact GBM simulation.
// rng must outlive the call. Throws std::invalid_argument on
// parameter validation failure.
MCResult
mc_european_call_exact(double S, double K, double r, double sigma, double T,
                       std::size_t n_paths,
                       std::mt19937_64& rng,
                       double confidence_level = 0.95);


// Price a European call by Monte Carlo with Euler-Maruyama paths.
//
// Carries a discretisation bias of order T / n_steps: the estimator
// converges to the BS price at weak rate 1 as n_steps grows.
MCResult
mc_european_call_euler(double S, double K, double r, double sigma, double T,
                       std::size_t n_steps, std::size_t n_paths,
                       std::mt19937_64& rng,
                       double confidence_level = 0.95);


// Price a European call by Monte Carlo with Milstein paths.
//
// Both Milstein and Euler have weak order 1 for European pricing,
// so this estimator is statistically indistinguishable from
// mc_european_call_euler at any practical sample size. Exposed for
// completeness and for the empirical convergence study.
MCResult
mc_european_call_milstein(double S, double K, double r, double sigma, double T,
                          std::size_t n_steps, std::size_t n_paths,
                          std::mt19937_64& rng,
                          double confidence_level = 0.95);

}  // namespace quant
