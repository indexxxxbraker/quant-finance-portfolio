// greeks.hpp
//
// Greek estimators for the European call by Monte Carlo.
//
// Three techniques are implemented for Delta and Vega:
//   - Bumping: finite differences with common random numbers.
//   - Pathwise: differentiation of the payoff under the integral.
//   - Likelihood ratio: differentiation of the kernel of the density.
//
// For Gamma, only bumping (central second-order finite differences) is
// implemented. Pathwise fails because the second derivative of
// max(.,0) produces a Dirac contribution that Monte Carlo cannot
// estimate; the LR method has very high variance for Gamma.
//
// All estimators return an MCResult with the same conventions as
// monte_carlo.hpp pricers (estimate, half_width, sample_variance,
// n_paths). The n_paths field stores the actual number of normals
// drawn; for bumping methods this is one Z per path even though the
// pricer is evaluated 2 or 3 times (CRN reuses the same Z across
// evaluations).
//
// References:
//   Phase 2 Block 4 writeup. Broadie and Glasserman, "Estimating
//   security price derivatives using simulation", Management Science
//   42(2), 269-285, 1996. Glasserman, Chapter 7.

#pragma once

#include "monte_carlo.hpp"   // for MCResult

#include <cstddef>
#include <random>

namespace quant {

// =====================================================================
// Bumping (common-random-numbers central differences)
// =====================================================================

// Delta of the European call by central finite differences with CRN.
// The bump is multiplicative: S0 -> S0 * (1 +/- h).
//
// rng must outlive the call. Throws std::invalid_argument on bad inputs.
MCResult
delta_bump(double S, double K, double r, double sigma, double T,
           std::size_t n_paths,
           std::mt19937_64& rng,
           double h = 1e-2,
           double confidence_level = 0.95);


// Vega of the European call by central finite differences with CRN.
// The bump is additive in absolute volatility units: sigma -> sigma +/- h.
MCResult
vega_bump(double S, double K, double r, double sigma, double T,
          std::size_t n_paths,
          std::mt19937_64& rng,
          double h = 1e-2,
          double confidence_level = 0.95);


// Gamma of the European call by central second-order finite differences
// with CRN. Three pricer evaluations: at S*(1+h), S, S*(1-h).
//
// Note: Gamma's variance scales like 1/h^2 even with CRN. The
// half-width is intrinsically wider than for Delta/Vega.
MCResult
gamma_bump(double S, double K, double r, double sigma, double T,
           std::size_t n_paths,
           std::mt19937_64& rng,
           double h = 1e-2,
           double confidence_level = 0.95);


// =====================================================================
// Pathwise sensitivities
// =====================================================================

// Delta of the European call by pathwise differentiation.
//
// Estimator: e^{-rT} * 1_{S_T > K} * S_T / S0.
//
// Unbiased and typically lower variance than LR. Requires the payoff
// to be Lipschitz, which the call is (Lipschitz constant 1).
MCResult
delta_pathwise(double S, double K, double r, double sigma, double T,
               std::size_t n_paths,
               std::mt19937_64& rng,
               double confidence_level = 0.95);


// Vega of the European call by pathwise differentiation.
//
// Estimator: e^{-rT} * 1_{S_T > K} * S_T * (sqrt(T)*Z - sigma*T).
MCResult
vega_pathwise(double S, double K, double r, double sigma, double T,
              std::size_t n_paths,
              std::mt19937_64& rng,
              double confidence_level = 0.95);


// =====================================================================
// Likelihood ratio
// =====================================================================

// Delta of the European call by likelihood ratio.
//
// Estimator: e^{-rT} * max(S_T - K, 0) * Z / (S0 * sigma * sqrt(T)).
//
// Score derivation: log S_T ~ N(log S0 + (r - sigma^2/2) T,
// sigma^2 T). The score d/dS0 log p evaluated at the path gives the
// factor Z / (S0 * sigma * sqrt(T)).
MCResult
delta_lr(double S, double K, double r, double sigma, double T,
         std::size_t n_paths,
         std::mt19937_64& rng,
         double confidence_level = 0.95);


// Vega of the European call by likelihood ratio.
//
// Estimator: e^{-rT} * max(S_T - K, 0) * ((Z^2 - 1)/sigma - Z*sqrt(T)).
MCResult
vega_lr(double S, double K, double r, double sigma, double T,
        std::size_t n_paths,
        std::mt19937_64& rng,
        double confidence_level = 0.95);

}  // namespace quant
