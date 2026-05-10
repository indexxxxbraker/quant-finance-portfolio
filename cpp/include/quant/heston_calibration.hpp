// heston_calibration.hpp
//
// Heston model: calibration to market data via Levenberg-Marquardt.
//
// Mirrors quantlib.heston_calibration on the Python side. The Fourier
// pricer of Block 2 is the engine inside the optimisation loop. The
// optimiser is a damped Gauss-Newton (Levenberg's variant) implemented
// to avoid external dependencies.
//
// Bounds are enforced via internal variable transformation rather than
// constrained optimisation:
//   z = (log kappa, log theta, log sigma, atanh rho, log v0)
// lives in R^5 with no constraints. This is simpler than constrained
// LM and just as accurate for typical Heston regimes.
//
// See theory/phase4/block6_heston_calibration_exotics.tex for the
// mathematical formulation, choice of objective function, and
// discussion of identifiability.

#pragma once

#include "heston_fourier.hpp"   // HestonParams

#include <cstddef>
#include <vector>

namespace quant::heston {

// =====================================================================
// Public types
// =====================================================================

// One observation in the calibration surface.
struct CalibrationQuote {
    double K;          // strike
    double T;          // maturity
    double C_market;   // observed call price
};


// Result of a calibration run.
struct CalibrationResult {
    HestonParams params;        // calibrated parameters
    std::vector<double> residuals;  // per-quote residuals (model - market)
    double rmse;                // root-mean-square of residuals
    int n_iter;                 // number of optimiser iterations
    bool success;               // converged within tolerance
};


// =====================================================================
// Public interface
// =====================================================================

// Compute Black-Scholes implied volatility from a call price.
//
// Uses Newton-Raphson with bisection fallback. Returns NaN if the price
// is outside no-arbitrage bounds [max(S0 - K*exp(-rT), 0), S0].
double implied_vol_bs(double C_target,
                       double K, double T,
                       double S0, double r,
                       double sigma0 = 0.2,
                       int max_iter = 50,
                       double tol = 1e-8);


// Calibrate Heston parameters to a market call surface.
//
// Parameters:
//   market           vector of (K, T, C_market) quotes; size >= 5
//   S0, r            spot and risk-free rate
//   initial_guess    starting parameters (must be inside the bounds:
//                    kappa, theta, sigma, v0 > 0, |rho| < 1)
//   weighted         if true, use 1/vega weighting for residuals
//   max_iter         maximum LM iterations; default 200
//
// Returns: CalibrationResult with fitted params, residuals, RMSE.
//
// Throws std::invalid_argument on bad inputs.
CalibrationResult
calibrate_heston(const std::vector<CalibrationQuote>& market,
                   double S0, double r,
                   const HestonParams& initial_guess,
                   bool weighted = true,
                   int max_iter = 200);

}  // namespace quant::heston
