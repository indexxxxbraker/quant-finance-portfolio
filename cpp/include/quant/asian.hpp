// asian.hpp
//
// Asian option pricers by Monte Carlo for the discretely-monitored
// arithmetic Asian call:
//
//     payoff = max( (1/N) * sum_{k=1}^N S_{t_k} - K, 0 )
//
// monitored on an equispaced grid t_k = k * T / N (k = 1, ..., N).
// S_0 is the known spot and is NOT included in the average.
//
// Three pricers are provided:
//
//   - mc_asian_call_arithmetic_iid: standard MC baseline.
//   - mc_asian_call_geometric_iid : same scheme but with the geometric
//     average (used to validate the closed form below).
//   - mc_asian_call_arithmetic_cv : arithmetic estimator with the
//     geometric Asian as control variate. Typical VRF > 1000 at our
//     canonical ATM parameters because the empirical correlation
//     between arithmetic and geometric per-path payoffs exceeds 0.999.
//
// Plus a closed-form geometric Asian call:
//
//   - geometric_asian_call_closed_form: the geometric average is
//     lognormal under GBM, so this is a Black-Scholes-like formula
//     with effective volatility sigma_eff and effective rate r_eff.
//     See theory/phase2/mc_asian.tex for the full derivation.
//
// The path simulator uses exact GBM stepping (no Euler), which avoids
// time-discretisation bias that would compound across the N
// observation dates.
//
// References:
//   Phase 2 Block 5 writeup. Kemna and Vorst (1990), J. Banking and
//   Finance, 14(1):113-129. Glasserman, Section 4.5.

#pragma once

#include "monte_carlo.hpp"   // for MCResult

#include <cstddef>
#include <random>

namespace quant {

// =====================================================================
// Closed form for the discretely-monitored geometric Asian call
// =====================================================================

// Closed-form price of the discretely-monitored geometric Asian call
// with N = n_steps observations on an equispaced grid in (0, T].
//
// The geometric average S_geom = (prod_k S_{t_k})^{1/N} is lognormal
// under risk-neutral GBM. The price has a Black-Scholes-like form
// with an effective volatility and effective rate:
//
//   sigma_eff = sigma * sqrt((N+1)(2N+1) / (6 N^2))
//   r_eff     = (N+1)/(2N) * (r - sigma^2/2) + sigma_eff^2 / 2
//
// then C_geom = e^{-rT} * [S * exp(r_eff * T) * Phi(d1) - K * Phi(d2)],
// where d1 = (log(S/K) + (r_eff + sigma_eff^2/2) T) / (sigma_eff*sqrt T),
// and d2 = d1 - sigma_eff*sqrt(T).
//
// Sanity checks: at N -> infinity, sigma_eff -> sigma/sqrt(3) (the
// classical continuous-monitoring result); at N = 1 the formula
// reduces to the vanilla Black-Scholes call.
//
// Throws std::invalid_argument on invalid inputs.
double
geometric_asian_call_closed_form(double S, double K, double r,
                                 double sigma, double T,
                                 std::size_t n_steps);


// =====================================================================
// IID pricers
// =====================================================================

// Price the arithmetic Asian call by IID Monte Carlo using exact GBM
// stepping for the path simulator.
//
// rng must outlive the call. Throws std::invalid_argument on bad input.
MCResult
mc_asian_call_arithmetic_iid(double S, double K, double r,
                             double sigma, double T,
                             std::size_t n_paths,
                             std::size_t n_steps,
                             std::mt19937_64& rng,
                             double confidence_level = 0.95);


// Price the geometric Asian call by IID Monte Carlo. Used to validate
// the closed form, not as a standalone pricer.
//
// Same path scheme as the arithmetic version (so they share the same
// rng consumption pattern).
MCResult
mc_asian_call_geometric_iid(double S, double K, double r,
                            double sigma, double T,
                            std::size_t n_paths,
                            std::size_t n_steps,
                            std::mt19937_64& rng,
                            double confidence_level = 0.95);


// =====================================================================
// Arithmetic Asian with geometric control variate
// =====================================================================

// Price the arithmetic Asian call with the geometric Asian as control
// variate. Both estimators are computed on the same paths (CRN); the
// control's known mean is the closed-form geometric_asian_call_closed_form.
//
// Returns an MCResult whose half-width is computed from the variance
// of (X - c_hat * (Y - EY)), i.e. (1 - rho^2) * Var(X). Typical VRF
// > 1000 at our canonical parameters.
MCResult
mc_asian_call_arithmetic_cv(double S, double K, double r,
                            double sigma, double T,
                            std::size_t n_paths,
                            std::size_t n_steps,
                            std::mt19937_64& rng,
                            double confidence_level = 0.95);

}  // namespace quant
