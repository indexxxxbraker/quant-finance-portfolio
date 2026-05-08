// heston_fourier.hpp
//
// Heston model: characteristic function and Lewis-quadrature pricing.
//
// Mirrors the design of quantlib.heston_fourier on the Python side, with
// two notable differences:
//
//   1. Parameters are bundled in a HestonParams struct, since C++ benefits
//      strongly from typed parameter aggregation (and call sites become
//      far more readable than long parameter lists).
//   2. Carr-Madan FFT pricing is NOT implemented in C++. The role of C++
//      in this project is high-precision single-strike pricing and
//      deterministic unit testing; FFT-driven surface calibration lives
//      on the Python side. This avoids a dependency on FFTW / pocketfft.
//
// The characteristic function uses the Albrecher-Mayer-Schoutens-Tistaert
// (2007) "Little Trap" formulation; see theory/phase4/block2_heston_fourier.tex
// for the derivation.

#pragma once

#include <complex>
#include <utility>

namespace quant::heston {

// =====================================================================
// Heston model parameters
// =====================================================================

// Parameter bundle for the Heston model under Q:
//
//     dS_t = r S_t dt + sqrt(v_t) S_t dW1_t
//     dv_t = kappa (theta - v_t) dt + sigma sqrt(v_t) dW2_t
//     d<W1, W2>_t = rho dt
//
struct HestonParams {
    double kappa;   // mean reversion speed of variance (>0)
    double theta;   // long-run variance (>0)
    double sigma;   // vol of vol (>0)
    double rho;     // correlation in [-1, 1]
    double v0;      // initial variance (>=0)

    // Throw std::invalid_argument if parameters are out of range.
    void validate() const;
};


// =====================================================================
// Characteristic function (AMSST / "Little Trap" formulation)
// =====================================================================

// Compute the (C, D) coefficients of the Heston characteristic function
// such that
//
//     phi(u; tau, X_t, v_t) = exp(C(tau; u) + D(tau; u) v_t + i u X_t).
//
// The Fourier variable u may be real or complex.
std::pair<std::complex<double>, std::complex<double>>
heston_cf_coefficients(std::complex<double> u,
                        double tau,
                        double r,
                        const HestonParams& p);

// Heston characteristic function of log(S_T):
//
//     phi(u; tau) = E[ exp(i u log S_T) | S_t = S0, v_t = v0 ]
//                 = exp(C + D v0 + i u log S0).
//
// Sanity checks:
//     phi(0;  ...) = 1
//     phi(-i; ...) = S0 * exp(r * tau)         (= E[S_T] under Q)
std::complex<double>
heston_cf(std::complex<double> u,
          double tau,
          double S0,
          double r,
          const HestonParams& p);


// =====================================================================
// European call: Lewis adaptive quadrature
// =====================================================================

// Heston European call price via Lewis (2000) inversion:
//
//     C(K, T) = S0 - sqrt(K S0) exp(-r tau) / pi
//               * integral_0^infty Re[ exp(i u log(S0/K))
//                                      * phi_norm(u - i/2; tau)
//                                      / (u^2 + 1/4) ] du,
//
// with phi_norm(u; tau) = exp(C(tau; u) + D(tau; u) v0). The integral is
// evaluated on [0, u_max] via composite 32-point Gauss-Legendre split at
// u = 30 to resolve the integrand's structure near zero.
//
// u_max defaults to 100, which is far past the support of the integrand
// for typical equity calibrations. Increase for very high vol-of-vol or
// extremely short maturities.
double heston_call_lewis(double K,
                          double tau,
                          double S0,
                          double r,
                          const HestonParams& p,
                          double u_max = 100.0);


// =====================================================================
// Reference: Black-Scholes and put-call parity
// =====================================================================

// Black-Scholes European call price.
//
// In the Heston BS limit (sigma -> 0, v0 = theta), the Heston call price
// converges to BS at vol sqrt(v0). Used as a sanity benchmark.
double black_scholes_call(double S0,
                           double K,
                           double tau,
                           double r,
                           double sigma);

// European put via put-call parity:
//
//     P = C - S0 + K * exp(-r * tau).
//
// Model-free; useful for converting call quotes to put quotes.
double put_via_parity(double call_price,
                      double S0,
                      double K,
                      double tau,
                      double r);

}  // namespace quant::heston
