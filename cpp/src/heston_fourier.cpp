// heston_fourier.cpp
//
// See heston_fourier.hpp for design notes and theory/phase4/block2_heston_fourier.tex
// for the mathematical derivation.

#include "heston_fourier.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

namespace quant::heston {

// =====================================================================
// Internal helpers (anonymous namespace)
// =====================================================================
namespace {

// pi -- portable constant rather than std::numbers::pi (C++20).
constexpr double PI = 3.14159265358979323846;

// 32-point Gauss-Legendre nodes on [-1, 1]
constexpr std::array<double, 32> GL32_NODES = {{
    -9.972638618494815699e-01, -9.856115115452683817e-01,
    -9.647622555875063899e-01, -9.349060759377396668e-01,
    -8.963211557660520912e-01, -8.493676137325699704e-01,
    -7.944837959679423856e-01, -7.321821187402897113e-01,
    -6.630442669302152314e-01, -5.877157572407623043e-01,
    -5.068999089322293594e-01, -4.213512761306353327e-01,
    -3.318686022821276671e-01, -2.392873622521370647e-01,
    -1.444719615827964876e-01, -4.830766568773832426e-02,
    +4.830766568773832426e-02, +1.444719615827964876e-01,
    +2.392873622521370647e-01, +3.318686022821276671e-01,
    +4.213512761306353327e-01, +5.068999089322293594e-01,
    +5.877157572407623043e-01, +6.630442669302152314e-01,
    +7.321821187402897113e-01, +7.944837959679423856e-01,
    +8.493676137325699704e-01, +8.963211557660520912e-01,
    +9.349060759377396668e-01, +9.647622555875063899e-01,
    +9.856115115452683817e-01, +9.972638618494815699e-01,
}};

// 32-point Gauss-Legendre weights
constexpr std::array<double, 32> GL32_WEIGHTS = {{
    +7.018610009470505756e-03, +1.627439473090574323e-02,
    +2.539206530926202410e-02, +3.427386291302176452e-02,
    +4.283589802222683568e-02, +5.099805926237609144e-02,
    +5.868409347853556501e-02, +6.582222277636168295e-02,
    +7.234579410884833806e-02, +7.819389578707022781e-02,
    +8.331192422694670696e-02, +8.765209300440378326e-02,
    +9.117387869576377979e-02, +9.384439908080451087e-02,
    +9.563872007927470831e-02, +9.654008851472765940e-02,
    +9.654008851472765940e-02, +9.563872007927470831e-02,
    +9.384439908080451087e-02, +9.117387869576377979e-02,
    +8.765209300440378326e-02, +8.331192422694670696e-02,
    +7.819389578707022781e-02, +7.234579410884833806e-02,
    +6.582222277636168295e-02, +5.868409347853556501e-02,
    +5.099805926237609144e-02, +4.283589802222683568e-02,
    +3.427386291302176452e-02, +2.539206530926202410e-02,
    +1.627439473090574323e-02, +7.018610009470505756e-03,
}};

// Integrate f on [a, b] via 32-point Gauss-Legendre.
//
// Maps [-1, 1] linearly to [a, b]:  x_i = mid + half_width * node_i,
// integral approximated by  half_width * sum_i w_i * f(x_i).
//
// Exact for polynomials of degree <= 63 on [a, b]; for our smooth,
// analytic integrand this gives near-double precision.
template <typename F>
double integrate_gl32(F&& f, double a, double b) {
    const double mid = 0.5 * (a + b);
    const double half_width = 0.5 * (b - a);
    double sum = 0.0;
    for (std::size_t i = 0; i < 32; ++i) {
        const double x = mid + half_width * GL32_NODES[i];
        sum += GL32_WEIGHTS[i] * f(x);
    }
    return half_width * sum;
}

// Standard normal CDF via std::erf:
//     N(x) = 0.5 * (1 + erf(x / sqrt(2))).
double standard_normal_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

// Common pre-validation for pricing functions.
void check_basic(double tau, double S0, const HestonParams& p) {
    p.validate();
    if (tau <= 0.0) {
        throw std::invalid_argument("tau must be positive");
    }
    if (S0 <= 0.0) {
        throw std::invalid_argument("S0 must be positive");
    }
    // Note: r is unrestricted (negative rates are admissible).
}

}  // namespace


// =====================================================================
// HestonParams::validate
// =====================================================================

void HestonParams::validate() const {
    if (kappa <= 0.0) {
        throw std::invalid_argument("HestonParams: kappa must be positive");
    }
    if (theta <= 0.0) {
        throw std::invalid_argument("HestonParams: theta must be positive");
    }
    if (sigma <= 0.0) {
        throw std::invalid_argument("HestonParams: sigma must be positive");
    }
    if (rho < -1.0 || rho > 1.0) {
        throw std::invalid_argument("HestonParams: rho must be in [-1, 1]");
    }
    if (v0 < 0.0) {
        throw std::invalid_argument("HestonParams: v0 must be non-negative");
    }
}


// =====================================================================
// Characteristic function: AMSST formulation
// =====================================================================

std::pair<std::complex<double>, std::complex<double>>
heston_cf_coefficients(std::complex<double> u,
                        double tau,
                        double r,
                        const HestonParams& p) {
    using namespace std::complex_literals;

    // beta(u) = kappa - i rho sigma u
    const auto beta = p.kappa - 1i * p.rho * p.sigma * u;

    // d(u) = sqrt(beta^2 + sigma^2 u (u + i))
    // std::sqrt on std::complex returns the principal branch (Re d >= 0),
    // which is the correct branch choice for the AMSST formulation.
    const auto d = std::sqrt(beta * beta
                              + p.sigma * p.sigma * u * (u + 1i));

    // g(u) = (beta - d) / (beta + d)        [AMSST / "Little Trap"]
    const auto g = (beta - d) / (beta + d);

    const auto exp_neg_dtau = std::exp(-d * tau);
    const auto one_minus_g_exp = 1.0 - g * exp_neg_dtau;

    // D(tau; u) = (beta - d) / sigma^2 * (1 - exp(-d tau)) / (1 - g exp(-d tau))
    const auto D = (beta - d) / (p.sigma * p.sigma)
                   * (1.0 - exp_neg_dtau) / one_minus_g_exp;

    // C(tau; u) = i u r tau + (kappa theta / sigma^2)
    //              * [(beta - d) tau - 2 ln((1 - g exp(-d tau)) / (1 - g))]
    const auto log_term = std::log(one_minus_g_exp / (1.0 - g));
    const auto C = 1i * u * r * tau
                   + (p.kappa * p.theta / (p.sigma * p.sigma))
                     * ((beta - d) * tau - 2.0 * log_term);

    return {C, D};
}


std::complex<double>
heston_cf(std::complex<double> u,
          double tau,
          double S0,
          double r,
          const HestonParams& p) {
    using namespace std::complex_literals;
    check_basic(tau, S0, p);

    const auto [C, D] = heston_cf_coefficients(u, tau, r, p);
    return std::exp(C + D * p.v0 + 1i * u * std::log(S0));
}


// =====================================================================
// Lewis call price
// =====================================================================

double heston_call_lewis(double K,
                          double tau,
                          double S0,
                          double r,
                          const HestonParams& p,
                          double u_max) {
    using namespace std::complex_literals;
    check_basic(tau, S0, p);

    if (K <= 0.0) {
        throw std::invalid_argument("K must be positive");
    }
    if (u_max <= 0.0) {
        throw std::invalid_argument("u_max must be positive");
    }

    const double log_S_over_K = std::log(S0 / K);
    const double v0 = p.v0;

    // Lewis integrand (real-valued, smooth on [0, infty)):
    //
    //     Re[ exp(i u log(S0/K)) * phi_norm(u - i/2; tau) / (u^2 + 1/4) ].
    //
    // The variable u in the lambda corresponds to the real part of the
    // shifted Fourier variable; the imaginary shift -i/2 lives in the
    // argument of phi_norm.
    auto integrand = [&](double u) -> double {
        const std::complex<double> u_shifted{u, -0.5};
        const auto [C, D] = heston_cf_coefficients(u_shifted, tau, r, p);
        const auto phi_norm = std::exp(C + D * v0);
        const auto val = std::exp(1i * u * log_S_over_K) * phi_norm
                         / (u * u + 0.25);
        return val.real();
    };

    // Composite Gauss-Legendre on three panels [0, 5], [5, 30], [30, u_max].
    // The Lewis integrand decays as 1/u^2 modulated by a fast Gaussian-like
    // factor coming from the characteristic function. Most of its mass sits
    // in [0, 5] (where 1/(u^2 + 1/4) is largest), with a slow tail beyond.
    // Three asymmetric panels resolve the peak without wasting nodes on
    // the deep tail. If u_max is small enough that the splits collapse,
    // we fall back to a single-panel evaluation on [0, u_max].
    double I = 0.0;
    if (u_max <= 5.0) {
        I = integrate_gl32(integrand, 0.0, u_max);
    } else if (u_max <= 30.0) {
        I = integrate_gl32(integrand, 0.0, 5.0)
          + integrate_gl32(integrand, 5.0, u_max);
    } else {
        I = integrate_gl32(integrand, 0.0, 5.0)
          + integrate_gl32(integrand, 5.0, 30.0)
          + integrate_gl32(integrand, 30.0, u_max);
    }

    return S0 - std::sqrt(S0 * K) * std::exp(-r * tau) / PI * I;
}


// =====================================================================
// Reference: Black-Scholes and put-call parity
// =====================================================================

double black_scholes_call(double S0,
                           double K,
                           double tau,
                           double r,
                           double sigma) {
    if (tau <= 0.0) {
        return std::max(S0 - K, 0.0);
    }
    if (sigma <= 0.0) {
        return std::max(S0 - K * std::exp(-r * tau), 0.0);
    }

    const double sqrt_tau = std::sqrt(tau);
    const double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * tau)
                      / (sigma * sqrt_tau);
    const double d2 = d1 - sigma * sqrt_tau;

    return S0 * standard_normal_cdf(d1)
           - K * std::exp(-r * tau) * standard_normal_cdf(d2);
}


double put_via_parity(double call_price,
                      double S0,
                      double K,
                      double tau,
                      double r) {
    return call_price - S0 + K * std::exp(-r * tau);
}

}  // namespace quant::heston
