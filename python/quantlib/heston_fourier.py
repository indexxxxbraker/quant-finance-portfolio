"""
Heston model: characteristic function and Fourier-based pricing.

This module implements the Heston (1993) stochastic-volatility model under
the risk-neutral measure, with two pricing routes:

    1. Carr-Madan FFT (production method, used for surface pricing).
    2. Lewis adaptive quadrature (validation / single-strike high-precision).

The characteristic function uses the Albrecher-Mayer-Schoutens-Tistaert (2007)
"Little Trap" formulation, which avoids the branch-cut issue of the original
Heston (1993) formulation. See ``theory/phase4/block2_heston_fourier.tex``
for the derivation and discussion.

Notation
--------
The Heston dynamics under Q are

    dS_t = r S_t dt + sqrt(v_t) S_t dW1_t
    dv_t = kappa (theta - v_t) dt + sigma sqrt(v_t) dW2_t
    d<W1, W2>_t = rho dt

with parameters (kappa, theta, sigma, rho, v0) and risk-free rate r.
All pricing functions take (tau, S0, v0, r, kappa, theta, sigma, rho)
in that order.

References
----------
[Heston1993]  Heston, S. (1993). A closed-form solution for options with
              stochastic volatility. Review of Financial Studies 6(2).
[AMSST2007]   Albrecher, Mayer, Schoutens, Tistaert (2007). The little
              Heston trap. Wilmott Magazine.
[CarrMadan]   Carr, P., Madan, D. (1999). Option valuation using the FFT.
              Journal of Computational Finance 2(4).
[Lewis2000]   Lewis, A. (2000). Option Valuation under Stochastic Volatility.
              Finance Press.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


# =====================================================================
# Input validation
# =====================================================================

def _check_heston_params(tau, S0, v0, kappa, theta, sigma, rho):
    """Validate Heston model parameters. Raises ValueError on failure."""
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    if S0 <= 0:
        raise ValueError(f"S0 must be positive, got {S0}")
    if v0 < 0:
        raise ValueError(f"v0 must be non-negative, got {v0}")
    if kappa <= 0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    if theta <= 0:
        raise ValueError(f"theta must be positive, got {theta}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if not (-1.0 <= rho <= 1.0):
        raise ValueError(f"rho must be in [-1, 1], got {rho}")


# =====================================================================
# Heston characteristic function (AMSST formulation)
# =====================================================================

def _heston_cf_coefficients(u, tau, r, kappa, theta, sigma, rho):
    """
    Compute the (C, D) coefficients of the Heston characteristic function.

    The characteristic function of X_T = log(S_T) under the risk-neutral
    measure, given (X_t = log S_t, v_t), takes the affine form

        phi(u; tau, X_t, v_t) = exp(C(tau; u) + D(tau; u) v_t + i u X_t).

    This helper returns the (C, D) pair so that callers can either build
    the full characteristic function (heston_cf below) or the normalized
    version exp(C + D v_t) used in Lewis's formula.
    """
    u = np.asarray(u, dtype=complex)

    beta = kappa - 1j * rho * sigma * u
    # Discriminant: beta^2 + sigma^2 (u^2 + i u) = beta^2 + sigma^2 u (u + i)
    d = np.sqrt(beta * beta + sigma * sigma * u * (u + 1j))

    g = (beta - d) / (beta + d)
    exp_neg_dtau = np.exp(-d * tau)

    one_minus_g_exp = 1.0 - g * exp_neg_dtau
    D = (beta - d) / (sigma * sigma) * (1.0 - exp_neg_dtau) / one_minus_g_exp

    log_term = np.log(one_minus_g_exp / (1.0 - g))
    C = (1j * u * r * tau
         + (kappa * theta / (sigma * sigma)) * ((beta - d) * tau - 2.0 * log_term))

    return C, D


def heston_cf(u, tau, S0, v0, r, kappa, theta, sigma, rho):
    """
    Heston characteristic function of log(S_T).

    Returns

        phi(u; tau, S0, v0) = E[ exp(i u log S_T) | S_t = S0, v_t = v0 ]
                            = exp(C + D v0 + i u log S0).

    Sanity checks satisfied by this formula (used as smoke tests):
        phi(0;  tau, ...) = 1
        phi(-i; tau, ...) = S0 * exp(r tau)            (=  E[S_T] under Q)
    """
    _check_heston_params(tau, S0, v0, kappa, theta, sigma, rho)

    C, D = _heston_cf_coefficients(u, tau, r, kappa, theta, sigma, rho)
    u_arr = np.asarray(u, dtype=complex)
    return np.exp(C + D * v0 + 1j * u_arr * np.log(S0))


# =====================================================================
# Carr-Madan FFT pricing
# =====================================================================

def heston_call_carr_madan(K, tau, S0, v0, r, kappa, theta, sigma, rho,
                            alpha=1.5, N=4096, eta=0.25):
    """
    Heston European call price(s) via Carr-Madan FFT.

    Following Carr & Madan (1999), the damped call price
    c(k) = exp(alpha k) C(k) is integrable for alpha large enough; its
    Fourier transform is

        psi(v) = exp(-r tau) phi(v - (alpha+1) i; tau)
                 / [(alpha + i v)(alpha + 1 + i v)],

    and the call price is recovered by

        C(k) = exp(-alpha k)/pi * integral_0^infty Re[exp(-i v k) psi(v)] dv.

    The FFT setup uses the Carr-Madan grid k_m = -b + m * lambda for
    m = 0, ..., N-1, with lambda = 2 pi / (N eta) and b = N lambda / 2.
    Strikes are interpolated linearly in log-strike to the requested K.
    """
    _check_heston_params(tau, S0, v0, kappa, theta, sigma, rho)

    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if N <= 0 or (N & (N - 1)):
        raise ValueError(f"N must be a positive power of 2, got {N}")
    if eta <= 0:
        raise ValueError(f"eta must be positive, got {eta}")

    K_array = np.atleast_1d(np.asarray(K, dtype=float))
    if np.any(K_array <= 0):
        raise ValueError("All strikes must be positive")

    # Carr-Madan natural strike grid (symmetric around log K = 0)
    lambd = 2.0 * np.pi / (N * eta)
    b = 0.5 * N * lambd
    k_grid = -b + np.arange(N) * lambd

    # Integration grid in v
    v = np.arange(N) * eta

    # Carr-Madan transform of the call: psi(v)
    u_complex = v - (alpha + 1.0) * 1j
    phi_at_u = heston_cf(u_complex, tau, S0, v0, r, kappa, theta, sigma, rho)
    denom = alpha * alpha + alpha - v * v + 1j * (2.0 * alpha + 1.0) * v
    psi = np.exp(-r * tau) * phi_at_u / denom

    # Simpson 1/3 weights, 0-indexed:
    #   w_0 = 1/3, w_n = 4/3 for n odd, w_n = 2/3 for n even (n >= 2).
    n_idx = np.arange(N)
    weights = np.where(n_idx == 0, 1.0 / 3.0,
                       (3.0 - (-1.0) ** n_idx) / 3.0)

    # FFT input: integration step, phase factor and weights folded in
    x = eta * np.exp(1j * v * b) * psi * weights
    fft_out = np.fft.fft(x)

    # Recover call prices on the natural grid k_m
    C_grid = np.real(np.exp(-alpha * k_grid) * fft_out / np.pi)

    # Linear interpolation in log-strike to the requested K values
    log_K_query = np.log(K_array)
    if np.any(log_K_query < k_grid[0]) or np.any(log_K_query > k_grid[-1]):
        raise ValueError(
            "Some requested strikes fall outside the FFT grid; "
            "increase N or adjust eta."
        )
    C_query = np.interp(log_K_query, k_grid, C_grid)

    if np.ndim(K) == 0:
        return float(C_query[0])
    return C_query


# =====================================================================
# Lewis adaptive quadrature pricing
# =====================================================================

def heston_call_lewis(K, tau, S0, v0, r, kappa, theta, sigma, rho,
                      u_max=200.0, quad_kwargs=None):
    """
    Heston European call price(s) via Lewis adaptive quadrature.

    Uses Lewis's (2000) inversion formula in the form

        C(K, T) = S0 - sqrt(K S0) exp(-r tau) / pi
                  * integral_0^infty Re[ exp(i u log(S0/K))
                                         * phi_norm(u - i/2; tau)
                                         / (u^2 + 1/4) ] du,

    where phi_norm(u; tau) = exp(C(tau; u) + D(tau; u) v_0) is the Heston
    characteristic function with X_t = 0 (i.e., S_t = 1). This is
    equivalent to the standard Lewis formula but expressed in terms of
    phi_norm rather than the centered char function of log(F_T/F_0); see
    block2_heston_fourier.tex for the derivation.

    For single-strike pricing this typically beats Carr-Madan in both
    accuracy and speed; for many strikes, Carr-Madan FFT wins.
    """
    _check_heston_params(tau, S0, v0, kappa, theta, sigma, rho)

    if u_max <= 0:
        raise ValueError(f"u_max must be positive, got {u_max}")

    if quad_kwargs is None:
        quad_kwargs = {}

    K_array = np.atleast_1d(np.asarray(K, dtype=float))
    if np.any(K_array <= 0):
        raise ValueError("All strikes must be positive")

    prices = np.empty(len(K_array), dtype=float)
    log_S0 = np.log(S0)

    for i, Ki in enumerate(K_array):
        log_S_over_K = log_S0 - np.log(Ki)

        def integrand(u):
            C_co, D_co = _heston_cf_coefficients(
                u - 0.5j, tau, r, kappa, theta, sigma, rho
            )
            phi_norm = np.exp(C_co + D_co * v0)
            integrand_complex = (np.exp(1j * u * log_S_over_K) * phi_norm
                                 / (u * u + 0.25))
            return float(np.real(integrand_complex))

        integral, _ = quad(integrand, 0.0, u_max, **quad_kwargs)
        prices[i] = (S0 - np.sqrt(S0 * Ki) * np.exp(-r * tau) / np.pi
                     * integral)

    if np.ndim(K) == 0:
        return float(prices[0])
    return prices


# =====================================================================
# Reference: Black-Scholes for sanity checks
# =====================================================================

def black_scholes_call(S0, K, tau, r, sigma):
    """
    Black-Scholes European call price.

    Used as a reference benchmark in the BS limit. When sigma_Heston -> 0
    in the Heston model, the variance v(t) becomes deterministic and
    integrated variance V(T) = theta T + (v_0 - theta)(1 - exp(-kappa T))/kappa.
    The Heston call price approaches the BS call with effective volatility
    sqrt(V(T)/T); in the special case v_0 = theta this is just sqrt(v_0).
    """
    if tau <= 0:
        return max(S0 - K, 0.0)
    if sigma <= 0:
        return max(S0 - K * np.exp(-r * tau), 0.0)

    sqrt_tau = np.sqrt(tau)
    d1 = ((np.log(S0 / K) + (r + 0.5 * sigma * sigma) * tau)
          / (sigma * sqrt_tau))
    d2 = d1 - sigma * sqrt_tau
    return S0 * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)


def put_via_parity(call_price, S0, K, tau, r):
    """European put price via put-call parity: P = C - S0 + K exp(-r tau)."""
    return call_price - S0 + K * np.exp(-r * tau)
