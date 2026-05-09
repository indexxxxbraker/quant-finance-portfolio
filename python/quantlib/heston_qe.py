"""
Heston model: Monte Carlo simulation via Andersen's Quadratic-Exponential (QE) scheme.

This module implements the QE scheme of Andersen (2008), an
approximate-but-accurate sampler for the Heston variance process based
on matching the first two conditional moments of v_{t+dt} | v_t. The
scheme uses two regimes:

    - Quadratic (Q): for psi = s^2/m^2 below psi_c, approximates
      v_{t+dt} as a (b + Z)^2 with Z ~ N(0,1), matching moments.
    - Exponential (E): for psi above psi_c, approximates as a mixture
      of an atom at zero and an exponential tail, again matching
      moments. This is the regime that captures the structural atom
      at zero of the CIR process when the Feller parameter is small.

The log-spot is integrated using Andersen's central discretisation,
which uses both v_n and v_{n+1} (the latter being available from QE),
giving a higher-order log-spot scheme than the simple Euler of Block 3.

Refer to ``theory/phase4/block4_heston_qe.tex`` for the derivation,
including the conditional moments of CIR, the moment-matching
construction of both regimes, and the log-spot scheme.

Public interface
----------------
- ``simulate_terminal_heston_qe``: terminal-only QE simulator with
  O(n_paths) memory.
- ``mc_european_call_heston_qe``: high-level pricer.

References
----------
[Andersen2008]   Andersen, L. (2008). Simple and efficient simulation
                 of the Heston stochastic volatility model. Journal of
                 Computational Finance 11(3).
"""

from __future__ import annotations

import numpy as np

from quantlib.monte_carlo import (
    MCResult,
    mc_estimator,
    _resolve_rng,
)
from quantlib.gbm import _standard_normals


# =====================================================================
# Input validation
# =====================================================================

def _validate_heston_params(kappa, theta, sigma, rho, v0):
    """Validate Heston model parameters. Raises ValueError on failure."""
    if kappa <= 0.0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    if theta <= 0.0:
        raise ValueError(f"theta must be positive, got {theta}")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if not (-1.0 <= rho <= 1.0):
        raise ValueError(f"rho must be in [-1, 1], got {rho}")
    if v0 < 0.0:
        raise ValueError(f"v0 must be non-negative, got {v0}")


def _validate_contract_spec(S0, T):
    if S0 <= 0.0:
        raise ValueError(f"S0 must be positive, got {S0}")
    if T <= 0.0:
        raise ValueError(f"T must be positive, got {T}")


def _validate_strike(K):
    if K <= 0.0:
        raise ValueError(f"K must be positive, got {K}")


def _validate_n_paths(n_paths):
    if not isinstance(n_paths, (int, np.integer)):
        raise TypeError(
            f"n_paths must be an integer, got {type(n_paths).__name__}"
        )
    if n_paths < 2:
        raise ValueError(
            f"n_paths must be at least 2, got {n_paths}"
        )


def _validate_n_steps(n_steps):
    if not isinstance(n_steps, (int, np.integer)):
        raise TypeError(
            f"n_steps must be an integer, got {type(n_steps).__name__}"
        )
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}")


def _validate_qe_scheme_params(psi_c, gamma1, gamma2):
    """Validate the QE scheme tuning parameters."""
    if not (1.0 < psi_c < 2.0):
        raise ValueError(
            f"psi_c must be in (1, 2), got {psi_c}. Andersen's "
            f"recommendation is 1.5."
        )
    if gamma1 < 0.0 or gamma2 < 0.0:
        raise ValueError(
            f"gamma1, gamma2 must be non-negative, got "
            f"({gamma1}, {gamma2})"
        )
    if not np.isclose(gamma1 + gamma2, 1.0):
        raise ValueError(
            f"gamma1 + gamma2 must equal 1, got "
            f"{gamma1} + {gamma2} = {gamma1 + gamma2}"
        )


# =====================================================================
# QE step kernel
# =====================================================================

def _qe_variance_step(v_n, kappa, theta, sigma, dt, psi_c, U, Z):
    """
    Sample v_{n+1} given v_n via Andersen's QE scheme.

    This is the heart of the QE simulator: vectorised over paths,
    branchless except for the regime selector (Q vs E), and matching
    the first two conditional moments of CIR exactly in both regimes.

    Parameters
    ----------
    v_n : ndarray of shape (n_paths,)
        Current variance values (entry-wise; may include zeros for
        paths that recently fell to zero in the E regime).
    kappa, theta, sigma : float
        Heston variance dynamics parameters.
    dt : float
        Time step size.
    psi_c : float
        Threshold between Q and E regimes; in (1, 2). Andersen
        recommends 1.5.
    U : ndarray of shape (n_paths,)
        Standard uniform draws (used in the E regime).
    Z : ndarray of shape (n_paths,)
        Standard normal draws (used in the Q regime).

    Returns
    -------
    v_next : ndarray of shape (n_paths,)
        Sampled v_{n+1}, non-negative by construction.

    Notes
    -----
    The conditional mean and variance of CIR are
        m = theta + (v_n - theta) * exp(-kappa * dt),
        s^2 = v_n * sigma^2 * exp(-kappa*dt) * (1 - exp(-kappa*dt)) / kappa
              + theta * sigma^2 * (1 - exp(-kappa*dt))^2 / (2 * kappa).
    See block4_heston_qe.tex Section 4 for the derivation.
    """
    exp_neg = np.exp(-kappa * dt)
    one_minus_exp = 1.0 - exp_neg

    # Conditional moments of v_{n+1} | v_n.
    m = theta + (v_n - theta) * exp_neg
    s2 = (v_n * sigma * sigma * exp_neg * one_minus_exp / kappa
          + theta * sigma * sigma * one_minus_exp * one_minus_exp
            / (2.0 * kappa))
    psi = s2 / (m * m)

    # Branch on the regime. Both regimes are vectorised; we just
    # combine the results with np.where based on psi.
    is_quadratic = psi <= psi_c

    # ---- Quadratic regime (psi <= psi_c) ----------------------------
    # b^2 = 2/psi - 1 + sqrt(2/psi (2/psi - 1)),  a = m / (1 + b^2)
    # v_next = a * (b + Z)^2.
    # Computed wherever it would matter; result selected by where().
    psi_safe = np.where(is_quadratic, psi, 1.0)  # avoid div-by-zero in inactive branch
    inv_psi = 1.0 / psi_safe
    b2 = 2.0 * inv_psi - 1.0 + np.sqrt(2.0 * inv_psi * (2.0 * inv_psi - 1.0))
    a = m / (1.0 + b2)
    b = np.sqrt(np.maximum(b2, 0.0))
    v_quadratic = a * (b + Z) * (b + Z)

    # ---- Exponential regime (psi > psi_c) ---------------------------
    # p = (psi - 1) / (psi + 1),  beta = 2 / (m * (psi + 1))
    # If U <= p: v_next = 0; else: v_next = -log((1-U)/(1-p)) / beta.
    psi_e = np.where(is_quadratic, 2.0, psi)  # avoid div-by-zero where regime irrelevant
    p = (psi_e - 1.0) / (psi_e + 1.0)
    beta = 2.0 / (m * (psi_e + 1.0))
    # log argument: (1-U)/(1-p), guarded against U > p
    above_p = U > p
    log_arg = np.where(above_p,
                        (1.0 - U) / np.maximum(1.0 - p, 1e-300),
                        1.0)  # log(1) = 0 when below p
    v_exponential_above = -np.log(log_arg) / np.maximum(beta, 1e-300)
    v_exponential = np.where(above_p, v_exponential_above, 0.0)

    return np.where(is_quadratic, v_quadratic, v_exponential)


# =====================================================================
# Log-spot step kernel
# =====================================================================

def _qe_logS_step(log_S_n, v_n, v_next, r, rho, kappa, theta, sigma,
                   dt, gamma1, gamma2, Z_indep):
    """
    Update log-spot from time t_n to t_{n+1} using Andersen's central
    discretisation, which uses both v_n and v_{n+1}.

    The update is
        log_S_{n+1} = log_S_n + K0 + K1 v_n + K2 v_{n+1}
                       + sqrt(K3 v_n + K4 v_{n+1}) * Z_indep,
    with coefficients
        K0 = (r - rho kappa theta / sigma) dt,
        K1 = gamma1 dt (kappa rho / sigma - 1/2) - rho/sigma,
        K2 = gamma2 dt (kappa rho / sigma - 1/2) + rho/sigma,
        K3 = gamma1 dt (1 - rho^2),
        K4 = gamma2 dt (1 - rho^2).
    Z_indep is a standard normal independent of the variance increment.

    See block4_heston_qe.tex Section 6 for the derivation.
    """
    # K coefficients (computed once per step — pass as args from caller
    # if profiling shows this is hot; for clarity we recompute here,
    # the cost is negligible compared to the variance step).
    K0 = (r - rho * kappa * theta / sigma) * dt
    K1 = gamma1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = gamma2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = gamma1 * dt * (1.0 - rho * rho)
    K4 = gamma2 * dt * (1.0 - rho * rho)

    # Variance integral (trapezoidal): K3 v_n + K4 v_{n+1}. Guard
    # against tiny negative values from floating point: v_next is
    # non-negative by construction in QE, but rounding can make
    # K3 * v_n + K4 * v_next slightly negative when both are zero.
    var_term = np.maximum(K3 * v_n + K4 * v_next, 0.0)

    return log_S_n + K0 + K1 * v_n + K2 * v_next + np.sqrt(var_term) * Z_indep


# =====================================================================
# Terminal simulator (O(M) memory)
# =====================================================================

def simulate_terminal_heston_qe(S0, v0, r, kappa, theta, sigma, rho, T,
                                  n_steps, n_paths,
                                  *,
                                  seed=None, rng=None,
                                  antithetic=False,
                                  psi_c=1.5,
                                  gamma1=0.5, gamma2=0.5):
    """
    Simulate the terminal spot S_T under Heston via Andersen QE.

    Memory-efficient: only the current values of (log_S, v) are held
    during the iteration; the trajectories are not stored. For
    European-style payoffs this is the right choice. For path-dependent
    payoffs, build a path simulator on top of the same _qe_step kernels.

    Parameters
    ----------
    S0 : float
        Initial spot, must be positive.
    v0 : float
        Initial variance, must be non-negative.
    r : float
        Risk-free rate (unconstrained).
    kappa, theta, sigma, rho : float
        Heston parameters; see module docstring for ranges.
    T : float
        Time horizon, must be positive.
    n_steps : int
        Number of discretisation steps; n_steps >= 1.
    n_paths : int
        Number of independent paths; n_paths >= 2 (and even when
        ``antithetic=True``).
    seed : int or None, keyword-only
        Seed for the internally-constructed generator. Pass exactly
        one of ``seed`` and ``rng``.
    rng : numpy.random.Generator or None, keyword-only
        Random generator. Pass exactly one of ``seed`` and ``rng``.
    antithetic : bool, keyword-only, optional
        If True, paths are generated in pairs sharing the magnitude of
        all stochastic inputs (uniform U for the E regime, normal Z for
        Q regime, normal Z_indep for the log-spot) with opposite signs
        on the normals. Requires n_paths to be even.
    psi_c : float, keyword-only, optional
        Threshold between Q and E regimes; default 1.5 (Andersen's
        recommendation).
    gamma1, gamma2 : float, keyword-only, optional
        Weights of v_n and v_{n+1} in the log-spot integration;
        gamma1 + gamma2 = 1, both non-negative. Default (0.5, 0.5).

    Returns
    -------
    S_T : ndarray of shape (n_paths,)
        Terminal spot values, all positive.

    Notes
    -----
    Antithetic semantics for QE need a comment: the stochastic inputs
    per step are (U, Z, Z_indep). For antithetic, we negate Z and
    Z_indep (which is standard) but for U we use 1 - U (which preserves
    Uniform(0,1)). This keeps the symmetric effect of antithetic on
    monotone payoffs.
    """
    _validate_heston_params(kappa, theta, sigma, rho, v0)
    _validate_contract_spec(S0, T)
    _validate_n_steps(n_steps)
    _validate_n_paths(n_paths)
    _validate_qe_scheme_params(psi_c, gamma1, gamma2)
    rng = _resolve_rng(seed, rng)

    if antithetic and (n_paths % 2 != 0):
        raise ValueError(
            f"antithetic=True requires n_paths to be even, got {n_paths}"
        )

    dt = T / n_steps

    log_S = np.full(n_paths, np.log(S0), dtype=np.float64)
    v     = np.full(n_paths, v0, dtype=np.float64)

    half = n_paths // 2 if antithetic else n_paths

    for _ in range(n_steps):
        # Generate the three stochastic inputs for this step:
        #   U: uniform draw used in E regime (and ignored in Q regime)
        #   Z: normal draw used in Q regime (and ignored in E regime)
        #   Z_indep: normal independent of variance, drives log-spot
        U_half = rng.uniform(size=half)
        Z_half = _standard_normals(half, rng)
        Z_indep_half = _standard_normals(half, rng)

        if antithetic:
            U = np.concatenate([U_half, 1.0 - U_half])
            Z = np.concatenate([Z_half, -Z_half])
            Z_indep = np.concatenate([Z_indep_half, -Z_indep_half])
        else:
            U = U_half
            Z = Z_half
            Z_indep = Z_indep_half

        v_next = _qe_variance_step(v, kappa, theta, sigma, dt,
                                     psi_c, U, Z)
        log_S = _qe_logS_step(log_S, v, v_next, r, rho, kappa, theta,
                                sigma, dt, gamma1, gamma2, Z_indep)
        v = v_next

    return np.exp(log_S)


# =====================================================================
# High-level pricer
# =====================================================================

def mc_european_call_heston_qe(S0, K, v0, r, kappa, theta, sigma, rho, T,
                                  n_steps, n_paths,
                                  *,
                                  seed=None, rng=None,
                                  antithetic=False,
                                  psi_c=1.5,
                                  gamma1=0.5, gamma2=0.5,
                                  confidence_level=0.95):
    """
    Price a European call under Heston by Andersen QE Monte Carlo.

    Pipeline: simulate ``n_paths`` of S_T using ``n_steps`` QE steps,
    evaluate the discounted payoff ``e^{-rT} (S_T - K)^+`` on each
    path, and reduce to a Monte Carlo result via ``mc_estimator``.

    With ``antithetic=True``, the n_paths discounted payoffs are paired
    and averaged into n_paths/2 antithetic samples BEFORE being passed
    to mc_estimator, exactly as in the full-truncation Euler version of
    Block 3. The MCResult.n_paths field reports n_paths/2 in that case.

    Parameters
    ----------
    Same as ``simulate_terminal_heston_qe``, plus K (strike) and
    confidence_level (for the asymptotic Gaussian interval).

    Returns
    -------
    MCResult
        Named tuple ``(estimate, half_width, sample_variance, n_paths)``.

    Notes
    -----
    The estimator carries a discretisation bias of order O(dt^2) in
    benign regimes (high Feller parameter) and approximately O(dt^2)
    even in low-Feller regimes thanks to QE's atom-at-zero handling.
    Compared to the Block 3 full-truncation Euler estimator
    (``mc_european_call_heston``), QE achieves the same precision with
    typically 4-10x fewer time steps, with the gap widening for low
    Feller parameter (see block4_heston_qe.tex Section 7).
    """
    _validate_strike(K)

    S_T = simulate_terminal_heston_qe(
        S0, v0, r, kappa, theta, sigma, rho, T,
        n_steps, n_paths,
        seed=seed, rng=rng, antithetic=antithetic,
        psi_c=psi_c, gamma1=gamma1, gamma2=gamma2,
    )

    Y = np.exp(-r * T) * np.maximum(S_T - K, 0.0)

    if antithetic:
        half = n_paths // 2
        Y = 0.5 * (Y[:half] + Y[half:])

    return mc_estimator(Y, confidence_level=confidence_level)


# =====================================================================
# Smoke test entry point
# =====================================================================

if __name__ == "__main__":
    S0, K, v0, r = 100.0, 100.0, 0.04, 0.05
    kappa, theta, sigma, rho = 1.5, 0.04, 0.3, -0.7
    T = 0.5

    n_steps = 50  # QE works well with few steps; FT-Euler needs more
    n_paths = 100_000

    print("Heston Monte Carlo: Andersen QE smoke test")
    print(f"S0={S0}, K={K}, T={T}, r={r}")
    print(f"kappa={kappa}, theta={theta}, sigma={sigma}, "
          f"rho={rho}, v0={v0}")
    print(f"n_steps={n_steps}, n_paths={n_paths}")
    print()

    result = mc_european_call_heston_qe(
        S0, K, v0, r, kappa, theta, sigma, rho, T,
        n_steps, n_paths, seed=42,
    )
    print(f"QE estimate     : {result.estimate:.6f}")
    print(f"Half-width (95%): {result.half_width:.6f}")
    print(f"Sample variance : {result.sample_variance:.6f}")
    print(f"Sample size     : {result.n_paths}")

    try:
        from quantlib.heston_fourier import heston_call_lewis
        C_ref = heston_call_lewis(K, T, S0, v0, r,
                                   kappa, theta, sigma, rho)
        print()
        print(f"Fourier reference: {C_ref:.6f}")
        print(f"QE - Fourier     : {result.estimate - C_ref:+.6f}")
        print(f"In half-width    : "
              f"{abs(result.estimate - C_ref) < result.half_width}")
    except ImportError:
        print("(quantlib.heston_fourier not available for cross-check)")

    # Compare with FT-Euler at the same n_steps
    try:
        from quantlib.heston_mc import mc_european_call_heston
        result_fte = mc_european_call_heston(
            S0, K, v0, r, kappa, theta, sigma, rho, T,
            n_steps, n_paths, seed=42,
        )
        print()
        print(f"FT-Euler (same n_steps={n_steps}):")
        print(f"  estimate: {result_fte.estimate:.6f}")
        print(f"  HW:       {result_fte.half_width:.6f}")
        print(f"  bias vs Fourier: "
              f"{result_fte.estimate - C_ref:+.6f}")
        print(f"  QE bias vs Fourier: "
              f"{result.estimate - C_ref:+.6f}")
    except ImportError:
        pass
