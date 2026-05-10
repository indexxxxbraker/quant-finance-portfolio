"""
Heston model: exotic option pricing via Andersen QE Monte Carlo.

This module implements three families of path-dependent European-style
exotic options under Heston:

    - Arithmetic-average Asian call
    - Floating-strike lookback call
    - Up-and-out barrier call

All three share the same simulation infrastructure --- the QE scheme
of Block 4 --- and differ only in the path-functional accumulated
during the time loop. The pattern mirrors the "infrastructure reusable,
recipe replaceable" lesson from Block 3-4: the simulator structure is
shared, the payoff is the recipe.

For early-exercise payoffs (American puts), see the PDE-based PSOR
extension in a separate module; Monte Carlo with forward-only paths
cannot natively price early exercise.

See ``theory/phase4/block6_heston_calibration_exotics.tex`` for the
mathematical statement of each payoff and the discussion of
discrete-monitoring biases for lookback and barrier options.

Public interface
----------------
- ``mc_asian_call_heston``: arithmetic-average Asian call.
- ``mc_lookback_call_heston``: floating-strike lookback call.
- ``mc_barrier_call_heston``: up-and-out barrier call.

References
----------
[Glasserman2003]  Glasserman, P. (2003). Monte Carlo Methods in
                  Financial Engineering. Springer.
"""

from __future__ import annotations

import numpy as np

from quantlib.monte_carlo import MCResult, mc_estimator, _resolve_rng
from quantlib.gbm import _standard_normals


# =====================================================================
# Input validation
# =====================================================================

def _validate_heston_params(kappa, theta, sigma, rho, v0):
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


def _validate_contract(S0, T):
    if S0 <= 0.0:
        raise ValueError(f"S0 must be positive, got {S0}")
    if T <= 0.0:
        raise ValueError(f"T must be positive, got {T}")


def _validate_strike(K):
    if K <= 0.0:
        raise ValueError(f"K must be positive, got {K}")


def _validate_n_paths(n_paths):
    if not isinstance(n_paths, (int, np.integer)):
        raise TypeError(f"n_paths must be an integer")
    if n_paths < 2:
        raise ValueError(f"n_paths must be at least 2, got {n_paths}")


def _validate_n_steps(n_steps):
    if not isinstance(n_steps, (int, np.integer)):
        raise TypeError(f"n_steps must be an integer")
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}")


def _validate_qe_scheme_params(psi_c, gamma1, gamma2):
    if not (1.0 < psi_c < 2.0):
        raise ValueError(f"psi_c must be in (1, 2), got {psi_c}")
    if gamma1 < 0.0 or gamma2 < 0.0:
        raise ValueError(f"gamma1, gamma2 must be non-negative")
    if not np.isclose(gamma1 + gamma2, 1.0):
        raise ValueError(f"gamma1 + gamma2 must equal 1")


# =====================================================================
# QE step kernels (vectorised over paths)
# =====================================================================
#
# Reproduce the QE kernels from Block 4 here for self-containment.
# An alternative would be to import them from quantlib.heston_qe, but
# their underscore-prefixed names mean they're not part of the stable
# public API; copying the small kernel keeps this module robust to
# future refactors of heston_qe.

def _qe_variance_step(v_n, kappa, theta, sigma, dt, psi_c, U, Z):
    """Sample v_{n+1}|v_n via Andersen QE; vectorised over paths."""
    exp_neg = np.exp(-kappa * dt)
    one_minus_exp = 1.0 - exp_neg
    m  = theta + (v_n - theta) * exp_neg
    s2 = (v_n * sigma * sigma * exp_neg * one_minus_exp / kappa
          + theta * sigma * sigma * one_minus_exp * one_minus_exp
            / (2.0 * kappa))
    psi = s2 / (m * m)
    is_quadratic = psi <= psi_c

    psi_safe = np.where(is_quadratic, psi, 1.0)
    inv_psi = 1.0 / psi_safe
    b2 = 2.0 * inv_psi - 1.0 + np.sqrt(2.0 * inv_psi * (2.0 * inv_psi - 1.0))
    a = m / (1.0 + b2)
    b = np.sqrt(np.maximum(b2, 0.0))
    v_quadratic = a * (b + Z) * (b + Z)

    psi_e = np.where(is_quadratic, 2.0, psi)
    p = (psi_e - 1.0) / (psi_e + 1.0)
    beta = 2.0 / (m * (psi_e + 1.0))
    above_p = U > p
    log_arg = np.where(above_p, (1.0 - U) / np.maximum(1.0 - p, 1e-300), 1.0)
    v_exp_above = -np.log(log_arg) / np.maximum(beta, 1e-300)
    v_exponential = np.where(above_p, v_exp_above, 0.0)

    return np.where(is_quadratic, v_quadratic, v_exponential)


def _qe_logS_step(log_S_n, v_n, v_next, r, rho, kappa, theta, sigma,
                   dt, gamma1, gamma2, Z_indep):
    """Update log-spot via Andersen central discretisation."""
    K0 = (r - rho * kappa * theta / sigma) * dt
    K1 = gamma1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = gamma2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = gamma1 * dt * (1.0 - rho * rho)
    K4 = gamma2 * dt * (1.0 - rho * rho)
    var_term = np.maximum(K3 * v_n + K4 * v_next, 0.0)
    return log_S_n + K0 + K1 * v_n + K2 * v_next + np.sqrt(var_term) * Z_indep


# =====================================================================
# Generic path-aware simulator
# =====================================================================

def _simulate_paths_qe_with_accumulator(
        S0, v0, r, kappa, theta, sigma, rho, T,
        n_steps, n_paths, rng,
        accumulator_init, accumulator_update,
        psi_c=1.5, gamma1=0.5, gamma2=0.5):
    """
    Run the QE simulator and accumulate a path-functional.

    The accumulator is provided as two callables:
        - accumulator_init(S_0, n_paths) -> any state
        - accumulator_update(state, S_t, t_idx) -> new state

    where S_t is an ndarray of shape (n_paths,) with the spot values at
    step t_idx. The state is opaque to this function; it can be a single
    array (running max, running sum, etc.) or a tuple.

    Returns (S_T, final_state) where S_T is the terminal spot and
    final_state is the result of all accumulator_update calls.

    Notes
    -----
    The simulator is step-outer (Python NumPy vectorised over paths,
    Python loop over steps), same pattern as heston_qe.py. The
    accumulator is called once per step, including the initial value
    (so it sees S_0, S_{t_1}, ..., S_T).
    """
    dt = T / n_steps
    log_S0 = np.log(S0)

    log_S = np.full(n_paths, log_S0, dtype=np.float64)
    v     = np.full(n_paths, v0,     dtype=np.float64)
    S_current = np.exp(log_S)  # = S_0 initially

    state = accumulator_init(S_current, n_paths)

    for k in range(1, n_steps + 1):
        U = rng.uniform(size=n_paths)
        Z = _standard_normals(n_paths, rng)
        Z_indep = _standard_normals(n_paths, rng)

        v_next = _qe_variance_step(v, kappa, theta, sigma, dt, psi_c, U, Z)
        log_S = _qe_logS_step(log_S, v, v_next, r, rho, kappa, theta, sigma,
                                dt, gamma1, gamma2, Z_indep)
        v = v_next

        S_current = np.exp(log_S)
        state = accumulator_update(state, S_current, k)

    return S_current, state


# =====================================================================
# Asian (arithmetic-average)
# =====================================================================

def mc_asian_call_heston(S0, K, v0, r, kappa, theta, sigma, rho, T,
                          n_steps, n_paths,
                          *,
                          n_avg=None,
                          seed=None, rng=None,
                          psi_c=1.5, gamma1=0.5, gamma2=0.5,
                          confidence_level=0.95):
    """
    Price an arithmetic-average Asian call under Heston via QE MC.

    The payoff is
        max(mean(S_{t_1}, ..., S_{t_n_avg}) - K, 0),
    where the averaging dates are uniformly spaced over [0, T] (i.e.,
    t_k = k * T / n_avg for k = 1, ..., n_avg). For a daily Asian over
    a year, n_avg ~ 252; for a monthly Asian over a year, n_avg = 12.

    Parameters
    ----------
    S0, K, v0, r, kappa, theta, sigma, rho, T : float
        Standard contract and Heston parameters.
    n_steps : int
        Number of QE simulation steps. Must be a multiple of n_avg if
        the averaging dates are to align with simulation steps.
    n_paths : int
        Number of MC paths.
    n_avg : int, keyword-only, optional
        Number of averaging dates. If None, defaults to n_steps (i.e.,
        averaging at every simulation step).
    seed, rng : pass exactly one
    psi_c, gamma1, gamma2 : QE scheme parameters
    confidence_level : float, default 0.95

    Returns
    -------
    MCResult

    Notes
    -----
    The averaging dates are placed at the simulation step indices
    {n_steps / n_avg, 2 n_steps / n_avg, ..., n_steps}; if n_steps is
    not divisible by n_avg the dates fall on rounded indices, slightly
    altering the precise averaging schedule. For best results choose
    n_steps = k * n_avg for some integer k.
    """
    _validate_heston_params(kappa, theta, sigma, rho, v0)
    _validate_contract(S0, T)
    _validate_strike(K)
    _validate_n_steps(n_steps)
    _validate_n_paths(n_paths)
    _validate_qe_scheme_params(psi_c, gamma1, gamma2)
    rng = _resolve_rng(seed, rng)

    if n_avg is None:
        n_avg = n_steps
    if n_avg < 1:
        raise ValueError(f"n_avg must be at least 1, got {n_avg}")

    # Compute the simulation step indices at which we sample for averaging
    sample_indices = set(int(round(k * n_steps / n_avg))
                          for k in range(1, n_avg + 1))

    def init(S0_arr, n_paths):
        # state = (running_sum, count_observed)
        return (np.zeros(n_paths), 0)

    def update(state, S_current, k):
        running_sum, count = state
        if k in sample_indices:
            return (running_sum + S_current, count + 1)
        return state

    _, (running_sum, count) = _simulate_paths_qe_with_accumulator(
        S0, v0, r, kappa, theta, sigma, rho, T,
        n_steps, n_paths, rng,
        init, update,
        psi_c=psi_c, gamma1=gamma1, gamma2=gamma2,
    )

    if count == 0:
        raise RuntimeError(
            "no averaging samples observed; check n_steps / n_avg")
    avg_S = running_sum / count
    payoffs = np.exp(-r * T) * np.maximum(avg_S - K, 0.0)
    return mc_estimator(payoffs, confidence_level=confidence_level)


# =====================================================================
# Lookback (floating-strike)
# =====================================================================

def mc_lookback_call_heston(S0, v0, r, kappa, theta, sigma, rho, T,
                              n_steps, n_paths,
                              *,
                              seed=None, rng=None,
                              psi_c=1.5, gamma1=0.5, gamma2=0.5,
                              confidence_level=0.95):
    """
    Price a floating-strike lookback call under Heston via QE MC.

    The payoff is
        S_T - min(S_{t_0}, S_{t_1}, ..., S_{t_n_steps}),
    where the minimum is taken over all simulation steps.

    Parameters
    ----------
    S0, v0, r, kappa, theta, sigma, rho, T : float
    n_steps : int
        Number of QE simulation steps. The minimum is monitored at
        each step (discrete monitoring). For finer monitoring, increase
        n_steps; the limit n_steps -> infinity recovers continuous
        monitoring (modulo simulation noise).
    n_paths : int
    seed, rng : pass exactly one
    psi_c, gamma1, gamma2 : QE scheme parameters
    confidence_level : float, default 0.95

    Returns
    -------
    MCResult

    Notes
    -----
    Discrete-monitoring bias: the minimum of a discretely-sampled path
    is a biased estimator of the true continuous-time minimum (the path
    can dip lower between samples). The bias is positive (true min is
    smaller) and scales as O(sqrt(dt)). For accurate continuous
    monitoring at moderate n_steps, use the Asmussen-Glynn-Pitman
    correction; not implemented here.
    """
    _validate_heston_params(kappa, theta, sigma, rho, v0)
    _validate_contract(S0, T)
    _validate_n_steps(n_steps)
    _validate_n_paths(n_paths)
    _validate_qe_scheme_params(psi_c, gamma1, gamma2)
    rng = _resolve_rng(seed, rng)

    def init(S0_arr, n_paths):
        # State is the running minimum
        return S0_arr.copy()

    def update(state, S_current, k):
        return np.minimum(state, S_current)

    S_T, S_min = _simulate_paths_qe_with_accumulator(
        S0, v0, r, kappa, theta, sigma, rho, T,
        n_steps, n_paths, rng,
        init, update,
        psi_c=psi_c, gamma1=gamma1, gamma2=gamma2,
    )

    payoffs = np.exp(-r * T) * (S_T - S_min)
    return mc_estimator(payoffs, confidence_level=confidence_level)


# =====================================================================
# Up-and-out barrier
# =====================================================================

def mc_barrier_call_heston(S0, K, H, v0, r, kappa, theta, sigma, rho, T,
                              n_steps, n_paths,
                              *,
                              seed=None, rng=None,
                              psi_c=1.5, gamma1=0.5, gamma2=0.5,
                              confidence_level=0.95):
    """
    Price an up-and-out barrier call under Heston via QE MC.

    The payoff is
        max(S_T - K, 0) * I(max(S_{t_0}, ..., S_{t_n_steps}) < H),
    where I(.) is the indicator. The option pays the European call
    payoff only if the path never crosses the barrier H; otherwise it
    "knocks out" to zero.

    Parameters
    ----------
    S0, K, v0, r, kappa, theta, sigma, rho, T : float
    H : float
        Barrier level, must be > S0 (otherwise the option is
        immediately knocked out at t=0).
    n_steps : int
        Number of QE simulation steps. The maximum is monitored at
        each step (discrete monitoring); see Notes.
    n_paths : int
    seed, rng : pass exactly one
    psi_c, gamma1, gamma2 : QE scheme parameters
    confidence_level : float, default 0.95

    Returns
    -------
    MCResult

    Notes
    -----
    Discrete-monitoring bias: the maximum of a discretely-sampled path
    is biased downward (true max is larger). For up-and-out barriers,
    this means the discrete-monitoring price overestimates the true
    continuous-monitoring price (paths that "should" knock out are
    counted as surviving). The bias scales as O(sqrt(dt)). Brownian-
    bridge corrections (Beaglehole-Dybvig-Zhou) reduce this; not
    implemented here.
    """
    _validate_heston_params(kappa, theta, sigma, rho, v0)
    _validate_contract(S0, T)
    _validate_strike(K)
    if H <= S0:
        raise ValueError(f"H must be greater than S0 (got H={H}, S0={S0})")
    _validate_n_steps(n_steps)
    _validate_n_paths(n_paths)
    _validate_qe_scheme_params(psi_c, gamma1, gamma2)
    rng = _resolve_rng(seed, rng)

    def init(S0_arr, n_paths):
        # State is the running maximum
        return S0_arr.copy()

    def update(state, S_current, k):
        return np.maximum(state, S_current)

    S_T, S_max = _simulate_paths_qe_with_accumulator(
        S0, v0, r, kappa, theta, sigma, rho, T,
        n_steps, n_paths, rng,
        init, update,
        psi_c=psi_c, gamma1=gamma1, gamma2=gamma2,
    )

    knocked_out = S_max >= H
    payoffs = np.where(
        knocked_out,
        0.0,
        np.exp(-r * T) * np.maximum(S_T - K, 0.0),
    )
    return mc_estimator(payoffs, confidence_level=confidence_level)


# =====================================================================
# Smoke test entry point
# =====================================================================

if __name__ == "__main__":
    S0, v0, r = 100.0, 0.04, 0.05
    kappa, theta, sigma, rho = 1.5, 0.04, 0.3, -0.7
    T = 0.5

    print("Heston exotics: smoke test")
    print(f"S0={S0}, T={T}, r={r}")
    print(f"kappa={kappa}, theta={theta}, sigma={sigma}, "
          f"rho={rho}, v0={v0}\n")

    print("=" * 60)
    print("Asian call (arithmetic average)")
    print("=" * 60)
    K = 100.0
    n_steps = 100
    n_paths = 100_000
    n_avg_options = [10, 50, 100]
    for n_avg in n_avg_options:
        result = mc_asian_call_heston(
            S0, K, v0, r, kappa, theta, sigma, rho, T,
            n_steps, n_paths, n_avg=n_avg, seed=42)
        print(f"  K={K}, n_avg={n_avg:>3}: "
              f"{result.estimate:.4f} +/- {result.half_width:.4f}")

    print()
    print("=" * 60)
    print("Lookback call (floating strike)")
    print("=" * 60)
    for n_steps in [50, 100, 200]:
        result = mc_lookback_call_heston(
            S0, v0, r, kappa, theta, sigma, rho, T,
            n_steps, n_paths, seed=42)
        print(f"  n_steps={n_steps:>3}: "
              f"{result.estimate:.4f} +/- {result.half_width:.4f}")
    # Convergence note: lookback price increases as n_steps -> infinity
    # because the discrete-monitoring bias is positive (true min is smaller).

    print()
    print("=" * 60)
    print("Up-and-out barrier call")
    print("=" * 60)
    K = 100.0
    n_steps = 100
    print(f"  K={K}, varying barrier H:")
    for H in [110.0, 120.0, 150.0, 1e6]:
        result = mc_barrier_call_heston(
            S0, K, H, v0, r, kappa, theta, sigma, rho, T,
            n_steps, n_paths, seed=42)
        h_label = "infty" if H >= 1e5 else f"{H:.0f}"
        print(f"    H={h_label:>6}: "
              f"{result.estimate:.4f} +/- {result.half_width:.4f}")
    # Note: H=infty should match the European call price.

    try:
        from quantlib.heston_fourier import heston_call_lewis
        C_eur = heston_call_lewis(K, T, S0, v0, r,
                                   kappa, theta, sigma, rho)
        print(f"\n  European reference (H -> infty): {C_eur:.4f}")
    except ImportError:
        pass
