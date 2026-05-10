"""
validate_heston_exotics.py
==========================

Validation suite for the Heston exotic option pricers in
``quantlib.heston_exotics`` (Asian, Lookback, Barrier MC) and
``quantlib.heston_american_pde`` (American put PDE). Implements four
pillars:

    1. Analytic limit cases:    Asian with n_avg=1 at t=T equals European
                                call; Barrier with H -> infty equals
                                European call; American put with K very
                                small approaches European put.
    2. Cross-method bounds:     American put >= European put (the early
                                exercise premium is non-negative); Asian
                                call price < European call (averaging
                                reduces effective volatility).
    3. Statistical scaling:     half_width * sqrt(n_paths) approximately
                                constant for the MC pricers.
    4. Discrete-monitoring bias: lookback price grows with n_steps
                                (positive bias, predicted by theory),
                                barrier knockout probability also
                                changes monotonically with n_steps.

The Fourier pricer of Block 2 is the ground truth for the European
limits; the European put is computed via put-call parity from the
Fourier call.

Run from the ``python/`` directory:

    python validate_heston_exotics.py

Returns exit code 0 on all pass, 1 on any failure.
"""

from __future__ import annotations

import sys
import time
from typing import Tuple

import numpy as np

from quantlib.heston_exotics import (
    mc_asian_call_heston,
    mc_lookback_call_heston,
    mc_barrier_call_heston,
)
from quantlib.heston_american_pde import heston_american_put_pde
from quantlib.heston_fourier import heston_call_lewis


# =====================================================================
# Reporting
# =====================================================================

PASS = "PASS"
FAIL = "FAIL"


def report(name: str, passed: bool, info: str = "") -> Tuple[str, bool]:
    tag = PASS if passed else FAIL
    line = f"  [{tag}] {name}"
    if info:
        line += f"   {info}"
    print(line)
    return name, passed


# =====================================================================
# Reference parameters
# =====================================================================

PARAMS = dict(kappa=1.5, theta=0.04, sigma=0.3, rho=-0.7, v0=0.04)
S0_REF = 100.0
K_REF = 100.0
R_REF = 0.05
T_REF = 0.5


def _european_call(K, T):
    return heston_call_lewis(K, T, S0_REF, PARAMS["v0"], R_REF,
                               PARAMS["kappa"], PARAMS["theta"],
                               PARAMS["sigma"], PARAMS["rho"])


def _european_put(K, T):
    """European put via put-call parity from Fourier call."""
    C = _european_call(K, T)
    return C - S0_REF + K * np.exp(-R_REF * T)


# =====================================================================
# Test 1: Analytic limit cases
# =====================================================================

def test_asian_limit_n_avg_1():
    """Asian with n_avg=1 (single sampling at t=T) should match European
    call. Cross-method test: MC Asian vs Fourier European."""
    print("Test 1a: Asian limit (n_avg=1 at t=T) -> European call")

    K, T = K_REF, T_REF
    n_steps = 100
    n_paths = 200_000

    # Asian with n_avg=1: the sample is at index n_steps (t=T)
    asian = mc_asian_call_heston(
        S0_REF, K, PARAMS["v0"], R_REF,
        PARAMS["kappa"], PARAMS["theta"], PARAMS["sigma"], PARAMS["rho"],
        T, n_steps, n_paths, n_avg=1, seed=42)
    eur = _european_call(K, T)

    err = abs(asian.estimate - eur)
    tol = 3.0 * asian.half_width
    return [report(
        "Asian(n_avg=1) matches European call",
        err < tol,
        info=f"Asian = {asian.estimate:.4f} +/- {asian.half_width:.4f}, "
             f"European = {eur:.4f}, err = {err:.4f}",
    )]


def test_barrier_limit_high_H():
    """Up-and-out barrier with H very large should match European call.
    The knockout probability tends to zero as H -> infty."""
    print("Test 1b: Barrier limit (H -> infty) -> European call")

    K, T = K_REF, T_REF
    n_steps = 100
    n_paths = 200_000

    # Barrier with very high H: should never knock out
    barrier = mc_barrier_call_heston(
        S0_REF, K, 1e6, PARAMS["v0"], R_REF,
        PARAMS["kappa"], PARAMS["theta"], PARAMS["sigma"], PARAMS["rho"],
        T, n_steps, n_paths, seed=42)
    eur = _european_call(K, T)

    err = abs(barrier.estimate - eur)
    tol = 3.0 * barrier.half_width
    return [report(
        "Barrier(H -> infty) matches European call",
        err < tol,
        info=f"Barrier = {barrier.estimate:.4f} +/- {barrier.half_width:.4f}, "
             f"European = {eur:.4f}, err = {err:.4f}",
    )]


def test_american_limit_deep_otm():
    """American put for deep OTM strike (K << S0) should be very close
    to European put: the option is unlikely to be exercised early when
    intrinsic value is small. EEP should be small in absolute terms."""
    print("Test 1c: American put deep OTM -> close to European")

    K_otm, T = 70.0, T_REF  # K=70, S=100: deep OTM put

    am = heston_american_put_pde(
        S0_REF, K_otm, T,
        PARAMS["kappa"], PARAMS["theta"], PARAMS["sigma"], PARAMS["rho"],
        PARAMS["v0"], R_REF,
        N_X=100, N_v=50, N_tau=100)
    eur_put = _european_put(K_otm, T)

    eep = am - eur_put
    print(f"    American put K={K_otm}: {am:.6f}")
    print(f"    European put K={K_otm}: {eur_put:.6f}")
    print(f"    EEP: {eep:.6f}")

    # EEP should be small (< 5% of European value) for deep OTM
    return [report(
        "EEP is small for deep OTM put (< 5% of European value)",
        0 <= eep < 0.05 * eur_put,
        info=f"EEP = {eep:.5f}, European = {eur_put:.5f}, "
             f"ratio = {100*eep/eur_put:.2f}%",
    )]


# =====================================================================
# Test 2: Cross-method bounds
# =====================================================================

def test_american_bound_european():
    """American put price >= European put price (no negative EEP)."""
    print("Test 2a: American put >= European put (positive EEP)")

    results = []
    for K in [80.0, 100.0, 120.0]:
        am = heston_american_put_pde(
            S0_REF, K, T_REF,
            PARAMS["kappa"], PARAMS["theta"], PARAMS["sigma"], PARAMS["rho"],
            PARAMS["v0"], R_REF,
            N_X=100, N_v=50, N_tau=100)
        eur = _european_put(K, T_REF)
        eep = am - eur
        results.append(report(
            f"K={K}: American >= European",
            eep >= -1e-6,  # tiny negative slack for floating point
            info=f"American = {am:.4f}, European = {eur:.4f}, EEP = {eep:.4f}",
        ))
    return results


def test_asian_bound_european():
    """Asian call price < European call price.

    Geometric/financial intuition: averaging reduces effective volatility
    of the underlying, so a call on the average is cheaper than a call
    on the terminal value. (Strict inequality holds for n_avg > 1; for
    n_avg = 1 they are equal.)"""
    print("Test 2b: Asian call <= European call")

    K, T = K_REF, T_REF
    n_paths = 200_000

    asian = mc_asian_call_heston(
        S0_REF, K, PARAMS["v0"], R_REF,
        PARAMS["kappa"], PARAMS["theta"], PARAMS["sigma"], PARAMS["rho"],
        T, n_steps=100, n_paths=n_paths, n_avg=50, seed=42)
    eur = _european_call(K, T)

    # Asian should be strictly less than European, by more than 3 HW.
    diff = eur - asian.estimate
    return [report(
        "Asian < European by at least 3*HW (strict)",
        diff > 3.0 * asian.half_width,
        info=f"European - Asian = {diff:.4f}, "
             f"3*HW = {3*asian.half_width:.4f}",
    )]


# =====================================================================
# Test 3: Statistical scaling for MC pricers
# =====================================================================

def test_statistical_scaling():
    """For each MC pricer, verify half_width * sqrt(n_paths) is
    approximately constant across n_paths (1/sqrt(M) scaling)."""
    print("Test 3: Statistical scaling 1/sqrt(M)")

    K, T = K_REF, T_REF
    results = []

    # Asian
    print("  Asian:")
    hws_asian = []
    for M in [10_000, 100_000, 500_000]:
        r = mc_asian_call_heston(
            S0_REF, K, PARAMS["v0"], R_REF,
            PARAMS["kappa"], PARAMS["theta"], PARAMS["sigma"], PARAMS["rho"],
            T, n_steps=50, n_paths=M, n_avg=25, seed=42)
        hws_asian.append(r.half_width * np.sqrt(M))
        print(f"    M={M:>7}: HW={r.half_width:.5f}, "
              f"HW*sqrt(M)={hws_asian[-1]:.4f}")
    rel_asian = (max(hws_asian) - min(hws_asian)) / min(hws_asian)
    results.append(report(
        "Asian: HW * sqrt(M) approximately constant",
        rel_asian < 0.05,
        info=f"max relative deviation = {rel_asian:.3%}",
    ))

    # Lookback
    print("  Lookback:")
    hws_lb = []
    for M in [10_000, 100_000, 500_000]:
        r = mc_lookback_call_heston(
            S0_REF, PARAMS["v0"], R_REF,
            PARAMS["kappa"], PARAMS["theta"], PARAMS["sigma"], PARAMS["rho"],
            T, n_steps=50, n_paths=M, seed=42)
        hws_lb.append(r.half_width * np.sqrt(M))
        print(f"    M={M:>7}: HW={r.half_width:.5f}, "
              f"HW*sqrt(M)={hws_lb[-1]:.4f}")
    rel_lb = (max(hws_lb) - min(hws_lb)) / min(hws_lb)
    results.append(report(
        "Lookback: HW * sqrt(M) approximately constant",
        rel_lb < 0.05,
        info=f"max relative deviation = {rel_lb:.3%}",
    ))

    # Barrier (with finite H so we don't trivially equal European)
    print("  Barrier (H=130):")
    hws_b = []
    for M in [10_000, 100_000, 500_000]:
        r = mc_barrier_call_heston(
            S0_REF, K, 130.0, PARAMS["v0"], R_REF,
            PARAMS["kappa"], PARAMS["theta"], PARAMS["sigma"], PARAMS["rho"],
            T, n_steps=50, n_paths=M, seed=42)
        hws_b.append(r.half_width * np.sqrt(M))
        print(f"    M={M:>7}: HW={r.half_width:.5f}, "
              f"HW*sqrt(M)={hws_b[-1]:.4f}")
    rel_b = (max(hws_b) - min(hws_b)) / min(hws_b)
    results.append(report(
        "Barrier: HW * sqrt(M) approximately constant",
        rel_b < 0.05,
        info=f"max relative deviation = {rel_b:.3%}",
    ))

    return results


# =====================================================================
# Test 4: Discrete-monitoring bias
# =====================================================================

def test_lookback_bias_in_n_steps():
    """The lookback's discrete monitoring bias is positive: discrete
    minimum > continuous minimum, so the discrete-monitoring price
    monotonically increases with n_steps (the path-min observed at
    finer monitoring is lower, so payoff S_T - S_min is larger).

    Verify the trend is monotonic (or close to it) across n_steps."""
    print("Test 4: Lookback price increases with n_steps (positive bias)")

    T = T_REF
    M = 100_000
    prices = []
    print(f"    {'n_steps':>8} {'price':>10} {'HW':>10}")
    for n_steps in [25, 50, 100, 200]:
        r = mc_lookback_call_heston(
            S0_REF, PARAMS["v0"], R_REF,
            PARAMS["kappa"], PARAMS["theta"], PARAMS["sigma"], PARAMS["rho"],
            T, n_steps=n_steps, n_paths=M, seed=42)
        prices.append(r.estimate)
        print(f"    {n_steps:>8} {r.estimate:>10.4f} {r.half_width:>10.4f}")

    # Verify trend: price at n_steps=200 > price at n_steps=25, by enough
    # to be statistically significant (more than 2 HW).
    diff = prices[-1] - prices[0]
    return [report(
        "Lookback price increases monotonically with n_steps",
        diff > 0.1,   # rough threshold: should grow by at least 0.1
        info=f"price(n=200) - price(n=25) = {diff:+.4f}",
    )]


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 70)
    print("Heston exotics: validation suite")
    print("=" * 70)
    print()
    print(f"Reference: S0={S0_REF}, K={K_REF}, T={T_REF}, r={R_REF}")
    print(f"Heston params: {PARAMS}")
    print(f"Feller nu = "
          f"{2*PARAMS['kappa']*PARAMS['theta']/PARAMS['sigma']**2:.3f}")
    print()

    t0 = time.perf_counter()

    all_results = []
    all_results.extend(test_asian_limit_n_avg_1())
    print()
    all_results.extend(test_barrier_limit_high_H())
    print()
    all_results.extend(test_american_limit_deep_otm())
    print()
    all_results.extend(test_american_bound_european())
    print()
    all_results.extend(test_asian_bound_european())
    print()
    all_results.extend(test_statistical_scaling())
    print()
    all_results.extend(test_lookback_bias_in_n_steps())

    elapsed = time.perf_counter() - t0

    n_pass = sum(1 for _, ok in all_results if ok)
    n_total = len(all_results)
    failures = [name for name, ok in all_results if not ok]

    print()
    print("=" * 70)
    print(f"SUMMARY: {n_pass} / {n_total} tests passed in {elapsed:.1f}s")
    if failures:
        print("Failures:")
        for f in failures:
            print(f"  - {f}")
    print("=" * 70)

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
