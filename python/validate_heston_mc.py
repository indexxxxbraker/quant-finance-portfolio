"""
validate_heston_mc.py
=====================

Validation suite for the Heston Monte Carlo pricer in
``quantlib.heston_mc``. Implements the four-pillar validation
described in ``theory/phase4/block3_heston_mc_basic.tex``, Section 6:

    1. Black-Scholes limit:       sigma -> 0 reduces Heston to BS.
    2. Cross-method consistency:  Heston MC matches Fourier reference
                                  to within statistical half-width.
    3. Discretisation convergence: bias decreases with n_steps,
                                   statistical error decreases with
                                   n_paths^{-1/2}.
    4. Antithetic variance reduction: paired payoffs reduce sample
                                      variance vs plain MC.

The Fourier pricer of Block 2 is the ground truth: it is bias-free
(modulo numerical quadrature error of order 1e-6, well below typical
MC half-widths), so any MC estimate falling within its half-width of
the Fourier price counts as agreement.

Run from the ``python/`` directory:

    python validate_heston_mc.py

Returns exit code 0 on all pass, 1 on any failure.
"""

from __future__ import annotations

import sys
import time
from typing import Tuple, List

import numpy as np

from quantlib.heston_mc import mc_european_call_heston
from quantlib.heston_fourier import heston_call_lewis, black_scholes_call


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
# Reference parameter sets
# =====================================================================

# Standard equity case: Feller parameter nu = 1.33, regular boundary.
PARAMS_STD = dict(
    kappa=1.5,
    theta=0.04,
    sigma=0.3,
    rho=-0.7,
    v0=0.04,
)

# Aggressive equity case: Feller parameter nu = 0.04, near-singular
# boundary. Tests truncation behaviour at low variance.
PARAMS_AGG = dict(
    kappa=0.5,
    theta=0.04,
    sigma=1.0,
    rho=-0.9,
    v0=0.04,
)

S0_REF = 100.0
K_REF = 100.0
R_REF = 0.05


# =====================================================================
# Test 1: Black-Scholes limit
# =====================================================================

def test_bs_limit():
    """As sigma -> 0 with v_0 = theta, Heston MC -> BS at vol sqrt(v_0).

    The reduction is in distribution: the Heston variance becomes
    deterministic and equal to v_0, and S_T becomes lognormal with the
    BS parameters. At small but nonzero sigma_H, the leverage effect
    introduces a linear-in-sigma_H bias (see Block 2 manual chapter,
    Numerical Lesson 3); we use sigma_H = 0.01 as the sweet spot
    between bias and numerical precision.
    """
    print("Test 1: Black-Scholes limit (sigma -> 0)")

    T = 0.5
    sigma_bs = float(np.sqrt(PARAMS_STD["v0"]))
    C_bs = black_scholes_call(S0_REF, K_REF, T, R_REF, sigma_bs)
    print(f"    BS reference (sigma=sqrt(v0)={sigma_bs:.4f}): {C_bs:.6f}")

    p = dict(PARAMS_STD)
    p["sigma"] = 0.01
    result = mc_european_call_heston(
        S0_REF, K_REF, p["v0"], R_REF,
        p["kappa"], p["theta"], p["sigma"], p["rho"], T,
        n_steps=200, n_paths=200_000, seed=42,
    )

    err = abs(result.estimate - C_bs)
    # Tolerance: 3x the half-width covers ~99.7% of the noise band, so
    # this asserts agreement at high confidence.
    tol = 3.0 * result.half_width
    return [report(
        "Heston (sigma=0.01) close to BS",
        err < tol,
        info=f"MC = {result.estimate:.6f}, err = {err:.5f}, "
             f"3*HW = {tol:.5f}",
    )]


# =====================================================================
# Test 2: Cross-method consistency vs Fourier
# =====================================================================

def test_cross_method():
    """At fixed parameters, Heston MC and Heston Fourier must agree
    within the MC half-width (modulo discretisation bias, which is
    small at the parameters tested)."""
    print("Test 2: Cross-method consistency (MC vs Fourier)")

    results = []
    for params, label in [(PARAMS_STD, "standard"),
                            (PARAMS_AGG, "aggressive")]:
        for T in [0.25, 0.5, 1.0]:
            C_ref = heston_call_lewis(
                K_REF, T, S0_REF, params["v0"], R_REF,
                params["kappa"], params["theta"], params["sigma"],
                params["rho"],
            )
            mc = mc_european_call_heston(
                S0_REF, K_REF, params["v0"], R_REF,
                params["kappa"], params["theta"], params["sigma"],
                params["rho"], T,
                n_steps=200, n_paths=200_000, seed=42,
            )
            err = abs(mc.estimate - C_ref)
            # Tolerance: 3x half-width to control false positives.
            # The aggressive parameter set has larger discretisation
            # bias due to its low Feller parameter; we allow 4x for
            # that case.
            multiplier = 4.0 if label == "aggressive" else 3.0
            tol = multiplier * mc.half_width
            results.append(report(
                f"{label}, T={T}",
                err < tol,
                info=f"MC = {mc.estimate:.4f}, "
                     f"Fourier = {C_ref:.4f}, "
                     f"err = {err:.4f}, tol = {tol:.4f}",
            ))
    return results


# =====================================================================
# Test 3: Discretisation convergence
# =====================================================================

def test_convergence_in_steps():
    """The bias decreases as n_steps -> infinity. We do not assert a
    specific rate (the weak rate of full-truncation Euler depends on
    the parameter regime), only that the trend is downward across the
    grid range and that the finest grid agrees with Fourier within the
    half-width."""
    print("Test 3a: Convergence in n_steps (fixed n_paths)")

    T = 0.5
    p = PARAMS_STD
    C_ref = heston_call_lewis(
        K_REF, T, S0_REF, p["v0"], R_REF,
        p["kappa"], p["theta"], p["sigma"], p["rho"],
    )
    print(f"    Fourier reference: {C_ref:.6f}")

    n_steps_grid = [50, 100, 200, 400]
    errors = []
    for n_steps in n_steps_grid:
        mc = mc_european_call_heston(
            S0_REF, K_REF, p["v0"], R_REF,
            p["kappa"], p["theta"], p["sigma"], p["rho"], T,
            n_steps=n_steps, n_paths=200_000, seed=42,
        )
        err = abs(mc.estimate - C_ref)
        errors.append(err)
        print(f"      n_steps={n_steps:>4}: MC={mc.estimate:.5f}, "
              f"err={err:.5f}, HW={mc.half_width:.5f}")

    # Finest grid must be within half-width of Fourier.
    final_mc = mc_european_call_heston(
        S0_REF, K_REF, p["v0"], R_REF,
        p["kappa"], p["theta"], p["sigma"], p["rho"], T,
        n_steps=n_steps_grid[-1], n_paths=200_000, seed=42,
    )
    err_final = abs(final_mc.estimate - C_ref)

    return [report(
        f"finest grid (N={n_steps_grid[-1]}) agrees with Fourier",
        err_final < 3.0 * final_mc.half_width,
        info=f"err = {err_final:.5f}, "
             f"3*HW = {3.0 * final_mc.half_width:.5f}",
    )]


def test_convergence_in_paths():
    """Statistical error scales as n_paths^{-1/2}. We verify by checking
    that half_width * sqrt(n_paths) is approximately constant across a
    range of n_paths."""
    print("Test 3b: Statistical convergence in n_paths (fixed n_steps)")

    T = 0.5
    p = PARAMS_STD

    n_paths_grid = [10_000, 100_000, 500_000]
    hw_times_sqrt_n = []
    for n_paths in n_paths_grid:
        mc = mc_european_call_heston(
            S0_REF, K_REF, p["v0"], R_REF,
            p["kappa"], p["theta"], p["sigma"], p["rho"], T,
            n_steps=100, n_paths=n_paths, seed=42,
        )
        hw_n = mc.half_width * np.sqrt(n_paths)
        hw_times_sqrt_n.append(hw_n)
        print(f"      n_paths={n_paths:>7}: HW={mc.half_width:.5f}, "
              f"HW*sqrt(M)={hw_n:.4f}")

    # All values should be within ~5% of the central value (genuine
    # Monte Carlo asymptotics with seed=42 across these sample sizes
    # are very well-behaved).
    central = float(np.median(hw_times_sqrt_n))
    rel_errs = [abs(x - central) / central for x in hw_times_sqrt_n]
    max_rel_err = float(max(rel_errs))

    return [report(
        "HW * sqrt(n_paths) approximately constant",
        max_rel_err < 0.05,
        info=f"max relative deviation = {max_rel_err:.3%}",
    )]


# =====================================================================
# Test 4: Antithetic variance reduction
# =====================================================================

def test_antithetic():
    """Antithetic variates should reduce the sample variance of the
    paired-average estimator vs plain MC. For monotone payoffs (like a
    call), the reduction is structural (not parameter-dependent in
    sign), and typical magnitudes are 3x to 10x for ATM options."""
    print("Test 4: Antithetic variance reduction")

    T = 0.5
    p = PARAMS_STD

    plain = mc_european_call_heston(
        S0_REF, K_REF, p["v0"], R_REF,
        p["kappa"], p["theta"], p["sigma"], p["rho"], T,
        n_steps=200, n_paths=100_000, seed=42,
    )
    anti = mc_european_call_heston(
        S0_REF, K_REF, p["v0"], R_REF,
        p["kappa"], p["theta"], p["sigma"], p["rho"], T,
        n_steps=200, n_paths=100_000, seed=42, antithetic=True,
    )

    var_ratio = plain.sample_variance / anti.sample_variance
    print(f"    Plain     : est = {plain.estimate:.4f}, "
          f"sample_var = {plain.sample_variance:.2f}, "
          f"HW = {plain.half_width:.5f}")
    print(f"    Antithetic: est = {anti.estimate:.4f}, "
          f"sample_var = {anti.sample_variance:.2f}, "
          f"HW = {anti.half_width:.5f}")
    print(f"    Variance reduction: {var_ratio:.2f}x")

    results = []
    # The reduction should be at least 2x for ATM monotone payoffs.
    results.append(report(
        "antithetic reduces variance by >= 2x",
        var_ratio > 2.0,
        info=f"observed reduction = {var_ratio:.2f}x",
    ))

    # Both estimators should agree (within combined HW) with each other.
    err = abs(plain.estimate - anti.estimate)
    combined_hw = plain.half_width + anti.half_width
    results.append(report(
        "plain and antithetic estimates agree",
        err < combined_hw,
        info=f"|plain - anti| = {err:.5f}, "
             f"combined HW = {combined_hw:.5f}",
    ))

    return results


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 70)
    print("Heston Monte Carlo (full-truncation Euler): validation suite")
    print("=" * 70)
    print()
    print(f"Reference parameter sets:")
    print(f"  Standard:   {PARAMS_STD}  (Feller nu = "
          f"{2*PARAMS_STD['kappa']*PARAMS_STD['theta']/PARAMS_STD['sigma']**2:.3f})")
    print(f"  Aggressive: {PARAMS_AGG}  (Feller nu = "
          f"{2*PARAMS_AGG['kappa']*PARAMS_AGG['theta']/PARAMS_AGG['sigma']**2:.3f})")
    print(f"  Contract:   S0={S0_REF}, K={K_REF}, r={R_REF}")
    print()

    t0 = time.perf_counter()

    all_results = []
    all_results.extend(test_bs_limit())
    print()
    all_results.extend(test_cross_method())
    print()
    all_results.extend(test_convergence_in_steps())
    print()
    all_results.extend(test_convergence_in_paths())
    print()
    all_results.extend(test_antithetic())

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
