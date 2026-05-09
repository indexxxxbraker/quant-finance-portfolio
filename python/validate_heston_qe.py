"""
validate_heston_qe.py
=====================

Validation suite for the Heston QE Monte Carlo pricer in
``quantlib.heston_qe``. Implements a five-pillar validation:

    1. Black-Scholes limit:       sigma -> 0 reduces Heston to BS.
    2. Cross-method consistency:  QE matches Fourier reference within
                                  statistical half-width.
    3. Statistical scaling:       half_width * sqrt(n_paths) approximately
                                  constant.
    4. Antithetic variance reduction.
    5. Bias comparison vs FT-Euler:  the central empirical claim of
                                     the block. QE should have bias
                                     substantially smaller than FT-Euler
                                     at the same (n_steps, n_paths),
                                     especially in the low-Feller
                                     regime.

The Fourier pricer of Block 2 is the ground truth.

Run from the ``python/`` directory:

    python validate_heston_qe.py

Returns exit code 0 on all pass, 1 on any failure.
"""

from __future__ import annotations

import sys
import time
from typing import Tuple

import numpy as np

from quantlib.heston_qe import mc_european_call_heston_qe
from quantlib.heston_mc import mc_european_call_heston   # FT-Euler baseline
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

PARAMS_STD = dict(
    kappa=1.5,
    theta=0.04,
    sigma=0.3,
    rho=-0.7,
    v0=0.04,
)

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
    """As sigma -> 0 with v_0 = theta, Heston QE -> BS at vol sqrt(v_0)."""
    print("Test 1: Black-Scholes limit (sigma -> 0)")

    T = 0.5
    sigma_bs = float(np.sqrt(PARAMS_STD["v0"]))
    C_bs = black_scholes_call(S0_REF, K_REF, T, R_REF, sigma_bs)
    print(f"    BS reference (sigma=sqrt(v0)={sigma_bs:.4f}): {C_bs:.6f}")

    p = dict(PARAMS_STD)
    p["sigma"] = 0.01
    result = mc_european_call_heston_qe(
        S0_REF, K_REF, p["v0"], R_REF,
        p["kappa"], p["theta"], p["sigma"], p["rho"], T,
        n_steps=100, n_paths=200_000, seed=42,
    )

    err = abs(result.estimate - C_bs)
    tol = 3.0 * result.half_width
    return [report(
        "QE (sigma=0.01) close to BS",
        err < tol,
        info=f"QE = {result.estimate:.6f}, err = {err:.5f}, "
             f"3*HW = {tol:.5f}",
    )]


# =====================================================================
# Test 2: Cross-method consistency vs Fourier
# =====================================================================

def test_cross_method():
    """QE estimates must agree with Fourier within half-widths."""
    print("Test 2: Cross-method consistency (QE vs Fourier)")

    results = []
    # QE has small bias even at modest n_steps, so we can use a relatively
    # small grid (n_steps=50) and rely on the statistical test.
    for params, label in [(PARAMS_STD, "standard"),
                            (PARAMS_AGG, "aggressive")]:
        for T in [0.25, 0.5, 1.0]:
            C_ref = heston_call_lewis(
                K_REF, T, S0_REF, params["v0"], R_REF,
                params["kappa"], params["theta"], params["sigma"],
                params["rho"],
            )
            mc = mc_european_call_heston_qe(
                S0_REF, K_REF, params["v0"], R_REF,
                params["kappa"], params["theta"], params["sigma"],
                params["rho"], T,
                n_steps=50, n_paths=200_000, seed=42,
            )
            err = abs(mc.estimate - C_ref)
            tol = 3.0 * mc.half_width
            results.append(report(
                f"{label}, T={T}",
                err < tol,
                info=f"QE = {mc.estimate:.4f}, "
                     f"Fourier = {C_ref:.4f}, "
                     f"err = {err:.4f}, tol = {tol:.4f}",
            ))
    return results


# =====================================================================
# Test 3: Statistical scaling
# =====================================================================

def test_statistical_scaling():
    """Half-width * sqrt(n_paths) should be approximately constant."""
    print("Test 3: Statistical convergence in n_paths (fixed n_steps)")

    T = 0.5
    p = PARAMS_STD

    n_paths_grid = [10_000, 100_000, 500_000]
    hw_times_sqrt_n = []
    for n_paths in n_paths_grid:
        mc = mc_european_call_heston_qe(
            S0_REF, K_REF, p["v0"], R_REF,
            p["kappa"], p["theta"], p["sigma"], p["rho"], T,
            n_steps=50, n_paths=n_paths, seed=42,
        )
        hw_n = mc.half_width * np.sqrt(n_paths)
        hw_times_sqrt_n.append(hw_n)
        print(f"      n_paths={n_paths:>7}: HW={mc.half_width:.5f}, "
              f"HW*sqrt(M)={hw_n:.4f}")

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
    """QE antithetic should reduce variance by >= 2x and remain
    statistically consistent with the plain estimator."""
    print("Test 4: Antithetic variance reduction")

    T = 0.5
    p = PARAMS_STD

    plain = mc_european_call_heston_qe(
        S0_REF, K_REF, p["v0"], R_REF,
        p["kappa"], p["theta"], p["sigma"], p["rho"], T,
        n_steps=50, n_paths=100_000, seed=42,
    )
    anti = mc_european_call_heston_qe(
        S0_REF, K_REF, p["v0"], R_REF,
        p["kappa"], p["theta"], p["sigma"], p["rho"], T,
        n_steps=50, n_paths=100_000, seed=42, antithetic=True,
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
    results.append(report(
        "antithetic reduces variance by >= 2x",
        var_ratio > 2.0,
        info=f"observed reduction = {var_ratio:.2f}x",
    ))
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
# Test 5: Bias comparison QE vs FT-Euler
# =====================================================================

def test_bias_vs_fte():
    """The central empirical claim of Block 4: QE has bias substantially
    smaller than full-truncation Euler at the same (n_steps, n_paths),
    most dramatically so in the low-Feller regime.

    To resolve the bias from statistical noise we average over multiple
    seeds with a large M. The expected ratio |bias_FTE / bias_QE| in
    aggressive parameters is in the range 5-50x at the n_steps tested;
    we assert >= 5x with comfortable margin.
    """
    print("Test 5: Bias comparison QE vs FT-Euler")

    p = PARAMS_AGG
    T = 0.5
    C_ref = heston_call_lewis(
        K_REF, T, S0_REF, p["v0"], R_REF,
        p["kappa"], p["theta"], p["sigma"], p["rho"],
    )
    print(f"    Aggressive params (Feller="
          f"{2*p['kappa']*p['theta']/p['sigma']**2:.3f})")
    print(f"    Fourier reference: {C_ref:.6f}")
    print(f"    Averaging across 5 seeds, n_steps=50, n_paths=200000:")

    qe_estimates = []
    fte_estimates = []
    for seed in range(5):
        r_qe = mc_european_call_heston_qe(
            S0_REF, K_REF, p["v0"], R_REF,
            p["kappa"], p["theta"], p["sigma"], p["rho"], T,
            n_steps=50, n_paths=200_000, seed=seed,
        )
        r_fte = mc_european_call_heston(
            S0_REF, K_REF, p["v0"], R_REF,
            p["kappa"], p["theta"], p["sigma"], p["rho"], T,
            n_steps=50, n_paths=200_000, seed=seed,
        )
        qe_estimates.append(r_qe.estimate)
        fte_estimates.append(r_fte.estimate)

    qe_bias = np.mean(qe_estimates) - C_ref
    fte_bias = np.mean(fte_estimates) - C_ref
    bias_ratio = abs(fte_bias) / max(abs(qe_bias), 1e-10)

    print(f"      QE bias  : {qe_bias:+.6f}")
    print(f"      FTE bias : {fte_bias:+.6f}")
    print(f"      |FTE / QE|: {bias_ratio:.2f}x")

    results = []
    # Assert QE has clearly smaller bias than FTE.
    results.append(report(
        "QE bias < FTE bias (aggressive regime)",
        abs(qe_bias) < abs(fte_bias),
        info=f"|QE|={abs(qe_bias):.5f}, |FTE|={abs(fte_bias):.5f}",
    ))
    # Assert the reduction is at least 5x (conservative; typical 20-70x).
    results.append(report(
        "bias reduction >= 5x in aggressive regime",
        bias_ratio >= 5.0,
        info=f"observed ratio = {bias_ratio:.2f}x (expected 5-50x)",
    ))
    return results


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 70)
    print("Heston QE (Andersen 2008): validation suite")
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
    all_results.extend(test_statistical_scaling())
    print()
    all_results.extend(test_antithetic())
    print()
    all_results.extend(test_bias_vs_fte())

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
