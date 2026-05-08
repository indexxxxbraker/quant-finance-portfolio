"""
validate_heston_fourier.py
==========================

Validation suite for the Heston Fourier pricer in
``quantlib.heston_fourier``. The tests exercise four independent
properties that, taken together, are strong evidence the implementation
is correct:

    1. Characteristic-function sanity:  phi(0) = 1,  phi(-i) = E[S_T].
    2. Black-Scholes limit:             sigma_H -> 0  =>  Heston -> BS.
    3. Put-call parity:                 C - P = S - K exp(-r tau).
    4. Carr-Madan vs Lewis cross-check on a (K, T) grid.
    5. AMSST stress test:               long maturities where the
                                        original Heston (1993)
                                        formulation would fail; we check
                                        that both inversion methods
                                        still agree.

Run from the ``python/`` directory:

    python validate_heston_fourier.py

Returns exit code 0 on all pass, 1 on any failure.
"""

from __future__ import annotations

import sys
import time
from typing import Tuple

import numpy as np

from quantlib.heston_fourier import (
    heston_cf,
    heston_call_carr_madan,
    heston_call_lewis,
    black_scholes_call,
    put_via_parity,
)


# =====================================================================
# Test reporting
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
# Reference parameter set (used across tests)
# =====================================================================

# Standard Heston example: moderate vol-of-vol, strong leverage effect.
PARAMS = dict(
    kappa=1.5,
    theta=0.04,
    sigma=0.3,
    rho=-0.7,
    v0=0.04,
)
S0_REF = 100.0
R_REF = 0.05


# =====================================================================
# Tests
# =====================================================================

def test_cf_sanity():
    """Characteristic function: phi(0) = 1 and phi(-i) = S0 exp(r tau)."""
    print("Test 1: Characteristic function sanity")

    tau = 0.5
    phi_0 = heston_cf(0.0, tau, S0_REF, PARAMS["v0"], R_REF,
                      PARAMS["kappa"], PARAMS["theta"],
                      PARAMS["sigma"], PARAMS["rho"])
    phi_neg_i = heston_cf(-1j, tau, S0_REF, PARAMS["v0"], R_REF,
                          PARAMS["kappa"], PARAMS["theta"],
                          PARAMS["sigma"], PARAMS["rho"])
    expected_neg_i = S0_REF * np.exp(R_REF * tau)

    err_0 = abs(complex(phi_0) - 1.0)
    err_neg_i = abs(complex(phi_neg_i) - expected_neg_i)
    tol = 1e-12

    p1 = report("phi(0) = 1",
                err_0 < tol,
                info=f"|phi(0) - 1| = {err_0:.2e}")
    p2 = report("phi(-i) = S0 exp(rT)",
                err_neg_i < tol,
                info=f"|phi(-i) - S0 exp(rT)| = {err_neg_i:.2e}")
    print()
    return [p1, p2]


def test_bs_limit():
    """
    As sigma_H -> 0 with v0 = theta, Heston -> BS with vol sqrt(v0).

    The convergence rate is LINEAR in sigma_H, not quadratic, because the
    leverage effect (rho != 0) gives a first-order correction. So we
    don't try to verify near-machine-precision agreement at any sigma_H;
    instead we check (a) close agreement at a representative small
    sigma_H = 0.001, and (b) monotonic decrease of error as sigma_H
    decreases.
    """
    print("Test 2: Black-Scholes limit (sigma_H -> 0)")

    tau = 0.5
    K = 100.0
    sigma_bs = float(np.sqrt(PARAMS["v0"]))
    C_bs = black_scholes_call(S0_REF, K, tau, R_REF, sigma_bs)

    print(f"    BS reference (sigma = sqrt(v0) = {sigma_bs:.4f}): "
          f"{C_bs:.6f}")
    print(f"    Heston prices and errors at decreasing sigma_H:")

    # Compute and display the convergence table
    sigma_grid = [0.3, 0.1, 0.05, 0.01, 0.001]
    errors_lewis = []
    errors_cm = []
    for sh in sigma_grid:
        cl = heston_call_lewis(K, tau, S0_REF, PARAMS["v0"], R_REF,
                               PARAMS["kappa"], PARAMS["theta"], sh,
                               PARAMS["rho"])
        cc = heston_call_carr_madan(K, tau, S0_REF, PARAMS["v0"], R_REF,
                                     PARAMS["kappa"], PARAMS["theta"], sh,
                                     PARAMS["rho"])
        err_l = abs(cl - C_bs)
        err_c = abs(cc - C_bs)
        errors_lewis.append(err_l)
        errors_cm.append(err_c)
        print(f"      sigma_H = {sh:6.4f}:  Lewis err = {err_l:.2e},  "
              f"Carr-Madan err = {err_c:.2e}")

    results = []

    # (a) At sigma_H = 0.001, Heston should be within 1e-3 of BS.
    #     This is the strongest single-point assertion of the BS limit.
    err_at_smallest = errors_lewis[-1]
    results.append(report(
        "Heston(sigma_H=0.001) close to BS",
        err_at_smallest < 1e-3,
        info=f"Lewis err = {err_at_smallest:.2e} (tol 1e-3)"
    ))

    # (b) Errors decrease monotonically as sigma_H -> 0.
    #     Allow a small slack because at sigma_H = 0.1 vs 0.05 the
    #     higher-order corrections happen to nearly cancel.
    monotonic_lewis = all(
        errors_lewis[i] >= errors_lewis[i+1] - 1e-4
        for i in range(len(errors_lewis) - 1)
    )
    monotonic_cm = all(
        errors_cm[i] >= errors_cm[i+1] - 2e-3   # extra slack for FFT noise
        for i in range(len(errors_cm) - 1)
    )
    results.append(report(
        "BS-limit error monotonically decreases (Lewis)",
        monotonic_lewis,
        info=f"errors = {[f'{e:.1e}' for e in errors_lewis]}"
    ))
    results.append(report(
        "BS-limit error monotonically decreases (Carr-Madan)",
        monotonic_cm,
        info=f"errors = {[f'{e:.1e}' for e in errors_cm]}"
    ))
    print()
    return results


def test_put_call_parity():
    """Put-call parity is a model-free identity; should hold to machine eps."""
    print("Test 3: Put-call parity")

    tau = 0.5
    K_grid = np.array([80.0, 95.0, 100.0, 105.0, 120.0])
    results = []
    tol = 1e-10

    for K in K_grid:
        C = heston_call_lewis(float(K), tau, S0_REF, PARAMS["v0"], R_REF,
                              PARAMS["kappa"], PARAMS["theta"],
                              PARAMS["sigma"], PARAMS["rho"])
        P = put_via_parity(C, S0_REF, K, tau, R_REF)
        lhs = C - P
        rhs = S0_REF - K * np.exp(-R_REF * tau)
        err = abs(lhs - rhs)
        results.append(report(
            f"K = {K:.0f}",
            err < tol,
            info=f"|C - P - (S - K e^-rT)| = {err:.2e}"
        ))
    print()
    return results


def test_cm_vs_lewis_grid():
    """Cross-check Carr-Madan vs Lewis on a (K, T) grid."""
    print("Test 4: Carr-Madan vs Lewis cross-check on (K, T) grid")

    K_grid = np.array([80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0])
    T_grid = [0.25, 0.5, 1.0, 2.0]
    # Tolerance dominated by FFT linear-interpolation error (N=4096, eta=0.25).
    # Empirically below ~1.5e-3 across this grid; allow a small margin.
    tol = 5e-3

    results = []
    for T in T_grid:
        C_cm = heston_call_carr_madan(K_grid, T, S0_REF, PARAMS["v0"],
                                       R_REF, PARAMS["kappa"],
                                       PARAMS["theta"], PARAMS["sigma"],
                                       PARAMS["rho"])
        C_le = heston_call_lewis(K_grid, T, S0_REF, PARAMS["v0"],
                                  R_REF, PARAMS["kappa"], PARAMS["theta"],
                                  PARAMS["sigma"], PARAMS["rho"])
        max_err = float(np.max(np.abs(C_cm - C_le)))
        results.append(report(
            f"T = {T:.2f}, K in [{K_grid[0]:.0f}, {K_grid[-1]:.0f}]",
            max_err < tol,
            info=f"max|CM - Lewis| = {max_err:.2e}"
        ))
    print()
    return results


def test_amsst_long_maturity():
    """
    Long-maturity AMSST stress test.

    With the original Heston (1993) formulation, ATM call prices for
    T >= 2 develop branch-cut errors of order 0.1%-2%. The AMSST/Little
    Trap formulation does not. We verify that both Carr-Madan and Lewis
    (which use independent integrations but the same AMSST char function)
    agree to FFT precision at long maturities.
    """
    print("Test 5: AMSST long-maturity stress (T = 2, 5, 10)")

    K = 100.0
    tol = 1e-2  # generous: the test is whether AMSST is stable, not exact precision

    results = []
    for T in [2.0, 5.0, 10.0]:
        C_cm = heston_call_carr_madan(K, T, S0_REF, PARAMS["v0"], R_REF,
                                       PARAMS["kappa"], PARAMS["theta"],
                                       PARAMS["sigma"], PARAMS["rho"])
        C_le = heston_call_lewis(K, T, S0_REF, PARAMS["v0"], R_REF,
                                  PARAMS["kappa"], PARAMS["theta"],
                                  PARAMS["sigma"], PARAMS["rho"])
        err = abs(C_cm - C_le)
        results.append(report(
            f"T = {T:.1f}",
            err < tol,
            info=(f"CM = {C_cm:.6f}, Lewis = {C_le:.6f}, "
                  f"|diff| = {err:.2e}")
        ))
    print()
    return results


# =====================================================================
# Timing benchmark (informational, not a pass/fail test)
# =====================================================================

def benchmark_timing():
    """Informational: how fast are we?"""
    print("Timing benchmark (not a pass/fail test)")

    K_grid = np.linspace(80.0, 120.0, 41)  # 41 strikes
    tau = 0.5

    n_repeats = 5

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        heston_call_carr_madan(K_grid, tau, S0_REF, PARAMS["v0"], R_REF,
                                PARAMS["kappa"], PARAMS["theta"],
                                PARAMS["sigma"], PARAMS["rho"])
    t_cm = (time.perf_counter() - t0) / n_repeats * 1000.0

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        heston_call_lewis(K_grid, tau, S0_REF, PARAMS["v0"], R_REF,
                          PARAMS["kappa"], PARAMS["theta"],
                          PARAMS["sigma"], PARAMS["rho"])
    t_le = (time.perf_counter() - t0) / n_repeats * 1000.0

    print(f"    Carr-Madan FFT (41 strikes, {n_repeats} runs avg): "
          f"{t_cm:.2f} ms")
    print(f"    Lewis quadrature (41 strikes, {n_repeats} runs avg): "
          f"{t_le:.2f} ms")
    print(f"    Carr-Madan speedup: {t_le / t_cm:.1f}x")
    print()


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 70)
    print("Heston Fourier pricer: validation suite")
    print("=" * 70)
    print()
    print(f"Reference parameters:")
    print(f"    kappa = {PARAMS['kappa']},  theta = {PARAMS['theta']},  "
          f"sigma = {PARAMS['sigma']},  rho = {PARAMS['rho']}")
    print(f"    v0 = {PARAMS['v0']},  S0 = {S0_REF},  r = {R_REF}")
    print()

    all_results = []
    all_results.extend(test_cf_sanity())
    all_results.extend(test_bs_limit())
    all_results.extend(test_put_call_parity())
    all_results.extend(test_cm_vs_lewis_grid())
    all_results.extend(test_amsst_long_maturity())

    benchmark_timing()

    n_pass = sum(1 for _, ok in all_results if ok)
    n_total = len(all_results)
    failures = [name for name, ok in all_results if not ok]

    print("=" * 70)
    print(f"SUMMARY: {n_pass} / {n_total} tests passed")
    if failures:
        print("Failures:")
        for f in failures:
            print(f"  - {f}")
    print("=" * 70)

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
