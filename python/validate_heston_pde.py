"""
validate_heston_pde.py
======================

Validation suite for the Heston PDE pricer in ``quantlib.heston_pde``.
Implements a four-pillar validation:

    1. Black-Scholes limit:     sigma -> 0 reduces Heston to BS.
    2. Cross-method vs Fourier: PDE matches Fourier across a panel of
                                 strikes and maturities.
    3. Spatial convergence:     halving (dX, dv) reduces error by ~4x
                                 (consistent with O(dx^2 + dv^2) order).
    4. Sanity vs QE:            PDE and QE Monte Carlo agree within
                                 the MC half-width.

The Fourier pricer of Block 2 is the bias-free ground truth.

Run from the ``python/`` directory:

    python validate_heston_pde.py

Returns exit code 0 on all pass, 1 on any failure.
"""

from __future__ import annotations

import sys
import time
from typing import Tuple

import numpy as np

from quantlib.heston_pde import heston_call_pde
from quantlib.heston_qe import mc_european_call_heston_qe
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

S0_REF = 100.0
K_REF = 100.0
R_REF = 0.05


# =====================================================================
# Test 1: Black-Scholes limit
# =====================================================================

def test_bs_limit():
    """As sigma -> 0 with v0 = theta, Heston PDE -> BS at constant vol."""
    print("Test 1: Black-Scholes limit (sigma -> 0)")

    T = 0.5
    sigma_bs = float(np.sqrt(PARAMS_STD["v0"]))
    C_bs = black_scholes_call(S0_REF, K_REF, T, R_REF, sigma_bs)
    print(f"    BS reference (sigma=sqrt(v0)={sigma_bs:.4f}): {C_bs:.6f}")

    p = dict(PARAMS_STD)
    p["sigma"] = 0.01

    price = heston_call_pde(
        S0_REF, K_REF, T,
        p["kappa"], p["theta"], p["sigma"], p["rho"], p["v0"], R_REF,
        N_X=200, N_v=100, N_tau=200,
    )
    err = abs(price - C_bs)

    # Tolerance: same logic as in Block 3, but here the PDE bias is
    # deterministic (~ 0.005 at this grid), so 0.05 is comfortable.
    return [report(
        "PDE (sigma=0.01) close to BS",
        err < 0.05,
        info=f"PDE = {price:.6f}, err = {err:.5f}",
    )]


# =====================================================================
# Test 2: Cross-method consistency vs Fourier
# =====================================================================

def test_cross_method():
    """For a panel of strikes and maturities, the PDE must agree with
    Fourier within reasonable tolerance. We use grid (200, 100, 200)
    where the spatial error is ~0.005 in absolute terms."""
    print("Test 2: Cross-method consistency (PDE vs Fourier)")

    p = PARAMS_STD
    results = []

    # We use a reasonably tight tolerance: 0.02 absolute. At grid
    # (200, 100, 200) the PDE typically agrees with Fourier to 3
    # decimals for ATM/ITM, less precise for far OTM where the payoff
    # is small in absolute terms.
    for K in [90.0, 100.0, 110.0]:
        for T in [0.25, 0.5, 1.0]:
            C_ref = heston_call_lewis(
                K, T, S0_REF, p["v0"], R_REF,
                p["kappa"], p["theta"], p["sigma"], p["rho"],
            )
            price = heston_call_pde(
                S0_REF, K, T,
                p["kappa"], p["theta"], p["sigma"], p["rho"], p["v0"], R_REF,
                N_X=200, N_v=100, N_tau=200,
            )
            err = abs(price - C_ref)
            tol = 0.02
            results.append(report(
                f"K={K}, T={T}",
                err < tol,
                info=f"PDE = {price:.4f}, Fourier = {C_ref:.4f}, "
                     f"err = {err:.4f}",
            ))
    return results


# =====================================================================
# Test 3: Spatial convergence (halving study)
# =====================================================================

def test_spatial_convergence():
    """Halve dX and dv (double N_X and N_v) and verify the error
    decreases by a factor close to 4, consistent with O(dx^2 + dv^2)
    spatial order. Time step is kept constant at a small value to make
    sure spatial errors dominate."""
    print("Test 3: Spatial convergence (halving N_X, N_v)")

    p = PARAMS_STD
    T = 0.5
    K = 100.0
    C_ref = heston_call_lewis(
        K, T, S0_REF, p["v0"], R_REF,
        p["kappa"], p["theta"], p["sigma"], p["rho"],
    )
    print(f"    Fourier reference: {C_ref:.6f}")

    # Use a generous N_tau so temporal error doesn't pollute the spatial study
    N_tau_fixed = 200

    grid_sizes = [(50, 25), (100, 50), (200, 100)]
    errors = []
    print(f"    {'N_X':>5} {'N_v':>5}  {'PDE':>10}  {'error':>10}  {'ratio':>8}")
    for (N_X, N_v) in grid_sizes:
        price = heston_call_pde(
            S0_REF, K, T,
            p["kappa"], p["theta"], p["sigma"], p["rho"], p["v0"], R_REF,
            N_X=N_X, N_v=N_v, N_tau=N_tau_fixed,
        )
        err = abs(price - C_ref)
        ratio_str = "—" if not errors else f"{errors[-1] / err:.2f}"
        print(f"    {N_X:>5} {N_v:>5}  {price:>10.6f}  {err:>10.6f}  "
              f"{ratio_str:>8}")
        errors.append(err)

    # Check: ratio of errors when halving should be close to 4 (with tolerance).
    # At small grids the convergence may not yet be in the asymptotic
    # regime, so we check the ratio at the second halving (medium-to-fine),
    # which should be closer to 4.
    ratio_fine = errors[1] / errors[2]
    return [report(
        "error halves by factor ~4 between N=(100,50) and N=(200,100)",
        2.5 < ratio_fine < 6.0,
        info=f"observed ratio = {ratio_fine:.2f} (expected ~4 for O(h^2))",
    )]


# =====================================================================
# Test 4: Sanity vs QE Monte Carlo
# =====================================================================

def test_pde_vs_qe():
    """Compare PDE against QE Monte Carlo as a third independent method.
    Both should agree with each other within the QE half-width plus the
    PDE discretisation error.

    This catches systematic biases that Fourier might not catch (e.g.,
    PDE BC error, or PDE convergence in a regime where Fourier is also
    misbehaving for some reason)."""
    print("Test 4: Sanity vs QE Monte Carlo")

    p = PARAMS_STD
    results = []
    for T in [0.25, 0.5, 1.0]:
        # PDE
        pde_price = heston_call_pde(
            S0_REF, K_REF, T,
            p["kappa"], p["theta"], p["sigma"], p["rho"], p["v0"], R_REF,
            N_X=200, N_v=100, N_tau=200,
        )
        # QE MC: enough paths/steps so HW is small relative to PDE error
        qe = mc_european_call_heston_qe(
            S0_REF, K_REF, p["v0"], R_REF,
            p["kappa"], p["theta"], p["sigma"], p["rho"], T,
            n_steps=50, n_paths=200_000, seed=42,
        )
        err = abs(pde_price - qe.estimate)
        # Combined tolerance: 3 * QE_HW + 0.02 (typical PDE error)
        tol = 3.0 * qe.half_width + 0.02
        results.append(report(
            f"T={T}",
            err < tol,
            info=f"PDE = {pde_price:.4f}, QE = {qe.estimate:.4f} "
                 f"(+/-{qe.half_width:.4f}), err = {err:.4f}, tol = {tol:.4f}",
        ))
    return results


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 70)
    print("Heston PDE (Douglas ADI): validation suite")
    print("=" * 70)
    print()
    print(f"Reference parameter set:")
    print(f"  Standard: {PARAMS_STD}  (Feller nu = "
          f"{2*PARAMS_STD['kappa']*PARAMS_STD['theta']/PARAMS_STD['sigma']**2:.3f})")
    print(f"  Contract: S0={S0_REF}, K={K_REF}, r={R_REF}")
    print()

    t0 = time.perf_counter()

    all_results = []
    all_results.extend(test_bs_limit())
    print()
    all_results.extend(test_cross_method())
    print()
    all_results.extend(test_spatial_convergence())
    print()
    all_results.extend(test_pde_vs_qe())

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
