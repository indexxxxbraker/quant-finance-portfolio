"""
validate_heston_calibration.py
==============================

Validation suite for the Heston calibration pipeline in
``quantlib.heston_calibration``. Implements three pillars:

    1. Round-trip test:        synthesize prices from known parameters,
                               then calibrate. The recovered parameters
                               should match the truth to machine
                               precision (since the optimiser is using
                               the same Fourier pricer that generated
                               the data).
    2. Residual analysis:      after calibration, the per-quote
                               residuals should be small and unbiased
                               (no systematic structure).
    3. Parameter stability:    perturbing the market data slightly
                               should produce a small change in the
                               calibrated parameters. Sudden jumps
                               indicate the optimiser is hitting local
                               minima.

The first two pillars are validation of the calibration machinery; the
third is validation of the well-posedness of the optimisation.

Run from the ``python/`` directory:

    python validate_heston_calibration.py

Returns exit code 0 on all pass, 1 on any failure.
"""

from __future__ import annotations

import sys
import time
from typing import Tuple

import numpy as np

from quantlib.heston_calibration import (
    calibrate_heston,
    implied_vol_bs,
)
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
# Reference parameters
# =====================================================================

TRUTH = {"kappa": 1.5, "theta": 0.04, "sigma": 0.3,
          "rho": -0.7, "v0": 0.04}
S0_REF = 100.0
R_REF = 0.05


def _make_synthetic_surface(params, S0, r, strikes, maturities):
    """Generate a synthetic call surface from given Heston params."""
    surface = []
    for K in strikes:
        for T in maturities:
            C = heston_call_lewis(K, T, S0, params["v0"], r,
                                    params["kappa"], params["theta"],
                                    params["sigma"], params["rho"])
            surface.append({"K": K, "T": T, "C_market": C})
    return surface


# =====================================================================
# Test 1: Round-trip
# =====================================================================

def test_round_trip():
    """Calibrate from synthetic data and recover the truth parameters
    to high precision. This is the cleanest possible test of the
    calibration machinery."""
    print("Test 1: Round-trip (synthetic data with known truth)")

    strikes = [90.0, 100.0, 110.0]
    maturities = [0.25, 0.5, 1.0]
    market = _make_synthetic_surface(TRUTH, S0_REF, R_REF,
                                       strikes, maturities)

    initial = {"kappa": 1.0, "theta": 0.05, "sigma": 0.5,
                "rho": -0.3, "v0": 0.05}

    result = calibrate_heston(market, S0_REF, R_REF,
                                initial_guess=initial)

    print(f"    Synthesised {len(market)} quotes from truth: {TRUTH}")
    print(f"    Initial guess: {initial}")
    print(f"    Recovered    : {{")
    for k, v in result.params.items():
        print(f"      {k!r}: {v:.10f},")
    print(f"    }}")
    print(f"    RMSE: {result.rmse:.6e}, n_iter: {result.n_iter}")

    # Per-parameter recovery to high precision (10^-6)
    results = []
    for name in ["kappa", "theta", "sigma", "rho", "v0"]:
        err = abs(result.params[name] - TRUTH[name])
        results.append(report(
            f"recovered {name} to 10^-6",
            err < 1e-6,
            info=f"err = {err:.2e}",
        ))
    # Aggregate RMSE check
    results.append(report(
        "RMSE below 10^-6",
        result.rmse < 1e-6,
        info=f"RMSE = {result.rmse:.2e}",
    ))
    return results


# =====================================================================
# Test 2: Residual analysis
# =====================================================================

def test_residual_analysis():
    """After calibration, residuals should be small (already verified
    by the RMSE in Test 1) AND unbiased: no systematic pattern across
    K or T. We check that mean(residuals) is close to zero and that
    the largest residual is comparable to the smallest, indicating no
    one quote dominates."""
    print("Test 2: Residual analysis")

    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    maturities = [0.25, 0.5, 1.0, 2.0]
    market = _make_synthetic_surface(TRUTH, S0_REF, R_REF,
                                       strikes, maturities)

    initial = {"kappa": 1.0, "theta": 0.05, "sigma": 0.5,
                "rho": -0.3, "v0": 0.05}
    result = calibrate_heston(market, S0_REF, R_REF,
                                initial_guess=initial)

    residuals = result.residuals
    mean_resid = float(np.mean(residuals))
    max_abs_resid = float(np.max(np.abs(residuals)))
    median_abs_resid = float(np.median(np.abs(residuals)))

    print(f"    {len(market)} observations, residual stats:")
    print(f"      mean    = {mean_resid:+.4e}")
    print(f"      median  = {median_abs_resid:.4e}")
    print(f"      max abs = {max_abs_resid:.4e}")

    results = []
    # Mean residual should be near zero (no overall bias).
    results.append(report(
        "mean residual near zero (unbiased)",
        abs(mean_resid) < 1e-6,
        info=f"mean = {mean_resid:+.2e}",
    ))
    # Max residual should not be a huge multiple of median (no
    # systematic structure where one quote dominates the fit).
    if median_abs_resid > 0:
        ratio = max_abs_resid / median_abs_resid
        results.append(report(
            "max/median residual ratio bounded (no outlier)",
            ratio < 100.0,  # generous; with synthetic data should be ~1
            info=f"ratio = {ratio:.1f}",
        ))
    else:
        results.append(report(
            "max/median residual ratio bounded (no outlier)",
            True,
            info="all residuals at machine precision",
        ))
    return results


# =====================================================================
# Test 3: Parameter stability under perturbation
# =====================================================================

def test_parameter_stability():
    """Perturb the market prices by a small relative amount (1e-4) and
    verify the recovered parameters change by a similarly small amount.
    Discontinuous jumps would indicate the optimiser is hitting local
    minima or that the inverse problem is severely ill-conditioned."""
    print("Test 3: Parameter stability under data perturbation")

    strikes = [90.0, 100.0, 110.0]
    maturities = [0.25, 0.5, 1.0]
    market_clean = _make_synthetic_surface(TRUTH, S0_REF, R_REF,
                                              strikes, maturities)

    initial = {"kappa": 1.0, "theta": 0.05, "sigma": 0.5,
                "rho": -0.3, "v0": 0.05}

    # Baseline calibration
    cal_clean = calibrate_heston(market_clean, S0_REF, R_REF,
                                    initial_guess=initial)

    # Perturbation seeds: 5 different small random perturbations
    rng = np.random.default_rng(seed=0)
    perturbed_results = []
    eps_rel = 1e-4
    for trial in range(5):
        market_perturbed = []
        for d in market_clean:
            # Multiplicative perturbation; preserves no-arbitrage bounds
            factor = 1.0 + eps_rel * rng.standard_normal()
            market_perturbed.append({
                "K": d["K"], "T": d["T"],
                "C_market": d["C_market"] * factor,
            })
        cal = calibrate_heston(market_perturbed, S0_REF, R_REF,
                                  initial_guess=initial)
        perturbed_results.append(cal)

    # Compute parameter stability: for each parameter, the standard
    # deviation across the 5 perturbed runs should be small (a few
    # times the perturbation level), as expected for a well-posed
    # optimisation.
    print(f"    Baseline params: {{")
    for k, v in cal_clean.params.items():
        print(f"      {k!r}: {v:.6f},")
    print(f"    }}")
    print(f"    {len(perturbed_results)} perturbed runs (eps_rel={eps_rel}):")
    print(f"    {'param':>6} {'mean':>12} {'std':>12} {'max dev':>12}")

    results = []
    for name in ["kappa", "theta", "sigma", "rho", "v0"]:
        vals = [r.params[name] for r in perturbed_results]
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        max_dev = max(abs(v - cal_clean.params[name]) for v in vals)
        print(f"    {name:>6} {mean:>12.6f} {std:>12.4e} {max_dev:>12.4e}")
        # Each parameter should not change more than 0.05 (absolute) under
        # a 1e-4 relative perturbation. This is a generous threshold;
        # the actual deviation for the standard params is much smaller.
        results.append(report(
            f"{name} stable under perturbation",
            max_dev < 0.05,
            info=f"max deviation = {max_dev:.2e}",
        ))
    return results


# =====================================================================
# Test 4: Implied volatility inversion
# =====================================================================

def test_iv_inversion():
    """The IV inversion is a building block of the calibrator. Verify
    it inverts BS exactly across a panel of strikes/maturities/vols."""
    print("Test 4: Implied volatility inversion")

    test_cases = [
        # (sigma_truth, K, T)
        (0.10, 100.0, 0.25),
        (0.20, 100.0, 0.5),
        (0.30, 100.0, 1.0),
        (0.20, 80.0, 0.5),   # ITM
        (0.20, 120.0, 0.5),  # OTM
        (0.50, 100.0, 0.25), # high vol
    ]
    results = []
    print(f"    {'sigma_truth':>12} {'K':>6} {'T':>6} {'recovered':>12} {'err':>10}")
    all_ok = True
    for sigma_truth, K, T in test_cases:
        C = black_scholes_call(S0_REF, K, T, R_REF, sigma_truth)
        sigma_inv = implied_vol_bs(C, K, T, S0_REF, R_REF)
        err = abs(sigma_inv - sigma_truth)
        ok = err < 1e-8
        print(f"    {sigma_truth:>12.3f} {K:>6.0f} {T:>6.2f} "
              f"{sigma_inv:>12.6f} {err:>10.2e}")
        all_ok = all_ok and ok

    results.append(report(
        f"IV inversion accurate to 10^-8 across {len(test_cases)} cases",
        all_ok,
        info="see table above for details",
    ))
    return results


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 70)
    print("Heston calibration: validation suite")
    print("=" * 70)
    print()
    print(f"Truth parameters: {TRUTH}")
    print(f"S0 = {S0_REF}, r = {R_REF}")
    print(f"Feller parameter (truth): "
          f"{2*TRUTH['kappa']*TRUTH['theta']/TRUTH['sigma']**2:.3f}")
    print()

    t0 = time.perf_counter()

    all_results = []
    all_results.extend(test_round_trip())
    print()
    all_results.extend(test_residual_analysis())
    print()
    all_results.extend(test_parameter_stability())
    print()
    all_results.extend(test_iv_inversion())

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
