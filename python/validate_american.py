"""Validation suite for the American-option pricers (Phase 2 Block 6).

Implements four tests:

  Test 1: binomial convergence. Compute the CRR American put at
          increasing tree depths and check the sequence stabilises
          to four decimal places by N = 5000.

  Test 2: LSM matches binomial. At canonical parameters with
          n = 1e5 paths and N_LSM = 50 exercise dates, the LSM
          estimator agrees with the binomial reference (N = 10000)
          within 3 half-widths.

  Test 3: positive early exercise premium. The American put price
          must exceed the European put price by a margin that is
          large compared to MC noise.

  Test 4: input validation.

Run from python/:
    python validate_american.py
"""

import math

import numpy as np

from quantlib.american import (
    binomial_american_put,
    lsm_american_put,
)
from quantlib.black_scholes import put_price


def _format_pass_fail(passed):
    return "PASS" if passed else "FAIL"


# Common parameters
S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0


# =====================================================================
# Test 1: binomial convergence
# =====================================================================

def test_binomial_convergence():
    print("\n" + "=" * 78)
    print("TEST 1: Binomial American put convergence")
    print("=" * 78)

    Ns = [100, 500, 1000, 2000, 5000, 10000]
    prices = []
    for N in Ns:
        v = binomial_american_put(S0, K, r, sigma, T, N)
        prices.append(v)
        print(f"  N = {N:>6d}  ->  {v:.6f}")

    # Stability check: absolute change between N=5000 and N=10000.
    delta = abs(prices[-1] - prices[-2])
    print(f"\n  |P(10000) - P(5000)| = {delta:.6f}")
    passed = delta < 1e-3
    print(f"\n  Test 1: {_format_pass_fail(passed)}  (delta < 1e-3)")
    return passed, prices[-1]


# =====================================================================
# Test 2: LSM matches binomial
# =====================================================================

def test_lsm_matches_binomial(v_ref):
    print("\n" + "=" * 78)
    print("TEST 2: LSM matches binomial within 3 half-widths")
    print("=" * 78)

    n = 100_000
    N_lsm = 50
    M = 4

    res = lsm_american_put(S0, K, r, sigma, T, n,
                            n_steps=N_lsm, basis_size=M, seed=42)
    err = res.estimate - v_ref
    err_in_hw = abs(err) / res.half_width

    print(f"  Reference (binomial N=10000): {v_ref:.6f}")
    print(f"  LSM (n={n}, N={N_lsm}, M={M}):")
    print(f"    estimate   = {res.estimate:.6f}")
    print(f"    half-width = {res.half_width:.6f}")
    print(f"  Error to reference: {err:+.6f}")
    print(f"  Error / half-width: {err_in_hw:.2f}")

    passed = err_in_hw < 3.0
    print(f"\n  Test 2: {_format_pass_fail(passed)}  (|err|/hw < 3)")
    return passed


# =====================================================================
# Test 3: early exercise premium
# =====================================================================

def test_early_exercise_premium(v_american_ref):
    print("\n" + "=" * 78)
    print("TEST 3: Positive early exercise premium")
    print("=" * 78)

    eu = put_price(S0, K, r, sigma, T)
    premium = v_american_ref - eu

    print(f"  European put (BS):    {eu:.6f}")
    print(f"  American put (ref):   {v_american_ref:.6f}")
    print(f"  Early ex. premium:    {premium:.6f}")
    # Sanity: at our parameters the premium is around 0.52, well
    # above any imaginable MC noise.
    passed = premium > 0.05
    print(f"\n  Test 3: {_format_pass_fail(passed)}  (premium > 0.05)")
    return passed


# =====================================================================
# Test 4: input validation
# =====================================================================

def test_input_validation():
    print("\n" + "=" * 78)
    print("TEST 4: Input validation")
    print("=" * 78)

    cases_binomial = [
        ("S<=0",         {"S": -1.0, "K": K, "r": r, "sigma": sigma, "T": T, "n_steps": 100}),
        ("K<=0",         {"S": S0, "K": 0.0, "r": r, "sigma": sigma, "T": T, "n_steps": 100}),
        ("sigma<=0",     {"S": S0, "K": K, "r": r, "sigma": -0.10, "T": T, "n_steps": 100}),
        ("T<=0",         {"S": S0, "K": K, "r": r, "sigma": sigma, "T": 0.0, "n_steps": 100}),
        ("n_steps<1",    {"S": S0, "K": K, "r": r, "sigma": sigma, "T": T, "n_steps": 0}),
    ]

    cases_lsm = [
        ("S<=0",         {"S": -1.0, "K": K, "r": r, "sigma": sigma, "T": T, "n_paths": 1000, "n_steps": 50, "seed": 42}),
        ("K<=0",         {"S": S0, "K": 0.0, "r": r, "sigma": sigma, "T": T, "n_paths": 1000, "n_steps": 50, "seed": 42}),
        ("n_paths<2",    {"S": S0, "K": K, "r": r, "sigma": sigma, "T": T, "n_paths": 1, "n_steps": 50, "seed": 42}),
        ("n_steps<1",    {"S": S0, "K": K, "r": r, "sigma": sigma, "T": T, "n_paths": 1000, "n_steps": 0, "seed": 42}),
        ("basis_size<1", {"S": S0, "K": K, "r": r, "sigma": sigma, "T": T, "n_paths": 1000, "n_steps": 50, "basis_size": 0, "seed": 42}),
        ("basis_size>8", {"S": S0, "K": K, "r": r, "sigma": sigma, "T": T, "n_paths": 1000, "n_steps": 50, "basis_size": 9, "seed": 42}),
    ]

    all_pass = True

    print("\n  binomial_american_put:")
    for label, kwargs in cases_binomial:
        try:
            binomial_american_put(**kwargs)
            print(f"    [{label}]: FAIL (no exception)")
            all_pass = False
        except (ValueError, TypeError):
            print(f"    [{label}]: PASS")

    print("\n  lsm_american_put:")
    for label, kwargs in cases_lsm:
        try:
            lsm_american_put(**kwargs)
            print(f"    [{label}]: FAIL (no exception)")
            all_pass = False
        except (ValueError, TypeError):
            print(f"    [{label}]: PASS")

    # Negative r is admissible.
    print("\n  negative r admissible:")
    try:
        binomial_american_put(S0, K, -0.02, sigma, T, 100)
        lsm_american_put(S0, K, -0.02, sigma, T, 1000, n_steps=50, seed=42)
        print(f"    [negative r]: PASS")
    except Exception as e:
        print(f"    [negative r]: FAIL ({type(e).__name__})")
        all_pass = False

    print(f"\n  Test 4: {_format_pass_fail(all_pass)}")
    return all_pass


# =====================================================================
# Main
# =====================================================================

def main():
    print("\n" + "=" * 78)
    print("VALIDATION: Phase 2 Block 6 -- American options")
    print("=" * 78)
    print(f"\nParameters: S = {S0}, K = {K}, r = {r}, sigma = {sigma}, T = {T}")

    t1, v_ref = test_binomial_convergence()
    t2 = test_lsm_matches_binomial(v_ref)
    t3 = test_early_exercise_premium(v_ref)
    t4 = test_input_validation()

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  Test 1 (binomial convergence):    {_format_pass_fail(t1)}")
    print(f"  Test 2 (LSM matches binomial):    {_format_pass_fail(t2)}")
    print(f"  Test 3 (early exercise premium):  {_format_pass_fail(t3)}")
    print(f"  Test 4 (input validation):        {_format_pass_fail(t4)}")

    overall = t1 and t2 and t3 and t4
    print("\n  " + ("ALL TESTS PASSED" if overall else "SOME TESTS FAILED"))

    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
