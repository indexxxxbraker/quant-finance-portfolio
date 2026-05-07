"""Validation suite for the Asian-option pricers (Phase 2 Block 5).

Implements four tests:

  Test 1: closed-form match for the geometric Asian. The IID
          estimator at n = 1e5 should agree with the closed-form
          price within 3 half-widths.

  Test 2: VRF for arithmetic Asian via geometric CV vs IID at the
          same budget. Predicted: VRF > 500 (typical > 1000).

  Test 3: empirical correlation between arithmetic and geometric
          payoffs. Predicted: rho > 0.999.

  Test 4: input validation.

Run from python/:
    python validate_mc_asian.py
"""

import math

import numpy as np

from quantlib.asian import (
    geometric_asian_call_closed_form,
    mc_asian_call_arithmetic_iid,
    mc_asian_call_geometric_iid,
    mc_asian_call_arithmetic_cv,
    _gbm_paths,
    _arithmetic_geometric_payoffs,
)


def _format_pass_fail(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


# Common parameters.
S0, K, r, sigma, T, N = 100.0, 100.0, 0.05, 0.20, 1.0, 50


# =====================================================================
# Test 1: closed-form match for geometric Asian
# =====================================================================

def test_geometric_closed_form() -> bool:
    print("\n" + "=" * 78)
    print("TEST 1: Closed-form match for geometric Asian")
    print("=" * 78)

    n = 100_000
    closed = geometric_asian_call_closed_form(S0, K, r, sigma, T, N)

    res = mc_asian_call_geometric_iid(S0, K, r, sigma, T, n,
                                       n_steps=N, seed=42)
    err = abs(res.estimate - closed)
    err_in_hw = err / res.half_width

    print(f"  Closed form (N={N})    : {closed:.6f}")
    print(f"  IID estimate (n={n}): {res.estimate:.6f}")
    print(f"  half-width            : {res.half_width:.6f}")
    print(f"  abs error             : {err:.6f}")
    print(f"  error / half-width    : {err_in_hw:.2f}")

    passed = err_in_hw < 3.0
    print(f"\n  Test 1: {_format_pass_fail(passed)}  (err < 3 hw)")
    return passed


# =====================================================================
# Test 2: VRF for arithmetic Asian via geometric CV
# =====================================================================

def test_vrf_arithmetic_cv() -> bool:
    print("\n" + "=" * 78)
    print("TEST 2: VRF for arithmetic Asian via geometric CV")
    print("        Predicted: VRF > 500")
    print("=" * 78)

    n = 100_000

    res_iid = mc_asian_call_arithmetic_iid(S0, K, r, sigma, T, n,
                                            n_steps=N, seed=42)
    res_cv  = mc_asian_call_arithmetic_cv(S0, K, r, sigma, T, n,
                                            n_steps=N, seed=42)

    print(f"  Arithmetic IID  (n={n}):")
    print(f"    estimate   = {res_iid.estimate:.6f}")
    print(f"    half-width = {res_iid.half_width:.6f}")
    print(f"\n  Arithmetic CV   (n={n}):")
    print(f"    estimate   = {res_cv.estimate:.6f}")
    print(f"    half-width = {res_cv.half_width:.6f}")

    vrf = (res_iid.half_width / res_cv.half_width) ** 2
    hw_ratio = res_iid.half_width / res_cv.half_width

    print(f"\n  VRF (= hw_iid^2 / hw_cv^2): {vrf:.0f}x")
    print(f"  Half-width ratio          : {hw_ratio:.1f}x")

    passed = vrf > 500
    print(f"\n  Test 2: {_format_pass_fail(passed)}  (VRF > 500)")
    return passed


# =====================================================================
# Test 3: empirical correlation between arithmetic and geometric
# =====================================================================

def test_empirical_correlation() -> bool:
    print("\n" + "=" * 78)
    print("TEST 3: Empirical correlation between arithmetic and geometric")
    print("        Predicted: rho > 0.999")
    print("=" * 78)

    n = 50_000
    rng = np.random.default_rng(42)
    paths = _gbm_paths(S0, r, sigma, T, n, N, rng)
    Pi_A, Pi_G = _arithmetic_geometric_payoffs(paths, K, r, T)

    # Pearson correlation.
    rho = float(np.corrcoef(Pi_A, Pi_G)[0, 1])
    print(f"  n_paths            : {n}")
    print(f"  empirical rho      : {rho:.6f}")
    print(f"  rho^2              : {rho ** 2:.6f}")
    print(f"  1 - rho^2          : {1 - rho ** 2:.6f}")
    print(f"  predicted VRF      : {1 / (1 - rho ** 2):.0f}x")

    passed = rho > 0.999
    print(f"\n  Test 3: {_format_pass_fail(passed)}  (rho > 0.999)")
    return passed


# =====================================================================
# Test 4: input validation
# =====================================================================

def test_input_validation() -> bool:
    print("\n" + "=" * 78)
    print("TEST 4: Input validation")
    print("=" * 78)

    pricers = [
        ("arithmetic_iid", mc_asian_call_arithmetic_iid),
        ("geometric_iid",  mc_asian_call_geometric_iid),
        ("arithmetic_cv",  mc_asian_call_arithmetic_cv),
    ]

    # Each call passes seed=42 to avoid the rng-resolution check.
    cases = [
        ("S<=0",            {"S": -1.0, "K": K, "r": r, "sigma": sigma, "T": T, "n_paths": 1000, "n_steps": N, "seed": 42}),
        ("K<=0",            {"S": S0, "K": 0.0, "r": r, "sigma": sigma, "T": T, "n_paths": 1000, "n_steps": N, "seed": 42}),
        ("sigma<=0",        {"S": S0, "K": K, "r": r, "sigma": -0.10, "T": T, "n_paths": 1000, "n_steps": N, "seed": 42}),
        ("T<=0",            {"S": S0, "K": K, "r": r, "sigma": sigma, "T": 0.0, "n_paths": 1000, "n_steps": N, "seed": 42}),
        ("n_paths < 2",     {"S": S0, "K": K, "r": r, "sigma": sigma, "T": T, "n_paths": 1, "n_steps": N, "seed": 42}),
        ("n_steps < 1",     {"S": S0, "K": K, "r": r, "sigma": sigma, "T": T, "n_paths": 1000, "n_steps": 0, "seed": 42}),
    ]

    all_pass = True
    for name, fn in pricers:
        print(f"\n  {name}:")
        for label, kwargs in cases:
            try:
                fn(**kwargs)
                print(f"    [{label}]: FAIL (no exception)")
                all_pass = False
            except (ValueError, TypeError):
                print(f"    [{label}]: PASS")
        # Negative r is admissible.
        try:
            fn(S=S0, K=K, r=-0.02, sigma=sigma, T=T, n_paths=1000,
               n_steps=N, seed=42)
            print(f"    [negative r is admissible]: PASS")
        except Exception as e:
            print(f"    [negative r is admissible]: FAIL ({type(e).__name__})")
            all_pass = False

    print(f"\n  Test 4 overall: {_format_pass_fail(all_pass)}")
    return all_pass


# =====================================================================
# Main
# =====================================================================

def main():
    print("\n" + "=" * 78)
    print("VALIDATION: Phase 2 Block 5 -- Asian options by Monte Carlo")
    print("=" * 78)
    print(f"\nParameters: S = {S0}, K = {K}, r = {r}, sigma = {sigma}, T = {T}, N = {N}")

    results = {
        "Test 1 (geometric closed-form)": test_geometric_closed_form(),
        "Test 2 (CV VRF)"               : test_vrf_arithmetic_cv(),
        "Test 3 (correlation)"          : test_empirical_correlation(),
        "Test 4 (input validation)"     : test_input_validation(),
    }

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for name, passed in results.items():
        print(f"  {name:38}: {_format_pass_fail(passed)}")

    overall = all(results.values())
    print("\n  " + ("ALL TESTS PASSED" if overall else "SOME TESTS FAILED"))

    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
