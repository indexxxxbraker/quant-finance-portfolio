"""Validation suite for the Monte Carlo Greeks (Phase 2 Block 4).

Implements four tests:

  Test 1: BS coherence for all 7 estimators (3 methods x 2 Greeks =
          Delta + Vega, plus bump-only Gamma). Each estimate should
          fall within 3 half-widths of its closed-form BS counterpart.

  Test 2: Variance ranking for Delta. Predicted: pathwise <= bump < LR.

  Test 3: Variance ranking for Vega. Predicted: pathwise <= bump < LR.

  Test 4: Input validation: bad inputs should raise.

Run from python/:
    python validate_greeks_mc.py
"""

import math

import numpy as np

from quantlib.black_scholes import (
    call_delta, gamma, vega,
)
from quantlib.greeks import (
    delta_bump, delta_pathwise, delta_lr,
    vega_bump,  vega_pathwise,  vega_lr,
    gamma_bump,
)


def _format_pass_fail(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


# Common parameters.
S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0


# =====================================================================
# Test 1: BS coherence
# =====================================================================

def test_bs_coherence() -> bool:
    print("\n" + "=" * 78)
    print("TEST 1: BS coherence for all Greek estimators")
    print("=" * 78)

    n = 100_000

    bs_delta = call_delta(S0, K, r, sigma, T)
    bs_vega  = vega(S0, K, r, sigma, T)
    bs_gamma = gamma(S0, K, r, sigma, T)

    print(f"  BS reference Greeks:")
    print(f"    Delta = {bs_delta:.6f}")
    print(f"    Vega  = {bs_vega:.6f}")
    print(f"    Gamma = {bs_gamma:.6f}")

    cases = [
        ("Delta bump",     delta_bump,     bs_delta),
        ("Delta pathwise", delta_pathwise, bs_delta),
        ("Delta LR",       delta_lr,       bs_delta),
        ("Vega bump",      vega_bump,      bs_vega),
        ("Vega pathwise",  vega_pathwise,  bs_vega),
        ("Vega LR",        vega_lr,        bs_vega),
        ("Gamma bump",     gamma_bump,     bs_gamma),
    ]

    all_pass = True
    print(f"\n  {'estimator':<16}  {'estimate':>12}  {'BS':>12}  {'hw':>10}  {'within 3hw'}")
    print(f"  {'-' * 16}  {'-' * 12}  {'-' * 12}  {'-' * 10}  {'-' * 10}")
    for name, fn, bs_val in cases:
        result = fn(S0, K, r, sigma, T, n, seed=42)
        within = abs(result.estimate - bs_val) <= 3.0 * result.half_width
        passed_str = _format_pass_fail(within)
        all_pass = all_pass and within
        print(f"  {name:<16}  {result.estimate:>12.6f}  {bs_val:>12.6f}  "
              f"{result.half_width:>10.6f}  {passed_str}")

    print(f"\n  Test 1 overall: {_format_pass_fail(all_pass)}")
    return all_pass


# =====================================================================
# Test 2: Variance ranking Delta
# =====================================================================

def test_variance_ranking_delta() -> bool:
    print("\n" + "=" * 78)
    print("TEST 2: Variance ranking for Delta")
    print("        Predicted: pathwise <= bump < LR")
    print("=" * 78)

    n = 100_000

    res_bump     = delta_bump(S0, K, r, sigma, T, n, seed=11)
    res_pathwise = delta_pathwise(S0, K, r, sigma, T, n, seed=11)
    res_lr       = delta_lr(S0, K, r, sigma, T, n, seed=11)

    print(f"  Bump      : hw = {res_bump.half_width:.6f}, "
          f"sample_var = {res_bump.sample_variance:.6f}")
    print(f"  Pathwise  : hw = {res_pathwise.half_width:.6f}, "
          f"sample_var = {res_pathwise.sample_variance:.6f}")
    print(f"  LR        : hw = {res_lr.half_width:.6f}, "
          f"sample_var = {res_lr.sample_variance:.6f}")

    print(f"\n  Half-width ratios:")
    print(f"    LR / pathwise = {res_lr.half_width / res_pathwise.half_width:.2f}x")
    print(f"    LR / bump     = {res_lr.half_width / res_bump.half_width:.2f}x")
    print(f"    bump / pathwise = {res_bump.half_width / res_pathwise.half_width:.2f}x")

    # Acceptance: LR has more variance than pathwise (with margin),
    # bump is roughly comparable to pathwise.
    pass_lr_pathwise = (res_lr.half_width > 1.5 * res_pathwise.half_width)
    pass_bump_pathwise = (0.7 * res_pathwise.half_width
                           <= res_bump.half_width
                           <= 1.5 * res_pathwise.half_width)

    overall = pass_lr_pathwise and pass_bump_pathwise
    print(f"\n  Test 2 (LR > 1.5x pathwise)         : {_format_pass_fail(pass_lr_pathwise)}")
    print(f"  Test 2 (bump comparable to pathwise): {_format_pass_fail(pass_bump_pathwise)}")
    print(f"  Test 2 overall: {_format_pass_fail(overall)}")
    return overall


# =====================================================================
# Test 3: Variance ranking Vega
# =====================================================================

def test_variance_ranking_vega() -> bool:
    print("\n" + "=" * 78)
    print("TEST 3: Variance ranking for Vega")
    print("        Predicted: pathwise <= bump < LR")
    print("=" * 78)

    n = 100_000

    res_bump     = vega_bump(S0, K, r, sigma, T, n, seed=11)
    res_pathwise = vega_pathwise(S0, K, r, sigma, T, n, seed=11)
    res_lr       = vega_lr(S0, K, r, sigma, T, n, seed=11)

    print(f"  Bump      : hw = {res_bump.half_width:.6f}")
    print(f"  Pathwise  : hw = {res_pathwise.half_width:.6f}")
    print(f"  LR        : hw = {res_lr.half_width:.6f}")

    print(f"\n  Half-width ratios:")
    print(f"    LR / pathwise   = {res_lr.half_width / res_pathwise.half_width:.2f}x")
    print(f"    LR / bump       = {res_lr.half_width / res_bump.half_width:.2f}x")
    print(f"    bump / pathwise = {res_bump.half_width / res_pathwise.half_width:.2f}x")

    pass_lr_pathwise = (res_lr.half_width > 1.5 * res_pathwise.half_width)
    pass_bump_pathwise = (0.7 * res_pathwise.half_width
                           <= res_bump.half_width
                           <= 1.5 * res_pathwise.half_width)

    overall = pass_lr_pathwise and pass_bump_pathwise
    print(f"\n  Test 3 (LR > 1.5x pathwise)         : {_format_pass_fail(pass_lr_pathwise)}")
    print(f"  Test 3 (bump comparable to pathwise): {_format_pass_fail(pass_bump_pathwise)}")
    print(f"  Test 3 overall: {_format_pass_fail(overall)}")
    return overall


# =====================================================================
# Test 4: Input validation
# =====================================================================

def test_input_validation() -> bool:
    print("\n" + "=" * 78)
    print("TEST 4: Input validation")
    print("=" * 78)

    # Sample one estimator from each method.
    pricers = [
        ("delta_bump",     delta_bump),
        ("delta_pathwise", delta_pathwise),
        ("delta_lr",       delta_lr),
        ("gamma_bump",     gamma_bump),
    ]

    # Each call passes seed=42 to ensure _resolve_rng does not raise
    # for an unrelated reason (its own check for both seed and rng
    # being None).
    cases = [
        ("S must be positive",         {"S": -1.0, "K": K, "r": r, "sigma": sigma, "T": T, "n_paths": 1000, "seed": 42}),
        ("K must be positive",         {"S": S0, "K": 0.0, "r": r, "sigma": sigma, "T": T, "n_paths": 1000, "seed": 42}),
        ("sigma must be positive",     {"S": S0, "K": K, "r": r, "sigma": -0.10, "T": T, "n_paths": 1000, "seed": 42}),
        ("T must be positive",         {"S": S0, "K": K, "r": r, "sigma": sigma, "T": 0.0, "n_paths": 1000, "seed": 42}),
        ("n_paths must be at least 2", {"S": S0, "K": K, "r": r, "sigma": sigma, "T": T, "n_paths": 1, "seed": 42}),
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
            fn(S=S0, K=K, r=-0.02, sigma=sigma, T=T, n_paths=1000, seed=42)
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
    print("VALIDATION: Phase 2 Block 4 -- Greeks by Monte Carlo")
    print("=" * 78)
    print(f"\nParameters: S = {S0}, K = {K}, r = {r}, sigma = {sigma}, T = {T}")

    results = {
        "Test 1 (BS coherence)"             : test_bs_coherence(),
        "Test 2 (Delta variance ranking)"   : test_variance_ranking_delta(),
        "Test 3 (Vega variance ranking)"    : test_variance_ranking_vega(),
        "Test 4 (input validation)"         : test_input_validation(),
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
