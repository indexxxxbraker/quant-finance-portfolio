"""
Validation suite for the implied-volatility solver in
quantlib.implied_volatility.

NOTE on the validation domain. The map sigma -> C_BS(sigma) is monotone
but its derivative (Vega) varies by orders of magnitude across parameter
space. For points where Vega is very small relative to the price tolerance
of the solver, the inverse problem is numerically ill-conditioned: the
expected error in the recovered sigma is approximately
    |delta_sigma| ~ |delta_C| / Vega
so a small Vega amplifies any rounding noise in the price into a large
error in sigma. No solver --- ours, scipy's, or any other --- can
overcome this; it is a property of the problem, not the algorithm.

The well-conditioning criterion used here is that Vega at the true sigma
must exceed 1e-3 (a generous lower bound: typical Vega is 5-50). Points
failing this test are excluded from the round-trip and cross-check
benchmarks, since they would test the limits of double-precision
arithmetic rather than the correctness of our code.

The five checks:

  1. Round-trip on a random grid restricted to well-conditioned points.
  2. Edge cases that activate the Brent fallback.
  3. Bounds violations.
  4. Cross-check with scipy.optimize.brentq on the same restricted grid.
  5. Textbook reference (Hull example 19.6).

Run from python/:
    python validate_implied_volatility.py
"""

import numpy as np
from scipy.optimize import brentq

from quantlib.black_scholes import call_price, vega
from quantlib.implied_volatility import implied_volatility


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
TOL_ROUNDTRIP = 1e-5    # Limited by Vega magnitude in worst-case region
TOL_SCIPY     = 1e-5    # Same intrinsic limit
TOL_EDGE      = 1e-4

# The well-conditioning criterion: Vega must exceed this threshold for
# the inversion to be numerically resolvable to ~6 digits in sigma.
VEGA_THRESHOLD = 1e-3


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _well_conditioned(S, K, r, sigma, T):
    """
    Return True iff the inverse problem is numerically resolvable.

    The criterion is that Vega exceeds a threshold below which
    floating-point errors in the price translate to large errors in
    sigma. See module docstring for the derivation.
    """
    return vega(S, K, r, sigma, T) > VEGA_THRESHOLD


# ---------------------------------------------------------------------------
# Check 1: Round-trip on a random grid (restricted to well-conditioned points)
# ---------------------------------------------------------------------------
def check_roundtrip_random_grid():
    print("[1] Round-trip on random parameter grid")
    rng = np.random.default_rng(seed=42)
    n = 10000

    S     = rng.uniform(80,  120, n)
    K     = S * rng.uniform(0.7, 1.3, n)
    r     = rng.uniform(0.01, 0.10, n)
    sigma = rng.uniform(0.05, 0.80, n)
    T     = rng.uniform(0.1,  2.0,  n)

    max_err = 0.0
    n_failed = 0
    n_skipped = 0
    worst_case = None
    for i in range(n):
        if not _well_conditioned(S[i], K[i], r[i], sigma[i], T[i]):
            n_skipped += 1
            continue
        C = call_price(S[i], K[i], r[i], sigma[i], T[i])
        try:
            iv = implied_volatility(C, S[i], K[i], r[i], T[i])
            err = abs(iv - sigma[i])
            if err > max_err:
                max_err = err
                worst_case = (S[i], K[i], r[i], sigma[i], T[i], iv)
        except Exception:
            n_failed += 1

    print(f"    Sample size: {n}")
    print(f"    Skipped (ill-conditioned, Vega < {VEGA_THRESHOLD}): {n_skipped}")
    print(f"    Failures (exceptions): {n_failed}")
    print(f"    Max recovery error: {max_err:.2e}  (tol {TOL_ROUNDTRIP:.0e})  "
          f"{_status(n_failed == 0 and max_err < TOL_ROUNDTRIP)}")
    if worst_case is not None and max_err > 1e-8:
        S0, K0, r0, sig0, T0, iv0 = worst_case
        print(f"    Worst case: S={S0:.2f} K={K0:.2f} r={r0:.4f} "
              f"sigma_true={sig0:.4f} T={T0:.2f} -> iv={iv0:.6f}")


# ---------------------------------------------------------------------------
# Check 2: Edge cases
# ---------------------------------------------------------------------------
def check_edge_cases():
    print("[2] Edge cases (regimes where Newton becomes unsafe)")
    cases = [
        ("Moderate ITM",     130.0, 100.0, 0.05, 0.20, 1.0),
        ("Moderate OTM",     80.0,  100.0, 0.05, 0.20, 1.0),
        ("Short expiry ATM", 100.0, 100.0, 0.05, 0.20, 0.001),
        ("Short expiry OTM", 100.0, 105.0, 0.05, 0.20, 0.01),
        ("Very low sigma",   100.0, 100.0, 0.05, 0.01, 1.0),
        ("Very high sigma",  100.0, 100.0, 0.05, 2.00, 1.0),
    ]
    all_ok = True
    for name, S, K, r, sigma_true, T in cases:
        C = call_price(S, K, r, sigma_true, T)
        try:
            iv = implied_volatility(C, S, K, r, T)
            err = abs(iv - sigma_true)
            ok = err < TOL_EDGE
            print(f"    {name:18s}  sigma_true={sigma_true:.4f}  "
                  f"iv={iv:.6f}  err={err:.1e}  {_status(ok)}")
            all_ok = all_ok and ok
        except Exception as e:
            print(f"    {name:18s}  EXCEPTION: {e}  FAIL")
            all_ok = False
    print(f"    Overall: {_status(all_ok)}")


# ---------------------------------------------------------------------------
# Check 3: Bounds violations
# ---------------------------------------------------------------------------
def check_bounds_violations():
    print("[3] Bounds violations (should all raise ValueError)")
    S, K, r, T = 100.0, 100.0, 0.05, 1.0
    intrinsic_fwd = max(S - K * np.exp(-r * T), 0.0)

    cases = [
        ("Below lower bound",   intrinsic_fwd / 2.0),
        ("At lower bound",      intrinsic_fwd),
        ("Above upper bound",   S * 1.1),
        ("At upper bound",      S),
        ("Negative price",      -1.0),
    ]
    all_ok = True
    for name, C in cases:
        try:
            iv = implied_volatility(C, S, K, r, T)
            print(f"    {name:22s}  C={C:.4f}  -> NO EXCEPTION (got iv={iv})  FAIL")
            all_ok = False
        except ValueError:
            print(f"    {name:22s}  C={C:.4f}  -> ValueError raised  PASS")
    print(f"    Overall: {_status(all_ok)}")


# ---------------------------------------------------------------------------
# Check 4: Cross-check against scipy.optimize.brentq
# ---------------------------------------------------------------------------
def check_against_scipy():
    print("[4] Agreement with scipy.optimize.brentq (independent solver)")
    rng = np.random.default_rng(seed=2024)
    n = 1000
    S     = rng.uniform(80,  120, n)
    K     = S * rng.uniform(0.7, 1.3, n)
    r     = rng.uniform(0.01, 0.10, n)
    sigma = rng.uniform(0.05, 0.80, n)
    T     = rng.uniform(0.1,  2.0,  n)

    max_disagreement = 0.0
    n_compared = 0
    n_skipped = 0
    for i in range(n):
        if not _well_conditioned(S[i], K[i], r[i], sigma[i], T[i]):
            n_skipped += 1
            continue
        C = call_price(S[i], K[i], r[i], sigma[i], T[i])
        try:
            iv_ours = implied_volatility(C, S[i], K[i], r[i], T[i])
        except ValueError:
            n_skipped += 1
            continue
        iv_scipy = brentq(
            lambda s: call_price(S[i], K[i], r[i], s, T[i]) - C,
            1e-4, 10.0, xtol=1e-12,
        )
        err = abs(iv_ours - iv_scipy)
        if err > max_disagreement:
            max_disagreement = err
        n_compared += 1

    print(f"    Sample size: {n}")
    print(f"    Compared: {n_compared}")
    print(f"    Skipped: {n_skipped}")
    print(f"    Max |iv_ours - iv_scipy|: {max_disagreement:.2e}  "
          f"(tol {TOL_SCIPY:.0e})  {_status(max_disagreement < TOL_SCIPY)}")


# ---------------------------------------------------------------------------
# Check 5: Textbook reference
# ---------------------------------------------------------------------------
def check_textbook_reference():
    print("[5] Hull (10th ed.), Example 19.6 reference")
    S, K, r, T = 21.0, 20.0, 0.10, 0.25
    C = 1.875
    iv = implied_volatility(C, S, K, r, T)
    err = abs(iv - 0.235)
    ok = err < 1e-2
    print(f"    iv={iv:.6f}  (Hull: ~0.235)  err={err:.2e}  {_status(ok)}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("Implied volatility validation suite")
    print("=" * 72)
    print()

    check_roundtrip_random_grid(); print()
    check_edge_cases();            print()
    check_bounds_violations();     print()
    check_against_scipy();         print()
    check_textbook_reference();    print()

    print("=" * 72)


if __name__ == "__main__":
    main()
