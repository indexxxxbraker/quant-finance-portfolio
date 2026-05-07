"""
Validation script for the BTCS finite-difference pricer.

Five checks:

1. Cross-validation against Black-Scholes closed-form on a parameter grid.
2. Quadratic convergence under balanced refinement (N -> 2N, M -> 4M).
3. First-order convergence in time at fixed (large) N (M -> 2M).
4. Put-call parity.
5. Unconditional stability: BTCS at very large dtau produces a finite,
   sensible price (although imprecise). FTCS at the same configuration
   would diverge.

Exit code 0 on success, 1 on failure. Run from python/ as
    python -m validate_btcs
or directly.
"""

import numpy as np

from quantlib.black_scholes import call_price as bs_call_price
from quantlib.black_scholes import put_price  as bs_put_price
from quantlib.btcs import btcs_european_call, btcs_european_put


PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

_n_pass = 0
_n_fail = 0


def check(label, condition, *, detail=""):
    global _n_pass, _n_fail
    tag = PASS if condition else FAIL
    print(f"  [{tag}] {label}" + (f"   ({detail})" if detail else ""))
    if condition:
        _n_pass += 1
    else:
        _n_fail += 1


# ---------------------------------------------------------------------------
def test_cross_validation():
    print("[1] Cross-validation against Black-Scholes closed form")
    test_cases = [
        # (S,    K,    r,    sigma, T,    N,   M)
        (100.0, 100.0, 0.05, 0.20, 1.00, 200, 800),
        ( 90.0, 100.0, 0.05, 0.20, 1.00, 200, 800),
        (110.0, 100.0, 0.05, 0.20, 1.00, 200, 800),
        (100.0, 100.0, 0.05, 0.30, 1.00, 200, 1800),
        (100.0, 100.0, 0.05, 0.20, 0.25, 200, 800),
        (100.0, 100.0, 0.10, 0.20, 1.00, 200, 800),
        (100.0, 100.0, 0.00, 0.20, 1.00, 200, 800),
        ( 42.0,  40.0, 0.10, 0.20, 0.50, 200, 800),
    ]
    tol = 5e-3

    for S, K, r, sigma, T, N, M in test_cases:
        c_btcs = btcs_european_call(S, K, r, sigma, T, N=N, M=M)
        p_btcs = btcs_european_put (S, K, r, sigma, T, N=N, M=M)
        c_bs   = bs_call_price(S, K, r, sigma, T)
        p_bs   = bs_put_price (S, K, r, sigma, T)

        err_c = abs(c_btcs - c_bs)
        err_p = abs(p_btcs - p_bs)

        check(f"call S={S:6.2f} K={K:.0f} sigma={sigma:.2f} T={T:.2f}",
              err_c < tol,
              detail=f"BTCS={c_btcs:.4f} BS={c_bs:.4f} err={err_c:.2e}")
        check(f"put  S={S:6.2f} K={K:.0f} sigma={sigma:.2f} T={T:.2f}",
              err_p < tol,
              detail=f"BTCS={p_btcs:.4f} BS={p_bs:.4f} err={err_p:.2e}")


def test_quadratic_convergence():
    """N -> 2N, M -> 4M: error decreases by ~4 (O(dx^2 + dtau) with
    dtau ~ dx^2)."""
    print("[2] Quadratic convergence under balanced refinement")
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    c_bs = bs_call_price(S, K, r, sigma, T)

    levels = [(100, 200), (200, 800), (400, 3200)]
    errors = []
    for N, M in levels:
        c = btcs_european_call(S, K, r, sigma, T, N=N, M=M)
        e = abs(c - c_bs)
        errors.append(e)
        print(f"    N={N:4d}, M={M:5d}: error = {e:.4e}")

    ratio_01 = errors[0] / errors[1]
    ratio_12 = errors[1] / errors[2]
    print(f"    Ratio level0/level1 = {ratio_01:.2f}  (expected ~4)")
    print(f"    Ratio level1/level2 = {ratio_12:.2f}  (expected ~4)")

    check("quadratic convergence ratio level0/level1 in [3.0, 5.5]",
          3.0 <= ratio_01 <= 5.5, detail=f"got {ratio_01:.2f}")
    check("quadratic convergence ratio level1/level2 in [3.0, 5.5]",
          3.0 <= ratio_12 <= 5.5, detail=f"got {ratio_12:.2f}")


def test_first_order_in_time():
    """Fixed large N, M -> 2M: error decreases by ~2 (O(dtau)
    dominates because spatial error is already saturated at large N)."""
    print("[3] First-order convergence in time at fixed N")
    # Large N so spatial error is below time error.
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    c_bs = bs_call_price(S, K, r, sigma, T)

    N = 800
    Ms = [10, 20, 40]
    errors = []
    for M in Ms:
        c = btcs_european_call(S, K, r, sigma, T, N=N, M=M)
        e = abs(c - c_bs)
        errors.append(e)
        print(f"    N={N}, M={M:3d}: dtau={T/M:.4f}, error = {e:.4e}")

    ratio_01 = errors[0] / errors[1]
    ratio_12 = errors[1] / errors[2]
    print(f"    Ratio M=10/M=20  = {ratio_01:.2f}  (expected ~2)")
    print(f"    Ratio M=20/M=40  = {ratio_12:.2f}  (expected ~2)")

    check("first-order time ratio M10/M20 in [1.5, 2.5]",
          1.5 <= ratio_01 <= 2.5, detail=f"got {ratio_01:.2f}")
    check("first-order time ratio M20/M40 in [1.5, 2.5]",
          1.5 <= ratio_12 <= 2.5, detail=f"got {ratio_12:.2f}")


def test_put_call_parity():
    print("[4] Put-call parity")
    K, r, sigma, T = 100.0, 0.05, 0.20, 1.0
    N, M = 200, 800
    spots = [85.0, 95.0, 100.0, 105.0, 115.0]

    diffs = []
    for S in spots:
        c = btcs_european_call(S, K, r, sigma, T, N=N, M=M)
        p = btcs_european_put (S, K, r, sigma, T, N=N, M=M)
        rhs = S - K * np.exp(-r * T)
        diffs.append(abs((c - p) - rhs))

    max_diff = max(diffs)
    check("parity holds for all spots",
          max_diff < 1e-2, detail=f"max |C - P - (S - Ke^-rT)| = {max_diff:.2e}")


def test_unconditional_stability():
    """The signature test of BTCS: very large dtau (well beyond what
    FTCS could handle) produces a finite, plausible price.
    """
    print("[5] Unconditional stability: large-dtau test")
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    c_bs = bs_call_price(S, K, r, sigma, T)

    # M=10 gives dtau=0.1. With dx ~ 0.008 this would be alpha ~ 312,
    # 600x the FTCS stability bound. FTCS would produce numbers of
    # order 10^200 (if it didn't overflow first).
    c_btcs = btcs_european_call(S, K, r, sigma, T, N=200, M=10)
    err = abs(c_btcs - c_bs)
    print(f"    BTCS M=10:  c = {c_btcs:.4f}  (BS = {c_bs:.4f}, err = {err:.4f})")

    check("M=10 produces a finite price",
          np.isfinite(c_btcs))
    check("M=10 price is within 5% of BS",
          err / c_bs < 0.05, detail=f"relative error = {err/c_bs:.2%}")

    # Even more extreme: M=2 (dtau = 0.5).
    c_extreme = btcs_european_call(S, K, r, sigma, T, N=200, M=2)
    err_extreme = abs(c_extreme - c_bs)
    print(f"    BTCS M=2:   c = {c_extreme:.4f}  (err = {err_extreme:.4f})")
    check("M=2 still finite (no explosion)",
          np.isfinite(c_extreme))
    # No precision claim at M=2 — this is just to confirm BTCS doesn't blow up.


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 72)
    print("Phase 3 Block 2 - BTCS pricer validation")
    print("=" * 72)

    test_cross_validation()
    test_quadratic_convergence()
    test_first_order_in_time()
    test_put_call_parity()
    test_unconditional_stability()

    print()
    print("=" * 72)
    total = _n_pass + _n_fail
    if _n_fail == 0:
        print(f"  {PASS}: {_n_pass}/{total} checks succeeded.")
        raise SystemExit(0)
    else:
        print(f"  {FAIL}: {_n_pass}/{total} succeeded, {_n_fail} failed.")
        raise SystemExit(1)
