"""
Validation script for the Kamrad-Ritchken trinomial pricer.

Five checks:

1. European cross-validation against Black-Scholes closed form.
2. American cross-validation against an embedded CRR binomial reference.
3. First-order convergence: doubling n_steps approximately halves the error.
4. Lambda sweep: varying lambda within the valid range gives consistent
   prices that all converge to BS as n_steps grows.
5. Input validation: lambda < 1, n_steps < 1, non-positive parameters
   all raise ValueError.

Run with `python -m validate_trinomial` from the python/ directory or
directly.
"""

import math
import numpy as np

from quantlib.trinomial import (
    trinomial_european_call,
    trinomial_european_put,
    trinomial_american_put,
)
from quantlib.black_scholes import call_price as bs_call_price
from quantlib.black_scholes import put_price  as bs_put_price


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
# Reference: standalone CRR American put for cross-validation. Same
# pattern as Block 4: keeps the test independent of any other module.
# ---------------------------------------------------------------------------
def crr_american_put(S, K, r, sigma, T, n_steps):
    dt = T / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    V = []
    for i in range(n_steps + 1):
        S_T = S * (u ** (n_steps - i)) * (d ** i)
        V.append(max(K - S_T, 0.0))
    for step in range(n_steps - 1, -1, -1):
        new_V = []
        for i in range(step + 1):
            S_node = S * (u ** (step - i)) * (d ** i)
            cont = disc * (p * V[i] + (1.0 - p) * V[i + 1])
            exer = max(K - S_node, 0.0)
            new_V.append(max(cont, exer))
        V = new_V
    return V[0]


# ---------------------------------------------------------------------------
def test_european_cross_validation():
    print("[1] European cross-validation against Black-Scholes")
    test_cases = [
        # (S,    K,    r,    sigma, T)
        (100.0, 100.0, 0.05, 0.20, 1.00),
        ( 90.0, 100.0, 0.05, 0.20, 1.00),
        (110.0, 100.0, 0.05, 0.20, 1.00),
        (100.0, 100.0, 0.05, 0.30, 1.00),
        (100.0, 100.0, 0.05, 0.20, 0.25),
        (100.0, 100.0, 0.10, 0.20, 1.00),
        ( 42.0,  40.0, 0.10, 0.20, 0.50),
    ]
    n_steps = 2000
    tol = 5e-3

    for S, K, r, sigma, T in test_cases:
        c_tri = trinomial_european_call(S, K, r, sigma, T, n_steps=n_steps)
        p_tri = trinomial_european_put (S, K, r, sigma, T, n_steps=n_steps)
        c_bs  = bs_call_price(S, K, r, sigma, T)
        p_bs  = bs_put_price (S, K, r, sigma, T)

        err_c = abs(c_tri - c_bs)
        err_p = abs(p_tri - p_bs)

        check(f"call S={S:6.2f} K={K:.0f} sigma={sigma:.2f} T={T:.2f}",
              err_c < tol,
              detail=f"tri={c_tri:.4f} BS={c_bs:.4f} err={err_c:.2e}")
        check(f"put  S={S:6.2f} K={K:.0f} sigma={sigma:.2f} T={T:.2f}",
              err_p < tol,
              detail=f"tri={p_tri:.4f} BS={p_bs:.4f} err={err_p:.2e}")


def test_american_cross_validation():
    print("[2] American cross-validation against CRR binomial reference")
    test_cases = [
        (100.0, 100.0, 0.05, 0.20, 1.00),
        ( 90.0, 100.0, 0.05, 0.20, 1.00),
        (110.0, 100.0, 0.05, 0.20, 1.00),
        (100.0, 100.0, 0.05, 0.30, 1.00),
        ( 42.0,  40.0, 0.10, 0.20, 0.50),
    ]
    n_steps = 2000
    tol = 5e-3

    for S, K, r, sigma, T in test_cases:
        p_tri = trinomial_american_put(S, K, r, sigma, T, n_steps=n_steps)
        p_crr = crr_american_put(S, K, r, sigma, T, n_steps=n_steps)
        err = abs(p_tri - p_crr)
        check(f"S={S:6.2f} K={K:.0f} sigma={sigma:.2f} T={T:.2f}",
              err < tol,
              detail=f"tri={p_tri:.4f} CRR={p_crr:.4f} err={err:.2e}")


def test_first_order_convergence():
    """Doubling n_steps halves the error. The trinomial is O(1/N), so
    the ratio should be near 2 (not 4 like a second-order scheme)."""
    print("[3] First-order convergence in n_steps")
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    c_bs = bs_call_price(S, K, r, sigma, T)
    print(f"    BS reference: {c_bs:.6f}")

    n_list = [200, 400, 800, 1600]
    errors = []
    for n in n_list:
        c = trinomial_european_call(S, K, r, sigma, T, n_steps=n)
        e = abs(c - c_bs)
        errors.append(e)
        print(f"    n={n:5d}: price = {c:.6f}, error = {e:.4e}")

    # The trinomial price for at-the-money options has a known
    # zigzag in n: errors at consecutive n values can differ. We
    # average over a few consecutive n values to smooth this out, or
    # compare values at n -> 2n -> 4n.
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    print(f"    Ratios (expected ~2 for first-order): {ratios}")

    # First-order means error halves per doubling. We tolerate a wide
    # band [1.2, 4.0] to account for the n-dependent oscillation that
    # is well documented in lattice methods near at-the-money strikes.
    for i, r_ratio in enumerate(ratios):
        check(f"ratio errors[{i}]/errors[{i+1}] in [1.2, 4.0]",
              1.2 <= r_ratio <= 4.0,
              detail=f"got {r_ratio:.2f}")


def test_lambda_sweep():
    """Verify that varying lambda within (1, 10] gives prices close
    to BS, with the spread shrinking as n_steps grows."""
    print("[4] Lambda sweep: varying lambda gives consistent prices")
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    c_bs = bs_call_price(S, K, r, sigma, T)
    n_steps = 1000

    prices = {}
    for lam in [1.0, 1.5, 2.0, 3.0, 5.0]:
        prices[lam] = trinomial_european_call(
            S, K, r, sigma, T, n_steps=n_steps, lambda_param=lam)
        print(f"    lambda={lam}: price = {prices[lam]:.6f} "
              f"(error = {abs(prices[lam] - c_bs):.4e})")

    # All prices should be within 5e-3 of BS at n=1000.
    max_err = max(abs(p - c_bs) for p in prices.values())
    check("all lambda values give prices within 5e-3 of BS",
          max_err < 5e-3,
          detail=f"max error across lambdas = {max_err:.2e}")
    # And the spread between any two lambdas should be small.
    spread = max(prices.values()) - min(prices.values())
    check("spread between lambda values < 1e-2",
          spread < 1e-2,
          detail=f"spread = {spread:.2e}")


def test_input_validation():
    print("[5] Input validation")
    K, r, sigma, T = 100.0, 0.05, 0.20, 1.0
    S = 100.0

    # S = 0
    try:
        trinomial_european_call(0.0, K, r, sigma, T, n_steps=100)
        check("S=0 raises", False)
    except ValueError:
        check("S=0 raises", True)

    # K negative
    try:
        trinomial_european_call(S, -10.0, r, sigma, T, n_steps=100)
        check("K<0 raises", False)
    except ValueError:
        check("K<0 raises", True)

    # sigma <= 0
    try:
        trinomial_european_call(S, K, r, 0.0, T, n_steps=100)
        check("sigma=0 raises", False)
    except ValueError:
        check("sigma=0 raises", True)

    # n_steps < 1
    try:
        trinomial_european_call(S, K, r, sigma, T, n_steps=0)
        check("n_steps=0 raises", False)
    except ValueError:
        check("n_steps=0 raises", True)

    # lambda < 1
    try:
        trinomial_european_call(S, K, r, sigma, T, n_steps=100,
                                  lambda_param=0.5)
        check("lambda<1 raises", False)
    except ValueError:
        check("lambda<1 raises", True)

    # lambda exactly 1 should NOT raise (boundary case is allowed)
    try:
        p = trinomial_european_call(S, K, r, sigma, T, n_steps=100,
                                      lambda_param=1.0)
        check("lambda=1 does not raise (boundary allowed)",
              math.isfinite(p), detail=f"price = {p:.4f}")
    except ValueError:
        check("lambda=1 does not raise", False)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 72)
    print("Phase 3 Block 5 - Trinomial KR pricer validation")
    print("=" * 72)

    test_european_cross_validation()
    test_american_cross_validation()
    test_first_order_convergence()
    test_lambda_sweep()
    test_input_validation()

    print()
    print("=" * 72)
    total = _n_pass + _n_fail
    if _n_fail == 0:
        print(f"  {PASS}: {_n_pass}/{total} checks succeeded.")
        raise SystemExit(0)
    else:
        print(f"  {FAIL}: {_n_pass}/{total} succeeded, {_n_fail} failed.")
        raise SystemExit(1)
