"""
Validation suite for the Black-Scholes Greeks.

Three independent angles:

  1. Finite differences. The closed-form Greek must agree with the numerical
     derivative of the price computed via central differences. Probes
     pointwise correctness across all five Greeks for both calls and puts.

  2. Black-Scholes PDE residual. The combination
        Theta + 0.5*sigma^2*S^2*Gamma + r*S*Delta - r*C
     must vanish identically. This is a structural test: it does not check
     any individual Greek but tests that the whole tuple satisfies the
     no-arbitrage relation.

  3. Vega-Gamma identity:  Vega = S^2 * sigma * T * Gamma.
     Algebraic identity derived from the closed forms; must hold to machine
     precision.

Run from python/:
    python validate_greeks.py
"""

import numpy as np

from quantlib.black_scholes import (
    call_price, put_price,
    call_delta, put_delta, gamma, vega,
    call_theta, put_theta, call_rho, put_rho,
)


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
TOL_FD       = 1e-6     # Central differences are O(h^2) with h=1e-5.
TOL_PDE      = 1e-10    # PDE residual at machine precision (FP noise only).
TOL_IDENTITY = 1e-12    # Vega-Gamma identity is algebraic.


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


# ---------------------------------------------------------------------------
# Check 1: Finite-difference verification
# ---------------------------------------------------------------------------
def check_finite_differences():
    """For each Greek, central-difference the price and compare."""
    print("[1] Finite-difference verification of all Greeks")

    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    h = 1e-5

    # Bumps for each parameter.
    cases = [
        # (name, analytic, fn_numerical)
        ("Call Delta",
         call_delta(S, K, r, sigma, T),
         (call_price(S + h, K, r, sigma, T) - call_price(S - h, K, r, sigma, T)) / (2 * h)),
        ("Put Delta",
         put_delta(S, K, r, sigma, T),
         (put_price(S + h, K, r, sigma, T) - put_price(S - h, K, r, sigma, T)) / (2 * h)),
        ("Gamma",
         gamma(S, K, r, sigma, T),
         (call_price(S + h, K, r, sigma, T) - 2 * call_price(S, K, r, sigma, T) + call_price(S - h, K, r, sigma, T)) / (h ** 2)),
        ("Vega",
         vega(S, K, r, sigma, T),
         (call_price(S, K, r, sigma + h, T) - call_price(S, K, r, sigma - h, T)) / (2 * h)),
        # Theta = dC/dt = -dC/dT, so we bump T and flip the sign.
        ("Call Theta",
         call_theta(S, K, r, sigma, T),
         -(call_price(S, K, r, sigma, T + h) - call_price(S, K, r, sigma, T - h)) / (2 * h)),
        ("Put Theta",
         put_theta(S, K, r, sigma, T),
         -(put_price(S, K, r, sigma, T + h) - put_price(S, K, r, sigma, T - h)) / (2 * h)),
        ("Call Rho",
         call_rho(S, K, r, sigma, T),
         (call_price(S, K, r + h, sigma, T) - call_price(S, K, r - h, sigma, T)) / (2 * h)),
        ("Put Rho",
         put_rho(S, K, r, sigma, T),
         (put_price(S, K, r + h, sigma, T) - put_price(S, K, r - h, sigma, T)) / (2 * h)),
    ]

    for name, analytic, numerical in cases:
        err = abs(analytic - numerical)
        ok = err < TOL_FD
        print(f"    {name:11s}: analytic={analytic:+.6e}  numerical={numerical:+.6e}  err={err:.1e}  {_status(ok)}")


# ---------------------------------------------------------------------------
# Check 2: Black-Scholes PDE residual
# ---------------------------------------------------------------------------
def check_pde_residual():
    """
    BS PDE:  Theta + 0.5 * sigma^2 * S^2 * Gamma + r*S*Delta - r*C = 0.

    This must hold for any point in parameter space. We probe a random grid.
    """
    print("[2] Black-Scholes PDE residual on random grid")
    rng = np.random.default_rng(seed=7)
    n = 10000
    S     = rng.uniform(50,  150, n)
    K     = rng.uniform(50,  150, n)
    r     = rng.uniform(0.01, 0.10, n)
    sigma = rng.uniform(0.10, 0.50, n)
    T     = rng.uniform(0.1,  2.0,  n)

    C  = call_price(S, K, r, sigma, T)
    Th = call_theta(S, K, r, sigma, T)
    Ga = gamma     (S, K, r, sigma, T)
    De = call_delta(S, K, r, sigma, T)

    residual = Th + 0.5 * sigma ** 2 * S ** 2 * Ga + r * S * De - r * C
    max_res = float(np.abs(residual).max())

    print(f"    Sample size: {n}")
    print(f"    Max |residual|: {max_res:.2e}  (tol {TOL_PDE:.0e})  {_status(max_res < TOL_PDE)}")

    # Also verify on the put side. The BS PDE applies to any portfolio of
    # contingent claims on S, so the same identity must hold replacing
    # (C, Theta_C, Delta_C) by (P, Theta_P, Delta_P), with Gamma unchanged.
    P  = put_price(S, K, r, sigma, T)
    ThP = put_theta(S, K, r, sigma, T)
    DeP = put_delta(S, K, r, sigma, T)
    residual_put = ThP + 0.5 * sigma ** 2 * S ** 2 * Ga + r * S * DeP - r * P
    max_res_put = float(np.abs(residual_put).max())
    print(f"    Max |residual| (put): {max_res_put:.2e}  (tol {TOL_PDE:.0e})  {_status(max_res_put < TOL_PDE)}")


# ---------------------------------------------------------------------------
# Check 3: Vega-Gamma identity
# ---------------------------------------------------------------------------
def check_vega_gamma():
    """
    Vega = S^2 * sigma * T * Gamma.

    Pure algebraic identity from the closed forms; should hold at machine
    precision for any (S, K, r, sigma, T).
    """
    print("[3] Vega-Gamma identity on random grid")
    rng = np.random.default_rng(seed=99)
    n = 10000
    S     = rng.uniform(50,  150, n)
    K     = rng.uniform(50,  150, n)
    r     = rng.uniform(0.01, 0.10, n)
    sigma = rng.uniform(0.10, 0.50, n)
    T     = rng.uniform(0.1,  2.0,  n)

    V = vega (S, K, r, sigma, T)
    G = gamma(S, K, r, sigma, T)
    residual = V - S ** 2 * sigma * T * G
    max_res = float(np.abs(residual).max())

    print(f"    Sample size: {n}")
    print(f"    Max |residual|: {max_res:.2e}  (tol {TOL_IDENTITY:.0e})  {_status(max_res < TOL_IDENTITY)}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("Greeks validation suite")
    print("=" * 72)
    print()

    check_finite_differences(); print()
    check_pde_residual();       print()
    check_vega_gamma();         print()

    print("=" * 72)


if __name__ == "__main__":
    main()
