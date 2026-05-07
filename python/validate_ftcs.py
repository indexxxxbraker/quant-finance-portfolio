"""
Validation script for the FTCS finite-difference pricer of Phase 3
Block 1.

Performs five checks:

1. Cross-validation against the Black-Scholes closed form on a grid
   of (S, K, r, sigma, T) parameters, with a tolerance scaled to the
   expected discretisation error O(dx^2).
2. Empirical convergence rate: doubling N and quadrupling M (to
   preserve CFL) should reduce the error by a factor of ~4.
3. Put-call parity: C - P = S - K * exp(-r * T) at every grid point
   to within twice the discretisation tolerance.
4. CFL violation triggers ValueError with an informative message.
5. Out-of-domain spot triggers ValueError.

Exit code 0 on success, 1 on failure. Run from python/ as
    python -m validate_ftcs
or directly.
"""

import numpy as np

from quantlib.black_scholes import call_price as bs_call_price
from quantlib.black_scholes import put_price  as bs_put_price
from quantlib.ftcs import (
    ftcs_european_call,
    ftcs_european_put,
    ftcs_min_M_for_cfl,
    _ftcs_march,
)
from quantlib.pde import (
    build_grid,
    call_initial_condition,
    call_boundary_lower,
    call_boundary_upper,
)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
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
# Tests
# ---------------------------------------------------------------------------
def test_cross_validation():
    """Compare FTCS to BS closed-form on a representative parameter grid."""
    print("[1] Cross-validation against Black-Scholes closed form")

    # Grid of parameters spanning the project's typical range.
    test_cases = [
        # (S,    K,    r,    sigma, T,    N,   M)
        (100.0, 100.0, 0.05, 0.20, 1.00, 200, 800),    # ATM, 1Y
        ( 90.0, 100.0, 0.05, 0.20, 1.00, 200, 800),    # OTM call / ITM put
        (110.0, 100.0, 0.05, 0.20, 1.00, 200, 800),    # ITM call / OTM put
        (100.0, 100.0, 0.05, 0.30, 1.00, 200, 1800),   # higher vol
        (100.0, 100.0, 0.05, 0.20, 0.25, 200, 800),    # short maturity
        (100.0, 100.0, 0.10, 0.20, 1.00, 200, 800),    # higher rate
        (100.0, 100.0, 0.00, 0.20, 1.00, 200, 800),    # zero rate
        ( 42.0,  40.0, 0.10, 0.20, 0.50, 200, 800),    # Hull 15.6
    ]

    # Tolerance: expected error is O(dx^2). For these parameters,
    # dx ~ 0.008, dx^2 ~ 6e-5; the constant in front depends on the
    # smoothness near the kink. Empirically the worst-case absolute
    # error on this set is around 5e-3 for prices of order 5-15.
    tol = 5e-3

    for S, K, r, sigma, T, N, M in test_cases:
        # Verify M satisfies CFL with margin
        M_min = ftcs_min_M_for_cfl(N=N, T=T, sigma=sigma, target_alpha=0.4)
        if M < M_min:
            print(f"    NOTE: M={M} < M_min={M_min}, increasing.")
            M = M_min

        c_ftcs = ftcs_european_call(S, K, r, sigma, T, N=N, M=M)
        p_ftcs = ftcs_european_put (S, K, r, sigma, T, N=N, M=M)
        c_bs   = bs_call_price(S, K, r, sigma, T)
        p_bs   = bs_put_price (S, K, r, sigma, T)

        err_c = abs(c_ftcs - c_bs)
        err_p = abs(p_ftcs - p_bs)

        check(
            f"call S={S:6.2f} K={K:.0f} sigma={sigma:.2f} T={T:.2f}",
            err_c < tol,
            detail=f"FTCS={c_ftcs:.4f} BS={c_bs:.4f} err={err_c:.2e}",
        )
        check(
            f"put  S={S:6.2f} K={K:.0f} sigma={sigma:.2f} T={T:.2f}",
            err_p < tol,
            detail=f"FTCS={p_ftcs:.4f} BS={p_bs:.4f} err={err_p:.2e}",
        )


def test_convergence_rate():
    """Doubling N and quadrupling M should reduce the error by ~4."""
    print("[2] Empirical convergence rate")
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    c_bs = bs_call_price(S, K, r, sigma, T)

    refinement_levels = [
        (100, 200),
        (200, 800),
        (400, 3200),
    ]

    errors = []
    for N, M in refinement_levels:
        c_ftcs = ftcs_european_call(S, K, r, sigma, T, N=N, M=M)
        err = abs(c_ftcs - c_bs)
        errors.append(err)
        print(f"    N={N:4d}, M={M:5d}: error = {err:.4e}")

    # Ratio of errors should be close to 4 (one order of dx^2 reduction
    # per doubling of N). Accept ratios in [3.0, 5.5] to absorb the
    # constant prefactor variability and finite-grid noise.
    ratio_1 = errors[0] / errors[1]
    ratio_2 = errors[1] / errors[2]
    print(f"    Ratio level0/level1 = {ratio_1:.2f}  (expected ~4)")
    print(f"    Ratio level1/level2 = {ratio_2:.2f}  (expected ~4)")

    check("error ratio level0/level1 in [3.0, 5.5]",
          3.0 <= ratio_1 <= 5.5,
          detail=f"got {ratio_1:.2f}")
    check("error ratio level1/level2 in [3.0, 5.5]",
          3.0 <= ratio_2 <= 5.5,
          detail=f"got {ratio_2:.2f}")


def test_put_call_parity():
    """Put-call parity must hold to within 2 * discretisation tolerance."""
    print("[3] Put-call parity")
    K, r, sigma, T = 100.0, 0.05, 0.20, 1.0
    N, M = 200, 800
    spots = np.array([85, 95, 100, 105, 115], dtype=float)

    rhs = spots - K * np.exp(-r * T)
    diffs = []
    for S in spots:
        c = ftcs_european_call(S, K, r, sigma, T, N=N, M=M)
        p = ftcs_european_put (S, K, r, sigma, T, N=N, M=M)
        lhs = c - p
        diffs.append(abs(lhs - rhs[spots == S][0]))

    max_diff = max(diffs)
    # Both call and put have O(dx^2) error; their difference can have
    # error up to twice that. Use a margin of 2x the cross-validation
    # tolerance.
    tol = 1e-2
    check("put-call parity holds for all spots",
          max_diff < tol,
          detail=f"max |C - P - (S - K e^-rT)| = {max_diff:.2e}")


def test_cfl_violation():
    """CFL violation must raise ValueError."""
    print("[4] CFL enforcement")

    # Choose M small enough that alpha > 0.5. With sigma=0.2, T=1,
    # n_sigma=4, N=100, dx = 8/100 = 0.08, alpha_safe < 0.5 needs
    # dtau < 0.32; M >= ceil(1 / 0.32) = 4. Choose M = 3 to violate.
    try:
        ftcs_european_call(100.0, 100.0, 0.05, 0.20, 1.0, N=100, M=3)
        check("CFL violation raises ValueError", False,
              detail="no exception raised")
    except ValueError as e:
        check("CFL violation raises ValueError", True,
              detail=f"message starts with '{str(e)[:30]}...'")
    except Exception as e:
        check("CFL violation raises ValueError", False,
              detail=f"raised {type(e).__name__}, not ValueError")


def test_invalid_spot():
    """Spot outside the truncated domain must raise ValueError."""
    print("[5] Out-of-domain spot")

    K, r, sigma, T = 100.0, 0.05, 0.20, 1.0
    # x_max = 4 * 0.2 * 1 = 0.8; S_max = 100 * exp(0.8) ~ 222.55.
    # Pick S = 1000 to be safely above.
    try:
        ftcs_european_call(1000.0, K, r, sigma, T, N=200, M=800)
        check("out-of-domain spot raises ValueError", False)
    except ValueError as e:
        check("out-of-domain spot raises ValueError", True,
              detail=f"message starts with '{str(e)[:40]}...'")
    except Exception as e:
        check("out-of-domain spot raises ValueError", False,
              detail=f"raised {type(e).__name__}")

    # S = 0 must also raise (validated separately)
    try:
        ftcs_european_call(0.0, K, r, sigma, T, N=200, M=800)
        check("S = 0 raises ValueError", False)
    except ValueError:
        check("S = 0 raises ValueError", True)


def test_explosion_demo():
    """Bypass CFL check, observe the sawtooth blow-up. Diagnostic only:
    not a pass/fail check, but a printout of the final-vector norm so
    the lesson is visible.
    """
    print("[6] CFL-violating diagnostic (visualises the explosion)")

    # Pick (N, M) that violates CFL. With sigma=0.2, T=1, n_sigma=4:
    # dx = 8 sigma sqrt(T) / N = 1.6 / N. The stability boundary is
    # alpha = (sigma^2 / 2) * dtau / dx^2 = 1/2, i.e.
    # M_min = T sigma^2 / (2 * (dx^2)) = T sigma^2 N^2 / (2 * 1.6^2).
    # For N=100 this is M_min = 0.04 * 10000 / 5.12 = 78. Picking M=50
    # well below that gives alpha ~ 0.6, comfortably unstable.
    K, r, sigma, T = 100.0, 0.05, 0.20, 1.0
    grid = build_grid(N=100, M=50, T=T, sigma=sigma, r=r, K=K)
    alpha = 0.5 * sigma ** 2 * grid.dtau / grid.dx ** 2
    print(f"    Grid: N=100, M=50,  dx={grid.dx:.4f}, dtau={grid.dtau:.4f}")
    print(f"    Fourier number alpha = {alpha:.4f}  "
          f"({'unstable' if alpha > 0.5 else 'stable'})")

    V0 = call_initial_condition(grid.xs, grid.K)
    bc_lo = call_boundary_lower(grid)
    bc_hi = call_boundary_upper(grid)

    V_unstable = _ftcs_march(grid, V0, bc_lo, bc_hi, validate_cfl=False)

    max_abs = np.max(np.abs(V_unstable))
    sign_changes = np.sum(np.diff(np.sign(V_unstable)) != 0)
    print(f"    max(|V|) = {max_abs:.3e}  (BS price ~ 10)")
    print(f"    sign changes in V (sawtooth signature) = {sign_changes}")

    # Confirm we see the predicted blow-up: max value way above the
    # true price, and many sign changes.
    check("alpha > 0.5 produces |V| above any sane price ceiling",
          max_abs > 1e6,
          detail=f"max(|V|) = {max_abs:.3e}")
    check("alpha > 0.5 produces oscillatory sign-changing V",
          sign_changes > 10,
          detail=f"{sign_changes} sign changes")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 72)
    print("Phase 3 Block 1 - FTCS pricer validation")
    print("=" * 72)

    test_cross_validation()
    test_convergence_rate()
    test_put_call_parity()
    test_cfl_violation()
    test_invalid_spot()
    test_explosion_demo()

    print()
    print("=" * 72)
    total = _n_pass + _n_fail
    if _n_fail == 0:
        print(f"  {PASS}: {_n_pass}/{total} checks succeeded.")
        raise SystemExit(0)
    else:
        print(f"  {FAIL}: {_n_pass}/{total} succeeded, {_n_fail} failed.")
        raise SystemExit(1)
