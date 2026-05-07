"""
Validation script for the Thomas algorithm.

Five checks:

1. Diagonally-dominant random systems: cross-validation against
   numpy.linalg.solve.
2. The BTCS-style matrix (constant tridiagonal): hand-checkable.
3. thomas_solve_factored gives the same result as thomas_solve.
4. Edge cases: n = 1, n = 2.
5. Input validation: shape mismatches raise ValueError.

Run with
    python -m validate_thomas
or directly.
"""

import numpy as np

from quantlib.thomas import (
    thomas_solve,
    thomas_factor,
    thomas_solve_factored,
    ThomasFactor,
)


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
# Helpers
# ---------------------------------------------------------------------------
def assemble_tridiagonal(sub, diag, sup):
    """Reconstruct the dense tridiagonal matrix for cross-validation."""
    return np.diag(diag) + np.diag(sup, 1) + np.diag(sub, -1)


def random_diagonally_dominant(n, rng, dominance_ratio=2.0):
    """Generate a random tridiagonal system that is strictly diagonally
    dominant. Returns (sub, diag, sup, rhs)."""
    sub = rng.standard_normal(n - 1)
    sup = rng.standard_normal(n - 1)
    # Make diag strictly dominant: |diag[i]| >= dominance_ratio * (|sub[i-1]| + |sup[i]|).
    diag = np.empty(n)
    for i in range(n):
        adj = (abs(sub[i - 1]) if i > 0 else 0.0) + (abs(sup[i]) if i < n - 1 else 0.0)
        # Random sign times a magnitude that dominates by the given ratio.
        diag[i] = (1.0 if rng.random() < 0.5 else -1.0) * (1.0 + dominance_ratio * adj)
    rhs = rng.standard_normal(n)
    return sub, diag, sup, rhs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_random_systems():
    print("[1] Random diagonally-dominant systems")
    rng = np.random.default_rng(seed=42)
    sizes = [3, 5, 10, 50, 200, 1000]

    for n in sizes:
        sub, diag, sup, rhs = random_diagonally_dominant(n, rng)
        x_thomas = thomas_solve(sub, diag, sup, rhs)
        x_numpy  = np.linalg.solve(assemble_tridiagonal(sub, diag, sup), rhs)
        max_err = np.max(np.abs(x_thomas - x_numpy))
        # Backward stability gives errors of O(n * eps * cond(A)). For
        # diagonally dominant matrices cond(A) is moderate, so we set
        # the tolerance to 1e-10 across all sizes.
        check(f"n = {n:4d}", max_err < 1e-10,
              detail=f"max |x_thomas - x_numpy| = {max_err:.2e}")


def test_btcs_matrix():
    print("[2] BTCS-style constant-coefficient tridiagonal matrix")
    n = 100
    # Coefficients similar to BTCS at sigma=0.2, dtau=0.005, dx=0.008,
    # mu=0.03, r=0.05.
    alpha = 0.5 * 0.04 * 0.005 / (0.008 ** 2)        # ~ 1.56
    nu_signed = 0.03 * 0.005 / (2.0 * 0.008)          # ~ 9.4e-3
    b_minus = -alpha + nu_signed
    b_zero  = 1.0 + 2.0 * alpha + 0.05 * 0.005
    b_plus  = -alpha - nu_signed

    sub  = np.full(n - 1, b_minus)
    diag = np.full(n,     b_zero)
    sup  = np.full(n - 1, b_plus)
    rhs  = np.linspace(0.0, 1.0, n)

    x_thomas = thomas_solve(sub, diag, sup, rhs)
    x_numpy  = np.linalg.solve(assemble_tridiagonal(sub, diag, sup), rhs)
    max_err = np.max(np.abs(x_thomas - x_numpy))
    check("BTCS-style matrix matches numpy", max_err < 1e-12,
          detail=f"max err = {max_err:.2e}")


def test_factored_consistency():
    print("[3] thomas_solve_factored is consistent with thomas_solve")
    rng = np.random.default_rng(seed=7)
    n = 50
    sub, diag, sup, rhs = random_diagonally_dominant(n, rng)

    x_one_shot = thomas_solve(sub, diag, sup, rhs)
    factor     = thomas_factor(sub, diag, sup)
    x_factored = thomas_solve_factored(factor, rhs)

    check("factored == one-shot, single rhs",
          np.allclose(x_factored, x_one_shot, atol=1e-14),
          detail=f"max diff = {np.max(np.abs(x_factored - x_one_shot)):.2e}")

    # Multiple RHSs against the same factor.
    max_diff = 0.0
    for k in range(5):
        rhs_k = rng.standard_normal(n)
        a = thomas_solve(sub, diag, sup, rhs_k)
        b = thomas_solve_factored(factor, rhs_k)
        max_diff = max(max_diff, np.max(np.abs(a - b)))
    check("factored == one-shot, 5 distinct rhs",
          max_diff < 1e-14, detail=f"max diff over all = {max_diff:.2e}")


def test_edge_cases():
    print("[4] Edge cases: small systems")
    # n = 1: the system is just diag[0] * x[0] = rhs[0]; sub and sup are
    # empty.
    sub  = np.array([], dtype=float)
    diag = np.array([3.0])
    sup  = np.array([], dtype=float)
    rhs  = np.array([6.0])
    x = thomas_solve(sub, diag, sup, rhs)
    check("n=1: 3 x = 6 -> x = 2", np.isclose(x[0], 2.0))

    # n = 2: 2x2 system,
    #   2 -1   x1   3
    #  -1  2   x2 = 0
    # Solution: x1 = 2, x2 = 1.
    sub  = np.array([-1.0])
    diag = np.array([ 2.0,  2.0])
    sup  = np.array([-1.0])
    rhs  = np.array([ 3.0,  0.0])
    x = thomas_solve(sub, diag, sup, rhs)
    check("n=2: hand-checkable system", np.allclose(x, [2.0, 1.0]),
          detail=f"got {x}")

    # n=2 with factor.
    factor = thomas_factor(sub, diag, sup)
    x_f = thomas_solve_factored(factor, rhs)
    check("n=2: factored agrees", np.allclose(x_f, [2.0, 1.0]))


def test_input_validation():
    print("[5] Input validation")
    sub  = np.array([1.0, 2.0])
    diag = np.array([3.0, 4.0, 5.0])
    sup  = np.array([6.0, 7.0])
    rhs  = np.array([8.0, 9.0, 10.0])

    # Correct sizes — should succeed.
    try:
        thomas_solve(sub, diag, sup, rhs)
        check("correct sizes do not raise", True)
    except Exception as e:
        check("correct sizes do not raise", False, detail=str(e))

    # Wrong sub size.
    try:
        thomas_solve(np.array([1.0]), diag, sup, rhs)
        check("wrong sub size raises", False)
    except ValueError:
        check("wrong sub size raises", True)

    # Wrong sup size.
    try:
        thomas_solve(sub, diag, np.array([1.0]), rhs)
        check("wrong sup size raises", False)
    except ValueError:
        check("wrong sup size raises", True)

    # Wrong rhs size.
    try:
        thomas_solve(sub, diag, sup, np.array([1.0, 2.0]))
        check("wrong rhs size raises", False)
    except ValueError:
        check("wrong rhs size raises", True)

    # Empty diag.
    try:
        thomas_solve(np.array([]), np.array([]), np.array([]), np.array([]))
        check("empty system raises", False)
    except ValueError:
        check("empty system raises", True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 72)
    print("Phase 3 Block 2 - Thomas algorithm validation")
    print("=" * 72)

    test_random_systems()
    test_btcs_matrix()
    test_factored_consistency()
    test_edge_cases()
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
