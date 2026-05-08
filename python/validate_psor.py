"""
Validation script for the PSOR tridiagonal LCP solver.

Five checks:

1. Unconstrained problem (obstacle = -infty): PSOR should reduce to
   plain SOR for the linear system, agreeing with numpy.linalg.solve.
2. Constrained problem with hand-checkable answer (small system).
3. omega-sweep: total iterations as a function of omega should have
   a minimum interior to (0, 2).
4. Input validation: shape mismatches, omega outside (0, 2),
   non-feasible x0.
5. max_iter exceeded raises RuntimeError.
"""

import numpy as np

from quantlib.psor import psor_solve


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


def assemble_dense(sub, diag, sup):
    n = diag.size
    A = np.diag(diag) + np.diag(sup, 1) + np.diag(sub, -1)
    return A


# ---------------------------------------------------------------------------
def test_unconstrained_matches_linear_solve():
    print("[1] Unconstrained LCP reduces to linear solve")
    rng = np.random.default_rng(seed=42)

    for n in [3, 10, 50, 200]:
        # Build a strictly diagonally dominant tridiagonal.
        sub = rng.standard_normal(n - 1)
        sup = rng.standard_normal(n - 1)
        diag = np.array([
            (1.0 if rng.random() < 0.5 else -1.0)
              * (1.0 + 2.0 * (
                  (abs(sub[i - 1]) if i > 0 else 0.0)
                  + (abs(sup[i]) if i < n - 1 else 0.0)
              ))
            for i in range(n)
        ])
        # Make sure diagonal is positive (PSOR assumes this implicitly).
        diag = np.abs(diag)
        rhs = rng.standard_normal(n)
        # Trivial obstacle: way below any plausible solution.
        obstacle = np.full(n, -1e10)

        x_psor, n_iter = psor_solve(
            sub, diag, sup, rhs, obstacle,
            omega=1.2, tol_abs=1e-12, tol_rel=1e-12, max_iter=10000,
        )
        x_ref = np.linalg.solve(assemble_dense(sub, diag, sup), rhs)
        max_err = np.max(np.abs(x_psor - x_ref))
        check(f"n={n:3d}", max_err < 1e-6,
              detail=f"max |PSOR - numpy| = {max_err:.2e}, iter = {n_iter}")


def test_constrained_small_system():
    print("[2] Constrained LCP, hand-checkable")
    # 4x4 problem with active constraints.
    n = 4
    sub  = np.full(n - 1, -1.0)
    diag = np.full(n,      4.0)
    sup  = np.full(n - 1, -1.0)
    rhs  = np.array([1.0, 2.0, 3.0, 4.0])
    obstacle = np.full(n, 1.0)

    x, n_iter = psor_solve(sub, diag, sup, rhs, obstacle, omega=1.2)
    # Verify all components >= obstacle.
    check("solution >= obstacle", np.all(x >= obstacle - 1e-10),
          detail=f"x = {x}")

    # Verify complementarity: where x = obstacle, the residual A x - rhs
    # should be > 0; where x > obstacle, residual should be ~ 0.
    A = assemble_dense(sub, diag, sup)
    residual = A @ x - rhs
    active = np.isclose(x, obstacle, atol=1e-8)
    inactive = ~active

    if active.any():
        check("active set has positive residual",
              np.all(residual[active] >= -1e-7),
              detail=f"min residual on active = {residual[active].min():.2e}")
    if inactive.any():
        check("inactive set has zero residual",
              np.allclose(residual[inactive], 0.0, atol=1e-6),
              detail=f"max |residual| on inactive = "
                     f"{np.max(np.abs(residual[inactive])):.2e}")


def test_omega_sweep():
    print("[3] omega sweep: iterations should have an interior minimum")
    # Build a moderately conditioned tridiagonal LCP.
    rng = np.random.default_rng(seed=7)
    n = 100
    diag_val = 2.5
    off_val  = -1.0
    sub  = np.full(n - 1, off_val)
    diag = np.full(n,     diag_val)
    sup  = np.full(n - 1, off_val)
    rhs  = rng.standard_normal(n)
    obstacle = np.full(n, -10.0)

    omegas = [0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
    iters = []
    for w in omegas:
        _, n_iter = psor_solve(
            sub, diag, sup, rhs, obstacle, omega=w,
            tol_abs=1e-9, tol_rel=1e-9, max_iter=20000,
        )
        iters.append(n_iter)
        print(f"    omega={w:.1f}: iter = {n_iter}")

    # The minimum should be at an interior omega, and the extremes
    # should take more iterations than the middle.
    idx_min = int(np.argmin(iters))
    check("optimum is interior (not at boundary)",
          0 < idx_min < len(omegas) - 1,
          detail=f"omega* = {omegas[idx_min]}")
    check("omega=1.4 strictly faster than omega=0.5",
          iters[omegas.index(1.4)] < iters[omegas.index(0.5)],
          detail=f"{iters[omegas.index(1.4)]} vs {iters[omegas.index(0.5)]}")
    check("omega=1.4 strictly faster than omega=1.8",
          iters[omegas.index(1.4)] < iters[omegas.index(1.8)],
          detail=f"{iters[omegas.index(1.4)]} vs {iters[omegas.index(1.8)]}")


def test_input_validation():
    print("[4] Input validation")
    sub  = np.array([1.0, 2.0])
    diag = np.array([5.0, 5.0, 5.0])
    sup  = np.array([1.0, 2.0])
    rhs  = np.array([1.0, 2.0, 3.0])
    obstacle = np.array([0.0, 0.0, 0.0])

    # Correct sizes succeed.
    try:
        psor_solve(sub, diag, sup, rhs, obstacle)
        check("correct sizes do not raise", True)
    except Exception as e:
        check("correct sizes do not raise", False, detail=str(e))

    # omega outside (0, 2)
    for bad_omega in [0.0, 2.0, -0.1, 2.1]:
        try:
            psor_solve(sub, diag, sup, rhs, obstacle, omega=bad_omega)
            check(f"omega={bad_omega} raises", False)
        except ValueError:
            check(f"omega={bad_omega} raises", True)

    # rhs size mismatch
    try:
        psor_solve(sub, diag, sup, np.array([1.0, 2.0]), obstacle)
        check("rhs size mismatch raises", False)
    except ValueError:
        check("rhs size mismatch raises", True)

    # obstacle size mismatch
    try:
        psor_solve(sub, diag, sup, rhs, np.array([0.0, 0.0]))
        check("obstacle size mismatch raises", False)
    except ValueError:
        check("obstacle size mismatch raises", True)

    # Infeasible x0 (below obstacle)
    try:
        psor_solve(sub, diag, sup, rhs, np.full(3, 100.0),
                    x0=np.zeros(3))
        check("infeasible x0 raises", False)
    except ValueError:
        check("infeasible x0 raises", True)


def test_max_iter_exceeded():
    print("[5] max_iter exceeded raises RuntimeError")
    # Use very tight tolerance and tiny max_iter to force failure.
    n = 50
    sub  = np.full(n - 1, -1.0)
    diag = np.full(n,      2.5)
    sup  = np.full(n - 1, -1.0)
    rhs  = np.ones(n)
    obstacle = np.full(n, -100.0)

    try:
        psor_solve(sub, diag, sup, rhs, obstacle,
                    omega=1.2, tol_abs=1e-15, tol_rel=1e-15,
                    max_iter=5)
        check("max_iter raises RuntimeError", False)
    except RuntimeError:
        check("max_iter raises RuntimeError", True)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 72)
    print("Phase 3 Block 4 - PSOR solver validation")
    print("=" * 72)

    test_unconstrained_matches_linear_solve()
    test_constrained_small_system()
    test_omega_sweep()
    test_input_validation()
    test_max_iter_exceeded()

    print()
    print("=" * 72)
    total = _n_pass + _n_fail
    if _n_fail == 0:
        print(f"  {PASS}: {_n_pass}/{total} checks succeeded.")
        raise SystemExit(0)
    else:
        print(f"  {FAIL}: {_n_pass}/{total} succeeded, {_n_fail} failed.")
        raise SystemExit(1)
