"""
Projected Successive Over-Relaxation (PSOR) solver for tridiagonal
linear complementarity problems.

Phase 3 Block 4. Solves the LCP

    A x >= rhs           (componentwise)
    x   >= obstacle      (componentwise)
    (A x - rhs) (x - obstacle) = 0   (componentwise)

where A is tridiagonal with sub-diagonal `sub`, main diagonal `diag`,
and super-diagonal `sup`. The method iterates SOR-style updates
followed by a max-projection onto the obstacle:

    x_i^{(k+1)} = max{ obstacle_i,
                       (1 - omega) x_i^{(k)}
                       + omega/diag_i * (rhs_i - sub_{i-1} x_{i-1}^{(k+1)}
                                                 - sup_i  x_{i+1}^{(k)}) }

Convergence is geometric for omega in (0, 2) when A is symmetric
positive-definite (Cryer 1971), and for diagonally-dominant M-matrices
in the non-symmetric case (Cottle-Pang-Stone, ch. 5). The optimal
omega is approximately 2 / (1 + sqrt(1 - rho_J^2)) where rho_J is
the spectral radius of the Jacobi iteration, typically in [1.2, 1.5]
for the project's CN matrices.

Used by quantlib.cn_american (Block 4) for the per-step solve in
American option pricing.
"""

import numpy as np


def psor_solve(sub, diag, sup, rhs, obstacle, *,
               omega=1.2, tol_abs=1e-8, tol_rel=1e-7,
               max_iter=10000, x0=None):
    """Solve a tridiagonal LCP by Projected SOR.

    Parameters
    ----------
    sub : ndarray, shape (n-1,)
        Sub-diagonal of A.
    diag : ndarray, shape (n,)
        Main diagonal of A. Must have non-zero entries.
    sup : ndarray, shape (n-1,)
        Super-diagonal of A.
    rhs : ndarray, shape (n,)
        Right-hand side vector.
    obstacle : ndarray, shape (n,)
        Lower-bound constraint: solution must satisfy x >= obstacle.
    omega : float, default 1.2
        Relaxation parameter; must be in (0, 2). Optimal for typical
        CN/BTCS matrices is approximately 1.2-1.5.
    tol_abs : float, default 1e-8
        Absolute tolerance on the iteration increment.
    tol_rel : float, default 1e-7
        Relative tolerance on the iteration increment.
    max_iter : int, default 10000
        Maximum number of iterations before declaring failure.
    x0 : ndarray or None, default None
        Initial guess. Must satisfy x0 >= obstacle. If None, uses
        x0 = obstacle.

    Returns
    -------
    x : ndarray, shape (n,)
        Solution to the LCP.
    n_iter : int
        Number of iterations actually performed.

    Raises
    ------
    ValueError
        On shape mismatch, omega outside (0, 2), or infeasible x0.
    RuntimeError
        If max_iter is reached without satisfying the tolerance.
        This indicates either an inadequate omega, a poorly
        conditioned matrix, or both.
    """
    sub      = np.asarray(sub,      dtype=np.float64)
    diag     = np.asarray(diag,     dtype=np.float64)
    sup      = np.asarray(sup,      dtype=np.float64)
    rhs      = np.asarray(rhs,      dtype=np.float64)
    obstacle = np.asarray(obstacle, dtype=np.float64)

    # Shape validation.
    n = diag.size
    if n == 0:
        raise ValueError("psor_solve: empty system")
    if sub.size != n - 1:
        raise ValueError(
            f"psor_solve: sub must have length n-1 = {n-1}, got {sub.size}"
        )
    if sup.size != n - 1:
        raise ValueError(
            f"psor_solve: sup must have length n-1 = {n-1}, got {sup.size}"
        )
    if rhs.size != n:
        raise ValueError(
            f"psor_solve: rhs must have length n = {n}, got {rhs.size}"
        )
    if obstacle.size != n:
        raise ValueError(
            f"psor_solve: obstacle must have length n = {n}, "
            f"got {obstacle.size}"
        )

    # Parameter validation.
    if not (0.0 < omega < 2.0):
        raise ValueError(
            f"psor_solve: omega must lie strictly in (0, 2), got {omega}"
        )
    if max_iter <= 0:
        raise ValueError(
            f"psor_solve: max_iter must be positive, got {max_iter}"
        )

    # Initial guess.
    if x0 is None:
        x = obstacle.copy()
    else:
        x = np.asarray(x0, dtype=np.float64).copy()
        if x.size != n:
            raise ValueError(
                f"psor_solve: x0 has length {x.size}, expected {n}"
            )
        if np.any(x < obstacle - 1e-14):
            raise ValueError(
                "psor_solve: x0 must satisfy x0 >= obstacle"
            )

    # Pre-compute inverse diagonals once.
    inv_diag = 1.0 / diag
    if np.any(~np.isfinite(inv_diag)):
        raise ValueError(
            "psor_solve: diag has zero or infinite entries"
        )

    # PSOR iteration. The first and last rows pick up only one
    # off-diagonal term; the interior rows pick up two.
    for k in range(max_iter):
        max_change = 0.0

        for i in range(n):
            # Compute the un-projected SOR update.
            if i == 0:
                gs_update = (rhs[0] - sup[0] * x[1]) * inv_diag[0]
            elif i == n - 1:
                gs_update = (rhs[n-1] - sub[n-2] * x[n-2]) * inv_diag[n-1]
            else:
                gs_update = (
                    rhs[i] - sub[i-1] * x[i-1] - sup[i] * x[i+1]
                ) * inv_diag[i]

            new_val = (1.0 - omega) * x[i] + omega * gs_update

            # Project onto the obstacle.
            new_val = max(new_val, obstacle[i])

            # Track the largest change for the stopping criterion.
            change = abs(new_val - x[i])
            if change > max_change:
                max_change = change

            x[i] = new_val

        # Stopping criterion: combined absolute and relative tolerance.
        threshold = tol_abs + tol_rel * np.max(np.abs(x))
        if max_change < threshold:
            return x, k + 1

    raise RuntimeError(
        f"psor_solve: did not converge in {max_iter} iterations "
        f"(last increment {max_change:.2e}). "
        f"Try a different omega (current {omega}) or check that "
        f"the matrix is diagonally dominant."
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 4x4 problem with no active constraint: should reduce to a
    # standard linear solve and match numpy.linalg.solve.
    n = 4
    sub  = np.full(n - 1, -1.0)
    diag = np.full(n,      4.0)
    sup  = np.full(n - 1, -1.0)
    rhs  = np.array([1.0, 2.0, 3.0, 4.0])
    obstacle = np.full(n, -1e10)   # essentially no constraint

    A = np.diag(diag) + np.diag(sup, 1) + np.diag(sub, -1)
    x_ref = np.linalg.solve(A, rhs)
    x_psor, n_iter = psor_solve(sub, diag, sup, rhs, obstacle,
                                 omega=1.2)
    print(f"4x4 unconstrained LCP (should match linear solve):")
    print(f"  numpy.linalg.solve: {x_ref}")
    print(f"  psor_solve:         {x_psor}  (n_iter = {n_iter})")
    print(f"  max |diff|        = {np.max(np.abs(x_psor - x_ref)):.2e}")

    # Active constraint: obstacle = [1.0, 1.0, 1.0, 1.0]. Only the
    # first node has reference solution below the obstacle, so the
    # constraint is active there.
    print()
    obstacle = np.full(n, 1.0)
    x_psor, n_iter = psor_solve(sub, diag, sup, rhs, obstacle,
                                 omega=1.2)
    print(f"4x4 LCP with obstacle = 1.0 (some constraints active):")
    print(f"  unconstrained solution: {x_ref}")
    print(f"  PSOR solution:          {x_psor}  (n_iter = {n_iter})")
    print(f"  active set:             {np.isclose(x_psor, 1.0)}")
