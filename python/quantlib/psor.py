"""
Projected Successive Over-Relaxation (PSOR) solver for tridiagonal
linear complementarity problems.

Solves the LCP

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
positive-definite (Cryer 1971). For non-symmetric diagonally dominant
M-matrices the general SOR guarantee is only 0 < omega <
2 / (1 + rho(|J|)), J the Jacobi matrix (Varga, Matrix Iterative
Analysis); the non-symmetric LCP theory is in Cottle-Pang-Stone, ch. 5.
The project's CN matrices have same-signed off-diagonals and are
diagonally similar to symmetric positive-definite ones, so the full
range (0, 2) applies to them. The optimal omega of linear SOR is
2 / (1 + sqrt(1 - rho_J^2)) where rho_J is the spectral radius of the
Jacobi iteration, about 1.2-1.35 for the project's CN matrices; with
the projection active the measured optimum is between 1.2 and 1.6
(see the comment in psor_solve).

Used by quantlib.cn_american for the per-step solve in
American option pricing.
"""

import numpy as np


def psor_solve(sub, diag, sup, rhs, obstacle, *,
               omega=1.2, tol_abs=1e-10, tol_rel=1e-9,
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
        CN/BTCS matrices is approximately 1.2-1.6.
    tol_abs : float, default 1e-10
        Absolute tolerance on the iteration increment.
    tol_rel : float, default 1e-9
        Relative tolerance on the iteration increment. See the note on
        the stopping rule in the function body before loosening these.
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
    # omega = 1 is projected Gauss-Seidel. The best omega depends on the
    # matrix: about 1.2 on the small problem swept in validate_psor.py, and
    # 1.5-1.6 on the 400 x 200 Crank-Nicolson grid of benchmark_phase3.py
    # (about 18 sweeps per step, against about 20 at the omega = 1.4 that
    # the benchmark uses). The SOR formula in the module docstring is only a
    # heuristic once the projection is active, so omega is left as a
    # parameter rather than fixed.
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

        # Stopping criterion on the size of the last update. For a
        # contraction with factor q the remaining error is about q/(1-q)
        # times that update, so a loose tolerance under-converges most when
        # omega is far from its optimum. With the earlier defaults
        # (tol_rel = 1e-7) the CN-PSOR American put of benchmark_phase3.py
        # moved by 1.7e-3 between omega = 1.0 and omega = 1.6; with
        # tol_rel = 1e-9 the spread is below 2e-5, under the grid's own
        # discretisation error (about 1e-4 there).
        threshold = tol_abs + tol_rel * np.max(np.abs(x))
        if max_change < threshold:
            return x, k + 1

    raise RuntimeError(
        f"psor_solve: did not converge in {max_iter} iterations "
        f"(last increment {max_change:.2e}). "
        f"Try a different omega (current {omega}) or check that "
        f"the matrix is diagonally dominant."
    )
