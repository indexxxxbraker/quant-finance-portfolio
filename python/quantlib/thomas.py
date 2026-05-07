"""
Thomas algorithm for tridiagonal linear systems.

Phase 3 Block 2. Provides an O(n) solver for systems of the form

    diag[0]   sup[0]                                   x[0]   = rhs[0]
    sub[0]    diag[1]   sup[1]                         x[1]   = rhs[1]
              sub[1]    diag[2]   sup[2]               x[2]   = rhs[2]
                        ...                            ...
                        sub[n-2]  diag[n-1]            x[n-1] = rhs[n-1]

The implementation does Gaussian elimination *without pivoting*. This
is provably backward-stable when the matrix is diagonally dominant
(|diag[i]| >= |sub[i-1]| + |sup[i]|; Higham 2002 Theorem 9.5). For the
BTCS application of Block 2, diagonal dominance follows from the grid
Pe'clet condition |mu| * dx / sigma^2 <= 1, which holds with comfortable
margin in our parameter range.

For repeated solves with a fixed matrix and varying right-hand side
(exactly the BTCS situation: M time steps with the same A), use
    factor = thomas_factor(sub, diag, sup)
    for each rhs:
        x = thomas_solve_factored(factor, rhs)
to avoid recomputing the (n-1) divisions of the forward sweep at each
solve.
"""

from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------------
# Pre-factored representation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ThomasFactor:
    """Pre-computed quantities for repeated solves with a fixed
    tridiagonal matrix.

    Attributes
    ----------
    sub : ndarray, shape (n-1,)
        A copy of the original sub-diagonal. Needed during the d'
        recursion of the per-rhs solve.
    c_prime : ndarray, shape (n-1,)
        Modified super-diagonal (c[i] / m_i, where m_i is the effective
        pivot at row i).
    m : ndarray, shape (n,)
        Effective pivots, i.e. m[i] = diag[i] - sub[i-1] * c_prime[i-1]
        with m[0] = diag[0]. Stored (rather than 1/m) for clarity at
        the cost of one extra division per row in the per-rhs solve.
    """
    sub: np.ndarray
    c_prime: np.ndarray
    m: np.ndarray


# ---------------------------------------------------------------------------
# One-shot solver
# ---------------------------------------------------------------------------
def thomas_solve(sub, diag, sup, rhs):
    """Solve a tridiagonal linear system A x = rhs in O(n) operations.

    Parameters
    ----------
    sub : ndarray, shape (n-1,)
        Sub-diagonal: sub[i] is the matrix entry at row i+1, column i.
    diag : ndarray, shape (n,)
        Main diagonal.
    sup : ndarray, shape (n-1,)
        Super-diagonal: sup[i] is the matrix entry at row i, column i+1.
    rhs : ndarray, shape (n,)
        Right-hand side vector.

    Returns
    -------
    ndarray, shape (n,)
        Solution x.

    Raises
    ------
    ValueError
        On shape mismatch or empty input.

    Notes
    -----
    No pivoting. Backward-stable on diagonally-dominant matrices; may
    give large errors on non-diagonally-dominant matrices and will
    raise FloatingPointError if a pivot is exactly zero (which is
    impossible under the diagonal-dominance hypothesis of Theorem 9.5
    of Higham 2002).
    """
    sub  = np.asarray(sub,  dtype=np.float64)
    diag = np.asarray(diag, dtype=np.float64)
    sup  = np.asarray(sup,  dtype=np.float64)
    rhs  = np.asarray(rhs,  dtype=np.float64)

    n = diag.size
    if n == 0:
        raise ValueError("thomas_solve: empty system")
    if sub.size != n - 1:
        raise ValueError(
            f"thomas_solve: sub must have length n-1 = {n-1}, got {sub.size}"
        )
    if sup.size != n - 1:
        raise ValueError(
            f"thomas_solve: sup must have length n-1 = {n-1}, got {sup.size}"
        )
    if rhs.size != n:
        raise ValueError(
            f"thomas_solve: rhs must have length n = {n}, got {rhs.size}"
        )

    # Allocate working buffers; never modify the input arrays.
    c_prime = np.empty(n - 1, dtype=np.float64)
    d_prime = np.empty(n,     dtype=np.float64)

    # Forward sweep.
    if diag[0] == 0.0:
        raise ZeroDivisionError("thomas_solve: zero pivot at row 0")
    if n > 1:
        c_prime[0] = sup[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]
    for i in range(1, n):
        m_i = diag[i] - sub[i - 1] * (c_prime[i - 1] if i < n else 0.0)
        if m_i == 0.0:
            raise ZeroDivisionError(
                f"thomas_solve: zero effective pivot at row {i}; "
                "the matrix is singular or not diagonally dominant"
            )
        if i < n - 1:
            c_prime[i] = sup[i] / m_i
        d_prime[i] = (rhs[i] - sub[i - 1] * d_prime[i - 1]) / m_i

    # Backward substitution.
    x = np.empty(n, dtype=np.float64)
    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


# ---------------------------------------------------------------------------
# Pre-factored interface for repeated solves
# ---------------------------------------------------------------------------
def thomas_factor(sub, diag, sup):
    """Compute the pre-factorisation of a tridiagonal matrix.

    The factorisation depends only on the matrix entries; once
    computed, it can be applied to many different right-hand sides
    via thomas_solve_factored. This is the right thing for BTCS,
    where the matrix is constant across all M time steps.

    Parameters
    ----------
    sub, diag, sup : ndarray
        Tridiagonal matrix entries; same shapes as in thomas_solve.

    Returns
    -------
    ThomasFactor
        Immutable factorisation. Pass to thomas_solve_factored.

    Raises
    ------
    ValueError
        On shape mismatch.
    ZeroDivisionError
        If a pivot is zero during the forward sweep.
    """
    sub  = np.asarray(sub,  dtype=np.float64)
    diag = np.asarray(diag, dtype=np.float64)
    sup  = np.asarray(sup,  dtype=np.float64)

    n = diag.size
    if n == 0:
        raise ValueError("thomas_factor: empty system")
    if sub.size != n - 1:
        raise ValueError(
            f"thomas_factor: sub must have length n-1 = {n-1}, got {sub.size}"
        )
    if sup.size != n - 1:
        raise ValueError(
            f"thomas_factor: sup must have length n-1 = {n-1}, got {sup.size}"
        )

    c_prime = np.empty(n - 1, dtype=np.float64)
    m       = np.empty(n,     dtype=np.float64)

    m[0] = diag[0]
    if m[0] == 0.0:
        raise ZeroDivisionError("thomas_factor: zero pivot at row 0")
    if n > 1:
        c_prime[0] = sup[0] / m[0]
    for i in range(1, n):
        m[i] = diag[i] - sub[i - 1] * (c_prime[i - 1] if i < n else 0.0)
        if m[i] == 0.0:
            raise ZeroDivisionError(
                f"thomas_factor: zero effective pivot at row {i}"
            )
        if i < n - 1:
            c_prime[i] = sup[i] / m[i]

    return ThomasFactor(sub=sub.copy(), c_prime=c_prime, m=m)


def thomas_solve_factored(factor, rhs):
    """Apply a pre-computed factorisation to solve A x = rhs.

    Parameters
    ----------
    factor : ThomasFactor
        From thomas_factor(sub, diag, sup).
    rhs : ndarray, shape (n,)
        Right-hand side vector. Must match the matrix size.

    Returns
    -------
    ndarray, shape (n,)
        Solution x.

    Raises
    ------
    ValueError
        On size mismatch.
    """
    rhs = np.asarray(rhs, dtype=np.float64)
    n = factor.m.size
    if rhs.size != n:
        raise ValueError(
            f"thomas_solve_factored: rhs has length {rhs.size}, expected {n}"
        )

    # Forward d-recursion.
    d_prime = np.empty(n, dtype=np.float64)
    d_prime[0] = rhs[0] / factor.m[0]
    for i in range(1, n):
        d_prime[i] = (rhs[i] - factor.sub[i - 1] * d_prime[i - 1]) / factor.m[i]

    # Backward substitution.
    x = np.empty(n, dtype=np.float64)
    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - factor.c_prime[i] * x[i + 1]

    return x


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Hand-checkable 4x4 system. Diagonally dominant.
    #
    #   4 -1  0  0   x0    1
    #  -1  4 -1  0   x1  = 2
    #   0 -1  4 -1   x2    3
    #   0  0 -1  4   x3    4
    #
    # Reference solution via numpy.linalg.solve:
    n = 4
    sub  = np.full(n - 1, -1.0)
    diag = np.full(n,      4.0)
    sup  = np.full(n - 1, -1.0)
    rhs  = np.array([1.0, 2.0, 3.0, 4.0])

    A = np.diag(diag) + np.diag(sup, 1) + np.diag(sub, -1)
    x_ref = np.linalg.solve(A, rhs)
    x_thomas = thomas_solve(sub, diag, sup, rhs)
    factor = thomas_factor(sub, diag, sup)
    x_factored = thomas_solve_factored(factor, rhs)

    print("4x4 reference test:")
    print(f"  numpy.linalg.solve:        {x_ref}")
    print(f"  thomas_solve:              {x_thomas}")
    print(f"  thomas_solve_factored:     {x_factored}")
    print(f"  max |thomas - numpy|     = {np.max(np.abs(x_thomas - x_ref)):.2e}")
    print(f"  max |factored - numpy|   = {np.max(np.abs(x_factored - x_ref)):.2e}")
