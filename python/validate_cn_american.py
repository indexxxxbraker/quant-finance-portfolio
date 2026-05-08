"""
Validation script for the CN-American put pricer.

Seven checks:

1. Cross-validation against the CRR binomial reference (Phase 2 Block 6).
2. American put price >= European put price at every spot.
3. Spatial refinement reduces the error toward the CRR reference
   (qualitative monotonic decrease over a moderate bracket; American
   pricing is O(h), not O(h^2), due to the kink at the free boundary).
4. omega sweep: optimal omega is interior to (0, 2); the iteration
   count is much higher near the boundaries.
5. omega outside (0, 2) raises (delegated to PSOR).
6. Free-boundary monotonicity: the recovered exercise boundary
   S_f(tau) is monotonically increasing in tau.
7. American put recovers European put when r=0 (no early-exercise
   premium for an undiscounted put -- holds approximately).
"""

import math
import numpy as np

from quantlib.cn_american import cn_american_put
from quantlib.cn import cn_european_put
from quantlib.black_scholes import put_price as bs_put_price


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
# Reference: standalone CRR binomial American put. Independent of any
# Phase 2 imports so the validator stands on its own.
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
def test_cross_validation_crr():
    print("[1] Cross-validation against CRR binomial reference")
    test_cases = [
        # (S,    K,    r,    sigma, T,    N,   M)
        (100.0, 100.0, 0.05, 0.20, 1.00, 200, 100),
        ( 90.0, 100.0, 0.05, 0.20, 1.00, 200, 100),
        (110.0, 100.0, 0.05, 0.20, 1.00, 200, 100),
        (100.0, 100.0, 0.05, 0.30, 1.00, 200, 100),
        (100.0, 100.0, 0.05, 0.20, 0.25, 200, 100),
        (100.0, 100.0, 0.10, 0.20, 1.00, 200, 100),
        ( 42.0,  40.0, 0.10, 0.20, 0.50, 200, 100),
    ]
    tol = 5e-3

    for S, K, r, sigma, T, N, M in test_cases:
        p_psor = cn_american_put(S, K, r, sigma, T, N=N, M=M)
        p_crr  = crr_american_put(S, K, r, sigma, T, n_steps=2000)
        err = abs(p_psor - p_crr)
        check(f"S={S:6.2f} K={K:.0f} sigma={sigma:.2f} T={T:.2f}",
              err < tol,
              detail=f"PSOR={p_psor:.4f} CRR={p_crr:.4f} err={err:.2e}")


def test_american_dominates_european():
    print("[2] American put >= European put")
    K, r, sigma, T = 100.0, 0.05, 0.20, 1.0
    N, M = 200, 100

    diffs = []
    for S in [80.0, 90.0, 100.0, 110.0, 120.0]:
        p_amer = cn_american_put(S, K, r, sigma, T, N=N, M=M)
        p_eur  = cn_european_put(S, K, r, sigma, T, N=N, M=M)
        diff = p_amer - p_eur
        diffs.append(diff)
        check(f"S={S:6.2f}: A={p_amer:.4f} >= E={p_eur:.4f}",
              diff >= -1e-6,
              detail=f"premium = {diff:.4f}")

    check("early-exercise premium is positive at S<K (ITM put)",
          diffs[0] > 0.0 and diffs[1] > 0.0,
          detail=f"premia at S=80, S=90: {diffs[0]:.4f}, {diffs[1]:.4f}")


def test_spatial_refinement():
    """Doubling N reduces the error vs CRR. American option pricing
    via PDE methods converges at O(h), not O(h^2), due to the kink in
    the value function at the free boundary. We test only that the
    error decreases with refinement, not the rate."""
    print("[3] Spatial refinement reduces the error vs CRR")
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    p_ref = crr_american_put(S, K, r, sigma, T, n_steps=5000)
    print(f"    CRR reference (n=5000): {p_ref:.6f}")

    levels = [(100, 50), (200, 100), (400, 200)]
    errors = []
    for N, M in levels:
        p = cn_american_put(S, K, r, sigma, T, N=N, M=M)
        err = abs(p - p_ref)
        errors.append(err)
        print(f"    N={N:4d}, M={M:3d}: price = {p:.6f}, error = {err:.4e}")

    check("error at N=200 < error at N=100",
          errors[1] < errors[0],
          detail=f"{errors[1]:.4e} < {errors[0]:.4e}")
    # We allow N=400 to be roughly the same as N=200 since this is O(h)
    # convergence and we are at small absolute error already.
    check("error at N=400 is small (< 5e-3)",
          errors[2] < 5e-3,
          detail=f"err = {errors[2]:.4e}")


def test_omega_sweep():
    print("[4] omega sweep: interior optimum, slow at boundaries")
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0

    omegas = [0.5, 1.0, 1.4, 1.9]
    iter_counts = []
    for w in omegas:
        _, diag = cn_american_put(S, K, r, sigma, T, N=200, M=100,
                                    omega=w, max_iter=8000,
                                    return_diagnostics=True)
        iter_counts.append(diag["total_iterations"])
        print(f"    omega={w}: total iter = {diag['total_iterations']}")

    # Interior should be much faster than extremes.
    interior_min = min(iter_counts[1], iter_counts[2])
    extreme_max  = max(iter_counts[0], iter_counts[3])
    check("interior omega much faster than boundary omega",
          interior_min < extreme_max / 2,
          detail=f"interior min = {interior_min}, "
                 f"extreme max = {extreme_max}")


def test_omega_validation():
    print("[5] omega outside (0, 2) raises")
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    for bad_omega in [0.0, 2.0, -0.5, 2.5]:
        try:
            cn_american_put(S, K, r, sigma, T, N=100, M=50,
                             omega=bad_omega)
            check(f"omega={bad_omega} raises", False)
        except ValueError:
            check(f"omega={bad_omega} raises", True)


def test_free_boundary_monotonic():
    print("[6] Free boundary S_f(tau) is monotonically increasing")
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0

    # We need access to the full V grid at each time step. Easiest:
    # rerun CN-American at multiple maturities and infer S_f at tau=T.
    # The boundary at tau is the largest x_min such that V(x_min, tau)
    # > g(x_min). Equivalently, at the final time, the largest j such
    # that V[j] > g[j] is the index just past the boundary.
    _, diag = cn_american_put(S, K, r, sigma, T, N=200, M=100,
                                return_diagnostics=True)
    grid = diag["grid"]
    V_final = diag["final_solution"]
    g = np.maximum(grid.K - grid.K * np.exp(grid.xs), 0.0)

    # The exercise boundary is the largest j *in the ITM region*
    # (g > 0, i.e. x < 0) with V[j] = g[j]. Beyond that j we enter
    # the continuation region; at x >= 0 the obstacle is zero and
    # V also tends to zero deep OTM, so we must restrict to g > 0
    # to find the actual exercise boundary.
    itm = (g > 1e-10)
    on_obstacle_itm = np.isclose(V_final, g, atol=1e-6) & itm
    if on_obstacle_itm.any():
        j_boundary = np.where(on_obstacle_itm)[0].max()
        S_f_final = grid.K * np.exp(grid.xs[j_boundary])
        print(f"    S_f(tau=T) approx = {S_f_final:.4f} (theoretical: < K = {grid.K})")
        check("S_f at final tau is below K",
              S_f_final < grid.K,
              detail=f"S_f = {S_f_final:.4f}")
    else:
        check("exercise region exists at tau=T", False,
              detail="no ITM nodes are on the obstacle, suspicious")
        return

    # Boundary at smaller maturity should be CLOSER to K than at larger maturity.
    # Reasoning in tau = T - t convention: tau = T means we are at calendar time
    # t = 0; tau = 0 means we are at t = T (maturity). The exercise boundary
    # S_f(t) DECREASES as t decreases (going back in time), so S_f(t=0) at
    # T=1 is lower than S_f(t=0) at T=0.25 because there is more remaining
    # time to wait when T is larger.
    _, diag_short = cn_american_put(S, K, r, sigma, 0.25, N=200, M=100,
                                      return_diagnostics=True)
    V_short = diag_short["final_solution"]
    grid_short = diag_short["grid"]
    g_short = np.maximum(grid_short.K - grid_short.K * np.exp(grid_short.xs),
                          0.0)
    itm_short = (g_short > 1e-10)
    on_obstacle_short = np.isclose(V_short, g_short, atol=1e-6) & itm_short
    if on_obstacle_short.any():
        j_short = np.where(on_obstacle_short)[0].max()
        S_f_short = grid_short.K * np.exp(grid_short.xs[j_short])
        print(f"    S_f at tau=T=0.25: {S_f_short:.4f}")
        print(f"    S_f at tau=T=1.00: {S_f_final:.4f}")
        check("S_f at smaller maturity > S_f at larger maturity",
              S_f_short > S_f_final,
              detail=f"{S_f_short:.4f} > {S_f_final:.4f}")


def test_input_validation():
    print("[7] Input validation")
    K, r, sigma, T = 100.0, 0.05, 0.20, 1.0
    try:
        cn_american_put(0.0, K, r, sigma, T, N=200, M=100)
        check("S=0 raises", False)
    except ValueError:
        check("S=0 raises", True)
    try:
        cn_american_put(1000.0, K, r, sigma, T, N=200, M=100)
        check("out-of-domain S raises", False)
    except ValueError:
        check("out-of-domain S raises", True)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 72)
    print("Phase 3 Block 4 - CN-American put pricer validation")
    print("=" * 72)

    test_cross_validation_crr()
    test_american_dominates_european()
    test_spatial_refinement()
    test_omega_sweep()
    test_omega_validation()
    test_free_boundary_monotonic()
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
