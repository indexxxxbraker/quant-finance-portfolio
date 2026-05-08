"""
Validation script for the Crank-Nicolson pricer of Phase 3 Block 3.

Six checks:

1. Cross-validation against Black-Scholes closed form on a parameter
   grid.
2. Quadratic convergence in time (with Rannacher): halving dtau at
   fixed large N reduces the error by ~4 (CN's signature).
3. Vanilla CN (without Rannacher) shows degraded convergence near
   the kink: the ratio drops from ~4 toward ~2. This is the empirical
   evidence of the strike-induced oscillation phenomenon.
4. Reproduction of BTCS via theta_march with theta=1: the generic
   stepper agrees with the dedicated BTCS pricer of Block 2 to
   round-off.
5. Put-call parity holds.
6. Input validation.

Run from python/ as
    python -m validate_cn
or directly.
"""

import numpy as np

from quantlib.black_scholes import call_price as bs_call_price
from quantlib.black_scholes import put_price  as bs_put_price
from quantlib.cn import cn_european_call, cn_european_put
from quantlib.btcs import btcs_european_call
from quantlib.theta_scheme import theta_march
from quantlib.pde import (
    build_grid, call_initial_condition,
    call_boundary_lower, call_boundary_upper,
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
def test_cross_validation():
    print("[1] Cross-validation against Black-Scholes closed form")
    test_cases = [
        # (S,    K,    r,    sigma, T,    N,   M)
        (100.0, 100.0, 0.05, 0.20, 1.00, 400, 200),
        ( 90.0, 100.0, 0.05, 0.20, 1.00, 400, 200),
        (110.0, 100.0, 0.05, 0.20, 1.00, 400, 200),
        (100.0, 100.0, 0.05, 0.30, 1.00, 400, 200),
        (100.0, 100.0, 0.05, 0.20, 0.25, 400, 200),
        (100.0, 100.0, 0.10, 0.20, 1.00, 400, 200),
        (100.0, 100.0, 0.00, 0.20, 1.00, 400, 200),
        ( 42.0,  40.0, 0.10, 0.20, 0.50, 400, 200),
    ]
    tol = 5e-3

    for S, K, r, sigma, T, N, M in test_cases:
        c_cn = cn_european_call(S, K, r, sigma, T, N=N, M=M)
        p_cn = cn_european_put (S, K, r, sigma, T, N=N, M=M)
        c_bs = bs_call_price(S, K, r, sigma, T)
        p_bs = bs_put_price (S, K, r, sigma, T)

        err_c = abs(c_cn - c_bs)
        err_p = abs(p_cn - p_bs)

        check(f"call S={S:6.2f} K={K:.0f} sigma={sigma:.2f} T={T:.2f}",
              err_c < tol,
              detail=f"CN={c_cn:.4f} BS={c_bs:.4f} err={err_c:.2e}")
        check(f"put  S={S:6.2f} K={K:.0f} sigma={sigma:.2f} T={T:.2f}",
              err_p < tol,
              detail=f"CN={p_cn:.4f} BS={p_bs:.4f} err={err_p:.2e}")


def test_quadratic_time_convergence():
    """The signature test of CN: halving dtau reduces error by ~4."""
    print("[2] Quadratic convergence in time (with Rannacher)")
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    c_bs = bs_call_price(S, K, r, sigma, T)

    # Fix N large so spatial error is below temporal. Vary M.
    N = 800
    Ms = [10, 20, 40]
    errors = []
    for M in Ms:
        c = cn_european_call(S, K, r, sigma, T, N=N, M=M,
                              rannacher_steps=2)
        e = abs(c - c_bs)
        errors.append(e)
        print(f"    N={N}, M={M:3d}: dtau={T/M:.4f}, error = {e:.4e}")

    ratio_01 = errors[0] / errors[1]
    ratio_12 = errors[1] / errors[2]
    print(f"    Ratio M=10/M=20  = {ratio_01:.2f}  (CN expects ~4)")
    print(f"    Ratio M=20/M=40  = {ratio_12:.2f}  (CN expects ~4)")

    # Loose lower bound 3.0 on the ratio. CN with Rannacher gives ~3.5-4
    # in practice; BTCS gives ~2. Anything above 3.0 is conclusively
    # better than BTCS, confirming O(dtau^2).
    check("Rannacher CN ratio M10/M20 > 3.0 (above BTCS first-order)",
          ratio_01 > 3.0, detail=f"got {ratio_01:.2f}")
    check("Rannacher CN ratio M20/M40 > 3.0",
          ratio_12 > 3.0, detail=f"got {ratio_12:.2f}")


def test_kink_oscillation_signature():
    """Without Rannacher, the kink artefact degrades convergence
    toward first order. This is the empirical signature of Block 3.
    """
    print("[3] CN without Rannacher: degraded convergence (kink artefact)")
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    c_bs = bs_call_price(S, K, r, sigma, T)

    N = 800
    Ms = [10, 20, 40]
    errors = []
    for M in Ms:
        c = cn_european_call(S, K, r, sigma, T, N=N, M=M,
                              rannacher_steps=0)
        e = abs(c - c_bs)
        errors.append(e)
        print(f"    N={N}, M={M:3d}: dtau={T/M:.4f}, error = {e:.4e}")

    ratio_01 = errors[0] / errors[1]
    ratio_12 = errors[1] / errors[2]
    print(f"    Ratio M=10/M=20  = {ratio_01:.2f}  (degraded toward 2)")
    print(f"    Ratio M=20/M=40  = {ratio_12:.2f}  (degraded toward 2)")

    # Vanilla CN ratios should be noticeably below the Rannacher
    # values: empirically 2.0-2.5, never above 3.0 in this regime.
    check("vanilla CN ratio M10/M20 < 3.0 (degraded from CN)",
          ratio_01 < 3.0, detail=f"got {ratio_01:.2f}")
    check("vanilla CN ratio M20/M40 < 3.0",
          ratio_12 < 3.0, detail=f"got {ratio_12:.2f}")
    # And the errors themselves should be MUCH larger than with Rannacher.
    check("vanilla CN error at M=10 is at least 5x worse than Rannacher CN",
          errors[0] > 5 * 8e-3,
          detail=f"vanilla M=10 err = {errors[0]:.2e} vs Rannacher ~8e-3")


def test_btcs_reproduced_via_theta_scheme():
    """theta_march(theta=1) should reproduce Block 2's BTCS pricer."""
    print("[4] Reproduction of BTCS via theta-scheme with theta=1")
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    K_, N, M = 100.0, 200, 800

    grid = build_grid(N=N, M=M, T=T, sigma=sigma, r=r, K=K_)
    V0 = call_initial_condition(grid.xs, grid.K)
    bc_lo = call_boundary_lower(grid)
    bc_hi = call_boundary_upper(grid)
    V_via_theta = theta_march(grid, V0, 1.0, bc_lo, bc_hi)

    x_0 = np.log(S / K_)
    c_via_theta = float(np.interp(x_0, grid.xs, V_via_theta))
    c_btcs = btcs_european_call(S, K_, r, sigma, T, N=N, M=M)
    diff = abs(c_via_theta - c_btcs)
    check("theta_march(theta=1) matches btcs_european_call",
          diff < 1e-12,
          detail=f"|diff| = {diff:.2e}")


def test_put_call_parity():
    print("[5] Put-call parity")
    K, r, sigma, T = 100.0, 0.05, 0.20, 1.0
    N, M = 400, 200
    diffs = []
    for S in [85.0, 95.0, 100.0, 105.0, 115.0]:
        c = cn_european_call(S, K, r, sigma, T, N=N, M=M)
        p = cn_european_put (S, K, r, sigma, T, N=N, M=M)
        rhs = S - K * np.exp(-r * T)
        diffs.append(abs((c - p) - rhs))
    check("parity holds for all spots",
          max(diffs) < 1e-2,
          detail=f"max |C-P-(S-Ke^-rT)| = {max(diffs):.2e}")


def test_input_validation():
    print("[6] Input validation")
    # S = 0, S < 0
    try:
        cn_european_call(0.0, 100.0, 0.05, 0.20, 1.0, N=200, M=200)
        check("S=0 raises", False)
    except ValueError:
        check("S=0 raises", True)

    # rannacher_steps < 0
    try:
        cn_european_call(100, 100.0, 0.05, 0.20, 1.0, N=200, M=200,
                          rannacher_steps=-1)
        check("rannacher_steps=-1 raises", False)
    except ValueError:
        check("rannacher_steps=-1 raises", True)

    # rannacher_steps > M
    try:
        cn_european_call(100, 100.0, 0.05, 0.20, 1.0, N=200, M=2,
                          rannacher_steps=5)
        check("rannacher_steps > M raises", False)
    except ValueError:
        check("rannacher_steps > M raises", True)

    # Out of domain
    try:
        cn_european_call(1000, 100.0, 0.05, 0.20, 1.0, N=200, M=200)
        check("out-of-domain spot raises", False)
    except ValueError:
        check("out-of-domain spot raises", True)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 72)
    print("Phase 3 Block 3 - CN pricer validation")
    print("=" * 72)

    test_cross_validation()
    test_quadratic_time_convergence()
    test_kink_oscillation_signature()
    test_btcs_reproduced_via_theta_scheme()
    test_put_call_parity()
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
