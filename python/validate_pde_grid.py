"""
Validation script for the Phase 3 Block 0 grid infrastructure.

Runs the following checks:

1.  Grid construction: sizes, spacings, endpoint values, monotonicity.
2.  Initial conditions: at every node, the IC equals the payoff
    (S - K)^+ resp. (K - S)^+ evaluated at S = K * e^x.
3.  Boundary conditions: agree with the Black-Scholes analytical
    asymptotics in the large-|x| limit, and with the IC at tau = 0.
4.  Internal consistency at corners: BC(tau=0) at the boundary node
    must equal IC at the same node.
5.  Stability numbers: closed-form check of fourier_number and
    courant_number formulae.
6.  Input validation: build_grid raises ValueError on every kind of
    bad input.

The exit code is 0 on success, 1 on failure. Designed to be run with
    python -m quantlib.validate_pde_grid
or directly.
"""

import numpy as np

from quantlib.pde import (
    Grid,
    build_grid,
    call_initial_condition,
    put_initial_condition,
    call_boundary_lower,
    call_boundary_upper,
    put_boundary_lower,
    put_boundary_upper,
    fourier_number,
    courant_number,
    is_explicit_stable,
)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

_n_pass = 0
_n_fail = 0


def check(label, condition, *, detail=""):
    """Record a single check result. condition must be a bool."""
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
def test_grid_construction():
    print("[1] Grid construction")
    N, M, T, sigma, r, K = 200, 400, 1.0, 0.20, 0.05, 100.0
    n_sigma = 4.0
    g = build_grid(N=N, M=M, T=T, sigma=sigma, r=r, K=K, n_sigma=n_sigma)

    check("g is a frozen Grid", isinstance(g, Grid))
    check("xs has length N+1",  g.xs.shape == (N + 1,))
    check("taus has length M+1", g.taus.shape == (M + 1,))

    expected_half = n_sigma * sigma * np.sqrt(T)
    check("x_min = -n_sigma * sigma * sqrt(T)",
          np.isclose(g.x_min, -expected_half),
          detail=f"got {g.x_min:.6f}, expected {-expected_half:.6f}")
    check("x_max = +n_sigma * sigma * sqrt(T)",
          np.isclose(g.x_max, +expected_half))

    check("dx   = (x_max - x_min) / N",
          np.isclose(g.dx, (g.x_max - g.x_min) / N))
    check("dtau = T / M",
          np.isclose(g.dtau, T / M))

    check("xs[0] = x_min",        np.isclose(g.xs[0],  g.x_min))
    check("xs[N] = x_max",        np.isclose(g.xs[-1], g.x_max))
    check("taus[0] = 0",          np.isclose(g.taus[0], 0.0))
    check("taus[M] = T",          np.isclose(g.taus[-1], T))

    # Spacing uniformity to machine precision.
    dx_actual = np.diff(g.xs)
    dt_actual = np.diff(g.taus)
    check("xs is uniformly spaced",
          np.allclose(dx_actual, g.dx, atol=1e-14))
    check("taus is uniformly spaced",
          np.allclose(dt_actual, g.dtau, atol=1e-14))

    check("mu = r - sigma^2/2",
          np.isclose(g.mu, r - 0.5 * sigma ** 2))


def test_initial_conditions():
    print("[2] Initial conditions")
    N, M, T, sigma, r, K = 100, 100, 1.0, 0.20, 0.05, 100.0
    g = build_grid(N=N, M=M, T=T, sigma=sigma, r=r, K=K)

    Ss = K * np.exp(g.xs)
    expected_call = np.maximum(Ss - K, 0.0)
    expected_put  = np.maximum(K - Ss, 0.0)

    ic_call = call_initial_condition(g.xs, K)
    ic_put  = put_initial_condition (g.xs, K)

    check("Call IC matches (S - K)^+ pointwise",
          np.allclose(ic_call, expected_call, atol=1e-12))
    check("Put IC matches (K - S)^+ pointwise",
          np.allclose(ic_put,  expected_put,  atol=1e-12))

    # Spot checks on representative nodes.
    j_atm = np.argmin(np.abs(g.xs))
    check("Call IC = 0 at the at-the-money node",
          np.isclose(ic_call[j_atm], 0.0, atol=1e-12),
          detail=f"x={g.xs[j_atm]:+.6f}, V={ic_call[j_atm]:.2e}")
    check("Put IC  = 0 at the at-the-money node",
          np.isclose(ic_put[j_atm],  0.0, atol=1e-12))

    # Monotonicity: call IC is non-decreasing, put IC non-increasing.
    check("Call IC non-decreasing in j",
          np.all(np.diff(ic_call) >= -1e-12))
    check("Put IC  non-increasing in j",
          np.all(np.diff(ic_put)  <= +1e-12))


def test_boundary_conditions():
    print("[3] Boundary conditions")
    N, M, T, sigma, r, K = 200, 200, 1.0, 0.20, 0.05, 100.0
    g = build_grid(N=N, M=M, T=T, sigma=sigma, r=r, K=K)

    bc_call_lo = call_boundary_lower(g)
    bc_call_hi = call_boundary_upper(g)
    bc_put_lo  = put_boundary_lower (g)
    bc_put_hi  = put_boundary_upper (g)

    check("All BC arrays have length M+1",
          all(arr.shape == (M + 1,) for arr in
              (bc_call_lo, bc_call_hi, bc_put_lo, bc_put_hi)))

    # Lower call and upper put BC are identically zero.
    check("Call lower BC is identically zero",
          np.allclose(bc_call_lo, 0.0, atol=1e-14))
    check("Put  upper BC is identically zero",
          np.allclose(bc_put_hi,  0.0, atol=1e-14))

    # Closed-form BS asymptotics. Call upper at tau = T is
    # K * (e^{x_max} - e^{-rT}); call upper at tau = 0 is
    # K * (e^{x_max} - 1) = call IC at j=N.
    expected_hi_T = K * (np.exp(g.x_max) - np.exp(-r * T))
    check("Call upper BC at tau = T matches asymptotic",
          np.isclose(bc_call_hi[-1], expected_hi_T, atol=1e-12))
    expected_hi_0 = K * (np.exp(g.x_max) - 1.0)
    check("Call upper BC at tau = 0 = K*(e^{x_max} - 1)",
          np.isclose(bc_call_hi[0], expected_hi_0, atol=1e-12))

    # Put lower at tau = T is K * (e^{-rT} - e^{x_min}); put lower at
    # tau = 0 is K * (1 - e^{x_min}) = put IC at j=0.
    expected_lo_T = K * (np.exp(-r * T) - np.exp(g.x_min))
    check("Put lower BC at tau = T matches asymptotic",
          np.isclose(bc_put_lo[-1], expected_lo_T, atol=1e-12))
    expected_lo_0 = K * (1.0 - np.exp(g.x_min))
    check("Put lower BC at tau = 0 = K*(1 - e^{x_min})",
          np.isclose(bc_put_lo[0], expected_lo_0, atol=1e-12))

    # Corner consistency: BC at tau=0 must agree with IC at that node.
    ic_call = call_initial_condition(g.xs, K)
    ic_put  = put_initial_condition (g.xs, K)
    check("Corner consistency: call BC_lo(0) = IC[0]",
          np.isclose(bc_call_lo[0], ic_call[0], atol=1e-12))
    check("Corner consistency: call BC_hi(0) = IC[N]",
          np.isclose(bc_call_hi[0], ic_call[-1], atol=1e-12))
    check("Corner consistency: put  BC_lo(0) = IC[0]",
          np.isclose(bc_put_lo[0],  ic_put[0],  atol=1e-12))
    check("Corner consistency: put  BC_hi(0) = IC[N]",
          np.isclose(bc_put_hi[0],  ic_put[-1], atol=1e-12))

    # Monotonicity in tau. Call upper is increasing in tau (intrinsic
    # value e^{x_max} - e^{-r tau}); put lower is decreasing.
    check("Call upper BC is non-decreasing in tau",
          np.all(np.diff(bc_call_hi) >= -1e-14))
    check("Put  lower BC is non-increasing in tau (for r > 0)",
          np.all(np.diff(bc_put_lo)  <= +1e-14))


def test_stability_numbers():
    print("[4] Stability numbers")
    sigma = 0.20
    dx = 0.008
    dtau = 0.0025
    r = 0.05
    mu = r - 0.5 * sigma * sigma   # = 0.03

    expected_alpha = 0.5 * sigma ** 2 * dtau / dx ** 2
    expected_nu    = 0.5 * abs(mu) * dtau / dx

    alpha = fourier_number(sigma, dtau, dx)
    nu    = courant_number(mu,    dtau, dx)

    check("fourier_number formula",
          np.isclose(alpha, expected_alpha, atol=1e-14),
          detail=f"got {alpha:.10f}, expected {expected_alpha:.10f}")
    check("courant_number formula",
          np.isclose(nu, expected_nu, atol=1e-14))
    check("courant_number is non-negative for negative drift",
          courant_number(-0.5, dtau, dx) >= 0)

    # Boundary case: alpha = 0.5 is the CFL limit, exactly stable.
    check("is_explicit_stable(0.5)  == True  (CFL boundary)",
          is_explicit_stable(0.5) is True)
    check("is_explicit_stable(0.5 + 1e-12) == False",
          is_explicit_stable(0.5 + 1e-12) is False)
    check("is_explicit_stable(0.49) == True",
          is_explicit_stable(0.49) is True)


def test_input_validation():
    print("[5] Input validation")
    base = dict(N=100, M=100, T=1.0, sigma=0.20, r=0.05, K=100.0)

    def expect_raise(label, **overrides):
        kwargs = {**base, **overrides}
        try:
            build_grid(**kwargs)
        except ValueError:
            check(label, True)
        except Exception as e:
            check(label, False, detail=f"raised {type(e).__name__}, not ValueError")
        else:
            check(label, False, detail="did not raise")

    expect_raise("N = 1 raises",      N=1)
    expect_raise("N = 0 raises",      N=0)
    expect_raise("M = 0 raises",      M=0)
    expect_raise("T = 0 raises",      T=0.0)
    expect_raise("T < 0 raises",      T=-1.0)
    expect_raise("sigma = 0 raises",  sigma=0.0)
    expect_raise("sigma < 0 raises",  sigma=-0.1)
    expect_raise("K = 0 raises",      K=0.0)
    expect_raise("K < 0 raises",      K=-100.0)
    expect_raise("n_sigma = 0 raises", n_sigma=0.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 72)
    print("Phase 3 Block 0 - Grid infrastructure validation")
    print("=" * 72)

    test_grid_construction()
    test_initial_conditions()
    test_boundary_conditions()
    test_stability_numbers()
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
