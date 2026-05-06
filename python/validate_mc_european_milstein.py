"""Validation suite for the Milstein Monte Carlo pricer of the
European call under geometric Brownian motion.

Implements four tests as in Phase 2 Block 1.2.2 writeup, Section 4.

The most distinctive of these is Test 1, which runs Euler and
Milstein on the *same* grid of step sizes and on the *same* Brownian
paths, and reports both fitted slopes side by side. The expected
result is slope +0.5 for Euler and slope +1.0 for Milstein, the
direct empirical confirmation of the strong-order asymmetry that is
the conceptual heart of Block 1.2.

Run from python/:

    python validate_mc_european_milstein.py
"""

import numpy as np

from quantlib.black_scholes import call_price
from quantlib.monte_carlo import mc_european_call_milstein
from quantlib.gbm import (
    simulate_terminal_euler,
    simulate_terminal_milstein,
    _gbm_exact_from_brownian,
)


def _format_pass_fail(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


# Common parameters used across multiple tests.
S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.00
BS_PRICE = call_price(S0, K, r, sigma, T)


# =====================================================================
# Test 1: side-by-side strong order, Euler vs Milstein
# =====================================================================

def test_strong_order_side_by_side() -> bool:
    """Verify strong order 1/2 (Euler) and 1 (Milstein) on a shared grid.

    For each step count N in the grid, both schemes are run on the
    *same* matrix of Brownian increments, so their pathwise errors
    can be compared directly. The exact solution is built from
    W_T = sum of those increments via the closed-form GBM formula.
    """
    print("\n" + "=" * 78)
    print("TEST 1: Side-by-side strong order")
    print("        Expected: slope +0.5 (Euler), +1.0 (Milstein)")
    print("=" * 78)

    n_values = [10, 30, 100, 300, 1000, 3000]
    n_paths = 10_000
    rng_master = np.random.default_rng(0)

    log_h = []
    log_strong_euler = []
    log_strong_milstein = []

    print(f"  {'N':>5}  {'h':>9}  {'strong_euler':>15}  "
          f"{'strong_milstein':>15}  ratio")
    for N in n_values:
        h = T / N
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        delta_W = rng.normal(loc=0.0, scale=np.sqrt(h),
                             size=(n_paths, N))

        # Both schemes consume the same delta_W.
        S_T_euler = simulate_terminal_euler(
            S0, r, sigma, T, N, n_paths, delta_W=delta_W
        )
        S_T_milstein = simulate_terminal_milstein(
            S0, r, sigma, T, N, n_paths, delta_W=delta_W
        )

        # Exact terminal on the same Brownian path.
        W_T = delta_W.sum(axis=1)
        S_T_exact = _gbm_exact_from_brownian(S0, r, sigma, T, W_T)

        strong_euler = float(np.mean(np.abs(S_T_exact - S_T_euler)))
        strong_milstein = float(np.mean(np.abs(S_T_exact - S_T_milstein)))
        ratio = strong_euler / strong_milstein

        log_h.append(np.log(h))
        log_strong_euler.append(np.log(strong_euler))
        log_strong_milstein.append(np.log(strong_milstein))

        print(f"  {N:>5d}  {h:>9.5f}  {strong_euler:>15.5e}  "
              f"{strong_milstein:>15.5e}  {ratio:>5.2f}x")

    slope_euler, _ = np.polyfit(log_h, log_strong_euler, 1)
    slope_milstein, _ = np.polyfit(log_h, log_strong_milstein, 1)

    tol_euler = 0.10
    tol_milstein = 0.15

    pass_euler = abs(slope_euler - 0.5) < tol_euler
    pass_milstein = abs(slope_milstein - 1.0) < tol_milstein

    print(f"\n  Fitted slopes:")
    print(f"    Euler   : {slope_euler:+.4f}  "
          f"(expected +0.5, tolerance {tol_euler}) "
          f"[{_format_pass_fail(pass_euler)}]")
    print(f"    Milstein: {slope_milstein:+.4f}  "
          f"(expected +1.0, tolerance {tol_milstein}) "
          f"[{_format_pass_fail(pass_milstein)}]")

    overall = pass_euler and pass_milstein
    print(f"\n  Test 1 overall: {_format_pass_fail(overall)}")
    return overall


# =====================================================================
# Test 2: empirical weak order under common random numbers (Milstein)
# =====================================================================

def test_weak_order() -> bool:
    """Verify weak order 1 of Milstein on the call payoff.

    Same CRN methodology as the Block 1.2.1 weak order test: drive
    both Milstein and exact with the same Brownian path, look at
    the pathwise difference of payoffs.
    """
    print("\n" + "=" * 78)
    print("TEST 2: Empirical weak order under CRN (Milstein)")
    print("        Expected: slope +1.0 (same as Euler)")
    print("=" * 78)

    n_values = [10, 20, 50, 100]
    n_paths = 200_000
    discount = np.exp(-r * T)

    rng_master = np.random.default_rng(7)

    log_h = []
    log_weak = []

    for N in n_values:
        h = T / N
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        delta_W = rng.normal(loc=0.0, scale=np.sqrt(h),
                             size=(n_paths, N))

        S_T_milstein = simulate_terminal_milstein(
            S0, r, sigma, T, N, n_paths, delta_W=delta_W
        )
        W_T = delta_W.sum(axis=1)
        S_T_exact = _gbm_exact_from_brownian(S0, r, sigma, T, W_T)

        payoff_milstein = discount * np.maximum(S_T_milstein - K, 0.0)
        payoff_exact = discount * np.maximum(S_T_exact - K, 0.0)
        weak_err = float(np.abs(np.mean(payoff_milstein - payoff_exact)))

        log_h.append(np.log(h))
        log_weak.append(np.log(weak_err))
        print(f"  N={N:5d}  h={h:.5f}  weak_err={weak_err:.5e}")

    slope, _ = np.polyfit(log_h, log_weak, 1)

    expected_slope = 1.0
    tolerance = 0.20
    passed = abs(slope - expected_slope) < tolerance

    print(f"\n  Fitted slope: {slope:+.4f}  (expected {expected_slope:+.4f})")
    print(f"  Tolerance   : {tolerance}")
    print(f"\n  Test 2 overall: {_format_pass_fail(passed)}")
    return passed


# =====================================================================
# Test 3: coherence with exact pricer at large n_steps
# =====================================================================

def test_coherence_with_exact() -> bool:
    """Sanity check: the Milstein pricer at large n_steps agrees with
    BS within a few half-widths.
    """
    print("\n" + "=" * 78)
    print("TEST 3: Coherence with exact pricer at large n_steps")
    print("=" * 78)

    n_paths = 20_000
    for N in [500, 2000]:
        result = mc_european_call_milstein(
            S0, K, r, sigma, T, n_steps=N, n_paths=n_paths, seed=11,
        )
        within = abs(result.estimate - BS_PRICE) <= 3.0 * result.half_width
        print(f"  N={N:5d}  estimate={result.estimate:.5f}  "
              f"BS={BS_PRICE:.5f}  hw={result.half_width:.5f}  "
              f"|err|={abs(result.estimate - BS_PRICE):.5f}  "
              f"[{_format_pass_fail(within)}]")

        if not within:
            print("\n  Test 3 overall: FAIL")
            return False

    print(f"\n  Test 3 overall: PASS")
    return True


# =====================================================================
# Test 4: input validation
# =====================================================================

def test_input_validation() -> bool:
    """Mechanical checks: the pricer rejects bad inputs."""
    print("\n" + "=" * 78)
    print("TEST 4: Input validation and basic invariances (Milstein)")
    print("=" * 78)

    rng = np.random.default_rng(0)

    cases = [
        ("S must be positive",
         lambda: mc_european_call_milstein(
             -1.0, K, r, sigma, T, 50, 1000, rng=rng)),
        ("K must be positive",
         lambda: mc_european_call_milstein(
             S0, 0.0, r, sigma, T, 50, 1000, rng=rng)),
        ("sigma must be positive",
         lambda: mc_european_call_milstein(
             S0, K, r, -0.10, T, 50, 1000, rng=rng)),
        ("T must be positive",
         lambda: mc_european_call_milstein(
             S0, K, r, sigma, 0.0, 50, 1000, rng=rng)),
        ("n_steps must be at least 1",
         lambda: mc_european_call_milstein(
             S0, K, r, sigma, T, 0, 1000, rng=rng)),
        ("n_paths must be at least 2",
         lambda: mc_european_call_milstein(
             S0, K, r, sigma, T, 50, 1, rng=rng)),
    ]

    all_pass = True
    for label, call in cases:
        try:
            call()
            print(f"  [{label}]: FAIL (no exception raised)")
            all_pass = False
        except (ValueError, TypeError):
            print(f"  [{label}]: PASS")

    try:
        mc_european_call_milstein(S0, K, -0.02, sigma, T, 50, 1000, rng=rng)
        print(f"  [negative r is admissible]: PASS")
    except Exception as e:
        print(f"  [negative r is admissible]: FAIL ({type(e).__name__})")
        all_pass = False

    print(f"\n  Test 4 overall: {_format_pass_fail(all_pass)}")
    return all_pass


# =====================================================================
# Main
# =====================================================================

def main():
    np.set_printoptions(linewidth=120, precision=6, suppress=True)

    results = {
        "Test 1 (strong: Euler vs Milstein)" : test_strong_order_side_by_side(),
        "Test 2 (weak order, CRN)"           : test_weak_order(),
        "Test 3 (coherence with exact)"      : test_coherence_with_exact(),
        "Test 4 (input validation)"          : test_input_validation(),
    }

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for name, passed in results.items():
        print(f"  {name:38}: {_format_pass_fail(passed)}")

    overall = all(results.values())
    print("\n  " + ("ALL TESTS PASSED" if overall else "SOME TESTS FAILED"))

    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
