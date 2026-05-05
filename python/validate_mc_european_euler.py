"""Validation suite for the Euler-Maruyama Monte Carlo pricer of the
European call under geometric Brownian motion.

Implements the four tests of Phase 2 Block 1.2.1 writeup, Section 4:

    Test 1: empirical strong order. Slope -1/2 in log-log of
            E[|S_T_exact - S_T_euler|] vs h.
    Test 2: empirical weak order. Slope -1 in log-log of
            |E[Phi(S_T_euler) - Phi(S_T_exact)]| vs h, computed
            under common random numbers (CRN) for variance control.
    Test 3: coherence with exact pricer. As n_steps -> infty at fixed
            n_paths (and shared Brownian), the Euler estimate must
            approach the exact-sampler estimate.
    Test 4: input validation and basic invariances.

Run from python/:

    python validate_mc_european_euler.py
"""

import numpy as np

from quantlib.black_scholes import call_price
from quantlib.monte_carlo import (
    mc_european_call_euler,
    simulate_path_euler,
    simulate_terminal_euler,
    _gbm_exact_from_brownian,
)


def _format_pass_fail(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


# Common parameters used across multiple tests.
S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.00
BS_PRICE = call_price(S0, K, r, sigma, T)


# =====================================================================
# Test 1: empirical strong order (expected slope = -1/2)
# =====================================================================

def test_strong_order() -> bool:
    """Verify strong order 1/2 of Euler-Maruyama on GBM.

    For each step count N in a logarithmic grid, simulates n_paths
    Euler paths sharing their Brownian increments with the exact
    closed-form solution, and estimates E[|S_T_exact - S_T_euler|].
    Fits a line in log-log; the slope should be -0.5.
    """
    print("\n" + "=" * 68)
    print("TEST 1: Empirical strong order (expected slope = -1/2)")
    print("=" * 68)

    n_values = [10, 30, 100, 300, 1000, 3000]
    n_paths = 10_000
    rng_master = np.random.default_rng(0)

    log_h = []
    log_strong = []

    for N in n_values:
        h = T / N
        # Use independent rng per N so successive runs do not share noise.
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        delta_W = rng.normal(loc=0.0, scale=np.sqrt(h),
                             size=(n_paths, N))

        # Euler path on these increments.
        S_T_euler = simulate_terminal_euler(
            S0, r, sigma, T, N, n_paths, delta_W=delta_W
        )
        # Exact terminal on the same Brownian (W_T = sum of increments).
        W_T = delta_W.sum(axis=1)
        S_T_exact = _gbm_exact_from_brownian(S0, r, sigma, T, W_T)

        strong_err = float(np.mean(np.abs(S_T_exact - S_T_euler)))
        log_h.append(np.log(h))
        log_strong.append(np.log(strong_err))
        print(f"  N={N:5d}  h={h:.5f}  strong_err={strong_err:.5e}")

    slope, _ = np.polyfit(log_h, log_strong, 1)

    expected_slope = 0.5
    tolerance = 0.10
    passed = abs(slope - expected_slope) < tolerance

    print(f"\n  Fitted slope: {slope:+.4f}  (expected {expected_slope:+.4f})")
    print(f"  Tolerance   : {tolerance}")
    print(f"\n  Test 1 overall: {_format_pass_fail(passed)}")
    return passed


# =====================================================================
# Test 2: empirical weak order (expected slope = -1)
# =====================================================================

def test_weak_order() -> bool:
    """Verify weak order 1 of Euler-Maruyama on the call payoff.

    Uses common random numbers between Euler and exact: both
    estimators consume the same Brownian increments. The weak error
    is then estimated as the mean of (Phi_euler - Phi_exact) over
    paths, which has dramatically smaller variance than estimating
    each estimator independently and subtracting.

    See Block 1.2.1 writeup, Section 4.3, for the variance argument
    that motivates CRN here.
    """
    print("\n" + "=" * 68)
    print("TEST 2: Empirical weak order under CRN (expected slope = -1)")
    print("=" * 68)

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

        S_T_euler = simulate_terminal_euler(
            S0, r, sigma, T, N, n_paths, delta_W=delta_W
        )
        W_T = delta_W.sum(axis=1)
        S_T_exact = _gbm_exact_from_brownian(S0, r, sigma, T, W_T)

        # Pathwise difference of the discounted call payoff.
        payoff_euler = discount * np.maximum(S_T_euler - K, 0.0)
        payoff_exact = discount * np.maximum(S_T_exact - K, 0.0)
        weak_err = float(np.abs(np.mean(payoff_euler - payoff_exact)))

        log_h.append(np.log(h))
        log_weak.append(np.log(weak_err))
        print(f"  N={N:5d}  h={h:.5f}  weak_err={weak_err:.5e}")

    slope, _ = np.polyfit(log_h, log_weak, 1)

    expected_slope = 1.0
    # Wider tolerance than strong: weak order 1 is more affected by
    # the moderate-h regime and by the residual MC noise on the CRN
    # difference, especially for the smallest N where the weak error
    # itself is largest and the slope estimate is least sensitive.
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
    """As n_steps -> infty, the Euler pricer at fixed n_paths and
    shared Brownian must approach the BS price.

    This is a sanity check: it verifies that the implementation
    pipeline (sampler + payoff + reducer) does not have a structural
    bug. A fast convergence to BS at n_steps in the thousands
    confirms the algorithm is consistent.
    """
    print("\n" + "=" * 68)
    print("TEST 3: Coherence with exact pricer at large n_steps")
    print("=" * 68)

    n_paths = 20_000
    # Two large step counts; the Euler estimate at large N should be
    # within 2-3 half-widths of the BS price.
    for N in [500, 2000]:
        result = mc_european_call_euler(
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
# Test 4: input validation and basic invariances
# =====================================================================

def test_input_validation() -> bool:
    """Mechanical checks: the pricer rejects bad inputs and accepts
    the documented admissible ones.
    """
    print("\n" + "=" * 68)
    print("TEST 4: Input validation and basic invariances")
    print("=" * 68)

    rng = np.random.default_rng(0)

    cases = [
        ("S must be positive",
         lambda: mc_european_call_euler(
             -1.0, K, r, sigma, T, 50, 1000, rng=rng)),
        ("K must be positive",
         lambda: mc_european_call_euler(
             S0, 0.0, r, sigma, T, 50, 1000, rng=rng)),
        ("sigma must be positive",
         lambda: mc_european_call_euler(
             S0, K, r, -0.10, T, 50, 1000, rng=rng)),
        ("T must be positive",
         lambda: mc_european_call_euler(
             S0, K, r, sigma, 0.0, 50, 1000, rng=rng)),
        ("n_steps must be at least 1",
         lambda: mc_european_call_euler(
             S0, K, r, sigma, T, 0, 1000, rng=rng)),
        ("n_paths must be at least 2",
         lambda: mc_european_call_euler(
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

    # Negative rates are admissible.
    try:
        mc_european_call_euler(S0, K, -0.02, sigma, T, 50, 1000, rng=rng)
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
        "Test 1 (strong order)"          : test_strong_order(),
        "Test 2 (weak order, CRN)"       : test_weak_order(),
        "Test 3 (coherence with exact)"  : test_coherence_with_exact(),
        "Test 4 (input validation)"      : test_input_validation(),
    }

    print("\n" + "=" * 68)
    print("SUMMARY")
    print("=" * 68)
    for name, passed in results.items():
        print(f"  {name:35}: {_format_pass_fail(passed)}")

    overall = all(results.values())
    print("\n  " + ("ALL TESTS PASSED" if overall else "SOME TESTS FAILED"))

    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
