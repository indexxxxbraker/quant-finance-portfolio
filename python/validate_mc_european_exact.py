"""Validation suite for the Monte Carlo European call pricer with
exact GBM simulation.

Triangulated validation (Phase 2 Block 1.1 writeup, Section 6):

    Test 1: containment frequency of the BS price by the MC CI.
    Test 2: empirical convergence rate (slope -1/2 in log-log).
    Test 3: agreement of sample variance with closed form.
    Test 4: monotonicities (with common random numbers) and
            asymptotic limits.

Run from the python/ directory:

    cd python/
    python validate_mc_european_exact.py
"""

import numpy as np

from quantlib.black_scholes import call_price, call_payoff_variance
from quantlib.monte_carlo import mc_european_call_exact


# =====================================================================
# Reusable utilities
# =====================================================================

def _format_pass_fail(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


# =====================================================================
# Test 1: containment frequency of the BS price
# =====================================================================

def test_containment_frequency() -> bool:
    """For multiple parameter sets, verify that the empirical
    containment frequency of the BS price by the MC CI is close to
    the nominal 0.95.

    Acceptance interval is calibrated to the binomial sampling
    uncertainty: with n_seeds = 200 trials at p = 0.95, two standard
    deviations correspond to roughly +/- 0.03, hence the [0.92, 0.98]
    band.
    """
    print("\n" + "=" * 68)
    print("TEST 1: Containment frequency of the BS price by the MC CI")
    print("=" * 68)

    grid = [
        ("ATM",          100.0, 100.0, 0.05, 0.20, 1.00),
        ("ITM",          110.0, 100.0, 0.05, 0.20, 1.00),
        ("OTM",           90.0, 100.0, 0.05, 0.20, 1.00),
        ("LongMaturity", 100.0, 100.0, 0.05, 0.30, 5.00),
    ]

    n_seeds = 200
    n_paths = 10_000
    confidence_level = 0.95
    lower_acceptance = 0.92
    upper_acceptance = 0.98

    all_pass = True
    for label, S, K, r, sigma, T in grid:
        bs_price = call_price(S, K, r, sigma, T)

        contained = 0
        for seed in range(n_seeds):
            result = mc_european_call_exact(
                S, K, r, sigma, T, n_paths,
                seed=seed, confidence_level=confidence_level,
            )
            if abs(result.estimate - bs_price) <= result.half_width:
                contained += 1

        rate = contained / n_seeds
        case_pass = lower_acceptance <= rate <= upper_acceptance
        all_pass &= case_pass

        print(f"  [{label:13}] BS={bs_price:8.4f}  "
              f"containment={rate:5.3f}  "
              f"[{_format_pass_fail(case_pass)}]")

    print(f"\n  Test 1 overall: {_format_pass_fail(all_pass)}")
    return all_pass


# =====================================================================
# Test 2: empirical convergence rate
# =====================================================================

def test_convergence_rate() -> bool:
    """For a fixed parameter set, verify that the RMSE of the MC
    estimator decays as n^{-1/2}, by linear fit on log-log axes.
    """
    print("\n" + "=" * 68)
    print("TEST 2: Empirical convergence rate (expected slope = -0.5)")
    print("=" * 68)

    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.00
    bs_price = call_price(S, K, r, sigma, T)

    # Logarithmic grid of n spanning four orders of magnitude.
    n_values = [100, 316, 1_000, 3_162, 10_000, 31_623, 100_000]
    n_seeds_per_n = 50

    log_n = []
    log_rmse = []

    for n in n_values:
        squared_errors = np.empty(n_seeds_per_n)
        for k in range(n_seeds_per_n):
            # Distinct seeds per (n, k) so that no two runs share noise.
            seed = 1_000_000 * n + k
            result = mc_european_call_exact(
                S, K, r, sigma, T, n, seed=seed,
            )
            squared_errors[k] = (result.estimate - bs_price) ** 2

        rmse = float(np.sqrt(np.mean(squared_errors)))
        log_n.append(np.log(n))
        log_rmse.append(np.log(rmse))
        print(f"  n={n:7d}  RMSE={rmse:.5e}")

    slope, intercept = np.polyfit(log_n, log_rmse, 1)

    expected_slope = -0.5
    tolerance = 0.10
    passed = abs(slope - expected_slope) < tolerance

    print(f"\n  Fitted slope: {slope:+.4f}  (expected {expected_slope:+.4f})")
    print(f"  Tolerance   : {tolerance}")
    print(f"\n  Test 2 overall: {_format_pass_fail(passed)}")
    return passed


# =====================================================================
# Test 3: agreement of sample variance with closed form
# =====================================================================

def test_variance_agreement() -> bool:
    """For multiple parameter sets, verify that the sample variance of
    the discounted payoff agrees with the closed-form variance derived
    in the writeup, within a tolerance commensurate with the variance
    of the variance estimator at the prescribed sample size.

    The 5% relative tolerance accommodates the high kurtosis of the
    call payoff (mass at zero plus a long right tail), which makes the
    sample variance estimator converge slower than for nicer
    distributions.
    """
    print("\n" + "=" * 68)
    print("TEST 3: Sample variance vs closed-form variance")
    print("=" * 68)

    grid = [
        ("ATM",     100.0, 100.0, 0.05, 0.20, 1.00),
        ("ITM",     110.0, 100.0, 0.05, 0.20, 1.00),
        ("OTM",      90.0, 100.0, 0.05, 0.20, 1.00),
        ("HighVol", 100.0, 100.0, 0.05, 0.50, 1.00),
    ]

    n_paths = 1_000_000
    rel_tol = 0.05

    all_pass = True
    for label, S, K, r, sigma, T in grid:
        result = mc_european_call_exact(
            S, K, r, sigma, T, n_paths, seed=0,
        )
        var_closed = float(call_payoff_variance(S, K, r, sigma, T))
        rel_error = abs(result.sample_variance - var_closed) / var_closed
        case_pass = rel_error < rel_tol
        all_pass &= case_pass
        print(f"  [{label:8}] var_closed={var_closed:11.6f}  "
              f"var_sample={result.sample_variance:11.6f}  "
              f"rel_err={rel_error:.5f}  "
              f"[{_format_pass_fail(case_pass)}]")

    print(f"\n  Test 3 overall: {_format_pass_fail(all_pass)}")
    return all_pass


# =====================================================================
# Test 4: monotonicities and asymptotic limits
# =====================================================================

def test_monotonicities_and_limits() -> bool:
    """Using common random numbers (the same seed in two perturbed
    calls), verify the qualitative properties of the BS price.

    Monotonicities in S and K hold pathwise under CRN: with the same
    Z, increasing S or decreasing K can only increase the payoff on
    each path. Monotonicity in sigma and T is *not* pathwise but holds
    in expectation; with CRN it is verified at high confidence because
    the noise on the difference of two CRN estimates is much smaller
    than on independent estimates.
    """
    print("\n" + "=" * 68)
    print("TEST 4: Monotonicities (CRN) and asymptotic limits")
    print("=" * 68)

    n_paths = 10_000
    seed = 42

    # ---- Monotonicity in S (pathwise under CRN, increasing) ----
    base = dict(K=100.0, r=0.05, sigma=0.20, T=1.0,
                n_paths=n_paths, seed=seed)
    mc_low_S  = mc_european_call_exact(S=90.0,  **base).estimate
    mc_high_S = mc_european_call_exact(S=110.0, **base).estimate
    pass_mono_S = mc_high_S >= mc_low_S
    print(f"  Monotone in S    : {mc_low_S:8.4f} (S=90) <= "
          f"{mc_high_S:8.4f} (S=110)   [{_format_pass_fail(pass_mono_S)}]")

    # ---- Anti-monotonicity in K (pathwise under CRN, decreasing) ----
    base = dict(S=100.0, r=0.05, sigma=0.20, T=1.0,
                n_paths=n_paths, seed=seed)
    mc_low_K  = mc_european_call_exact(K=90.0,  **base).estimate
    mc_high_K = mc_european_call_exact(K=110.0, **base).estimate
    pass_anti_K = mc_high_K <= mc_low_K
    print(f"  Antimonotone in K: {mc_low_K:8.4f} (K=90) >= "
          f"{mc_high_K:8.4f} (K=110)  [{_format_pass_fail(pass_anti_K)}]")

    # ---- Monotonicity in sigma (in-expectation, increasing) ----
    base = dict(S=100.0, K=100.0, r=0.05, T=1.0,
                n_paths=n_paths, seed=seed)
    mc_low_sig  = mc_european_call_exact(sigma=0.20, **base).estimate
    mc_high_sig = mc_european_call_exact(sigma=0.40, **base).estimate
    pass_mono_sig = mc_high_sig >= mc_low_sig
    print(f"  Monotone in sigma: {mc_low_sig:8.4f} (s=0.2) <= "
          f"{mc_high_sig:8.4f} (s=0.4)  [{_format_pass_fail(pass_mono_sig)}]")

    # ---- Monotonicity in T (in-expectation, increasing for r>=0) ----
    base = dict(S=100.0, K=100.0, r=0.05, sigma=0.20,
                n_paths=n_paths, seed=seed)
    mc_low_T  = mc_european_call_exact(T=1.0, **base).estimate
    mc_high_T = mc_european_call_exact(T=2.0, **base).estimate
    pass_mono_T = mc_high_T >= mc_low_T
    print(f"  Monotone in T    : {mc_low_T:8.4f} (T=1) <= "
          f"{mc_high_T:8.4f} (T=2)    [{_format_pass_fail(pass_mono_T)}]")

    # ---- Asymptotic: S -> infinity implies C / S -> 1 ----
    # At finite S the ratio differs from 1 by two contributions:
    #   * Structural (deterministic): K*exp(-rT)/S. For S=1e5 this is
    #     about 1e-3.
    #   * MC noise (statistical): half-width relative to S is
    #     ~ z*sigma*sqrt(T/n). For sigma=0.2, T=1, n=1e4, this is
    #     about 4e-3.
    # A tolerance of 1e-2 covers both with margin.
    huge_S = 100_000.0
    res_huge = mc_european_call_exact(
        S=huge_S, K=100.0, r=0.05, sigma=0.20, T=1.0,
        n_paths=n_paths, seed=seed,
    )
    ratio = res_huge.estimate / huge_S
    pass_lim_inf = abs(ratio - 1.0) < 1e-2
    print(f"  S -> infty       : C/S = {ratio:.6f}    "
          f"(expected -> 1)        [{_format_pass_fail(pass_lim_inf)}]")

    # ---- Asymptotic: S -> 0 implies C -> 0 ----
    tiny_S = 1e-6
    res_tiny = mc_european_call_exact(
        S=tiny_S, K=100.0, r=0.05, sigma=0.20, T=1.0,
        n_paths=n_paths, seed=seed,
    )
    pass_lim_zero = res_tiny.estimate < 1e-10
    print(f"  S -> 0           : C = {res_tiny.estimate:.3e}     "
          f"(expected -> 0)        [{_format_pass_fail(pass_lim_zero)}]")

    all_pass = (pass_mono_S and pass_anti_K and pass_mono_sig and pass_mono_T
                and pass_lim_inf and pass_lim_zero)
    print(f"\n  Test 4 overall: {_format_pass_fail(all_pass)}")
    return all_pass


# =====================================================================
# Main
# =====================================================================

def main():
    np.set_printoptions(linewidth=120, precision=6, suppress=True)

    results = {
        "Test 1 (containment frequency)" : test_containment_frequency(),
        "Test 2 (convergence rate)"      : test_convergence_rate(),
        "Test 3 (variance agreement)"    : test_variance_agreement(),
        "Test 4 (monotonicities/limits)" : test_monotonicities_and_limits(),
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
