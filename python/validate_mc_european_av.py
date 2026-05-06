"""Validation suite for the antithetic-variates pricer of the
European call under exact GBM.

Implements four tests as in Phase 2 Block 2.1 writeup, Section 5:

  Test 1: empirical variance reduction factor (VRF), measuring the
          correlation Corr(f(Z), f(-Z)) and computing the predicted
          factor 1 / (1 + rho).

  Test 2: consistency with closed-form BS price within a few
          half-widths.

  Test 3: VxT comparison vs IID estimator at equal computational
          budget. Reports both half-widths and the empirical ratio,
          which should match the VRF from Test 1.

  Test 4: input validation.

Run from python/:

    python validate_mc_european_av.py
"""

import numpy as np

from quantlib.black_scholes import call_price
from quantlib.variance_reduction import mc_european_call_exact_av
from quantlib.monte_carlo import mc_european_call_exact


def _format_pass_fail(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


# Common parameters used across tests.
S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.00
BS_PRICE = call_price(S0, K, r, sigma, T)


# =====================================================================
# Test 1: empirical VRF
# =====================================================================

def test_vrf_empirical() -> bool:
    """Measure Corr(f(Z), f(-Z)) and compute the VRF.

    For monotone payoffs the correlation is negative, so VRF > 1.
    Predicted range for our ATM call: VRF in [1.4, 3.3].
    """
    print("\n" + "=" * 78)
    print("TEST 1: Empirical variance reduction factor")
    print("        Predicted: VRF in [1.4, 3.3] for ATM call")
    print("=" * 78)

    n_paths = 1_000_000
    rng = np.random.default_rng(0)
    Z = rng.standard_normal(n_paths)

    drift     = (r - 0.5 * sigma * sigma) * T
    diffusion = sigma * np.sqrt(T)
    discount  = np.exp(-r * T)

    f_Z       = discount * np.maximum(S0 * np.exp(drift + diffusion * Z) - K, 0.0)
    f_neg_Z   = discount * np.maximum(S0 * np.exp(drift - diffusion * Z) - K, 0.0)

    rho = float(np.corrcoef(f_Z, f_neg_Z)[0, 1])
    vrf = 1.0 / (1.0 + rho)

    print(f"  Empirical Corr(f(Z), f(-Z)) = {rho:+.4f}")
    print(f"  Empirical VRF               = {vrf:.4f}")

    # The test passes if the correlation is negative and VRF is
    # meaningfully above 1.
    passed = (rho < 0.0) and (vrf >= 1.3)
    print(f"\n  Test 1 overall: {_format_pass_fail(passed)}")
    return passed


# =====================================================================
# Test 2: consistency with BS
# =====================================================================

def test_bs_consistency() -> bool:
    """The AV estimator should agree with BS within a few half-widths."""
    print("\n" + "=" * 78)
    print("TEST 2: Consistency with closed-form BS")
    print("=" * 78)

    n_paths = 50_000
    result = mc_european_call_exact_av(
        S0, K, r, sigma, T, n_paths=n_paths, seed=42,
    )

    err = abs(result.estimate - BS_PRICE)
    within = err <= 3.0 * result.half_width

    print(f"  AV estimate = {result.estimate:.6f}")
    print(f"  BS price    = {BS_PRICE:.6f}")
    print(f"  half-width  = {result.half_width:.6f}")
    print(f"  |error|     = {err:.6f}  (threshold: 3*hw = {3*result.half_width:.6f})")
    print(f"\n  Test 2 overall: {_format_pass_fail(within)}")
    return within


# =====================================================================
# Test 3: VxT comparison vs IID
# =====================================================================

def test_vxt_vs_iid() -> bool:
    """Compare AV (n_pairs = M) vs IID (n_paths = 2M) at fixed budget.

    Same number of payoff evaluations: AV does 2M, IID does 2M.
    The AV estimator should produce a smaller half-width by a factor
    of approximately sqrt(VRF).
    """
    print("\n" + "=" * 78)
    print("TEST 3: VxT comparison vs IID at fixed budget")
    print("        Both estimators use 2M = 100000 payoff evaluations")
    print("=" * 78)

    n_pairs = 50_000   # 2 * n_pairs = 100_000 payoff evaluations
    n_iid   = 2 * n_pairs

    result_av  = mc_european_call_exact_av(
        S0, K, r, sigma, T, n_paths=n_pairs, seed=11,
    )
    result_iid = mc_european_call_exact(
        S0, K, r, sigma, T, n_paths=n_iid,   seed=11,
    )

    var_ratio_paired = result_iid.sample_variance / result_av.sample_variance
    hw_ratio  = result_iid.half_width / result_av.half_width

    # The VRF (variance reduction factor of the *estimator*) is the
    # squared ratio of half-widths. The ratio of sample variances
    # is 2 * VRF because the AV sample variance is the variance of
    # the paired payoff, computed over n_pairs samples, whereas the
    # IID sample variance is computed over 2*n_pairs individual
    # payoff samples. The two effects together produce the factor 2.
    vrf_estimator = hw_ratio ** 2

    print(f"  AV  (n_pairs={n_pairs}): "
          f"hw={result_av.half_width:.6f}  "
          f"sample_var={result_av.sample_variance:.4f}")
    print(f"  IID (n_paths={n_iid}): "
          f"hw={result_iid.half_width:.6f}  "
          f"sample_var={result_iid.sample_variance:.4f}")
    print(f"  Half-width ratio (IID / AV)              = {hw_ratio:.4f}")
    print(f"  VRF (estimator variance ratio)            = {vrf_estimator:.4f}  "
          f"(should match VRF from Test 1)")
    print(f"  Sample-var ratio (IID / AV-paired)        = {var_ratio_paired:.4f}  "
          f"(= 2 * VRF, see comment)")

    # The AV half-width should be smaller, and VRF should be > 1.
    passed = (vrf_estimator > 1.2) and (hw_ratio > 1.0)
    print(f"\n  Test 3 overall: {_format_pass_fail(passed)}")
    return passed


# =====================================================================
# Test 4: input validation
# =====================================================================

def test_input_validation() -> bool:
    """Mechanical: the AV pricer rejects bad inputs."""
    print("\n" + "=" * 78)
    print("TEST 4: Input validation")
    print("=" * 78)

    rng = np.random.default_rng(0)

    cases = [
        ("S must be positive",
         lambda: mc_european_call_exact_av(
             -1.0, K, r, sigma, T, 1000, rng=rng)),
        ("K must be positive",
         lambda: mc_european_call_exact_av(
             S0, 0.0, r, sigma, T, 1000, rng=rng)),
        ("sigma must be positive",
         lambda: mc_european_call_exact_av(
             S0, K, r, -0.10, T, 1000, rng=rng)),
        ("T must be positive",
         lambda: mc_european_call_exact_av(
             S0, K, r, sigma, 0.0, 1000, rng=rng)),
        ("n_paths must be at least 2",
         lambda: mc_european_call_exact_av(
             S0, K, r, sigma, T, 1, rng=rng)),
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
        mc_european_call_exact_av(S0, K, -0.02, sigma, T, 1000, rng=rng)
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
        "Test 1 (empirical VRF)"    : test_vrf_empirical(),
        "Test 2 (BS consistency)"   : test_bs_consistency(),
        "Test 3 (VxT vs IID)"       : test_vxt_vs_iid(),
        "Test 4 (input validation)" : test_input_validation(),
    }

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for name, passed in results.items():
        print(f"  {name:34}: {_format_pass_fail(passed)}")

    overall = all(results.values())
    print("\n  " + ("ALL TESTS PASSED" if overall else "SOME TESTS FAILED"))

    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
