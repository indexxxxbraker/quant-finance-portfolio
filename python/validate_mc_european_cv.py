"""Validation suite for the control-variates pricers of the
European call under exact GBM.

Implements four tests for each of the two controls:
  Control 1: discounted underlying X_1 = e^{-rT} S_T,
             E[X_1] = S_0.
  Control 2: discounted asset-or-nothing X_2 = e^{-rT} S_T * 1_{S_T > K},
             E[X_2] = S_0 * Phi(d_1).

  Test 1: empirical correlation Corr(Y, X) and VRF = 1/(1 - rho^2),
          measured on a single large sample. Checks the predicted
          ranges from the Block 2.2 writeup.

  Test 2: BS coherence within a few half-widths.

  Test 3: half-width comparison vs IID at fixed budget.

  Test 4: input validation.

Run from python/:
    python validate_mc_european_cv.py
"""

import numpy as np
from scipy.stats import norm

from quantlib.black_scholes import call_price
from quantlib.monte_carlo import mc_european_call_exact
from quantlib.variance_reduction import (
    mc_european_call_exact_cv_underlying,
    mc_european_call_exact_cv_aon,
)


def _format_pass_fail(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


# Common parameters used across tests.
S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.00
BS_PRICE = call_price(S0, K, r, sigma, T)


# =====================================================================
# Test 1: empirical correlations and VRF for both controls
# =====================================================================

def test_empirical_rho_and_vrf() -> bool:
    """Measure Corr(Y, X) for both controls and compute predicted VRF.

    Predictions (from Block 2.2 writeup, Section 4):
      Control 1 (underlying):  rho approx 0.92, VRF approx 6.9.
      Control 2 (AON):         rho approx 0.77, VRF approx 2.5.

    Note: an earlier "naive shape-matching" prediction had control 2
    outperforming control 1; this prediction is wrong and the writeup
    discusses why. The values above are from the closed-form analysis.
    """
    print("\n" + "=" * 78)
    print("TEST 1: Empirical correlation and VRF for each control")
    print("=" * 78)

    n_paths = 1_000_000
    rng = np.random.default_rng(0)
    Z = rng.standard_normal(n_paths)

    drift     = (r - 0.5 * sigma * sigma) * T
    diffusion = sigma * np.sqrt(T)
    discount  = np.exp(-r * T)

    S_T = S0 * np.exp(drift + diffusion * Z)

    # Target: discounted call payoff.
    Y = discount * np.maximum(S_T - K, 0.0)

    # Control 1: discounted underlying.
    X1 = discount * S_T
    rho_1 = float(np.corrcoef(Y, X1)[0, 1])
    vrf_1 = 1.0 / (1.0 - rho_1 * rho_1)

    # Control 2: discounted asset-or-nothing.
    X2 = discount * S_T * (S_T > K).astype(np.float64)
    rho_2 = float(np.corrcoef(Y, X2)[0, 1])
    vrf_2 = 1.0 / (1.0 - rho_2 * rho_2)

    print(f"  Control 1 (underlying):")
    print(f"    Corr(Y, X_1)    = {rho_1:+.4f}  "
          f"(predicted: ~ 0.92)")
    print(f"    VRF_1           = {vrf_1:.4f}  "
          f"(predicted: ~ 6.9)")
    print(f"  Control 2 (AON):")
    print(f"    Corr(Y, X_2)    = {rho_2:+.4f}  "
          f"(predicted: ~ 0.77)")
    print(f"    VRF_2           = {vrf_2:.4f}  "
          f"(predicted: ~ 2.5)")

    # Tolerance bands from the closed-form analysis. The correlations
    # are deterministic in the population; sampling fluctuations at
    # n_paths = 10^6 give us roughly 3 significant figures.
    pass_1 = (0.90 < rho_1 < 0.94) and (5.5 < vrf_1 < 9.0)
    pass_2 = (0.75 < rho_2 < 0.80) and (2.0 < vrf_2 < 3.0)

    overall = pass_1 and pass_2
    print(f"\n  Test 1 (control 1): {_format_pass_fail(pass_1)}")
    print(f"  Test 1 (control 2): {_format_pass_fail(pass_2)}")
    print(f"\n  Test 1 overall: {_format_pass_fail(overall)}")
    return overall


# =====================================================================
# Test 2: BS coherence
# =====================================================================

def test_bs_consistency() -> bool:
    """Each CV pricer should agree with BS within a few half-widths."""
    print("\n" + "=" * 78)
    print("TEST 2: Consistency with closed-form BS (each control)")
    print("=" * 78)

    n_paths = 50_000

    result_cv1 = mc_european_call_exact_cv_underlying(
        S0, K, r, sigma, T, n_paths=n_paths, seed=42,
    )
    err_1 = abs(result_cv1.estimate - BS_PRICE)
    within_1 = err_1 <= 3.0 * result_cv1.half_width

    result_cv2 = mc_european_call_exact_cv_aon(
        S0, K, r, sigma, T, n_paths=n_paths, seed=42,
    )
    err_2 = abs(result_cv2.estimate - BS_PRICE)
    within_2 = err_2 <= 3.0 * result_cv2.half_width

    print(f"  Control 1 (underlying):")
    print(f"    estimate = {result_cv1.estimate:.6f}  "
          f"BS = {BS_PRICE:.6f}  "
          f"hw = {result_cv1.half_width:.6f}  "
          f"|err| = {err_1:.6f}  "
          f"[{_format_pass_fail(within_1)}]")
    print(f"  Control 2 (AON):")
    print(f"    estimate = {result_cv2.estimate:.6f}  "
          f"BS = {BS_PRICE:.6f}  "
          f"hw = {result_cv2.half_width:.6f}  "
          f"|err| = {err_2:.6f}  "
          f"[{_format_pass_fail(within_2)}]")

    overall = within_1 and within_2
    print(f"\n  Test 2 overall: {_format_pass_fail(overall)}")
    return overall


# =====================================================================
# Test 3: half-width comparison vs IID at fixed budget
# =====================================================================

def test_vxt_vs_iid() -> bool:
    """Compare half-widths of CV-1, CV-2, and IID at equal n_paths.

    Each pricer uses the same number of payoff evaluations (n_paths),
    so the squared half-width ratio gives the VRF directly.
    """
    print("\n" + "=" * 78)
    print("TEST 3: Half-width comparison vs IID at fixed budget")
    print("        All estimators use the same n_paths = 100000 paths.")
    print("=" * 78)

    n_paths = 100_000

    result_iid = mc_european_call_exact(
        S0, K, r, sigma, T, n_paths=n_paths, seed=11,
    )
    result_cv1 = mc_european_call_exact_cv_underlying(
        S0, K, r, sigma, T, n_paths=n_paths, seed=11,
    )
    result_cv2 = mc_european_call_exact_cv_aon(
        S0, K, r, sigma, T, n_paths=n_paths, seed=11,
    )

    hw_iid = result_iid.half_width
    hw_cv1 = result_cv1.half_width
    hw_cv2 = result_cv2.half_width

    vrf_1_emp = (hw_iid / hw_cv1) ** 2
    vrf_2_emp = (hw_iid / hw_cv2) ** 2

    print(f"  IID         : hw = {hw_iid:.6f}")
    print(f"  CV-1 (under): hw = {hw_cv1:.6f}  "
          f"(IID/CV-1)^2 = {vrf_1_emp:.4f}")
    print(f"  CV-2 (AON)  : hw = {hw_cv2:.6f}  "
          f"(IID/CV-2)^2 = {vrf_2_emp:.4f}")

    pass_1 = vrf_1_emp > 5.0   # control 1 VRF predicted ~ 6.9
    pass_2 = vrf_2_emp > 2.0   # control 2 VRF predicted ~ 2.5

    overall = pass_1 and pass_2
    print(f"\n  Test 3 (control 1): {_format_pass_fail(pass_1)}")
    print(f"  Test 3 (control 2): {_format_pass_fail(pass_2)}")
    print(f"\n  Test 3 overall: {_format_pass_fail(overall)}")
    return overall


# =====================================================================
# Test 4: input validation
# =====================================================================

def test_input_validation() -> bool:
    """Mechanical: both pricers reject bad inputs."""
    print("\n" + "=" * 78)
    print("TEST 4: Input validation (both pricers)")
    print("=" * 78)

    rng = np.random.default_rng(0)

    pricers = [
        ("CV-underlying", mc_european_call_exact_cv_underlying),
        ("CV-AON",        mc_european_call_exact_cv_aon),
    ]

    cases = [
        ("S must be positive",   (-1.0, K, r, sigma, T, 1000)),
        ("K must be positive",   (S0, 0.0, r, sigma, T, 1000)),
        ("sigma must be positive", (S0, K, r, -0.10, T, 1000)),
        ("T must be positive",   (S0, K, r, sigma, 0.0, 1000)),
        ("n_paths must be at least 2", (S0, K, r, sigma, T, 1)),
    ]

    all_pass = True
    for pricer_name, pricer in pricers:
        print(f"\n  {pricer_name}:")
        for label, args in cases:
            try:
                pricer(*args, rng=rng)
                print(f"    [{label}]: FAIL (no exception raised)")
                all_pass = False
            except (ValueError, TypeError):
                print(f"    [{label}]: PASS")
        try:
            pricer(S0, K, -0.02, sigma, T, 1000, rng=rng)
            print(f"    [negative r is admissible]: PASS")
        except Exception as e:
            print(f"    [negative r is admissible]: FAIL ({type(e).__name__})")
            all_pass = False

    print(f"\n  Test 4 overall: {_format_pass_fail(all_pass)}")
    return all_pass


# =====================================================================
# Main
# =====================================================================

def main():
    np.set_printoptions(linewidth=120, precision=6, suppress=True)

    results = {
        "Test 1 (empirical rho and VRF)" : test_empirical_rho_and_vrf(),
        "Test 2 (BS consistency)"        : test_bs_consistency(),
        "Test 3 (half-width vs IID)"     : test_vxt_vs_iid(),
        "Test 4 (input validation)"      : test_input_validation(),
    }

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for name, passed in results.items():
        print(f"  {name:35}: {_format_pass_fail(passed)}")

    overall = all(results.values())
    print("\n  " + ("ALL TESTS PASSED" if overall else "SOME TESTS FAILED"))

    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
