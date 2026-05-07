"""Validation suite for the QMC and RQMC European call pricers
(Phase 2 Block 3.1).

Implements four tests:

  Test 1: Convergence rate slopes.
    Measures error vs n on a logarithmic grid for IID-MC, QMC-Halton,
    QMC-Sobol (deterministic), and RQMC-Sobol. Fits a linear regression
    of log error on log n and reports the slope. Predicted:
        IID-MC: slope ~ -0.5
        Halton: slope -0.7 to -0.9
        Sobol-det: slope ~ -1.0
        Sobol-RQMC: slope ~ -1.0

  Test 2: BS coherence of RQMC-Sobol.
    The RQMC estimate should agree with C^BS modulo the Euler bias.

  Test 3: Half-width comparison vs IID at equal payoff budget.
    RQMC-Sobol with n_paths * R payoffs should produce half-widths
    significantly below IID-MC at the same total cost.

  Test 4: Input validation.
    Mechanical: bad inputs should raise.

Run from python/:
    python validate_mc_european_qmc.py
"""

import math

import numpy as np
from scipy.stats import linregress

from quantlib.black_scholes import call_price
from quantlib.monte_carlo import mc_european_call_euler
from quantlib.qmc import (
    mc_european_call_euler_qmc,
    mc_european_call_euler_rqmc,
)


def _format_pass_fail(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


# Common parameters.
S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.00
N_STEPS = 20
BS_PRICE = call_price(S0, K, r, sigma, T)


def _high_precision_euler_target() -> float:
    """Compute C^Euler at N = 20 via a high-precision RQMC run.

    We use RQMC rather than IID-MC because RQMC's half-width at the
    same total budget is much smaller; this matters because the
    convergence-rate Test 1 measures errors as small as 1e-3, and
    a noisy target would contaminate the slope measurement.

    With n = 65536 per replication and R = 50 replications (3.3M
    total payoffs), the half-width is approximately 1.5e-3, i.e.
    the target is precise to ~3 decimal digits. This is one order
    of magnitude better than the smallest errors we are trying to
    measure in Test 1, so the slope measurement is not floor-limited
    by the target's own noise.
    """
    print("    Computing high-precision C^Euler reference (RQMC, n*R = 3.3M)...")
    result = mc_european_call_euler_rqmc(
        S0, K, r, sigma, T,
        n_paths=65536, n_steps=N_STEPS,
        n_replications=50, seed=99999,
    )
    print(f"    C^Euler ~ {result.estimate:.6f} +/- {result.half_width:.6f}  "
          f"(BS = {BS_PRICE:.6f}, Euler bias = "
          f"{result.estimate - BS_PRICE:+.4f})")
    return float(result.estimate)


# =====================================================================
# Test 1: convergence rate slopes
# =====================================================================

def test_convergence_rates(c_euler: float) -> bool:
    print("\n" + "=" * 78)
    print("TEST 1: Convergence rate slopes (log-log regression)")
    print("=" * 78)
    print("\n    Methodology:")
    print("      IID-MC: standard deviation across 30 seeds (rigorous quantity")
    print("              with clean theoretical prediction sigma_n = O(n^-0.5)).")
    print("      QMC, RQMC: absolute error vs high-precision RQMC target.")

    # Use exact powers of 2 to give Sobol its best behaviour.
    n_grid = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    log_n = np.log(n_grid)

    err_iid    = np.zeros(len(n_grid))   # IID: stdev across seeds
    err_halton = np.zeros(len(n_grid))   # QMC: abs error
    err_sobol  = np.zeros(len(n_grid))
    err_rqmc   = np.zeros(len(n_grid))

    print(f"\n    {'n':>7} {'IID stdev':>10} {'Halton err':>11} "
          f"{'Sobol det err':>14} {'RQMC err':>10}")
    print(f"    {'-'*7} {'-'*10} {'-'*11} {'-'*14} {'-'*10}")

    rng = np.random.default_rng(0)

    for idx, n in enumerate(n_grid):
        # IID-MC: collect 30 estimates with different seeds, compute
        # standard deviation. This is what 1/sqrt(n) predicts.
        N_SEEDS = 30
        iid_estimates = np.empty(N_SEEDS, dtype=np.float64)
        for s in range(N_SEEDS):
            res = mc_european_call_euler(
                S0, K, r, sigma, T,
                n_paths=n, n_steps=N_STEPS, seed=int(rng.integers(1, 10**9)),
            )
            iid_estimates[s] = res.estimate
        err_iid[idx] = float(np.std(iid_estimates, ddof=1))

        # QMC Halton: deterministic.
        est_h = mc_european_call_euler_qmc(
            S0, K, r, sigma, T, n_paths=n,
            n_steps=N_STEPS, sequence="halton",
        )
        err_halton[idx] = abs(est_h - c_euler)

        # QMC Sobol: deterministic.
        est_s = mc_european_call_euler_qmc(
            S0, K, r, sigma, T, n_paths=n,
            n_steps=N_STEPS, sequence="sobol",
        )
        err_sobol[idx] = abs(est_s - c_euler)

        # RQMC Sobol.
        n_per_rep = max(2, n // 20)
        res_rqmc = mc_european_call_euler_rqmc(
            S0, K, r, sigma, T,
            n_paths=n_per_rep, n_steps=N_STEPS,
            n_replications=20, seed=int(rng.integers(1, 10**9)),
        )
        err_rqmc[idx] = abs(res_rqmc.estimate - c_euler)

        print(f"    {n:>7} {err_iid[idx]:>10.6f} {err_halton[idx]:>11.6f} "
              f"{err_sobol[idx]:>14.6f} {err_rqmc[idx]:>10.6f}")

    # Linear regression slopes.
    def safe_slope(errs):
        valid = errs > 1e-12
        if valid.sum() < 3:
            return float("nan")
        return float(linregress(log_n[valid], np.log(errs[valid])).slope)

    slope_iid    = safe_slope(err_iid)
    slope_halton = safe_slope(err_halton)
    slope_sobol  = safe_slope(err_sobol)
    slope_rqmc   = safe_slope(err_rqmc)

    print(f"\n    Convergence slopes (linear regression of log err vs log n):")
    print(f"      IID-MC stdev: {slope_iid:+.4f}  (predicted ~ -0.50)")
    print(f"      QMC-Halton  : {slope_halton:+.4f}  (predicted -0.7 to -0.9)")
    print(f"      QMC-Sobol   : {slope_sobol:+.4f}  (predicted ~ -1.00)")
    print(f"      RQMC-Sobol  : {slope_rqmc:+.4f}  (predicted ~ -1.00)")

    # Tolerance bands. IID stdev has a clean prediction, so the band
    # is tight. QMC/RQMC absolute error has noise from the choice of
    # n grid points, so the bands are wider.
    pass_iid    = -0.65 < slope_iid    < -0.35
    pass_halton = -1.30 < slope_halton < -0.50
    pass_sobol  = -1.50 < slope_sobol  < -0.70
    pass_rqmc   = -1.50 < slope_rqmc   < -0.50

    overall = pass_iid and pass_halton and pass_sobol and pass_rqmc
    print(f"\n    Test 1 (IID slope)        : {_format_pass_fail(pass_iid)}")
    print(f"    Test 1 (Halton slope)     : {_format_pass_fail(pass_halton)}")
    print(f"    Test 1 (Sobol slope)      : {_format_pass_fail(pass_sobol)}")
    print(f"    Test 1 (RQMC slope)       : {_format_pass_fail(pass_rqmc)}")
    print(f"    Test 1 overall            : {_format_pass_fail(overall)}")

    return overall


# =====================================================================
# Test 2: BS coherence of RQMC
# =====================================================================

def test_bs_coherence_rqmc(c_euler: float) -> bool:
    print("\n" + "=" * 78)
    print("TEST 2: BS coherence of RQMC-Sobol")
    print("=" * 78)

    n_per_rep = 4096
    R = 20

    result = mc_european_call_euler_rqmc(
        S0, K, r, sigma, T,
        n_paths=n_per_rep, n_steps=N_STEPS,
        n_replications=R, seed=42,
    )

    # Expected target: C^Euler (which differs from C^BS by the Euler bias).
    # Tolerance: 3 half-widths around C^Euler.
    err_vs_euler = abs(result.estimate - c_euler)
    err_vs_bs    = abs(result.estimate - BS_PRICE)
    within_euler = err_vs_euler <= 3.0 * result.half_width

    print(f"    n per rep = {n_per_rep}, R = {R}")
    print(f"    estimate          = {result.estimate:.6f}")
    print(f"    half-width        = {result.half_width:.6f}")
    print(f"    C^Euler reference = {c_euler:.6f}  (|err| = {err_vs_euler:.6f})")
    print(f"    C^BS reference    = {BS_PRICE:.6f}  (|err| = {err_vs_bs:.6f})")
    print(f"    within 3*hw of Euler: {_format_pass_fail(within_euler)}")
    return within_euler


# =====================================================================
# Test 3: half-width vs IID at equal budget
# =====================================================================

def test_halfwidth_vs_iid() -> bool:
    print("\n" + "=" * 78)
    print("TEST 3: Half-width comparison vs IID at equal payoff budget")
    print("=" * 78)

    # Common total budget: 100,000 payoff evaluations.
    # IID: n_paths = 100_000.
    # RQMC: n_per_rep = 5_000, R = 20.
    n_iid_total = 100_000
    n_per_rep = 5_000
    R = 20

    iid = mc_european_call_euler(
        S0, K, r, sigma, T,
        n_paths=n_iid_total, n_steps=N_STEPS, seed=11,
    )
    rqmc = mc_european_call_euler_rqmc(
        S0, K, r, sigma, T,
        n_paths=n_per_rep, n_steps=N_STEPS,
        n_replications=R, seed=11,
    )

    ratio = iid.half_width / rqmc.half_width

    print(f"    IID-MC  : n = {n_iid_total} paths, hw = {iid.half_width:.6f}")
    print(f"    RQMC    : n = {n_per_rep} per-rep x R = {R} = "
          f"{n_per_rep * R} payoffs, hw = {rqmc.half_width:.6f}")
    print(f"    half-width ratio (IID / RQMC) = {ratio:.2f}x")
    print(f"    estimator variance ratio      = {ratio**2:.2f}x")

    # We require RQMC to win, with at least a factor of 3.
    passed = ratio > 3.0
    print(f"\n    Test 3 overall: {_format_pass_fail(passed)}")
    return passed


# =====================================================================
# Test 4: input validation
# =====================================================================

def test_input_validation() -> bool:
    print("\n" + "=" * 78)
    print("TEST 4: Input validation")
    print("=" * 78)

    # We test each pricer with the standard set of bad inputs.
    pricers = [
        ("QMC-deterministic", mc_european_call_euler_qmc),
        ("RQMC",              mc_european_call_euler_rqmc),
    ]

    cases = [
        ("S must be positive",  (-1.0, K, r, sigma, T, 1000)),
        ("K must be positive",  (S0, 0.0, r, sigma, T, 1000)),
        ("sigma must be positive", (S0, K, r, -0.10, T, 1000)),
        ("T must be positive",  (S0, K, r, sigma, 0.0, 1000)),
        ("n_paths must be at least 2", (S0, K, r, sigma, T, 1)),
    ]

    all_pass = True
    for name, pricer in pricers:
        print(f"\n    {name}:")
        for label, args in cases:
            try:
                pricer(*args)
                print(f"      [{label}]: FAIL (no exception)")
                all_pass = False
            except (ValueError, TypeError):
                print(f"      [{label}]: PASS")

        # Negative r is admissible.
        try:
            pricer(S0, K, -0.02, sigma, T, 1000)
            print(f"      [negative r is admissible]: PASS")
        except Exception as e:
            print(f"      [negative r is admissible]: FAIL ({type(e).__name__})")
            all_pass = False

    print(f"\n    Test 4 overall: {_format_pass_fail(all_pass)}")
    return all_pass


# =====================================================================
# Main
# =====================================================================

def main():
    print("\n" + "=" * 78)
    print("VALIDATION: Phase 2 Block 3.1 -- QMC and RQMC pricers")
    print("=" * 78)
    print(f"\nParameters: S = {S0}, K = {K}, r = {r}, sigma = {sigma}, T = {T}")
    print(f"Discretisation: Euler with N = {N_STEPS} steps")
    print(f"BS reference price = {BS_PRICE:.6f}\n")

    print("    Computing C^Euler reference value...")
    c_euler = _high_precision_euler_target()

    results = {
        "Test 1 (convergence rate slopes)" : test_convergence_rates(c_euler),
        "Test 2 (BS coherence of RQMC)"    : test_bs_coherence_rqmc(c_euler),
        "Test 3 (half-width vs IID)"       : test_halfwidth_vs_iid(),
        "Test 4 (input validation)"        : test_input_validation(),
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
