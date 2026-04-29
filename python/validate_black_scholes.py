"""
Validation suite for the Black-Scholes implementation in quantlib.black_scholes.

This is *not* a unit-test suite. Unit tests check that the code returns the
expected value at a few representative inputs and protect against regressions.
This script interrogates the implementation from multiple mathematically
independent angles to give us justified confidence that it correctly
implements the Black-Scholes model.

The strategy is triangulation: each check is a necessary condition the code
must satisfy, and the checks are designed so that a bug consistent with
passing one is not necessarily consistent with passing the others.

Run from the python/ directory:
    python validate_black_scholes.py
"""

import numpy as np

from quantlib.black_scholes import call_price, put_price


# ---------------------------------------------------------------------------
# Tolerances. Tight enough to catch genuine bugs, loose enough to absorb
# unavoidable floating-point noise.
# ---------------------------------------------------------------------------
ATOL_HULL = 1e-3       # Hull rounds reference values to four decimals.
ATOL_PARITY = 1e-12    # Put-call parity should hold to machine precision.
ATOL_LIMIT = 1e-6      # Asymptotic limits, looser due to non-zero T.
ATOL_MONO = 1e-12      # Monotonicity checks: tiny tolerance for FP noise.


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


# ---------------------------------------------------------------------------
# Check 1: Textbook reference value
# ---------------------------------------------------------------------------
def check_hull_reference():
    """
    Compare against Hull, Options, Futures, and Other Derivatives (10th ed.),
    Example 15.6. This is the weakest test in isolation --- it only probes
    one point in parameter space --- but it anchors the implementation in a
    published source and would catch any gross error immediately.
    """
    print("[1] Hull (10th ed.), Example 15.6")
    S, K, r, sigma, T = 42.0, 40.0, 0.10, 0.20, 0.5

    C = call_price(S, K, r, sigma, T)
    P = put_price(S, K, r, sigma, T)
    err_C = abs(C - 4.7594)
    err_P = abs(P - 0.8086)

    print(f"    Call: {C:.6f}  (target 4.7594, err {err_C:.1e})  {_status(err_C < ATOL_HULL)}")
    print(f"    Put:  {P:.6f}  (target 0.8086, err {err_P:.1e})  {_status(err_P < ATOL_HULL)}")


# ---------------------------------------------------------------------------
# Check 2: Put-call parity (model-free)
# ---------------------------------------------------------------------------
def check_put_call_parity():
    """
    Put-call parity:   C - P = S - K * exp(-r*T).

    This is a model-free identity --- it follows from no-arbitrage alone,
    without any assumption about the dynamics of S. Any correct pricer must
    satisfy it to machine precision. Failure here is a definitive bug,
    either in call_price, in put_price, or in both.

    We probe 10000 random parameter combinations to make sure the identity
    holds across the whole regime of interest.
    """
    print("[2] Put-call parity on random parameter grid")
    rng = np.random.default_rng(seed=42)
    n = 10000
    S     = rng.uniform(50,  150, n)
    K     = rng.uniform(50,  150, n)
    r     = rng.uniform(0.01, 0.10, n)
    sigma = rng.uniform(0.10, 0.50, n)
    T     = rng.uniform(0.1,  2.0,  n)

    C = call_price(S, K, r, sigma, T)
    P = put_price(S, K, r, sigma, T)
    residual = (C - P) - (S - K * np.exp(-r * T))
    max_res = float(np.abs(residual).max())

    print(f"    Sample size: {n}")
    print(f"    Max |residual|: {max_res:.2e}  (tol {ATOL_PARITY:.0e})  {_status(max_res < ATOL_PARITY)}")


# ---------------------------------------------------------------------------
# Check 3: Limit T -> 0+
# ---------------------------------------------------------------------------
def check_limit_T_to_zero():
    """
    As T -> 0+, the European option price converges to its intrinsic value:
        C -> max(S - K, 0),    P -> max(K - S, 0).

    The formula has 1/sqrt(T) in d1, d2 and is undefined at T=0 itself, so we
    probe at T = 1e-8. This simultaneously tests:
        (a) that the discounting limit is correct (exp(-r*T) -> 1),
        (b) that the tails of the normal CDF behave correctly when d1, d2
            blow up to +/- infinity in the ITM/OTM cases,
        (c) that the ATM case (where d1, d2 stay near 0) gives the right
            cancellation between the two terms.
    """
    print("[3] Limit T -> 0+ (probed at T = 1e-8)")
    T = 1e-8
    cases = [
        ("ITM call", call_price, 110.0, 100.0, 10.0),
        ("OTM call", call_price,  90.0, 100.0,  0.0),
        ("ATM call", call_price, 100.0, 100.0,  0.0),
        ("ITM put",  put_price,   90.0, 100.0, 10.0),
        ("OTM put",  put_price,  110.0, 100.0,  0.0),
    ]
    for name, fn, S, K, target in cases:
        price = fn(S, K, 0.05, 0.20, T)
        err = abs(price - target)
        ok = err < ATOL_LIMIT
        print(f"    {name:8s} (S={S:.0f}, K={K:.0f}): price={price:.6f}  (target {target:.4f}, err {err:.1e})  {_status(ok)}")


# ---------------------------------------------------------------------------
# Check 4: Deep ITM / deep OTM asymptotics
# ---------------------------------------------------------------------------
def check_deep_ITM_OTM():
    """
    Deep ITM call (S >> K): exercise is essentially certain, so
        C -> S - K * exp(-r*T)    (the discounted forward minus PV(strike)).

    Deep OTM call (S << K): exercise is essentially impossible, so
        C -> 0.

    These probe the tails of the normal CDF, where the formula is most
    susceptible to floating-point underflow. Symmetric statements hold for
    the put.
    """
    print("[4] Deep ITM and OTM behaviour")
    K, r, sigma, T = 100.0, 0.05, 0.20, 1.0

    # Deep ITM call: S = 1000.
    S = 1000.0
    C = call_price(S, K, r, sigma, T)
    target = S - K * np.exp(-r * T)
    err = abs(C - target)
    print(f"    Deep ITM call (S={S:.0f}): {C:.4f}  (target {target:.4f}, err {err:.1e})  {_status(err < ATOL_LIMIT)}")

    # Deep OTM call: S = 10.
    S = 10.0
    C = call_price(S, K, r, sigma, T)
    err = abs(C - 0.0)
    print(f"    Deep OTM call (S={S:.0f}):   {C:.2e}  (target 0, err {err:.1e})  {_status(err < ATOL_LIMIT)}")


# ---------------------------------------------------------------------------
# Check 5: Monotonicity in each parameter
# ---------------------------------------------------------------------------
def check_monotonicities():
    """
    The Black-Scholes call price is:
        - monotonically increasing in S (Delta = N(d1) >= 0),
        - monotonically decreasing in K,
        - monotonically increasing in sigma (Vega > 0),
        - monotonically increasing in T (for non-dividend-paying stocks).

    Each of these is a different partial derivative; a sign error in d1, d2,
    or the formula structure would surface in at least one of them. This is
    a check on the *shape* of the function, complementary to the pointwise
    checks above.
    """
    print("[5] Monotonicities of the call")
    base = dict(S=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0)
    n = 50

    cases = [
        ("S",     np.linspace( 50.0,  200.0, n), "increasing"),
        ("K",     np.linspace( 50.0,  200.0, n), "decreasing"),
        ("sigma", np.linspace(  0.05,  0.80, n), "increasing"),
        ("T",     np.linspace(  0.01,  5.0,  n), "increasing"),
    ]

    for name, grid, expected in cases:
        params = dict(base)
        params[name] = grid
        prices = call_price(**params)
        diffs = np.diff(prices)
        if expected == "increasing":
            ok = bool(np.all(diffs >= -ATOL_MONO))
        else:
            ok = bool(np.all(diffs <= ATOL_MONO))
        print(f"    Call {expected:11s} in {name:5s}:  {_status(ok)}")


# ---------------------------------------------------------------------------
# Check 6: No-arbitrage bounds
# ---------------------------------------------------------------------------
def check_arbitrage_bounds():
    """
    No-arbitrage bounds (model-free):
        max(S - K*exp(-r*T), 0) <= C <= S
        max(K*exp(-r*T) - S, 0) <= P <= K*exp(-r*T)

    The lower bounds are the discounted intrinsic forward values (an option
    is worth at least its intrinsic value at the forward). The upper bound
    on the call comes from the call payoff being dominated by S_T, whose
    present value is S_t. Failure of any bound on a Black-Scholes price
    would either indicate a bug, or imply the model itself violates
    no-arbitrage --- which we know it doesn't.
    """
    print("[6] No-arbitrage bounds")
    rng = np.random.default_rng(seed=123)
    n = 10000
    S     = rng.uniform(50,  150, n)
    K     = rng.uniform(50,  150, n)
    r     = rng.uniform(0.01, 0.10, n)
    sigma = rng.uniform(0.10, 0.50, n)
    T     = rng.uniform(0.1,  2.0,  n)

    C = call_price(S, K, r, sigma, T)
    P = put_price(S, K, r, sigma, T)
    pv_K = K * np.exp(-r * T)
    intrinsic_call = np.maximum(S - pv_K, 0.0)
    intrinsic_put  = np.maximum(pv_K - S, 0.0)

    tol = ATOL_PARITY
    call_lower = bool(np.all(C >= intrinsic_call - tol))
    call_upper = bool(np.all(C <= S + tol))
    put_lower  = bool(np.all(P >= intrinsic_put - tol))
    put_upper  = bool(np.all(P <= pv_K + tol))

    print(f"    Call >= max(S - K*exp(-rT), 0):  {_status(call_lower)}")
    print(f"    Call <= S:                       {_status(call_upper)}")
    print(f"    Put  >= max(K*exp(-rT) - S, 0):  {_status(put_lower)}")
    print(f"    Put  <= K*exp(-rT):              {_status(put_upper)}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("Black-Scholes validation suite")
    print("=" * 72)
    print()

    check_hull_reference();    print()
    check_put_call_parity();   print()
    check_limit_T_to_zero();   print()
    check_deep_ITM_OTM();      print()
    check_monotonicities();    print()
    check_arbitrage_bounds();  print()

    print("=" * 72)


if __name__ == "__main__":
    main()
