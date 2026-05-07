"""American options by binomial tree and by Longstaff-Schwartz.

This module implements two pricers for the American put:

  - binomial_american_put: Cox-Ross-Rubinstein binomial tree. The
    backward induction is exact on the lattice; the only error is the
    discretisation gap between the Bermudan (with N exercise dates)
    and the true American (with continuum exercise). Used as the
    ground truth in the limit N -> infty.

  - lsm_american_put: Longstaff-Schwartz regression-based Monte Carlo,
    with Laguerre polynomials of degree 0 through M-1 as the basis
    (default M=4) on the normalised state S/K. Restricted to
    in-the-money paths at each exercise date.

The American put is the canonical test bed for both methods. It has
no closed form (the early-exercise boundary depends on time and
underlying in a non-trivial way), so cross-validation between two
independent methods is the standard way to gain confidence in the
implementations.

References
----------
Phase 2 Block 6 writeup. Longstaff and Schwartz (2001), Review of
Financial Studies 14(1):113-147. Cox, Ross and Rubinstein (1979),
JFE 7(3):229-263. Glasserman, Chapter 8.
"""

import math

import numpy as np

from quantlib.gbm import (
    validate_model_params,
    validate_n_paths,
    validate_strike,
)
from quantlib.monte_carlo import MCResult, _resolve_rng, mc_estimator


# =====================================================================
# Validators
# =====================================================================

def _validate_n_steps(n_steps):
    if not isinstance(n_steps, (int, np.integer)):
        raise TypeError(f"n_steps must be int, got {type(n_steps).__name__}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")


def _validate_basis_size(basis_size):
    if not isinstance(basis_size, (int, np.integer)):
        raise TypeError(f"basis_size must be int, got {type(basis_size).__name__}")
    if basis_size < 1:
        raise ValueError(f"basis_size must be >= 1, got {basis_size}")
    if basis_size > 8:
        raise ValueError(f"basis_size must be <= 8 (numerical conditioning), got {basis_size}")


# =====================================================================
# Cox-Ross-Rubinstein binomial tree (ground truth)
# =====================================================================

def binomial_american_put(S, K, r, sigma, T, n_steps):
    """Price an American put by the Cox-Ross-Rubinstein binomial tree.

    Lattice parameters:
        u = exp(sigma * sqrt(dt))
        d = 1/u
        p = (exp(r*dt) - d) / (u - d)

    Backward induction: at each non-terminal node, the value is the
    maximum of intrinsic (K - S)^+ and the discounted risk-neutral
    expectation of the next-step values. The American constraint is
    encoded by this max.

    Returns the price as a float (no Monte Carlo error -- the only
    error is the lattice discretisation, which vanishes as
    n_steps -> infty at rate ~1/sqrt(n_steps)).
    """
    validate_model_params(S, r, sigma, T)
    validate_strike(K)
    _validate_n_steps(n_steps)

    dt = T / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp(r * dt) - d) / (u - d)

    if not (0.0 < p < 1.0):
        raise ValueError(
            f"Risk-neutral probability p={p} not in (0, 1); "
            "increase n_steps or check r, sigma."
        )

    # Terminal-node asset prices: S_N(j) = S * u^(2j - N) for j=0..N.
    j_arr = np.arange(n_steps + 1)
    S_terminal = S * u ** (2 * j_arr - n_steps)
    V = np.maximum(K - S_terminal, 0.0)

    # Backward induction: at step k, V has indices 0..k.
    for k in range(n_steps - 1, -1, -1):
        # Continuation values at the k+1 nodes of step k.
        V_cont = disc * (p * V[1:k + 2] + (1.0 - p) * V[:k + 1])
        # Intrinsic values at step k.
        j_arr = np.arange(k + 1)
        S_k = S * u ** (2 * j_arr - k)
        intrinsic = np.maximum(K - S_k, 0.0)
        # American constraint: take the larger of the two.
        V[:k + 1] = np.maximum(intrinsic, V_cont)

    return float(V[0])


# =====================================================================
# Laguerre basis functions
# =====================================================================

def _laguerre_basis(x, M):
    """Evaluate the first M Laguerre polynomials L_0, ..., L_{M-1} at
    each entry of x. Returns a (len(x), M) array.

    Recurrence (Bonnet's):
        L_0(x) = 1
        L_1(x) = 1 - x
        (n+1) L_{n+1}(x) = (2n + 1 - x) L_n(x) - n L_{n-1}(x)

    The recurrence is numerically stable on [0, 1] (where ITM put paths
    live, since x = S/K < 1 when in the money), and remains usable
    well beyond [0, 2].
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    L = np.empty((n, M))
    if M >= 1:
        L[:, 0] = 1.0
    if M >= 2:
        L[:, 1] = 1.0 - x
    for k in range(1, M - 1):
        L[:, k + 1] = ((2 * k + 1 - x) * L[:, k] - k * L[:, k - 1]) / (k + 1)
    return L


# =====================================================================
# Longstaff-Schwartz Monte Carlo
# =====================================================================

def lsm_american_put(S, K, r, sigma, T, n_paths,
                     *, n_steps=50, basis_size=4,
                     seed=None, rng=None,
                     confidence_level=0.95):
    """American put by the Longstaff-Schwartz algorithm.

    Algorithm (see Phase 2 Block 6 writeup, Section 3):

      1. Simulate n_paths exact GBM paths on the equispaced grid
         t_k = k*T/n_steps, k = 0, ..., n_steps.
      2. Initialise cash flows at maturity: Y_i = max(K - S_N^i, 0).
      3. Backward induction: for k = n_steps-1, ..., 1:
            a. Discount one step: Y *= exp(-r*dt).
            b. Find ITM paths (K - S_k^i > 0).
            c. Regress {Y_i}_ITM on Laguerre basis of S_k^i / K.
            d. Compute continuation estimates C_hat_i for ITM paths.
            e. Where intrinsic >= C_hat, set Y_i to intrinsic.
      4. Final discount Y *= exp(-r*dt).
      5. Estimator = max(intrinsic at S_0, sample mean of Y).

    The estimator is generally not exactly unbiased (in-sample fitting
    biases up, sub-optimal exercise rule biases down), but the two
    biases approximately cancel for large n_paths and reasonable
    basis_size.

    Returns an MCResult; the half-width is computed under the
    standard MC asymptotic CI.
    """
    validate_model_params(S, r, sigma, T)
    validate_strike(K)
    validate_n_paths(n_paths)
    _validate_n_steps(n_steps)
    _validate_basis_size(basis_size)
    rng = _resolve_rng(seed, rng)

    dt = T / n_steps
    drift = (r - 0.5 * sigma * sigma) * dt
    diffusion = sigma * math.sqrt(dt)
    disc_step = math.exp(-r * dt)

    # ---- Step 1: simulate path matrix shape (n_paths, n_steps + 1) ----
    Z = rng.standard_normal(size=(n_paths, n_steps))
    increments = np.exp(drift + diffusion * Z)
    factors = np.cumprod(increments, axis=1)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S
    paths[:, 1:] = S * factors

    # ---- Step 2: cash flows initialised at maturity ----
    Y = np.maximum(K - paths[:, -1], 0.0)

    # ---- Step 3: backward induction ----
    for k in range(n_steps - 1, 0, -1):
        Y *= disc_step
        S_k = paths[:, k]
        intrinsic = K - S_k
        itm = intrinsic > 0.0

        if itm.sum() < basis_size:
            # Not enough ITM paths to identify the regression.
            # Skip exercise at this step (just discount; already done).
            continue

        x_itm = S_k[itm] / K
        Psi = _laguerre_basis(x_itm, basis_size)
        # Solve Psi^T Psi beta = Psi^T Y_itm. lstsq uses SVD; for our
        # small basis (M=4) this is essentially as fast as Cholesky and
        # robust to near-rank-deficiency.
        beta, *_ = np.linalg.lstsq(Psi, Y[itm], rcond=None)
        C_hat = Psi @ beta

        # Exercise where intrinsic >= continuation estimate.
        ex_mask = intrinsic[itm] >= C_hat
        idx_itm = np.where(itm)[0]
        idx_ex = idx_itm[ex_mask]
        Y[idx_ex] = intrinsic[idx_ex]

    # ---- Step 4: final discount to t_0 ----
    Y *= disc_step

    # ---- Step 5: t_0 exercise comparison ----
    intrinsic_0 = max(K - S, 0.0)
    cont_result = mc_estimator(Y, confidence_level=confidence_level)
    if intrinsic_0 > cont_result.estimate:
        # Immediate exercise dominates: deterministic answer.
        return MCResult(
            estimate=intrinsic_0,
            half_width=0.0,
            sample_variance=0.0,
            n_paths=n_paths,
        )
    return cont_result


# =====================================================================
# Smoke test
# =====================================================================

if __name__ == "__main__":
    from quantlib.black_scholes import put_price

    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0

    # European put for reference (no early exercise).
    eu_put = put_price(S, K, r, sigma, T)
    print(f"European put (BS):                      {eu_put:.6f}")

    # Binomial American convergence.
    print("\nBinomial American put convergence:")
    for N in [100, 500, 1000, 5000, 10000]:
        v = binomial_american_put(S, K, r, sigma, T, N)
        print(f"  N = {N:>5d}  ->  {v:.6f}")

    # Reference (high-N binomial).
    v_ref = binomial_american_put(S, K, r, sigma, T, 10000)
    print(f"\nReference American put (binomial N=10000): {v_ref:.6f}")
    print(f"Early exercise premium:                  {v_ref - eu_put:.6f}")

    # LSM
    n = 100_000
    print(f"\nLSM American put (n={n}, N_LSM=50, basis=4):")
    res = lsm_american_put(S, K, r, sigma, T, n,
                            n_steps=50, basis_size=4, seed=42)
    print(f"  estimate    = {res.estimate:.6f}")
    print(f"  half-width  = {res.half_width:.6f}")
    print(f"  reference   = {v_ref:.6f}")
    err = res.estimate - v_ref
    err_in_hw = err / res.half_width if res.half_width > 0 else float('inf')
    print(f"  err to ref  = {err:+.6f} ({err_in_hw:+.2f} hw)")
