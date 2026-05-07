"""Quasi-Monte Carlo and Randomized QMC pricers for the European call.

This module implements the Block 3 part of the variance-reduction
toolkit: deterministic QMC sequences (Halton, Sobol) for pricing the
European call under the Euler discretisation, plus a randomized
QMC variant with digital shifting that recovers a usable half-width.

The integration problem is d-dimensional with d = N (the number of
Euler steps). We use d = 20 as the canonical setup for validation;
this is dimensional enough to exhibit non-trivial QMC behaviour
without entering the high-dimensional regime where QMC's logarithmic
factor catastrophe dominates.

Functions
---------
halton(n, d):
    First n entries of the d-dimensional Halton sequence, from
    scratch via radical inverse. Returns shape (n, d) array.

sobol(n, d, scramble=False, seed=None):
    First n entries of the d-dimensional Sobol sequence, via
    scipy.stats.qmc.Sobol (which uses Joe-Kuo tables internally).
    If scramble=True, returns a digitally shifted version.

mc_european_call_euler_qmc:
    Deterministic QMC pricer using either Halton or Sobol. Returns
    a single estimate; no half-width because no statistical model.

mc_european_call_euler_rqmc:
    Randomized QMC pricer using Sobol with R independent digital
    shifts. Returns a full MCResult (n_paths field stores R, the
    number of replications, not the n*R individual payoffs).

References
----------
Phase 2 Block 3.0 writeup (foundations); Block 3.1 writeup (this
implementation). Glasserman, *Monte Carlo Methods in Financial
Engineering*, Chapter 5.
"""

import math

import numpy as np
from scipy.stats import qmc, norm

from quantlib.gbm import (
    validate_model_params,
    validate_strike,
    validate_n_paths,
    validate_n_steps,
)
from quantlib.monte_carlo import MCResult


# =====================================================================
# Halton sequence (from scratch)
# =====================================================================

# The first 20 prime numbers, used as the bases for the d-dimensional
# Halton sequence. Beyond d = 20 we would need to extend this list,
# but the construction would degrade rapidly anyway: for d > 8 Halton
# is known to exhibit visible projection artefacts.
_FIRST_PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
]


def _radical_inverse(i: int, base: int) -> float:
    """The base-b radical inverse of the integer i.

    Writes i in base b, reverses the digits, places a decimal point.
    Returns a value in [0, 1).

    Computed iteratively to avoid forming large powers and the
    associated floating-point loss; the running accumulator stays
    bounded by 1.
    """
    result = 0.0
    f = 1.0 / base
    while i > 0:
        result += f * (i % base)
        i //= base
        f /= base
    return result


def halton(n: int, d: int) -> np.ndarray:
    """Generate the first n entries of the d-dimensional Halton sequence.

    Each coordinate j uses the prime base p_{j+1} (so coordinate 0
    uses base 2, coordinate 1 uses base 3, etc.). The first index
    skipped is 0 (which would give the all-zero point); we start at
    i = 1 to avoid this degenerate point.

    Returns
    -------
    np.ndarray of shape (n, d), entries in [0, 1).
    """
    if d > len(_FIRST_PRIMES):
        raise ValueError(
            f"Halton sequence supports at most d = {len(_FIRST_PRIMES)} "
            f"dimensions (would need primes beyond {_FIRST_PRIMES[-1]})"
        )
    if n < 1:
        raise ValueError("n must be at least 1")

    out = np.empty((n, d), dtype=np.float64)
    for j in range(d):
        base = _FIRST_PRIMES[j]
        # Start at index 1: index 0 produces the all-zero point.
        for i in range(n):
            out[i, j] = _radical_inverse(i + 1, base)
    return out


# =====================================================================
# Sobol sequence (wrapper around scipy.stats.qmc.Sobol)
# =====================================================================

def sobol(n: int, d: int, *, scramble: bool = False,
          seed=None) -> np.ndarray:
    """Generate the first n entries of the d-dimensional Sobol sequence.

    Uses scipy.stats.qmc.Sobol, which loads the Joe-Kuo direction-number
    tables internally (covering dimensions up to 21201).

    Parameters
    ----------
    n : int
        Number of points. Sobol's optimal points are at powers of 2;
        for n = 2^k the equidistribution is best. For non-power-of-2
        n, scipy returns the first n points without warning, but the
        practitioner should prefer powers of 2 when possible.
    d : int
        Dimension.
    scramble : bool, default False
        If True, applies Owen scrambling (a randomisation more
        sophisticated than digital shift). For our deterministic-QMC
        pricer we use scramble=False; for RQMC we apply our own
        digital shift externally.
    seed : optional
        Seed for the scrambling, if scramble=True.

    Returns
    -------
    np.ndarray of shape (n, d), entries in [0, 1).
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if d < 1:
        raise ValueError("d must be at least 1")

    sampler = qmc.Sobol(d=d, scramble=scramble, seed=seed)
    return sampler.random(n)


# =====================================================================
# Deterministic QMC pricer
# =====================================================================

def _euler_payoffs(u: np.ndarray, S: float, K: float, r: float,
                   sigma: float, T: float) -> np.ndarray:
    """Convert a (n, N) array of uniform points into n discounted Euler
    European-call payoffs.

    Each row of u becomes one path: u[i, k] -> Z_k = Phi^{-1}(u[i, k]),
    then S_T^{(N)} accumulates via the Euler recursion.
    """
    n, N = u.shape
    h = T / N
    discount = math.exp(-r * T)

    # Convert uniforms to standard normals via inversion.
    # ppf is the inverse normal CDF (Phi^{-1}).
    Z = norm.ppf(u)

    # Euler recursion: S_{k+1} = S_k * (1 + r*h + sigma*sqrt(h)*Z_k).
    # Vectorised over the n paths simultaneously.
    sqrt_h = math.sqrt(h)
    S_path = np.full(n, S, dtype=np.float64)
    for k in range(N):
        S_path *= 1.0 + r * h + sigma * sqrt_h * Z[:, k]

    return discount * np.maximum(S_path - K, 0.0)


def mc_european_call_euler_qmc(S, K, r, sigma, T, n_paths,
                               *,
                               n_steps: int = 20,
                               sequence: str = "sobol"):
    """Deterministic QMC pricer for the European call (Euler).

    Generates a single QMC point set in dimension d = n_steps,
    inverts to normals via Phi^{-1}, runs the Euler scheme, returns
    the average discounted payoff. No half-width: the estimator is
    deterministic.

    Parameters
    ----------
    sequence : 'halton' or 'sobol'
        Which low-discrepancy sequence to use.

    Returns
    -------
    float
        The deterministic QMC estimate of the European call price.

    Notes
    -----
    Returning a bare float (not an MCResult) is deliberate: an
    MCResult with a half-width of zero would be misleading, since
    the estimator has unknown statistical error (the Koksma-Hlawka
    bound requires V(f), which is intractable for option payoffs).
    The user is forced to acknowledge the lack of an error bar.
    For a proper error estimate, use mc_european_call_euler_rqmc.
    """
    validate_model_params(S, r, sigma, T)
    validate_strike(K)
    validate_n_paths(n_paths)
    validate_n_steps(n_steps)

    if sequence == "halton":
        u = halton(n_paths, n_steps)
    elif sequence == "sobol":
        u = sobol(n_paths, n_steps, scramble=False)
    else:
        raise ValueError(
            f"sequence must be 'halton' or 'sobol', got {sequence!r}"
        )

    # Avoid u = 0 or u = 1, which would map to +/- infinity under
    # Phi^{-1}. This can happen at the first Sobol point (which is
    # the origin) or at the radical inverse boundaries. We clip into
    # the open interval with a small epsilon.
    eps = np.finfo(np.float64).tiny
    u = np.clip(u, eps, 1.0 - eps)

    payoffs = _euler_payoffs(u, S, K, r, sigma, T)
    return float(np.mean(payoffs))


# =====================================================================
# RQMC pricer with digital shift
# =====================================================================

def _digital_shift(u: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Apply a digital shift (XOR on the binary expansions) to a QMC
    point set.

    For each entry u_{i,j} and shift component xi_j, both in [0, 1),
    convert to 53-bit integers (the mantissa precision of float64),
    XOR, and convert back. This is the standard digital shift in
    floating-point form.

    Parameters
    ----------
    u : ndarray of shape (n, d), values in [0, 1).
    shift : ndarray of shape (d,), values in [0, 1).

    Returns
    -------
    ndarray of shape (n, d) with each row digitally shifted by
    `shift`.
    """
    # Convert to 53-bit unsigned integers via bit-shifting.
    # A float in [0, 1) maps to an integer in [0, 2^53).
    SCALE = float(1 << 53)
    u_int = (u * SCALE).astype(np.uint64)
    shift_int = (shift * SCALE).astype(np.uint64)

    # XOR (broadcast: shift_int has shape (d,), u_int has shape (n, d)).
    shifted_int = u_int ^ shift_int[np.newaxis, :]

    # Convert back to float in [0, 1).
    return shifted_int.astype(np.float64) / SCALE


def mc_european_call_euler_rqmc(S, K, r, sigma, T, n_paths,
                                *,
                                n_steps: int = 20,
                                n_replications: int = 20,
                                seed=None,
                                confidence_level: float = 0.95):
    """Randomized QMC pricer for the European call (Euler) with
    Sobol + digital shift.

    Procedure:
      1. Generate a single deterministic Sobol point set of size n.
      2. For each of R replications:
         a. Draw a uniform random shift xi in [0, 1)^d.
         b. XOR the Sobol points with xi to get a shifted point set.
         c. Run the Euler pricer on the shifted points.
      3. The R replication estimates are i.i.d.; their sample mean
         and sample variance give the RQMC estimate and half-width.

    Parameters
    ----------
    n_paths : int
        Per-replication number of QMC points. The total payoff budget
        is n_paths * n_replications.
    n_replications : int, default 20
        Number of independent digital shifts.
    seed : optional
        Seed for the shifts. The Sobol point set itself is deterministic
        regardless of the seed.

    Returns
    -------
    MCResult
        result.estimate is the average of R replication estimates.
        result.half_width is z * sigma_rep / sqrt(R), where sigma_rep
        is the empirical sample sd across the R replications.
        result.n_paths stores R (the number of i.i.d. units), not
        n_paths * n_replications.
    """
    validate_model_params(S, r, sigma, T)
    validate_strike(K)
    validate_n_paths(n_paths)
    validate_n_steps(n_steps)
    if n_replications < 2:
        raise ValueError("n_replications must be at least 2 for a "
                         "meaningful sample variance")

    rng = np.random.default_rng(seed)

    # Generate the base Sobol point set ONCE, deterministically.
    u_base = sobol(n_paths, n_steps, scramble=False)
    eps = np.finfo(np.float64).tiny
    u_base = np.clip(u_base, eps, 1.0 - eps)

    # Build R replication estimates.
    replication_estimates = np.empty(n_replications, dtype=np.float64)
    for r_idx in range(n_replications):
        shift = rng.random(n_steps)
        u_shifted = _digital_shift(u_base, shift)
        # Re-clip after XOR: in rare cases the XOR can produce exactly 0.
        u_shifted = np.clip(u_shifted, eps, 1.0 - eps)

        payoffs = _euler_payoffs(u_shifted, S, K, r, sigma, T)
        replication_estimates[r_idx] = float(np.mean(payoffs))

    # Reduce: estimate, sample variance, half-width over the R
    # replications. The CLT applies to these R i.i.d. estimates.
    estimate = float(np.mean(replication_estimates))
    sample_variance = float(np.var(replication_estimates, ddof=1))
    z = float(norm.ppf(0.5 + confidence_level / 2.0))
    half_width = z * math.sqrt(sample_variance / n_replications)

    return MCResult(
        estimate=estimate,
        half_width=half_width,
        sample_variance=sample_variance,
        n_paths=n_replications,
    )


# =====================================================================
# Smoke test entry point
# =====================================================================

if __name__ == "__main__":
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.00
    N = 20

    # Deterministic QMC pricers. Using a power of 2 for Sobol's
    # optimal balance properties (no warning emitted by scipy).
    est_halton = mc_european_call_euler_qmc(
        S, K, r, sigma, T, n_paths=8192,
        n_steps=N, sequence="halton",
    )
    print(f"QMC Halton (n=8192, N={N}): estimate = {est_halton:.6f}")

    est_sobol = mc_european_call_euler_qmc(
        S, K, r, sigma, T, n_paths=8192,
        n_steps=N, sequence="sobol",
    )
    print(f"QMC Sobol  (n=8192, N={N}): estimate = {est_sobol:.6f}")

    # RQMC pricer.
    result_rqmc = mc_european_call_euler_rqmc(
        S, K, r, sigma, T, n_paths=4096,
        n_steps=N, n_replications=20, seed=42,
    )
    print(f"\nRQMC Sobol (n=4096, R=20, N={N}, total payoffs=81920):")
    print(f"  estimate     : {result_rqmc.estimate:.6f}")
    print(f"  half-width   : {result_rqmc.half_width:.6f}")
    print(f"  sample var   : {result_rqmc.sample_variance:.6f}")
    print(f"  n (= R)      : {result_rqmc.n_paths}")
