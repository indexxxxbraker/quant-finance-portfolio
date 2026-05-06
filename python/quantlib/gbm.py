"""GBM-specific samplers for Monte Carlo simulation.

This module contains all samplers of the geometric Brownian motion
state under the risk-neutral measure, plus the input validators that
are shared across them. It is designed to be paired with
``quantlib.monte_carlo``, which provides the model-agnostic statistical
reducer and the high-level pricers that orchestrate samplers, payoffs
and the reducer.

Public samplers:

- ``simulate_terminal_gbm``: exact sampler of the terminal price
  ``S_T`` under GBM, using the closed-form solution of the SDE
  (Block 1.1).

- ``simulate_path_euler``: Euler-Maruyama path sampler returning the
  full path including the initial value (Block 1.2.1).
- ``simulate_terminal_euler``: convenience that returns only the
  terminal column of ``simulate_path_euler`` (Block 1.2.1).

- ``simulate_path_milstein``: Milstein path sampler returning the
  full path (Block 1.2.2).
- ``simulate_terminal_milstein``: terminal-value convenience for
  Milstein (Block 1.2.2).

Public validators (used by the pricers in ``monte_carlo.py``):

- ``validate_model_params``: checks ``S0 > 0``, ``sigma > 0``, ``T > 0``;
  ``r`` is unconstrained.
- ``validate_strike``: checks ``K > 0``.
- ``validate_n_paths``: checks ``n_paths >= 2``.
- ``validate_n_steps``: checks ``n_steps >= 1``.

References
----------
Phase 2 Block 0 writeup (foundations); Block 1.1 writeup (exact);
Block 1.2.0 writeup (SDE discretisation theory); Block 1.2.1 writeup
(Euler); Block 1.2.2 writeup (Milstein). Glasserman, *Monte Carlo
Methods in Financial Engineering*, Chapters 1, 3, and 6.
"""

import numpy as np
from scipy.stats import norm


# =====================================================================
# Input validators
# =====================================================================

def validate_model_params(S0, r, sigma, T):
    """Validate the model parameters of geometric Brownian motion.

    Note that ``r`` is unconstrained: negative rates are admissible.
    """
    if S0 <= 0:
        raise ValueError(f"S0 must be positive, got {S0}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")


def validate_strike(K):
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")


def validate_n_paths(n_paths):
    if not isinstance(n_paths, (int, np.integer)):
        raise TypeError(
            f"n_paths must be an integer, got {type(n_paths).__name__}"
        )
    if n_paths < 2:
        raise ValueError(
            f"n_paths must be at least 2 (for sample variance with "
            f"Bessel's correction), got {n_paths}"
        )


def validate_n_steps(n_steps):
    if not isinstance(n_steps, (int, np.integer)):
        raise TypeError(
            f"n_steps must be an integer, got {type(n_steps).__name__}"
        )
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}")


# =====================================================================
# Internal helpers
# =====================================================================

def _standard_normals(n, rng):
    """Generate ``n`` independent standard normal samples by inversion.

    Inversion is used (rather than the faster Ziggurat method that
    NumPy provides via ``rng.standard_normal``) to maintain
    coordinate-by-coordinate compatibility with low-discrepancy
    sequences, which will be introduced in Phase 2 Block 3 (QMC).
    See the Phase 2 Block 0 writeup, Section 3.3, for the rationale.
    """
    u = rng.uniform(size=n)
    return norm.ppf(u)


def _gbm_exact_from_brownian(S0, r, sigma, T, W_T):
    """Exact GBM terminal price given the terminal Brownian value(s).

    Computes ``S_T = S_0 * exp((r - 0.5 * sigma^2) * T + sigma * W_T)``,
    the closed-form solution of the GBM SDE evaluated at the supplied
    ``W_T``. Used in validation to build an exact reference path that
    shares its Brownian driver with a discretised path, enabling
    pathwise (strong) and CRN-based (weak) error estimation.

    Notes
    -----
    Private to this module: only the validation scripts import it.
    The function is not part of the user-facing API.
    """
    drift = (r - 0.5 * sigma * sigma) * T
    return S0 * np.exp(drift + sigma * W_T)


# =====================================================================
# Exact GBM sampler (Block 1.1)
# =====================================================================

def simulate_terminal_gbm(S0, r, sigma, T, n_paths, rng):
    """Simulate ``n_paths`` samples of ``S_T`` under geometric Brownian
    motion, exactly.

    Uses the closed-form solution of the GBM SDE,
    ``S_T = S_0 * exp((r - sigma^2/2) * T + sigma * sqrt(T) * Z)``,
    with ``Z ~ N(0, 1)`` sampled by inversion. No time-discretisation
    error.

    See Phase 2 Block 1.1 writeup, Section 3.
    """
    validate_model_params(S0, r, sigma, T)
    validate_n_paths(n_paths)

    Z = _standard_normals(n_paths, rng)
    drift = (r - 0.5 * sigma * sigma) * T
    diffusion = sigma * np.sqrt(T)
    return S0 * np.exp(drift + diffusion * Z)


# =====================================================================
# Internal: sample Brownian increments and resolve rng vs delta_W
# =====================================================================

def _resolve_increments(n_paths, n_steps, h, rng, delta_W):
    """Either generate Brownian increments from rng or validate the
    pre-supplied ones. Used by both Euler and Milstein samplers.

    Exactly one of ``rng`` and ``delta_W`` must be provided.
    """
    if (rng is None) == (delta_W is None):
        raise ValueError(
            "Pass exactly one of `rng` (Generator) or `delta_W` "
            "(pre-sampled increments)."
        )

    if delta_W is None:
        return rng.normal(loc=0.0, scale=np.sqrt(h),
                          size=(n_paths, n_steps))

    delta_W = np.asarray(delta_W, dtype=np.float64)
    if delta_W.shape != (n_paths, n_steps):
        raise ValueError(
            f"delta_W shape must be ({n_paths}, {n_steps}), "
            f"got {delta_W.shape}"
        )
    return delta_W


# =====================================================================
# Euler-Maruyama scheme (Block 1.2.1)
# =====================================================================

def simulate_path_euler(S0, r, sigma, T, n_steps, n_paths,
                        *,
                        rng=None,
                        delta_W=None):
    """Simulate full Euler-Maruyama paths of GBM.

    Implements the recursion
        S_{n+1} = S_n * (1 + r * h + sigma * dW_n),
    with h = T / n_steps and dW_n ~ N(0, h) independent across n.
    Returns the full path including the initial value.

    Parameters
    ----------
    rng : numpy.random.Generator or None, keyword-only
        Random generator for sampling Brownian increments. Pass
        exactly one of ``rng`` and ``delta_W``.
    delta_W : ndarray of shape (n_paths, n_steps) or None, keyword-only
        Pre-sampled Brownian increments. Used by the validation suite
        for common-random-numbers comparisons.

    Returns
    -------
    ndarray of shape (n_paths, n_steps + 1)
        Column 0 is ``S0``, column ``n_steps`` is the
        Euler-discretised terminal value.

    Notes
    -----
    Euler does not preserve positivity in principle: a sufficiently
    negative ``delta_W`` can produce a negative ``S``. For typical
    finance parameters this is astronomically rare and not handled
    here. See Phase 2 Block 1.2.1 writeup, Section 2.2.
    """
    validate_model_params(S0, r, sigma, T)
    validate_n_steps(n_steps)
    validate_n_paths(n_paths)

    h = T / n_steps
    delta_W = _resolve_increments(n_paths, n_steps, h, rng, delta_W)

    # Multiplicative recursion: S_{n+1} = S_n * factor_n
    # where factor_n = 1 + r * h + sigma * delta_W_n.
    factors = 1.0 + r * h + sigma * delta_W
    cumulative = np.cumprod(factors, axis=1)

    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = S0
    paths[:, 1:] = S0 * cumulative
    return paths


def simulate_terminal_euler(S0, r, sigma, T, n_steps, n_paths,
                            *,
                            rng=None,
                            delta_W=None):
    """Convenience: Euler-Maruyama terminal value only.

    Returns only the terminal column of ``simulate_path_euler``. See
    that function for the full parameter description.
    """
    paths = simulate_path_euler(S0, r, sigma, T, n_steps, n_paths,
                                rng=rng, delta_W=delta_W)
    return paths[:, -1]


# =====================================================================
# Milstein scheme (Block 1.2.2)
# =====================================================================

def simulate_path_milstein(S0, r, sigma, T, n_steps, n_paths,
                           *,
                           rng=None,
                           delta_W=None):
    """Simulate full Milstein paths of GBM.

    Implements the recursion
        S_{n+1} = S_n * (1 + r*h + sigma*dW_n
                          + 0.5 * sigma^2 * (dW_n^2 - h)),
    with h = T / n_steps and dW_n ~ N(0, h) independent across n.

    The recursion is multiplicative but quadratic in dW_n. The extra
    term, called the Milstein corrector, has zero mean (because
    E[dW_n^2 - h] = 0 since Var(dW_n) = h) but is pathwise nonzero
    of typical magnitude h. Including it lifts the strong order from
    1/2 (Euler) to 1 (Milstein); the weak order remains 1.

    Parameters and return value identical to ``simulate_path_euler``.

    Notes
    -----
    Milstein has marginally better positivity behaviour than Euler:
    the squared-increment term is always nonnegative, so the
    multiplicative factor is bounded below by
    ``1 + r*h + sigma*dW_n - 0.5 * sigma^2 * h``. For typical finance
    parameters, neither scheme produces nonpositive prices.

    See Phase 2 Block 1.2.2 writeup.
    """
    validate_model_params(S0, r, sigma, T)
    validate_n_steps(n_steps)
    validate_n_paths(n_paths)

    h = T / n_steps
    delta_W = _resolve_increments(n_paths, n_steps, h, rng, delta_W)

    # Multiplicative recursion with quadratic corrector.
    # factor_n = 1 + r*h + sigma * dW_n + 0.5 * sigma^2 * (dW_n^2 - h)
    sigma_sq = sigma * sigma
    factors = (1.0 + r * h
               + sigma * delta_W
               + 0.5 * sigma_sq * (delta_W * delta_W - h))
    cumulative = np.cumprod(factors, axis=1)

    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = S0
    paths[:, 1:] = S0 * cumulative
    return paths


def simulate_terminal_milstein(S0, r, sigma, T, n_steps, n_paths,
                               *,
                               rng=None,
                               delta_W=None):
    """Convenience: Milstein terminal value only.

    Returns only the terminal column of ``simulate_path_milstein``.
    See that function for the full parameter description.
    """
    paths = simulate_path_milstein(S0, r, sigma, T, n_steps, n_paths,
                                   rng=rng, delta_W=delta_W)
    return paths[:, -1]
