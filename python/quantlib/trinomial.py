"""
Trinomial tree of Kamrad-Ritchken for European and American options.

Implements the lambda-parametrised trinomial tree:

    x -> x + dx  with prob p_u = 1/(2 lambda) + nu sqrt(dt) / (2 sigma sqrt(lambda))
    x -> x       with prob p_m = 1 - 1/lambda
    x -> x - dx  with prob p_d = 1/(2 lambda) - nu sqrt(dt) / (2 sigma sqrt(lambda))

with dx = sigma sqrt(lambda dt) and nu = r - sigma^2/2. Kamrad and
Ritchken write the step as lambda_KR sigma sqrt(dt), so the lambda used
here is lambda_KR^2.

The default lambda = 3 gives (p_u, p_m, p_d) ~ (1/6, 2/3, 1/6) with
nodes 0 and +-sigma sqrt(3 dt): the three-point Gauss-Hermite rule for
N(0, sigma^2 dt), exact for polynomials up to degree 5. At leading order
the step then matches the Gaussian moments up to the fifth: the mean and
variance match for any lambda, while E[dX^3] = lambda sigma^2 nu dt^2
and E[dX^4] = lambda sigma^4 dt^2 equal the Gaussian 3 sigma^2 nu dt^2
and 3 sigma^4 dt^2 only for lambda = 3. For lambda = 1, p_m = 0 and the
tree degenerates to a binomial-equivalent.

The trinomial backward induction is structurally equivalent to the
FTCS finite-difference scheme (quantlib.ftcs) with alpha = 1/(2 lambda).
This is a structural insight, not a coincidence: see
theory/phase3/trinomial.tex for the explicit identification of coefficients.

References
----------
Boyle (1986), International Options Journal 3, 7-12.
Kamrad and Ritchken (1991), Management Science 37(12), 1640-1652.
"""

import numpy as np


def _validate_inputs(S, K, r, sigma, T, n_steps, lambda_param):
    """Validate inputs common to all trinomial pricers.

    Raises ValueError on any invalid combination.
    """
    if S <= 0:
        raise ValueError(f"S must be positive, got {S}")
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if lambda_param < 1.0:
        raise ValueError(
            f"lambda_param must be >= 1 (else p_m < 0); got {lambda_param}"
        )


def _trinomial_probabilities(r, sigma, dt, lambda_param):
    """Compute the KR probabilities p_u, p_m, p_d.

    Returns
    -------
    (p_u, p_m, p_d) : tuple of float
        Each in [0, 1] under the validation conditions above. Their sum
        is 1 by construction.

    Notes
    -----
    For project parameters (|nu| sqrt(dt) << sigma sqrt(lambda)) the
    drift correction is small. We do not check positivity of p_d at
    runtime: if it failed, it would only be because the user picked
    pathological parameters (extreme dt, near-zero sigma, etc.) and
    the diagnostic would surface as an immediately visible bug.
    """
    nu = r - 0.5 * sigma * sigma
    drift_term = nu * np.sqrt(dt) / (2.0 * sigma * np.sqrt(lambda_param))
    p_u = 0.5 / lambda_param + drift_term
    p_m = 1.0 - 1.0 / lambda_param
    p_d = 0.5 / lambda_param - drift_term
    return p_u, p_m, p_d


def _trinomial_backward(S, K, r, sigma, T, n_steps, lambda_param,
                        payoff_type):
    """Run the backward induction for any trinomial pricer.

    Parameters
    ----------
    payoff_type : str
        One of 'european_call', 'european_put', 'american_put'.

    Returns
    -------
    float
        Price at the root of the tree.

    Notes
    -----
    Uses an in-place array of length 2 * n_steps + 1 for memory
    efficiency. Cost: O(n_steps^2) flops, O(n_steps) memory.
    """
    dt = T / n_steps
    dx = sigma * np.sqrt(lambda_param * dt)
    p_u, p_m, p_d = _trinomial_probabilities(r, sigma, dt, lambda_param)
    disc = np.exp(-r * dt)

    # Terminal payoffs at j = -n_steps, ..., +n_steps.
    n_nodes = 2 * n_steps + 1
    j_grid = np.arange(-n_steps, n_steps + 1, dtype=np.float64)
    S_terminal = S * np.exp(j_grid * dx)
    if payoff_type == 'european_call':
        V = np.maximum(S_terminal - K, 0.0)
    elif payoff_type in ('european_put', 'american_put'):
        V = np.maximum(K - S_terminal, 0.0)
    else:
        raise ValueError(f"unknown payoff_type: {payoff_type!r}")

    # Backward induction. At step n we have 2*(N - n) + 1 active nodes;
    # we shrink the array by 2 per step (one off each end).
    is_american = (payoff_type == 'american_put')
    # We reuse the same buffer, shrinking by reading from a sliced view.
    for n in range(n_steps - 1, -1, -1):
        # New centre is at the same position; new j range is -n..+n.
        # Active nodes at step n+1 are V[start:end] where start = n_steps - (n+1)
        # and end = n_steps + (n+1) + 1.
        # New nodes at step n correspond to indices V[start+1:end-1].
        # The recurrence reads V[i-1], V[i], V[i+1] (at step n+1) to
        # produce V_new[i] (at step n).
        # We compute new values into a fresh slice and write them back.
        start = n_steps - (n + 1)
        end   = n_steps + (n + 1) + 1
        # Continuation values for the next layer (length 2n+1).
        cont = disc * (
            p_u * V[start + 2 : end]
          + p_m * V[start + 1 : end - 1]
          + p_d * V[start     : end - 2]
        )
        if is_american:
            # Spot prices at this layer's nodes: j = -n .. +n.
            j_n = np.arange(-n, n + 1, dtype=np.float64)
            S_n = S * np.exp(j_n * dx)
            exer = np.maximum(K - S_n, 0.0)
            V_new = np.maximum(cont, exer)
        else:
            V_new = cont
        # Write back into the corresponding slice of V.
        V[start + 1 : end - 1] = V_new

    # The root is at index n_steps (the centre of the original array).
    return float(V[n_steps])


# ---------------------------------------------------------------------------
# High-level pricers
# ---------------------------------------------------------------------------
def trinomial_european_call(S, K, r, sigma, T, *, n_steps,
                            lambda_param=3.0):
    """Price a European call by the Kamrad-Ritchken trinomial tree.

    Parameters
    ----------
    S, K, r, sigma, T : float
        Standard Black-Scholes parameters.
    n_steps : int (keyword-only)
        Number of time steps in the tree.
    lambda_param : float, default 3.0
        Stretching parameter; must be >= 1. Default 3 (lambda_KR =
        sqrt(3)) gives the nodes and weights of the three-point
        Gauss-Hermite rule. Set to 1 to recover binomial-equivalent
        behaviour (p_m = 0). For any lambda the recursion matches FTCS
        with alpha = 1/(2 lambda) up to where the discount factor is
        applied, an O(dt) difference; the two coincide only when r = 0.

    Returns
    -------
    float
        Trinomial approximation to the Black-Scholes call price.
    """
    _validate_inputs(S, K, r, sigma, T, n_steps, lambda_param)
    return _trinomial_backward(S, K, r, sigma, T, n_steps, lambda_param,
                                'european_call')


def trinomial_european_put(S, K, r, sigma, T, *, n_steps,
                           lambda_param=3.0):
    """Price a European put by the Kamrad-Ritchken trinomial tree.
    See trinomial_european_call.
    """
    _validate_inputs(S, K, r, sigma, T, n_steps, lambda_param)
    return _trinomial_backward(S, K, r, sigma, T, n_steps, lambda_param,
                                'european_put')


def trinomial_american_put(S, K, r, sigma, T, *, n_steps,
                           lambda_param=3.0):
    """Price an American put by the Kamrad-Ritchken trinomial tree.

    The early-exercise check is applied at every node at every step:
    V_node = max(intrinsic value, discounted continuation value).

    Note: there is no trinomial_american_call function because, for a
    non-dividend-paying stock, the American call equals the European
    call (Merton 1973).
    """
    _validate_inputs(S, K, r, sigma, T, n_steps, lambda_param)
    return _trinomial_backward(S, K, r, sigma, T, n_steps, lambda_param,
                                'american_put')
