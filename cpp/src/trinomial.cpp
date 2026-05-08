// trinomial.cpp - implementation of the Kamrad-Ritchken trinomial tree.

#include "trinomial.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace quant {
namespace pde {

namespace {

// ---------------------------------------------------------------------
// Common input validation.
// ---------------------------------------------------------------------
void validate_inputs(double S, double K, double r, double sigma, double T,
                     int n_steps, double lambda_param) {
    (void)r;  // r can be any sign in principle; no constraint
    if (S <= 0.0) {
        throw std::invalid_argument(
            "trinomial: S must be positive, got " + std::to_string(S));
    }
    if (K <= 0.0) {
        throw std::invalid_argument(
            "trinomial: K must be positive, got " + std::to_string(K));
    }
    if (sigma <= 0.0) {
        throw std::invalid_argument(
            "trinomial: sigma must be positive, got " + std::to_string(sigma));
    }
    if (T <= 0.0) {
        throw std::invalid_argument(
            "trinomial: T must be positive, got " + std::to_string(T));
    }
    if (n_steps < 1) {
        throw std::invalid_argument(
            "trinomial: n_steps must be >= 1, got " + std::to_string(n_steps));
    }
    if (lambda_param < 1.0) {
        throw std::invalid_argument(
            "trinomial: lambda_param must be >= 1 (else p_m < 0), got " +
            std::to_string(lambda_param));
    }
}

// ---------------------------------------------------------------------
// Backward induction core. payoff_type:
//   0 = european_call, 1 = european_put, 2 = american_put.
//
// The recurrence V_new[i] = f(V_old[i-1], V_old[i], V_old[i+1]) requires
// a separate buffer for the new values: an in-place sweep in either
// direction would corrupt at least one of the three reads. We use a
// pre-allocated scratch buffer of the same size as V to keep memory
// allocation out of the hot loop.
// ---------------------------------------------------------------------
double trinomial_backward(double S, double K, double r, double sigma,
                          double T, int n_steps, double lambda_param,
                          int payoff_type) {
    const double dt = T / static_cast<double>(n_steps);
    const double dx = sigma * std::sqrt(lambda_param * dt);
    const double nu = r - 0.5 * sigma * sigma;

    const double drift_term =
        nu * std::sqrt(dt) / (2.0 * sigma * std::sqrt(lambda_param));
    const double p_u = 0.5 / lambda_param + drift_term;
    const double p_m = 1.0 - 1.0 / lambda_param;
    const double p_d = 0.5 / lambda_param - drift_term;
    const double disc = std::exp(-r * dt);

    // Terminal payoffs: 2*n_steps + 1 nodes at j = -n_steps, ..., +n_steps.
    const std::size_t n_nodes = static_cast<std::size_t>(2 * n_steps + 1);
    std::vector<double> V(n_nodes);
    std::vector<double> V_scratch(n_nodes);

    for (std::size_t idx = 0; idx < n_nodes; ++idx) {
        const int j = static_cast<int>(idx) - n_steps;
        const double S_j = S * std::exp(static_cast<double>(j) * dx);
        switch (payoff_type) {
            case 0:  // european call
                V[idx] = std::max(S_j - K, 0.0);
                break;
            case 1:  // european put
            case 2:  // american put (same terminal payoff)
                V[idx] = std::max(K - S_j, 0.0);
                break;
            default:
                throw std::invalid_argument(
                    "trinomial_backward: unknown payoff_type");
        }
    }

    const bool is_american = (payoff_type == 2);

    // Backward induction. At step n the active nodes occupy V[start..end).
    for (int n = n_steps - 1; n >= 0; --n) {
        const std::size_t start = static_cast<std::size_t>(n_steps - (n + 1));
        const std::size_t end   = static_cast<std::size_t>(n_steps + (n + 1) + 1);

        // Each new node at step n (j = -n .. +n) is at index in (start, end-1).
        for (std::size_t i = start + 1; i + 1 < end; ++i) {
            const double cont =
                disc * (p_u * V[i + 1] + p_m * V[i] + p_d * V[i - 1]);

            if (is_american) {
                const int j_node = static_cast<int>(i) - n_steps;
                const double S_node =
                    S * std::exp(static_cast<double>(j_node) * dx);
                const double exer = std::max(K - S_node, 0.0);
                V_scratch[i] = std::max(cont, exer);
            } else {
                V_scratch[i] = cont;
            }
        }

        // Copy the new layer back into V. Only the active range is updated.
        for (std::size_t i = start + 1; i + 1 < end; ++i) {
            V[i] = V_scratch[i];
        }
    }

    // The root is at the centre index n_steps.
    return V[static_cast<std::size_t>(n_steps)];
}

}  // namespace

// ---------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------
double trinomial_european_call(double S, double K, double r, double sigma,
                                double T, int n_steps, double lambda_param) {
    validate_inputs(S, K, r, sigma, T, n_steps, lambda_param);
    return trinomial_backward(S, K, r, sigma, T, n_steps, lambda_param, 0);
}

double trinomial_european_put(double S, double K, double r, double sigma,
                                double T, int n_steps, double lambda_param) {
    validate_inputs(S, K, r, sigma, T, n_steps, lambda_param);
    return trinomial_backward(S, K, r, sigma, T, n_steps, lambda_param, 1);
}

double trinomial_american_put(double S, double K, double r, double sigma,
                                double T, int n_steps, double lambda_param) {
    validate_inputs(S, K, r, sigma, T, n_steps, lambda_param);
    return trinomial_backward(S, K, r, sigma, T, n_steps, lambda_param, 2);
}

}  // namespace pde
}  // namespace quant
