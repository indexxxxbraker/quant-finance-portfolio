#pragma once

// trinomial.hpp - Kamrad-Ritchken trinomial tree for European and American options.
//
// Phase 3, Block 5. The lambda-parametrised trinomial:
//
//     x -> x + dx  with prob p_u = 1/(2L) + nu sqrt(dt) / (2 sigma sqrt(L))
//     x -> x       with prob p_m = 1 - 1/L
//     x -> x - dx  with prob p_d = 1/(2L) - nu sqrt(dt) / (2 sigma sqrt(L))
//
// where L = lambda_param, dx = sigma sqrt(L dt), nu = r - sigma^2/2.
//
// The default lambda = 3 corresponds to (p_u, p_m, p_d) ~ (1/6, 2/3, 1/6),
// the weights of Simpson's rule for numerical integration.
//
// This scheme is structurally equivalent to FTCS (Block 1) with
// alpha = 1/(2 lambda) up to an O(dt) discount-treatment difference.

namespace quant {
namespace pde {

// European call price by trinomial tree.
//
// Throws std::invalid_argument if any parameter is invalid:
//   - S, K, sigma, T must be positive
//   - n_steps must be >= 1
//   - lambda_param must be >= 1 (else p_m < 0)
double trinomial_european_call(double S, double K, double r, double sigma,
                                double T, int n_steps,
                                double lambda_param = 3.0);

// European put price by trinomial tree. Same parameter contract.
double trinomial_european_put(double S, double K, double r, double sigma,
                                double T, int n_steps,
                                double lambda_param = 3.0);

// American put price by trinomial tree. Same parameter contract.
//
// Note: there is no trinomial_american_call function. For a non-dividend-
// paying stock, the American call equals the European call (Merton 1973);
// users should call trinomial_european_call instead.
double trinomial_american_put(double S, double K, double r, double sigma,
                                double T, int n_steps,
                                double lambda_param = 3.0);

}  // namespace pde
}  // namespace quant
