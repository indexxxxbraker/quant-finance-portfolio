// main.cpp
//
// Smoke test: prints prices, Greeks, implied volatility recovery at
// the Hull example point, plus a Hull example for IV, plus the five
// Monte Carlo pricers (exact sampler from Block 1.1, Euler-Maruyama
// from Block 1.2.1, Milstein from Block 1.2.2, antithetic variates
// from Block 2.1, control variates -- both controls -- from Block 2.2).

#include "black_scholes.hpp"
#include "implied_volatility.hpp"
#include "monte_carlo.hpp"
#include "variance_reduction.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

int main() {
    // Hull (10th ed.), Example 15.6: pricing.
    const double S = 42.0, K = 40.0, r = 0.10, sigma = 0.20, T = 0.5;

    const double C = quant::call_price(S, K, r, sigma, T);
    const double P = quant::put_price(S, K, r, sigma, T);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Call: " << C << "   (Hull: 4.7594)\n";
    std::cout << "Put:  " << P << "   (Hull: 0.8086)\n";

    const double residual = std::abs((C - P) - (S - K * std::exp(-r * T)));
    std::cout << std::scientific << std::setprecision(2);
    std::cout << "Put-call parity residual: " << residual << "\n\n";

    // Round-trip test for implied volatility.
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Implied volatility round-trip:\n";
    const double iv = quant::implied_volatility(C, S, K, r, T);
    std::cout << "  sigma_true=" << sigma << "  iv=" << iv
              << "  err=" << std::scientific << std::abs(iv - sigma) << "\n\n";

    // Hull (10th ed.), Example 19.6: implied volatility.
    std::cout << std::fixed << std::setprecision(6);
    const double iv_hull = quant::implied_volatility(1.875, 21.0, 20.0, 0.10, 0.25);
    std::cout << "Hull Example 19.6: iv=" << iv_hull << "  (Hull: ~0.235)\n\n";

    // Monte Carlo benchmarks (ATM, 1y, sigma=0.20). BS price ~10.4506.
    std::cout << std::fixed << std::setprecision(6);
    {
        std::mt19937_64 rng(42);
        const auto mc_exact = quant::mc_european_call_exact(
            100.0, 100.0, 0.05, 0.20, 1.0, 100'000, rng);
        std::cout << "Exact MC pricer (Block 1.1, n_paths=100000):\n"
                  << "  Estimate    : " << mc_exact.estimate
                  << "  (BS: 10.4506)\n"
                  << "  Half-width  : " << mc_exact.half_width << "\n"
                  << "  Sample var  : " << mc_exact.sample_variance << "\n"
                  << "  N paths     : " << mc_exact.n_paths << "\n\n";
    }
    {
        std::mt19937_64 rng(42);
        const auto mc_euler = quant::mc_european_call_euler(
            100.0, 100.0, 0.05, 0.20, 1.0, 100, 100'000, rng);
        std::cout << "Euler MC pricer (Block 1.2.1, n_steps=100, n_paths=100000):\n"
                  << "  Estimate    : " << mc_euler.estimate
                  << "  (BS: 10.4506)\n"
                  << "  Half-width  : " << mc_euler.half_width << "\n"
                  << "  Sample var  : " << mc_euler.sample_variance << "\n"
                  << "  N paths     : " << mc_euler.n_paths << "\n\n";
    }
    {
        std::mt19937_64 rng(42);
        const auto mc_milstein = quant::mc_european_call_milstein(
            100.0, 100.0, 0.05, 0.20, 1.0, 100, 100'000, rng);
        std::cout << "Milstein MC pricer (Block 1.2.2, n_steps=100, n_paths=100000):\n"
                  << "  Estimate    : " << mc_milstein.estimate
                  << "  (BS: 10.4506)\n"
                  << "  Half-width  : " << mc_milstein.half_width << "\n"
                  << "  Sample var  : " << mc_milstein.sample_variance << "\n"
                  << "  N paths     : " << mc_milstein.n_paths << "\n\n";
    }
    {
        std::mt19937_64 rng(42);
        const auto mc_av = quant::mc_european_call_exact_av(
            100.0, 100.0, 0.05, 0.20, 1.0, 50'000, rng);
        std::cout << "AV MC pricer (Block 2.1, n_pairs=50000 = 100000 payoffs):\n"
                  << "  Estimate    : " << mc_av.estimate
                  << "  (BS: 10.4506)\n"
                  << "  Half-width  : " << mc_av.half_width << "\n"
                  << "  Sample var  : " << mc_av.sample_variance << "\n"
                  << "  N pairs     : " << mc_av.n_paths << "\n\n";
    }
    {
        std::mt19937_64 rng(42);
        const auto mc_cv1 = quant::mc_european_call_exact_cv_underlying(
            100.0, 100.0, 0.05, 0.20, 1.0, 100'000, rng);
        std::cout << "CV MC pricer (Block 2.2, control=underlying, n_paths=100000):\n"
                  << "  Estimate    : " << mc_cv1.estimate
                  << "  (BS: 10.4506)\n"
                  << "  Half-width  : " << mc_cv1.half_width << "\n"
                  << "  Sample var  : " << mc_cv1.sample_variance << "\n"
                  << "  N paths     : " << mc_cv1.n_paths << "\n\n";
    }
    {
        std::mt19937_64 rng(42);
        const auto mc_cv2 = quant::mc_european_call_exact_cv_aon(
            100.0, 100.0, 0.05, 0.20, 1.0, 100'000, rng);
        std::cout << "CV MC pricer (Block 2.2, control=AON, n_paths=100000):\n"
                  << "  Estimate    : " << mc_cv2.estimate
                  << "  (BS: 10.4506)\n"
                  << "  Half-width  : " << mc_cv2.half_width << "\n"
                  << "  Sample var  : " << mc_cv2.sample_variance << "\n"
                  << "  N paths     : " << mc_cv2.n_paths << "\n";
    }

    return 0;
}
