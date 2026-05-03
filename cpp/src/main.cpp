// main.cpp
//
// Smoke test: prints prices, Greeks, and implied volatility recovery at the
// Hull example point, plus a Hull example for IV.

#include "black_scholes.hpp"
#include "implied_volatility.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>

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
    std::cout << "Hull Example 19.6: iv=" << iv_hull << "  (Hull: ~0.235)\n";

    return 0;
}
