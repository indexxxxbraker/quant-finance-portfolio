// main.cpp
//
// Smoke test: replicates the output of python/quantlib/black_scholes.py
// when run as a script. If the C++ output matches the Python output, the
// two implementations agree at this single anchor point.

#include "black_scholes.hpp"

#include <cmath>     // std::abs, std::exp
#include <iomanip>   // std::setprecision, std::fixed, std::scientific
#include <iostream>  // std::cout

int main() {
    // Hull, Options, Futures, and Other Derivatives (10th ed.), Example 15.6.
    const double S     = 42.0;
    const double K     = 40.0;
    const double r     = 0.10;
    const double sigma = 0.20;
    const double T     = 0.5;

    const double C = quant::call_price(S, K, r, sigma, T);
    const double P = quant::put_price(S, K, r, sigma, T);

    // std::cout works like Python's print but uses the << operator to chain
    // values. std::fixed + std::setprecision(4) gives 4 decimals, like
    // Python's f"{x:.4f}".
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Call: " << C << "   (Hull: 4.7594)\n";
    std::cout << "Put:  " << P << "   (Hull: 0.8086)\n";

    // Put-call parity residual.
    const double residual = std::abs((C - P) - (S - K * std::exp(-r * T)));
    std::cout << std::scientific << std::setprecision(2);
    std::cout << "Put-call parity residual: " << residual << "\n";

    return 0;  // Convention: 0 means "executed successfully".
}
