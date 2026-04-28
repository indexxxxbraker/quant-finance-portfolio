#include "normal_distribution.h"
#include <iostream>
#include <iomanip>
#include <vector>

int main() {
    const std::vector<double> test_points = {-3.0, -1.0, 0.0, 1.0, 3.0};

    std::cout << "Standard normal CDF values:\n";
    std::cout << std::fixed << std::setprecision(6);

    for (double x : test_points) {
        std::cout << "  Phi(" << std::showpos << x << std::noshowpos
                  << ") = " << quant::standard_normal_cdf(x) << "\n";
    }

    return 0;
}