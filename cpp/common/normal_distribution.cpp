#include "normal_distribution.h"
#include <cmath>

namespace quant {

double standard_normal_cdf(double x) {
    // C++17 onwards provides std::erfc.
    // Phi(x) = 0.5 * erfc(-x / sqrt(2))
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double standard_normal_pdf(double x) {
    static const double inv_sqrt_2pi = 1.0 / std::sqrt(2.0 * M_PI);
    return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

}  // namespace quant