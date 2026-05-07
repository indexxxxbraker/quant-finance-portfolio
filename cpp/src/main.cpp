// main.cpp
//
// Smoke test: prints prices, Greeks, implied volatility recovery at
// the Hull example point, plus a Hull example for IV, plus the seven
// Monte Carlo pricers (exact sampler from Block 1.1, Euler-Maruyama
// from Block 1.2.1, Milstein from Block 1.2.2, antithetic variates
// from Block 2.1, control variates -- both controls -- from Block 2.2,
// QMC-Sobol and RQMC-Sobol from Block 3), plus a comparison of Greek
// estimators (Delta via three methods, Vega via three methods, Gamma
// via bumping) from Block 4.

#include "black_scholes.hpp"
#include "implied_volatility.hpp"
#include "monte_carlo.hpp"
#include "variance_reduction.hpp"
#include "qmc.hpp"
#include "greeks.hpp"
#include "asian.hpp"

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
                  << "  N paths     : " << mc_cv2.n_paths << "\n\n";
    }
    {
        const double est = quant::mc_european_call_euler_qmc(
            100.0, 100.0, 0.05, 0.20, 1.0, 8192, 20, "sobol");
        std::cout << "QMC Sobol pricer (Block 3, n=8192, N=20):\n"
                  << "  Estimate    : " << est
                  << "  (BS: 10.4506)\n"
                  << "  Half-width  : N/A (deterministic estimator)\n\n";
    }
    {
        std::mt19937_64 rng(42);
        const auto rqmc = quant::mc_european_call_euler_rqmc(
            100.0, 100.0, 0.05, 0.20, 1.0, 4096, 20, 20, rng);
        std::cout << "RQMC Sobol pricer (Block 3, n=4096, R=20, N=20):\n"
                  << "  Estimate    : " << rqmc.estimate
                  << "  (BS: 10.4506)\n"
                  << "  Half-width  : " << rqmc.half_width << "\n"
                  << "  Sample var  : " << rqmc.sample_variance << "\n"
                  << "  N (= R)     : " << rqmc.n_paths << "\n\n";
    }
    {
        const double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
        const std::size_t n = 100'000;

        std::cout << "Greek estimators (Block 4, n=100000):\n";
        std::cout << "  BS Delta = " << quant::call_delta(S, K, r, sigma, T) << "\n";
        std::cout << "  BS Vega  = " << quant::vega(S, K, r, sigma, T) << "\n";
        std::cout << "  BS Gamma = " << quant::gamma(S, K, r, sigma, T) << "\n\n";

        {
            std::mt19937_64 rng_b(42), rng_p(42), rng_l(42);
            const auto db = quant::delta_bump(S, K, r, sigma, T, n, rng_b);
            const auto dp = quant::delta_pathwise(S, K, r, sigma, T, n, rng_p);
            const auto dl = quant::delta_lr(S, K, r, sigma, T, n, rng_l);
            std::cout << "  Delta bump    : " << db.estimate << "  hw " << db.half_width << "\n";
            std::cout << "  Delta pathwise: " << dp.estimate << "  hw " << dp.half_width << "\n";
            std::cout << "  Delta LR      : " << dl.estimate << "  hw " << dl.half_width << "\n";
        }
        {
            std::mt19937_64 rng_b(42), rng_p(42), rng_l(42);
            const auto vb = quant::vega_bump(S, K, r, sigma, T, n, rng_b);
            const auto vp = quant::vega_pathwise(S, K, r, sigma, T, n, rng_p);
            const auto vl = quant::vega_lr(S, K, r, sigma, T, n, rng_l);
            std::cout << "  Vega bump     : " << vb.estimate << "  hw " << vb.half_width << "\n";
            std::cout << "  Vega pathwise : " << vp.estimate << "  hw " << vp.half_width << "\n";
            std::cout << "  Vega LR       : " << vl.estimate << "  hw " << vl.half_width << "\n";
        }
        {
            std::mt19937_64 rng(42);
            const auto gb = quant::gamma_bump(S, K, r, sigma, T, n, rng);
            std::cout << "  Gamma bump    : " << gb.estimate << "  hw " << gb.half_width << "\n";
        }
    }
    // ===== Asian options (Block 5) =====
    {
        std::cout << "\n--- Phase 2 Block 5: Asian options ---\n";

        constexpr double S0 = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
        constexpr std::size_t N = 50;
        constexpr std::size_t n = 100'000;
        std::mt19937_64 rng(42);

        const double cg_closed = quant::geometric_asian_call_closed_form(
                                    S0, K, r, sigma, T, N);
        std::cout << "Geometric Asian closed form (N=" << N << "): "
                << cg_closed << "\n";

        std::mt19937_64 rng_g(42);
        const auto res_g = quant::mc_asian_call_geometric_iid(
                                S0, K, r, sigma, T, n, N, rng_g);
        std::cout << "Geometric IID (n=" << n << "): "
                << res_g.estimate << " +- " << res_g.half_width
                << " (err/hw = "
                << std::abs(res_g.estimate - cg_closed) / res_g.half_width
                << ")\n";

        std::mt19937_64 rng_a(42);
        const auto res_a_iid = quant::mc_asian_call_arithmetic_iid(
                                    S0, K, r, sigma, T, n, N, rng_a);
        std::cout << "Arithmetic IID (n=" << n << "): "
                << res_a_iid.estimate << " +- " << res_a_iid.half_width << "\n";

        std::mt19937_64 rng_cv(42);
        const auto res_a_cv = quant::mc_asian_call_arithmetic_cv(
                                    S0, K, r, sigma, T, n, N, rng_cv);
        std::cout << "Arithmetic CV  (n=" << n << "): "
                << res_a_cv.estimate << " +- " << res_a_cv.half_width << "\n";

        const double vrf = (res_a_iid.half_width / res_a_cv.half_width)
                            * (res_a_iid.half_width / res_a_cv.half_width);
        std::cout << "VRF (CV vs IID): " << vrf << "x\n";
    }

    return 0;
}
