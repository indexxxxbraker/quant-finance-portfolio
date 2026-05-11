"""
Phase 4 Block 7 - Final benchmark.

Orchestrates all Heston pricers in `quantlib` to produce reproducible
artifacts in python/results/phase4/ demonstrating:

  - Section 1 (essential): cross-validation of 4 pricing methods
        (Fourier, MC Euler, MC QE, PDE ADI) at K in [80, 120]
  - Section 2 (essential): calibration to synthetic noisy vol surface
        (LM with vega weighting, 4 maturities)
  - Section 3 (essential): empirical verification of PDE O(h^2)
        convergence rate (log-log)
  - Section 4 (optional):  exotic option prices (Asian, Lookback,
        Barrier up-and-out, American put)
  - Section 5 (optional):  Andersen QE vs full-truncation Euler bias
        as a function of n_steps

Outputs: per-section CSV (machine-readable) and PNG (presentation) in
python/results/phase4/, plus tables printed to stdout.

Run from the python/ directory:

    python benchmark_phase4.py

Total runtime ~5-10 minutes at production sizes.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Pricer imports
# --------------------------------------------------------------------------
from quantlib.heston_fourier import heston_call_lewis, black_scholes_call
from quantlib.heston_mc import mc_european_call_heston
from quantlib.heston_qe import mc_european_call_heston_qe
from quantlib.heston_pde import heston_call_pde
from quantlib.heston_calibration import calibrate_heston, implied_vol_bs
from quantlib.heston_exotics import (
    mc_asian_call_heston,
    mc_lookback_call_heston,
    mc_barrier_call_heston,
)
from quantlib.heston_american_pde import heston_american_put_pde


# ==========================================================================
# Configuration
# ==========================================================================

# Canonical Heston parameters (well inside Feller-OK regime: 2*kappa*theta/sigma^2 = 1.78)
HESTON = dict(kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, v0=0.04)
S0 = 100.0
R  = 0.03
T  = 1.0

# Strikes for cross-validation (Section 1)
STRIKES_CROSS = np.arange(80.0, 121.0, 5.0)  # 80, 85, ..., 120  (9 strikes)

# MC and PDE production sizes (Section 1)
MC_N_PATHS = 200_000
MC_N_STEPS = 200
PDE_N_X    = 200
PDE_N_V    = 100
PDE_N_TAU  = 100

# Calibration (Section 2)
MATURITIES_CALIB = [0.25, 0.5, 1.0, 2.0]
STRIKES_CALIB    = np.arange(80.0, 121.0, 5.0)
IV_NOISE_STD     = 0.005  # 0.5% Gaussian noise in IV space

# Convergence grids (Section 3): doubling (N_X, N_v) together
CONV_GRIDS = [
    ( 25,  12,  50),
    ( 50,  25,  50),
    (100,  50, 100),
    (200, 100, 200),
    (400, 200, 400),
]
CONV_K = 100.0   # ATM strike for convergence study

# Exotics (Section 4)
EXOTIC_N_PATHS = 100_000
EXOTIC_N_STEPS = 200
EXOTIC_K       = 100.0
BARRIER_H      = 130.0   # up-and-out barrier above S0

# QE vs Euler bias (Section 5).
# Uses aggressive Heston params (Feller << 1) where the schemes differ
# significantly. At canonical params (Feller=1.78) both methods are
# similarly accurate and the comparison is not informative.
QE_VS_EULER_HESTON  = dict(kappa=0.5, theta=0.04, sigma=1.0, rho=-0.9, v0=0.04)
# Feller = 2 * 0.5 * 0.04 / 1.0**2 = 0.04 (aggressive regime)
QE_VS_EULER_N_STEPS = [25, 50, 100, 200, 400]
QE_VS_EULER_M       = 100_000
QE_VS_EULER_K       = 100.0
QE_VS_EULER_N_SEEDS = 3

RNG_SEED = 42

# Output directory
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "phase4"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================================
# Section 1: cross-validation
# ==========================================================================

def cross_validation_section():
    """Compute prices for the European call via 4 methods at each strike."""
    print()
    print("=" * 102)
    print("  SECTION 1: Cross-validation of 4 pricing methods")
    print(f"  S0={S0}, K in {STRIKES_CROSS.tolist()}, T={T}, r={R}")
    print(f"  Heston: kappa={HESTON['kappa']}, theta={HESTON['theta']}, "
          f"sigma={HESTON['sigma']}, rho={HESTON['rho']}, v0={HESTON['v0']}")
    feller = 2 * HESTON["kappa"] * HESTON["theta"] / HESTON["sigma"]**2
    print(f"  Feller parameter 2*kappa*theta/sigma^2 = {feller:.3f}")
    print(f"  MC: n_paths={MC_N_PATHS}, n_steps={MC_N_STEPS}")
    print(f"  PDE: N_X={PDE_N_X}, N_v={PDE_N_V}, N_tau={PDE_N_TAU}")
    print("=" * 102)

    print(f"\n  {'K':>5}  {'Fourier':>11}  {'MC Euler':>22}  "
          f"{'MC QE':>22}  {'PDE ADI':>11}")
    print("  " + "-" * 80)

    rows = []
    for K in STRIKES_CROSS:
        # Fourier (reference)
        price_f = heston_call_lewis(
            K, T, S0, HESTON["v0"], R,
            HESTON["kappa"], HESTON["theta"],
            HESTON["sigma"], HESTON["rho"])

        # MC Euler
        t0 = time.perf_counter()
        r_e = mc_european_call_heston(
            S0, K, HESTON["v0"], R,
            HESTON["kappa"], HESTON["theta"],
            HESTON["sigma"], HESTON["rho"],
            T, MC_N_STEPS, MC_N_PATHS, seed=RNG_SEED)
        t_e = time.perf_counter() - t0

        # MC QE
        t0 = time.perf_counter()
        r_q = mc_european_call_heston_qe(
            S0, K, HESTON["v0"], R,
            HESTON["kappa"], HESTON["theta"],
            HESTON["sigma"], HESTON["rho"],
            T, MC_N_STEPS, MC_N_PATHS, seed=RNG_SEED)
        t_q = time.perf_counter() - t0

        # PDE
        t0 = time.perf_counter()
        price_pde = heston_call_pde(
            S0, K, T,
            HESTON["kappa"], HESTON["theta"],
            HESTON["sigma"], HESTON["rho"],
            HESTON["v0"], R,
            N_X=PDE_N_X, N_v=PDE_N_V, N_tau=PDE_N_TAU)
        t_pde = time.perf_counter() - t0

        rows.append({
            "K": float(K),
            "fourier": price_f,
            "euler_price": r_e.estimate, "euler_hw": r_e.half_width,
            "qe_price":    r_q.estimate, "qe_hw":    r_q.half_width,
            "pde_price":   price_pde,
            "time_euler_s": t_e, "time_qe_s": t_q, "time_pde_s": t_pde,
        })
        print(f"  {K:>5.1f}  {price_f:>11.6f}  "
              f"{r_e.estimate:>11.6f}+/-{r_e.half_width:.4f}  "
              f"{r_q.estimate:>11.6f}+/-{r_q.half_width:.4f}  "
              f"{price_pde:>11.6f}")

    return rows


def plot_cross_validation(rows):
    """Two-panel plot: absolute prices on top, relative differences below."""
    Ks       = np.array([r["K"]           for r in rows])
    fourier  = np.array([r["fourier"]     for r in rows])
    euler    = np.array([r["euler_price"] for r in rows])
    euler_hw = np.array([r["euler_hw"]    for r in rows])
    qe       = np.array([r["qe_price"]    for r in rows])
    qe_hw    = np.array([r["qe_hw"]       for r in rows])
    pde      = np.array([r["pde_price"]   for r in rows])

    rel_euler    = (euler - fourier) / fourier
    rel_euler_hw = euler_hw / fourier
    rel_qe       = (qe    - fourier) / fourier
    rel_qe_hw    = qe_hw / fourier
    rel_pde      = (pde   - fourier) / fourier

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 9), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.2]})

    # --- Top: absolute prices ---
    ax1.plot(Ks, fourier, "k-", linewidth=2.5, label="Fourier (Lewis)",
              zorder=10)
    ax1.errorbar(Ks, euler, yerr=euler_hw, fmt="o", color="C0", capsize=4,
                  label="MC full-truncation Euler", alpha=0.85)
    ax1.errorbar(Ks, qe, yerr=qe_hw, fmt="s", color="C3", capsize=4,
                  label="MC Andersen QE", alpha=0.85)
    ax1.plot(Ks, pde, "D-", color="C2", markersize=9, linewidth=1.5,
              label="PDE 2D Douglas ADI", alpha=0.9)
    ax1.set_ylabel("European call price")
    ax1.set_title(
        "Heston cross-validation: 4 methods at canonical parameters\n"
        f"S0={S0}, T={T}, r={R},  "
        f"κ={HESTON['kappa']}, θ={HESTON['theta']}, σ={HESTON['sigma']}, "
        f"ρ={HESTON['rho']}, v0={HESTON['v0']}")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # --- Bottom: relative differences ---
    ax2.axhline(0, color="black", linewidth=2.5, label="Fourier (reference)")
    ax2.fill_between(Ks, rel_euler - rel_euler_hw, rel_euler + rel_euler_hw,
                       color="C0", alpha=0.20)
    ax2.plot(Ks, rel_euler, "o-", color="C0", linewidth=1.5, markersize=8,
              label="MC full-truncation Euler (95% CI)")
    ax2.fill_between(Ks, rel_qe - rel_qe_hw, rel_qe + rel_qe_hw,
                       color="C3", alpha=0.20)
    ax2.plot(Ks, rel_qe, "s-", color="C3", linewidth=1.5, markersize=8,
              label="MC Andersen QE (95% CI)")
    ax2.plot(Ks, rel_pde, "D-", color="C2", linewidth=1.5, markersize=8,
              label="PDE 2D Douglas ADI")
    ax2.set_xlabel("Strike K")
    ax2.set_ylabel(r"$(C_{\mathrm{method}} - C_{\mathrm{Fourier}}) "
                     r"/ C_{\mathrm{Fourier}}$")
    ax2.set_title("Relative difference from Fourier (95% CI shaded for MC)")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = RESULTS_DIR / "heston_cross_validation.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out}")


def save_cross_validation_csv(rows):
    path = RESULTS_DIR / "heston_cross_validation.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "K", "C_Fourier",
            "C_MC_Euler", "MC_Euler_half_width", "rel_diff_Euler",
            "C_MC_QE", "MC_QE_half_width", "rel_diff_QE",
            "C_PDE", "rel_diff_PDE",
            "time_Euler_s", "time_QE_s", "time_PDE_s",
        ])
        for r in rows:
            fref = r["fourier"]
            w.writerow([
                f"{r['K']:.2f}",
                f"{r['fourier']:.8f}",
                f"{r['euler_price']:.8f}",
                f"{r['euler_hw']:.6e}",
                f"{(r['euler_price'] - fref) / fref:.6e}",
                f"{r['qe_price']:.8f}",
                f"{r['qe_hw']:.6e}",
                f"{(r['qe_price'] - fref) / fref:.6e}",
                f"{r['pde_price']:.8f}",
                f"{(r['pde_price'] - fref) / fref:.6e}",
                f"{r['time_euler_s']:.3f}",
                f"{r['time_qe_s']:.3f}",
                f"{r['time_pde_s']:.3f}",
            ])
    print(f"  Saved: {path}")


# ==========================================================================
# Section 2: calibration
# ==========================================================================

def synthesize_vol_surface(truth_params, maturities, strikes,
                              noise_std=IV_NOISE_STD, seed=RNG_SEED):
    """Generate a synthetic noisy Heston-friendly call surface."""
    rng = np.random.default_rng(seed)
    quotes = []
    for T_ in maturities:
        for K in strikes:
            C_true = heston_call_lewis(
                K, T_, S0, truth_params["v0"], R,
                truth_params["kappa"], truth_params["theta"],
                truth_params["sigma"], truth_params["rho"])
            iv_true = implied_vol_bs(C_true, K, T_, S0, R)
            if np.isnan(iv_true):
                continue
            iv_noisy = iv_true + noise_std * rng.standard_normal()
            iv_noisy = max(0.01, iv_noisy)  # numerical safety
            C_noisy = black_scholes_call(S0, K, T_, R, iv_noisy)
            quotes.append({"K": float(K), "T": float(T_),
                            "C_market": float(C_noisy)})
    return quotes


def calibration_section():
    """Calibrate Heston to a synthetic noisy surface."""
    print()
    print("=" * 102)
    print("  SECTION 2: Calibration to synthetic noisy vol surface")
    print(f"  Maturities: {MATURITIES_CALIB}")
    print(f"  Strikes:    {STRIKES_CALIB.tolist()}")
    print(f"  IV noise:   N(0, {IV_NOISE_STD}) added to clean Heston IVs")
    print("=" * 102)

    # Truth = canonical Heston
    truth = dict(HESTON)
    market = synthesize_vol_surface(truth, MATURITIES_CALIB, STRIKES_CALIB)
    print(f"\n  Generated {len(market)} synthetic quotes")

    # Initial guess: deliberately distant from truth
    initial = dict(kappa=1.0, theta=0.05, sigma=0.5, rho=-0.3, v0=0.05)
    print(f"  Truth params:  {truth}")
    print(f"  Initial guess: {initial}")

    print("\n  Calibrating (LM with vega weighting)...")
    t0 = time.perf_counter()
    result = calibrate_heston(market, S0, R, initial_guess=initial)
    elapsed = time.perf_counter() - t0
    print(f"  Elapsed: {elapsed:.2f}s, n_iter: {result.n_iter}, "
          f"success: {result.success}")
    print(f"  Calibrated:")
    for k in ["kappa", "theta", "sigma", "rho", "v0"]:
        diff = result.params[k] - truth[k]
        print(f"    {k:>6}: {result.params[k]:>10.6f}  "
              f"(truth {truth[k]:>10.6f}, err {diff:>+9.5f})")
    print(f"  RMSE (price): {result.rmse:.6e}")

    return market, result, truth


def plot_calibration(market, result, truth):
    """2x2 grid: one IV-vs-K subplot per maturity."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    p = result.params

    for ax, T_target in zip(axes, MATURITIES_CALIB):
        slice_quotes = [q for q in market if abs(q["T"] - T_target) < 1e-9]
        if not slice_quotes:
            continue
        K_mkt  = np.array([q["K"] for q in slice_quotes])
        iv_mkt = np.array([implied_vol_bs(q["C_market"], q["K"], q["T"],
                                            S0, R)
                            for q in slice_quotes])

        # Calibrated curve at finer grid
        K_fine   = np.linspace(STRIKES_CALIB[0], STRIKES_CALIB[-1], 50)
        iv_calib = []
        for K in K_fine:
            C_c = heston_call_lewis(K, T_target, S0, p["v0"], R,
                                      p["kappa"], p["theta"],
                                      p["sigma"], p["rho"])
            iv_calib.append(implied_vol_bs(C_c, K, T_target, S0, R))
        iv_calib = np.array(iv_calib)

        # RMSE in IV space at this maturity slice
        iv_at_K = np.array([
            implied_vol_bs(
                heston_call_lewis(K, T_target, S0, p["v0"], R,
                                    p["kappa"], p["theta"],
                                    p["sigma"], p["rho"]),
                K, T_target, S0, R)
            for K in K_mkt])
        valid = ~(np.isnan(iv_mkt) | np.isnan(iv_at_K))
        rmse  = (float(np.sqrt(np.mean((iv_at_K[valid] - iv_mkt[valid])**2)))
                  if valid.any() else float('nan'))

        ax.scatter(K_mkt, iv_mkt, s=60, color="C0",
                    edgecolors="black", linewidth=0.7,
                    label="market (noisy)", zorder=3)
        ax.plot(K_fine, iv_calib, color="C3", linewidth=2,
                 label="Heston calibrated", zorder=2)
        ax.set_xlabel("Strike K")
        ax.set_ylabel("Implied volatility")
        ax.set_title(f"T = {T_target},  RMSE(IV) = {rmse:.4f}")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Heston calibration on synthetic noisy vol surface\n"
        f"Calibrated: κ={p['kappa']:.3f}, θ={p['theta']:.4f}, "
        f"σ={p['sigma']:.3f}, ρ={p['rho']:.3f}, v0={p['v0']:.4f}     "
        f"Truth: κ={truth['kappa']}, θ={truth['theta']}, "
        f"σ={truth['sigma']}, ρ={truth['rho']}, v0={truth['v0']}",
        fontsize=11,
    )
    fig.tight_layout()
    out = RESULTS_DIR / "heston_calibration.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out}")


def save_calibration_csv(market, result, truth):
    path = RESULTS_DIR / "heston_calibration.csv"
    p = result.params
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# calibration_result"])
        w.writerow(["parameter", "truth", "calibrated", "error"])
        for k in ["kappa", "theta", "sigma", "rho", "v0"]:
            w.writerow([k, truth[k], p[k], p[k] - truth[k]])
        w.writerow(["rmse_price", "", result.rmse, ""])
        w.writerow(["n_iter", "", result.n_iter, ""])
        w.writerow(["success", "", result.success, ""])
        w.writerow([])
        w.writerow(["# market_quotes_and_model_fit"])
        w.writerow(["K", "T", "C_market", "iv_market",
                     "C_calibrated", "iv_calibrated", "iv_residual"])
        for q in market:
            iv_m = implied_vol_bs(q["C_market"], q["K"], q["T"], S0, R)
            C_c  = heston_call_lewis(q["K"], q["T"], S0, p["v0"], R,
                                       p["kappa"], p["theta"],
                                       p["sigma"], p["rho"])
            iv_c = implied_vol_bs(C_c, q["K"], q["T"], S0, R)
            w.writerow([
                q["K"], q["T"],
                f"{q['C_market']:.6f}",
                f"{iv_m:.6f}" if not np.isnan(iv_m) else "",
                f"{C_c:.6f}",
                f"{iv_c:.6f}" if not np.isnan(iv_c) else "",
                f"{iv_c - iv_m:.6e}"
                  if not (np.isnan(iv_m) or np.isnan(iv_c)) else "",
            ])
    print(f"  Saved: {path}")


# ==========================================================================
# Section 3: PDE convergence
# ==========================================================================

def convergence_section():
    """PDE error vs grid spacing log-log; fit slope to verify O(h^2)."""
    print()
    print("=" * 102)
    print("  SECTION 3: PDE 2D Douglas ADI spatial convergence")
    print(f"  ATM strike K={CONV_K}, T={T}")
    print("=" * 102)

    ref = heston_call_lewis(CONV_K, T, S0, HESTON["v0"], R,
                              HESTON["kappa"], HESTON["theta"],
                              HESTON["sigma"], HESTON["rho"])
    print(f"\n  Fourier reference: {ref:.8f}\n")
    print(f"  {'N_X':>5} {'N_v':>5} {'N_tau':>6}  {'h_X':>10}  "
          f"{'PDE price':>12}  {'|error|':>14}  {'time (s)':>10}")
    print("  " + "-" * 75)

    v_max = 5.0 * HESTON["theta"]
    half_width = 4.0 * np.sqrt(v_max * T)

    rows = []
    for (N_X, N_v, N_tau) in CONV_GRIDS:
        t0 = time.perf_counter()
        price = heston_call_pde(
            S0, CONV_K, T,
            HESTON["kappa"], HESTON["theta"],
            HESTON["sigma"], HESTON["rho"],
            HESTON["v0"], R,
            N_X=N_X, N_v=N_v, N_tau=N_tau)
        elapsed = time.perf_counter() - t0
        h_X = 2.0 * half_width / N_X
        err = abs(price - ref)
        rows.append({"N_X": N_X, "N_v": N_v, "N_tau": N_tau,
                      "h_X": h_X, "price": price, "error": err,
                      "time_s": elapsed})
        print(f"  {N_X:>5d} {N_v:>5d} {N_tau:>6d}  {h_X:>10.5f}  "
              f"{price:>12.6f}  {err:>14.6e}  {elapsed:>10.2f}")

    h_arr = np.array([r["h_X"]   for r in rows])
    e_arr = np.array([r["error"] for r in rows])
    valid = e_arr > 0
    slope, _ = np.polyfit(np.log(h_arr[valid]), np.log(e_arr[valid]), 1)
    print(f"\n  Fitted slope on log-log: {slope:.3f}  (theory: 2.000)")

    return rows, ref, float(slope)


def plot_convergence(rows, ref, slope):
    """Log-log plot of PDE error vs h with O(h^2) reference line."""
    h_arr = np.array([r["h_X"]   for r in rows])
    e_arr = np.array([r["error"] for r in rows])

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.loglog(h_arr, e_arr, "o-", linewidth=2.5, markersize=12, color="C2",
                label="Douglas ADI (observed)")

    # Reference slope = 2 anchored at finest grid
    h_ref = np.array([h_arr.min(), h_arr.max()])
    e_ref = e_arr[-1] * (h_ref / h_arr[-1]) ** 2
    ax.loglog(h_ref, e_ref, "k--", linewidth=1.5, alpha=0.6,
                label=r"$O(h^2)$ reference (slope = 2)")

    ax.set_xlabel(r"grid spacing $h_X$  (log scale)")
    ax.set_ylabel(r"$|C_{\mathrm{PDE}} - C_{\mathrm{Fourier}}|$  (log scale)")
    ax.set_title(
        f"Heston PDE 2D Douglas ADI: spatial convergence "
        f"(K={CONV_K}, T={T})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=11)
    ax.text(0.05, 0.05,
             f"fitted slope = {slope:.3f}\ntheoretical = 2.000",
             transform=ax.transAxes, fontsize=11,
             verticalalignment="bottom",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                        edgecolor="gray", alpha=0.9))

    fig.tight_layout()
    out = RESULTS_DIR / "pde_adi_convergence.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out}")


def save_convergence_csv(rows, ref, slope):
    path = RESULTS_DIR / "pde_adi_convergence.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# reference_fourier_price", f"{ref:.10f}"])
        w.writerow(["# fitted_loglog_slope",      f"{slope:.4f}"])
        w.writerow(["# theoretical_slope",        "2.0"])
        w.writerow([])
        w.writerow(["N_X", "N_v", "N_tau", "h_X",
                     "PDE_price", "abs_error", "time_s"])
        for r in rows:
            w.writerow([
                r["N_X"], r["N_v"], r["N_tau"],
                f"{r['h_X']:.6e}",
                f"{r['price']:.8f}",
                f"{r['error']:.6e}",
                f"{r['time_s']:.3f}",
            ])
    print(f"  Saved: {path}")


# ==========================================================================
# Section 4: exotics (optional)
# ==========================================================================

def exotics_section():
    """Price 4 exotic payoffs under canonical Heston."""
    print()
    print("=" * 102)
    print("  SECTION 4: Exotic options under Heston")
    print(f"  MC: n_paths={EXOTIC_N_PATHS}, n_steps={EXOTIC_N_STEPS}, "
          f"K={EXOTIC_K}, T={T}")
    print("=" * 102)

    results = {}

    print("\n  European call (Fourier reference)...")
    C_eur = heston_call_lewis(EXOTIC_K, T, S0, HESTON["v0"], R,
                                HESTON["kappa"], HESTON["theta"],
                                HESTON["sigma"], HESTON["rho"])
    results["European call\n(Fourier)"] = (C_eur, 0.0)
    print(f"    Price: {C_eur:.4f}")

    print("\n  Asian arithmetic call (MC QE, n_avg=50)...")
    t0 = time.perf_counter()
    r_asian = mc_asian_call_heston(
        S0, EXOTIC_K, HESTON["v0"], R,
        HESTON["kappa"], HESTON["theta"],
        HESTON["sigma"], HESTON["rho"],
        T, EXOTIC_N_STEPS, EXOTIC_N_PATHS, n_avg=50, seed=RNG_SEED)
    print(f"    Price: {r_asian.estimate:.4f} +/- {r_asian.half_width:.4f}, "
          f"{(time.perf_counter()-t0):.1f}s")
    results["Asian\narithmetic"] = (r_asian.estimate, r_asian.half_width)

    print("\n  Lookback floating-strike call (MC QE)...")
    t0 = time.perf_counter()
    r_lb = mc_lookback_call_heston(
        S0, HESTON["v0"], R,
        HESTON["kappa"], HESTON["theta"],
        HESTON["sigma"], HESTON["rho"],
        T, EXOTIC_N_STEPS, EXOTIC_N_PATHS, seed=RNG_SEED)
    print(f"    Price: {r_lb.estimate:.4f} +/- {r_lb.half_width:.4f}, "
          f"{(time.perf_counter()-t0):.1f}s")
    results["Lookback\nfloating"] = (r_lb.estimate, r_lb.half_width)

    print(f"\n  Barrier up-and-out call (MC QE, H={BARRIER_H})...")
    t0 = time.perf_counter()
    r_bar = mc_barrier_call_heston(
        S0, EXOTIC_K, BARRIER_H, HESTON["v0"], R,
        HESTON["kappa"], HESTON["theta"],
        HESTON["sigma"], HESTON["rho"],
        T, EXOTIC_N_STEPS, EXOTIC_N_PATHS, seed=RNG_SEED)
    print(f"    Price: {r_bar.estimate:.4f} +/- {r_bar.half_width:.4f}, "
          f"{(time.perf_counter()-t0):.1f}s")
    results[f"Barrier UO\n(H={BARRIER_H:.0f})"] = (r_bar.estimate,
                                                       r_bar.half_width)

    print(f"\n  American put (PDE Douglas ADI + projection)...")
    t0 = time.perf_counter()
    p_am = heston_american_put_pde(
        S0, EXOTIC_K, T,
        HESTON["kappa"], HESTON["theta"],
        HESTON["sigma"], HESTON["rho"],
        HESTON["v0"], R,
        N_X=PDE_N_X, N_v=PDE_N_V, N_tau=PDE_N_TAU)
    print(f"    Price: {p_am:.4f}, {(time.perf_counter()-t0):.1f}s")
    results["American put\n(PDE)"] = (p_am, 0.0)

    return results


def plot_exotics(results):
    """Bar chart of exotic prices with 95% CI for MC methods."""
    fig, ax = plt.subplots(figsize=(11, 6))

    names  = list(results.keys())
    values = [v[0] for v in results.values()]
    hws    = [v[1] for v in results.values()]

    colors = []
    for name in names:
        if "Fourier" in name or "PDE" in name:
            colors.append("C2")   # deterministic
        else:
            colors.append("C0")   # MC

    x = np.arange(len(names))
    bars = ax.bar(x, values, yerr=hws, capsize=6,
                    color=colors, edgecolor="black", linewidth=0.7,
                    alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Option price")
    ax.set_title(
        f"Heston exotic option prices  "
        f"(S0={S0}, K={EXOTIC_K}, T={T}, n_paths={EXOTIC_N_PATHS})\n"
        f"95% CI shown for MC methods; deterministic for Fourier/PDE")
    ax.grid(True, axis="y", alpha=0.3)

    ymax = max(v + h for v, h in zip(values, hws))
    for bar, v, hw in zip(bars, values, hws):
        height = bar.get_height()
        label = f"{v:.3f}±{hw:.3f}" if hw > 0 else f"{v:.3f}"
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.04 * ymax,
                 label, ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, ymax * 1.18)
    fig.tight_layout()
    out = RESULTS_DIR / "heston_exotics.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out}")


def save_exotics_csv(results):
    path = RESULTS_DIR / "heston_exotics.csv"
    method_map = {
        "European call\n(Fourier)":    "Fourier",
        "Asian\narithmetic":           "MC Andersen QE",
        "Lookback\nfloating":          "MC Andersen QE",
        f"Barrier UO\n(H={BARRIER_H:.0f})": "MC Andersen QE",
        "American put\n(PDE)":         "PDE Douglas ADI + projection",
    }
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["payoff_type", "price", "half_width_95", "method"])
        for name, (price, hw) in results.items():
            clean_name = name.replace("\n", " ")
            w.writerow([
                clean_name,
                f"{price:.6f}",
                f"{hw:.6e}" if hw > 0 else "",
                method_map.get(name, ""),
            ])
    print(f"  Saved: {path}")


# ==========================================================================
# Section 5: QE vs Euler bias (optional)
# ==========================================================================

def qe_vs_euler_section():
    """Compare QE bias vs Euler bias as function of n_steps.

    Uses aggressive Heston parameters (Feller=0.04) where the schemes
    differ significantly. At canonical params (Feller=1.78) both methods
    are similarly accurate and the comparison would not be informative.
    """
    h = QE_VS_EULER_HESTON
    feller_agg = 2 * h["kappa"] * h["theta"] / h["sigma"]**2
    print()
    print("=" * 102)
    print("  SECTION 5: Andersen QE vs full-truncation Euler bias")
    print(f"  Aggressive Heston: kappa={h['kappa']}, theta={h['theta']}, "
          f"sigma={h['sigma']}, rho={h['rho']}, v0={h['v0']}")
    print(f"  Feller parameter: {feller_agg:.3f}  "
          f"(canonical was 1.778; this is where QE was designed to dominate)")
    print(f"  M={QE_VS_EULER_M}, K={QE_VS_EULER_K}, T={T}, "
          f"{QE_VS_EULER_N_SEEDS} seeds averaged")
    print("=" * 102)

    K = QE_VS_EULER_K
    ref = heston_call_lewis(K, T, S0, h["v0"], R,
                              h["kappa"], h["theta"],
                              h["sigma"], h["rho"])
    print(f"\n  Fourier reference: {ref:.8f}\n")
    print(f"  {'N':>5}  {'Euler bias':>14}  {'Euler HW':>11}  "
          f"{'QE bias':>14}  {'QE HW':>11}")
    print("  " + "-" * 65)

    rows = []
    for N in QE_VS_EULER_N_STEPS:
        euler_est = []
        qe_est    = []
        euler_hws = []
        qe_hws    = []
        for seed in range(QE_VS_EULER_N_SEEDS):
            r_e = mc_european_call_heston(
                S0, K, h["v0"], R,
                h["kappa"], h["theta"],
                h["sigma"], h["rho"],
                T, N, QE_VS_EULER_M, seed=seed)
            r_q = mc_european_call_heston_qe(
                S0, K, h["v0"], R,
                h["kappa"], h["theta"],
                h["sigma"], h["rho"],
                T, N, QE_VS_EULER_M, seed=seed)
            euler_est.append(r_e.estimate)
            qe_est.append(r_q.estimate)
            euler_hws.append(r_e.half_width)
            qe_hws.append(r_q.half_width)

        euler_bias = abs(np.mean(euler_est) - ref)
        qe_bias    = abs(np.mean(qe_est)    - ref)
        euler_hw_avg = np.mean(euler_hws) / np.sqrt(QE_VS_EULER_N_SEEDS)
        qe_hw_avg    = np.mean(qe_hws)    / np.sqrt(QE_VS_EULER_N_SEEDS)

        rows.append({
            "N": N,
            "euler_bias": euler_bias, "euler_hw_avg": euler_hw_avg,
            "qe_bias": qe_bias,       "qe_hw_avg":    qe_hw_avg,
            "euler_est": euler_est,   "qe_est":       qe_est,
        })
        print(f"  {N:>5d}  {euler_bias:>14.6e}  {euler_hw_avg:>11.4e}  "
              f"{qe_bias:>14.6e}  {qe_hw_avg:>11.4e}")

    return rows, ref


def plot_qe_vs_euler(rows, ref):
    """Log-log bias vs N for both Euler and QE in aggressive regime.

    Drops error bars (rendering issue on log-scale when value < HW) and
    instead shows a horizontal noise floor band so the reader can judge
    which bias values are statistically significant.
    """
    Ns          = np.array([r["N"]            for r in rows])
    euler_bias  = np.array([r["euler_bias"]   for r in rows])
    qe_bias     = np.array([r["qe_bias"]      for r in rows])
    euler_hw_av = np.array([r["euler_hw_avg"] for r in rows])
    qe_hw_av    = np.array([r["qe_hw_avg"]    for r in rows])

    # Average HW across all points (Euler and QE) gives a representative
    # noise floor.
    noise_floor = float(np.mean(np.concatenate([euler_hw_av, qe_hw_av])))

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.loglog(Ns, euler_bias, "o-",
                linewidth=2.5, markersize=12, color="C0",
                label="Full-truncation Euler")
    ax.loglog(Ns, qe_bias, "s-",
                linewidth=2.5, markersize=12, color="C3",
                label="Andersen QE")

    # Noise floor: bias values below this line are not statistically
    # significant (within one half-width of zero).
    ax.axhline(noise_floor, color="gray", linestyle=":", linewidth=1.5,
                 alpha=0.8,
                 label=f"Noise floor "
                         f"(avg 95% half-width = {noise_floor:.3f})")

    # Reference slope -1 for Euler weak rate
    N_ref = np.array([Ns.min(), Ns.max()])
    e_ref = euler_bias[0] * (Ns[0] / N_ref)
    ax.loglog(N_ref, e_ref, "k--", linewidth=1.0, alpha=0.5,
                label=r"$O(1/N)$ (Euler weak rate reference)")

    ax.set_xlabel(r"$N$  (number of time steps)")
    ax.set_ylabel(r"$|bias| = |\bar{C}_{\mathrm{MC}} "
                    r"- C_{\mathrm{Fourier}}|$")
    h = QE_VS_EULER_HESTON
    feller_agg = 2 * h["kappa"] * h["theta"] / h["sigma"]**2
    ax.set_title(
        f"Heston MC: bias vs $N$ in aggressive regime "
        f"(Feller = {feller_agg:.2f})\n"
        f"κ={h['kappa']}, θ={h['theta']}, σ={h['sigma']}, "
        f"ρ={h['rho']}, v0={h['v0']}     "
        f"(M={QE_VS_EULER_M}, {QE_VS_EULER_N_SEEDS} seeds avg, "
        f"K={int(QE_VS_EULER_K)}, T={T})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

    fig.tight_layout()
    out = RESULTS_DIR / "heston_qe_vs_euler.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out}")


def save_qe_vs_euler_csv(rows, ref):
    path = RESULTS_DIR / "heston_qe_vs_euler.csv"
    h = QE_VS_EULER_HESTON
    feller_agg = 2 * h["kappa"] * h["theta"] / h["sigma"]**2
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# regime", "aggressive"])
        w.writerow(["# kappa", h["kappa"]])
        w.writerow(["# theta", h["theta"]])
        w.writerow(["# sigma", h["sigma"]])
        w.writerow(["# rho",   h["rho"]])
        w.writerow(["# v0",    h["v0"]])
        w.writerow(["# feller", f"{feller_agg:.4f}"])
        w.writerow(["# reference_fourier", f"{ref:.10f}"])
        w.writerow(["# M", QE_VS_EULER_M])
        w.writerow(["# n_seeds", QE_VS_EULER_N_SEEDS])
        w.writerow([])
        w.writerow(["N",
                     "euler_bias_abs", "euler_hw_avg",
                     "qe_bias_abs",    "qe_hw_avg",
                     "euler_estimates_per_seed",
                     "qe_estimates_per_seed"])
        for r in rows:
            w.writerow([
                r["N"],
                f"{r['euler_bias']:.6e}",
                f"{r['euler_hw_avg']:.6e}",
                f"{r['qe_bias']:.6e}",
                f"{r['qe_hw_avg']:.6e}",
                "|".join(f"{x:.6f}" for x in r["euler_est"]),
                "|".join(f"{x:.6f}" for x in r["qe_est"]),
            ])
    print(f"  Saved: {path}")


# ==========================================================================
# Main
# ==========================================================================

def main():
    t_start = time.perf_counter()
    print(f"Phase 4 Block 7 - Final benchmark")
    print(f"Output directory: {RESULTS_DIR}")

    # Section 1: cross-validation
    rows_cross = cross_validation_section()
    save_cross_validation_csv(rows_cross)
    plot_cross_validation(rows_cross)

    # Section 2: calibration
    market, calib_result, truth = calibration_section()
    save_calibration_csv(market, calib_result, truth)
    plot_calibration(market, calib_result, truth)

    # Section 3: PDE convergence
    rows_conv, ref_conv, slope = convergence_section()
    save_convergence_csv(rows_conv, ref_conv, slope)
    plot_convergence(rows_conv, ref_conv, slope)

    # Section 4: exotics
    results_exotics = exotics_section()
    save_exotics_csv(results_exotics)
    plot_exotics(results_exotics)

    # Section 5: QE vs Euler bias
    rows_qe_eul, ref_qe = qe_vs_euler_section()
    save_qe_vs_euler_csv(rows_qe_eul, ref_qe)
    plot_qe_vs_euler(rows_qe_eul, ref_qe)

    # Summary
    elapsed = time.perf_counter() - t_start
    print()
    print("=" * 102)
    print(f"  Benchmark complete in {elapsed:.1f} s ({elapsed/60:.1f} min)")
    print(f"  Outputs in: {RESULTS_DIR}")
    print("=" * 102)
    for f in sorted(RESULTS_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:50s} {size_kb:>8.1f} KB")


if __name__ == "__main__":
    main()
