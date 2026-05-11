"""
Phase 1 benchmark: reproducible visual artifacts for the Black-Scholes
pricer, the Greeks, and the implied-volatility solver.

Run from the repository root:
    python python/benchmark_phase1.py

Outputs land in python/results/phase1/:
  - greeks_panel.png + greeks_panel.csv
  - put_call_parity.png + put_call_parity.csv
  - iv_recovery.png + iv_recovery.csv

Each artifact targets one structural property of the algorithms:
  1. greeks_panel: visual sanity check that the five Greeks behave as
     theory predicts across moneyness.
  2. put_call_parity: demonstration that C - P = S - K*exp(-rT) holds at
     machine precision by algebraic construction, not by tuned tolerance.
  3. iv_recovery: round-trip test of the IV solver, showing convergence
     to ~1e-8 across a representative volatility range.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for reproducibility on any host.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantlib.black_scholes import (
    call_price, put_price,
    call_delta, put_delta, gamma, vega,
    call_theta, put_theta, call_rho, put_rho,
)
from quantlib.implied_volatility import implied_volatility


# ---------------------------------------------------------------------------
# Output configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "phase1"
DPI = 120
SAVEFIG_KW = dict(dpi=DPI, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Artifact 1: Greeks panel
# ---------------------------------------------------------------------------
def build_greeks_panel(verbose: bool = True):
    """
    A 2x3 grid of the five Greeks as functions of spot S, with the sixth
    sub-panel reserved for a description of the experiment parameters.
    Call and put are overlaid for Delta, Theta, and Rho.
    """
    if verbose:
        print("[1/3] Building Greeks panel...")

    K, T, r, sigma = 100.0, 1.0, 0.05, 0.20
    S = np.linspace(0.5 * K, 1.5 * K, 200)

    dc = call_delta(S, K, r, sigma, T)
    dp = put_delta (S, K, r, sigma, T)
    g  = gamma     (S, K, r, sigma, T)
    v  = vega      (S, K, r, sigma, T)
    tc = call_theta(S, K, r, sigma, T)
    tp = put_theta (S, K, r, sigma, T)
    rc = call_rho  (S, K, r, sigma, T)
    rp = put_rho   (S, K, r, sigma, T)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    ax_delta, ax_gamma, ax_vega   = axes[0]
    ax_theta, ax_rho,   ax_desc   = axes[1]

    ax_delta.plot(S, dc, label="Call", lw=1.6)
    ax_delta.plot(S, dp, label="Put",  lw=1.6, linestyle="--")
    ax_delta.axvline(K, color="grey", lw=0.8, alpha=0.5)
    ax_delta.axhline(0, color="grey", lw=0.5, alpha=0.5)
    ax_delta.set_xlabel("S"); ax_delta.set_ylabel(r"$\Delta$")
    ax_delta.set_title("Delta")
    ax_delta.legend(loc="best", frameon=False)
    ax_delta.grid(True, alpha=0.3)

    ax_gamma.plot(S, g, lw=1.6, color="C2")
    ax_gamma.axvline(K, color="grey", lw=0.8, alpha=0.5)
    ax_gamma.set_xlabel("S"); ax_gamma.set_ylabel(r"$\Gamma$")
    ax_gamma.set_title("Gamma (call = put)")
    ax_gamma.grid(True, alpha=0.3)

    ax_vega.plot(S, v, lw=1.6, color="C3")
    ax_vega.axvline(K, color="grey", lw=0.8, alpha=0.5)
    ax_vega.set_xlabel("S"); ax_vega.set_ylabel(r"$\mathcal{V}$")
    ax_vega.set_title("Vega (call = put)")
    ax_vega.grid(True, alpha=0.3)

    ax_theta.plot(S, tc, label="Call", lw=1.6)
    ax_theta.plot(S, tp, label="Put",  lw=1.6, linestyle="--")
    ax_theta.axvline(K, color="grey", lw=0.8, alpha=0.5)
    ax_theta.axhline(0, color="grey", lw=0.5, alpha=0.5)
    ax_theta.set_xlabel("S"); ax_theta.set_ylabel(r"$\Theta$ (per year)")
    ax_theta.set_title("Theta")
    ax_theta.legend(loc="best", frameon=False)
    ax_theta.grid(True, alpha=0.3)

    ax_rho.plot(S, rc, label="Call", lw=1.6)
    ax_rho.plot(S, rp, label="Put",  lw=1.6, linestyle="--")
    ax_rho.axvline(K, color="grey", lw=0.8, alpha=0.5)
    ax_rho.axhline(0, color="grey", lw=0.5, alpha=0.5)
    ax_rho.set_xlabel("S"); ax_rho.set_ylabel(r"$\rho$")
    ax_rho.set_title("Rho")
    ax_rho.legend(loc="best", frameon=False)
    ax_rho.grid(True, alpha=0.3)

    ax_desc.axis("off")
    desc_text = (
        "Experiment\n"
        "----------\n"
        f"Strike:        K = {K:.0f}\n"
        f"Maturity:      T = {T:.1f} year\n"
        f"Rate:          r = {r:.2%}\n"
        f"Volatility:    $\\sigma$ = {sigma:.2%}\n"
        f"Spot range:    S $\\in$ [{S.min():.0f}, {S.max():.0f}]\n"
        f"Grid points:   {len(S)}\n\n"
        "Theta convention: $\\partial C/\\partial t$ (calendar time)\n"
        "Greek values per unit input change\n"
        "(divide $\\Theta$ by 365 for per-day)."
    )
    ax_desc.text(
        0.02, 0.98, desc_text,
        transform=ax_desc.transAxes,
        family="monospace", fontsize=9.5,
        verticalalignment="top",
    )

    fig.suptitle("Black-Scholes Greeks across moneyness", fontsize=14, y=1.00)
    fig.tight_layout()

    df = pd.DataFrame({
        "S": S,
        "delta_call": dc, "delta_put": dp,
        "gamma": g,
        "vega": v,
        "theta_call": tc, "theta_put": tp,
        "rho_call": rc, "rho_put": rp,
    })

    fig.savefig(RESULTS_DIR / "greeks_panel.png", **SAVEFIG_KW)
    df.to_csv(RESULTS_DIR / "greeks_panel.csv", index=False)

    if verbose:
        print(f"      -> greeks_panel.png ({len(S)} points)")
        print(f"      -> greeks_panel.csv")
    return fig


# ---------------------------------------------------------------------------
# Artifact 2: Put-call parity
# ---------------------------------------------------------------------------
def build_put_call_parity(seed: int, verbose: bool = True):
    """
    Demonstrate that C - P = S - K*exp(-rT) holds at machine precision.

    Left: scatter of (C - P) vs (S - K*exp(-rT)) over 200 random tuples;
          all points must lie on y = x.
    Right: log10 histogram of nonzero residuals; expect a peak around -15.
           Cases where the residual is exactly zero (algebraic cancellation
           down to the last bit) are reported in the title.
    """
    if verbose:
        print("[2/3] Building put-call parity figure...")

    rng = np.random.default_rng(seed)
    N = 200
    S     = rng.uniform( 50.0, 150.0, N)
    K     = rng.uniform( 50.0, 150.0, N)
    r     = rng.uniform(  0.01, 0.10, N)
    sigma = rng.uniform(  0.10, 0.50, N)
    T     = rng.uniform(  0.10, 2.00, N)

    C = call_price(S, K, r, sigma, T)
    P = put_price (S, K, r, sigma, T)

    lhs = C - P
    rhs = S - K * np.exp(-r * T)
    residual = lhs - rhs
    abs_res = np.abs(residual)

    # Separate the cases of exact algebraic cancellation from the FP-noise
    # cases. The histogram shows only the latter to keep the scale readable;
    # the count of exact cancellations is reported in the title.
    nonzero_mask = abs_res > 0
    n_exact_zero = int((~nonzero_mask).sum())
    log_abs_res = np.log10(abs_res[nonzero_mask])

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5))

    ax_l.scatter(rhs, lhs, s=18, alpha=0.7, color="C0", edgecolor="none")
    lo, hi = min(rhs.min(), lhs.min()), max(rhs.max(), lhs.max())
    pad = 0.05 * (hi - lo)
    line = np.linspace(lo - pad, hi + pad, 100)
    ax_l.plot(line, line, color="grey", lw=1.0, linestyle="--", label="y = x")
    ax_l.set_xlabel(r"$S - K e^{-rT}$")
    ax_l.set_ylabel(r"$C - P$")
    ax_l.set_title("Put-call parity (algebraic identity)")
    ax_l.legend(loc="best", frameon=False)
    ax_l.grid(True, alpha=0.3)

    ax_r.hist(log_abs_res, bins=20, color="C1", edgecolor="white", linewidth=0.8)
    ax_r.set_xlabel(r"$\log_{10}|\text{residual}|$")
    ax_r.set_ylabel("Frequency")
    ax_r.set_title(
        f"Residual magnitudes  (N = {N}, "
        f"{n_exact_zero} cases at exact zero excluded)"
    )
    ax_r.axvline(-15, color="grey", lw=0.8, linestyle="--",
                 label="machine precision (-15)")
    ax_r.legend(loc="best", frameon=False)
    ax_r.grid(True, alpha=0.3)

    fig.suptitle(
        "$C - P = S - K e^{-rT}$  holds at machine precision by algebraic "
        "construction",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    df = pd.DataFrame({
        "S": S, "K": K, "T": T, "r": r, "sigma": sigma,
        "C": C, "P": P,
        "lhs_C_minus_P": lhs,
        "rhs_S_minus_PV_K": rhs,
        "residual": residual,
    })

    fig.savefig(RESULTS_DIR / "put_call_parity.png", **SAVEFIG_KW)
    df.to_csv(RESULTS_DIR / "put_call_parity.csv", index=False)

    if verbose:
        max_res = float(abs_res.max())
        print(f"      -> put_call_parity.png (N = {N})")
        print(f"      -> put_call_parity.csv")
        print(f"      max |residual| = {max_res:.2e}, "
              f"{n_exact_zero} exact-zero cases")
    return fig


# ---------------------------------------------------------------------------
# Artifact 3: Implied volatility round-trip
# ---------------------------------------------------------------------------
def build_iv_recovery(verbose: bool = True):
    """
    Round-trip test: compute the BS price for a grid of sigma_true, feed it
    into the IV solver, recover sigma_recovered, plot recovery vs truth.

    Left: scatter sigma_recovered vs sigma_true over y = x.
    Right: log10 histogram of nonzero absolute errors. Cases where the
           recovery is bit-exact are reported in the title.
    """
    if verbose:
        print("[3/3] Building implied volatility recovery figure...")

    S, K, r, T = 100.0, 100.0, 0.05, 1.0
    sigma_true = np.linspace(0.05, 0.60, 40)

    sigma_rec = np.empty_like(sigma_true)
    for i, s in enumerate(sigma_true):
        C = call_price(S, K, r, s, T)
        sigma_rec[i] = implied_volatility(C, S, K, r, T)

    err = np.abs(sigma_rec - sigma_true)
    nonzero_mask = err > 0
    n_exact_zero = int((~nonzero_mask).sum())
    log_abs_err = np.log10(err[nonzero_mask])

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5))

    ax_l.scatter(sigma_true, sigma_rec, s=24, alpha=0.8, color="C0",
                 edgecolor="none", label="Recovered")
    lo, hi = sigma_true.min(), sigma_true.max()
    pad = 0.02 * (hi - lo)
    line = np.linspace(lo - pad, hi + pad, 100)
    ax_l.plot(line, line, color="grey", lw=1.0, linestyle="--", label="y = x")
    ax_l.set_xlabel(r"$\sigma_{\mathrm{true}}$")
    ax_l.set_ylabel(r"$\sigma_{\mathrm{recovered}}$")
    ax_l.set_title("Implied volatility recovery")
    ax_l.legend(loc="best", frameon=False)
    ax_l.grid(True, alpha=0.3)

    ax_r.hist(log_abs_err, bins=15, color="C1", edgecolor="white", linewidth=0.8)
    ax_r.set_xlabel(
        r"$\log_{10}|\sigma_{\mathrm{recovered}} - \sigma_{\mathrm{true}}|$"
    )
    ax_r.set_ylabel("Frequency")
    ax_r.set_title(
        f"Recovery error  (N = {len(sigma_true)}, "
        f"{n_exact_zero} cases at exact zero excluded)"
    )
    ax_r.axvline(-8, color="grey", lw=0.8, linestyle="--",
                 label="solver tolerance ($10^{-8}$)")
    ax_r.legend(loc="best", frameon=False)
    ax_r.grid(True, alpha=0.3)

    fig.suptitle(
        f"IV round-trip on grid $\\sigma \\in [{sigma_true.min():.2f}, "
        f"{sigma_true.max():.2f}]$  (S=K={K:.0f}, r={r:.2%}, T={T:.1f})",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    df = pd.DataFrame({
        "sigma_true": sigma_true,
        "sigma_recovered": sigma_rec,
        "abs_error": err,
    })

    fig.savefig(RESULTS_DIR / "iv_recovery.png", **SAVEFIG_KW)
    df.to_csv(RESULTS_DIR / "iv_recovery.csv", index=False)

    if verbose:
        print(f"      -> iv_recovery.png ({len(sigma_true)} points)")
        print(f"      -> iv_recovery.csv")
        print(f"      max abs error = {err.max():.2e}, "
              f"{n_exact_zero} exact-zero cases")
    return fig


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate reproducible visual artifacts for Phase 1.")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for the random tuples in the put-call parity figure "
             "(default: 42).")
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-step output.")
    args = parser.parse_args()

    verbose = not args.quiet
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Writing artifacts to {RESULTS_DIR}")
        print()

    build_greeks_panel(verbose=verbose)
    build_put_call_parity(seed=args.seed, verbose=verbose)
    build_iv_recovery(verbose=verbose)

    if verbose:
        print()
        print("Done.")


if __name__ == "__main__":
    main()
