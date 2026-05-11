"""
Phase 2 Block 6 - Final benchmark.

Orchestrates the variance-reduction techniques, SDE schemes, and
QMC samplers of Phase 2 to produce reproducible visual artefacts
of the headline results.

Outputs:
  - vr_scoreboard.png + vr_scoreboard.csv
        Variance-reduction factor of every VR technique vs an IID
        baseline. Horizontal bar chart on log scale; the geometric
        Asian CV stands ~3 orders of magnitude above the rest.

  - euler_milstein_convergence.png + euler_milstein_convergence.csv
        Strong-error scaling of the Euler and Milstein schemes as
        Delta t -> 0, using a Brownian-coupled sampler so that the
        L^2 error |S_T^scheme - S_T^exact| is well-defined pathwise.
        Reference slopes 1/2 (Euler) and 1 (Milstein).

  - qmc_vs_mc.png + qmc_vs_mc.csv
        Error-vs-N scaling of IID MC and Sobol RQMC for the European
        call, both measured by RMS error across multiple seeds /
        shifts. IID slope ~-1/2; RQMC sits below IID by a constant
        factor at all N tested, but its slope does not reach the
        asymptotic -1 because the (log N)^d prefactor of Sobol
        remains significant in our d=20 setting.

Run from the python/ directory:

    python benchmark_phase2.py

Total runtime ~3-4 minutes at the production sizes set below.
"""

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# quantlib imports
# --------------------------------------------------------------------------
from quantlib.black_scholes import call_price as bs_call
from quantlib.monte_carlo import mc_european_call_exact
from quantlib.variance_reduction import (
    mc_european_call_exact_av,
    mc_european_call_exact_cv_underlying,
    mc_european_call_exact_cv_aon,
)
from quantlib.qmc import mc_european_call_euler_rqmc
from quantlib.asian import (
    mc_asian_call_arithmetic_iid,
    mc_asian_call_arithmetic_cv,
)


# ==========================================================================
# Configuration
# ==========================================================================
CANONICAL = dict(S=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0)
RNG_SEED  = 42

# --- Artefact 1: VR scoreboard ---
# We target ~1_000_000 total payoffs in every technique for apples-to-apples
# variance ratios. The exact n passed to each pricer is chosen so that the
# total payoff count is fixed; see _measure_vrf_* below for the bookkeeping.
# 2**20 = 1_048_576 is picked (slightly above 1M) so that 1_048_576 / R is
# a power of 2 for R = 16 -- this keeps Sobol RQMC in its balance-friendly
# regime and avoids the UserWarning from scipy's Sobol.
N_VR_TARGET_PAYOFFS = 2**20  # 1_048_576

# RQMC: R replications, each of N_RQMC_PATHS paths. R * N_RQMC_PATHS should
# equal N_VR_TARGET_PAYOFFS so the comparison is by total Brownian draws.
RQMC_REPLICATIONS = 16
RQMC_PATHS_PER_REP = N_VR_TARGET_PAYOFFS // RQMC_REPLICATIONS  # 65_536 = 2^16

# --- Artefact 2: Euler/Milstein strong convergence ---
N_PATHS_STRONG = 50_000
DT_INVS_STRONG = [16, 32, 64, 128, 256, 512, 1024]   # n_steps = 1/dt

# --- Artefact 3: QMC vs MC error scaling ---
# All N are powers of 2, starting at 2^11 = 2048 to skip the warm-up
# regime where Sobol's balance properties are not yet active.
# N_RQMC_REPLICATIONS = 16 divides every N to give a power-of-2 n_per.
N_LIST_QMC = [2**k for k in range(11, 17)]           # 2048 .. 65536
N_SEEDS_QMC = 30                                     # replications for RMS
N_RQMC_REPLICATIONS = 16                             # internal RQMC R

# Output directory
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "phase2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================================
# Section utilities
# ==========================================================================
@dataclass
class VRRecord:
    name: str
    family: str         # "iid", "vr_plain", "qmc", "structural"
    underlying: str     # "european_call" or "asian_arithmetic"
    n_paths_arg: int    # n passed to the pricer
    total_payoffs: int  # bookkeeping for the report
    estimate: float
    half_width: float
    vrf: float          # relative to IID baseline on the same underlying


def _print_header(title: str) -> None:
    print()
    print("=" * 100)
    print(f"  {title}")
    print("=" * 100)


# ==========================================================================
# Section 1: Variance-reduction scoreboard
# ==========================================================================
def _measure_iid_baseline_european(seed: int) -> VRRecord:
    """IID baseline on the European call: n payoffs."""
    n = N_VR_TARGET_PAYOFFS
    t0 = time.perf_counter()
    res = mc_european_call_exact(**CANONICAL, n_paths=n, seed=seed)
    dt = time.perf_counter() - t0
    print(f"    IID baseline (European)        n={n:>8d}  "
          f"hw={res.half_width:.4e}  t={dt:.2f}s")
    return VRRecord("IID baseline", "iid", "european_call",
                    n, n, res.estimate, res.half_width, vrf=1.0)


def _measure_antithetic(baseline_hw: float, seed: int) -> VRRecord:
    """Antithetic variates: n pairs => 2*n payoffs. Set n = TARGET / 2 so
    total payoff count matches the baseline.
    """
    n_pairs = N_VR_TARGET_PAYOFFS // 2
    t0 = time.perf_counter()
    res = mc_european_call_exact_av(**CANONICAL, n_paths=n_pairs, seed=seed)
    dt = time.perf_counter() - t0
    vrf = (baseline_hw / res.half_width) ** 2
    total = 2 * n_pairs
    print(f"    Antithetic                     n_pairs={n_pairs:>7d}  "
          f"hw={res.half_width:.4e}  VRF={vrf:.2f}  t={dt:.2f}s")
    return VRRecord("Antithetic variates", "vr_plain", "european_call",
                    n_pairs, total, res.estimate, res.half_width, vrf)


def _measure_cv_underlying(baseline_hw: float, seed: int) -> VRRecord:
    n = N_VR_TARGET_PAYOFFS
    t0 = time.perf_counter()
    res = mc_european_call_exact_cv_underlying(**CANONICAL, n_paths=n,
                                                 seed=seed)
    dt = time.perf_counter() - t0
    vrf = (baseline_hw / res.half_width) ** 2
    print(f"    CV underlying                  n={n:>8d}  "
          f"hw={res.half_width:.4e}  VRF={vrf:.2f}  t={dt:.2f}s")
    return VRRecord("CV (underlying)", "vr_plain", "european_call",
                    n, n, res.estimate, res.half_width, vrf)


def _measure_cv_aon(baseline_hw: float, seed: int) -> VRRecord:
    n = N_VR_TARGET_PAYOFFS
    t0 = time.perf_counter()
    res = mc_european_call_exact_cv_aon(**CANONICAL, n_paths=n, seed=seed)
    dt = time.perf_counter() - t0
    vrf = (baseline_hw / res.half_width) ** 2
    print(f"    CV asset-or-nothing            n={n:>8d}  "
          f"hw={res.half_width:.4e}  VRF={vrf:.2f}  t={dt:.2f}s")
    return VRRecord("CV (asset-or-nothing)", "vr_plain", "european_call",
                    n, n, res.estimate, res.half_width, vrf)


def _measure_rqmc(baseline_hw: float, seed: int) -> VRRecord:
    """RQMC: R replications x N paths/rep = total payoffs."""
    R = RQMC_REPLICATIONS
    n_per = RQMC_PATHS_PER_REP
    total = R * n_per
    t0 = time.perf_counter()
    res = mc_european_call_euler_rqmc(
        **CANONICAL, n_paths=n_per, n_steps=20,
        n_replications=R, seed=seed)
    dt = time.perf_counter() - t0
    vrf = (baseline_hw / res.half_width) ** 2
    print(f"    Sobol RQMC                     R={R}, n_per={n_per:>6d}  "
          f"(total={total})  hw={res.half_width:.4e}  VRF={vrf:.2f}  "
          f"t={dt:.2f}s")
    return VRRecord("Sobol RQMC", "qmc", "european_call",
                    n_per, total, res.estimate, res.half_width, vrf)


def _measure_iid_baseline_asian(seed: int) -> Tuple[VRRecord, float]:
    """IID baseline on the Asian arithmetic call. Returns (record, hw)."""
    n = N_VR_TARGET_PAYOFFS
    t0 = time.perf_counter()
    res = mc_asian_call_arithmetic_iid(**CANONICAL, n_paths=n,
                                        n_steps=50, seed=seed)
    dt = time.perf_counter() - t0
    print(f"    IID baseline (Asian)           n={n:>8d}  "
          f"hw={res.half_width:.4e}  t={dt:.2f}s")
    rec = VRRecord("IID baseline (Asian)", "iid", "asian_arithmetic",
                   n, n, res.estimate, res.half_width, vrf=1.0)
    return rec, res.half_width


def _measure_cv_geometric_asian(baseline_hw: float, seed: int) -> VRRecord:
    n = N_VR_TARGET_PAYOFFS
    t0 = time.perf_counter()
    res = mc_asian_call_arithmetic_cv(**CANONICAL, n_paths=n,
                                        n_steps=50, seed=seed)
    dt = time.perf_counter() - t0
    vrf = (baseline_hw / res.half_width) ** 2
    print(f"    CV geometric Asian             n={n:>8d}  "
          f"hw={res.half_width:.4e}  VRF={vrf:.2f}  t={dt:.2f}s")
    return VRRecord("CV (geometric Asian)", "structural", "asian_arithmetic",
                    n, n, res.estimate, res.half_width, vrf)


def run_vr_scoreboard() -> List[VRRecord]:
    _print_header("ARTEFACT 1: Variance-reduction scoreboard")
    print("  Target total payoff count per technique: "
          f"{N_VR_TARGET_PAYOFFS:_}")
    print()
    print("  European-call techniques:")
    iid_eu = _measure_iid_baseline_european(RNG_SEED)
    ant    = _measure_antithetic(iid_eu.half_width, RNG_SEED + 1)
    cvu    = _measure_cv_underlying(iid_eu.half_width, RNG_SEED + 2)
    cva    = _measure_cv_aon(iid_eu.half_width, RNG_SEED + 3)
    rqmc   = _measure_rqmc(iid_eu.half_width, RNG_SEED + 4)

    print()
    print("  Asian-call techniques (separate baseline, different "
          "underlying):")
    iid_as, hw_as = _measure_iid_baseline_asian(RNG_SEED + 5)
    cvga = _measure_cv_geometric_asian(hw_as, RNG_SEED + 6)

    return [iid_eu, ant, cvu, cva, rqmc, iid_as, cvga]


def save_vr_scoreboard_csv(records: List[VRRecord]) -> None:
    path = RESULTS_DIR / "vr_scoreboard.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["technique", "family", "underlying",
                    "n_paths_arg", "total_payoffs",
                    "estimate", "half_width", "vrf"])
        for r in records:
            w.writerow([r.name, r.family, r.underlying,
                        r.n_paths_arg, r.total_payoffs,
                        f"{r.estimate:.6f}", f"{r.half_width:.6e}",
                        f"{r.vrf:.4f}"])
    print(f"\n  Saved: {path}")


def plot_vr_scoreboard(records: List[VRRecord]) -> None:
    # Exclude baselines from the plot (their VRF = 1 by construction).
    # Keep them implicitly via the gridline at x=1.
    bars = [r for r in records if r.family != "iid"]
    # Order: ascending VRF, so the geometric Asian sits at the top
    # visually emphasising the result.
    bars = sorted(bars, key=lambda r: r.vrf)

    color_map = {
        "vr_plain":   "#4C72B0",  # blue
        "qmc":        "#55A868",  # green
        "structural": "#C44E52",  # red
    }

    names  = [r.name for r in bars]
    vrfs   = [r.vrf for r in bars]
    colors = [color_map[r.family] for r in bars]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, vrfs, color=colors, edgecolor="black",
            linewidth=0.5, alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel("Variance reduction factor "
                  r"$\mathrm{VRF} = (\mathrm{hw}_{\mathrm{IID}}"
                  r"/ \mathrm{hw}_{\mathrm{technique}})^2$",
                  fontsize=11)
    ax.set_title("Phase 2: variance reduction scoreboard",
                 fontsize=12)
    ax.grid(True, axis="x", which="both", alpha=0.3)
    ax.axvline(1.0, color="black", linewidth=0.8, alpha=0.5)
    ax.text(1.05, -0.65, "IID baseline (= 1)",
            fontsize=8, color="black", alpha=0.6,
            ha="left", va="top")

    # Numeric annotation at the end of each bar.
    for i, v in enumerate(vrfs):
        ax.text(v * 1.1, i, f"{v:.1f}\u00D7", va="center",
                fontsize=9, color="black")

    # Special annotation for the geometric Asian.
    geo_idx = next((i for i, r in enumerate(bars)
                    if r.name == "CV (geometric Asian)"), None)
    if geo_idx is not None:
        v = vrfs[geo_idx]
        ax.annotate(r"thanks to AM-GM: $\rho > 0.999$",
                    xy=(v, geo_idx), xytext=(v * 0.04, geo_idx + 0.45),
                    fontsize=9, color="#C44E52",
                    arrowprops=dict(arrowstyle="->", color="#C44E52",
                                    lw=0.8))

    # Legend by family.
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_map["vr_plain"],
                      label="Plain variance reduction"),
        plt.Rectangle((0, 0), 1, 1, color=color_map["qmc"],
                      label="Quasi-Monte Carlo"),
        plt.Rectangle((0, 0), 1, 1, color=color_map["structural"],
                      label="Structural (problem-specific)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

    # Caveat footnote on comparability.
    fig.text(0.5, -0.02,
             "VRF normalised by total payoff count. Geometric Asian CV is "
             "measured on the Asian arithmetic call,\nnot the European "
             "call; other techniques on the European call. Same canonical "
             "params (S=K=100, r=0.05, $\\sigma$=0.20, T=1).",
             ha="center", fontsize=8, color="gray", style="italic")

    fig.tight_layout()
    out = RESULTS_DIR / "vr_scoreboard.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ==========================================================================
# Section 2: Euler / Milstein strong convergence
# ==========================================================================
def _coupled_paths(S0: float, r: float, sigma: float, T: float,
                   n_steps: int, n_paths: int,
                   rng: np.random.Generator
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three coupled samplers sharing the same Brownian increments.

    Returns (S_T_euler, S_T_milstein, S_T_exact) of length n_paths.
    All three are deterministic functions of the SAME dW_1, ..., dW_n,
    which is what makes the strong error pathwise meaningful.
    """
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    # Brownian increments: shape (n_paths, n_steps).
    dW = rng.standard_normal(size=(n_paths, n_steps)) * sqrt_dt

    # --- Exact: S_T = S0 * exp((r - sigma^2/2) T + sigma * sum(dW)) ---
    W_T = dW.sum(axis=1)
    S_exact = S0 * np.exp((r - 0.5 * sigma * sigma) * T + sigma * W_T)

    # --- Euler: multiplicative recursion ---
    S_eu = np.full(n_paths, S0)
    for k in range(n_steps):
        S_eu = S_eu * (1.0 + r * dt + sigma * dW[:, k])

    # --- Milstein: additional 0.5 * sigma^2 * (dW^2 - dt) term ---
    S_mi = np.full(n_paths, S0)
    for k in range(n_steps):
        dw = dW[:, k]
        S_mi = S_mi * (1.0 + r * dt + sigma * dw
                        + 0.5 * sigma * sigma * (dw * dw - dt))

    return S_eu, S_mi, S_exact


def run_strong_convergence() -> Dict[str, Tuple[List[float], List[float]]]:
    _print_header("ARTEFACT 2: Euler vs Milstein strong convergence")
    print(f"  n_paths = {N_PATHS_STRONG:_}; "
          f"Delta t = 1 / N for N in {DT_INVS_STRONG}")
    print()

    S0, r, sigma, T = (CANONICAL["S"], CANONICAL["r"],
                       CANONICAL["sigma"], CANONICAL["T"])

    dt_list = []
    err_eu_list = []
    err_mi_list = []

    rng_master = np.random.default_rng(RNG_SEED)

    print(f"    {'dt':>10s} {'n_steps':>8s} {'err_euler':>14s} "
          f"{'err_milstein':>14s} {'t (s)':>8s}")
    print("    " + "-" * 60)
    for inv_dt in DT_INVS_STRONG:
        n_steps = inv_dt
        dt = 1.0 / inv_dt
        rng = np.random.default_rng(rng_master.integers(2**63))
        t0 = time.perf_counter()
        S_eu, S_mi, S_ex = _coupled_paths(S0, r, sigma, T,
                                            n_steps, N_PATHS_STRONG, rng)
        elapsed = time.perf_counter() - t0
        # Strong (L^2) error: sqrt(E[(S^scheme - S^exact)^2])
        err_eu = float(np.sqrt(np.mean((S_eu - S_ex) ** 2)))
        err_mi = float(np.sqrt(np.mean((S_mi - S_ex) ** 2)))
        dt_list.append(dt)
        err_eu_list.append(err_eu)
        err_mi_list.append(err_mi)
        print(f"    {dt:>10.4e} {n_steps:>8d} {err_eu:>14.4e} "
              f"{err_mi:>14.4e} {elapsed:>8.2f}")

    return {"Euler":    (dt_list, err_eu_list),
            "Milstein": (dt_list, err_mi_list)}


def save_strong_convergence_csv(results: Dict[str, Tuple[List[float],
                                                          List[float]]]
                                 ) -> None:
    path = RESULTS_DIR / "euler_milstein_convergence.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "dt", "n_steps", "strong_error_L2"])
        for scheme, (dts, errs) in results.items():
            for dt, err in zip(dts, errs):
                w.writerow([scheme, f"{dt:.6e}", int(round(1.0 / dt)),
                            f"{err:.6e}"])
    print(f"\n  Saved: {path}")


def plot_strong_convergence(results: Dict[str, Tuple[List[float],
                                                      List[float]]]
                             ) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    dt_eu, err_eu = results["Euler"]
    dt_mi, err_mi = results["Milstein"]

    ax.loglog(dt_eu, err_eu, "o-", color="#4C72B0",
              label="Euler-Maruyama", markersize=8, linewidth=1.6,
              alpha=0.9)
    ax.loglog(dt_mi, err_mi, "s-", color="#C44E52",
              label="Milstein", markersize=8, linewidth=1.6,
              alpha=0.9)

    # Reference slope lines. We shift them visibly above the data by a
    # factor of ~1.8x so they appear as distinct guides rather than
    # overlapping with the empirical curves.
    dt_arr = np.array(dt_eu)
    # Slope 1/2, anchored above Euler's leftmost point.
    ref_half = 1.8 * err_eu[0] * (dt_arr / dt_arr[0]) ** 0.5
    ax.loglog(dt_arr, ref_half, "k--", linewidth=1.1, alpha=0.6,
              label=r"slope $\propto \sqrt{\Delta t}$ (Euler, strong order 1/2)")
    # Slope 1, anchored above Milstein's leftmost point.
    ref_one = 1.8 * err_mi[0] * (dt_arr / dt_arr[0]) ** 1.0
    ax.loglog(dt_arr, ref_one, "k:", linewidth=1.1, alpha=0.6,
              label=r"slope $\propto \Delta t$ (Milstein, strong order 1)")

    ax.set_xlabel(r"$\Delta t$", fontsize=11)
    ax.set_ylabel(r"strong $L^2$ error "
                  r"$\sqrt{\,\mathbb{E}[\,(S_T^{\Delta t}"
                  r" - S_T^{\mathrm{exact}})^2\,]\,}$",
                  fontsize=11)
    ax.set_title("Phase 2: Euler vs Milstein strong convergence "
                 "(Brownian-coupled paths)",
                 fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.invert_xaxis()

    fig.text(0.5, -0.02,
             f"GBM at S=K=100, r=0.05, $\\sigma$=0.20, T=1. "
             f"n_paths = {N_PATHS_STRONG:_}. Same Brownian increments "
             "used for all three schemes per path.",
             ha="center", fontsize=8, color="gray", style="italic")

    fig.tight_layout()
    out = RESULTS_DIR / "euler_milstein_convergence.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ==========================================================================
# Section 3: QMC vs MC error scaling
# ==========================================================================
def run_qmc_vs_mc() -> Dict[str, Tuple[List[int], List[float]]]:
    _print_header("ARTEFACT 3: QMC vs MC error scaling")
    ref = bs_call(**CANONICAL)
    print(f"  BS analytical reference: {ref:.8f}")
    print(f"  N in {N_LIST_QMC}")
    print(f"  Replications per N: {N_SEEDS_QMC}")
    print()

    rms_iid = []
    rms_rqmc = []

    print(f"    {'N':>7s} {'RMS IID':>14s} {'RMS RQMC':>14s} "
          f"{'t IID (s)':>10s} {'t RQMC (s)':>11s}")
    print("    " + "-" * 60)

    for N in N_LIST_QMC:
        # --- IID MC: average squared error across many seeds ---
        errs_iid = []
        t0 = time.perf_counter()
        for k in range(N_SEEDS_QMC):
            res = mc_european_call_exact(
                **CANONICAL, n_paths=N, seed=RNG_SEED + 100 + k)
            errs_iid.append((res.estimate - ref) ** 2)
        rms_i = float(np.sqrt(np.mean(errs_iid)))
        t_iid = time.perf_counter() - t0
        rms_iid.append(rms_i)

        # --- Sobol RQMC: same RMS measure ---
        # We choose n_paths_per_rep so that total = N exactly, with R
        # fixed at N_RQMC_REPLICATIONS. Skip very small N where there
        # is not enough budget per replication.
        n_per = max(N // N_RQMC_REPLICATIONS, 16)
        errs_rqmc = []
        t0 = time.perf_counter()
        for k in range(N_SEEDS_QMC):
            res = mc_european_call_euler_rqmc(
                **CANONICAL, n_paths=n_per, n_steps=20,
                n_replications=N_RQMC_REPLICATIONS,
                seed=RNG_SEED + 200 + k)
            errs_rqmc.append((res.estimate - ref) ** 2)
        rms_r = float(np.sqrt(np.mean(errs_rqmc)))
        t_rqmc = time.perf_counter() - t0
        rms_rqmc.append(rms_r)

        print(f"    {N:>7d} {rms_i:>14.4e} {rms_r:>14.4e} "
              f"{t_iid:>10.2f} {t_rqmc:>11.2f}")

    return {"IID MC":     (N_LIST_QMC, rms_iid),
            "Sobol RQMC": (N_LIST_QMC, rms_rqmc)}


def save_qmc_vs_mc_csv(results: Dict[str, Tuple[List[int], List[float]]]
                        ) -> None:
    path = RESULTS_DIR / "qmc_vs_mc.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "N", "rms_error"])
        for method, (Ns, errs) in results.items():
            for N, err in zip(Ns, errs):
                w.writerow([method, N, f"{err:.6e}"])
    print(f"\n  Saved: {path}")


def _fit_slope_loglog(x: List[float], y: List[float]) -> float:
    """Slope of a log-log fit of y vs x."""
    lx = np.log(np.asarray(x, dtype=float))
    ly = np.log(np.asarray(y, dtype=float))
    slope, _ = np.polyfit(lx, ly, 1)
    return float(slope)


def plot_qmc_vs_mc(results: Dict[str, Tuple[List[int], List[float]]]
                    ) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    Ns_iid, rms_iid = results["IID MC"]
    Ns_qmc, rms_qmc = results["Sobol RQMC"]

    slope_iid = _fit_slope_loglog(Ns_iid, rms_iid)
    slope_qmc = _fit_slope_loglog(Ns_qmc, rms_qmc)

    ax.loglog(Ns_iid, rms_iid, "o-", color="#4C72B0",
              label=f"IID MC  (fit slope = {slope_iid:.2f})",
              markersize=8, linewidth=1.6, alpha=0.9)
    ax.loglog(Ns_qmc, rms_qmc, "s-", color="#55A868",
              label=f"Sobol RQMC  (fit slope = {slope_qmc:.2f})",
              markersize=8, linewidth=1.6, alpha=0.9)

    # Reference slopes for visual calibration.
    N_arr = np.array(Ns_iid, dtype=float)
    ref_half = rms_iid[0] * (N_arr / N_arr[0]) ** (-0.5)
    ax.loglog(N_arr, ref_half, "k--", linewidth=0.9, alpha=0.5,
              label=r"slope $\propto N^{-1/2}$")
    ref_one = rms_qmc[0] * (N_arr / N_arr[0]) ** (-1.0)
    ax.loglog(N_arr, ref_one, "k:", linewidth=0.9, alpha=0.6,
              label=r"slope $\propto N^{-1}$")

    ax.set_xlabel("N (total sample size)", fontsize=11)
    ax.set_ylabel("RMS error vs Black-Scholes "
                  f"(over {N_SEEDS_QMC} seeds)",
                  fontsize=11)
    ax.set_title("Phase 2: convergence rate of IID MC vs Sobol RQMC",
                 fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)

    fig.text(0.5, -0.05,
             "European call at S=K=100, r=0.05, $\\sigma$=0.20, T=1. "
             "Euler discretisation with n_steps=20 (problem dimension d=20).\n"
             "RQMC uses Sobol with 16 random shifts. Sobol's asymptotic "
             "$N^{-1}$ rate carries a $(\\log N)^d$ prefactor that remains "
             "significant\n"
             "at this dimensionality and sample size, so the empirical slope "
             "settles above $-1$ in the observed range "
             "(Glasserman, Sec. 5.2).",
             ha="center", va="top",
             fontsize=8, color="gray", style="italic")

    fig.tight_layout()
    out = RESULTS_DIR / "qmc_vs_mc.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")



# ==========================================================================
# Main
# ==========================================================================
def main() -> None:
    t_start = time.perf_counter()
    print(f"Phase 2 Block 6 - Final benchmark")
    print(f"Output directory: {RESULTS_DIR}")

    # --- Artefact 1 ---
    vr_records = run_vr_scoreboard()
    save_vr_scoreboard_csv(vr_records)
    plot_vr_scoreboard(vr_records)

    # --- Artefact 2 ---
    strong_results = run_strong_convergence()
    save_strong_convergence_csv(strong_results)
    plot_strong_convergence(strong_results)

    # --- Artefact 3 ---
    qmc_results = run_qmc_vs_mc()
    save_qmc_vs_mc_csv(qmc_results)
    plot_qmc_vs_mc(qmc_results)

    # --- Summary ---
    elapsed = time.perf_counter() - t_start
    print()
    print("=" * 100)
    print(f"  Benchmark complete in {elapsed:.1f} s "
          f"({elapsed/60:.1f} min)")
    print(f"  Outputs in: {RESULTS_DIR}")
    print("=" * 100)
    for f in sorted(RESULTS_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:40s} {size_kb:>8.1f} KB")


if __name__ == "__main__":
    main()
