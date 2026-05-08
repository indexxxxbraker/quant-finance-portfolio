"""
Phase 3 Block 6 - Final benchmark.

Orchestrates all available pricers in `quantlib` to produce the closing
report of Phase 3:

  - Section 1: pricing comparison at canonical and stress cases
  - Section 2: convergence study (log-log plots)
  - Section 3: cost-precision frontier
  - Section 4: free boundary recovery (CN-PSOR vs trinomial)

Outputs:
  - Tables to stdout (human-readable)
  - CSV files in python/results/phase3/ (machine-readable)
  - PNG plots in python/results/phase3/

Run from the python/ directory:

    python benchmark_phase3.py

Total runtime ~2-3 minutes at production sizes (n_paths=500_000).
"""

import csv
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Any, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Pricer imports (verified by preflight_phase3.py)
# --------------------------------------------------------------------------
from quantlib.black_scholes import call_price as bs_call, put_price as bs_put
from quantlib.monte_carlo import (
    mc_european_call_exact,
    mc_european_call_euler,
    mc_european_call_milstein,
    MCResult,
)
from quantlib.variance_reduction import (
    mc_european_call_exact_av,
    mc_european_call_exact_cv_underlying,
    mc_european_call_exact_cv_aon,
)
from quantlib.qmc import (
    mc_european_call_euler_qmc,
    mc_european_call_euler_rqmc,
)
from quantlib.american import binomial_american_put, lsm_american_put
from quantlib.ftcs import ftcs_european_call, ftcs_min_M_for_cfl
from quantlib.btcs import btcs_european_call
from quantlib.cn import cn_european_call
from quantlib.cn_american import cn_american_put
from quantlib.trinomial import (
    trinomial_european_call,
    trinomial_american_put,
)


# ==========================================================================
# Configuration
# ==========================================================================
CANONICAL = dict(S=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0)
STRESS_HIGH_VOL = dict(S=100.0, K=100.0, r=0.05, sigma=0.40, T=1.0)
STRESS_SHORT_T  = dict(S=100.0, K=100.0, r=0.05, sigma=0.20, T=0.10)

# Production sizes
N_PATHS              = 500_000   # MC methods at canonical
N_STEPS_LATTICE      = 1000      # CRR / trinomial
N_GRID_PDE           = 400       # spatial nodes for PDE
M_GRID_PDE           = 200       # time steps for PDE
N_PSOR_REF           = 800       # PSOR reference (high precision)
M_PSOR_REF           = 400
RNG_SEED             = 42

# Convergence study sizes
CONV_N_PATHS         = [10_000, 50_000, 250_000, 1_250_000]
CONV_N_STEPS_LATTICE = [50, 100, 200, 400, 800, 1600]
CONV_N_GRID_PDE      = [50, 100, 200, 400, 800]

# Free boundary sampling
FB_N_TAUS            = 25
FB_N_GRID            = 200
FB_M_GRID            = 100
FB_N_STEPS_TRI       = 800

# Output directory
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "phase3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================================
# MethodSpec abstraction
# ==========================================================================
@dataclass
class MethodSpec:
    """Configuration for one pricing method.

    Each method is a (name, fn, extract_price, extract_se, kwargs) bundle
    that decouples the calling convention from the benchmark loop.
    """
    name: str
    fn: Callable
    extract_price: Callable[[Any], float]
    extract_se: Optional[Callable[[Any], float]] = None  # half_width if MC
    category: str = "other"


@dataclass
class RunResult:
    method: str
    category: str
    price: float
    error: float
    half_width: Optional[float]
    elapsed_ms: float


def run_method(spec: MethodSpec, params: dict, kwargs: dict,
               reference: float) -> RunResult:
    """Execute a single pricing method, return a RunResult row."""
    t0 = time.perf_counter()
    raw = spec.fn(**params, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    price = spec.extract_price(raw)
    half_width = spec.extract_se(raw) if spec.extract_se else None
    return RunResult(
        method=spec.name,
        category=spec.category,
        price=price,
        error=abs(price - reference),
        half_width=half_width,
        elapsed_ms=elapsed_ms,
    )


# ==========================================================================
# Method registries
# ==========================================================================
def european_call_methods(seed: int = RNG_SEED) -> List[Tuple[MethodSpec, dict]]:
    """All methods to benchmark for European call."""
    return [
        # --- Closed form ---
        (MethodSpec("BS closed form", bs_call,
                    extract_price=lambda x: x,
                    category="closed_form"),
         {}),

        # --- Vanilla MC ---
        (MethodSpec("MC exact (terminal)", mc_european_call_exact,
                    extract_price=lambda r: r.estimate,
                    extract_se=lambda r: r.half_width,
                    category="mc_vanilla"),
         dict(n_paths=N_PATHS, seed=seed)),
        (MethodSpec("MC Euler (n_steps=20)", mc_european_call_euler,
                    extract_price=lambda r: r.estimate,
                    extract_se=lambda r: r.half_width,
                    category="mc_vanilla"),
         dict(n_steps=20, n_paths=N_PATHS, seed=seed)),
        (MethodSpec("MC Milstein (n_steps=20)", mc_european_call_milstein,
                    extract_price=lambda r: r.estimate,
                    extract_se=lambda r: r.half_width,
                    category="mc_vanilla"),
         dict(n_steps=20, n_paths=N_PATHS, seed=seed)),

        # --- Variance reduction (all on the exact terminal MC) ---
        (MethodSpec("MC + antithetic variates", mc_european_call_exact_av,
                    extract_price=lambda r: r.estimate,
                    extract_se=lambda r: r.half_width,
                    category="mc_vr"),
         dict(n_paths=N_PATHS, seed=seed)),
        (MethodSpec("MC + CV (underlying)",
                    mc_european_call_exact_cv_underlying,
                    extract_price=lambda r: r.estimate,
                    extract_se=lambda r: r.half_width,
                    category="mc_vr"),
         dict(n_paths=N_PATHS, seed=seed)),
        (MethodSpec("MC + CV (asset-or-nothing)",
                    mc_european_call_exact_cv_aon,
                    extract_price=lambda r: r.estimate,
                    extract_se=lambda r: r.half_width,
                    category="mc_vr"),
         dict(n_paths=N_PATHS, seed=seed)),

        # --- QMC ---
        (MethodSpec("Sobol QMC (deterministic)",
                    mc_european_call_euler_qmc,
                    extract_price=lambda x: x,
                    category="qmc"),
         dict(n_paths=N_PATHS, n_steps=20)),
        (MethodSpec("Sobol RQMC (randomised)",
                    mc_european_call_euler_rqmc,
                    extract_price=lambda r: r.estimate,
                    extract_se=lambda r: r.half_width,
                    category="qmc"),
         dict(n_paths=max(1024, N_PATHS // 20),
              n_steps=20, n_replications=20, seed=seed)),

        # --- Lattice ---
        (MethodSpec(f"Trinomial KR (lambda=3)", trinomial_european_call,
                    extract_price=lambda x: x,
                    category="lattice"),
         dict(n_steps=N_STEPS_LATTICE)),

        # --- PDE ---
        (MethodSpec("FTCS", ftcs_european_call,
                    extract_price=lambda x: x,
                    category="pde"),
         dict(N=N_GRID_PDE,
              M=ftcs_min_M_for_cfl(N=N_GRID_PDE, T=1.0,
                                    sigma=0.20, n_sigma=4.0),
              n_sigma=4.0)),
        (MethodSpec("BTCS", btcs_european_call,
                    extract_price=lambda x: x,
                    category="pde"),
         dict(N=N_GRID_PDE, M=M_GRID_PDE, n_sigma=4.0)),
        (MethodSpec("Crank-Nicolson", cn_european_call,
                    extract_price=lambda x: x,
                    category="pde"),
         dict(N=N_GRID_PDE, M=M_GRID_PDE, n_sigma=4.0)),
    ]


def american_put_methods(seed: int = RNG_SEED) -> List[Tuple[MethodSpec, dict]]:
    """All methods to benchmark for American put."""
    return [
        (MethodSpec("CRR binomial American", binomial_american_put,
                    extract_price=lambda x: x,
                    category="lattice"),
         dict(n_steps=N_STEPS_LATTICE)),
        (MethodSpec("Trinomial KR American", trinomial_american_put,
                    extract_price=lambda x: x,
                    category="lattice"),
         dict(n_steps=N_STEPS_LATTICE)),
        (MethodSpec("CN-PSOR", cn_american_put,
                    extract_price=lambda x: x,
                    category="pde"),
         dict(N=N_GRID_PDE, M=M_GRID_PDE, omega=1.4, n_sigma=4.0)),
        (MethodSpec("LSM (Longstaff-Schwartz)", lsm_american_put,
                    extract_price=lambda r: r.estimate,
                    extract_se=lambda r: r.half_width,
                    category="mc"),
         dict(n_paths=N_PATHS, n_steps=50, basis_size=4, seed=seed)),
    ]


# ==========================================================================
# Reference computation
# ==========================================================================
def american_reference(params: dict) -> float:
    """Lattice consensus reference for the American put.

    Average of CRR and trinomial at n=8000. Each lattice method has
    truncation error ~1/8000 ~ 1.25e-4 at this size, much smaller than
    CN-PSOR at moderate (N, M). Averaging cancels residual bias from
    each method's odd-even oscillation in the at-the-money region.
    """
    crr = binomial_american_put(**params, n_steps=8000)
    tri = trinomial_american_put(**params, n_steps=8000)
    return 0.5 * (crr + tri)


def consensus_check(params: dict, ref_value: float) -> dict:
    """Cross-check: at moderate n, do the methods agree on the American put?

    With the reference now being the lattice consensus at n=8000,
    this probes whether the methods at smaller n (and CN-PSOR at moderate
    N, M) have already converged.
    """
    crr_4k  = binomial_american_put(**params, n_steps=4000)
    tri_4k  = trinomial_american_put(**params, n_steps=4000)
    cn_800  = cn_american_put(**params, N=800, M=400, omega=1.4, n_sigma=4.0)
    methods = {"CRR (n=4000)": crr_4k,
               "Trinomial (n=4000)": tri_4k,
               "CN-PSOR (N=800,M=400)": cn_800}
    spread = max(methods.values()) - min(methods.values())
    return {
        "Reference (lattice avg, n=8000)": ref_value,
        **methods,
        "spread (across cross-check)": spread,
    }


# ==========================================================================
# Section 1: pricing benchmark
# ==========================================================================
def fmt_row(method: str, price: float, error: float, ref: float,
            half_width: Optional[float], elapsed_ms: float) -> str:
    """Single-line formatted row."""
    rel = error / ref * 100 if ref else 0.0
    hw_str = f"{half_width:8.4f}" if half_width is not None else "       —"
    return (f"  {method:38s} {price:10.5f}  {error:.2e}  "
            f"({rel:5.2f}%)  {hw_str}  {elapsed_ms:9.1f} ms")


def print_table_header(title: str) -> None:
    print()
    print(f"  {title}")
    print("  " + "─" * 102)
    print(f"  {'method':38s} {'price':>10s}  {'abs err':>8s}"
          f"  {'%err':>6s}   {'half_w':>8s}  {'time':>13s}")
    print("  " + "─" * 102)


def benchmark_pricing(params: dict, label: str,
                      ref_call: float, ref_put_amer: float) -> dict:
    """Run all pricers at a given parameter set, print tables, return data."""
    print()
    print("=" * 102)
    print(f"  {label}")
    print(f"  S={params['S']}, K={params['K']}, r={params['r']}, "
          f"sigma={params['sigma']}, T={params['T']}")
    print(f"  BS reference (call): {ref_call:.6f}")
    print(f"  Lattice consensus reference (American put): {ref_put_amer:.6f}")
    print("=" * 102)

    out = {"label": label, "params": params,
           "european": [], "american": []}

    print_table_header("EUROPEAN CALL")
    for spec, kwargs in european_call_methods():
        try:
            res = run_method(spec, params, kwargs, ref_call)
            print(fmt_row(spec.name, res.price, res.error, ref_call,
                          res.half_width, res.elapsed_ms))
            out["european"].append(res)
        except Exception as e:
            print(f"  [FAIL] {spec.name}: {type(e).__name__}: {e}")

    print_table_header("AMERICAN PUT")
    for spec, kwargs in american_put_methods():
        try:
            res = run_method(spec, params, kwargs, ref_put_amer)
            print(fmt_row(spec.name, res.price, res.error, ref_put_amer,
                          res.half_width, res.elapsed_ms))
            out["american"].append(res)
        except Exception as e:
            print(f"  [FAIL] {spec.name}: {type(e).__name__}: {e}")

    return out


def save_pricing_csv(filename: str, run_data: dict, ref_call: float,
                     ref_put_amer: float) -> None:
    """Save a pricing table to CSV."""
    path = RESULTS_DIR / filename
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "option_type", "method", "category",
                    "price", "reference", "abs_error", "rel_error_pct",
                    "half_width", "elapsed_ms"])
        for r in run_data["european"]:
            w.writerow([run_data["label"], "european_call",
                        r.method, r.category,
                        f"{r.price:.6f}", f"{ref_call:.6f}",
                        f"{r.error:.6e}",
                        f"{r.error/ref_call*100:.4f}" if ref_call else "",
                        f"{r.half_width:.6e}" if r.half_width else "",
                        f"{r.elapsed_ms:.2f}"])
        for r in run_data["american"]:
            w.writerow([run_data["label"], "american_put",
                        r.method, r.category,
                        f"{r.price:.6f}", f"{ref_put_amer:.6f}",
                        f"{r.error:.6e}",
                        f"{r.error/ref_put_amer*100:.4f}" if ref_put_amer else "",
                        f"{r.half_width:.6e}" if r.half_width else "",
                        f"{r.elapsed_ms:.2f}"])
    print(f"  Saved: {path}")


# ==========================================================================
# Section 2: convergence study
# ==========================================================================
def convergence_european_call(params: dict, ref_call: float) -> dict:
    """Run convergence study for European call at canonical case."""
    print("\n" + "=" * 102)
    print("  CONVERGENCE STUDY: European call")
    print("=" * 102)

    results = {}

    # Stochastic methods: vary n_paths
    print("\n  Stochastic (vary n_paths):")
    stochastic_specs = [
        ("MC exact",
         lambda n: mc_european_call_exact(**params, n_paths=n,
                                            seed=RNG_SEED).estimate),
        ("MC + antithetic",
         lambda n: mc_european_call_exact_av(**params, n_paths=n,
                                               seed=RNG_SEED).estimate),
        ("MC + CV underlying",
         lambda n: mc_european_call_exact_cv_underlying(**params, n_paths=n,
                                                          seed=RNG_SEED).estimate),
        ("Sobol QMC",
         lambda n: mc_european_call_euler_qmc(**params, n_paths=n,
                                                n_steps=20)),
    ]
    for name, fn in stochastic_specs:
        errs = []
        for n in CONV_N_PATHS:
            err = abs(fn(n) - ref_call)
            errs.append(err)
            print(f"    {name:30s} n={n:>9d}: err={err:.4e}")
        results[name] = (CONV_N_PATHS, errs)

    # Lattice: vary n_steps
    print("\n  Lattice (vary n_steps):")
    errs_tri = []
    for n in CONV_N_STEPS_LATTICE:
        p = trinomial_european_call(**params, n_steps=n)
        e = abs(p - ref_call)
        errs_tri.append(e)
        print(f"    Trinomial KR                   n_steps={n:>4d}: err={e:.4e}")
    results["Trinomial KR"] = (CONV_N_STEPS_LATTICE, errs_tri)

    # PDE: vary N
    print("\n  PDE (vary N, M scaled):")
    pde_specs = [
        ("FTCS", ftcs_european_call,
         lambda N: ftcs_min_M_for_cfl(N=N, T=params["T"],
                                       sigma=params["sigma"], n_sigma=4.0)),
        ("BTCS", btcs_european_call, lambda N: max(N // 4, 50)),
        ("Crank-Nicolson", cn_european_call, lambda N: max(N // 4, 50)),
    ]
    for name, fn, m_for_n in pde_specs:
        errs = []
        for N in CONV_N_GRID_PDE:
            M = m_for_n(N)
            try:
                p = fn(**params, N=N, M=M, n_sigma=4.0)
                e = abs(p - ref_call)
            except Exception as ex:
                p = np.nan
                e = np.nan
            errs.append(e)
            print(f"    {name:30s} N={N:>4d} M={M:>5d}: err={e:.4e}")
        results[name] = (CONV_N_GRID_PDE, errs)

    return results


def convergence_american_put(params: dict, ref_put: float) -> dict:
    """Run convergence study for American put at canonical case."""
    print("\n" + "=" * 102)
    print("  CONVERGENCE STUDY: American put")
    print("=" * 102)

    results = {}

    # Lattice
    print("\n  Lattice (vary n_steps):")
    for name, fn in [("CRR American", binomial_american_put),
                     ("Trinomial American", trinomial_american_put)]:
        errs = []
        for n in CONV_N_STEPS_LATTICE:
            p = fn(**params, n_steps=n)
            e = abs(p - ref_put)
            errs.append(e)
            print(f"    {name:30s} n_steps={n:>4d}: err={e:.4e}")
        results[name] = (CONV_N_STEPS_LATTICE, errs)

    # PDE: CN-PSOR
    print("\n  PDE (vary N):")
    errs = []
    for N in CONV_N_GRID_PDE:
        M = max(N // 4, 50)
        p = cn_american_put(**params, N=N, M=M, omega=1.4, n_sigma=4.0)
        e = abs(p - ref_put)
        errs.append(e)
        print(f"    CN-PSOR                        N={N:>4d} M={M:>5d}: err={e:.4e}")
    results["CN-PSOR"] = (CONV_N_GRID_PDE, errs)

    # LSM
    print("\n  Stochastic (vary n_paths):")
    errs = []
    for n in CONV_N_PATHS:
        r = lsm_american_put(**params, n_paths=n, n_steps=50,
                              basis_size=4, seed=RNG_SEED)
        e = abs(r.estimate - ref_put)
        errs.append(e)
        print(f"    LSM                            n_paths={n:>9d}: err={e:.4e}")
    results["LSM"] = (CONV_N_PATHS, errs)

    return results


def plot_convergence(eur_results: dict, amer_results: dict) -> None:
    """Plot log-log convergence: 2x2 grid (eur stoch, eur det, amer stoch, amer det)."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Helper to add reference slope lines
    def add_ref_line(ax, x_arr, y_anchor, slope, label):
        x = np.array([x_arr[0], x_arr[-1]])
        y = y_anchor * (x / x[0]) ** slope
        ax.loglog(x, y, "k--", linewidth=0.8, alpha=0.5, label=label)

    # European call: stochastic
    ax = axes[0, 0]
    for name, (xs, ys) in eur_results.items():
        if "MC" in name or "QMC" in name:
            ax.loglog(xs, ys, "o-", label=name, alpha=0.85)
    ax.set_xlabel("n_paths")
    ax.set_ylabel("|error| vs BS")
    ax.set_title("European call: stochastic methods")
    ax.grid(True, which="both", alpha=0.3)
    if eur_results:
        # Reference slope -1/2 from a representative MC curve
        for name, (xs, ys) in eur_results.items():
            if "MC exact" in name and ys[0] > 0:
                add_ref_line(ax, xs, ys[0], -0.5, "slope -1/2")
                break
    ax.legend(fontsize=8, loc="best")

    # European call: deterministic
    ax = axes[0, 1]
    for name, (xs, ys) in eur_results.items():
        if "MC" not in name and "QMC" not in name:
            ax.loglog(xs, ys, "s-", label=name, alpha=0.85)
    ax.set_xlabel("size (N or n_steps)")
    ax.set_ylabel("|error| vs BS")
    ax.set_title("European call: deterministic methods")
    ax.grid(True, which="both", alpha=0.3)
    if "Trinomial KR" in eur_results:
        xs, ys = eur_results["Trinomial KR"]
        if ys[0] > 0:
            add_ref_line(ax, xs, ys[0], -1.0, "slope -1")
            add_ref_line(ax, xs, ys[0]/4, -2.0, "slope -2")
    ax.legend(fontsize=8, loc="best")

    # American put: lattice + PDE
    ax = axes[1, 0]
    for name, (xs, ys) in amer_results.items():
        if name != "LSM":
            ax.loglog(xs, ys, "s-", label=name, alpha=0.85)
    ax.set_xlabel("size (N or n_steps)")
    ax.set_ylabel("|error| vs CN-PSOR ref")
    ax.set_title("American put: deterministic methods")
    ax.grid(True, which="both", alpha=0.3)
    if "Trinomial American" in amer_results:
        xs, ys = amer_results["Trinomial American"]
        if ys[0] > 0:
            add_ref_line(ax, xs, ys[0], -1.0, "slope -1")
    ax.legend(fontsize=8, loc="best")

    # American put: LSM
    ax = axes[1, 1]
    if "LSM" in amer_results:
        xs, ys = amer_results["LSM"]
        ax.loglog(xs, ys, "o-", label="LSM", alpha=0.85, color="C3")
    ax.set_xlabel("n_paths")
    ax.set_ylabel("|error| vs CN-PSOR ref")
    ax.set_title("American put: LSM")
    ax.grid(True, which="both", alpha=0.3)
    if "LSM" in amer_results:
        xs, ys = amer_results["LSM"]
        if ys[0] > 0:
            add_ref_line(ax, xs, ys[0], -0.5, "slope -1/2")
    ax.legend(fontsize=8, loc="best")

    fig.suptitle("Phase 3 Block 6: convergence study (canonical case)")
    fig.tight_layout()
    out = RESULTS_DIR / "convergence.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ==========================================================================
# Section 3: cost-precision frontier
# ==========================================================================
def plot_cost_precision(eur_run: dict, amer_run: dict) -> None:
    """Plot wall-clock time vs absolute error, both option types."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    color_map = {
        "closed_form": "k",
        "mc_vanilla":  "C0",
        "mc_vr":       "C1",
        "qmc":         "C2",
        "lattice":     "C3",
        "pde":         "C4",
        "mc":          "C0",
    }

    for ax, run, title in [(ax1, eur_run, "European call"),
                            (ax2, amer_run, "American put")]:
        for r in run:
            if r.error == 0 or np.isnan(r.error):
                continue
            color = color_map.get(r.category, "gray")
            ax.scatter(r.elapsed_ms, r.error, s=80, color=color,
                        edgecolors="black", linewidth=0.5,
                        label=r.method)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("wall-clock time (ms)")
        ax.set_ylabel("|absolute error|")
        ax.set_title(f"{title}: cost vs precision")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle("Phase 3 Block 6: cost-precision frontier (canonical)")
    fig.tight_layout()
    out = RESULTS_DIR / "cost_precision.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ==========================================================================
# Section 4: free boundary recovery
# ==========================================================================
def free_boundary_trinomial(params: dict, n_steps: int = FB_N_STEPS_TRI,
                             lambda_param: float = 3.0
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """One trinomial backward sweep, recording the free boundary at every τ.

    Returns (taus, S_f) of length n_steps. The boundary S_f(τ) is the
    largest spot at which exercise dominates continuation at the
    backward-induction layer corresponding to time-to-maturity τ.
    """
    S, K, r, sigma, T = (params["S"], params["K"], params["r"],
                         params["sigma"], params["T"])
    dt = T / n_steps
    dx = sigma * np.sqrt(lambda_param * dt)
    nu = r - 0.5 * sigma * sigma
    drift_term = nu * np.sqrt(dt) / (2.0 * sigma * np.sqrt(lambda_param))
    p_u = 0.5 / lambda_param + drift_term
    p_m = 1.0 - 1.0 / lambda_param
    p_d = 0.5 / lambda_param - drift_term
    disc = np.exp(-r * dt)

    n_nodes = 2 * n_steps + 1
    j_grid = np.arange(-n_steps, n_steps + 1, dtype=np.float64)
    S_grid = S * np.exp(j_grid * dx)

    V = np.maximum(K - S_grid, 0.0)

    taus = []
    Sfs = []

    for n in range(n_steps - 1, -1, -1):
        start = n_steps - (n + 1)
        end   = n_steps + (n + 1) + 1
        cont = disc * (
            p_u * V[start + 2 : end]
          + p_m * V[start + 1 : end - 1]
          + p_d * V[start     : end - 2]
        )
        j_n = np.arange(-n, n + 1, dtype=np.float64)
        S_n = S * np.exp(j_n * dx)
        exer = np.maximum(K - S_n, 0.0)
        V_new = np.maximum(cont, exer)
        V[start + 1 : end - 1] = V_new

        # Free boundary: largest S_n where exer > cont (i.e., exercise active)
        # In the put case, exercise is for low S; the boundary is the upper
        # edge of the exercise region.
        is_exercise = (V_new <= exer + 1e-12) & (exer > 0.0)
        tau = (n_steps - n) * dt
        if np.any(is_exercise):
            S_f = S_n[is_exercise].max()
        else:
            S_f = np.nan
        taus.append(tau)
        Sfs.append(S_f)

    return np.array(taus), np.array(Sfs)


def free_boundary_cn_psor(params: dict, n_taus: int = FB_N_TAUS,
                           N: int = FB_N_GRID, M: int = FB_M_GRID
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Repeated CN-PSOR calls at varying T, extracting S_f from each result.

    For each τ in a uniform grid, run cn_american_put with T=τ, and ask for
    diagnostics. The final-solution slice plus the spatial grid give
    S_f(τ) directly.
    """
    S, K, r, sigma, T_full = (params["S"], params["K"], params["r"],
                              params["sigma"], params["T"])
    taus = np.linspace(T_full / n_taus, T_full, n_taus)

    Sfs = []
    for tau in taus:
        try:
            result = cn_american_put(S=S, K=K, r=r, sigma=sigma, T=tau,
                                      N=N, M=M, omega=1.4, n_sigma=4.0,
                                      return_diagnostics=True)
            if isinstance(result, tuple):
                price, diag = result
            else:
                # Defensive: if signature differs, skip
                Sfs.append(np.nan)
                continue
            V_final = diag["final_solution"]
            grid = diag["grid"]

            # Reconstruct S array from grid. We probe common attribute names.
            S_arr = None
            for attr in ("S", "S_grid", "spots", "asset_grid"):
                if hasattr(grid, attr):
                    S_arr = np.asarray(getattr(grid, attr))
                    break
            if S_arr is None and hasattr(grid, "x"):
                S_arr = S * np.exp(np.asarray(grid.x))
            elif S_arr is None and hasattr(grid, "x_grid"):
                S_arr = S * np.exp(np.asarray(grid.x_grid))
            if S_arr is None:
                # Last resort: build from grid attributes.
                if hasattr(grid, "x_min") and hasattr(grid, "x_max") \
                        and hasattr(grid, "N"):
                    x = np.linspace(grid.x_min, grid.x_max, grid.N + 1)
                    S_arr = S * np.exp(x)

            if S_arr is None:
                Sfs.append(np.nan)
                continue

            exer = np.maximum(K - S_arr, 0.0)
            is_exercise = (V_final <= exer + 1e-9) & (exer > 0.0)
            if np.any(is_exercise):
                Sfs.append(S_arr[is_exercise].max())
            else:
                Sfs.append(np.nan)
        except Exception as e:
            print(f"    [warn] CN-PSOR boundary at tau={tau:.3f} failed: {e}")
            Sfs.append(np.nan)

    return taus, np.array(Sfs)


def plot_free_boundary(params: dict) -> None:
    """Plot S_f(τ) recovered from both CN-PSOR and trinomial."""
    print("\n" + "=" * 102)
    print("  FREE BOUNDARY RECOVERY")
    print("=" * 102)

    print("\n  Trinomial backward sweep (single instrumented call)...")
    t0 = time.perf_counter()
    taus_tri, Sfs_tri = free_boundary_trinomial(params)
    print(f"    {len(taus_tri)} points in {(time.perf_counter()-t0):.2f} s")

    print("  CN-PSOR (repeated calls at varying T)...")
    t0 = time.perf_counter()
    taus_cn, Sfs_cn = free_boundary_cn_psor(params)
    print(f"    {len(taus_cn)} points in {(time.perf_counter()-t0):.2f} s")

    # Save CSV
    csv_path = RESULTS_DIR / "free_boundary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "tau", "S_f"])
        for t, s in zip(taus_tri, Sfs_tri):
            w.writerow(["trinomial", f"{t:.6f}",
                        f"{s:.6f}" if np.isfinite(s) else ""])
        for t, s in zip(taus_cn, Sfs_cn):
            w.writerow(["cn_psor", f"{t:.6f}",
                        f"{s:.6f}" if np.isfinite(s) else ""])
    print(f"  Saved: {csv_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    valid_tri = np.isfinite(Sfs_tri)
    valid_cn  = np.isfinite(Sfs_cn)
    if valid_tri.any():
        ax.plot(taus_tri[valid_tri], Sfs_tri[valid_tri],
                ".-", linewidth=0.8, label="Trinomial KR (n_steps=800)",
                alpha=0.85, color="C3")
    if valid_cn.any():
        ax.plot(taus_cn[valid_cn], Sfs_cn[valid_cn],
                "o-", markersize=6, linewidth=1.3,
                label=f"CN-PSOR (N={FB_N_GRID}, M={FB_M_GRID}, "
                       f"{FB_N_TAUS} maturities)",
                color="C4")
    ax.axhline(params["K"], color="gray", linestyle=":", linewidth=0.7,
               label=f"Strike K={params['K']}")
    ax.set_xlabel(r"time-to-maturity $\tau$")
    ax.set_ylabel(r"free boundary $S_f(\tau)$")
    ax.set_title(f"American put: exercise boundary "
                 f"(S={params['S']}, K={params['K']}, "
                 f"r={params['r']}, sigma={params['sigma']})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    out = RESULTS_DIR / "free_boundary.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def save_convergence_csv(eur: dict, amer: dict) -> None:
    path = RESULTS_DIR / "convergence.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["option", "method", "size", "abs_error"])
        for name, (xs, ys) in eur.items():
            for x, y in zip(xs, ys):
                w.writerow(["european_call", name, x,
                            f"{y:.6e}" if np.isfinite(y) else ""])
        for name, (xs, ys) in amer.items():
            for x, y in zip(xs, ys):
                w.writerow(["american_put", name, x,
                            f"{y:.6e}" if np.isfinite(y) else ""])
    print(f"  Saved: {path}")


# ==========================================================================
# Main
# ==========================================================================
def main():
    t_start = time.perf_counter()
    print(f"Phase 3 Block 6 - Final benchmark")
    print(f"Output directory: {RESULTS_DIR}")

    # ----- References -----
    print("\n" + "=" * 102)
    print("  REFERENCES")
    print("=" * 102)
    refs = {}
    for label, params in [("canonical", CANONICAL),
                          ("high_vol",  STRESS_HIGH_VOL),
                          ("short_T",   STRESS_SHORT_T)]:
        ref_call = bs_call(**params)
        ref_amer = american_reference(params)
        refs[label] = (ref_call, ref_amer)
        print(f"\n  {label}: {params}")
        print(f"    BS call:  {ref_call:.6f}")
        print(f"    Am. put:  {ref_amer:.6f}")

    # Consensus check at canonical
    print("\n  Consensus check on American reference (canonical):")
    consensus = consensus_check(CANONICAL, refs["canonical"][1])
    for k, v in consensus.items():
        if k == "spread":
            tag = "OK" if v < 5e-3 else "warn"
            print(f"    [{tag}] {k}: {v:.4e}")
        else:
            print(f"    {k:25s}: {v:.6f}")

    # ----- Section 1: pricing tables -----
    runs = {}
    for label, params in [("canonical", CANONICAL),
                          ("high_vol",  STRESS_HIGH_VOL),
                          ("short_T",   STRESS_SHORT_T)]:
        ref_call, ref_amer = refs[label]
        runs[label] = benchmark_pricing(params, label, ref_call, ref_amer)

    # Save CSVs
    print("\n  CSV outputs:")
    for label, run_data in runs.items():
        ref_call, ref_amer = refs[label]
        save_pricing_csv(f"pricing_{label}.csv", run_data, ref_call, ref_amer)

    # ----- Section 2: convergence -----
    eur_conv = convergence_european_call(CANONICAL, refs["canonical"][0])
    amer_conv = convergence_american_put(CANONICAL, refs["canonical"][1])
    save_convergence_csv(eur_conv, amer_conv)
    plot_convergence(eur_conv, amer_conv)

    # ----- Section 3: cost-precision -----
    print("\n" + "=" * 102)
    print("  COST-PRECISION FRONTIER")
    print("=" * 102)
    plot_cost_precision(runs["canonical"]["european"],
                        runs["canonical"]["american"])

    # ----- Section 4: free boundary -----
    plot_free_boundary(CANONICAL)

    # ----- Summary -----
    elapsed = time.perf_counter() - t_start
    print()
    print("=" * 102)
    print(f"  Benchmark complete in {elapsed:.1f} s ({elapsed/60:.1f} min)")
    print(f"  Outputs in: {RESULTS_DIR}")
    print("=" * 102)
    for f in sorted(RESULTS_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:40s} {size_kb:>8.1f} KB")


if __name__ == "__main__":
    main()
