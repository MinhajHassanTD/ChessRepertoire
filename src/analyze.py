"""
C10 — Analysis (src/analyze.py)
Load all run pickles, compute statistics, and produce tables and plots for the paper.

Depends on: C9 outputs (runs/*.pkl).

Outputs
-------
results/main_table.csv       — one row per method; mean/std/median heldout,
                                Holm-corrected Wilcoxon p vs STATIC, A12 vs STATIC
results/convergence.png      — best-training-fitness curves with 95 % CI bands
results/sensitivity_table.csv — (method, lambda) rows; mean/std heldout
results/diagnostic_table.csv — COEVOLVE per-generation averages across seeds
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — keeps output deterministic
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ── Constants ─────────────────────────────────────────────────────────────────

METHODS_MAIN = ["most_played_baseline", "STATIC", "COEVOLVE_FROZEN", "COEVOLVE"]
NON_STATIC_METHODS = ["most_played_baseline", "COEVOLVE_FROZEN", "COEVOLVE"]
MAIN_SEEDS = range(1000, 1015)   # 15 seeds
SENS_SEEDS = range(2000, 2005)   # 5 seeds
MAIN_LAMBDA = 1.0
SENSITIVITY_LAMBDAS = [0.0, 1.0, 2.0]

_CONVERGENCE_COLORS = {
    "STATIC": "steelblue",
    "COEVOLVE_FROZEN": "darkorange",
    "COEVOLVE": "forestgreen",
}


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_runs(runs_dir: str) -> list[dict]:
    """
    Load every .pkl file from *runs_dir*.
    Files are sorted by name so the order is deterministic across calls.
    """
    runs: list[dict] = []
    for path in sorted(Path(runs_dir).glob("*.pkl")):
        with open(path, "rb") as fh:
            runs.append(pickle.load(fh))
    return runs


# ── Internal helpers ──────────────────────────────────────────────────────────

def _select(
    runs: list[dict],
    *,
    mode: Optional[str] = None,
    seeds: Optional[range] = None,
    lambda_weight: Optional[float] = None,
) -> list[dict]:
    """Filter runs by mode, seed set, and lambda_weight."""
    out = []
    for r in runs:
        if mode is not None and r["mode"] != mode:
            continue
        if seeds is not None and r["seed"] not in seeds:
            continue
        if lambda_weight is not None:
            if r["config"].get("lambda_weight") != lambda_weight:
                continue
        out.append(r)
    return out


def _scores_by_seed(runs: list[dict]) -> dict[int, float]:
    """Return {seed: heldout_score} for a filtered run list."""
    return {r["seed"]: r["heldout_score"] for r in runs}


# ── A12 effect size ───────────────────────────────────────────────────────────

def _a12(x: list[float], y: list[float]) -> float:
    """
    Vargha-Delaney A12 effect size.

    A12(X, Y) = ( #{(xi,yj) : xi > yj} + 0.5 * #{(xi,yj) : xi == yj} )
                / ( |X| * |Y| )

    A value > 0.5 means X tends to be larger than Y.
    """
    n = len(x) * len(y)
    if n == 0:
        return float("nan")
    wins = sum(1 for xi in x for yj in y if xi > yj)
    ties = sum(1 for xi in x for yj in y if xi == yj)
    return (wins + 0.5 * ties) / n


# ── Holm correction ───────────────────────────────────────────────────────────

def _holm_correct(p_values: list[float]) -> list[float]:
    """
    Holm step-down correction.

    Sort p-values ascending; multiply the smallest by k, the next by k-1, …,
    clip each at 1.0.  Returns corrected values in the *same order* as the
    input list.

    Spec (Section E, C10): "sort p-values ascending, multiply the smallest by k,
    the next by k-1, etc., and clip at 1.0.  With three comparisons against
    STATIC, k = 3."
    """
    k = len(p_values)
    if k == 0:
        return []
    order = sorted(range(k), key=lambda i: p_values[i])
    corrected: list[Optional[float]] = [None] * k
    for rank, idx in enumerate(order):
        multiplier = k - rank          # k, k-1, …, 1
        corrected[idx] = min(1.0, p_values[idx] * multiplier)
    return corrected  # type: ignore[return-value]


# ── Main table ────────────────────────────────────────────────────────────────

def compute_main_table(runs: list[dict]) -> pd.DataFrame:
    """
    Compute results/main_table.csv.

    One row per method (most_played_baseline, STATIC, COEVOLVE_FROZEN, COEVOLVE).
    Columns: method, mean_heldout, std_heldout, median_heldout,
             wilcoxon_p_vs_STATIC, A12_vs_STATIC.

    Uses main-experiment runs: seeds 1000–1014, lambda=1.0.
    Wilcoxon p-values are Holm-corrected across the three non-STATIC methods.
    """
    # Per-method {seed: heldout_score}
    method_scores: dict[str, dict[int, float]] = {}
    for m in METHODS_MAIN:
        method_scores[m] = _scores_by_seed(
            _select(runs, mode=m, seeds=MAIN_SEEDS, lambda_weight=MAIN_LAMBDA)
        )

    static_by_seed = method_scores["STATIC"]

    # --- Raw Wilcoxon p-values for the three non-STATIC methods ---------------
    raw_p: list[float] = []
    for m in NON_STATIC_METHODS:
        m_by_seed = method_scores[m]
        shared_seeds = sorted(set(m_by_seed) & set(static_by_seed))
        if len(shared_seeds) < 2:
            raw_p.append(float("nan"))
            continue
        x = [m_by_seed[s] for s in shared_seeds]
        y = [static_by_seed[s] for s in shared_seeds]
        diffs = [xi - yi for xi, yi in zip(x, y)]
        if all(d == 0.0 for d in diffs):
            raw_p.append(1.0)
        else:
            try:
                _, pval = stats.wilcoxon(x, y, zero_method="wilcox")
                raw_p.append(float(pval))
            except ValueError:
                raw_p.append(1.0)

    # Handle NaNs: pass only non-NaN values to Holm, keep NaN positions
    nan_mask = [np.isnan(p) for p in raw_p]
    finite_p = [p for p in raw_p if not np.isnan(p)]
    corrected_finite = _holm_correct(finite_p)
    corrected_p: list[float] = []
    fi = 0
    for is_nan in nan_mask:
        if is_nan:
            corrected_p.append(float("nan"))
        else:
            corrected_p.append(corrected_finite[fi])
            fi += 1

    non_static_corrected = {m: corrected_p[i] for i, m in enumerate(NON_STATIC_METHODS)}

    # --- Build rows -----------------------------------------------------------
    rows = []
    for m in METHODS_MAIN:
        scores = list(method_scores[m].values())
        if scores:
            mean_h   = float(np.mean(scores))
            std_h    = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
            median_h = float(np.median(scores))
        else:
            mean_h = std_h = median_h = float("nan")

        if m == "STATIC":
            wilcoxon_p = float("nan")
            a12        = float("nan")
        else:
            wilcoxon_p = non_static_corrected.get(m, float("nan"))
            m_by_seed  = method_scores[m]
            shared     = sorted(set(m_by_seed) & set(static_by_seed))
            x = [m_by_seed[s] for s in shared]
            y = [static_by_seed[s] for s in shared]
            a12 = _a12(x, y)

        rows.append({
            "method":               m,
            "mean_heldout":         mean_h,
            "std_heldout":          std_h,
            "median_heldout":       median_h,
            "wilcoxon_p_vs_STATIC": wilcoxon_p,
            "A12_vs_STATIC":        a12,
        })

    return pd.DataFrame(rows)


# ── Convergence plot ──────────────────────────────────────────────────────────

def plot_convergence(runs: list[dict], out_path: str) -> None:
    """
    Save results/convergence.png.

    One line per GA method (STATIC, COEVOLVE_FROZEN, COEVOLVE).
    X-axis: generation.  Y-axis: best training fitness.
    Shaded band = 95 % CI across seeds ( mean ± 1.96 * SE ).
    Uses main-experiment runs: seeds 1000–1014, lambda=1.0.
    most_played_baseline has no history and is omitted.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for m in ["STATIC", "COEVOLVE_FROZEN", "COEVOLVE"]:
        m_runs = [
            r for r in _select(runs, mode=m, seeds=MAIN_SEEDS, lambda_weight=MAIN_LAMBDA)
            if r["history"]
        ]
        if not m_runs:
            continue

        n_gens = len(m_runs[0]["history"])
        matrix = np.array([
            [h["best_training_fitness"] for h in r["history"]]
            for r in m_runs
            if len(r["history"]) == n_gens
        ])
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            continue

        gens   = np.arange(n_gens)
        mean_f = np.mean(matrix, axis=0)
        n      = matrix.shape[0]
        se     = np.std(matrix, axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(n_gens)
        ci     = 1.96 * se

        color = _CONVERGENCE_COLORS.get(m, "black")
        ax.plot(gens, mean_f, label=m, color=color)
        ax.fill_between(gens, mean_f - ci, mean_f + ci, alpha=0.2, color=color)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Training Fitness")
    ax.set_title("Convergence (95 % CI across seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Sensitivity table ─────────────────────────────────────────────────────────

def compute_sensitivity_table(runs: list[dict]) -> pd.DataFrame:
    """
    Compute results/sensitivity_table.csv.

    One row per (method, lambda_weight) combination.
    Columns: method, lambda_weight, mean_heldout, std_heldout.
    Uses sensitivity-experiment runs: seeds 2000–2004.
    """
    rows = []
    for m in ["STATIC", "COEVOLVE"]:
        for lam in SENSITIVITY_LAMBDAS:
            subset = _select(runs, mode=m, seeds=SENS_SEEDS, lambda_weight=lam)
            scores = [r["heldout_score"] for r in subset]
            if scores:
                mean_h = float(np.mean(scores))
                std_h  = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
            else:
                mean_h = std_h = float("nan")
            rows.append({
                "method":        m,
                "lambda_weight": lam,
                "mean_heldout":  mean_h,
                "std_heldout":   std_h,
            })

    return pd.DataFrame(rows)


# ── Diagnostic table ──────────────────────────────────────────────────────────

def compute_diagnostic_table(runs: list[dict]) -> pd.DataFrame:
    """
    Compute results/diagnostic_table.csv.

    COEVOLVE main-experiment runs (seeds 1000–1014, lambda=1.0) only.
    Per-generation averages across seeds for:
        repertoire_diversity, opponent_diversity, hof_size.
    """
    coevolve_runs = [
        r for r in _select(
            runs, mode="COEVOLVE", seeds=MAIN_SEEDS, lambda_weight=MAIN_LAMBDA
        )
        if r["history"]
    ]

    if not coevolve_runs:
        return pd.DataFrame(
            columns=["gen", "repertoire_diversity", "opponent_diversity", "hof_size"]
        )

    n_gens = len(coevolve_runs[0]["history"])
    rows = []
    for g in range(n_gens):
        rep_divs:  list[float] = []
        opp_divs:  list[float] = []
        hof_sizes: list[float] = []
        for r in coevolve_runs:
            if g >= len(r["history"]):
                continue
            h = r["history"][g]
            rep_divs.append(h["repertoire_diversity"])
            if h["opponent_diversity"] is not None:
                opp_divs.append(h["opponent_diversity"])
            hof_sizes.append(float(h["hof_size"]))
        rows.append({
            "gen":                  g,
            "repertoire_diversity": float(np.mean(rep_divs)) if rep_divs else float("nan"),
            "opponent_diversity":   float(np.mean(opp_divs)) if opp_divs else float("nan"),
            "hof_size":             float(np.mean(hof_sizes)) if hof_sizes else float("nan"),
        })

    return pd.DataFrame(rows)


# ── Top-level entry point ─────────────────────────────────────────────────────

def run_analysis(runs_dir: str = "runs", results_dir: str = "results") -> None:
    """
    Load all runs and write every output artefact to *results_dir*.

    Calling this function twice with the same run data produces bit-identical
    CSV files (all operations are deterministic).
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading runs from '{runs_dir}' ...")
    runs = load_runs(runs_dir)
    print(f"  {len(runs)} run(s) loaded.\n")

    # 1. Main table
    print("Computing main_table.csv ...")
    main_df = compute_main_table(runs)
    main_df.to_csv(os.path.join(results_dir, "main_table.csv"), index=False)
    print(main_df.to_string(index=False))

    # 2. Convergence plot
    print("\nPlotting convergence.png ...")
    plot_convergence(runs, os.path.join(results_dir, "convergence.png"))
    print("  saved.")

    # 3. Sensitivity table
    print("\nComputing sensitivity_table.csv ...")
    sens_df = compute_sensitivity_table(runs)
    sens_df.to_csv(os.path.join(results_dir, "sensitivity_table.csv"), index=False)
    print(sens_df.to_string(index=False))

    # 4. Diagnostic table
    print("\nComputing diagnostic_table.csv ...")
    diag_df = compute_diagnostic_table(runs)
    diag_df.to_csv(os.path.join(results_dir, "diagnostic_table.csv"), index=False)
    print(diag_df.to_string(index=False))

    print(f"\nDone — results saved to '{results_dir}/'.")


if __name__ == "__main__":
    run_analysis()
