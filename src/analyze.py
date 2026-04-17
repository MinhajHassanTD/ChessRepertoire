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

from src.fitness import BANDS, walk


# ── Constants ─────────────────────────────────────────────────────────────────

METHODS_MAIN = [
    "most_played_baseline",
    "STATIC",
    "STATIC_DIV",
    "COEVOLVE_FROZEN",
    "COEVOLVE",
    "COEVOLVE_B",
    "COEVOLVE_C",
    "COEVOLVE_D",
]
NON_STATIC_METHODS = [
    "most_played_baseline",
    "STATIC_DIV",
    "COEVOLVE_FROZEN",
    "COEVOLVE",
    "COEVOLVE_B",
    "COEVOLVE_C",
    "COEVOLVE_D",
]
MAIN_SEEDS = range(1000, 1015)   # 15 seeds
SENS_SEEDS = range(2000, 2005)   # 5 seeds
MAIN_LAMBDA = 1.0
SENSITIVITY_LAMBDAS = [0.0, 1.0, 2.0]

# Metrics reported in the main table (Path A multi-metric robustness view)
HELDOUT_METRICS = [
    "heldout_uniform_mean",              # mean score under uniform mixture (λ-free)
    "heldout_worst_band",                # min over 3 bands (pure robustness / CVaR α=1/3)
    "heldout_adversarial_q10",           # 10th-pct mean score over 200 Dirichlet mixtures
    "heldout_uniform_fitness",           # mean + λ·cvar under uniform (legacy heldout_score)
    "heldout_adversarial_coevolved",     # testv1: worst fitness found by a small adversarial GA on held-out
]

# Seed for the Dirichlet samples used in adversarial metric — fixed so the
# analysis is deterministic across re-runs.
_ADVERSARIAL_RNG_SEED = 7
_ADVERSARIAL_N_SAMPLES = 200
_ADVERSARIAL_DIRICHLET_ALPHA = 1.0    # symmetric Dirichlet(1,1,1) = uniform over simplex
_ADVERSARIAL_QUANTILE = 0.10          # lower tail: pessimistic mixture realization

_CONVERGENCE_COLORS = {
    "STATIC": "steelblue",
    "STATIC_DIV": "teal",
    "COEVOLVE_FROZEN": "darkorange",
    "COEVOLVE": "forestgreen",
    "COEVOLVE_B": "crimson",
    "COEVOLVE_C": "mediumorchid",
    "COEVOLVE_D": "saddlebrown",
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
        if mode is not None and r.get("mode") != mode:
            continue
        if seeds is not None and r.get("seed") not in seeds:
            continue
        if lambda_weight is not None:
            if r.get("config", {}).get("lambda_weight") != lambda_weight:
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


# ── Held-out metric recomputation ─────────────────────────────────────────────

class _FrozenRep:
    """Thin stand-in for a Repertoire when walking the held-out graph.

    Reconstructs committed / reached from the serialised candidate; walk() only
    reads color, committed, reached (graph is passed explicitly).
    """
    __slots__ = ("color", "committed", "reached")

    def __init__(self, color: str, committed: dict, reached):
        self.color = color
        self.committed = committed
        self.reached = set(reached)


def _band_scores_heldout(
    candidate_ser: dict,
    eval_cache_heldout: dict,
    base_policies_train: dict,
    graph_heldout: dict,
) -> dict[str, float]:
    """Compute per-band held-out scores (player-perspective mean of white/black).

    Mirrors the logic in fitness.evaluate(...) but reads directly from the
    serialised best candidate so we don't need the original Candidate object.
    """
    white = _FrozenRep("white", candidate_ser["white_committed"], candidate_ser["white_reached"])
    black = _FrozenRep("black", candidate_ser["black_committed"], candidate_ser["black_reached"])

    band_scores: dict[str, float] = {}
    for band in BANDS:
        white_ws = walk(white, band, eval_cache_heldout, base_policies_train, graph_heldout)
        black_ws = walk(black, band, eval_cache_heldout, base_policies_train, graph_heldout)
        band_scores[band] = 0.5 * white_ws + 0.5 * (1.0 - black_ws)
    return band_scores


def _compute_heldout_metrics(
    candidate_ser: dict,
    eval_cache_heldout: dict,
    base_policies_train: dict,
    graph_heldout: dict,
    lambda_weight: float,
) -> dict[str, float]:
    """Return the four held-out metrics for one serialised best candidate.

    - heldout_uniform_mean:   mean band score under uniform mixture (1/3,1/3,1/3)
    - heldout_worst_band:     min over the 3 band scores (α=1/3 CVaR)
    - heldout_adversarial_q10: 10th-pct of mean score over 200 Dirichlet(1,1,1) mixtures
    - heldout_uniform_fitness: mean + λ·cvar under uniform (matches old heldout_score)
    """
    bs = _band_scores_heldout(candidate_ser, eval_cache_heldout, base_policies_train, graph_heldout)
    band_vec = np.array([bs[b] for b in BANDS])

    uniform_mean = float(band_vec.mean())
    worst_band   = float(band_vec.min())

    rng = np.random.default_rng(_ADVERSARIAL_RNG_SEED)
    mixtures = rng.dirichlet(np.full(3, _ADVERSARIAL_DIRICHLET_ALPHA), size=_ADVERSARIAL_N_SAMPLES)
    sampled_means = mixtures @ band_vec
    adversarial_q10 = float(np.quantile(sampled_means, _ADVERSARIAL_QUANTILE))

    uniform_fitness = uniform_mean + lambda_weight * worst_band

    return {
        "heldout_uniform_mean":     uniform_mean,
        "heldout_worst_band":       worst_band,
        "heldout_adversarial_q10":  adversarial_q10,
        "heldout_uniform_fitness":  uniform_fitness,
    }
    # Note: heldout_adversarial_coevolved is produced by run_coevolution itself
    # and surfaced through _scores_by_seed_metric below.


def _load_heldout_artifacts(data_dir: str = "data"):
    """Load graph_heldout, eval_cache_heldout, base_policies_train from pickles."""
    with open(Path(data_dir) / "graph_heldout.pkl", "rb") as fh:
        graph_heldout = pickle.load(fh)
    with open(Path(data_dir) / "eval_cache_heldout.pkl", "rb") as fh:
        eval_cache_heldout = pickle.load(fh)
    with open(Path(data_dir) / "base_policies.pkl", "rb") as fh:
        base_policies_train = pickle.load(fh)
    return graph_heldout, eval_cache_heldout, base_policies_train


def _augment_runs_with_metrics(runs: list[dict], data_dir: str = "data") -> None:
    """In-place: attach a 'heldout_metrics' dict to every run that has a
    serialised best candidate.  Idempotent (skips runs already augmented)."""
    if all("heldout_metrics" in r for r in runs):
        return
    graph_heldout, eval_cache_heldout, base_policies_train = _load_heldout_artifacts(data_dir)
    for r in runs:
        if "heldout_metrics" in r:
            continue
        if "final_best_candidate" not in r:
            continue
        lam = float(r["config"].get("lambda_weight", MAIN_LAMBDA))
        r["heldout_metrics"] = _compute_heldout_metrics(
            r["final_best_candidate"],
            eval_cache_heldout,
            base_policies_train,
            graph_heldout,
            lam,
        )


def _scores_by_seed_metric(runs: list[dict], metric: str) -> dict[int, float]:
    """{seed: metric_value} for a pre-filtered run list.  Falls back to the
    legacy scalar heldout_score if metric == 'heldout_uniform_fitness' and no
    heldout_metrics dict is present (should not happen after augmentation).

    The testv1 metric `heldout_adversarial_coevolved` is stored at the top
    level of the run dict (produced by run_coevolution), so we read it directly.
    Runs that predate the metric are silently skipped.
    """
    out: dict[int, float] = {}
    for r in runs:
        if metric == "heldout_adversarial_coevolved":
            if "heldout_adversarial_coevolved" in r:
                out[r["seed"]] = r["heldout_adversarial_coevolved"]
            continue
        hm = r.get("heldout_metrics")
        if hm is not None and metric in hm:
            out[r["seed"]] = hm[metric]
        elif metric == "heldout_uniform_fitness":
            out[r["seed"]] = r["heldout_score"]
    return out


# ── Main table ────────────────────────────────────────────────────────────────

def compute_main_table(runs: list[dict]) -> pd.DataFrame:
    """
    Compute results/main_table.csv.

    One row per (metric, method), covering every metric in HELDOUT_METRICS.
    Columns: metric, method, mean, std, median, wilcoxon_p_vs_STATIC, A12_vs_STATIC.

    Uses main-experiment runs: seeds 1000–1014, lambda=1.0.
    For each metric independently:
      - paired Wilcoxon vs STATIC is computed across shared seeds
      - p-values are Holm-corrected across the three non-STATIC methods
      - A12 effect size is computed against STATIC
    """
    filtered_by_method: dict[str, list[dict]] = {
        m: _select(runs, mode=m, seeds=MAIN_SEEDS, lambda_weight=MAIN_LAMBDA)
        for m in METHODS_MAIN
    }

    all_rows: list[dict] = []

    for metric in HELDOUT_METRICS:
        method_scores: dict[str, dict[int, float]] = {
            m: _scores_by_seed_metric(filtered_by_method[m], metric)
            for m in METHODS_MAIN
        }
        static_by_seed = method_scores["STATIC"]

        # Raw Wilcoxon p-values vs STATIC for each non-STATIC method
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

        # Holm correction skipping NaNs
        nan_mask = [np.isnan(p) for p in raw_p]
        finite_p = [p for p in raw_p if not np.isnan(p)]
        corrected_finite = _holm_correct(finite_p)
        corrected_p: list[float] = []
        fi = 0
        for is_nan in nan_mask:
            if is_nan:
                corrected_p.append(float("nan"))
            else:
                corrected_p.append(corrected_finite[fi]); fi += 1
        non_static_corrected = {m: corrected_p[i] for i, m in enumerate(NON_STATIC_METHODS)}

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

            all_rows.append({
                "metric":               metric,
                "method":               m,
                "mean":                 mean_h,
                "std":                  std_h,
                "median":               median_h,
                "wilcoxon_p_vs_STATIC": wilcoxon_p,
                "A12_vs_STATIC":        a12,
            })

    return pd.DataFrame(all_rows)


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

    for m in ["STATIC", "COEVOLVE_FROZEN", "COEVOLVE", "COEVOLVE_B", "COEVOLVE_C", "COEVOLVE_D"]:
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

    One row per (metric, method, lambda_weight) combination.
    Columns: metric, method, lambda_weight, mean, std.
    Uses sensitivity-experiment runs: seeds 2000–2004.
    """
    rows = []
    for metric in HELDOUT_METRICS:
        for m in ["STATIC", "COEVOLVE"]:
            for lam in SENSITIVITY_LAMBDAS:
                subset = _select(runs, mode=m, seeds=SENS_SEEDS, lambda_weight=lam)
                scores = list(_scores_by_seed_metric(subset, metric).values())
                if scores:
                    mean_h = float(np.mean(scores))
                    std_h  = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
                else:
                    mean_h = std_h = float("nan")
                rows.append({
                    "metric":        metric,
                    "method":        m,
                    "lambda_weight": lam,
                    "mean":          mean_h,
                    "std":           std_h,
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
    rows = []
    for variant in ["COEVOLVE", "COEVOLVE_B", "COEVOLVE_C", "COEVOLVE_D"]:
        variant_runs = [
            r for r in _select(
                runs, mode=variant, seeds=MAIN_SEEDS, lambda_weight=MAIN_LAMBDA
            )
            if r["history"]
        ]
        if not variant_runs:
            continue
        n_gens = len(variant_runs[0]["history"])
        for g in range(n_gens):
            rep_divs:  list[float] = []
            opp_divs:  list[float] = []
            hof_sizes: list[float] = []
            hof_informs: list[float] = []
            opp_perts:   list[float] = []
            for r in variant_runs:
                if g >= len(r["history"]):
                    continue
                h = r["history"][g]
                rep_divs.append(h["repertoire_diversity"])
                if h["opponent_diversity"] is not None:
                    opp_divs.append(h["opponent_diversity"])
                hof_sizes.append(float(h["hof_size"]))
                if h.get("hof_mean_informativeness") is not None:
                    hof_informs.append(float(h["hof_mean_informativeness"]))
                if h.get("opp_mean_perturbations") is not None:
                    opp_perts.append(float(h["opp_mean_perturbations"]))
            rows.append({
                "variant":                 variant,
                "gen":                     g,
                "repertoire_diversity":    float(np.mean(rep_divs)) if rep_divs else float("nan"),
                "opponent_diversity":      float(np.mean(opp_divs)) if opp_divs else float("nan"),
                "hof_size":                float(np.mean(hof_sizes)) if hof_sizes else float("nan"),
                "hof_mean_informativeness": float(np.mean(hof_informs)) if hof_informs else float("nan"),
                "opp_mean_perturbations":   float(np.mean(opp_perts)) if opp_perts else float("nan"),
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

    print("Recomputing held-out metrics (uniform mean / worst band / adversarial-q10) ...")
    _augment_runs_with_metrics(runs)
    print("  done.\n")

    # 1. Main table
    print("Computing main_table.csv ...")
    main_df = compute_main_table(runs)
    main_df.to_csv(os.path.join(results_dir, "main_table.csv"), index=False)
    # Print one metric block at a time for readability
    for metric, sub in main_df.groupby("metric", sort=False):
        print(f"\n[{metric}]")
        print(sub.drop(columns=["metric"]).to_string(index=False))

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
