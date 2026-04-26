"""
Analysis: load all run pickles, compute statistics, and produce tables + plots.

Outputs (all written to results/):
    main_table.csv              — one row per (metric, method): mean/std, Wilcoxon p,
                                  A12 effect size (all vs STATIC as reference)
    convergence.png             — GA training curves with 95% CI across seeds
    score_distributions.png     — box + strip for both held-out metrics, all methods
    band_breakdown.png          — per-rating-band score grouped by method
    closure_ablation.png        — closure ON vs OFF for STATIC and COEVOLVE
    white_black_breakdown.png   — white score vs black score strip plots per method
    ga_vs_nonga.png             — Non-GA / GA groups separated and sorted
    repertoire_structure.png    — committed moves + subgraph size per method
    coevolve_dynamics.png       — COEVOLVE internal dynamics (3-panel)
    diagnostic_table.csv        — COEVOLVE per-generation averages
    repertoire_tree.txt         — best candidate per GA mode as a decision tree
    repertoire_grouped.txt      — best candidate grouped by opening family
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.fitness import BANDS, walk
from src.config import (
    MAIN_SEEDS,
    CLOSURE_ABLATION_SEEDS,
    MAIN_LAMBDA,
    BUDGET,
)


# ── Constants ─────────────────────────────────────────────────────────────────

METHODS_MAIN = [
    "most_played_baseline",  # human heuristic baseline
    "RANDOM_SEARCH",         # random search with same eval budget as GA
    "GREEDY_HILLCLIMB",      # greedy hill-climb with same eval budget as GA
    "STATIC",                # GA with fixed uniform opponent
    "COEVOLVE",              # GA with co-evolving opponent population
]

HELDOUT_METRICS = [
    "heldout_uniform_mean",   # average win-rate across all 3 bands under uniform opponent
    "heldout_worst_band",     # win-rate against the hardest band (robustness / CVaR)
]

# ── Display constants ─────────────────────────────────────────────────────────

# Wong colorblind-safe palette — works in greyscale and for common colour-blindness types.
_METHOD_COLORS = {
    "most_played_baseline":   "#999999",  # grey
    "RANDOM_SEARCH":          "#E69F00",  # orange
    "GREEDY_HILLCLIMB":       "#56B4E9",  # sky blue
    "STATIC":                 "#009E73",  # green
    "COEVOLVE":               "#0072B2",  # blue
    # No-closure ablation variants get lighter versions of their parent's colour.
    "STATIC_NOCLOSURE":       "#66C2A5",  # light green
    "COEVOLVE_NOCLOSURE":     "#80B1D3",  # light blue
}

_METHOD_LABELS = {
    "most_played_baseline":   "Most Played",
    "RANDOM_SEARCH":          "Rand. Search",
    "GREEDY_HILLCLIMB":       "Hill-climb",
    "STATIC":                 "Static GA",
    "COEVOLVE":               "CoEvolve",
    "STATIC_NOCLOSURE":       "Static (no closure)",
    "COEVOLVE_NOCLOSURE":     "CoEvolve (no closure)",
}

_BAND_COLORS = ["#d62728", "#ff7f0e", "#1f77b4"]   # red, orange, blue per band
_BAND_LABELS = {
    "1000-1399": "1000–1399",
    "1400-1799": "1400–1799",
    "1800-2199": "1800–2199",
}

# GA methods shown in convergence curves (only those with generation history).
GA_METHODS_CONV = ["STATIC", "COEVOLVE"]


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_runs(runs_dir: str) -> list[dict]:
    runs: list[dict] = []
    for path in sorted(Path(runs_dir).glob("*.pkl")):
        with open(path, "rb") as fh:
            runs.append(pickle.load(fh))
    return runs


# ── Filtering ─────────────────────────────────────────────────────────────────

def _select(
    runs: list[dict],
    *,
    mode: Optional[str] = None,
    seeds: Optional[range] = None,
    lambda_weight: Optional[float] = None,
) -> list[dict]:
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


# ── Statistics helpers ────────────────────────────────────────────────────────

def _a12(x: list[float], y: list[float]) -> float:
    """Vargha-Delaney A12 effect size. >0.5 means X tends larger than Y."""
    n = len(x) * len(y)
    if n == 0:
        return float("nan")
    wins = sum(1 for xi in x for yj in y if xi > yj)
    ties = sum(1 for xi in x for yj in y if xi == yj)
    return (wins + 0.5 * ties) / n


def _holm_correct(p_values: list[float]) -> list[float]:
    k = len(p_values)
    if k == 0:
        return []
    order = sorted(range(k), key=lambda i: p_values[i])
    corrected: list[Optional[float]] = [None] * k
    for rank, idx in enumerate(order):
        multiplier = k - rank
        corrected[idx] = min(1.0, p_values[idx] * multiplier)
    return corrected  # type: ignore[return-value]


# ── Held-out metric recomputation ─────────────────────────────────────────────


class _FrozenRep:
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
) -> dict:
    white = _FrozenRep("white", candidate_ser["white_committed"], candidate_ser["white_reached"])
    black = _FrozenRep("black", candidate_ser["black_committed"], candidate_ser["black_reached"])

    band_scores, white_band_scores, black_band_scores = {}, {}, {}
    for band in BANDS:
        w_ws = walk(white, band, eval_cache_heldout, base_policies_train, graph_heldout)
        b_ws = walk(black, band, eval_cache_heldout, base_policies_train, graph_heldout)
        b_score = 1.0 - b_ws
        white_band_scores[band] = float(w_ws)
        black_band_scores[band] = float(b_score)
        band_scores[band]       = 0.5 * w_ws + 0.5 * b_score

    band_vec = np.array([band_scores[b] for b in BANDS])
    return {
        "heldout_uniform_mean": float(band_vec.mean()),
        "heldout_worst_band":   float(band_vec.min()),
        "band_scores":          {b: float(band_scores[b])       for b in BANDS},
        "white_band_scores":    white_band_scores,
        "black_band_scores":    black_band_scores,
        "white_mean":           float(np.mean([white_band_scores[b] for b in BANDS])),
        "black_mean":           float(np.mean([black_band_scores[b] for b in BANDS])),
    }


def _load_heldout_artifacts(data_dir: str = "data"):
    with open(Path(data_dir) / "graph_heldout.pkl", "rb") as fh:
        graph_heldout = pickle.load(fh)
    with open(Path(data_dir) / "eval_cache_heldout.pkl", "rb") as fh:
        eval_cache_heldout = pickle.load(fh)
    with open(Path(data_dir) / "base_policies.pkl", "rb") as fh:
        base_policies_train = pickle.load(fh)
    return graph_heldout, eval_cache_heldout, base_policies_train


def _augment_runs_with_metrics(runs: list[dict], data_dir: str = "data") -> None:
    # Recompute for any run that is missing heldout_metrics or the per-band breakdown.
    needs = [
        r for r in runs
        if "final_best_candidate" in r
        and "white_band_scores" not in r.get("heldout_metrics", {})
    ]
    if not needs:
        return
    graph_heldout, eval_cache_heldout, base_policies_train = _load_heldout_artifacts(data_dir)
    for r in needs:
        lam = float(r["config"].get("lambda_weight", MAIN_LAMBDA))
        r["heldout_metrics"] = _compute_heldout_metrics(
            r["final_best_candidate"],
            eval_cache_heldout,
            base_policies_train,
            graph_heldout,
            lam,
        )


def _scores_by_seed_metric(runs: list[dict], metric: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for r in runs:
        hm = r.get("heldout_metrics")
        if hm is not None and metric in hm:
            out[r["seed"]] = hm[metric]
        elif metric == "heldout_uniform_fitness":
            out[r["seed"]] = r["heldout_score"]
    return out


# ── Main table ────────────────────────────────────────────────────────────────

def compute_summary_table(runs: list[dict]) -> pd.DataFrame:
    """Mean, std, median per method per metric — no statistical tests."""
    filtered = {
        m: _select(runs, mode=m, seeds=MAIN_SEEDS, lambda_weight=MAIN_LAMBDA)
        for m in METHODS_MAIN
    }
    rows: list[dict] = []
    for metric in HELDOUT_METRICS:
        for m in METHODS_MAIN:
            scores = list(_scores_by_seed_metric(filtered[m], metric).values())
            rows.append({
                "metric": metric,
                "method": m,
                "n":      len(scores),
                "mean":   float(np.mean(scores))            if scores else float("nan"),
                "std":    float(np.std(scores, ddof=1))     if len(scores) > 1 else 0.0,
                "median": float(np.median(scores))          if scores else float("nan"),
            })
    return pd.DataFrame(rows)


def _wilcoxon_safe(x: list[float], y: list[float]) -> float:
    """Paired Wilcoxon p-value; returns 1.0 on degenerate input."""
    if len(x) < 2:
        return float("nan")
    diffs = [xi - yi for xi, yi in zip(x, y)]
    if all(d == 0.0 for d in diffs):
        return 1.0
    try:
        _, pval = stats.wilcoxon(x, y, zero_method="wilcox")
        return float(pval)
    except ValueError:
        return 1.0


def compute_pairwise_table(runs: list[dict]) -> pd.DataFrame:
    """All-pairs Wilcoxon signed-rank + A12 with Holm correction per metric.

    Every ordered pair (method_a, method_b) where method_a appears before
    method_b in METHODS_MAIN.  A12 > 0.5 means method_a tends higher.
    """
    filtered = {
        m: _select(runs, mode=m, seeds=MAIN_SEEDS, lambda_weight=MAIN_LAMBDA)
        for m in METHODS_MAIN
    }
    pairs = [
        (METHODS_MAIN[i], METHODS_MAIN[j])
        for i in range(len(METHODS_MAIN))
        for j in range(i + 1, len(METHODS_MAIN))
    ]

    rows: list[dict] = []
    for metric in HELDOUT_METRICS:
        method_scores = {
            m: _scores_by_seed_metric(filtered[m], metric) for m in METHODS_MAIN
        }

        # Collect raw p-values for all pairs in this metric
        pair_data: list[tuple] = []
        raw_p: list[float] = []
        for a, b in pairs:
            shared = sorted(set(method_scores[a]) & set(method_scores[b]))
            if len(shared) < 2:
                raw_p.append(float("nan"))
                pair_data.append((a, b, [], []))
                continue
            x = [method_scores[a][s] for s in shared]
            y = [method_scores[b][s] for s in shared]
            raw_p.append(_wilcoxon_safe(x, y))
            pair_data.append((a, b, x, y))

        # Holm-correct all finite p-values for this metric together
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

        for (a, b, x, y), cp in zip(pair_data, corrected_p):
            mean_a = float(np.mean(x)) if x else float("nan")
            mean_b = float(np.mean(y)) if y else float("nan")
            rows.append({
                "metric":          metric,
                "method_a":        a,
                "method_b":        b,
                "mean_a":          mean_a,
                "mean_b":          mean_b,
                "delta":           mean_a - mean_b if x and y else float("nan"),
                "wilcoxon_p":      cp,
                "A12":             _a12(x, y) if x and y else float("nan"),
                "significant_p05": bool(not np.isnan(cp) and cp < 0.05),
            })

    return pd.DataFrame(rows)


def compute_closure_ablation_table(runs: list[dict]) -> pd.DataFrame:
    """Paired Wilcoxon + A12 for closure ON vs OFF (Holm-corrected across all tests).

    Uses CLOSURE_ABLATION_SEEDS (2000–2014), not MAIN_SEEDS.
    A12 > 0.5 means the with-closure method tends higher.
    """
    pairs = [
        ("STATIC",   "STATIC_NOCLOSURE"),
        ("COEVOLVE", "COEVOLVE_NOCLOSURE"),
    ]

    # Gather all (metric, pair) combos together for a single Holm correction pass
    all_raw_p: list[float] = []
    all_info:  list[tuple] = []

    for metric in HELDOUT_METRICS:
        for with_m, without_m in pairs:
            with_scores    = _scores_by_seed_metric(
                _select(runs, mode=with_m,    seeds=CLOSURE_ABLATION_SEEDS, lambda_weight=MAIN_LAMBDA),
                metric,
            )
            without_scores = _scores_by_seed_metric(
                _select(runs, mode=without_m, seeds=CLOSURE_ABLATION_SEEDS, lambda_weight=MAIN_LAMBDA),
                metric,
            )
            shared = sorted(set(with_scores) & set(without_scores))
            if len(shared) < 2:
                all_raw_p.append(float("nan"))
                all_info.append((metric, with_m, without_m, [], []))
                continue
            x = [with_scores[s]    for s in shared]
            y = [without_scores[s] for s in shared]
            all_raw_p.append(_wilcoxon_safe(x, y))
            all_info.append((metric, with_m, without_m, x, y))

    nan_mask = [np.isnan(p) for p in all_raw_p]
    finite_p = [p for p in all_raw_p if not np.isnan(p)]
    corrected_finite = _holm_correct(finite_p)
    corrected_p2: list[float] = []
    fi = 0
    for is_nan in nan_mask:
        if is_nan:
            corrected_p2.append(float("nan"))
        else:
            corrected_p2.append(corrected_finite[fi]); fi += 1

    rows: list[dict] = []
    for (metric, with_m, without_m, x, y), cp in zip(all_info, corrected_p2):
        mean_with    = float(np.mean(x)) if x else float("nan")
        mean_without = float(np.mean(y)) if y else float("nan")
        rows.append({
            "metric":          metric,
            "method_with":     with_m,
            "method_without":  without_m,
            "mean_with":       mean_with,
            "mean_without":    mean_without,
            "delta":           mean_with - mean_without if x and y else float("nan"),
            "wilcoxon_p":      cp,
            "A12":             _a12(x, y) if x and y else float("nan"),
            "significant_p05": bool(not np.isnan(cp) and cp < 0.05),
        })

    return pd.DataFrame(rows)


# ── Convergence plot ──────────────────────────────────────────────────────────

def plot_convergence(runs: list[dict], out_path: str) -> None:
    """GA methods only — clean convergence curves with 95% CI bands."""
    fig, ax = plt.subplots(figsize=(7, 4))

    for m in GA_METHODS_CONV:
        selected = _select(runs, mode=m, seeds=MAIN_SEEDS, lambda_weight=MAIN_LAMBDA)
        history_runs = [r for r in selected if r.get("history")]
        if not history_runs:
            continue
        n_gens = len(history_runs[0]["history"])
        matrix = np.array([
            [h["best_training_fitness"] for h in r["history"]]
            for r in history_runs
            if len(r["history"]) == n_gens
        ])
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            continue
        gens  = np.arange(n_gens)
        mean_f = np.mean(matrix, axis=0)
        n     = matrix.shape[0]
        se    = np.std(matrix, axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(n_gens)
        ci    = 1.96 * se
        color = _METHOD_COLORS[m]
        ax.plot(gens, mean_f, label=_METHOD_LABELS[m], color=color, linewidth=2)
        ax.fill_between(gens, mean_f - ci, mean_f + ci, alpha=0.15, color=color)

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Best Training Fitness", fontsize=12)
    ax.set_title("Training Convergence — GA Methods  (95% CI across seeds)", fontsize=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_score_distributions(runs: list[dict], out_path: str) -> None:
    """Box + individual-seed strip plots for the two held-out metrics, side by side."""
    metrics_info = [
        ("heldout_uniform_mean", "Mean Score\n(uniform opponent)",  "(a)"),
        ("heldout_worst_band",   "Worst-Band Score\n(robustness)", "(b)"),
    ]
    filtered = {
        m: _select(runs, mode=m, seeds=MAIN_SEEDS, lambda_weight=MAIN_LAMBDA)
        for m in METHODS_MAIN
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    rng_j = np.random.default_rng(0)

    for ax, (metric, ylabel, panel) in zip(axes, metrics_info):
        all_data, positions, colors, labels = [], [], [], []
        for i, m in enumerate(METHODS_MAIN):
            vals = list(_scores_by_seed_metric(filtered[m], metric).values())
            if not vals:
                continue
            all_data.append(vals)
            positions.append(i)
            colors.append(_METHOD_COLORS[m])
            labels.append(_METHOD_LABELS[m])

        bp = ax.boxplot(
            all_data,
            positions=positions,
            widths=0.45,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 2},
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.2},
            showfliers=False,
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.45)

        for pos, vals, c in zip(positions, all_data, colors):
            jitter = rng_j.uniform(-0.15, 0.15, len(vals))
            ax.scatter(
                [pos + j for j in jitter], vals,
                color=c, s=28, alpha=0.85, zorder=4,
                edgecolors="white", linewidths=0.4,
            )

        ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.35)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9.5)
        ax.set_ylabel(ylabel, fontsize=10.5)
        ax.set_title(f"{panel} {ylabel.replace(chr(10), ' ')}", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Held-out Score Distributions Across Seeds  (dashed = random play 0.5)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_band_breakdown(runs: list[dict], out_path: str) -> None:
    """Grouped bar chart: held-out score per rating band, per method."""
    bands = list(BANDS)
    filtered = {
        m: _select(runs, mode=m, seeds=MAIN_SEEDS, lambda_weight=MAIN_LAMBDA)
        for m in METHODS_MAIN
    }

    n_m = len(METHODS_MAIN)
    n_b = len(bands)
    means = np.full((n_m, n_b), np.nan)
    stds  = np.full((n_m, n_b), np.nan)

    for mi, m in enumerate(METHODS_MAIN):
        for bi, band in enumerate(bands):
            vals = [
                r["heldout_metrics"]["band_scores"][band]
                for r in filtered[m]
                if "band_scores" in r.get("heldout_metrics", {})
            ]
            if vals:
                means[mi, bi] = np.mean(vals)
                stds[mi, bi]  = np.std(vals, ddof=1) if len(vals) > 1 else 0.0

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x      = np.arange(n_m)
    width  = 0.25
    offsets = np.array([-width, 0.0, width])

    for bi, (band, color) in enumerate(zip(bands, _BAND_COLORS)):
        mask = ~np.isnan(means[:, bi])
        ax.bar(
            x[mask] + offsets[bi],
            means[mask, bi],
            width=width,
            yerr=stds[mask, bi],
            label=_BAND_LABELS[band],
            color=color,
            alpha=0.75,
            capsize=3,
            error_kw={"linewidth": 1.2},
        )

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.35)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_METHOD_LABELS[m] for m in METHODS_MAIN], rotation=25, ha="right", fontsize=10
    )
    ax.set_ylabel("Held-out Mean Score", fontsize=11)
    ax.set_title("Per-Band Performance by Method  (error bars = ±1 SD)", fontsize=12)
    ax.legend(title="Rating Band", fontsize=10, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    all_vals = means[~np.isnan(means)]
    if len(all_vals):
        ax.set_ylim(max(0.0, all_vals.min() - 0.04), min(1.0, all_vals.max() + 0.04))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── White vs Black breakdown ──────────────────────────────────────────────────

def plot_white_black_breakdown(runs: list[dict], out_path: str) -> None:
    """Side-by-side strip plots: white score vs black score per method."""
    all_methods = ["most_played_baseline", "RANDOM_SEARCH", "GREEDY_HILLCLIMB", "STATIC", "COEVOLVE"]
    filtered = {
        m: _select(runs, mode=m, seeds=MAIN_SEEDS, lambda_weight=MAIN_LAMBDA)
        for m in all_methods
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle("White vs Black Repertoire Performance  (held-out, dashed = 0.5)", fontsize=13)

    for ax, color_key, title, ylabel in [
        (axes[0], "white_mean", "(a) Playing as White",  "Score Playing as White"),
        (axes[1], "black_mean", "(b) Playing as Black",  "Score Playing as Black"),
    ]:
        for xi, method in enumerate(all_methods):
            vals = [
                r["heldout_metrics"][color_key]
                for r in filtered[method]
                if color_key in r.get("heldout_metrics", {})
            ]
            if not vals:
                continue
            color = _METHOD_COLORS.get(method, "#333333")
            jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(
                [xi] * len(vals) + jitter,
                vals,
                color=color, alpha=0.75, s=30, zorder=3,
            )
            ax.plot([xi - 0.2, xi + 0.2], [np.mean(vals)] * 2,
                    color="black", linewidth=2.0, zorder=4)

        ax.axhline(0.5, color="grey", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_xticks(range(len(all_methods)))
        ax.set_xticklabels(
            [_METHOD_LABELS[m] for m in all_methods],
            rotation=25, ha="right", fontsize=10,
        )
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── GA vs Non-GA comparison ───────────────────────────────────────────────────

def plot_ga_vs_nonga(runs: list[dict], out_path: str) -> None:
    """Bar chart: methods sorted by score within Non-GA / GA groups."""
    from matplotlib.patches import Patch

    NGA = ["most_played_baseline", "RANDOM_SEARCH", "GREEDY_HILLCLIMB"]
    GA  = ["STATIC", "COEVOLVE"]
    metrics_info = [
        ("heldout_uniform_mean", "Mean Score (uniform opponent)"),
        ("heldout_worst_band",   "Worst-Band Score (robustness)"),
    ]
    filtered = {
        m: _select(runs, mode=m, seeds=MAIN_SEEDS, lambda_weight=MAIN_LAMBDA)
        for m in METHODS_MAIN
    }

    def _mean_std(m, metric):
        vals = list(_scores_by_seed_metric(filtered[m], metric).values())
        if not vals:
            return float("nan"), 0.0
        return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    GAP = 0.8

    for ax, (metric, ylabel) in zip(axes, metrics_info):
        nga_sorted = sorted(NGA, key=lambda m: _mean_std(m, metric)[0])
        ga_sorted  = sorted(GA,  key=lambda m: _mean_std(m, metric)[0])

        positions = {}
        for i, m in enumerate(nga_sorted):
            positions[m] = float(i)
        for i, m in enumerate(ga_sorted):
            positions[m] = float(len(NGA)) + GAP + float(i)

        for group in [nga_sorted, ga_sorted]:
            for m in group:
                mean, std = _mean_std(m, metric)
                if np.isnan(mean):
                    continue
                hatch = "//" if m in NGA else None
                ax.bar(positions[m], mean, width=0.62, color=_METHOD_COLORS[m],
                       alpha=0.82, hatch=hatch, edgecolor="black", linewidth=0.7)
                ax.errorbar(positions[m], mean, yerr=std, fmt="none",
                            color="black", capsize=3, linewidth=1.4)

        sep_x = len(NGA) - 0.5 + GAP / 2
        ax.axvline(sep_x, color="gray", linestyle=":", linewidth=1.3, alpha=0.7)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.35)

        all_pos = [positions[m] for m in nga_sorted + ga_sorted]
        all_lab = [_METHOD_LABELS[m] for m in nga_sorted + ga_sorted]
        ax.set_xticks(all_pos)
        ax.set_xticklabels(all_lab, rotation=30, ha="right", fontsize=9.5)
        ax.set_ylabel(ylabel, fontsize=10.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25)

        ymin, ymax = ax.get_ylim()
        ax.text(np.mean([positions[m] for m in nga_sorted]), ymax * 0.995,
                "Non-GA", ha="center", va="top", fontsize=8.5, color="gray", style="italic")
        ax.text(np.mean([positions[m] for m in ga_sorted]),  ymax * 0.995,
                "GA",     ha="center", va="top", fontsize=8.5, color="gray", style="italic")

    legend_handles = [
        Patch(facecolor="#cccccc", hatch="//", edgecolor="black", label="Non-GA baseline"),
        Patch(facecolor="#cccccc", edgecolor="black", label="GA method"),
    ]
    axes[0].legend(handles=legend_handles, fontsize=9, loc="lower right")
    fig.suptitle("GA vs Non-GA: Sorted by Score within Group  (error bars = ±1 SD)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Repertoire structure ───────────────────────────────────────────────────────

def plot_repertoire_structure(runs: list[dict], out_path: str) -> None:
    """Budget utilization and subgraph size per method."""
    filtered = {
        m: _select(runs, mode=m, seeds=MAIN_SEEDS, lambda_weight=MAIN_LAMBDA)
        for m in METHODS_MAIN
    }

    stats: dict[str, dict[str, list]] = {
        m: {"w_comm": [], "b_comm": [], "w_reach": [], "b_reach": []}
        for m in METHODS_MAIN
    }
    for m in METHODS_MAIN:
        for r in filtered[m]:
            cand = r.get("final_best_candidate")
            if cand is None:
                continue
            stats[m]["w_comm"].append(len(cand.get("white_committed", {})))
            stats[m]["b_comm"].append(len(cand.get("black_committed", {})))
            stats[m]["w_reach"].append(len(cand.get("white_reached", [])))
            stats[m]["b_reach"].append(len(cand.get("black_reached", [])))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x     = np.arange(len(METHODS_MAIN))
    width = 0.35
    labels_m = [_METHOD_LABELS[m] for m in METHODS_MAIN]

    for ax, key_w, key_b, ylabel, title, show_budget in [
        (axes[0], "w_comm",  "b_comm",  "Committed Moves",     "(a) Budget Utilization",  True),
        (axes[1], "w_reach", "b_reach", "Positions in Subgraph","(b) Subgraph Coverage",  False),
    ]:
        for offset, key, label, color, hatch in [
            (-width / 2, key_w, "White", "#2196F3", None),
            ( width / 2, key_b, "Black", "#F44336", "//"),
        ]:
            means = [float(np.mean(stats[m][key])) if stats[m][key] else 0.0 for m in METHODS_MAIN]
            stds  = [float(np.std(stats[m][key], ddof=1)) if len(stats[m][key]) > 1 else 0.0
                     for m in METHODS_MAIN]
            ax.bar(x + offset, means, width=width, yerr=stds, label=label,
                   color=color, alpha=0.75, capsize=3, hatch=hatch,
                   edgecolor="black", linewidth=0.5, error_kw={"linewidth": 1.2})
        if show_budget:
            ax.axhline(BUDGET, color="black", linestyle="--", linewidth=1.2,
                       alpha=0.6, label=f"Budget ({BUDGET})")
        ax.set_xticks(x)
        ax.set_xticklabels(labels_m, rotation=30, ha="right", fontsize=9.5)
        ax.set_ylabel(ylabel, fontsize=10.5)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Repertoire Structure Across Seeds  (error bars = ±1 SD)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── COEVOLVE dynamics ──────────────────────────────────────────────────────────

def plot_coevolve_dynamics(runs: list[dict], out_path: str) -> None:
    """3-panel COEVOLVE-only plot: best+mean fitness, repertoire diversity, opponent diversity."""
    coevolve_runs = [
        r for r in _select(runs, mode="COEVOLVE", seeds=MAIN_SEEDS, lambda_weight=MAIN_LAMBDA)
        if r.get("history")
    ]
    if not coevolve_runs:
        print("  [skip] no COEVOLVE history found.")
        return

    n_gens = len(coevolve_runs[0]["history"])
    valid  = [r for r in coevolve_runs if len(r["history"]) == n_gens]
    gens   = np.arange(n_gens)

    def _mat(key):
        rows = []
        for r in valid:
            row = [h.get(key) for h in r["history"]]
            if all(v is not None for v in row):
                rows.append(row)
        return np.array(rows) if rows else np.empty((0, n_gens))

    def _plot_band(ax, mat, color, label=None, linestyle="-"):
        if mat.size == 0:
            return
        m_line = np.mean(mat, axis=0)
        n = mat.shape[0]
        ci = 1.96 * np.std(mat, axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(n_gens)
        ax.plot(gens, m_line, linestyle, color=color, linewidth=2,
                label=label if label else "")
        ax.fill_between(gens, m_line - ci, m_line + ci, alpha=0.15, color=color)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # (a) best vs mean training fitness
    ax = axes[0]
    _plot_band(ax, _mat("best_training_fitness"), "#0072B2", label="Best", linestyle="-")
    _plot_band(ax, _mat("mean_training_fitness"), "#56B4E9", label="Mean", linestyle="--")
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Training Fitness", fontsize=11)
    ax.set_title("(a) Best vs Mean Fitness", fontsize=11)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    # (b) repertoire diversity
    ax = axes[1]
    _plot_band(ax, _mat("repertoire_diversity"), "#009E73")
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Mean Pairwise Jaccard Distance", fontsize=11)
    ax.set_title("(b) Repertoire Population Diversity", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    # (c) opponent diversity — only non-None rows
    ax = axes[2]
    opp_rows = []
    for r in valid:
        row = [h.get("opponent_diversity") for h in r["history"]]
        row_clean = [v if v is not None else float("nan") for v in row]
        opp_rows.append(row_clean)
    if opp_rows:
        opp_mat = np.array(opp_rows)
        _plot_band(ax, opp_mat, "#E69F00")
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Mean L2 Distance", fontsize=11)
    ax.set_title("(c) Opponent Population Diversity", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("COEVOLVE Dynamics Across Seeds  (95% CI)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Diagnostic table (COEVOLVE only) ──────────────────────────────────────────

def compute_diagnostic_table(runs: list[dict]) -> pd.DataFrame:
    rows = []
    variant_runs = [
        r for r in _select(runs, mode="COEVOLVE", seeds=MAIN_SEEDS, lambda_weight=MAIN_LAMBDA)
        if r["history"]
    ]
    if not variant_runs:
        return pd.DataFrame(rows)

    n_gens = len(variant_runs[0]["history"])
    for g in range(n_gens):
        rep_divs:  list[float] = []
        opp_divs:  list[float] = []
        hof_sizes: list[float] = []
        for r in variant_runs:
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


# ── Repertoire text report ────────────────────────────────────────────────────

# ── Opening name lookup (longest-prefix match) ───────────────────────────────

_OPENING_NAMES: dict[tuple, str] = {
    # ── 1.e4 ──────────────────────────────────────────────────────────────────
    ("e4",):                                         "King's Pawn",
    ("e4", "e5"):                                    "Open Game",
    ("e4", "e5", "Nf3", "Nc6", "Bb5"):              "Ruy Lopez",
    ("e4", "e5", "Nf3", "Nc6", "Bc4"):              "Italian Game",
    ("e4", "e5", "Nf3", "Nc6", "d4"):               "Scotch Game",
    ("e4", "e5", "Nf3", "Nf6"):                     "Petrov Defense",
    ("e4", "e5", "Nc3"):                             "Vienna Game",
    ("e4", "c5"):                                    "Sicilian Defense",
    ("e4", "c5", "Nf3", "d6"):                      "Sicilian — Najdorf/Dragon",
    ("e4", "c5", "Nf3", "Nc6"):                     "Sicilian — Classical",
    ("e4", "c5", "Nf3", "e6"):                      "Sicilian — Kan/Taimanov",
    ("e4", "c5", "c3"):                              "Sicilian — Alapin",
    ("e4", "e6"):                                    "French Defense",
    ("e4", "e6", "d4", "d5", "Nc3"):                "French — Classical",
    ("e4", "e6", "d4", "d5", "e5"):                 "French — Advance",
    ("e4", "e6", "d4", "d5", "exd5"):               "French — Exchange",
    ("e4", "c6"):                                    "Caro-Kann Defense",
    ("e4", "c6", "d4", "d5", "Nc3"):                "Caro-Kann — Classical",
    ("e4", "c6", "d4", "d5", "e5"):                 "Caro-Kann — Advance",
    ("e4", "c6", "d4", "d5", "exd5"):               "Caro-Kann — Exchange",
    ("e4", "d5"):                                    "Scandinavian Defense",
    ("e4", "d5", "exd5", "Qxd5", "Nc3"):            "Scandinavian — Classical",
    ("e4", "d5", "exd5", "Nf6"):                    "Scandinavian — Icelandic",
    ("e4", "d6"):                                    "Pirc Defense",
    ("e4", "g6"):                                    "Modern Defense",
    ("e4", "Nf6"):                                   "Alekhine Defense",
    # ── 1.d4 ──────────────────────────────────────────────────────────────────
    ("d4",):                                         "Queen's Pawn",
    ("d4", "d5", "c4"):                              "Queen's Gambit",
    ("d4", "d5", "c4", "c6"):                        "Slav Defense",
    ("d4", "d5", "c4", "c6", "Nf3"):                "Slav Defense",
    ("d4", "d5", "c4", "c6", "Nc3"):                "Slav Defense",
    ("d4", "d5", "c4", "c5"):                        "Symmetrical Defense",
    ("d4", "d5", "c4", "dxc4"):                      "Queen's Gambit Accepted",
    ("d4", "d5", "c4", "dxc4", "e3"):               "QGA — Classical",
    ("d4", "d5", "c4", "dxc4", "Nf3"):              "QGA",
    ("d4", "d5", "c4", "e5"):                        "Albin Counter-Gambit",
    ("d4", "d5", "c4", "e6"):                        "Queen's Gambit Declined",
    ("d4", "d5", "c4", "e6", "Nc3", "Bb4"):         "Nimzo-Indian Defense",
    ("d4", "d5", "c4", "e6", "Nc3", "Nf6"):         "QGD — Orthodox",
    ("d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5"): "QGD — Orthodox",
    ("d4", "d5", "c4", "Nf6"):                       "Anti-Indian",
    ("d4", "d5", "Bf4"):                             "London System",
    ("d4", "d5", "Nf3"):                             "Queen's Pawn",
    ("d4", "Nf6", "c4"):                             "Indian Defenses",
    ("d4", "Nf6", "c4", "c5"):                       "Benoni Defense",
    ("d4", "Nf6", "c4", "c5", "d5"):                "Modern Benoni",
    ("d4", "Nf6", "c4", "d5"):                       "Grünfeld Defense",
    ("d4", "Nf6", "c4", "d5", "cxd5"):              "Grünfeld Defense",
    ("d4", "Nf6", "c4", "d5", "Nc3"):               "Grünfeld / Anti-Grünfeld",
    ("d4", "Nf6", "c4", "e5"):                       "Budapest Gambit",
    ("d4", "Nf6", "c4", "e6"):                       "Indian / QGD",
    ("d4", "Nf6", "c4", "e6", "Nc3", "Bb4"):        "Nimzo-Indian Defense",
    ("d4", "Nf6", "c4", "e6", "Nc3", "Bb4", "Qc2"): "Nimzo-Indian — Classical",
    ("d4", "Nf6", "c4", "e6", "Nc3", "Bb4", "e3"):  "Nimzo-Indian — Rubinstein",
    ("d4", "Nf6", "c4", "e6", "Nc3", "Bb4", "Bg5"): "Nimzo-Indian — Leningrad",
    ("d4", "Nf6", "c4", "e6", "Nc3", "d5"):         "QGD / Hybrid",
    ("d4", "Nf6", "c4", "e6", "Nc3", "c5"):         "Benoni / Hedgehog",
    ("d4", "Nf6", "c4", "g6"):                       "King's Indian / Grünfeld",
    ("d4", "Nf6", "c4", "g6", "Nc3"):               "King's Indian Defense",
    ("d4", "Nf6", "c4", "g6", "Nc3", "Bg7"):        "King's Indian Defense",
    ("d4", "Nf6", "c4", "g6", "g3"):                "Catalan / Fianchetto",
    ("d4", "Nf6", "c4", "g6", "Nf3"):               "King's Indian",
    ("d4", "f5"):                                    "Dutch Defense",
    ("d4", "f5", "c4"):                              "Dutch Defense",
    ("d4", "f5", "g3"):                              "Dutch — Leningrad",
    ("d4", "g6"):                                    "Modern / King's Indian",
    ("d4", "b6"):                                    "Owen's Defense",
    # ── 1.c4 ──────────────────────────────────────────────────────────────────
    ("c4",):                                         "English Opening",
    ("c4", "e5"):                                    "English — Reversed Sicilian",
    ("c4", "c5"):                                    "English — Symmetrical",
    ("c4", "Nf6"):                                   "English Opening",
    ("c4", "e6"):                                    "English Opening",
    ("c4", "g6"):                                    "English — King's Indian",
    # ── 1.Nf3 / other ─────────────────────────────────────────────────────────
    ("Nf3",):                                        "Réti Opening",
    ("Nf3", "d5", "c4"):                             "Réti Opening",
    ("Nf3", "d5"):                                   "Réti Opening",
    ("Nf3", "Nf6", "c4"):                            "English / Indian",
    ("g3",):                                         "King's Fianchetto",
    ("b3",):                                         "Nimzo-Larsen Attack",
    ("f4",):                                         "Bird's Opening",
}


def _opening_name_for(san_moves: list[str]) -> str:
    """Longest-prefix match against _OPENING_NAMES; fallback to '1.X Opening'."""
    best = None
    for length in range(1, len(san_moves) + 1):
        name = _OPENING_NAMES.get(tuple(san_moves[:length]))
        if name:
            best = name
    return best or (f"1.{san_moves[0]} Opening" if san_moves else "Unknown Opening")


# ── Format A — decision tree with box-drawing ─────────────────────────────────

def _render_tree_A(committed: dict, reached: set, color: str, graph: dict) -> list[str]:
    """
    Render the repertoire as a branching decision tree.

    Branches happen only at opponent-choice points. Each branch line shows
    the opponent's move followed by "→ OurResponse". Sub-branches indent below.

    Example (White repertoire):
        1.d4
        ├─ 1...d5  →  2.c4
        │   ├─ 2...c6  →  3.Nf3
        │   │   ├─ 3...Bf5
        │   │   └─ 3...Nf6
        │   └─ 2...dxc4  →  3.e3
        └─ 1...Nf6  →  2.c4
    """
    nodes  = graph["nodes"]
    output: list[str] = []

    def _fmt(san: str, fen: str) -> str:
        node = nodes.get(fen)
        if node is None:
            return san
        n = node["ply_depth"] // 2 + 1
        return f"{n}.{san}" if node["turn"] == "white" else f"{n}...{san}"

    def _our_resp(fen: str):
        if fen not in committed:
            return None
        node = nodes.get(fen)
        if node is None:
            return None
        ci = node["children"].get(committed[fen])
        if ci is None or ci["child_fen"] not in reached:
            return None
        return _fmt(ci["move_san"], fen), ci["child_fen"]

    def _opp_covered(fen: str) -> list[tuple[str, str]]:
        node = nodes.get(fen)
        if node is None:
            return []
        return sorted(
            (_fmt(ci["move_san"], fen), ci["child_fen"])
            for ci in node["children"].values()
            if ci["child_fen"] in reached
        )

    def _render_opp(fen: str, indent: str, visited: frozenset) -> None:
        if fen in visited:
            return
        visited = visited | {fen}
        covered = _opp_covered(fen)
        for i, (opp_san, opp_child) in enumerate(covered):
            last = i == len(covered) - 1
            conn = "└─" if last else "├─"
            ext  = "   " if last else "│  "
            resp = _our_resp(opp_child)
            if resp is None:
                output.append(f"{indent}{conn} {opp_san}")
            else:
                our_san, our_child = resp
                output.append(f"{indent}{conn} {opp_san}  →  {our_san}")
                _render_opp(our_child, indent + ext, visited)

    root      = graph["root_fen"]
    root_node = nodes.get(root)
    if root_node is None:
        return ["  (graph data unavailable)"]

    if root_node["turn"] == color:
        resp = _our_resp(root)
        if resp is None:
            return ["  (no committed opening move)"]
        our_san, our_child = resp
        output.append(f"  {our_san}")
        _render_opp(our_child, "  ", frozenset({root}))
    else:
        _render_opp(root, "  ", frozenset())

    return output


# ── Format C — grouped by opening family ─────────────────────────────────────

def _collect_lines_for_color(
    committed: dict, reached: set, color: str, graph: dict
) -> list[list[str]]:
    """DFS collecting every root-to-leaf path as a list of SAN strings."""
    nodes = graph["nodes"]
    root  = graph["root_fen"]
    lines: list[list[str]] = []

    def _dfs(fen: str, path: list[str], visited: frozenset) -> None:
        if fen in visited:
            if path:
                lines.append(path)
            return
        visited = visited | {fen}
        node = nodes.get(fen)
        if node is None:
            if path:
                lines.append(path)
            return
        children = node.get("children", {})
        if node["turn"] == color:
            if fen not in committed:
                if path:
                    lines.append(path)
                return
            ci = children.get(committed[fen])
            if ci is None or ci["child_fen"] not in reached:
                if path:
                    lines.append(path)
                return
            _dfs(ci["child_fen"], path + [ci["move_san"]], visited)
        else:
            covered = [(ci["child_fen"], ci["move_san"])
                       for ci in children.values() if ci["child_fen"] in reached]
            if not covered:
                if path:
                    lines.append(path)
                return
            for child_fen, san in covered:
                _dfs(child_fen, path + [san], visited)

    _dfs(root, [], frozenset())
    return lines


def _format_line(san_moves: list[str]) -> str:
    """'1.e4  e5  2.Nf3  Nc6' from a flat SAN list (starting at ply 0)."""
    if not san_moves:
        return "(empty)"
    parts = []
    for i, san in enumerate(san_moves):
        parts.append(f"{i//2+1}.{san}" if i % 2 == 0 else san)
    return "  ".join(parts)


def _format_suffix(san_moves: list[str], start_ply: int) -> str:
    """Format a SAN continuation starting at *start_ply* (correct move numbers)."""
    if not san_moves:
        return "[end of line]"
    parts = []
    for i, san in enumerate(san_moves):
        ply = start_ply + i
        parts.append(f"{ply//2+1}.{san}" if ply % 2 == 0 else san)
    return "  ".join(parts)


def _render_format_C(lines: list[list[str]], group_depth: int = 4) -> list[str]:
    """
    Group lines by their first *group_depth* moves, label each group with the
    opening name, then list only the continuation moves for each line.
    """
    if not lines:
        return ["  (no lines)"]

    groups: dict[tuple, list[list[str]]] = {}
    for line in lines:
        key = tuple(line[:group_depth])
        groups.setdefault(key, []).append(line)

    output: list[str] = []
    for key in sorted(groups):
        group_lines = sorted(groups[key])
        prefix      = list(key)
        name        = _opening_name_for(prefix)
        header_pfx  = _format_line(prefix)
        n           = len(group_lines)
        bar         = "─" * max(2, 62 - len(name) - len(header_pfx))
        output.append(f"  ── {name}  ({header_pfx})  ── {n} line{'s' if n != 1 else ''}  {bar}")
        suffix_ply  = len(key)
        for line in group_lines:
            suffix = _format_suffix(line[suffix_ply:], suffix_ply)
            output.append(f"       {suffix}")
        output.append("")

    return output


# ── Shared writer ─────────────────────────────────────────────────────────────

def _write_mode_block(
    fh,
    best: dict,
    graph_train: dict,
    fmt: str,          # "A" or "C"
    SEP: str,
) -> None:
    cand   = best["final_best_candidate"]
    w_com  = cand["white_committed"]
    w_rea  = set(cand["white_reached"])
    b_com  = cand["black_committed"]
    b_rea  = set(cand["black_reached"])
    mode   = best.get("mode", "?")
    seed   = best.get("seed", "?")
    hm = best.get("heldout_metrics", {})
    heldout_legacy = best.get("heldout_score", float("nan"))
    heldout_uniform = hm.get("heldout_uniform_mean", heldout_legacy)
    white_heldout = hm.get("white_mean", float("nan"))
    black_heldout = hm.get("black_mean", float("nan"))

    if fmt == "A":
        w_lines = _render_tree_A(w_com, w_rea, "white", graph_train)
        b_lines = _render_tree_A(b_com, b_rea, "black", graph_train)
    else:
        raw_w   = _collect_lines_for_color(w_com, w_rea, "white", graph_train)
        raw_b   = _collect_lines_for_color(b_com, b_rea, "black", graph_train)
        w_lines = _render_format_C(raw_w)
        b_lines = _render_format_C(raw_b)

    fh.write(f"{SEP}\n")
    fh.write(
        f"  MODE : {mode}  |  SEED : {seed}"
        f"  |  HELDOUT_LEGACY : {heldout_legacy:.4f}"
        f"  |  HELDOUT_UNIFORM : {heldout_uniform:.4f}"
        f"  |  WHITE : {white_heldout:.4f}  |  BLACK : {black_heldout:.4f}\n"
    )
    fh.write(f"  White committed : {len(w_com)} moves  |  "
             f"Black committed : {len(b_com)} moves\n")
    fh.write(f"{SEP}\n\n")

    fh.write("  ── AS WHITE ──\n\n")
    for ln in w_lines:
        fh.write(f"{ln}\n")

    fh.write("\n  ── AS BLACK ──\n\n")
    for ln in b_lines:
        fh.write(f"{ln}\n")

    fh.write("\n")


def write_repertoire_report(
    runs: list[dict],
    results_dir: str,
    data_dir: str = "data",
) -> None:
    """
    For each GA method, pick the best seed and write two repertoire files:
      repertoire_tree.txt    — Format A: branching decision tree
      repertoire_grouped.txt — Format C: grouped by opening family
    """
    graph_train_path = os.path.join(data_dir, "graph_train.pkl")
    if not os.path.exists(graph_train_path):
        print("  [skip] graph_train.pkl not found.")
        return
    with open(graph_train_path, "rb") as fh:
        graph_train = pickle.load(fh)

    ga_methods = ["COEVOLVE", "STATIC", "GREEDY_HILLCLIMB"]
    SEP  = "=" * 70
    HSEP = "#" * 70

    for fmt, filename, title in [
        ("A", "repertoire_tree.txt",    "DECISION TREE — branch = opponent choice, → = your response"),
        ("C", "repertoire_grouped.txt", "GROUPED BY OPENING — continuations shown after the opening header"),
    ]:
        out_path = os.path.join(results_dir, filename)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(f"{SEP}\n")
            fh.write(f"  CHESS REPERTOIRE PROJECT\n")
            fh.write(f"  {title}\n")
            fh.write(f"  Ranked by held-out fitness (lambda={MAIN_LAMBDA}, best seed per mode)\n")
            fh.write(f"{SEP}\n\n")

            for mode in ga_methods:
                mode_runs = [
                    r for r in runs
                    if r.get("mode") == mode
                    and r.get("config", {}).get("lambda_weight") == MAIN_LAMBDA
                ]
                if not mode_runs:
                    continue
                best = max(mode_runs, key=lambda r: r.get("heldout_score", -float("inf")))
                _write_mode_block(fh, best, graph_train, fmt, HSEP)

        print(f"  saved -> {out_path}")


# ── Closure ablation plot ─────────────────────────────────────────────────────

def plot_closure_ablation(runs: list[dict], out_path: str) -> None:
    """Side-by-side box+strip comparing closure ON vs OFF for STATIC and COEVOLVE.

    This is the main novel-contribution validation plot. If the closure rule
    helps, WITH-closure methods should consistently score higher.
    """
    groups = [
        ("STATIC",            "STATIC_NOCLOSURE",   "Static GA"),
        ("COEVOLVE",          "COEVOLVE_NOCLOSURE",  "CoEvolve"),
    ]
    metrics_info = [
        ("heldout_uniform_mean", "Mean Score (uniform opponent)"),
        ("heldout_worst_band",   "Worst-Band Score (robustness)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    rng_j = np.random.default_rng(42)

    for ax, (metric, ylabel) in zip(axes, metrics_info):
        positions = []
        all_data, colors, labels = [], [], []
        tick_labels = []
        pos = 0
        group_centers = []

        for with_m, without_m, title in groups:
            group_start = pos
            for m, linestyle in [(with_m, "Closure"), (without_m, "No Closure")]:
                sel = _select(runs, mode=m, seeds=CLOSURE_ABLATION_SEEDS, lambda_weight=MAIN_LAMBDA)
                vals = list(_scores_by_seed_metric(sel, metric).values())
                if not vals:
                    pos += 1
                    continue
                all_data.append(vals)
                positions.append(pos)
                colors.append(_METHOD_COLORS.get(m, "#888888"))
                labels.append(f"{title}\n({linestyle})")
                tick_labels.append(f"{title}\n{linestyle}")
                pos += 1
            group_centers.append((group_start + pos - 1) / 2)
            pos += 0.6  # gap between groups

        if not all_data:
            ax.text(0.5, 0.5, "No ablation data yet", ha="center", va="center",
                    transform=ax.transAxes, color="grey", fontsize=11)
            ax.set_title(ylabel, fontsize=11)
            continue

        bp = ax.boxplot(
            all_data, positions=positions, widths=0.4, patch_artist=True,
            medianprops={"color": "black", "linewidth": 2},
            whiskerprops={"linewidth": 1.2}, capprops={"linewidth": 1.2},
            showfliers=False,
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)

        for pos_i, vals, c in zip(positions, all_data, colors):
            jitter = rng_j.uniform(-0.12, 0.12, len(vals))
            ax.scatter([pos_i + j for j in jitter], vals, color=c, s=26,
                       alpha=0.85, zorder=4, edgecolors="white", linewidths=0.4)

        ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.35)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10.5)
        ax.set_title(ylabel, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Closure Ablation: Does Forcing Opponent Reply Coverage Help?  (same seeds, paired)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Top-level entry point ─────────────────────────────────────────────────────

def run_analysis(
    runs_dir: str = "runs",
    results_dir: str = "results",
) -> None:
    """Run the full analysis pipeline and write all outputs to results_dir."""
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading runs from '{runs_dir}' ...")
    runs = load_runs(runs_dir)
    print(f"  {len(runs)} run(s) loaded.\n")

    print("Recomputing held-out metrics ...")
    _augment_runs_with_metrics(runs)
    print("  done.\n")

    print("Computing summary_table.csv ...")
    summary_df = compute_summary_table(runs)
    summary_df.to_csv(os.path.join(results_dir, "summary_table.csv"), index=False)
    for metric, sub in summary_df.groupby("metric", sort=False):
        print(f"\n[{metric}]")
        print(sub.drop(columns=["metric"]).to_string(index=False))

    print("\nComputing pairwise_table.csv ...")
    pairwise_df = compute_pairwise_table(runs)
    pairwise_df.to_csv(os.path.join(results_dir, "pairwise_table.csv"), index=False)
    for metric, sub in pairwise_df.groupby("metric", sort=False):
        print(f"\n[{metric}]")
        print(sub.drop(columns=["metric"]).to_string(index=False))

    print("\nComputing closure_ablation_table.csv ...")
    ablation_df = compute_closure_ablation_table(runs)
    ablation_df.to_csv(os.path.join(results_dir, "closure_ablation_table.csv"), index=False)
    print(ablation_df.to_string(index=False))

    print("\nPlotting convergence.png ...")
    plot_convergence(runs, os.path.join(results_dir, "convergence.png"))
    print("  saved.")

    print("\nPlotting score_distributions.png ...")
    plot_score_distributions(runs, os.path.join(results_dir, "score_distributions.png"))
    print("  saved.")

    print("\nPlotting band_breakdown.png ...")
    plot_band_breakdown(runs, os.path.join(results_dir, "band_breakdown.png"))
    print("  saved.")

    print("\nPlotting closure_ablation.png ...")
    plot_closure_ablation(runs, os.path.join(results_dir, "closure_ablation.png"))
    print("  saved.")

    print("\nPlotting white_black_breakdown.png ...")
    plot_white_black_breakdown(runs, os.path.join(results_dir, "white_black_breakdown.png"))
    print("  saved.")

    print("\nPlotting ga_vs_nonga.png ...")
    plot_ga_vs_nonga(runs, os.path.join(results_dir, "ga_vs_nonga.png"))
    print("  saved.")

    print("\nPlotting repertoire_structure.png ...")
    plot_repertoire_structure(runs, os.path.join(results_dir, "repertoire_structure.png"))
    print("  saved.")

    print("\nPlotting coevolve_dynamics.png ...")
    plot_coevolve_dynamics(runs, os.path.join(results_dir, "coevolve_dynamics.png"))
    print("  saved.")

    print("\nComputing diagnostic_table.csv ...")
    diag_df = compute_diagnostic_table(runs)
    diag_df.to_csv(os.path.join(results_dir, "diagnostic_table.csv"), index=False)
    print(diag_df.to_string(index=False))

    print("\nWriting repertoire_tree.txt + repertoire_grouped.txt ...")
    write_repertoire_report(runs, results_dir)

    print(f"\nDone - results saved to '{results_dir}'.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ChessRepertoire analysis")
    parser.add_argument("--runs-dir",    default="runs",    help="Directory with run pickles")
    parser.add_argument("--results-dir", default="results", help="Output directory")
    args = parser.parse_args()
    run_analysis(runs_dir=args.runs_dir, results_dir=args.results_dir)
