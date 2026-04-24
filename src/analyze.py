"""
Analysis: load all run pickles, compute statistics, produce tables and plots.

Outputs (under results/):
    main_table.csv            — one row per (metric, method); mean/std/median/
                                Holm-corrected Wilcoxon p vs STATIC, A12 vs STATIC
    convergence.png           — best-training-fitness curves with 95% CI
    sensitivity_table.csv     — (metric, method, lambda) rows; mean/std
    diagnostic_table.csv      — COEVOLVE per-generation averages across seeds
    repertoire_subgraphs.txt  — best candidate per GA mode written as chess lines
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
    SENSITIVITY_SEEDS,
    MAIN_LAMBDA,
    SENSITIVITY_LAMBDAS,
    ADVERSARIAL_RNG_SEED,
    ADVERSARIAL_N_SAMPLES,
    ADVERSARIAL_DIRICHLET_ALPHA,
    ADVERSARIAL_QUANTILE,
)


# ── Constants ─────────────────────────────────────────────────────────────────

METHODS_MAIN = [
    "most_played_baseline",
    "RANDOM_SEARCH",
    "GREEDY_HILLCLIMB",
    "STATIC",
    "COEVOLVE_FROZEN",
    "COEVOLVE",
]
NON_STATIC_METHODS = [
    "most_played_baseline",
    "RANDOM_SEARCH",
    "GREEDY_HILLCLIMB",
    "COEVOLVE_FROZEN",
    "COEVOLVE",
]
SENS_SEEDS = SENSITIVITY_SEEDS

HELDOUT_METRICS = [
    "heldout_uniform_mean",       # mean score under uniform mixture
    "heldout_worst_band",         # min over the N bands (CVaR α=1/N)
    "heldout_adversarial_q10",    # 10th-pct mean over Dirichlet mixtures
    "heldout_uniform_fitness",    # mean + λ·cvar (legacy heldout_score)
]

_CONVERGENCE_COLORS = {
    "most_played_baseline": "dimgray",
    "STATIC": "steelblue",
    "COEVOLVE_FROZEN": "darkorange",
    "COEVOLVE": "forestgreen",
    "RANDOM_SEARCH": "firebrick",
    "GREEDY_HILLCLIMB": "purple",
}


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
) -> dict[str, float]:
    bs = _band_scores_heldout(candidate_ser, eval_cache_heldout, base_policies_train, graph_heldout)
    band_vec = np.array([bs[b] for b in BANDS])

    uniform_mean = float(band_vec.mean())
    worst_band = float(band_vec.min())

    rng = np.random.default_rng(ADVERSARIAL_RNG_SEED)
    mixtures = rng.dirichlet(np.full(len(BANDS), ADVERSARIAL_DIRICHLET_ALPHA), size=ADVERSARIAL_N_SAMPLES)
    sampled_means = mixtures @ band_vec
    adversarial_q10 = float(np.quantile(sampled_means, ADVERSARIAL_QUANTILE))

    uniform_fitness = uniform_mean + lambda_weight * worst_band

    return {
        "heldout_uniform_mean":     uniform_mean,
        "heldout_worst_band":       worst_band,
        "heldout_adversarial_q10":  adversarial_q10,
        "heldout_uniform_fitness":  uniform_fitness,
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
    out: dict[int, float] = {}
    for r in runs:
        hm = r.get("heldout_metrics")
        if hm is not None and metric in hm:
            out[r["seed"]] = hm[metric]
        elif metric == "heldout_uniform_fitness":
            out[r["seed"]] = r["heldout_score"]
    return out


# ── Main table ────────────────────────────────────────────────────────────────

def compute_main_table(runs: list[dict]) -> pd.DataFrame:
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
                mean_h = float(np.mean(scores))
                std_h = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
                median_h = float(np.median(scores))
            else:
                mean_h = std_h = median_h = float("nan")

            if m == "STATIC":
                wilcoxon_p = float("nan")
                a12 = float("nan")
            else:
                wilcoxon_p = non_static_corrected.get(m, float("nan"))
                m_by_seed = method_scores[m]
                shared = sorted(set(m_by_seed) & set(static_by_seed))
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
    fig, ax = plt.subplots(figsize=(8, 5))

    method_order = [
        "most_played_baseline",
        "STATIC",
        "COEVOLVE_FROZEN",
        "COEVOLVE",
        "RANDOM_SEARCH",
        "GREEDY_HILLCLIMB",
    ]

    # Use GA runs to set a common generation axis for flat baseline curves.
    ga_lengths = [
        len(r["history"])
        for m in ["STATIC", "COEVOLVE_FROZEN", "COEVOLVE"]
        for r in _select(runs, mode=m, seeds=MAIN_SEEDS, lambda_weight=MAIN_LAMBDA)
        if r["history"]
    ]
    default_n_gens = max(ga_lengths) if ga_lengths else 1

    for m in method_order:
        selected = _select(runs, mode=m, seeds=MAIN_SEEDS, lambda_weight=MAIN_LAMBDA)
        if not selected:
            continue

        history_runs = [r for r in selected if r["history"]]
        if history_runs:
            n_gens = len(history_runs[0]["history"])
            matrix = np.array([
                [h["best_training_fitness"] for h in r["history"]]
                for r in history_runs
                if len(r["history"]) == n_gens
            ])
        else:
            scalar_vals = [
                float(r["final_training_fitness"])
                for r in selected
                if r.get("final_training_fitness") is not None
            ]
            if not scalar_vals:
                continue
            n_gens = default_n_gens
            matrix = np.repeat(np.array(scalar_vals)[:, None], n_gens, axis=1)

        if matrix.ndim != 2 or matrix.shape[0] == 0:
            continue

        gens = np.arange(n_gens)
        mean_f = np.mean(matrix, axis=0)
        n = matrix.shape[0]
        se = np.std(matrix, axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(n_gens)
        ci = 1.96 * se

        color = _CONVERGENCE_COLORS.get(m, "black")
        ax.plot(gens, mean_f, label=m, color=color)
        ax.fill_between(gens, mean_f - ci, mean_f + ci, alpha=0.2, color=color)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Training Fitness")
    ax.set_title("Convergence (95% CI across seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Sensitivity table ─────────────────────────────────────────────────────────

def compute_sensitivity_table(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for metric in HELDOUT_METRICS:
        for m in ["STATIC", "COEVOLVE"]:
            for lam in SENSITIVITY_LAMBDAS:
                subset = _select(runs, mode=m, seeds=SENS_SEEDS, lambda_weight=lam)
                scores = list(_scores_by_seed_metric(subset, metric).values())
                if scores:
                    mean_h = float(np.mean(scores))
                    std_h = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
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
    heldout = best.get("heldout_score", float("nan"))

    if fmt == "A":
        w_lines = _render_tree_A(w_com, w_rea, "white", graph_train)
        b_lines = _render_tree_A(b_com, b_rea, "black", graph_train)
    else:
        raw_w   = _collect_lines_for_color(w_com, w_rea, "white", graph_train)
        raw_b   = _collect_lines_for_color(b_com, b_rea, "black", graph_train)
        w_lines = _render_format_C(raw_w)
        b_lines = _render_format_C(raw_b)

    fh.write(f"{SEP}\n")
    fh.write(f"  MODE : {mode}  |  SEED : {seed}  |  HELDOUT FITNESS : {heldout:.4f}\n")
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

    ga_methods = ["COEVOLVE", "COEVOLVE_FROZEN", "STATIC", "GREEDY_HILLCLIMB"]
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


# ── Top-level entry point ─────────────────────────────────────────────────────

def run_analysis(runs_dir: str = "runs", results_dir: str = "results") -> None:
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading runs from '{runs_dir}' ...")
    runs = load_runs(runs_dir)
    print(f"  {len(runs)} run(s) loaded.\n")

    print("Recomputing held-out metrics ...")
    _augment_runs_with_metrics(runs)
    print("  done.\n")

    print("Computing main_table.csv ...")
    main_df = compute_main_table(runs)
    main_df.to_csv(os.path.join(results_dir, "main_table.csv"), index=False)
    for metric, sub in main_df.groupby("metric", sort=False):
        print(f"\n[{metric}]")
        print(sub.drop(columns=["metric"]).to_string(index=False))

    print("\nPlotting convergence.png ...")
    plot_convergence(runs, os.path.join(results_dir, "convergence.png"))
    print("  saved.")

    print("\nComputing sensitivity_table.csv ...")
    sens_df = compute_sensitivity_table(runs)
    sens_df.to_csv(os.path.join(results_dir, "sensitivity_table.csv"), index=False)
    print(sens_df.to_string(index=False))

    print("\nComputing diagnostic_table.csv ...")
    diag_df = compute_diagnostic_table(runs)
    diag_df.to_csv(os.path.join(results_dir, "diagnostic_table.csv"), index=False)
    print(diag_df.to_string(index=False))

    print("\nWriting repertoire_tree.txt + repertoire_grouped.txt ...")
    write_repertoire_report(runs, results_dir)

    print(f"\nDone — results saved to '{results_dir}/'.")


if __name__ == "__main__":
    run_analysis()
