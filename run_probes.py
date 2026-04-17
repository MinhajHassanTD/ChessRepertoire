"""testv2 — Phase 1 probe orchestrator.

Loads every saved run in runs/, runs Probe A (per-game trajectory sampling on the
held-out graph) on all of them, and runs Probe B (python-chess simulation against
Stockfish, with aggregate-freq fallback when Stockfish is not on PATH) on a
pilot subset.

Writes two CSVs under results/:

  - probe_report.csv    : one row per run, all per-run metrics
  - probe_summary.csv   : per-method aggregate + A12/Wilcoxon vs STATIC

Phase 1 gate: if COEVOLVE_D vs STATIC A12 >= 0.60 on any independent axis
(per_game_worst_decile or sim_mean / sim_worst_decile), proceed to Phase 2.

Usage:
    python run_probes.py                 # full Probe A + pilot Probe B
    python run_probes.py --probe-a-only  # skip Probe B (faster)
    python run_probes.py --full-sim      # Probe B on all methods/seeds (slow)
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats

from src.analyze import _a12, load_runs
from src.eval_cache import load_eval_cache
from src.graph import load_graph
from src.per_game_eval import sample_trajectories, summarise_trajectories
from src.policies import load_policies
from src.simulate import simulate_candidate


# ── Configuration ─────────────────────────────────────────────────────────────

PILOT_METHODS = ["STATIC", "STATIC_DIV", "COEVOLVE", "COEVOLVE_D"]
PILOT_SEEDS   = list(range(1000, 1005))   # first 5 seeds
PILOT_N_GAMES = 50
PROBE_A_SAMPLES_PER_BAND = 400

# Methods we report in the summary (mirrors analyze.METHODS_MAIN minus the
# baseline — the baseline has no `final_best_candidate` so it can't be probed).
SUMMARY_METHODS = [
    "STATIC",
    "STATIC_DIV",
    "COEVOLVE_FROZEN",
    "COEVOLVE",
    "COEVOLVE_B",
    "COEVOLVE_C",
    "COEVOLVE_D",
]


# ── Probe A driver ────────────────────────────────────────────────────────────

def run_probe_a(runs: list[dict], eval_cache_heldout: dict,
                base_policies: dict, graph_heldout: dict) -> pd.DataFrame:
    rows = []
    n = len(runs)
    for i, r in enumerate(runs, 1):
        method = r.get("mode")
        seed = r.get("seed")
        cand = r.get("final_best_candidate")
        if cand is None:
            print(f"[A {i:3}/{n}] SKIP {method} s{seed} — no final_best_candidate")
            continue
        t0 = time.time()
        traj = sample_trajectories(
            candidate_ser=cand,
            eval_cache=eval_cache_heldout,
            base_policies=base_policies,
            graph=graph_heldout,
            n_samples_per_band=PROBE_A_SAMPLES_PER_BAND,
            seed=seed + 50_000,
        )
        summary = summarise_trajectories(traj)
        dt = time.time() - t0
        print(f"[A {i:3}/{n}] {method:16s} s{seed}  "
              f"pg_mean={summary['per_game_mean']:.4f}  "
              f"pg_q10={summary['per_game_worst_decile']:.4f}  "
              f"({dt:.1f}s)")
        rows.append({
            "method": method,
            "seed":   seed,
            "heldout_score": r.get("heldout_score"),
            **summary,
        })
    return pd.DataFrame(rows)


# ── Probe B driver ────────────────────────────────────────────────────────────

def run_probe_b(runs: list[dict], eval_cache_heldout: dict, graph_heldout: dict,
                pilot_methods: list[str], pilot_seeds: list[int],
                n_games: int, stockfish_path: str | None) -> pd.DataFrame:
    rows = []
    selected = [
        r for r in runs
        if r.get("mode") in pilot_methods and r.get("seed") in pilot_seeds
    ]
    n = len(selected)
    if n == 0:
        print("[B] no runs matched pilot selection.")
        return pd.DataFrame(rows)

    mode_str = "Stockfish" if (stockfish_path and os.path.exists(stockfish_path)) else "aggregate-freq fallback"
    print(f"[B] pilot opponent: {mode_str}  ({n} runs × {n_games} games)")

    for i, r in enumerate(selected, 1):
        method = r["mode"]; seed = r["seed"]
        cand = r.get("final_best_candidate")
        if cand is None:
            continue
        t0 = time.time()
        summary = simulate_candidate(
            candidate_ser=cand,
            heldout_graph=graph_heldout,
            eval_cache_heldout=eval_cache_heldout,
            n_games=n_games,
            seed=seed + 70_000,
            stockfish_path=stockfish_path,
            engine_depth=8,
            max_plies=120,
        )
        dt = time.time() - t0
        print(f"[B {i:3}/{n}] {method:16s} s{seed}  "
              f"sim_mean={summary['sim_mean']:.4f}  "
              f"sim_q10={summary['sim_worst_decile']:.4f}  "
              f"trunc={summary['sim_truncated_frac']:.2f}  "
              f"({dt:.1f}s)")
        rows.append({
            "method": method,
            "seed":   seed,
            "sim_mean":           summary["sim_mean"],
            "sim_std":            summary["sim_std"],
            "sim_worst_decile":   summary["sim_worst_decile"],
            "sim_truncated_frac": summary["sim_truncated_frac"],
        })
    return pd.DataFrame(rows)


# ── Aggregation + A12 / Wilcoxon vs STATIC ────────────────────────────────────

_METRICS = [
    "per_game_mean", "per_game_worst_decile", "per_game_worst_band_mean",
    "per_game_std",
    "sim_mean", "sim_std", "sim_worst_decile", "sim_truncated_frac",
]


def summarise(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    if not df_a.empty and not df_b.empty:
        df = df_a.merge(df_b, on=["method", "seed"], how="left")
    else:
        df = df_a.copy()
        for c in ("sim_mean", "sim_std", "sim_worst_decile", "sim_truncated_frac"):
            if c not in df.columns:
                df[c] = np.nan

    rows = []
    static_by_metric = {
        m: df[df["method"] == "STATIC"][m].dropna().tolist()
        for m in _METRICS
    }
    for method in SUMMARY_METHODS:
        sub = df[df["method"] == method]
        for metric in _METRICS:
            vals = sub[metric].dropna().tolist()
            if not vals:
                continue
            static_vals = static_by_metric.get(metric, [])
            p = np.nan
            a12 = np.nan
            if method != "STATIC" and static_vals:
                common_seeds = sorted(
                    set(sub["seed"]).intersection(
                        df[df["method"] == "STATIC"]["seed"]
                    )
                )
                paired_me = sub[sub["seed"].isin(common_seeds)].sort_values("seed")[metric].tolist()
                paired_st = df[(df["method"] == "STATIC") & (df["seed"].isin(common_seeds))].sort_values("seed")[metric].tolist()
                paired_me = [x for x in paired_me if not np.isnan(x)]
                paired_st = [x for x in paired_st if not np.isnan(x)]
                if len(paired_me) == len(paired_st) and len(paired_me) >= 2:
                    try:
                        w = stats.wilcoxon(paired_me, paired_st, zero_method="wilcox",
                                           alternative="greater")
                        p = float(w.pvalue)
                    except ValueError:
                        p = np.nan
                a12 = _a12(vals, static_vals)
            rows.append({
                "method": method,
                "metric": metric,
                "n":      len(vals),
                "mean":   float(np.mean(vals)),
                "std":    float(np.std(vals)),
                "median": float(np.median(vals)),
                "wilcoxon_p_vs_STATIC": p,
                "A12_vs_STATIC":        a12,
            })
    return pd.DataFrame(rows)


# ── Go/no-go gate ─────────────────────────────────────────────────────────────

_GATE_METRICS = ["per_game_worst_decile", "sim_mean", "sim_worst_decile"]
_GATE_THRESHOLD = 0.60


def decide_phase2(summary_df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("Phase 1 → Phase 2 decision gate")
    print("=" * 70)
    cov_d = summary_df[(summary_df["method"] == "COEVOLVE_D") &
                       (summary_df["metric"].isin(_GATE_METRICS))]
    triggered = False
    for _, row in cov_d.iterrows():
        hit = (row["A12_vs_STATIC"] is not None
               and not np.isnan(row["A12_vs_STATIC"])
               and row["A12_vs_STATIC"] >= _GATE_THRESHOLD)
        tag = "  [GATE HIT]" if hit else ""
        if hit:
            triggered = True
        print(f"  COEVOLVE_D vs STATIC on {row['metric']:28s}: "
              f"A12={row['A12_vs_STATIC']:.3f}  p={row['wilcoxon_p_vs_STATIC']}{tag}")
    if triggered:
        print("\n-> Phase 2 triggered: scale up Probe B to full 15 seeds x all methods.")
    else:
        print("\n-> No Phase 1 gate hit. Fall back to diagnosis-paper narrative; "
              "Probe A/B results still corroborate parity.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-a-only", action="store_true",
                    help="Skip Probe B (simulation).")
    ap.add_argument("--full-sim", action="store_true",
                    help="Run Probe B on all methods × all seeds (slow).")
    ap.add_argument("--n-games", type=int, default=PILOT_N_GAMES,
                    help="Games per (method, seed) in Probe B.")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    load_dotenv()
    stockfish_path = os.environ.get("STOCKFISH_PATH")

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    print("Loading held-out artifacts ...")
    graph_heldout = load_graph(os.path.join(args.data_dir, "graph_heldout.pkl"))
    base_policies = load_policies(os.path.join(args.data_dir, "base_policies.pkl"))
    eval_cache_heldout = load_eval_cache(os.path.join(args.data_dir, "eval_cache_heldout.pkl"))
    print("Loading runs ...")
    runs = load_runs(args.runs_dir)
    print(f"Loaded {len(runs)} runs.")

    # Probe A
    print("\n== Probe A: per-game trajectory sampling ==")
    df_a = run_probe_a(runs, eval_cache_heldout, base_policies, graph_heldout)

    # Probe B
    if args.probe_a_only:
        df_b = pd.DataFrame()
    else:
        print("\n== Probe B: python-chess game simulation ==")
        if args.full_sim:
            pilot_methods = SUMMARY_METHODS
            pilot_seeds = sorted({r["seed"] for r in runs if r.get("mode") in SUMMARY_METHODS})
        else:
            pilot_methods = PILOT_METHODS
            pilot_seeds = PILOT_SEEDS
        df_b = run_probe_b(
            runs, eval_cache_heldout, graph_heldout,
            pilot_methods=pilot_methods,
            pilot_seeds=pilot_seeds,
            n_games=args.n_games,
            stockfish_path=stockfish_path,
        )

    # ── Write per-run report ─────────────────────────────────────────────────
    if df_b.empty:
        report = df_a
    else:
        report = df_a.merge(df_b, on=["method", "seed"], how="left")
    report_path = os.path.join(args.results_dir, "probe_report.csv")
    report.sort_values(["method", "seed"]).to_csv(report_path, index=False)
    print(f"\nWrote {report_path}  ({len(report)} rows)")

    # ── Write summary w/ A12/Wilcoxon vs STATIC ──────────────────────────────
    summary = summarise(df_a, df_b)
    summary_path = os.path.join(args.results_dir, "probe_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}  ({len(summary)} rows)")

    # ── Phase 2 decision ─────────────────────────────────────────────────────
    decide_phase2(summary)


if __name__ == "__main__":
    main()
