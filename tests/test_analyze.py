"""
Tests for C10 — Analysis (src/analyze.py).

Each test function maps directly to one acceptance criterion from Section E C10:

  AC1  re-running produce bit-identical CSV output files
  AC2  main_table.csv has the correct columns and one row per method
  AC3  wilcoxon_p_vs_STATIC is Holm-corrected (k=3) across non-STATIC methods
  AC4  A12_vs_STATIC is computed with the correct formula
  AC5  sensitivity_table.csv has one row per (method, lambda) pair
  AC6  diagnostic_table.csv contains only COEVOLVE data
  AC7  convergence.png is created (non-empty file)

Synthetic run fixtures are constructed in-process — no real data needed.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.analyze import (
    _a12,
    _holm_correct,
    compute_diagnostic_table,
    compute_main_table,
    compute_sensitivity_table,
    load_runs,
    plot_convergence,
    run_analysis,
)


# ── Synthetic data helpers ────────────────────────────────────────────────────

def _make_history(n_gens: int, mode: str, rng: np.random.Generator) -> list[dict]:
    """Build a plausible history list for *n_gens* generations."""
    history = []
    for g in range(n_gens):
        history.append({
            "gen":                  g,
            "best_training_fitness": float(rng.uniform(0.4, 0.6)),
            "mean_training_fitness": float(rng.uniform(0.35, 0.55)),
            "repertoire_diversity":  float(rng.uniform(0.1, 0.5)),
            "opponent_diversity":    float(rng.uniform(0.05, 0.3)) if mode == "COEVOLVE" else None,
            "hof_size":              int(rng.integers(0, 6)) if mode == "COEVOLVE" else 0,
        })
    return history


def _make_run(
    mode: str,
    seed: int,
    lambda_weight: float,
    heldout_score: float,
    n_gens: int = 5,
) -> dict:
    """Return a minimal run dict that analyze.py can consume."""
    rng = np.random.default_rng(seed + hash(mode) % 2**31)
    history = [] if mode == "most_played_baseline" else _make_history(n_gens, mode, rng)
    return {
        "mode":                   mode,
        "config":                 {"lambda_weight": lambda_weight, "alpha": 1 / 3},
        "seed":                   seed,
        "git_commit":             "deadbeef",
        "history":                history,
        "final_best_candidate":   {
            "white_committed": {}, "white_reached": [],
            "black_committed": {}, "black_reached": [],
        },
        "final_training_fitness": None if mode == "most_played_baseline" else float(rng.uniform(0.4, 0.6)),
        "heldout_score":          heldout_score,
        "wall_time_seconds":      1.0,
    }


def _write_runs(runs: list[dict], runs_dir: str) -> None:
    """Pickle each run into *runs_dir* using the standard filename scheme."""
    for r in runs:
        lam = r["config"]["lambda_weight"]
        fname = f"{r['mode']}_l{lam}_s{r['seed']}.pkl"
        with open(os.path.join(runs_dir, fname), "wb") as fh:
            pickle.dump(r, fh)


def _build_main_runs(seeds=range(1000, 1010)) -> list[dict]:
    """
    Create synthetic main-experiment runs for all four methods.
    COEVOLVE scores are deliberately higher so Wilcoxon rejects.
    """
    rng_base = np.random.default_rng(42)
    runs: list[dict] = []

    static_scores = {s: float(rng_base.uniform(0.45, 0.50)) for s in seeds}

    for s in seeds:
        runs.append(_make_run("most_played_baseline", s, 1.0, static_scores[s] - 0.02))
        runs.append(_make_run("STATIC",               s, 1.0, static_scores[s]))
        runs.append(_make_run("COEVOLVE_FROZEN",       s, 1.0, static_scores[s] + 0.01))
        runs.append(_make_run("COEVOLVE",              s, 1.0, static_scores[s] + 0.05))
    return runs


def _build_sensitivity_runs() -> list[dict]:
    """Sensitivity experiments: STATIC and COEVOLVE × 3 lambdas × seeds 2000–2004."""
    runs: list[dict] = []
    rng = np.random.default_rng(99)
    for m in ["STATIC", "COEVOLVE"]:
        for lam in [0.0, 1.0, 2.0]:
            for s in range(2000, 2005):
                runs.append(_make_run(m, s, lam, float(rng.uniform(0.44, 0.56))))
    return runs


# ── AC1: Bit-identical CSV re-runs ────────────────────────────────────────────

def test_bit_identical_reruns(tmp_path):
    """
    AC1 — running analyze.py twice with the same data produces bit-identical
    CSV files (main_table, sensitivity_table, diagnostic_table).
    """
    runs_dir    = tmp_path / "runs"
    results_dir = tmp_path / "results"
    runs_dir.mkdir()

    all_runs = _build_main_runs() + _build_sensitivity_runs()
    _write_runs(all_runs, str(runs_dir))

    run_analysis(str(runs_dir), str(results_dir))

    # Read first-pass bytes
    csv_files = ["main_table.csv", "sensitivity_table.csv", "diagnostic_table.csv"]
    first = {f: (results_dir / f).read_bytes() for f in csv_files}

    # Second pass into a fresh results dir
    results_dir2 = tmp_path / "results2"
    run_analysis(str(runs_dir), str(results_dir2))
    second = {f: (results_dir2 / f).read_bytes() for f in csv_files}

    for f in csv_files:
        assert first[f] == second[f], f"{f} is not bit-identical across two runs"


# ── AC2: main_table.csv columns and rows ─────────────────────────────────────

def test_main_table_columns_and_rows(tmp_path):
    """main_table.csv has the (metric, method) schema with required columns."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _write_runs(_build_main_runs(), str(runs_dir))

    runs = load_runs(str(runs_dir))
    df = compute_main_table(runs)

    required_columns = {
        "metric", "method", "mean", "std", "median",
        "wilcoxon_p_vs_STATIC", "A12_vs_STATIC",
    }
    assert required_columns == set(df.columns), f"Unexpected columns: {set(df.columns)}"

    expected_methods = {"most_played_baseline", "STATIC", "COEVOLVE_FROZEN", "COEVOLVE"}
    assert set(df["method"]) == expected_methods


def test_main_table_stat_values_are_finite(tmp_path):
    """STATIC rows have NaN p / A12 (self-comparison undefined)."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _write_runs(_build_main_runs(), str(runs_dir))

    df = compute_main_table(load_runs(str(runs_dir)))

    static_rows = df[df["method"] == "STATIC"]
    assert static_rows["wilcoxon_p_vs_STATIC"].isna().all()
    assert static_rows["A12_vs_STATIC"].isna().all()


# ── AC3: Holm correction ──────────────────────────────────────────────────────

def test_holm_correct_k3():
    """
    AC3 — _holm_correct with k=3 multiplies sorted p-values by [3, 2, 1]
    in ascending order, clips at 1.0, and returns values in original order.
    """
    # Input in arbitrary order: [p_medium, p_small, p_large]
    p_raw = [0.04, 0.01, 0.3]
    corrected = _holm_correct(p_raw)

    # Sorted ascending: 0.01, 0.04, 0.3
    # Multiplied by:       3,    2,   1
    # Clipped:           0.03, 0.08, 0.3
    # Returned in original order: [0.08, 0.03, 0.3]
    assert len(corrected) == 3
    assert abs(corrected[0] - 0.08) < 1e-12, f"Expected 0.08, got {corrected[0]}"
    assert abs(corrected[1] - 0.03) < 1e-12, f"Expected 0.03, got {corrected[1]}"
    assert abs(corrected[2] - 0.30) < 1e-12, f"Expected 0.30, got {corrected[2]}"


def test_holm_correct_clips_at_1():
    """AC3 — corrected p-values are capped at 1.0."""
    p_raw = [0.5, 0.6, 0.9]
    corrected = _holm_correct(p_raw)
    assert all(c <= 1.0 for c in corrected)


def test_main_table_p_values_are_holm_corrected(tmp_path):
    """The wilcoxon_p_vs_STATIC column reflects Holm correction (>= raw p)."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _write_runs(_build_main_runs(), str(runs_dir))

    runs = load_runs(str(runs_dir))
    df = compute_main_table(runs)
    df_uf = df[df["metric"] == "heldout_uniform_fitness"]

    static_runs = [r for r in runs if r["mode"] == "STATIC" and 1000 <= r["seed"] <= 1014]
    static_by_seed = {r["seed"]: r["heldout_score"] for r in static_runs}

    for m in ["most_played_baseline", "COEVOLVE_FROZEN", "COEVOLVE"]:
        m_runs = [r for r in runs if r["mode"] == m and 1000 <= r["seed"] <= 1014]
        m_by_seed = {r["seed"]: r["heldout_score"] for r in m_runs}
        seeds = sorted(set(m_by_seed) & set(static_by_seed))
        x = [m_by_seed[s] for s in seeds]
        y = [static_by_seed[s] for s in seeds]
        diffs = [xi - yi for xi, yi in zip(x, y)]
        if all(d == 0.0 for d in diffs):
            continue
        _, raw_p = stats.wilcoxon(x, y, zero_method="wilcox")

        table_p = df_uf.loc[df_uf["method"] == m, "wilcoxon_p_vs_STATIC"].iloc[0]
        assert table_p >= raw_p - 1e-12, (
            f"{m}: corrected p ({table_p:.6f}) < raw p ({raw_p:.6f})"
        )


# ── AC4: A12 formula ──────────────────────────────────────────────────────────

def test_a12_all_x_greater():
    """AC4 — A12 = 1.0 when every x > every y."""
    x = [1.0, 2.0, 3.0]
    y = [0.0, 0.0, 0.0]
    assert abs(_a12(x, y) - 1.0) < 1e-12


def test_a12_all_equal():
    """AC4 — A12 = 0.5 when all values are equal (all ties)."""
    x = [1.0, 1.0, 1.0]
    y = [1.0, 1.0, 1.0]
    assert abs(_a12(x, y) - 0.5) < 1e-12


def test_a12_empty():
    """AC4 — A12 is NaN when either input is empty."""
    assert np.isnan(_a12([], [1.0]))
    assert np.isnan(_a12([1.0], []))


def test_a12_known_value():
    """AC4 — hand-computed case: x=[1,2], y=[0,1] → wins=3, ties=1, n=4."""
    x = [1.0, 2.0]
    y = [0.0, 1.0]
    # pairs: (1,0)→win, (1,1)→tie, (2,0)→win, (2,1)→win  → wins=3 ties=1 n=4
    expected = (3 + 0.5 * 1) / 4  # = 0.875
    assert abs(_a12(x, y) - expected) < 1e-12


def test_main_table_a12_in_range(tmp_path):
    """A12 values in main_table.csv are in [0, 1] for non-STATIC methods."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _write_runs(_build_main_runs(), str(runs_dir))

    df = compute_main_table(load_runs(str(runs_dir)))
    non_static = df[df["method"] != "STATIC"]
    for _, row in non_static.iterrows():
        val = row["A12_vs_STATIC"]
        if np.isnan(val):
            continue
        assert 0.0 <= val <= 1.0, f"{row['method']}/{row['metric']}: A12={val} out of [0,1]"


# ── AC5: Sensitivity table ────────────────────────────────────────────────────

def test_sensitivity_table_structure(tmp_path):
    """sensitivity_table.csv has (metric, method, lambda) rows with mean/std."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _write_runs(_build_sensitivity_runs(), str(runs_dir))

    df = compute_sensitivity_table(load_runs(str(runs_dir)))

    assert {"metric", "method", "lambda_weight", "mean", "std"}.issubset(set(df.columns))

    df_uf = df[df["metric"] == "heldout_uniform_fitness"]
    for m in ["STATIC", "COEVOLVE"]:
        for lam in [0.0, 1.0, 2.0]:
            row = df_uf[(df_uf["method"] == m) & (df_uf["lambda_weight"] == lam)]
            assert len(row) == 1, f"Missing row for method={m} lambda={lam}"
            assert np.isfinite(row.iloc[0]["mean"])
            assert np.isfinite(row.iloc[0]["std"])


# ── AC6: Diagnostic table — COEVOLVE only ────────────────────────────────────

def test_diagnostic_table_coevolve_only(tmp_path):
    """
    AC6 — diagnostic_table.csv is derived exclusively from COEVOLVE runs and
    contains one row per generation with the three required metric columns.
    """
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _write_runs(_build_main_runs(), str(runs_dir))

    df = compute_diagnostic_table(load_runs(str(runs_dir)))

    required_cols = {"gen", "repertoire_diversity", "opponent_diversity", "hof_size"}
    assert required_cols.issubset(set(df.columns))
    assert len(df) > 0, "diagnostic_table should have at least one row"

    # Opponent diversity must be finite for COEVOLVE (not None/NaN)
    assert df["opponent_diversity"].notna().all(), (
        "opponent_diversity has NaN — COEVOLVE runs must populate it"
    )

    # hof_size must be non-negative
    assert (df["hof_size"] >= 0).all()


def test_diagnostic_table_empty_when_no_coevolve(tmp_path):
    """Empty DataFrame is returned when there are no COEVOLVE runs in scope."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    static_only = [_make_run("STATIC", s, 1.0, 0.5) for s in range(1000, 1005)]
    _write_runs(static_only, str(runs_dir))

    df = compute_diagnostic_table(load_runs(str(runs_dir)))
    assert len(df) == 0


# ── AC7: Convergence PNG created ─────────────────────────────────────────────

def test_convergence_png_created(tmp_path):
    """
    AC7 — plot_convergence writes a non-empty PNG file to the given path.
    """
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _write_runs(_build_main_runs(), str(runs_dir))

    out_png = tmp_path / "convergence.png"
    plot_convergence(load_runs(str(runs_dir)), str(out_png))

    assert out_png.exists(), "convergence.png was not created"
    assert out_png.stat().st_size > 0, "convergence.png is empty"


def test_convergence_png_created_via_run_analysis(tmp_path):
    """
    AC7 (via run_analysis) — the full run_analysis pipeline also produces
    convergence.png.
    """
    runs_dir    = tmp_path / "runs"
    results_dir = tmp_path / "results"
    runs_dir.mkdir()
    _write_runs(_build_main_runs() + _build_sensitivity_runs(), str(runs_dir))

    run_analysis(str(runs_dir), str(results_dir))

    assert (results_dir / "convergence.png").exists()
    assert (results_dir / "convergence.png").stat().st_size > 0


# ── load_runs determinism ─────────────────────────────────────────────────────

def test_load_runs_deterministic(tmp_path):
    """
    Helper test — load_runs returns runs in the same order on repeated calls
    (relies on sorted glob).
    """
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    all_runs = _build_main_runs(seeds=range(1000, 1005))
    _write_runs(all_runs, str(runs_dir))

    order_1 = [r["seed"] for r in load_runs(str(runs_dir))]
    order_2 = [r["seed"] for r in load_runs(str(runs_dir))]
    assert order_1 == order_2
