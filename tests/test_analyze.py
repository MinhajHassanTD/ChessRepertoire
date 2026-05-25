"""Tests for C10 — Analysis (src/analyze.py).

Focuses on the pure statistical helpers and the public table-building
functions. Uses synthetic in-memory runs only; no real data files needed.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pytest

from src.analyze import (
    HELDOUT_METRICS,
    MAIN_SEEDS,
    METHODS_MAIN,
    _a12,
    _holm_correct,
    compute_pairwise_table,
    compute_runtime_table,
    compute_summary_table,
    load_runs,
)


# ── Synthetic run helpers ────────────────────────────────────────────────────

def _make_run(mode: str, seed: int, score: float, lam: float = 1.0,
              wall: float = 1.0) -> dict:
    return {
        "mode": mode,
        "config": {"lambda_weight": lam, "alpha": 1 / 3},
        "seed": seed,
        "git_commit": "deadbeef",
        "history": [],
        "final_best_candidate": {
            "white_committed": {}, "white_reached": [],
            "black_committed": {}, "black_reached": [],
        },
        "final_training_fitness": score,
        "heldout_score": score,
        # Pre-populate heldout_metrics so _augment_runs_with_metrics is a no-op.
        "heldout_metrics": {
            "heldout_uniform_mean": score,
            "heldout_worst_band":   score - 0.01,
            "white_band_scores":    {},  # marker that no recomputation is needed
        },
        "wall_time_seconds": wall,
    }


def _write_runs(runs, runs_dir):
    for r in runs:
        lam = r["config"]["lambda_weight"]
        fname = f"{r['mode']}_l{lam}_s{r['seed']}.pkl"
        with open(os.path.join(runs_dir, fname), "wb") as fh:
            pickle.dump(r, fh)


# ── _a12 (Vargha–Delaney) ────────────────────────────────────────────────────

def test_a12_all_x_greater():
    assert _a12([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]) == 1.0


def test_a12_all_x_less():
    assert _a12([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == 0.0


def test_a12_all_equal():
    assert _a12([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]) == 0.5


def test_a12_empty_returns_nan():
    assert np.isnan(_a12([], [1.0]))
    assert np.isnan(_a12([1.0], []))


def test_a12_known_value():
    # x=[1,2], y=[0,1] → pairs: (1>0), (1=1), (2>0), (2>1) → wins=3 ties=1 n=4
    assert _a12([1.0, 2.0], [0.0, 1.0]) == (3 + 0.5 * 1) / 4


# ── _holm_correct ─────────────────────────────────────────────────────────────

def test_holm_correct_k3_known_values():
    # p_raw in arbitrary order; sorted ascending: 0.01, 0.04, 0.3
    # multiply by 3,2,1 -> 0.03, 0.08, 0.3 -> returned in original order
    corrected = _holm_correct([0.04, 0.01, 0.3])
    assert abs(corrected[0] - 0.08) < 1e-12
    assert abs(corrected[1] - 0.03) < 1e-12
    assert abs(corrected[2] - 0.30) < 1e-12


def test_holm_correct_clips_at_1():
    assert all(c <= 1.0 for c in _holm_correct([0.5, 0.6, 0.9]))


def test_holm_correct_empty():
    assert _holm_correct([]) == []


# ── compute_summary_table ─────────────────────────────────────────────────────

def _build_main_runs():
    """One run per (method, seed) with score = 0.50 + small offset per method."""
    rng = np.random.default_rng(0)
    offsets = {m: i * 0.005 for i, m in enumerate(METHODS_MAIN)}
    runs = []
    for s in MAIN_SEEDS:
        for m in METHODS_MAIN:
            runs.append(_make_run(m, s, 0.50 + offsets[m] + float(rng.normal(0, 0.001))))
    return runs


def test_summary_table_has_required_columns():
    df = compute_summary_table(_build_main_runs())
    assert {"metric", "method", "n", "mean", "std", "median"}.issubset(df.columns)


def test_summary_table_one_row_per_method_metric():
    df = compute_summary_table(_build_main_runs())
    assert len(df) == len(METHODS_MAIN) * len(HELDOUT_METRICS)


def test_summary_table_all_methods_present():
    df = compute_summary_table(_build_main_runs())
    assert set(df["method"]) == set(METHODS_MAIN)


def test_summary_table_n_equals_main_seeds():
    df = compute_summary_table(_build_main_runs())
    assert (df["n"] == len(MAIN_SEEDS)).all()


# ── compute_pairwise_table ────────────────────────────────────────────────────

def test_pairwise_table_has_expected_columns():
    df = compute_pairwise_table(_build_main_runs())
    expected = {"metric", "method_a", "method_b", "mean_a", "mean_b",
                "delta", "wilcoxon_p", "A12", "significant_p05"}
    assert expected.issubset(df.columns)


def test_pairwise_table_a12_in_unit_interval():
    df = compute_pairwise_table(_build_main_runs())
    finite = df.dropna(subset=["A12"])
    assert ((finite["A12"] >= 0.0) & (finite["A12"] <= 1.0)).all()


# ── compute_runtime_table ─────────────────────────────────────────────────────

def test_runtime_table_includes_methods():
    runs = _build_main_runs()
    df = compute_runtime_table(runs)
    # Just assert it returns a non-empty frame with a method column
    assert not df.empty
    assert "method" in df.columns or df.index.name == "method"


# ── load_runs ─────────────────────────────────────────────────────────────────

def test_load_runs_returns_all_pickles(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    runs = _build_main_runs()[:5]
    _write_runs(runs, str(runs_dir))
    loaded = load_runs(str(runs_dir))
    assert len(loaded) == 5


def test_load_runs_deterministic_order(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _write_runs(_build_main_runs()[:5], str(runs_dir))
    a = [(r["mode"], r["seed"]) for r in load_runs(str(runs_dir))]
    b = [(r["mode"], r["seed"]) for r in load_runs(str(runs_dir))]
    assert a == b
