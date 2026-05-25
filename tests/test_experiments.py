"""Tests for C9 — Experiment runner (src/experiments.py).

Verifies the experiment matrix structure and run_all's resume + git-gate
behavior via mocks; no real GA runs are executed.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.experiments import (
    ALL_ABLATION_EXPERIMENTS,
    ALL_EXPERIMENTS,
    BASELINE_EXPERIMENTS,
    BASELINE_METHODS,
    CLOSURE_ABLATION_EXPERIMENTS,
    CLOSURE_ABLATION_METHODS,
    MAIN_EXPERIMENTS,
    MAIN_METHODS,
    MAIN_SEEDS,
    check_clean_git,
    run_all,
    run_filename,
)


def _fake_result(method="STATIC", seed=1000, lam=1.0) -> dict:
    return {
        "mode": method,
        "config": {"lambda_weight": lam, "alpha": 1 / 3},
        "seed": seed,
        "git_commit": "abc123",
        "history": [],
        "final_best_candidate": {
            "white_committed": {}, "white_reached": [],
            "black_committed": {}, "black_reached": [],
        },
        "final_training_fitness": 0.5,
        "heldout_score": 0.48,
        "wall_time_seconds": 1.0,
    }


# ── Run matrix structure ─────────────────────────────────────────────────────

def test_main_experiments_count():
    assert len(MAIN_EXPERIMENTS) == len(MAIN_METHODS) * len(MAIN_SEEDS)


def test_baseline_experiments_count():
    assert len(BASELINE_EXPERIMENTS) == len(BASELINE_METHODS) * len(MAIN_SEEDS)


def test_closure_ablation_count():
    assert len(CLOSURE_ABLATION_EXPERIMENTS) == len(CLOSURE_ABLATION_METHODS) * len(MAIN_SEEDS)


def test_all_experiments_is_union():
    assert len(ALL_EXPERIMENTS) == len(MAIN_EXPERIMENTS) + len(BASELINE_EXPERIMENTS)


def test_all_ablation_is_union():
    assert len(ALL_ABLATION_EXPERIMENTS) == len(CLOSURE_ABLATION_EXPERIMENTS)


def test_main_methods_are_expected():
    methods = {r["method"] for r in MAIN_EXPERIMENTS}
    assert methods == set(MAIN_METHODS)


def test_main_seeds_are_expected():
    seeds = {r["seed"] for r in MAIN_EXPERIMENTS}
    assert seeds == set(MAIN_SEEDS)


def test_every_run_has_required_keys():
    required = {"method", "seed", "lambda_weight", "alpha"}
    for run in ALL_EXPERIMENTS + ALL_ABLATION_EXPERIMENTS:
        assert required <= run.keys()


def test_filenames_unique_across_all_experiments():
    fnames = [
        run_filename(r["method"], r["lambda_weight"], r["seed"])
        for r in ALL_EXPERIMENTS + ALL_ABLATION_EXPERIMENTS
    ]
    assert len(fnames) == len(set(fnames))


def test_filename_format():
    assert run_filename("STATIC", 1.0, 1000) == os.path.join("runs", "STATIC_l1.0_s1000.pkl")
    assert run_filename("COEVOLVE", 2.0, 2003) == os.path.join("runs", "COEVOLVE_l2.0_s2003.pkl")


def test_filename_honors_runs_dir():
    assert run_filename("STATIC", 1.0, 1000, "custom") == os.path.join("custom", "STATIC_l1.0_s1000.pkl")


# ── check_clean_git ──────────────────────────────────────────────────────────

_GIT_PATCH = "src.experiments.subprocess.check_output"


def test_check_clean_git_exits_on_dirty_tree(monkeypatch):
    monkeypatch.delenv("SKIP_GIT_CHECK", raising=False)
    def _dirty(cmd, **kw):
        if "--porcelain" in cmd:
            return b" M dirty_file.py\n"
        return b"deadbeef\n"
    with patch(_GIT_PATCH, side_effect=_dirty), pytest.raises(SystemExit) as exc:
        check_clean_git()
    assert exc.value.code != 0


def test_check_clean_git_passes_on_clean_tree(monkeypatch):
    monkeypatch.delenv("SKIP_GIT_CHECK", raising=False)
    def _clean(cmd, **kw):
        if "--porcelain" in cmd:
            return b""
        return b"deadbeef\n"
    with patch(_GIT_PATCH, side_effect=_clean):
        check_clean_git()  # must not raise


# ── run_all: resume behavior and git gate ────────────────────────────────────

_PATCH_LOAD_GRAPH = "src.experiments.load_graph"
_PATCH_LOAD_POLICIES = "src.experiments.load_policies"
_PATCH_LOAD_EVAL_CACHE = "src.experiments.load_eval_cache"
_PATCH_RUN_COEVO = "src.experiments.run_coevolution"
_PATCH_RUN_BASELINE = "src.experiments.run_baseline"
_PATCH_RUN_RANDOM = "src.experiments.run_random_search"
_PATCH_RUN_HILLCLIMB = "src.experiments.run_greedy_hillclimb"


def _mock_data_loaders():
    return (
        patch(_PATCH_LOAD_GRAPH, return_value={}),
        patch(_PATCH_LOAD_POLICIES, return_value={}),
        patch(_PATCH_LOAD_EVAL_CACHE, return_value={}),
    )


def test_run_all_skips_existing_files(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    ga_runs = [r for r in MAIN_EXPERIMENTS if r["method"] == "STATIC"][:5]
    for r in ga_runs[:2]:
        path = Path(run_filename(r["method"], r["lambda_weight"], r["seed"], str(runs_dir)))
        with open(path, "wb") as fh:
            pickle.dump(_fake_result(), fh)

    coevo_mock = MagicMock(
        side_effect=lambda mode, config, seed, **kw:
            _fake_result(mode, seed, config["lambda_weight"])
    )
    g, p, e = _mock_data_loaders()
    with g, p, e, \
         patch(_PATCH_RUN_COEVO, coevo_mock), \
         patch(_PATCH_RUN_BASELINE, MagicMock(return_value=_fake_result())), \
         patch(_PATCH_RUN_RANDOM, MagicMock(return_value=_fake_result())), \
         patch(_PATCH_RUN_HILLCLIMB, MagicMock(return_value=_fake_result())):
        run_all(data_dir=str(tmp_path), runs_dir=str(runs_dir),
                experiments=ga_runs, skip_git_check=True)

    assert coevo_mock.call_count == 3


def test_run_all_creates_files_when_clean(tmp_path):
    runs_dir = tmp_path / "runs"
    ga_runs = [r for r in MAIN_EXPERIMENTS if r["method"] == "STATIC"][:3]

    coevo_mock = MagicMock(
        side_effect=lambda mode, config, seed, **kw:
            _fake_result(mode, seed, config["lambda_weight"])
    )
    g, p, e = _mock_data_loaders()
    with g, p, e, patch(_PATCH_RUN_COEVO, coevo_mock):
        run_all(data_dir=str(tmp_path), runs_dir=str(runs_dir),
                experiments=ga_runs, skip_git_check=True)

    assert coevo_mock.call_count == 3
    assert len(list(runs_dir.glob("*.pkl"))) == 3


def test_run_all_refuses_on_dirty_git(tmp_path, monkeypatch):
    monkeypatch.delenv("SKIP_GIT_CHECK", raising=False)
    def _dirty(cmd, **kw):
        if "--porcelain" in cmd:
            return b" M src/x.py\n"
        return b"deadbeef\n"
    coevo_mock = MagicMock()
    g, p, e = _mock_data_loaders()
    with g, p, e, \
         patch(_GIT_PATCH, side_effect=_dirty), \
         patch(_PATCH_RUN_COEVO, coevo_mock), \
         pytest.raises(SystemExit):
        run_all(data_dir=str(tmp_path), runs_dir=str(tmp_path / "runs"))
    coevo_mock.assert_not_called()
