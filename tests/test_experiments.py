"""
Tests for C9 — Experiment runner (src/experiments.py).

Acceptance criteria from Section E of BLUEPRINT.md:
  AC1: A clean run produces all 90 pickle files in runs/.
  AC2: Interrupting and restarting the runner does not redo completed runs.
  AC3: The runner refuses to start if there are uncommitted git changes.

All tests use mocks so no real data files or GA runs are needed.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from src.experiments import (
    ALL_EXPERIMENTS,
    BASELINE_EXPERIMENTS,
    MAIN_EXPERIMENTS,
    SENSITIVITY_EXPERIMENTS,
    check_clean_git,
    run_all,
    run_baseline,
    run_filename,
)

# ── Shared helpers ────────────────────────────────────────────────────────────

def _fake_result(method='STATIC', seed=1000, lam=1.0) -> dict:
    """A minimal fake result dict that mirrors run_coevolution's output."""
    return {
        'mode': method,
        'config': {'lambda_weight': lam, 'alpha': 1 / 3},
        'seed': seed,
        'git_commit': 'abc123',
        'history': [],
        'final_best_candidate': {
            'white_committed': {}, 'white_reached': [],
            'black_committed': {}, 'black_reached': [],
        },
        'final_training_fitness': 0.5,
        'heldout_score': 0.48,
        'wall_time_seconds': 1.0,
    }


def _make_toy_graph():
    """Minimal graph for baseline tests (same as test_coevolution)."""
    root = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
    pos_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -"
    pos_e4e5 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -"
    pos_e4c5 = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -"
    pos_d4 = "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq -"

    def _bs(g, w, d, l):
        return {"games": g, "wins": w, "draws": d, "losses": l}

    return {
        "root_fen": root,
        "nodes": {
            root: {
                "fen": root, "ply_depth": 0, "total_games": 1000,
                "score_raw": 0.52, "turn": "white",
                "band_stats": {
                    "1600-1799": _bs(300, 156, 90, 54),
                    "1800-1999": _bs(400, 208, 120, 72),
                    "2000-2199": _bs(300, 156, 90, 54),
                },
                "children": {
                    "e2e4": {"child_fen": pos_e4, "move_san": "e4",
                             "aggregate_count": 700,
                             "band_counts": {"1600-1799": 200, "1800-1999": 280, "2000-2199": 220}},
                    "d2d4": {"child_fen": pos_d4, "move_san": "d4",
                             "aggregate_count": 300,
                             "band_counts": {"1600-1799": 100, "1800-1999": 120, "2000-2199": 80}},
                },
            },
            pos_e4: {
                "fen": pos_e4, "ply_depth": 1, "total_games": 700,
                "score_raw": 0.51, "turn": "black",
                "band_stats": {
                    "1600-1799": _bs(200, 102, 60, 38),
                    "1800-1999": _bs(280, 143, 84, 53),
                    "2000-2199": _bs(220, 112, 66, 42),
                },
                "children": {
                    "e7e5": {"child_fen": pos_e4e5, "move_san": "e5",
                             "aggregate_count": 420,
                             "band_counts": {"1600-1799": 120, "1800-1999": 168, "2000-2199": 132}},
                    "c7c5": {"child_fen": pos_e4c5, "move_san": "c5",
                             "aggregate_count": 280,
                             "band_counts": {"1600-1799": 80, "1800-1999": 112, "2000-2199": 88}},
                },
            },
            pos_e4e5: {
                "fen": pos_e4e5, "ply_depth": 2, "total_games": 420,
                "score_raw": 0.50, "turn": "white",
                "band_stats": {
                    "1600-1799": _bs(120, 60, 36, 24),
                    "1800-1999": _bs(168, 84, 50, 34),
                    "2000-2199": _bs(132, 66, 40, 26),
                },
                "children": {},
            },
            pos_e4c5: {
                "fen": pos_e4c5, "ply_depth": 2, "total_games": 280,
                "score_raw": 0.53, "turn": "white",
                "band_stats": {
                    "1600-1799": _bs(80, 42, 24, 14),
                    "1800-1999": _bs(112, 59, 34, 19),
                    "2000-2199": _bs(88, 47, 26, 15),
                },
                "children": {},
            },
            pos_d4: {
                "fen": pos_d4, "ply_depth": 1, "total_games": 300,
                "score_raw": 0.52, "turn": "black",
                "band_stats": {
                    "1600-1799": _bs(100, 52, 30, 18),
                    "1800-1999": _bs(120, 62, 36, 22),
                    "2000-2199": _bs(80, 42, 24, 14),
                },
                "children": {},
            },
        },
    }


def _make_toy_eval_cache(graph):
    prior_mean = 0.52
    tau = 20
    bands = ["1600-1799", "1800-1999", "2000-2199"]
    scores = {}
    for fen, node in graph["nodes"].items():
        bd = {}
        for band in bands:
            stats = node["band_stats"][band]
            n = stats["games"]
            if n == 0:
                s = prior_mean
            else:
                raw = (stats["wins"] + 0.5 * stats["draws"]) / n
                s = (n * raw + tau * prior_mean) / (n + tau)
            bd[band] = s
        scores[fen] = bd
    return {"prior_mean": prior_mean, "scores": scores}


def _make_toy_base_policies(graph):
    alpha = 5.0
    bands = ["1600-1799", "1800-1999", "2000-2199"]
    policies = {b: {} for b in bands}
    for fen, node in graph["nodes"].items():
        children = node["children"]
        if not children:
            continue
        agg_total = sum(c["aggregate_count"] for c in children.values())
        if agg_total == 0:
            continue
        agg_prob = {m: c["aggregate_count"] / agg_total for m, c in children.items()}
        for band in bands:
            band_total = sum(c["band_counts"][band] for c in children.values())
            dist = {}
            for m, c in children.items():
                bc = c["band_counts"][band]
                dist[m] = (bc + alpha * agg_prob[m]) / (band_total + alpha)
            policies[band][fen] = dist
    return policies


# Patch targets (module-level imports in experiments.py)
_PATCH_GIT = 'src.experiments.subprocess.check_output'
_PATCH_LOAD_GRAPH = 'src.experiments.load_graph'
_PATCH_LOAD_POLICIES = 'src.experiments.load_policies'
_PATCH_LOAD_EVAL_CACHE = 'src.experiments.load_eval_cache'
_PATCH_RUN_COEVO = 'src.experiments.run_coevolution'
_PATCH_RUN_BASELINE = 'src.experiments.run_baseline'
_PATCH_RUN_RANDOM = 'src.experiments.run_random_search'
_PATCH_RUN_HILLCLIMB = 'src.experiments.run_greedy_hillclimb'


# ═══════════════════════════════════════════════════════════════════════════════
# Run matrix structure (prerequisite for AC1)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunMatrix:
    """Verify the experiment matrix is constructed exactly per spec."""

    def test_main_experiments_count(self):
        """4 methods × 15 seeds = 60 main runs."""
        assert len(MAIN_EXPERIMENTS) == 60

    def test_sensitivity_experiments_count(self):
        """2 methods × 3 lambdas × 5 seeds = 30 sensitivity runs."""
        assert len(SENSITIVITY_EXPERIMENTS) == 30

    def test_baseline_experiments_count(self):
        """2 non-GA baselines × 15 seeds = 30 baseline runs."""
        assert len(BASELINE_EXPERIMENTS) == 30

    def test_all_experiments_total_count(self):
        """Total run count must be exactly 120 (60 main + 30 baselines + 30 sensitivity)."""
        assert len(ALL_EXPERIMENTS) == 120

    def test_main_methods(self):
        """Main experiments must contain exactly the four specified methods."""
        methods = {r['method'] for r in MAIN_EXPERIMENTS}
        assert methods == {'most_played_baseline', 'STATIC', 'COEVOLVE_FROZEN', 'COEVOLVE'}

    def test_main_seeds(self):
        """Main experiments must cover seeds 1000–1014 for every method."""
        seeds = {r['seed'] for r in MAIN_EXPERIMENTS}
        assert seeds == set(range(1000, 1015))

    def test_sensitivity_methods(self):
        """Sensitivity experiments must contain only STATIC and COEVOLVE."""
        methods = {r['method'] for r in SENSITIVITY_EXPERIMENTS}
        assert methods == {'STATIC', 'COEVOLVE'}

    def test_sensitivity_lambda_values(self):
        """Sensitivity experiments must cover λ ∈ {0.0, 1.0, 2.0}."""
        lambdas = {r['lambda_weight'] for r in SENSITIVITY_EXPERIMENTS}
        assert lambdas == {0.0, 1.0, 2.0}

    def test_sensitivity_seeds(self):
        """Sensitivity experiments must cover seeds 2000–2004."""
        seeds = {r['seed'] for r in SENSITIVITY_EXPERIMENTS}
        assert seeds == set(range(2000, 2005))

    def test_all_runs_have_required_keys(self):
        """Every run dict must contain method, seed, lambda_weight, alpha."""
        for run in ALL_EXPERIMENTS:
            assert {'method', 'seed', 'lambda_weight', 'alpha'} <= run.keys()

    def test_all_filenames_unique(self):
        """No two runs should map to the same output filename."""
        fnames = [
            run_filename(r['method'], r['lambda_weight'], r['seed'])
            for r in ALL_EXPERIMENTS
        ]
        assert len(fnames) == len(set(fnames)), "Duplicate output filenames detected"

    def test_filename_format(self):
        """run_filename returns the expected pattern."""
        assert run_filename('STATIC', 1.0, 1000) == os.path.join(
            'runs', 'STATIC_l1.0_s1000.pkl'
        )
        assert run_filename('COEVOLVE', 2.0, 2003) == os.path.join(
            'runs', 'COEVOLVE_l2.0_s2003.pkl'
        )


# ═══════════════════════════════════════════════════════════════════════════════
# AC1 — Clean run produces all 90 pickle files
# ═══════════════════════════════════════════════════════════════════════════════

class TestAC1CleanRunProduces90Files:
    """
    AC1: A clean run produces all 90 pickle files in runs/.
    Verified via mocks: every experiment slot triggers a save, and the
    resulting files are readable pickles with the expected shape.
    """

    @pytest.fixture
    def mock_git_clean(self):
        """Simulate a clean git status and a known HEAD hash."""
        def _side_effect(cmd, **kw):
            if '--porcelain' in cmd:
                return b''           # clean tree
            if 'rev-parse' in cmd:
                return b'deadbeef\n'
            return b''
        with patch(_PATCH_GIT, side_effect=_side_effect):
            yield

    @pytest.fixture
    def mock_data_loaders(self):
        with patch(_PATCH_LOAD_GRAPH, return_value={}), \
             patch(_PATCH_LOAD_POLICIES, return_value={}), \
             patch(_PATCH_LOAD_EVAL_CACHE, return_value={}):
            yield

    def _patch_all_runners(self):
        """Context manager that stubs all four runner functions."""
        return (
            patch(_PATCH_RUN_COEVO, side_effect=lambda mode, config, seed, **kw:
                  _fake_result(mode, seed, config['lambda_weight'])),
            patch(_PATCH_RUN_BASELINE, side_effect=lambda run, *a, **kw:
                  _fake_result('most_played_baseline', run['seed'], run['lambda_weight'])),
            patch(_PATCH_RUN_RANDOM, side_effect=lambda run, *a, **kw:
                  _fake_result('RANDOM_SEARCH', run['seed'], run['lambda_weight'])),
            patch(_PATCH_RUN_HILLCLIMB, side_effect=lambda run, *a, **kw:
                  _fake_result('GREEDY_HILLCLIMB', run['seed'], run['lambda_weight'])),
        )

    def test_all_90_files_created(self, tmp_path, mock_git_clean, mock_data_loaders):
        """run_all must produce exactly 120 pickle files when starting fresh."""
        coevo_p, base_p, rand_p, hill_p = self._patch_all_runners()
        with coevo_p, base_p, rand_p, hill_p:
            run_all(data_dir=str(tmp_path), runs_dir=str(tmp_path / 'runs'))

        pkl_files = list((tmp_path / 'runs').glob('*.pkl'))
        assert len(pkl_files) == 120, (
            f"Expected 120 .pkl files, found {len(pkl_files)}"
        )

    def test_all_expected_filenames_present(self, tmp_path, mock_git_clean, mock_data_loaders):
        """Each expected output filename must exist after a clean run."""
        coevo_p, base_p, rand_p, hill_p = self._patch_all_runners()
        with coevo_p, base_p, rand_p, hill_p:
            run_all(data_dir=str(tmp_path), runs_dir=str(tmp_path / 'runs'))

        runs_dir = tmp_path / 'runs'
        for run in ALL_EXPERIMENTS:
            expected = Path(
                run_filename(run['method'], run['lambda_weight'], run['seed'],
                             str(runs_dir))
            )
            assert expected.exists(), f"Missing output file: {expected.name}"

    def test_each_pickle_is_loadable_dict(self, tmp_path, mock_git_clean, mock_data_loaders):
        """Every saved pickle must be a loadable dict."""
        coevo_p, base_p, rand_p, hill_p = self._patch_all_runners()
        with coevo_p, base_p, rand_p, hill_p:
            run_all(data_dir=str(tmp_path), runs_dir=str(tmp_path / 'runs'))

        for pkl in (tmp_path / 'runs').glob('*.pkl'):
            with open(pkl, 'rb') as fh:
                result = pickle.load(fh)
            assert isinstance(result, dict), f"{pkl.name} is not a dict"
            assert 'heldout_score' in result, f"{pkl.name} missing heldout_score"

    def test_baseline_method_does_not_call_run_coevolution(
            self, tmp_path, mock_git_clean, mock_data_loaders):
        """most_played_baseline runs must go through run_baseline, not run_coevolution."""
        coevo_mock = MagicMock(return_value=_fake_result())
        baseline_mock = MagicMock(
            side_effect=lambda run, *a, **kw:
                _fake_result('most_played_baseline', run['seed'], run['lambda_weight'])
        )
        # Only run baseline experiments to keep the test fast
        baseline_runs = [r for r in ALL_EXPERIMENTS if r['method'] == 'most_played_baseline']
        with patch(_PATCH_RUN_COEVO, coevo_mock), \
             patch(_PATCH_RUN_BASELINE, baseline_mock):
            run_all(data_dir=str(tmp_path), runs_dir=str(tmp_path / 'runs'),
                    experiments=baseline_runs)

        assert coevo_mock.call_count == 0, (
            f"run_coevolution called {coevo_mock.call_count} times for baseline runs"
        )
        assert baseline_mock.call_count == len(baseline_runs)

    def test_ga_methods_call_run_coevolution(
            self, tmp_path, mock_git_clean, mock_data_loaders):
        """STATIC / COEVOLVE_FROZEN / COEVOLVE runs must call run_coevolution."""
        coevo_mock = MagicMock(
            side_effect=lambda mode, config, seed, **kw:
                _fake_result(mode, seed, config['lambda_weight'])
        )
        # Use just a small subset for speed
        ga_runs = [r for r in MAIN_EXPERIMENTS if r['method'] == 'STATIC'][:3]
        with patch(_PATCH_RUN_COEVO, coevo_mock), \
             patch(_PATCH_RUN_BASELINE, MagicMock(return_value=_fake_result())):
            run_all(data_dir=str(tmp_path), runs_dir=str(tmp_path / 'runs'),
                    experiments=ga_runs)

        assert coevo_mock.call_count == len(ga_runs)


# ═══════════════════════════════════════════════════════════════════════════════
# AC2 — Interrupting and restarting does not redo completed runs
# ═══════════════════════════════════════════════════════════════════════════════

class TestAC2ResumeSkipsCompletedRuns:
    """
    AC2: If a run's output file already exists, the runner skips it without
    calling run_coevolution or run_baseline again.
    """

    @pytest.fixture
    def mock_git_clean(self):
        def _side_effect(cmd, **kw):
            if '--porcelain' in cmd:
                return b''
            if 'rev-parse' in cmd:
                return b'deadbeef\n'
            return b''
        with patch(_PATCH_GIT, side_effect=_side_effect):
            yield

    @pytest.fixture
    def mock_data_loaders(self):
        with patch(_PATCH_LOAD_GRAPH, return_value={}), \
             patch(_PATCH_LOAD_POLICIES, return_value={}), \
             patch(_PATCH_LOAD_EVAL_CACHE, return_value={}):
            yield

    def test_existing_file_skipped(self, tmp_path, mock_git_clean, mock_data_loaders):
        """A run whose output file exists must not call run_coevolution."""
        runs_dir = tmp_path / 'runs'
        runs_dir.mkdir()

        # Pre-create the output file for the first GA run
        first_ga = next(r for r in ALL_EXPERIMENTS if r['method'] == 'STATIC')
        pre_existing = Path(run_filename(
            first_ga['method'], first_ga['lambda_weight'], first_ga['seed'],
            str(runs_dir)
        ))
        with open(pre_existing, 'wb') as fh:
            pickle.dump(_fake_result(), fh)

        coevo_mock = MagicMock(
            side_effect=lambda mode, config, seed, **kw:
                _fake_result(mode, seed, config['lambda_weight'])
        )
        baseline_mock = MagicMock(
            side_effect=lambda run, **kw:
                _fake_result('most_played_baseline', run['seed'], run['lambda_weight'])
        )
        # Only test that the single pre-existing run is skipped
        with patch(_PATCH_RUN_COEVO, coevo_mock), \
             patch(_PATCH_RUN_BASELINE, baseline_mock):
            run_all(data_dir=str(tmp_path), runs_dir=str(runs_dir),
                    experiments=[first_ga])

        assert coevo_mock.call_count == 0, (
            "run_coevolution was called for a run whose output file already existed"
        )

    def test_pre_existing_file_not_overwritten(self, tmp_path, mock_git_clean, mock_data_loaders):
        """The content of a pre-existing pickle must be unchanged after a run."""
        runs_dir = tmp_path / 'runs'
        runs_dir.mkdir()

        first_ga = next(r for r in ALL_EXPERIMENTS if r['method'] == 'STATIC')
        pre_existing = Path(run_filename(
            first_ga['method'], first_ga['lambda_weight'], first_ga['seed'],
            str(runs_dir)
        ))
        sentinel_result = {'heldout_score': 99.0, 'sentinel': True}
        with open(pre_existing, 'wb') as fh:
            pickle.dump(sentinel_result, fh)

        with patch(_PATCH_RUN_COEVO,
                   return_value=_fake_result()), \
             patch(_PATCH_RUN_BASELINE,
                   return_value=_fake_result()):
            run_all(data_dir=str(tmp_path), runs_dir=str(runs_dir),
                    experiments=[first_ga])

        with open(pre_existing, 'rb') as fh:
            loaded = pickle.load(fh)
        assert loaded == sentinel_result, (
            "Pre-existing pickle was overwritten during resume"
        )

    def test_only_missing_runs_are_executed(self, tmp_path, mock_git_clean, mock_data_loaders):
        """With N pre-existing files out of M total, exactly M-N runs execute."""
        runs_dir = tmp_path / 'runs'
        runs_dir.mkdir()

        ga_runs = [r for r in MAIN_EXPERIMENTS if r['method'] == 'STATIC'][:5]
        # Pre-create 2 of them
        for run in ga_runs[:2]:
            path = Path(run_filename(run['method'], run['lambda_weight'],
                                     run['seed'], str(runs_dir)))
            with open(path, 'wb') as fh:
                pickle.dump(_fake_result(), fh)

        coevo_mock = MagicMock(
            side_effect=lambda mode, config, seed, **kw:
                _fake_result(mode, seed, config['lambda_weight'])
        )
        with patch(_PATCH_RUN_COEVO, coevo_mock), \
             patch(_PATCH_RUN_BASELINE, MagicMock(return_value=_fake_result())):
            run_all(data_dir=str(tmp_path), runs_dir=str(runs_dir),
                    experiments=ga_runs)

        assert coevo_mock.call_count == 3, (
            f"Expected 3 calls (5 total - 2 pre-existing), got {coevo_mock.call_count}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# AC3 — Runner refuses to start with uncommitted git changes
# ═══════════════════════════════════════════════════════════════════════════════

class TestAC3RefusesOnDirtyGit:
    """
    AC3: The runner must call sys.exit (non-zero) when git status --porcelain
    returns any output, and must not load data or start any runs.
    """

    def test_refuses_when_git_is_dirty(self, tmp_path):
        """run_all must sys.exit when there are uncommitted changes."""
        def _dirty_git(cmd, **kw):
            if '--porcelain' in cmd:
                return b' M src/some_file.py\n'
            return b'deadbeef\n'

        with patch(_PATCH_GIT, side_effect=_dirty_git), \
             patch(_PATCH_LOAD_GRAPH) as load_mock, \
             pytest.raises(SystemExit) as exc_info:
            run_all(data_dir=str(tmp_path), runs_dir=str(tmp_path / 'runs'))

        assert exc_info.value.code != 0, "sys.exit should be called with non-zero code"
        load_mock.assert_not_called()

    def test_no_runs_executed_when_git_dirty(self, tmp_path):
        """No coevolution or baseline calls must happen on a dirty git tree."""
        def _dirty_git(cmd, **kw):
            if '--porcelain' in cmd:
                return b'?? untracked_file.py\n'
            return b'deadbeef\n'

        coevo_mock = MagicMock()
        baseline_mock = MagicMock()

        with patch(_PATCH_GIT, side_effect=_dirty_git), \
             patch(_PATCH_RUN_COEVO, coevo_mock), \
             patch(_PATCH_RUN_BASELINE, baseline_mock), \
             patch(_PATCH_LOAD_GRAPH, return_value={}), \
             patch(_PATCH_LOAD_POLICIES, return_value={}), \
             patch(_PATCH_LOAD_EVAL_CACHE, return_value={}), \
             pytest.raises(SystemExit):
            run_all(data_dir=str(tmp_path), runs_dir=str(tmp_path / 'runs'))

        coevo_mock.assert_not_called()
        baseline_mock.assert_not_called()

    def test_check_clean_git_exits_on_dirty_tree(self):
        """check_clean_git() itself must raise SystemExit on dirty tree."""
        with patch(_PATCH_GIT, return_value=b' M dirty_file.py\n'), \
             pytest.raises(SystemExit) as exc_info:
            check_clean_git()

        assert exc_info.value.code != 0

    def test_check_clean_git_passes_on_clean_tree(self):
        """check_clean_git() must not raise on a clean tree."""
        def _clean(cmd, **kw):
            if '--porcelain' in cmd:
                return b''
            return b'deadbeef\n'

        # Must not raise
        with patch(_PATCH_GIT, side_effect=_clean):
            check_clean_git()

    def test_proceeds_with_clean_git(self, tmp_path):
        """run_all must proceed normally when git status returns empty output."""
        def _clean_git(cmd, **kw):
            if '--porcelain' in cmd:
                return b''
            return b'deadbeef\n'

        coevo_mock = MagicMock(
            side_effect=lambda mode, config, seed, **kw:
                _fake_result(mode, seed, config['lambda_weight'])
        )
        one_run = [r for r in MAIN_EXPERIMENTS if r['method'] == 'STATIC'][:1]

        with patch(_PATCH_GIT, side_effect=_clean_git), \
             patch(_PATCH_LOAD_GRAPH, return_value={}), \
             patch(_PATCH_LOAD_POLICIES, return_value={}), \
             patch(_PATCH_LOAD_EVAL_CACHE, return_value={}), \
             patch(_PATCH_RUN_COEVO, coevo_mock), \
             patch(_PATCH_RUN_BASELINE, MagicMock(return_value=_fake_result())):
            run_all(data_dir=str(tmp_path), runs_dir=str(tmp_path / 'runs'),
                    experiments=one_run)

        assert coevo_mock.call_count == 1


# ═══════════════════════════════════════════════════════════════════════════════
# run_baseline unit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunBaseline:
    """Unit tests for the run_baseline helper function."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph = _make_toy_graph()
        self.eval_cache = _make_toy_eval_cache(self.graph)
        self.base_policies = _make_toy_base_policies(self.graph)

    @pytest.fixture
    def mock_git_hash(self):
        with patch(_PATCH_GIT, return_value=b'cafebabe\n'):
            yield

    def test_baseline_returns_required_keys(self, mock_git_hash):
        """run_baseline must return a dict with the expected keys."""
        required = {
            'mode', 'config', 'seed', 'git_commit',
            'history', 'final_best_candidate',
            'final_training_fitness', 'heldout_score', 'wall_time_seconds',
        }
        run = {'method': 'most_played_baseline', 'seed': 42,
               'lambda_weight': 1.0, 'alpha': 1 / 3}
        result = run_baseline(
            run, self.graph, self.graph,
            self.base_policies, self.eval_cache,
        )
        assert required <= result.keys()

    def test_baseline_mode_field(self, mock_git_hash):
        """mode field must be 'most_played_baseline'."""
        run = {'method': 'most_played_baseline', 'seed': 7,
               'lambda_weight': 1.0, 'alpha': 1 / 3}
        result = run_baseline(
            run, self.graph, self.graph,
            self.base_policies, self.eval_cache,
        )
        assert result['mode'] == 'most_played_baseline'

    def test_baseline_history_is_empty(self, mock_git_hash):
        """Baseline runs have no generational history (no GA)."""
        run = {'method': 'most_played_baseline', 'seed': 1,
               'lambda_weight': 1.0, 'alpha': 1 / 3}
        result = run_baseline(
            run, self.graph, self.graph,
            self.base_policies, self.eval_cache,
        )
        assert result['history'] == []

    def test_baseline_heldout_score_is_finite(self, mock_git_hash):
        """heldout_score must be a finite float."""
        import math
        run = {'method': 'most_played_baseline', 'seed': 3,
               'lambda_weight': 1.0, 'alpha': 1 / 3}
        result = run_baseline(
            run, self.graph, self.graph,
            self.base_policies, self.eval_cache,
        )
        assert math.isfinite(result['heldout_score'])

    def test_baseline_final_best_candidate_structure(self, mock_git_hash):
        """final_best_candidate must have the four serialization keys."""
        run = {'method': 'most_played_baseline', 'seed': 9,
               'lambda_weight': 1.0, 'alpha': 1 / 3}
        result = run_baseline(
            run, self.graph, self.graph,
            self.base_policies, self.eval_cache,
        )
        fbc = result['final_best_candidate']
        assert isinstance(fbc, dict)
        assert {'white_committed', 'white_reached',
                'black_committed', 'black_reached'} <= fbc.keys()

    def test_baseline_seed_stored(self, mock_git_hash):
        """The seed must be stored in the result dict."""
        run = {'method': 'most_played_baseline', 'seed': 1234,
               'lambda_weight': 1.0, 'alpha': 1 / 3}
        result = run_baseline(
            run, self.graph, self.graph,
            self.base_policies, self.eval_cache,
        )
        assert result['seed'] == 1234

    def test_baseline_deterministic(self, mock_git_hash):
        """Same seed must produce the same heldout_score."""
        run = {'method': 'most_played_baseline', 'seed': 55,
               'lambda_weight': 1.0, 'alpha': 1 / 3}
        r1 = run_baseline(run, self.graph, self.graph,
                          self.base_policies, self.eval_cache)
        r2 = run_baseline(run, self.graph, self.graph,
                          self.base_policies, self.eval_cache)
        assert r1['heldout_score'] == r2['heldout_score']
        assert r1['final_best_candidate'] == r2['final_best_candidate']
