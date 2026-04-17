"""
Pilot: end-to-end smoke test.
pop_size=10, generations=5, seed=42.
Runs each mode twice and checks bit-identical reproducibility (excluding wall time).
"""

import pickle, time, sys, os
sys.path.insert(0, os.path.dirname(__file__))


class _Tee:
    """Write to both the original stdout and a file simultaneously."""
    def __init__(self, file):
        self._file = file
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

import numpy as np
from src.coevolution import run_coevolution
from src.eval_cache import load_eval_cache
from src.fitness import walk
from src.graph import load_graph
from src.policies import load_policies
from src.repertoire import Candidate, Repertoire

DATA = "data"

PILOT_CONFIG = {
    "lambda_weight": 1.0,
    "alpha": 1 / 3,
    "pop_size_repertoires": 10,
    "pop_size_opponents": 10,
    "n_generations": 5,
    "tournament_size": 3,
    "crossover_rate": 0.7,
    "mutation_rate": 0.3,
    "hof_size": 5,
}

MODES = ["STATIC", "COEVOLVE_FROZEN", "COEVOLVE"]


BANDS = ["1600-1799", "1800-1999", "2000-2199"]


def print_fitness_breakdown(serialized_candidate, graph_train, eval_train, base_policies):
    """Reconstruct best candidate and print per-color per-band scores, separated."""
    sc = serialized_candidate
    white_rep = Repertoire(color="white", committed=sc["white_committed"],
                           reached=set(sc["white_reached"]), graph=graph_train)
    black_rep = Repertoire(color="black", committed=sc["black_committed"],
                           reached=set(sc["black_reached"]), graph=graph_train)
    best = Candidate(white=white_rep, black=black_rep, fitness=None, band_scores_cache=None)

    print(f"  {'Band':<13} {'White':>8} {'Black':>8} {'Combined':>10}")
    print(f"  {'-'*46}")
    white_per_band = []
    black_per_band = []
    combined_per_band = []
    for band in BANDS:
        w = walk(best.white, band, eval_train, base_policies, graph_train)
        b = 1.0 - walk(best.black, band, eval_train, base_policies, graph_train)
        c = 0.5 * w + 0.5 * b
        white_per_band.append(w)
        black_per_band.append(b)
        combined_per_band.append(c)
        print(f"  {band:<13} {w:>8.4f} {b:>8.4f} {c:>10.4f}")
    print(f"  {'-'*46}")
    mean_w = np.mean(white_per_band)
    mean_b = np.mean(black_per_band)
    mean_c = np.mean(combined_per_band)
    cvar_w = min(white_per_band)
    cvar_b = min(black_per_band)
    cvar   = min(combined_per_band)
    lam = PILOT_CONFIG["lambda_weight"]
    fit_w = mean_w + lam * cvar_w
    fit_b = mean_b + lam * cvar_b
    fitness = mean_c + lam * cvar
    print(f"  {'White score':<13}  mean={mean_w:.4f}  CVaR={cvar_w:.4f}  fitness={fit_w:.4f}")
    print(f"  {'Black score':<13}  mean={mean_b:.4f}  CVaR={cvar_b:.4f}  fitness={fit_b:.4f}")
    print(f"  {'Combined':<13}  mean={mean_c:.4f}  CVaR={cvar:.4f}  fitness={fitness:.4f}")


def _rep_lines(rep, graph) -> list[str]:
    """
    Convert a Repertoire into a human-readable move tree.

    Format (ply // 2 gives indent level):
      1. e4          <- white committed move (ply 0, indent 0)
      1... e5        <- opponent covered response (ply 1, indent 0)
        2. Nf3       <- our next committed move (ply 2, indent 1)
        2... Nc6     <- opponent response (ply 3, indent 1)
          3. Bc4 *   <- leaf (no further committed move below)

    * = leaf node (no committed move from here).
    [= transposition] = position already shown earlier via different path.
    """
    lines = []
    visited: set = set()

    def dfs(fen: str) -> None:
        if fen not in graph["nodes"]:
            return
        if fen in visited:
            node = graph["nodes"][fen]
            ply = node["ply_depth"]
            indent = "  " * (ply // 2)
            lines.append(f"{indent}[= transposition, already shown]")
            return
        visited.add(fen)

        node = graph["nodes"][fen]
        ply = node["ply_depth"]
        is_our = node["turn"] == rep.color
        indent = "  " * (ply // 2)
        move_num = ply // 2 + 1

        if is_our:
            if fen not in rep.committed:
                return                          # uncommitted leaf — stop here
            uci = rep.committed[fen]
            ci = node["children"].get(uci)
            if ci is None:
                return
            san = ci["move_san"]
            child_fen = ci["child_fen"]
            is_leaf = child_fen not in rep.reached or child_fen not in graph["nodes"]
            tag = " *" if is_leaf else ""
            lines.append(f"{indent}{move_num}. {san}{tag}")
            if not is_leaf:
                dfs(child_fen)
        else:
            # Opponent turn: every covered response is its own branch
            covered = [
                (ci["move_san"], ci["child_fen"])
                for ci in node["children"].values()
                if ci["child_fen"] in rep.reached
            ]
            for san, child_fen in covered:
                tag = " *" if child_fen not in graph["nodes"] else ""
                lines.append(f"{indent}{move_num}... {san}{tag}")
                dfs(child_fen)

    dfs(graph["root_fen"])
    return lines


def print_subgraphs(serialized_candidate, graph):
    """Print white and black repertoire trees from a serialized candidate."""
    sc = serialized_candidate
    for color in ("white", "black"):
        rep = Repertoire(
            color=color,
            committed=sc[f"{color}_committed"],
            reached=set(sc[f"{color}_reached"]),
            graph=graph,
        )
        committed_n = len(rep.committed)
        reached_n   = len(rep.reached)
        print(f"  --- {color.upper()} REPERTOIRE"
              f"  (committed: {committed_n}, reached: {reached_n}) ---")
        for line in _rep_lines(rep, graph):
            print(f"  {line}")
        print()


def print_repertoire_for_player(serialized_candidate, graph, eval_cache, base_policies, color: str):
    """Print the move tree and per-band scores for a single player (white or black).

    Use this when a real user only plays one side and wants their personalized
    opening guide: every line they must know, plus how well it scores at each
    opponent rating band.

    color: 'white' or 'black'
    """
    assert color in ("white", "black"), "color must be 'white' or 'black'"
    sc = serialized_candidate
    rep = Repertoire(
        color=color,
        committed=sc[f"{color}_committed"],
        reached=set(sc[f"{color}_reached"]),
        graph=graph,
    )
    committed_n = len(rep.committed)
    reached_n   = len(rep.reached)

    print(f"  === REPERTOIRE FOR {color.upper()} PLAYER ===")
    print(f"  Committed moves: {committed_n}   Positions covered: {reached_n}")
    print()

    # Move tree
    print("  OPENING LINES  (* = leaf — no further committed move)")
    print("  " + "-" * 50)
    for line in _rep_lines(rep, graph):
        print(f"  {line}")
    print()

    # Per-band score for this player only
    print(f"  SCORES BY OPPONENT BAND  (higher = better for {color})")
    print(f"  {'Band':<13} {'Score':>8}")
    print(f"  {'-'*24}")
    scores = []
    for band in BANDS:
        ws = walk(rep, band, eval_cache, base_policies, graph)
        player_score = ws if color == "white" else 1.0 - ws
        scores.append(player_score)
        print(f"  {band:<13} {player_score:>8.4f}")
    print(f"  {'-'*24}")
    print(f"  {'mean':<13} {np.mean(scores):>8.4f}")
    print(f"  {'worst band':<13} {min(scores):>8.4f}")
    print()


def compare_results(r1: dict, r2: dict) -> tuple[bool, list[str]]:
    """Compare two run results excluding wall_time_seconds. Return (ok, issues)."""
    issues = []

    for key in ("mode", "seed", "final_training_fitness", "heldout_score"):
        v1, v2 = r1.get(key), r2.get(key)
        if v1 != v2:
            issues.append(f"  {key}: run1={v1}  run2={v2}")

    # Compare history gen-by-gen
    h1, h2 = r1.get("history", []), r2.get("history", [])
    if len(h1) != len(h2):
        issues.append(f"  history length: {len(h1)} vs {len(h2)}")
    else:
        for i, (g1, g2) in enumerate(zip(h1, h2)):
            for k in g1:
                if g1[k] != g2[k]:
                    issues.append(f"  history[gen={i}][{k}]: {g1[k]} vs {g2[k]}")

    # Compare final candidate
    c1 = r1.get("final_best_candidate", {})
    c2 = r2.get("final_best_candidate", {})
    if c1 != c2:
        issues.append("  final_best_candidate differs")

    return len(issues) == 0, issues


def main():
    os.makedirs("runs", exist_ok=True)
    log_path = "runs/pilot_output.txt"
    _log_file = open(log_path, "w", encoding="utf-8")
    _tee = _Tee(_log_file)
    sys.stdout = _tee

    print("Loading data …")
    graph_train    = load_graph(f"{DATA}/graph_train.pkl")
    graph_heldout  = load_graph(f"{DATA}/graph_heldout.pkl")
    base_policies  = load_policies(f"{DATA}/base_policies.pkl")
    eval_train     = load_eval_cache(f"{DATA}/eval_cache_train.pkl")
    eval_heldout   = load_eval_cache(f"{DATA}/eval_cache_heldout.pkl")
    print("Data loaded.\n")

    results = {}

    for mode in MODES:
        print(f"{'='*60}")
        print(f"MODE: {mode}")

        run1_start = time.time()
        r1 = run_coevolution(
            mode=mode, config=dict(PILOT_CONFIG), seed=42,
            graph_train=graph_train, graph_heldout=graph_heldout,
            base_policies_train=base_policies,
            eval_cache_train=eval_train, eval_cache_heldout=eval_heldout,
        )
        r1["wall_time_seconds"] = time.time() - run1_start

        run2_start = time.time()
        r2 = run_coevolution(
            mode=mode, config=dict(PILOT_CONFIG), seed=42,
            graph_train=graph_train, graph_heldout=graph_heldout,
            base_policies_train=base_policies,
            eval_cache_train=eval_train, eval_cache_heldout=eval_heldout,
        )
        r2["wall_time_seconds"] = time.time() - run2_start

        ok, issues = compare_results(r1, r2)

        print(f"  Final training fitness : {r1['final_training_fitness']:.6f}")
        print(f"  Heldout score          : {r1['heldout_score']:.6f}")
        print(f"  Wall time (run 1)      : {r1['wall_time_seconds']:.1f}s")
        print(f"  Wall time (run 2)      : {r2['wall_time_seconds']:.1f}s")
        print(f"  Reproducibility        : {'PASS' if ok else 'FAIL'}")
        if not ok:
            for iss in issues:
                print(iss)
        print(f"  --- Fitness breakdown (train, best candidate) ---")
        print_fitness_breakdown(r1["final_best_candidate"], graph_train, eval_train, base_policies)
        print_subgraphs(r1["final_best_candidate"], graph_train)
        print(f"  --- Player-specific views ---")
        print_repertoire_for_player(r1["final_best_candidate"], graph_train, eval_train, base_policies, "white")
        print_repertoire_for_player(r1["final_best_candidate"], graph_train, eval_train, base_policies, "black")

        results[mode] = {"run1": r1, "run2": r2, "reproducible": ok}

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'Mode':<20} {'Fitness':>12} {'Heldout':>10} {'Repro':>8}")
    for mode in MODES:
        r = results[mode]["run1"]
        repro = "PASS" if results[mode]["reproducible"] else "FAIL"
        print(f"{mode:<20} {r['final_training_fitness']:>12.6f} {r['heldout_score']:>10.6f} {repro:>8}")

    # Save pilot results
    with open("runs/pilot_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("\nPilot results saved to runs/pilot_results.pkl")
    print(f"Pilot output saved to {log_path}")

    all_pass = all(results[m]["reproducible"] for m in MODES)

    sys.stdout = _tee._stdout
    _log_file.close()

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
