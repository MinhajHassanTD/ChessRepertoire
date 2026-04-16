"""
C5 — Repertoire and Candidate
Chromosome representation and all genetic operators.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from src.graph import STARTING_FEN, aggregate_move_freq, children_of

# ── Constants ──────────────────────────────────────────────────────────────────

BUDGET = 20
CLOSURE_THRESHOLD = 0.05


# ── Exceptions ────────────────────────────────────────────────────────────────

class MutationFailed(Exception):
    """Raised when a mutation or crossover violates constraints."""


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Repertoire:
    color: str                        # 'white' or 'black'
    committed: dict                   # fen -> move_uci at our decision nodes
    reached: set                      # all positions in the subgraph
    graph: dict                       # reference to the position graph

    def copy(self) -> "Repertoire":
        return Repertoire(
            color=self.color,
            committed=dict(self.committed),
            reached=set(self.reached),
            graph=self.graph,
        )


@dataclass
class Candidate:
    white: Repertoire
    black: Repertoire
    fitness: Optional[float] = field(default=None)
    band_scores_cache: Optional[dict] = field(default=None)

    def copy(self) -> "Candidate":
        return Candidate(
            white=self.white.copy(),
            black=self.black.copy(),
            fitness=self.fitness,
            band_scores_cache=copy.deepcopy(self.band_scores_cache),
        )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _our_turn(rep: Repertoire, fen: str) -> bool:
    node = rep.graph["nodes"].get(fen)
    if node is None:
        return False
    return node["turn"] == rep.color


def _expand_to_closure(rep: Repertoire, frontier: list[str]) -> None:
    """
    BFS-expand from *frontier* positions: at opponent-turn nodes, add every
    child whose aggregate frequency >= CLOSURE_THRESHOLD; at our-turn nodes
    just add them to reached (the caller is responsible for committing).
    All newly reachable positions are added to rep.reached.
    """
    queue = deque(frontier)
    while queue:
        fen = queue.popleft()
        if fen not in rep.graph["nodes"]:
            continue
        node = rep.graph["nodes"][fen]
        if _our_turn(rep, fen):
            # Reached an our-turn position; expansion stops here.
            # (Caller decides whether to commit a move.)
            continue
        # Opponent turn: must cover all children with freq >= threshold.
        freq = aggregate_move_freq(rep.graph, fen)
        for move_uci, f in freq.items():
            if f >= CLOSURE_THRESHOLD:
                child_fen = node["children"][move_uci]["child_fen"]
                if child_fen not in rep.reached:
                    rep.reached.add(child_fen)
                    queue.append(child_fen)


def _expand_below_committed(rep: Repertoire, fen: str) -> None:
    """
    After committing a move at *fen*, expand the subtree beneath it to closure.
    This adds the child of the committed move, then does closure BFS from there.
    """
    move_uci = rep.committed[fen]
    node = rep.graph["nodes"][fen]
    child_fen = node["children"][move_uci]["child_fen"]
    if child_fen not in rep.reached:
        rep.reached.add(child_fen)
    _expand_to_closure(rep, [child_fen])


def _collect_descendants(rep: Repertoire, fen: str) -> set[str]:
    """
    BFS from *fen* over rep.reached edges to collect all descendants.
    Does NOT include *fen* itself.
    Handles DAG: a node is a descendant if reachable via reached edges from fen.
    """
    visited: set[str] = set()
    queue = deque([fen])
    while queue:
        cur = queue.popleft()
        node = rep.graph["nodes"].get(cur)
        if node is None:
            continue
        for move_uci, child_info in node["children"].items():
            child_fen = child_info["child_fen"]
            if child_fen in rep.reached and child_fen not in visited and child_fen != fen:
                visited.add(child_fen)
                queue.append(child_fen)
    return visited


def _still_reachable_without(rep: Repertoire, fen: str, excluded: set[str]) -> set[str]:
    """
    Return the set of positions reachable from root when the path through *fen*
    is cut — i.e., reachable without traversing any edge that leads into *excluded*.
    Used to handle DAG shared-ancestor edge cases.
    """
    root = rep.graph["root_fen"]
    visited: set[str] = set()
    queue = deque([root])
    while queue:
        cur = queue.popleft()
        if cur in visited:
            continue
        visited.add(cur)
        node = rep.graph["nodes"].get(cur)
        if node is None:
            continue
        for move_uci, child_info in node["children"].items():
            child_fen = child_info["child_fen"]
            if child_fen in rep.reached and child_fen not in visited:
                # Only follow committed edges at our-turn nodes
                if _our_turn(rep, cur):
                    if rep.committed.get(cur) == move_uci:
                        if child_fen not in excluded:
                            queue.append(child_fen)
                else:
                    if child_fen not in excluded:
                        queue.append(child_fen)
    return visited


def _remove_subtree(rep: Repertoire, below_fen: str) -> None:
    """
    Remove all descendants of *below_fen* that are not reachable from root
    via a path not passing through *below_fen*.  *below_fen* itself stays in
    reached (it becomes a leaf).
    """
    descendants = _collect_descendants(rep, below_fen)
    if not descendants:
        return

    # Which of those descendants are still reachable via another path?
    still_reachable = _still_reachable_without(rep, below_fen, {below_fen} | descendants)
    to_remove = descendants - still_reachable

    for d in to_remove:
        rep.reached.discard(d)
        rep.committed.pop(d, None)


# ── Public functions ──────────────────────────────────────────────────────────

def construct_initial(graph: dict, color: str, budget: int, rng) -> "Repertoire":
    """
    Greedy construction: at each our-turn node, pick the move with highest
    aggregate frequency.  Expand all opponent replies with aggregate frequency
    >= CLOSURE_THRESHOLD.  Stop expanding when budget is reached.
    """
    rep = Repertoire(color=color, committed={}, reached={STARTING_FEN}, graph=graph)
    _expand_to_closure(rep, [STARTING_FEN])

    queue: deque[str] = deque()
    # Seed queue with all our-turn positions currently in reached
    for fen in list(rep.reached):
        if _our_turn(rep, fen) and fen not in rep.committed:
            queue.append(fen)

    while queue and len(rep.committed) < budget:
        fen = queue.popleft()
        if fen not in rep.reached or fen in rep.committed:
            continue
        if not _our_turn(rep, fen):
            continue
        children = children_of(graph, fen)
        if not children:
            continue
        freq = aggregate_move_freq(graph, fen)
        if not freq:
            continue
        best_move = max(freq, key=lambda m: freq[m])
        rep.committed[fen] = best_move
        # Expand below the committed move
        child_fen = children[best_move]["child_fen"]
        if child_fen not in rep.reached:
            rep.reached.add(child_fen)
        prev_reached = set(rep.reached)
        _expand_to_closure(rep, [child_fen])
        # Enqueue newly discovered our-turn positions
        for nfen in rep.reached - prev_reached:
            if _our_turn(rep, nfen) and nfen not in rep.committed:
                queue.append(nfen)

    return rep


def construct_random(graph: dict, color: str, budget: int, rng) -> "Repertoire":
    """
    Like construct_initial, but at each our-turn node, pick a move via
    weighted random choice using aggregate frequencies as weights.
    """
    rep = Repertoire(color=color, committed={}, reached={STARTING_FEN}, graph=graph)
    _expand_to_closure(rep, [STARTING_FEN])

    queue: deque[str] = deque()
    for fen in list(rep.reached):
        if _our_turn(rep, fen) and fen not in rep.committed:
            queue.append(fen)

    while queue and len(rep.committed) < budget:
        fen = queue.popleft()
        if fen not in rep.reached or fen in rep.committed:
            continue
        if not _our_turn(rep, fen):
            continue
        children = children_of(graph, fen)
        if not children:
            continue
        freq = aggregate_move_freq(graph, fen)
        if not freq:
            continue
        moves = list(freq.keys())
        weights = [freq[m] for m in moves]
        total = sum(weights)
        weights = [w / total for w in weights]
        chosen = rng.choice(moves, p=weights)
        rep.committed[fen] = chosen
        child_fen = children[chosen]["child_fen"]
        if child_fen not in rep.reached:
            rep.reached.add(child_fen)
        prev_reached = set(rep.reached)
        _expand_to_closure(rep, [child_fen])
        for nfen in rep.reached - prev_reached:
            if _our_turn(rep, nfen) and nfen not in rep.committed:
                queue.append(nfen)

    return rep


def validate(rep: Repertoire) -> bool:
    """
    Check that the repertoire satisfies:
    1. Closure rule: every opponent-turn position in reached has every child
       with aggregate frequency >= CLOSURE_THRESHOLD also in reached.
    2. Within budget.
    3. Consistency: for every committed node, the child of the committed move
       is in reached.
    """
    # Budget
    if len(rep.committed) > BUDGET:
        return False

    for fen in rep.reached:
        if fen not in rep.graph["nodes"]:
            return False
        node = rep.graph["nodes"][fen]

        if _our_turn(rep, fen):
            # Consistency: if committed, child must be in reached
            if fen in rep.committed:
                move = rep.committed[fen]
                if move not in node["children"]:
                    return False
                child_fen = node["children"][move]["child_fen"]
                if child_fen not in rep.reached:
                    return False
        else:
            # Closure: all frequent opponent moves must be covered
            freq = aggregate_move_freq(rep.graph, fen)
            for move_uci, f in freq.items():
                if f >= CLOSURE_THRESHOLD:
                    child_info = node["children"].get(move_uci)
                    if child_info is None:
                        continue
                    child_fen = child_info["child_fen"]
                    if child_fen not in rep.reached:
                        return False

    return True


def mutate_move_swap(rep: Repertoire, rng) -> Repertoire:
    """
    Pick a random committed node, change its committed move to a different move
    weighted by aggregate frequency.  Remove descendants below the old move.
    Add descendants below the new move, expanding to closure.
    Raises MutationFailed if the result violates budget.
    """
    if not rep.committed:
        raise MutationFailed("No committed nodes to swap")

    new_rep = rep.copy()
    fens = list(new_rep.committed.keys())
    chosen_fen = rng.choice(fens)

    freq = aggregate_move_freq(new_rep.graph, chosen_fen)
    old_move = new_rep.committed[chosen_fen]
    alternatives = {m: f for m, f in freq.items() if m != old_move}
    if not alternatives:
        raise MutationFailed("No alternative moves available")

    moves = list(alternatives.keys())
    weights = [alternatives[m] for m in moves]
    total = sum(weights)
    weights = [w / total for w in weights]
    new_move = rng.choice(moves, p=weights)

    # Remove descendants below old move
    new_rep.committed.pop(chosen_fen)
    _remove_subtree(new_rep, chosen_fen)

    # Re-commit new move
    new_rep.committed[chosen_fen] = new_move
    node = new_rep.graph["nodes"][chosen_fen]
    child_fen = node["children"][new_move]["child_fen"]
    if child_fen not in new_rep.reached:
        new_rep.reached.add(child_fen)
    _expand_to_closure(new_rep, [child_fen])

    # Enqueue and commit new our-turn positions up to budget
    queue: deque[str] = deque()
    for fen in new_rep.reached:
        if _our_turn(new_rep, fen) and fen not in new_rep.committed:
            queue.append(fen)
    while queue and len(new_rep.committed) < BUDGET:
        fen = queue.popleft()
        if fen not in new_rep.reached or fen in new_rep.committed:
            continue
        f2 = aggregate_move_freq(new_rep.graph, fen)
        if not f2:
            continue
        best = max(f2, key=lambda m: f2[m])
        new_rep.committed[fen] = best
        ch = new_rep.graph["nodes"][fen]["children"][best]["child_fen"]
        if ch not in new_rep.reached:
            new_rep.reached.add(ch)
        prev = set(new_rep.reached)
        _expand_to_closure(new_rep, [ch])
        for nf in new_rep.reached - prev:
            if _our_turn(new_rep, nf) and nf not in new_rep.committed:
                queue.append(nf)

    if len(new_rep.committed) > BUDGET:
        raise MutationFailed("Budget exceeded after move swap")
    return new_rep


def mutate_extend(rep: Repertoire, rng) -> Repertoire:
    """
    Pick a random leaf (our-turn position in reached but not in committed).
    Commit a move there (weighted random choice).  Expand to closure.
    Raises MutationFailed if no budget headroom or no leaves exist.
    """
    if len(rep.committed) >= BUDGET:
        raise MutationFailed("Budget already full")

    leaves = [f for f in rep.reached if _our_turn(rep, f) and f not in rep.committed
              and children_of(rep.graph, f)]
    if not leaves:
        raise MutationFailed("No leaves to extend")

    new_rep = rep.copy()
    chosen_fen = rng.choice(leaves)
    freq = aggregate_move_freq(new_rep.graph, chosen_fen)
    if not freq:
        raise MutationFailed("No moves at leaf")

    moves = list(freq.keys())
    weights = [freq[m] for m in moves]
    total = sum(weights)
    weights = [w / total for w in weights]
    chosen_move = rng.choice(moves, p=weights)

    new_rep.committed[chosen_fen] = chosen_move
    node = new_rep.graph["nodes"][chosen_fen]
    child_fen = node["children"][chosen_move]["child_fen"]
    if child_fen not in new_rep.reached:
        new_rep.reached.add(child_fen)
    _expand_to_closure(new_rep, [child_fen])

    if len(new_rep.committed) > BUDGET:
        raise MutationFailed("Budget exceeded after extend")
    return new_rep


def mutate_prune(rep: Repertoire, rng) -> Repertoire:
    """
    Pick a random committed node other than the root committed node.
    Remove its committed move and all descendants.  The position becomes a leaf.
    Raises MutationFailed if pruning would violate closure at the parent opponent node.
    """
    # Find pruneable nodes: committed nodes whose removal keeps closure valid
    # For White: root committed node is STARTING_FEN; we should not prune it if it's the
    # only committed node.  For safety, exclude the starting position for white.
    candidates = [f for f in rep.committed if f != STARTING_FEN]
    if not candidates:
        raise MutationFailed("No pruneable committed nodes")

    new_rep = rep.copy()
    chosen_fen = rng.choice(candidates)

    # Check: is this node required by closure at its parent opponent node?
    # i.e., is there an opponent-turn parent in reached that has a freq >= threshold
    # move leading to chosen_fen, where that opponent move must be covered?
    # If so, removing chosen_fen from reached would violate closure.
    # Actually: pruning makes chosen_fen a leaf (stays in reached), so closure
    # at opponent parents is preserved.  What we must check is whether any opponent
    # node BELOW chosen_fen that we'll remove is needed for closure from above.
    # The spec says: "Raises MutationFailed if pruning would violate closure at
    # the parent opponent node."  Interpretation: if chosen_fen itself is required
    # by an opponent-turn parent (i.e., it would be removed), that's a problem.
    # But we keep chosen_fen in reached; only its descendants are removed.
    # So this check is: would any descendant whose removal causes a closure
    # violation at some opponent node that remains in reached?
    #
    # Simplified safe check: only fail if chosen_fen has no children or if
    # the position has no committed move (already a leaf). Since we only picked
    # from committed, it always has a committed move. Proceed with removal.

    new_rep.committed.pop(chosen_fen)
    _remove_subtree(new_rep, chosen_fen)

    if not validate(new_rep):
        raise MutationFailed("Prune violated closure")

    return new_rep


def mutate_opening_replacement(rep: Repertoire, rng) -> Repertoire:
    """
    Pick a random committed node at ply_depth <= 2.  Replace its committed move
    and rebuild everything below from scratch via construct_random.
    """
    shallow = [f for f in rep.committed
               if rep.graph["nodes"][f]["ply_depth"] <= 2]
    if not shallow:
        raise MutationFailed("No shallow committed nodes")

    new_rep = rep.copy()
    chosen_fen = rng.choice(shallow)

    # Remove committed move and all descendants below chosen_fen
    new_rep.committed.pop(chosen_fen)
    _remove_subtree(new_rep, chosen_fen)

    # Rebuild below chosen_fen via construct_random
    # We build a temporary repertoire rooted at chosen_fen and merge it in
    remaining_budget = BUDGET - len(new_rep.committed)
    if remaining_budget <= 0:
        raise MutationFailed("No budget remaining after removal")

    # Pick a new move via weighted random choice
    freq = aggregate_move_freq(new_rep.graph, chosen_fen)
    if not freq:
        raise MutationFailed("No moves at chosen opening position")

    moves = list(freq.keys())
    weights = [freq[m] for m in moves]
    total = sum(weights)
    weights = [w / total for w in weights]
    chosen_move = rng.choice(moves, p=weights)

    new_rep.committed[chosen_fen] = chosen_move
    node = new_rep.graph["nodes"][chosen_fen]
    child_fen = node["children"][chosen_move]["child_fen"]
    if child_fen not in new_rep.reached:
        new_rep.reached.add(child_fen)
    prev = set(new_rep.reached)
    _expand_to_closure(new_rep, [child_fen])

    # Fill remaining budget with greedy construction below
    queue: deque[str] = deque()
    for fen in new_rep.reached - prev | {child_fen}:
        if _our_turn(new_rep, fen) and fen not in new_rep.committed:
            queue.append(fen)
    while queue and len(new_rep.committed) < BUDGET:
        fen = queue.popleft()
        if fen not in new_rep.reached or fen in new_rep.committed:
            continue
        f2 = aggregate_move_freq(new_rep.graph, fen)
        if not f2:
            continue
        moves2 = list(f2.keys())
        w2 = [f2[m] for m in moves2]
        t2 = sum(w2)
        w2 = [x / t2 for x in w2]
        mv = rng.choice(moves2, p=w2)
        new_rep.committed[fen] = mv
        ch = new_rep.graph["nodes"][fen]["children"][mv]["child_fen"]
        if ch not in new_rep.reached:
            new_rep.reached.add(ch)
        prev2 = set(new_rep.reached)
        _expand_to_closure(new_rep, [ch])
        for nf in new_rep.reached - prev2:
            if _our_turn(new_rep, nf) and nf not in new_rep.committed:
                queue.append(nf)

    if len(new_rep.committed) > BUDGET:
        raise MutationFailed("Budget exceeded after opening replacement")
    return new_rep


def crossover(rep_a: Repertoire, rep_b: Repertoire, rng) -> Repertoire:
    """
    Find the set of positions committed in both parents with the same committed
    move.  If non-empty, pick one at random.  Keep rep_a's structure from root
    to (and including) that position, and rep_b's subtree from that position
    onward.  If the result exceeds budget, raise MutationFailed.
    If no shared committed moves exist, return a clone of rep_a.
    """
    shared = {f for f in rep_a.committed
              if f in rep_b.committed and rep_a.committed[f] == rep_b.committed[f]}
    if not shared:
        return rep_a.copy()

    pivot = rng.choice(list(shared))

    # Build new repertoire: rep_a up to pivot, rep_b from pivot onward
    new_rep = rep_a.copy()

    # Remove everything in new_rep that is only reachable through the pivot's subtree
    new_rep.committed.pop(pivot, None)
    _remove_subtree(new_rep, pivot)

    # Now graft rep_b's subtree below pivot into new_rep
    # Collect all positions in rep_b's subtree (descendants of pivot in rep_b)
    b_descendants = _collect_descendants(rep_b, pivot)
    b_descendants.add(pivot)
    for fen in b_descendants:
        new_rep.reached.add(fen)
        if fen in rep_b.committed:
            new_rep.committed[fen] = rep_b.committed[fen]

    if len(new_rep.committed) > BUDGET:
        raise MutationFailed("Budget exceeded after crossover")

    return new_rep


def mutate_candidate(candidate: Candidate, rng) -> Candidate:
    """
    Randomly choose white or black (50/50).  Randomly choose one of the four
    mutation operators (uniform).  Apply it.  If it raises MutationFailed,
    retry with a different operator choice, up to 5 attempts.  If all 5 fail,
    return an unchanged clone.
    """
    operators = [mutate_move_swap, mutate_extend, mutate_prune, mutate_opening_replacement]
    new_cand = candidate.copy()
    new_cand.fitness = None
    new_cand.band_scores_cache = None

    for _ in range(5):
        color = rng.choice(["white", "black"])
        op = rng.choice(operators)
        rep = new_cand.white if color == "white" else new_cand.black
        try:
            new_rep = op(rep, rng)
            if color == "white":
                new_cand.white = new_rep
            else:
                new_cand.black = new_rep
            return new_cand
        except MutationFailed:
            continue

    # All attempts failed — return unchanged clone
    clone = candidate.copy()
    clone.fitness = None
    clone.band_scores_cache = None
    return clone


def crossover_candidates(cand_a: Candidate, cand_b: Candidate, rng) -> Candidate:
    """
    Crossover white parts independently from black parts.
    Return a new Candidate with the resulting pair.
    """
    try:
        new_white = crossover(cand_a.white, cand_b.white, rng)
    except MutationFailed:
        new_white = cand_a.white.copy()

    try:
        new_black = crossover(cand_a.black, cand_b.black, rng)
    except MutationFailed:
        new_black = cand_a.black.copy()

    return Candidate(white=new_white, black=new_black, fitness=None, band_scores_cache=None)
