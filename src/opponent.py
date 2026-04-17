import numpy as np
from typing import Optional


# Cap on how many position-specific perturbations a single opponent may carry.
MAX_PERTURBATIONS = 30

# Strength of a fresh perturbation when added by mutation.
DEFAULT_PERTURBATION_STRENGTH = 0.7


class Opponent:
    """Adversarial opponent.

    Two-part representation:
      * `mixture` — convex combination over the 3 rating bands (existing behavior).
      * `perturbations` — sparse dict mapping FEN -> (move_uci, strength in (0,1]).
        At a perturbed FEN where it is an opponent turn, the base policy is
        blended toward a delta on `move_uci`:
            effective[m] = (1-s)*base[m] + s * 1[m == move_uci]
        Empty by default; populated by mutate() when a graph is supplied.
    """

    def __init__(
        self,
        mixture: np.ndarray,
        perturbations: Optional[dict] = None,
    ):
        assert mixture.shape == (3,)
        assert abs(mixture.sum() - 1.0) < 1e-9
        assert (mixture >= 0).all()
        self.mixture = mixture
        self.perturbations: dict = dict(perturbations) if perturbations else {}
        self.fitness: Optional[float] = None

    @classmethod
    def uniform(cls) -> 'Opponent':
        return cls(np.ones(3) / 3.0)

    @classmethod
    def random(cls, rng) -> 'Opponent':
        return cls(rng.dirichlet(np.ones(3)))

    def mutate(
        self,
        rng,
        strength: float = 0.3,
        graph: Optional[dict] = None,
        max_perturbations: int = MAX_PERTURBATIONS,
    ) -> 'Opponent':
        """Mutate mixture and (if graph supplied) perturbations.

        With per-position perturbations enabled, one of four edits is chosen:
          * mixture jitter (existing behavior)
          * add a new perturbation at a random opponent-turn FEN
          * remove a random perturbation
          * change strength of a random perturbation
        Without a graph, only mixture jitter happens (backward-compatible).
        """
        new_perts = dict(self.perturbations)

        if graph is None or max_perturbations <= 0:
            kind = "mixture"
        else:
            r = rng.random()
            if r < 0.4:
                kind = "mixture"
            elif r < 0.7:
                kind = "add" if len(new_perts) < max_perturbations else "modify"
            elif r < 0.9:
                kind = "remove" if new_perts else "add"
            else:
                kind = "modify" if new_perts else "add"

        if kind == "mixture":
            noise = rng.dirichlet(np.ones(3))
            new_mix = (1.0 - strength) * self.mixture + strength * noise
            new_mix = new_mix / new_mix.sum()
            return Opponent(new_mix, new_perts)

        new_mix = self.mixture.copy()

        if kind == "add":
            fen, move = _sample_opponent_position_move(graph, rng, exclude=set(new_perts))
            if fen is not None:
                new_perts[fen] = (move, DEFAULT_PERTURBATION_STRENGTH)
        elif kind == "remove":
            fens = list(new_perts.keys())
            if fens:
                victim = fens[rng.integers(len(fens))]
                new_perts.pop(victim)
        elif kind == "modify":
            fens = list(new_perts.keys())
            if fens:
                target = fens[rng.integers(len(fens))]
                move, s = new_perts[target]
                s_new = float(np.clip(s + rng.uniform(-0.3, 0.3), 0.05, 1.0))
                new_perts[target] = (move, s_new)

        return Opponent(new_mix, new_perts)

    def crossover(
        self,
        other: 'Opponent',
        rng,
        max_perturbations: int = MAX_PERTURBATIONS,
    ) -> 'Opponent':
        """Convex blend of mixtures + uniform-random subset union of perturbations."""
        w = rng.uniform(0.0, 1.0)
        new_mix = w * self.mixture + (1.0 - w) * other.mixture
        new_mix = new_mix / new_mix.sum()

        # Union perturbation FENs; for shared FENs pick from one parent at random.
        # Then keep at most max_perturbations entries.
        all_fens = set(self.perturbations) | set(other.perturbations)
        new_perts: dict = {}
        for fen in all_fens:
            in_a = fen in self.perturbations
            in_b = fen in other.perturbations
            if in_a and in_b:
                src = self.perturbations if rng.random() < 0.5 else other.perturbations
            elif in_a:
                # 50% chance of carrying through a single-parent perturbation
                if rng.random() < 0.5:
                    src = self.perturbations
                else:
                    continue
            else:
                if rng.random() < 0.5:
                    src = other.perturbations
                else:
                    continue
            new_perts[fen] = src[fen]

        if len(new_perts) > max_perturbations:
            keep_keys = list(new_perts.keys())
            chosen = rng.choice(len(keep_keys), size=max_perturbations, replace=False)
            new_perts = {keep_keys[i]: new_perts[keep_keys[i]] for i in chosen}

        return Opponent(new_mix, new_perts)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _sample_opponent_position_move(graph: dict, rng, exclude: set):
    """Pick a random FEN with at least 2 children and one of its children.

    The 'opponent-turn' notion is symmetric in this codebase (one shared graph
    for both colors), so we just sample any branching position.  Returns
    (fen, move_uci) or (None, None) on failure.
    """
    nodes = graph.get("nodes", {})
    if not nodes:
        return None, None
    # Quick rejection sampling: cap attempts to keep this O(1)-ish.
    keys = list(nodes.keys())
    n_keys = len(keys)
    for _ in range(20):
        idx = rng.integers(n_keys)
        fen = keys[idx]
        if fen in exclude:
            continue
        children = nodes[fen].get("children", {})
        if len(children) < 2:
            continue
        moves = list(children.keys())
        move = moves[rng.integers(len(moves))]
        return fen, move
    return None, None
