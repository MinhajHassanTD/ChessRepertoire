"""
Opponent representation: a probability mixture over the three rating bands.

The opponent is a length-3 simplex vector. STATIC pins it to uniform; COEVOLVE
evolves a population via tournament selection, Dirichlet-noise mutation, and
convex-blend crossover.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.config import OPPONENT_MUTATION_STRENGTH, RATING_BANDS

_N_BANDS = len(RATING_BANDS)


class Opponent:
    def __init__(self, mixture: np.ndarray):
        assert mixture.shape == (_N_BANDS,)
        assert abs(mixture.sum() - 1.0) < 1e-9
        assert (mixture >= 0).all()
        self.mixture = mixture
        self.fitness: Optional[float] = None

    @classmethod
    def uniform(cls) -> "Opponent":
        return cls(np.ones(_N_BANDS) / _N_BANDS)

    @classmethod
    def random(cls, rng) -> "Opponent":
        return cls(rng.dirichlet(np.ones(_N_BANDS)))

    def mutate(self, rng, strength: float = OPPONENT_MUTATION_STRENGTH) -> "Opponent":
        noise = rng.dirichlet(np.ones(_N_BANDS))
        new_mix = (1.0 - strength) * self.mixture + strength * noise
        new_mix = new_mix / new_mix.sum()
        return Opponent(new_mix)

    def crossover(self, other: "Opponent", rng) -> "Opponent":
        w = rng.uniform(0.0, 1.0)
        new_mix = w * self.mixture + (1.0 - w) * other.mixture
        new_mix = new_mix / new_mix.sum()
        return Opponent(new_mix)

    def crossover_dirichlet(
        self, other: "Opponent", rng, concentration: float = 3.0
    ) -> "Opponent":
        """Dirichlet-resample crossover.

        Instead of returning the deterministic convex blend (which contracts the
        population toward a centroid), this draws a sample from a Dirichlet
        distribution whose concentration vector is proportional to the midpoint
        of the two parents.  Lower concentration = wider spread around the
        midpoint; higher = tighter (approaches the original convex blend).
        """
        blend = 0.5 * (self.mixture + other.mixture)
        alpha = np.maximum(blend * concentration * _N_BANDS, 0.1)
        new_mix = rng.dirichlet(alpha)
        return Opponent(new_mix)
