import numpy as np
from typing import Optional


class Opponent:
    def __init__(self, mixture: np.ndarray):
        assert mixture.shape == (3,)
        assert abs(mixture.sum() - 1.0) < 1e-9
        assert (mixture >= 0).all()
        self.mixture = mixture
        self.fitness: Optional[float] = None

    @classmethod
    def uniform(cls) -> 'Opponent':
        return cls(np.ones(3) / 3.0)

    @classmethod
    def random(cls, rng) -> 'Opponent':
        return cls(rng.dirichlet(np.ones(3)))

    def mutate(self, rng, strength: float = 0.3) -> 'Opponent':
        noise = rng.dirichlet(np.ones(3))
        new_mix = (1.0 - strength) * self.mixture + strength * noise
        new_mix = new_mix / new_mix.sum()
        return Opponent(new_mix)

    def crossover(self, other: 'Opponent', rng) -> 'Opponent':
        w = rng.uniform(0.0, 1.0)
        new_mix = w * self.mixture + (1.0 - w) * other.mixture
        new_mix = new_mix / new_mix.sum()
        return Opponent(new_mix)
