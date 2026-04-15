import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from opponent import Opponent

TOLERANCE = 1e-9


def _valid(opp: Opponent):
    """Helper: check mixture non-negative and sums to 1 within tolerance."""
    assert (opp.mixture >= 0).all(), "mixture has negative component"
    assert abs(opp.mixture.sum() - 1.0) < TOLERANCE, f"mixture sums to {opp.mixture.sum()}, not 1"


# --- AC: __init__ direct construction ---

def test_init_valid_mixture():
    """__init__ with a valid simplex point produces valid Opponent."""
    mix = np.array([0.2, 0.5, 0.3])
    opp = Opponent(mix)
    _valid(opp)


def test_init_mixture_non_negative_and_sums_to_one():
    """Acceptance criterion: mixture is non-negative and sums to 1."""
    mix = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
    opp = Opponent(mix)
    _valid(opp)


def test_init_rejects_wrong_shape():
    """__init__ asserts mixture.shape == (3,)."""
    with pytest.raises(AssertionError):
        Opponent(np.array([0.5, 0.5]))


def test_init_rejects_negative():
    """__init__ asserts mixture >= 0."""
    with pytest.raises(AssertionError):
        Opponent(np.array([-0.1, 0.6, 0.5]))


def test_init_rejects_bad_sum():
    """__init__ asserts mixture sums to 1 within 1e-9."""
    with pytest.raises(AssertionError):
        Opponent(np.array([0.4, 0.4, 0.4]))


def test_init_fitness_is_none():
    """fitness attribute starts as None."""
    opp = Opponent(np.array([0.2, 0.5, 0.3]))
    assert opp.fitness is None


def test_fitness_settable():
    """fitness can be set to a float."""
    opp = Opponent(np.array([0.2, 0.5, 0.3]))
    opp.fitness = 3.14
    assert opp.fitness == pytest.approx(3.14)


# --- AC: uniform() ---

def test_uniform_valid():
    """Acceptance criterion: uniform() produces valid Opponent."""
    opp = Opponent.uniform()
    _valid(opp)


def test_uniform_values():
    """uniform() gives equal weights 1/3 each."""
    opp = Opponent.uniform()
    expected = np.ones(3) / 3.0
    np.testing.assert_allclose(opp.mixture, expected, atol=TOLERANCE)


# --- AC: random() ---

def test_random_valid_single():
    """Acceptance criterion: random() produces valid Opponent."""
    rng = np.random.default_rng(42)
    opp = Opponent.random(rng)
    _valid(opp)


def test_random_valid_many():
    """Acceptance criterion: 1000 random() calls all produce valid Opponents."""
    rng = np.random.default_rng(0)
    for _ in range(1000):
        opp = Opponent.random(rng)
        _valid(opp)


def test_random_different_each_call():
    """Two consecutive random() calls with advancing RNG produce different mixtures."""
    rng = np.random.default_rng(7)
    a = Opponent.random(rng)
    b = Opponent.random(rng)
    assert not np.allclose(a.mixture, b.mixture), "Two random opponents should differ"


def test_random_deterministic():
    """Same RNG seed produces same mixture (determinism)."""
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    a = Opponent.random(rng1)
    b = Opponent.random(rng2)
    np.testing.assert_array_equal(a.mixture, b.mixture)


# --- AC: mutate() ---

def test_mutate_valid():
    """Acceptance criterion: mutate() produces valid Opponent."""
    rng = np.random.default_rng(1)
    opp = Opponent.uniform()
    mutated = opp.mutate(rng)
    _valid(mutated)


def test_mutate_valid_many():
    """Acceptance criterion: 1000 mutate() calls all produce valid Opponents."""
    rng = np.random.default_rng(2)
    opp = Opponent.uniform()
    for _ in range(1000):
        mutated = opp.mutate(rng)
        _valid(mutated)


def test_mutate_returns_new_object():
    """mutate() returns a new Opponent, leaving original unchanged."""
    rng = np.random.default_rng(3)
    opp = Opponent.uniform()
    original_mix = opp.mixture.copy()
    mutated = opp.mutate(rng)
    assert mutated is not opp
    np.testing.assert_array_equal(opp.mixture, original_mix)


def test_mutate_strength_zero_unchanged():
    """strength=0.0 leaves mixture unchanged (within floating point)."""
    rng = np.random.default_rng(4)
    opp = Opponent(np.array([0.2, 0.5, 0.3]))
    mutated = opp.mutate(rng, strength=0.0)
    np.testing.assert_allclose(mutated.mixture, opp.mixture, atol=1e-12)


def test_mutate_various_strengths_valid():
    """mutate() with various strength values always produces valid Opponent."""
    rng = np.random.default_rng(5)
    opp = Opponent.uniform()
    for strength in [0.0, 0.1, 0.5, 0.9, 1.0]:
        mutated = opp.mutate(rng, strength=strength)
        _valid(mutated)


# --- AC: crossover() ---

def test_crossover_valid():
    """Acceptance criterion: crossover() produces valid Opponent."""
    rng = np.random.default_rng(10)
    a = Opponent.uniform()
    b = Opponent.random(rng)
    child = a.crossover(b, rng)
    _valid(child)


def test_crossover_valid_many():
    """Acceptance criterion: 1000 crossover() calls all produce valid Opponents."""
    rng = np.random.default_rng(11)
    for _ in range(1000):
        a = Opponent.random(rng)
        b = Opponent.random(rng)
        child = a.crossover(b, rng)
        _valid(child)


def test_crossover_returns_new_object():
    """crossover() returns a new Opponent, leaving parents unchanged."""
    rng = np.random.default_rng(12)
    a = Opponent.uniform()
    b = Opponent.random(rng)
    mix_a = a.mixture.copy()
    mix_b = b.mixture.copy()
    child = a.crossover(b, rng)
    assert child is not a
    assert child is not b
    np.testing.assert_array_equal(a.mixture, mix_a)
    np.testing.assert_array_equal(b.mixture, mix_b)


def test_crossover_with_self_valid():
    """crossover(self, self) always produces valid Opponent equal to self."""
    rng = np.random.default_rng(13)
    a = Opponent.uniform()
    child = a.crossover(a, rng)
    _valid(child)
    np.testing.assert_allclose(child.mixture, a.mixture, atol=TOLERANCE)


def test_crossover_child_is_convex_combination():
    """Child mixture is a convex combination of parent mixtures."""
    rng = np.random.default_rng(14)
    a = Opponent(np.array([0.6, 0.3, 0.1]))
    b = Opponent(np.array([0.1, 0.2, 0.7]))
    # Run many crossovers; child must stay in convex hull of a and b
    for _ in range(200):
        child = a.crossover(b, rng)
        _valid(child)
        # Each component of child must be between min and max of parents
        for i in range(3):
            lo = min(a.mixture[i], b.mixture[i])
            hi = max(a.mixture[i], b.mixture[i])
            assert lo - 1e-9 <= child.mixture[i] <= hi + 1e-9, (
                f"component {i}: {child.mixture[i]} not in [{lo}, {hi}]"
            )


# --- AC: comprehensive simplex validity across all construction methods ---

def test_all_methods_produce_valid_opponents():
    """Comprehensive: __init__, uniform, random, mutate, crossover all valid."""
    rng = np.random.default_rng(99)

    # direct
    _valid(Opponent(np.array([0.25, 0.25, 0.5])))

    # uniform
    _valid(Opponent.uniform())

    # random x20
    for _ in range(20):
        _valid(Opponent.random(rng))

    # mutate x20
    base = Opponent.uniform()
    for _ in range(20):
        _valid(base.mutate(rng))

    # crossover x20
    a = Opponent.random(rng)
    b = Opponent.random(rng)
    for _ in range(20):
        _valid(a.crossover(b, rng))
