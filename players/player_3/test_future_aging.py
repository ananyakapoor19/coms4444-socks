"""Regression checks for the joint-aging model and preserved spending rules."""

from math import ceil
from uuid import UUID

import pytest

from models.player import GameContext, PlayerSnapshot, TurnContext
from models.sock import Color
from players.player_3.benchmarks.original import Player3 as Original3
from players.player_3.benchmarks.terminal_only import TerminalOnly3
from players.player_3.future_aging import mismatch_values, project_life_values
from players.player_3.player import Player3


def player(cls=Player3):
	return cls(PlayerSnapshot(UUID(int=0), 0), GameContext(40, 4, 4, 720))


def turn(day=1, spent=0.0, remaining=120.0):
	return TurnContext(
		day=day, total_spent=spent, total_embarrassment=0.0, budget_remaining=remaining
	)


@pytest.mark.parametrize('fade', [1, 2])
@pytest.mark.parametrize('horizon', [0.0, 0.2, 1.0, 7.3, 63.9, 64.0, 144.0])
@pytest.mark.parametrize('alpha,q', [(0.0, 0.999), (0.3, 0.985), (1.0, 0.8)])
def test_projection_against_direct_pair_enumeration(fade, horizon, alpha, q):
	"""Independent scalar oracle, including caps and a stationary target."""
	observed = [0.0] * 65
	for age, mass in [(0, 1.0), (5, 2.0), (32, 0.5), (63, 3.0), (64, 1.0)]:
		observed[age] = mass
	actual = project_life_values(observed, fade, alpha, q, 0.98, horizon)
	target = [(1.0 - q) * q**age for age in range(65)]
	target[64] = q**64
	discount = min(q, 0.98)

	def cost(a, b):
		gap = fade * abs(a - b)
		return gap if gap > 6 else 0.0

	for candidate in [0, 3, 7, 31, 63, 64]:
		expected = 0.0
		for step in range(ceil(horizon)):
			age = min(64, candidate + step)
			obs = sum(
				mass * cost(age, min(64, partner + step)) for partner, mass in enumerate(observed)
			) / sum(observed)
			geo = sum(mass * cost(age, partner) for partner, mass in enumerate(target))
			weight = discount**step - discount ** min(step + 1, horizon)
			expected += weight * ((1 - alpha) * obs + alpha * geo)
		assert actual[candidate] == pytest.approx(expected, abs=1e-10)
	assert all(value >= 0 for value in actual)


@pytest.mark.parametrize('fade,free_ages', [(1, 6), (2, 3)])
def test_free_gap_and_synchronized_aging(fade, free_ages):
	masses = [1.0] + [0.0] * 64
	costs = mismatch_values(masses, fade)
	assert costs[free_ages] == 0
	assert costs[free_ages + 1] == fade * (free_ages + 1)
	values = project_life_values(masses, fade, 0.0, 0.99, 0.98, 144.0)
	assert values[0] == 0  # A fresh candidate stays matched to a fresh cohort.
	assert values[free_ages] == 0
	assert values[free_ages + 1] > 0


def test_empty_histogram_and_capped_observed_pair():
	assert project_life_values([0.0] * 65, 2, 0, 0.99, 0.98, 144) == [0.0] * 65
	capped = [0.0] * 64 + [1.0]
	assert project_life_values(capped, 2, 0, 0.99, 0.98, 144)[64] == 0
	assert project_life_values(capped, 2, 1, 0.99, 0.98, 144)[64] > 0


def test_controller_and_histogram_match_original_through_budget_exhaustion():
	new, old = player(), player(Original3)
	for context in [
		turn(1),
		turn(19),
		turn(100, 10, 110),
		turn(300, 40, 80),
		turn(500, 120, 0),
		turn(719, 120, 0),
	]:
		for p in (new, old):
			p._observe((0, 40, 127, 239), context.day)
		assert new.histogram == old.histogram
		assert new._controller(context) == old._controller(context)
		for field in ('credit', 'my_spend', 'household_rate', 'alpha', 'geometric_survival'):
			assert getattr(new, field) == getattr(old, field)


def test_cache_refresh_and_zero_horizon_on_last_day():
	p = player()
	p.select_socks((0, 0, 255, 255), turn(1))
	first = p._future_life_cache[Color.WHITE]
	p.select_socks((20, 40, 127, 175), turn(200))
	assert p._future_life_cache[Color.WHITE] is not first
	assert p._future_life_cache[Color.WHITE] != first
	p.select_socks((20, 40, 127, 175), turn(720, 120, 0))
	assert p._future_wear_horizon == 0
	assert p._future_life_cache[Color.WHITE] == [0.0] * 65


@pytest.mark.parametrize('remaining,expected', [(120, (2, 3)), (9, ())])
def test_terminal_only_is_separate_and_explicitly_bypasses_credit(remaining, expected):
	offered = (0, 0, 64, 127)
	p = player(TerminalOnly3)
	selection = p.select_socks(offered, turn(1, 0, remaining))
	assert selection.wear == (0, 1)
	assert selection.discard == expected
	assert p.credit == p.my_spend == 0
	assert player().select_socks(offered, turn(1, 0, remaining)).discard == ()
