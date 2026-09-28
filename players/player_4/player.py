import math
from bisect import bisect_left, bisect_right
from collections import deque
from functools import lru_cache
from itertools import combinations
from math import ceil, isinf, sqrt

from core.engine import EMBARRASSMENT_THRESHOLD, PACK_COST, PACK_SIZE
from models.player import GameContext, PlayerSnapshot, Selection, TurnContext
from models.player import Player as BasePlayer

WHITE_CUTOFF = 64
WHITE_START = 255
WHITE_FADE = 2


def is_white(shade: int) -> bool:
	return shade > WHITE_CUTOFF


def wears(shade: int) -> float:
	"""Return wear age on a shared 0..64 scale for both colours."""
	if is_white(shade):
		return (WHITE_START - shade) / WHITE_FADE
	return float(shade)


def _embarrassment(a: int, b: int) -> int:
	difference = abs(a - b)
	return difference if difference > 6 else 0


def _age_shade(shade: int, steps: int) -> int:
	return max(127, shade - 2 * steps) if shade > 64 else min(64, shade + steps)


def _gap_geometry(shades: tuple[int, ...], gap: int) -> tuple[list[int], list[int]]:
	return (
		[bisect_right(shades, shade - gap) for shade in shades],
		[bisect_left(shades, shade + gap) for shade in shades],
	)


class Player4(BasePlayer):
	"""Match safely and spend a paced discard allowance on future hand value.

	White and black shades move at different rates, so observations are first
	converted to wear age. Recent samples maintain a separate mean and standard
	deviation for each colour. That makes a widely spread colour more urgent
	without treating its raw shade range as evidence by itself.

	The household bill lags behind discards until a same-colour group of six
	forms. Each instance therefore counts its own requested discards immediately
	and reserves the roommates' projected spending before claiming unused budget.
	Up to two leftovers are discarded per turn when the saved allowance supports
	it, pristine socks are always kept, and the last few days stop recycling.

	WHITE_MIN and BLACK_MAX remain class attributes so the existing sweep tools
	can tune the minimum wear age. The active cutoff becomes more conservative
	when the configured budget supports fewer discards per roommate per day.
	"""

	WHITE_MIN = 240
	BLACK_MAX = 7
	RESERVE = 30.0
	RESERVE_FRACTION = 0.10
	SAMPLE_WINDOW_MULTIPLIER = 2
	MIN_STAT_SAMPLES = 10
	MAX_DISCARDS = 2
	WIDE_Z = 2.5
	NARROW_Z = 0.25
	DISPERSION_SHIFT = 0.75
	WARMUP_DAYS = 20
	HORIZON_DAYS = 20
	PRESERVE_BASELINE_AGE = 3.0

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)
		window = max(24, self.capacity * self.SAMPLE_WINDOW_MULTIPLIER)
		self._samples = {
			False: deque(maxlen=window),
			True: deque(maxlen=window),
		}
		self._budget: float | None = None
		self._requested_discards = 0
		self._observations = deque(maxlen=max(40, self.capacity))
		self._gap_geometry = lru_cache(maxsize=512)(_gap_geometry)

	def select_socks(self, offered: tuple[int, ...], turn: TurnContext) -> Selection:
		"""Use the original selector when the budget permits frequent retirement."""
		initial_budget = turn.total_spent + turn.budget_remaining
		retirement_age = (
			10 * self.roommates * self.days / (3 * initial_budget)
			if initial_budget > 0
			else float('inf')
		)
		if retirement_age <= self.PRESERVE_BASELINE_AGE:
			return self._select_socks_baseline(offered, turn)

		self._observe(offered)
		self._observations.extend(offered)
		allowance, _ = self._discard_policy(turn)
		pairs = list(combinations(range(len(offered)), 2))
		baseline_pair = min(pairs, key=lambda pair: self._pair_key(offered, pair))
		if allowance == 0:
			return Selection(wear=baseline_pair)

		immediate_minimum = _embarrassment(*(offered[i] for i in baseline_pair))
		values = self._future_values(offered, turn)
		improvements = {
			i: values[255 if shade > 64 else 0] - values[shade]
			for i, shade in enumerate(offered)
			if wears(shade) > 1
		}
		best = None
		for pair in pairs:
			a, b = (offered[i] for i in pair)
			if _embarrassment(a, b) != immediate_minimum:
				continue
			gains = sorted(
				(change, i)
				for i, change in improvements.items()
				if i not in pair and change < -1e-12
			)[:allowance]
			discard = tuple(i for change, i in gains)
			key = (
				sum(change for change, i in gains),
				wears(a) + wears(b),
				abs(a - b),
				len(discard),
				pair,
			)
			if best is None or key < best[0]:
				best = (key, pair, discard)
		_, pair, discard = best
		self._requested_discards += len(discard)
		return Selection(wear=pair, discard=discard)

	def _select_socks_baseline(self, offered: tuple[int, ...], turn: TurnContext) -> Selection:
		self._observe(offered)
		discard_allowance, aggressiveness = self._discard_policy(turn)

		first, second = min(
			combinations(range(len(offered)), 2),
			key=lambda pair: self._pair_key(offered, pair),
		)
		wear = (first, second)

		if not discard_allowance:
			return Selection(wear=wear)

		candidates = [
			index
			for index in range(len(offered))
			if index not in wear
			and wears(offered[index]) > self._minimum_age(offered[index], aggressiveness)
		]
		if not candidates:
			return Selection(wear=wear)

		discard = tuple(
			sorted(
				candidates,
				key=lambda index: self._outlier_key(offered[index]),
				reverse=True,
			)[:discard_allowance]
		)
		self._requested_discards += len(discard)
		return Selection(wear=wear, discard=discard)

	def _conditional_values(
		self, masses: dict[int, float], targets: set[int], companions: int
	) -> dict[int, float]:
		"""Evaluate best-pair cost for iid companions drawn from local shade masses."""
		items = sorted((shade, mass) for shade, mass in masses.items() if mass > 0)
		shades = tuple(shade for shade, mass in items)
		total = sum(mass for shade, mass in items)
		weights = [mass / total for shade, mass in items]
		n = len(shades)
		values = {shade: 0.0 for shade in targets}
		permutations = math.factorial(companions)
		for gap in range(7, 65):
			left, right = self._gap_geometry(shades, gap)
			forward = [[1.0] * (n + 1)]
			backward = [[1.0] * (n + 1)]
			for _count in range(1, companions + 1):
				previous = forward[-1]
				current = [0.0] * (n + 1)
				for i in range(n):
					current[i + 1] = current[i] + weights[i] * previous[left[i]]
				forward.append(current)
				previous = backward[-1]
				current = [0.0] * (n + 1)
				for i in range(n - 1, -1, -1):
					current[i] = current[i + 1] + weights[i] * previous[right[i]]
				backward.append(current)
			greatest_probability = 0.0
			for shade in targets:
				li = bisect_right(shades, shade - gap)
				ri = bisect_left(shades, shade + gap)
				probability = permutations * sum(
					forward[k][li] * backward[companions - k][ri] for k in range(companions + 1)
				)
				greatest_probability = max(greatest_probability, probability)
				values[shade] += (7 if gap == 7 else 1) * probability
			if greatest_probability < 1e-14:
				break
		return values

	def _future_values(self, offered: tuple[int, ...], turn: TurnContext) -> dict[int, float]:
		"""Infer a future handful from this player's offers and public game fields."""
		days_left = max(1, self.days - turn.day)
		budget = turn.total_spent + turn.budget_remaining
		reserve = max(self.RESERVE, budget * self.RESERVE_FRACTION)
		replacements_per_day = max(0, turn.budget_remaining - reserve) / days_left * 0.6
		mix = min(1.0, replacements_per_day * self.HORIZON_DAYS / self.capacity)
		wear_rate = 2 * self.roommates
		survival = wear_rate / (wear_rate + replacements_per_day + 0.03 * self.roommates)
		histories = {False: {}, True: {}}
		for shade in self._observations:
			histogram = histories[shade > 64]
			histogram[shade] = histogram.get(shade, 0) + 1
		targets = set(offered) | {0, 255}
		values = {shade: 0.0 for shade in targets}
		shifts = (0, min(4, int(days_left * wear_rate / self.capacity)))
		for step in shifts:
			masses = {}
			for white, histogram in histories.items():
				if not histogram:
					histogram = {255 if white else 0: 1}
				total = sum(histogram.values())
				for shade, mass in histogram.items():
					future = _age_shade(shade, step)
					masses[future] = masses.get(future, 0) + 0.5 * (1 - mix) * mass / total
				for age in range(65):
					shade = 255 - 2 * age if white else age
					mass = survival**age * (1 - survival if age < 64 else 1)
					masses[shade] = masses.get(shade, 0) + 0.5 * mix * mass
			shifted_targets = {_age_shade(shade, step) for shade in targets}
			projected = self._conditional_values(masses, shifted_targets, self.selection_unit - 1)
			for shade in targets:
				values[shade] += projected[_age_shade(shade, step)] / len(shifts)
		return values

	def _observe(self, offered: tuple[int, ...]) -> None:
		for shade in offered:
			self._samples[is_white(shade)].append(wears(shade))

	def _stats(self, white: bool) -> tuple[float, float]:
		values = self._samples[white]
		if not values:
			return 0.0, 0.0
		mean = sum(values) / len(values)
		if len(values) < 2:
			return mean, 0.0
		variance = sum((value - mean) ** 2 for value in values) / len(values)
		return mean, sqrt(variance)

	def _outlier_key(self, shade: int) -> tuple[float, float]:
		age = wears(shade)
		mean, stddev = self._stats(is_white(shade))
		z_score = (age - mean) / max(1.0, stddev)
		return z_score + stddev / 8.0, age

	@staticmethod
	def _pair_key(offered: tuple[int, ...], pair: tuple[int, int]) -> tuple[float, float, int]:
		a, b = pair
		difference = abs(offered[a] - offered[b])
		embarrassment = difference if difference > EMBARRASSMENT_THRESHOLD else 0
		combined_age = wears(offered[a]) + wears(offered[b])

		return (
			float(embarrassment),
			combined_age if embarrassment == 0 else difference,
			difference,
		)

	def _base_minimum_age(self, shade: int) -> float:
		age = wears(self.WHITE_MIN) if is_white(shade) else float(self.BLACK_MAX)
		return min(64.0, max(1.0, age))

	def _minimum_age(self, shade: int, aggressiveness: float) -> float:
		"""Old edge of this colour's dynamically sized acceptance range."""
		base_age = self._base_minimum_age(shade)
		budget_floor = 1.0 + (base_age - 1.0) * (1.0 - aggressiveness)

		white = is_white(shade)
		if len(self._samples[white]) < self.MIN_STAT_SAMPLES:
			return budget_floor

		mean, stddev = self._stats(white)
		_, other_stddev = self._stats(not white)
		if stddev < 1.0:
			return max(budget_floor, mean + 1.0)

		z_width = self.WIDE_Z - (self.WIDE_Z - self.NARROW_Z) * aggressiveness
		dispersion_advantage = max(0.0, stddev - other_stddev) / max(1.0, stddev + other_stddev)
		z_width = max(0.0, z_width - self.DISPERSION_SHIFT * dispersion_advantage)
		cluster_edge = mean + z_width * stddev
		statistical_edge = min(64.0, max(budget_floor, cluster_edge))
		return aggressiveness * budget_floor + (1.0 - aggressiveness) * statistical_edge

	def _discard_policy(self, turn: TurnContext) -> tuple[int, float]:
		"""Return today's discard allowance and range aggressiveness."""
		days_left = self.days - turn.day + 1
		reuse_horizon = ceil(self.capacity / max(1, self.selection_unit * self.roommates))
		if days_left <= reuse_horizon:
			return 0, 0.0

		if isinf(turn.budget_remaining):
			return self.MAX_DISCARDS, 1.0
		if turn.budget_remaining < PACK_COST:
			return 0, 0.0

		if self._budget is None:
			self._budget = turn.total_spent + turn.budget_remaining

		reserve = max(self.RESERVE, self._budget * self.RESERVE_FRACTION)
		usable_budget = max(0.0, self._budget - reserve)
		if turn.day <= self.WARMUP_DAYS:
			local_budget = usable_budget / max(1, self.roommates)
		else:
			own_imputed_spend = self._requested_discards * PACK_COST / PACK_SIZE
			others_spend = max(0.0, turn.total_spent - own_imputed_spend)
			projected_others_spend = others_spend / turn.day * self.days
			local_budget = max(0.0, usable_budget - projected_others_spend)

		total_quota = local_budget / PACK_COST * PACK_SIZE
		discard_rate = total_quota / max(1, self.days)
		allowed_by_today = total_quota * turn.day / max(1, self.days)
		available_credit = int(allowed_by_today - self._requested_discards)

		if available_credit < 1:
			return 0, 0.0
		spend_target = usable_budget * turn.day / max(1, self.days)
		if turn.total_spent > spend_target + PACK_COST:
			return 0, 0.0

		aggressiveness = min(1.0, discard_rate / self.MAX_DISCARDS)
		allowance = min(self.MAX_DISCARDS, available_credit)
		return allowance, aggressiveness
