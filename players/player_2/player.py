"""Group 2 player: distribution-aware sock selection.

The policy has three parts, described in ``POLICY_CHANGES.md``:

1. Track the shade distribution of the socks we have seen, per colour, over a
   sliding window using Welford's online mean/variance update.
2. Use zero-embarrassment choices to improve the projected distribution.
3. Discard at most one leftover when a pristine replacement has lower
   estimated matching cost, and replacement needs and budget pace allow it.
"""

from collections import deque
from itertools import combinations
from math import ceil, sqrt
from statistics import mean

from core.engine import PACK_COST, PACK_SIZE
from models.player import GameContext, PlayerSnapshot, Selection, TurnContext
from models.player import Player as BasePlayer

EMBARRASSMENT_THRESHOLD = 6

# Shade semantics, mirrored from models/sock.py. A shade above BLACK_CEILING is
# a white sock and a shade at or below it is a black one; the ranges never
# overlap, so colour can be inferred from shade without ambiguity.
WHITE_FLOOR = 127
WHITE_FADE = 2
BLACK_CEILING = 64
BLACK_FADE = 1

WHITE = 'white'
BLACK = 'black'


def colour_of(shade: int) -> str:
	return WHITE if shade > BLACK_CEILING else BLACK


def aged_shade(shade: int) -> int:
	"""Shade a sock will have after being worn once and washed."""
	if colour_of(shade) == WHITE:
		return max(WHITE_FLOOR, shade - WHITE_FADE)
	return min(BLACK_CEILING, shade + BLACK_FADE)


def pair_embarrassment(a: int, b: int) -> float:
	diff = abs(a - b)
	return float(diff) if diff > EMBARRASSMENT_THRESHOLD else 0.0


def expected_remaining_wears(shade: float) -> float:
	"""Expected useful wears remaining for a sock with this observed shade."""
	terminal_wears = 1.0 / 0.25  # terminal socks survive 4 wears in expectation
	if colour_of(int(shade)) == WHITE:
		return (shade - WHITE_FLOOR) / WHITE_FADE + terminal_wears
	return BLACK_CEILING - shade + terminal_wears


EXPECTED_FRESH_WEAR_LIFETIME = 68.0


class WindowedStats:
	"""Mean and population std over the last ``window`` values.

	While fewer than ``window`` values have been seen this is plain Welford:
	each ``add`` folds one value into the running mean and M2 (sum of squared
	deviations). Once the window is full, adding a value first evicts the
	oldest one using the reverse Welford update, so mean/M2 always describe
	exactly the values currently in the deque without ever re-summing them.
	"""

	def __init__(self, window: int) -> None:
		self.window = window
		self.values: deque[float] = deque()
		self.n = 0
		self.mean = 0.0
		self.m2 = 0.0

	def add(self, x: float) -> None:
		if self.n >= self.window:
			self._remove(self.values.popleft())
		self.values.append(x)
		self.n += 1
		delta = x - self.mean
		self.mean += delta / self.n
		self.m2 += delta * (x - self.mean)

	def _remove(self, x: float) -> None:
		if self.n <= 1:
			self.n = 0
			self.mean = 0.0
			self.m2 = 0.0
			return
		new_mean = (self.n * self.mean - x) / (self.n - 1)
		self.m2 -= (x - self.mean) * (x - new_mean)
		# Floating point can leave M2 a hair below zero once the window is
		# nearly uniform; clamp so the std is never NaN.
		if self.m2 < 0.0:
			self.m2 = 0.0
		self.mean = new_mean
		self.n -= 1

	@property
	def variance(self) -> float:
		return self.m2 / self.n if self.n else 0.0

	@property
	def std(self) -> float:
		return sqrt(self.variance)

	def copy(self) -> 'WindowedStats':
		other = WindowedStats(self.window)
		other.values = deque(self.values)
		other.n = self.n
		other.mean = self.mean
		other.m2 = self.m2
		return other


class Player2(BasePlayer):
	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)

		# Projected post-action shade distribution used only for zero-cost pair
		# tie-breaking. This is deliberately separate from raw observations.
		self.running_window_size = 20
		self.stats: dict[str, WindowedStats] = {
			WHITE: WindowedStats(self.running_window_size),
			BLACK: WindowedStats(self.running_window_size),
		}

		# Raw shades actually offered to us. The discard policy uses these
		# observations rather than our own hypothetical post-action state.
		self.raw_window_size = 40
		self.raw_history: dict[str, deque[float]] = {
			WHITE: deque(maxlen=self.raw_window_size),
			BLACK: deque(maxlen=self.raw_window_size),
		}

		# Discard-policy knobs. The reserve calculation is deliberately
		# conservative because we cannot observe the true drawer size.
		self.min_dist_samples = 10
		self.max_discards = 1
		# Require a meaningful improvement in expected thresholded shade gap.
		self.replacement_gain_threshold = float(EMBARRASSMENT_THRESHOLD)
		self.reserve_capacity_buffer = 4
		self.reserve_safety_packs = 2
		self.budget_pace_margin = 0.05

		# Diagnostics, one entry per day: the min and mean embarrassment over
		# all pairs in ``offered``, a proxy for how well-matched the drawer is,
		# and the household budget remaining.
		self.offered_pair_min: list[float] = []
		self.offered_pair_mean: list[float] = []
		self.budget_per_day: list[float] = []
		self.days_seen = 0

	# ------------------------------------------------------------------ policy

	def select_socks(self, offered: tuple[int, ...], turn: TurnContext) -> Selection:
		self.days_seen += 1
		self.budget_per_day.append(turn.budget_remaining)

		# Record every shade we actually observed before making a decision.
		for shade in offered:
			self.raw_history[colour_of(shade)].append(float(shade))

		# 1. Pairwise embarrassment of everything we were handed. This gauges
		#    the drawer's distribution over time and drives the wear choice.
		pairs = pairwise_sock_embarassments(offered)
		scores = [p['embarrassment'] for p in pairs]
		self.offered_pair_min.append(min(scores))
		self.offered_pair_mean.append(sum(scores) / len(scores))

		# 2. If any pair is free (embarrassment 0), use that freedom to
		#    choose the pair that leaves the projected per-colour shade
		#    distributions tightest after the worn socks age. Only when every
		#    pair has positive embarrassment do we take the minimum-cost pair.
		wear = self.choose_pair(offered, pairs)
		leftovers = [i for i in range(len(offered)) if i not in wear]

		# 3. Replace at most one leftover whose pristine replacement is better
		#    matched to observations, subject to the existing reserve and pace.
		discard = self.choose_discards(offered, wear, leftovers, turn)

		# Commit the chosen action to the real distribution: worn socks come
		# back aged, returned socks come back unchanged, discarded socks leave.
		self.apply_action(self.stats, offered, wear, leftovers, discard)

		return Selection(wear=wear, discard=discard)

	def choose_pair(self, offered: tuple[int, ...], pairs: list[dict]) -> tuple[int, int]:
		"""Use zero-cost choices to improve the projected drawer distribution.

		If any pair has zero immediate embarrassment (shade gap <= 6), project
		tomorrow's tracked distribution for each such pair: worn socks return
		aged and all leftovers return unchanged. Pick the free pair with the
		smallest resulting sum of black + white std. If every pair has positive
		embarrassment, fall back to the minimum-embarrassment pair.
		"""
		free_pairs = [p for p in pairs if p['embarrassment'] == 0.0]
		if not free_pairs:
			best = min(
				pairs,
				key=lambda p: (
					p['embarrassment'],
					abs(p['shades'][0] - p['shades'][1]),
					p['pair'],
				),
			)
			return best['pair']

		best_pair: tuple[int, int] | None = None
		best_key: tuple[float, int, tuple[int, int]] | None = None
		for candidate in free_pairs:
			wear = candidate['pair']
			leftovers = [i for i in range(len(offered)) if i not in wear]
			trial = {c: s.copy() for c, s in self.stats.items()}
			self.apply_action(trial, offered, wear, leftovers, ())
			total_std = sum(s.std for s in trial.values())
			# Stable deterministic tie-breaks if the projected spreads match.
			key = (total_std, abs(candidate['shades'][0] - candidate['shades'][1]), wear)
			if best_key is None or key < best_key:
				best_key = key
				best_pair = wear

		assert best_pair is not None
		return best_pair

	def expected_replacement_reserve(self, turn: TurnContext) -> float:
		"""Estimate budget that should be protected for unavoidable future holes.

		We cannot observe the true drawer size, so approximate it from the initial
		capacity minus a small safety buffer. Recent raw shade observations estimate
		how much wear remains in each colour. This is a guardrail, not an exact
		state reconstruction.
		"""
		estimated_per_colour = max(1.0, (self.capacity - self.reserve_capacity_buffer) / 2)
		estimated_services = 0.0
		for colour in (WHITE, BLACK):
			history = self.raw_history[colour]
			avg_life = (
				mean(expected_remaining_wears(shade) for shade in history)
				if history
				else EXPECTED_FRESH_WEAR_LIFETIME
			)
			estimated_services += estimated_per_colour * avg_life

		remaining_days = self.days - turn.day + 1
		remaining_demand = 2 * self.roommates * remaining_days
		fresh_wears_needed = max(0.0, remaining_demand - estimated_services)
		wears_per_pack = EXPECTED_FRESH_WEAR_LIFETIME * PACK_SIZE
		packs_needed = ceil(fresh_wears_needed / wears_per_pack)
		return PACK_COST * (packs_needed + self.reserve_safety_packs)

	def can_discard(self, turn: TurnContext) -> bool:
		"""Whether replacement reserve and budget pace permit a discard."""
		# With five offered socks, matching is already easy in our tests and
		# voluntary discards only hurt, so preserve inventory.
		if self.selection_unit >= 5:
			return False

		if turn.budget_remaining < self.expected_replacement_reserve(turn):
			return False

		initial_budget = turn.total_spent + turn.budget_remaining
		if initial_budget == float('inf'):
			return True
		if initial_budget <= 0:
			return False

		budget_fraction = turn.budget_remaining / initial_budget
		time_fraction = max(0.0, (self.days - turn.day) / self.days)
		return budget_fraction - time_fraction > self.budget_pace_margin

	def choose_discards(
		self,
		offered: tuple[int, ...],
		wear: tuple[int, int],
		leftovers: list[int],
		turn: TurnContext,
	) -> tuple[int, ...]:
		"""Compare keeping each leftover with its eventual pristine replacement.

		Average the engine's thresholded mismatch against recent raw shades of
		the same colour. A symmetric outlier score can discard a young sock that
		will age into the population, only to buy another equally young sock.
		Instead require replacement to reduce the estimated mismatch by >6.

		This is a local proxy, not a prediction of our best pair in a future
		hand. Replacement is delayed until six household discards accumulate;
		the reserve/pace guards remain necessary and are not survival guarantees.
		"""
		if not leftovers or not self.can_discard(turn):
			return ()

		candidates: list[tuple[float, int]] = []
		for i in leftovers:
			shade = offered[i]
			history = self.raw_history[colour_of(shade)]
			if len(history) < self.min_dist_samples:
				continue

			pristine = 255 if colour_of(shade) == WHITE else 0
			gain = mean(
				pair_embarrassment(shade, observed) - pair_embarrassment(pristine, observed)
				for observed in history
			)
			if gain > self.replacement_gain_threshold:
				candidates.append((gain, i))

		if not candidates:
			return ()

		# Rank improvements; lower index breaks exact ties. Default limit is one.
		ranked = sorted(candidates, key=lambda item: (-item[0], item[1]))
		return tuple(index for _, index in ranked[: self.max_discards])

	@staticmethod
	def apply_action(
		stats: dict[str, WindowedStats],
		offered: tuple[int, ...],
		wear: tuple[int, int],
		leftovers: list[int],
		discard: tuple[int, ...],
	) -> None:
		"""Fold one day's outcome into ``stats`` (mutates in place)."""
		for i in wear:
			shade = aged_shade(offered[i])
			stats[colour_of(shade)].add(shade)
		for i in leftovers:
			if i in discard:
				continue
			stats[colour_of(offered[i])].add(offered[i])


def pairwise_sock_embarassments(offered: tuple[int, ...]) -> list[dict]:
	"""Every unordered pair of indices in ``offered`` with its shades and the
	embarrassment cost of wearing it."""
	results = []
	for i, j in combinations(range(len(offered)), 2):
		results.append(
			{
				'pair': (i, j),
				'shades': (offered[i], offered[j]),
				'embarrassment': pair_embarrassment(offered[i], offered[j]),
			}
		)
	return results
