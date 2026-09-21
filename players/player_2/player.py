"""Group 2 player: distribution-aware sock selection.

The policy has three parts, each described in ``plan.md`` and
in ``POLICY_CHANGES.md`` next to this file:

1. Track the shade distribution of the socks we have seen, per colour, over a
   sliding window using Welford's online mean/variance update.
2. Wear the pair on offer with the least embarrassment.
3. For the socks we did not wear, enumerate every discard subset and keep the
   one that leaves our tracked distribution with the smallest std.
"""

from collections import deque
from itertools import combinations
from math import sqrt

from core.engine import PACK_COST
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

		# Sliding-window distribution of the shades we believe are in the
		# drawer, tracked separately per colour. Black and white shades live in
		# disjoint ranges (0-64 vs 127-255), so a single mixed distribution
		# would just measure the black/white ratio rather than how well the
		# socks within a colour match each other.
		self.running_window_size = 20
		self.stats: dict[str, WindowedStats] = {
			WHITE: WindowedStats(self.running_window_size),
			BLACK: WindowedStats(self.running_window_size),
		}

		# Do not discard socks of a colour until we have seen at least this
		# many of them - a std over two or three samples says nothing.
		self.min_dist_samples = 10

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

		# 3. Among every discard subset of the leftovers, keep the one that
		#    leaves our tracked distribution with the smallest std.
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

	def can_discard(self, turn: TurnContext) -> bool:
		"""Cooperative budget guard: only discard while the household is on or
		under its average spending pace and can still afford a six-pack.

		With no ``--budget`` both ``budget_remaining`` and the pace threshold
		are ``inf``, so discarding is always allowed on an unlimited run.
		"""
		if turn.budget_remaining < PACK_COST:
			return False
		initial_budget = turn.total_spent + turn.budget_remaining
		pace = initial_budget / self.days
		return turn.total_spent / turn.day <= pace

	def choose_discards(
		self,
		offered: tuple[int, ...],
		wear: tuple[int, int],
		leftovers: list[int],
		turn: TurnContext,
	) -> tuple[int, ...]:
		if not leftovers or not self.can_discard(turn):
			return ()

		# A leftover is only eligible for discard once we have enough samples
		# of its colour to trust the std.
		eligible = [
			i for i in leftovers if self.stats[colour_of(offered[i])].n >= self.min_dist_samples
		]

		best_action: tuple[int, ...] = ()
		best_key: tuple[float, int] | None = None
		for r in range(len(eligible) + 1):
			for subset in combinations(eligible, r):
				trial = {c: s.copy() for c, s in self.stats.items()}
				self.apply_action(trial, offered, wear, leftovers, subset)
				total_std = sum(s.std for s in trial.values())
				# Ties go to the cheaper action (fewer discards).
				key = (total_std, len(subset))
				if best_key is None or key < best_key:
					best_key = key
					best_action = subset
		return tuple(sorted(best_action))

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
