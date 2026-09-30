"""Group 2 player: distribution-aware sock selection.

The policy has three parts, described in ``POLICY_CHANGES.md``:

1. Track the shade distribution of the socks we have seen, per colour, over a
   sliding window using Welford's online mean/variance update.
2. Use zero-embarrassment choices to improve the projected distribution.
3. Discard leftovers when a pristine replacement has lower estimated matching
   cost and replacement needs and budget pace allow it, up to the configured cap.
"""

from collections import deque
from itertools import combinations
from math import ceil, sqrt
from statistics import mean
from typing import NamedTuple

from core.engine import HOLE_PROBABILITY, PACK_COST, PACK_SIZE
from models.player import GameContext, PlayerSnapshot, Selection, TurnContext
from models.player import Player as BasePlayer

# Shade semantics, mirrored from models/sock.py. A shade above BLACK_CEILING is
# a white sock and a shade at or below it is a black one; the ranges never
# overlap, so colour can be inferred from shade without ambiguity.
WHITE_START = 255
WHITE_FLOOR = 127
WHITE_FADE = 2
BLACK_START = 0
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


def pair_embarrassment(a: int, b: int, threshold: int) -> float:
	diff = abs(a - b)
	return float(diff) if diff > threshold else 0.0


def expected_remaining_wears(shade: float) -> float:
	"""Expected useful wears remaining for a sock with this observed shade."""
	# A fully faded sock gets a hole with probability HOLE_PROBABILITY per wear,
	# so it survives 1 / 0.25 = 4 more wears in expectation.
	terminal_wears = 1.0 / HOLE_PROBABILITY
	if colour_of(int(shade)) == WHITE:
		return (shade - WHITE_FLOOR) / WHITE_FADE + terminal_wears
	return BLACK_CEILING - shade + terminal_wears


# 68 for either colour: 64 fading wears plus 4 terminal ones.
EXPECTED_FRESH_WEAR_LIFETIME = expected_remaining_wears(WHITE_START)


class Pair(NamedTuple):
	"""Two of today's offered socks and the embarrassment of wearing them."""

	indices: tuple[int, int]  # positions in 'offered'
	shades: tuple[int, int]
	embarrassment: float

	@property
	def gap(self) -> int:
		"""How far apart the two shades are."""
		return abs(self.shades[0] - self.shades[1])


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

		# If, for whatever reason, we want to reason about other embarrassment thresholds
		self.embarrassment_threshold = 6

		# Mean/std of the last 5 socks we put back, per colour (worn ones at their
		# faded shade). Used to break ties between zero-cost pairs.
		self.running_window_size = 5
		self.stats: dict[str, WindowedStats] = {
			WHITE: WindowedStats(self.running_window_size),
			BLACK: WindowedStats(self.running_window_size),
		}

		# The last 20 shades offered to us, per colour (5 in high-budget mode).
		# Used for discards, the money reserve and high-budget pair choice.
		self.raw_window_size = 20
		self.raw_history: dict[str, deque[float]] = {
			WHITE: deque(maxlen=self.raw_window_size),
			BLACK: deque(maxlen=self.raw_window_size),
		}

		# Discard policy knobs.
		# Fewest observed shades of a colour before we will discard a sock of that colour.
		self.min_dist_samples = 3
		# Most socks we discard in one day: 1 with 5+ sock hands, 2 with 4-sock hands.
		self.max_discards = 1 if self.selection_unit >= 5 else 2
		# A pristine replacement must cut the sock's mean embarrassment by more than this.
		self.replacement_gain_threshold = 6.0
		# Slots left out when estimating the drawer's remaining wears (min 2 * (PACK_SIZE - 1)).
		self.reserve_capacity_buffer = 14
		# Extra packs' worth of money held back on top of the estimated replacement need.
		self.reserve_safety_packs = 6
		# How far budget-left fraction must exceed days-left fraction before we discard.
		self.budget_pace_margin = 0
		# High-budget mode: 4-sock hands with at least this much budget use a
		# shorter raw window and a different zero-cost tie-break.
		self.high_budget_threshold = 400
		self.high_budget_raw_window = 5
		# Set on day 0: True if we can actually enter high budget mode
		self.high_budget_mode = False
		# Budget at the start of the game (inf if no limit). None until our first turn.
		self.initial_budget: float | None = None

	# ------------------------------------------------------------------ policy

	def select_socks(self, offered: tuple[int, ...], turn: TurnContext) -> Selection:
		"""Choose today's pair to wear and which leftovers to discard.

		Args:
			offered: The socks' shades drawn for us today.
			turn: we extract 'day', 'total_spent', and 'budget_remaining'

		Returns:
			'Selection(wear, discard)'

		Notes:
			- On the first turn we infer the initial budget and determine
				whether high-budget mode is on.
			- Every offered shade is added to 'raw_history' before any decision
		"""
		# One time basic setup at day 0; since "turn" variable is not available in constructor
		if self.initial_budget is None:
			self.initial_budget = turn.total_spent + turn.budget_remaining
			self.high_budget_mode = (
				self.selection_unit == 4 and self.initial_budget >= self.high_budget_threshold
			)
			if self.high_budget_mode:
				self.raw_window_size = self.high_budget_raw_window  # shorter memory
				self.raw_history = {c: deque(maxlen=self.raw_window_size) for c in self.raw_history}

		# Update the raw history with every shade we've observed just now
		for shade in offered:
			self.raw_history[colour_of(shade)].append(float(shade))
		pairs = pairwise_sock_embarrassments(offered, self.embarrassment_threshold)
		wear = self.choose_pair(offered, pairs)
		leftovers = [i for i in range(len(offered)) if i not in wear]
		discard = self.choose_discards(offered, wear, leftovers, turn)
		# Remember what we put back in the drawer today (updates self.stats)
		self.record_returns(self.stats, offered, wear, leftovers, discard)
		return Selection(wear=wear, discard=discard)

	def choose_pair(self, offered: tuple[int, ...], pairs: list[Pair]) -> tuple[int, int]:
		"""Choose which two of today's socks to wear.

		Args:
			offered: The socks' shades drawn for us today.
			pairs: Every 'Pair' of 'offered', from 'pairwise_sock_embarrassments'.

		Returns:
			The '(i, j)' indices into 'offered' of the pair to wear.

		Notes:
			- If no pair is free (every shade gap > 6), wear the pair with the
				smallest embarrassment (greedy selection)
			- Otherwise, score each free pair and wear the lowest score:
				- High-budget mode: 'distance_from_average_after_wearing'
				- Normal mode: 'drawer_spread_after_wearing'
			- Ties break by the pair's own shade gap, then by lower indices.
		"""

		free_pairs = [p for p in pairs if p.embarrassment == 0.0]

		# If there weren't any free pairs, we greedily pick the pair with the smallest embarrassment
		if not free_pairs:
			return min(pairs, key=lambda p: p.embarrassment).indices

		# How we score a free pair (lower is better) depends on the mode
		if self.high_budget_mode:
			score = self.distance_from_average_after_wearing
		else:
			score = self.drawer_spread_after_wearing

		# Break ties by score, then the pair's own shade gap (every free pair
		# has zero embarrassment, so the gap is what differs), then lower indices.
		def rank(candidate: Pair):
			return (score(offered, candidate), candidate.gap, candidate.indices)

		return min(free_pairs, key=rank).indices

	def distance_from_average_after_wearing(
		self, offered: tuple[int, ...], candidate: Pair
	) -> float:
		"""High-budget score: does wearing this pair fade it toward or away from typical?

		For each sock, compare its squared distance from the mean recent offered
		shade of its colour after washing vs. now. Negative means the wash pulls
		the socks back toward typical, so outliers get worn. 'offered' is unused
		but kept so both scores can be called the same way.
		"""
		score = 0.0

		for shade in candidate.shades:
			# the mean shade of the recently offered socks of this color
			center = mean(self.raw_history[colour_of(shade)])

			# squared distance of this sock from its color's mean, now and if we wore it
			distance_now = (shade - center) ** 2
			distance_after_wash = (aged_shade(shade) - center) ** 2

			# Recall: negative means wearing this sock makes the spread less (good)
			score += distance_after_wash - distance_now

		return score

	def drawer_spread_after_wearing(self, offered: tuple[int, ...], candidate: Pair) -> float:
		"""Normal-mode score: how spread out the socks we put back would be.

		Projects tomorrow's 'stats' as if we wore this pair (aged) and returned
		the rest unchanged, then adds the white and black std.
		"""
		wear = candidate.indices

		# The leftovers go back unworn
		leftovers = [i for i in range(len(offered)) if i not in wear]

		# Copy our stats (want real ones to stay untouched)
		trial = {color: window_stats.copy() for color, window_stats in self.stats.items()}

		# Project tomorrow's stats; no discards yet, hence discard=()
		self.record_returns(
			stats=trial,
			offered=offered,
			wear=wear,
			leftovers=leftovers,
			discard=(),
		)

		return sum(window_stats.std for window_stats in trial.values())

	def estimated_replacement_reserve(self, turn: TurnContext, lost_wears: float = 0.0) -> float:
		"""Estimate how much money to hold back for future replacement packs.

		Args:
			turn: we extract 'day'
			lost_wears: Wears we'd give up by the discards proposed so far today
				(from 'choose_discards'). 0 when just checking the reserve.

		Returns:
			The dollars to keep in reserve: packs needed to cover the shortfall,
			plus 'reserve_safety_packs', times 'PACK_COST'.

		Notes:
			- Supply: wears the drawer still holds, estimated as socks per color
				times the average wears left of recently offered socks of that color.
			- Demand: wears the whole household needs for the rest of the game,
				2 socks per roommate per day, including today.
			- Every pack we'd need to cover demand - supply is priced in, where
				each fresh sock is worth 'EXPECTED_FRESH_WEAR_LIFETIME' wears.
			- This is an estimate: we can't see the real drawer, only what we've
				been offered, and other players can spend the shared budget.
		"""

		# Sock slots we leave out of the estimate to stay conservative
		# Note: never fewer than 2 * (PACK_SIZE - 1), since each color can have up to
		# PACK_SIZE - 1 lost socks waiting for a pack after replenishment
		excluded_slots = max(2 * (PACK_SIZE - 1), self.reserve_capacity_buffer)

		# How many socks of each color we count on having in the drawer
		socks_per_color = max(0.0, (self.capacity - excluded_slots) / 2)

		# Wears the drawer can still provide, summed over both colors
		wears_available = 0.0

		for color in (WHITE, BLACK):
			history = self.raw_history[color]

			# Average wears left in a sock of this color
			# Note: a fresh sock's lifetime if we haven't seen this color yet
			if history:
				wears_left_per_sock = mean(expected_remaining_wears(s) for s in history)
			else:
				wears_left_per_sock = EXPECTED_FRESH_WEAR_LIFETIME

			wears_available += socks_per_color * wears_left_per_sock

		# Take away the wears lost to the discards proposed so far
		wears_available = max(0.0, wears_available - lost_wears)

		# Wears the household needs for the rest of the game: 2 socks per roommate
		# per day, including today
		wears_needed = 2 * self.roommates * (self.days - turn.day + 1)

		# Packs needed to cover the shortfall (each pack is PACK_SIZE fresh socks)
		shortfall = max(0.0, wears_needed - wears_available)
		packs_needed = ceil(shortfall / (EXPECTED_FRESH_WEAR_LIFETIME * PACK_SIZE))

		# Reserve money for those packs, plus a safety margin of extra packs
		return PACK_COST * (packs_needed + self.reserve_safety_packs)

	def can_discard(self, turn: TurnContext) -> bool:
		"""Whether replacement reserve and budget pace permit a discard."""
		if turn.budget_remaining < PACK_COST:
			return False

		if turn.budget_remaining < self.estimated_replacement_reserve(turn):
			return False

		if self.initial_budget == float('inf'):
			return True

		budget_fraction = turn.budget_remaining / self.initial_budget
		time_fraction = max(0.0, (self.days - turn.day) / self.days)
		return budget_fraction - time_fraction > self.budget_pace_margin

	def choose_discards(
		self,
		offered: tuple[int, ...],
		wear: tuple[int, int],
		leftovers: list[int],
		turn: TurnContext,
	) -> tuple[int, ...]:
		"""Choose which of today's leftover socks to throw out.

		Args:
			offered: The socks' shades drawn for us today.
			wear: The '(i, j)' indices of the pair we're wearing. Unused here, but
				kept so overrides in 'run_experiments' share the signature.
			leftovers: The indices of 'offered' we aren't wearing.
			turn: we extract 'day' and 'budget_remaining'

		Returns:
			The indices into 'offered' to discard (possibly none).

		Notes:
			- Nothing is discarded unless 'can_discard' allows it (money left,
				replacement reserve and budget pace).
			- Each leftover's gain is how much lower its average embarrassment
				against recent shades of its color would be if it were replaced
				by a pristine sock. Only gains > 'replacement_gain_threshold' qualify.
			- Qualifying socks are taken biggest gain first (ties: lower index),
				skipping any whose lost wears would break the replacement reserve,
				up to 'max_discards'.
		"""

		# Check whether we're allowed to discard anything today
		if not leftovers or self.max_discards <= 0 or not self.can_discard(turn):
			return ()

		# Leftovers whose replacement would be a meaningful improvement
		candidates = []

		for i in leftovers:
			shade = offered[i]
			history = self.raw_history[colour_of(shade)]

			# Too few shades seen to judge this color yet
			if len(history) < self.min_dist_samples:
				continue

			# The shade of a brand-new sock of this color
			pristine = WHITE_START if colour_of(shade) == WHITE else BLACK_START

			# For each recent shade, how much less embarrassing pairing with it
			# would be if this sock were replaced by a pristine one
			gains = []

			for other in history:
				# embarrassment of pairing this sock with the recent shade
				embarrassment_now = pair_embarrassment(shade, other, self.embarrassment_threshold)

				# embarrassment of pairing a pristine sock with the recent shade
				embarrassment_if_replaced = pair_embarrassment(
					pristine, other, self.embarrassment_threshold
				)

				# Recall: positive means replacing this sock helps (good)
				gains.append(embarrassment_now - embarrassment_if_replaced)

			# The gain for this sock is its average over the recent shades
			gain = mean(gains)

			# Only keep socks whose replacement is a meaningful improvement
			if gain > self.replacement_gain_threshold:
				candidates.append({'index': i, 'gain': gain})

		# Rank by biggest gain first, then lower indices.
		def rank(candidate):
			return (-candidate['gain'], candidate['index'])

		# Among the candidates, we pick the discards
		discard = []
		lost_wears = 0.0

		for candidate in sorted(candidates, key=rank):
			i = candidate['index']

			# Wears we'd lose by throwing out this sock, on top of those already chosen
			proposed_loss = lost_wears + expected_remaining_wears(offered[i])

			# Skip it if losing those wears leaves too little money for replacements
			if turn.budget_remaining < self.estimated_replacement_reserve(turn, proposed_loss):
				continue

			discard.append(i)
			lost_wears = proposed_loss

			# Stop once we hit the daily discard cap
			if len(discard) >= self.max_discards:
				break

		return tuple(discard)

	@staticmethod
	def record_returns(
		stats: dict[str, WindowedStats],
		offered: tuple[int, ...],
		wear: tuple[int, int],
		leftovers: list[int],
		discard: tuple[int, ...],
	) -> None:
		"""
		Fold one day's outcome into ``stats`` (mutates in place).
		Wear socks are added with newer/aged shades
		Put Back socks are added back as it is shade
		Discard shades do not impact the history (we do not remove them)
		"""
		for i in wear:
			shade = aged_shade(offered[i])
			stats[colour_of(shade)].add(shade)
		for i in leftovers:
			if i in discard:
				continue
			stats[colour_of(offered[i])].add(offered[i])


def pairwise_sock_embarrassments(offered: tuple[int, ...], threshold: int) -> list[Pair]:
	"""Every unordered pair of indices in ``offered`` with its shades and the
	embarrassment cost of wearing it."""
	results = []
	for i, j in combinations(range(len(offered)), 2):
		results.append(
			Pair(
				indices=(i, j),
				shades=(offered[i], offered[j]),
				embarrassment=pair_embarrassment(offered[i], offered[j], threshold),
			)
		)
	return results
