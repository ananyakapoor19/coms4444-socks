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

from core.engine import PACK_COST, PACK_SIZE
from models.player import GameContext, PlayerSnapshot, Selection, TurnContext
from models.player import Player as BasePlayer

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


def pair_embarrassment(a: int, b: int, threshold: int) -> float:
	diff = abs(a - b)
	return float(diff) if diff > threshold else 0.0


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

		# If, for whatever reason, we want to reason about other embarrassment thresholds
		self.embarrassment_threshold = 6

		# Used for breaking ties of multiple zero-cost pairs
		# We project what the shade distribution will look like post-action for each pair
		# How many of the most recently returned shades per colour the projected spread covers.
		self.running_window_size = 5

		# Per-colour running mean/std of the socks we put back; drives choosing wear socks
		# Note: the ones we wear will be added at their *aged* shade
		self.stats: dict[str, WindowedStats] = {
			WHITE: WindowedStats(self.running_window_size),
			BLACK: WindowedStats(self.running_window_size),
		}

		# The raw shades actually offered to us (used in discard policy)
		# How many recent offered shades per colour we remember (5 in high-budget mode).
		self.raw_window_size = 20
		# Per-colour offered shades; drives discards, lifetimes and high-budget pair choice.
		self.raw_history: dict[str, deque[float]] = {
			WHITE: deque(maxlen=self.raw_window_size),
			BLACK: deque(maxlen=self.raw_window_size),
		}

		# Discard policy knobs.
		# Fewest observed shades of a colour before we will discard a sock of that colour.
		self.min_dist_samples = 3
		# Most socks we discard in one day: 1 with 5-sock hands, 2 with 4-sock hands.
		self.max_discards = 1 if self.selection_unit >= 5 else 2
		# A pristine replacement must cut the sock's mean embarrassment by more than this.
		self.replacement_gain_threshold = 6.0
		# Slots left out when estimating the drawer's remaining wears (min 2 * (PACK_SIZE - 1)).
		self.reserve_capacity_buffer = 14
		# Extra packs' worth of money held back on top of the estimated replacement need.
		self.reserve_safety_packs = 6
		# How far budget-left fraction must exceed days-left fraction before we discard.
		self.budget_pace_margin = 0
		# Turns played so far; used
		self.days_seen = 0
		# Set on day 0: True if we can actually enter high budget mode
		self.high_budget_mode = False
		# Set on day 0: infer original budget from turn 0 spent + remaining
		self.initial_budget = float('inf')

	# ------------------------------------------------------------------ policy

	def select_socks(self, offered: tuple[int, ...], turn: TurnContext) -> Selection:
		"""Choose today's pair to wear and which leftovers to discard.

		Args:
			offered: The socks' shades drawn for us today.
			turn: we extract 'day', 'total_spent', and 'budget_remaining

		Returns:
			'Selection(wear, discard)'

		Notes:
			- On the first turn we infer the initial budget and determine
				whether high-budget mode is on.
			- Every offered shade is added to 'raw_history' before any decision
		"""
		# One time basic setup at day 0; since "turn" variable is not available in constructor
		if self.days_seen == 0:
			self.initial_budget = turn.total_spent + turn.budget_remaining
			self.high_budget_mode = self.selection_unit == 4 and self.initial_budget >= 400
			if self.high_budget_mode:
				self.raw_window_size = 5  # We pick a more aggressive window size of 5 (shorter memory)
				self.raw_history = {c: deque(maxlen=5) for c in self.raw_history}

		self.days_seen += 1
		# Update the raw history with every shade we've observed just now
		for shade in offered:
			self.raw_history[colour_of(shade)].append(float(shade))
		pairs = pairwise_sock_embarrassments(offered, self.embarrassment_threshold)
		wear = self.choose_pair(offered, pairs)
		leftovers = [i for i in range(len(offered)) if i not in wear]
		discard = self.choose_discards(offered, wear, leftovers, turn)
		# Commit the chosen action to the real distribution; Bookkeeping: track socks we wore + put back + discarded; modifies self.stats
		self.record_returns(self.stats, offered, wear, leftovers, discard)
		return Selection(wear=wear, discard=discard)

	def choose_pair(self, offered: tuple[int, ...], pairs: list[dict]) -> tuple[int, int]:
		"""Choose which two of today's socks to wear.

		Args:
			offered: The socks' shades drawn for us today.
			pairs: Every pair of 'offered' with its 'pair' indices, 'shades'
				and 'embarrassment', from 'pairwise_sock_embarrassments'.

		Returns:
			The '(i, j)' indices into 'offered' of the pair to wear.

		Notes:
			- If no pair is free (every shade gap > 6), wear the pair with the
				smallest embarrassment (greedy selection)
			- Otherwise, score each free pair and wear the lowest score:
				- High-budget mode: how much washing moves the two socks away
					from their colour's mean recent offered shade, i.e. the sum of
					'(aged - mean)^2 - (shade - mean)^2'. Negative means the wash
					pulls the socks back toward typical, so outliers get worn.
				- Normal mode: project tomorrow's 'stats' (worn pair aged, the
					rest returned unchanged, no discards) and take the white +
					black std, i.e. how tightly the returned shades cluster.
			- Ties break by the pair's own shade gap, then by lower indices.
		"""

		# Compute which pairs yielded 0 embarrassment
		free_pairs = [p for p in pairs if p['embarrassment'] == 0.0]

		# If there weren't any free pairs, we greedily pick the pair with the smallest embarrassment
		if not free_pairs:
			return min(pairs, key=lambda p: p['embarrassment'])['pair']

		# Among the free pairs, we find the best pair
		best_pair: tuple[int, int] | None = None
		best_key: tuple[float, int, tuple[int, int]] | None = None

		for candidate in free_pairs:
			# Pair we're considering
			wear = candidate['pair']

			if self.high_budget_mode:
				# Add up how much wearing this pair moves each sock away from its color's
				# typical shade
				# Note: negative score = the wash pulls the socks closer to typical
				score = 0.0

				for shade in candidate['shades']:
					# The recently offered shades of this color
					history = self.raw_history[colour_of(shade)]

					# the mean shade of this color
					center = mean(history)

					# squared distance of this sock from its color's mean, currently
					distance_now = (shade - center) ** 2

					# squared distance of this sock from its color's mean, if we were to wear it
					distance_after_wash = (aged_shade(shade) - center) ** 2

					# how much the spread changes after wearing this sock
					# Recall: negative means wearing this sock makes the spread less (good)
					score += distance_after_wash - distance_now

			# Not in high budget mode
			else:
				# The leftovers go back unworn
				leftovers = [i for i in range(len(offered)) if i not in wear]

				# Copy our stats (want real ones to stay untouched)
				trial = {color: window_stats.copy() for color, window_stats in self.stats.items()}

				# Project tomorrow's stats based on wearing these socks and leaving the rest
				# Note: no discards yet, hence discard=()
				self.record_returns(
					stats=trial,
					offered=offered,
					wear=wear,
					leftovers=leftovers,
					discard=(),
				)

				# The score for this worn pair is how spread out white
				# and black shades would be afterwards, measured via std.
				score = sum(window_stats.std for window_stats in trial.values())

			# Break ties by score, then the pair's own shade gap (every free pair
			# has zero embarrassment, so the gap is what differs), then lower indices.
			gap = abs(candidate['shades'][0] - candidate['shades'][1])
			key = (score, gap, wear)

			# Keep the first candidate with the smallest key.
			if best_key is None or key < best_key:
				best_key = key
				best_pair = wear

		assert best_pair is not None
		return best_pair

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
			# The recently offered shades of this color
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
		# Five-sock hands use a separately tuned discard limit.
		if turn.budget_remaining < PACK_COST:
			return False

		if turn.budget_remaining < self.estimated_replacement_reserve(turn):
			return False

		if self.initial_budget == float('inf'):
			return True
		if self.initial_budget <= 0:
			return False

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
			# Sock we're considering
			shade = offered[i]

			# The recently offered shades of this color
			history = self.raw_history[colour_of(shade)]

			# Too few shades seen to judge this color yet
			if len(history) < self.min_dist_samples:
				continue

			# The shade of a brand-new sock of this color
			pristine = 255 if colour_of(shade) == WHITE else 0

			# For each recent shade, how much less embarrassing pairing with it
			# would be if this sock were replaced by a pristine one
			gains = []

			for other in history:
				# embarrassment of pairing this sock with the recent shade
				embarrassment_now = pair_embarrassment(
					shade, other, self.embarrassment_threshold
				)

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
			# Sock we're considering
			i = candidate['index']

			# Wears we'd lose by throwing out this sock, on top of those already chosen
			proposed_loss = lost_wears + expected_remaining_wears(offered[i])

			# Skip it if losing those wears leaves too little money for replacements
			if turn.budget_remaining < self.estimated_replacement_reserve(turn, proposed_loss):
				continue

			# Keep it
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


def pairwise_sock_embarrassments(offered: tuple[int, ...], threshold: int) -> list[dict]:
	"""Every unordered pair of indices in ``offered`` with its shades and the
	embarrassment cost of wearing it."""
	results = []
	for i, j in combinations(range(len(offered)), 2):
		results.append(
			{
				'pair': (i, j),
				'shades': (offered[i], offered[j]),
				'embarrassment': pair_embarrassment(offered[i], offered[j], threshold),
			}
		)
	return results