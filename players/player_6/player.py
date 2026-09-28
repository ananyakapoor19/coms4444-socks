"""Starting point for a group's player.

Copy this whole directory to ``players/player_<k>/`` using your group number,
then rename the class to ``Player<k>``. Group 4 would end up with
``players/player_4/player.py`` containing ``class Player4``. The registry looks
for exactly that; nothing else needs editing.

Keep the ``__init__.py``. Discovery uses ``pkgutil.iter_modules``, which only
reports directories that have one, so a group directory without it is silently
invisible to the simulator - no error, just a player that never turns up.

This directory is not itself discovered - the registry only matches
``player_<digits>`` - so the template can never appear in a run as a competitor.
"""

from itertools import combinations
from math import exp

from models.player import GameContext, PlayerSnapshot, Selection, TurnContext
from models.player import Player as BasePlayer

THRESHOLD = 6
BUCKETS = 8
HIST_DECAY = 0.985
PACK_COST = 10.0
# WHITE_CUTOFF = 200
ENDGAME_START = 0.8
ENDGAME_RESERVE = 0.2
SPEND_RATE_ALPHA = 0.3
MIN_COMPATIBILITY = 0.02
MAX_COMPATIBILITY = 0.18

# tryign to add discard score calculations
DISCARD_THRESHOLD = 1.0
TERMINAL_DISCARD_SCORE = 3.0
WHITE_AGE_START = 200
WHITE_AGE_RANGE = 73
BLACK_AGE_START = 40
BLACK_AGE_RANGE = 24
OUTLIER_WEIGHT = 1.0
AGE_SCORE_WEIGHT = 1.0
AGGRESSION_SCORE_WEIGHT = 0.75

# Fine-grained age histogram for population estimation.
AGE_BINS = 65
FAST_HIST_WEIGHT = 0.80
SLOW_HIST_WEIGHT = 0.20
SLOW_HIST_DECAY = 0.985
PRIOR_MASS_PER_COLOUR = 10.0

# Batch discard budget.
RESERVE_PACKS = 4
RESERVE_FRACTION = 0.03
WARMUP_DAYS = 20
ENDGAME_DAYS = 20
CREDIT_CAP = 8.0
MAX_DISCARDS = 3
UNDERPACE_BONUS = 0.50

# A third discretionary discard is only available in the five-sock variant.
EXTRA_DISCARD_THRESHOLD = 1.35

# Conservative discard policy for large 5-sock households.
CROWDED_ROOMMATE_CUTOFF = 7
CROWDED_UNIT = 5
CROWDED_WARMUP_FRACTION = 0.35
CROWDED_AGGRESSION_SCALE = 0.65
CROWDED_DISCARD_THRESHOLD = 1.75
CROWDED_MAX_DISCARDS = 1

# Small tie-break signal for choosing among equally good pairs. Immediate
# embarrassment and compatibility remain more important.
LEFTOVER_VALUE_WEIGHT = 0.20

# Future population signal.
POPULATION_WEIGHT = 0.50
POPULATION_DISCARD_PENALTY = 0.75
SOCK_REPLACEMENT_COST = PACK_COST / 6.0


class Player6(BasePlayer):
	"""Rename me to Player<k>, where <k> is your group number."""

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)

		# super() has already set these from ctx and snapshot:
		#
		#   self.index           which roommate you are (0-based)
		#   self.id              your UUID, stable for the whole simulation
		#   self.capacity        C, the drawer size at the start
		#   self.roommates       n, how many of you share the drawer
		#   self.selection_unit  how many socks you are handed each day
		#   self.days            how long the simulation runs
		#
		# The engine constructs you once, before day 1, and it constructs you
		# itself - you cannot preload state into an already-built object. Anything
		# you want to carry between days lives on self, so initialise it here.
		self.days_seen = 0
		self.white_hist = [1.0] * BUCKETS
		self.black_hist = [1.0] * BUCKETS
		self.estimated_budget = None
		self.last_spend = 0.0
		self.last_spend_day = 0
		self.recent_spend_rate = 0.0

		# Fine-grained population beliefs.
		self.fast_age_hist = [[0.0] * AGE_BINS for _ in range(2)]
		self.slow_age_hist = [[0.0] * AGE_BINS for _ in range(2)]
		for colour in range(2):
			self.fast_age_hist[colour][0] = PRIOR_MASS_PER_COLOUR
			self.slow_age_hist[colour][0] = PRIOR_MASS_PER_COLOUR

		# Local discard credit.
		self.discard_credit = 0.0
		self.my_virtual_discard_spend = 0.0
		self._population_cache = {}

	def _is_black(self, shade: int) -> bool:
		return shade <= 64

	def _bucket(self, shade: int) -> int:
		value = shade / 65 if self._is_black(shade) else (shade - 127) / 129
		index = int(value * BUCKETS)
		if index < 0:
			return 0
		if index >= BUCKETS:
			return BUCKETS - 1
		return index

	def _update_histograms(self, offered: tuple[int, ...]) -> None:
		for hist in (self.white_hist, self.black_hist):
			for i in range(BUCKETS):
				hist[i] *= HIST_DECAY
		for shade in offered:
			hist = self.black_hist if self._is_black(shade) else self.white_hist
			hist[self._bucket(shade)] += 1.0

	# Map both colours to a common age coordinate.
	def _age(self, shade: int) -> int:
		if self._is_black(shade):
			return max(0, min(64, shade))
		return max(0, min(64, (255 - shade) // 2))

	def _shade_from_age(self, colour: int, age: int) -> int:
		age = max(0, min(64, age))
		if colour == 0:  # black
			return age
		return 255 - 2 * age

	# Update the fine-grained population estimate.
	def _update_population_histograms(self, offered: tuple[int, ...]) -> None:
		self._population_cache.clear()
		fast_decay = exp(-self.selection_unit / max(1, self.capacity))
		for colour in range(2):
			for age in range(AGE_BINS):
				self.fast_age_hist[colour][age] *= fast_decay
				self.slow_age_hist[colour][age] *= SLOW_HIST_DECAY
		for shade in offered:
			colour = 0 if self._is_black(shade) else 1
			age = self._age(shade)
			self.fast_age_hist[colour][age] += 1.0
			self.slow_age_hist[colour][age] += 1.0

	def _population_potential(self, shade: int) -> float:
		"""Estimate how useful this sock is for finding a same-colour partner."""
		if shade in self._population_cache:
			return self._population_cache[shade]
		colour = 0 if self._is_black(shade) else 1
		age = self._age(shade)
		fast = self.fast_age_hist[colour]
		slow = self.slow_age_hist[colour]
		fast_total = sum(fast)
		slow_total = sum(slow)
		if fast_total <= 0.0 or slow_total <= 0.0:
			return 0.0
		age_radius = 6 if colour == 0 else 3
		lo = max(0, age - age_radius)
		hi = min(AGE_BINS - 1, age + age_radius)
		fast_mass = sum(fast[lo : hi + 1]) / fast_total
		slow_mass = sum(slow[lo : hi + 1]) / slow_total
		potential = FAST_HIST_WEIGHT * fast_mass + SLOW_HIST_WEIGHT * slow_mass
		self._population_cache[shade] = potential
		return potential

	def _budget_discard_limit(
		self,
		turn: TurnContext,
		unit: int,
	) -> tuple[float, int]:
		"""Return discard aggression and today's discretionary discard limit."""
		max_leftovers = max(0, unit - 2)
		if turn.budget_remaining == float('inf'):
			return self._discard_aggression(turn), min(MAX_DISCARDS, max_leftovers)

		current_budget = turn.total_spent + turn.budget_remaining
		if self.estimated_budget is None or current_budget > self.estimated_budget:
			self.estimated_budget = current_budget
		reserve = max(
			RESERVE_PACKS * PACK_COST,
			self.estimated_budget * RESERVE_FRACTION,
		)
		slack = max(0.0, turn.budget_remaining - reserve)
		days_left = max(1, self.days - turn.day + 1)
		if days_left <= ENDGAME_DAYS:
			return 0.0, 0

		warmup = turn.day <= WARMUP_DAYS
		target_to_date = self.estimated_budget * turn.day / max(1, self.days)
		underpace = max(0.0, target_to_date - turn.total_spent)
		underpace_fraction = underpace / max(1.0, self.estimated_budget)
		share = min(
			1.0,
			1.0 / max(1, self.roommates) + UNDERPACE_BONUS * underpace_fraction,
		)
		allowed_daily_dollars = slack / days_left * share
		self.discard_credit = min(
			CREDIT_CAP,
			self.discard_credit + allowed_daily_dollars / SOCK_REPLACEMENT_COST,
		)
		if warmup:
			return 0.0, 0

		aggression = self._discard_aggression(turn)
		allowed = min(MAX_DISCARDS, max_leftovers, int(self.discard_credit))
		if self.roommates >= CROWDED_ROOMMATE_CUTOFF and unit >= CROWDED_UNIT:
			if turn.day <= int(self.days * CROWDED_WARMUP_FRACTION):
				return 0.0, 0
			aggression *= CROWDED_AGGRESSION_SCALE
			allowed = min(allowed, CROWDED_MAX_DISCARDS)
		return aggression, allowed

	def _compatibility(self, shade: int) -> float:
		hist = self.black_hist if self._is_black(shade) else self.white_hist
		total = sum(hist)
		return hist[self._bucket(shade)] / total if total else 0.0

	def _embarrassment(self, a: int, b: int) -> int:
		diff = abs(a - b)
		return 0 if diff <= THRESHOLD else diff

	def _observe_spending(self, turn: TurnContext) -> None:
		elapsed = turn.day - self.last_spend_day
		if elapsed > 0:
			daily_spend = max(0.0, turn.total_spent - self.last_spend) / elapsed
			if self.last_spend_day == 0:
				self.recent_spend_rate = daily_spend
			else:
				self.recent_spend_rate = (
					SPEND_RATE_ALPHA * daily_spend
					+ (1.0 - SPEND_RATE_ALPHA) * self.recent_spend_rate
				)
		self.last_spend = turn.total_spent
		self.last_spend_day = turn.day

	def _discard_aggression(self, turn: TurnContext) -> float:
		"""Scale discarding by time left and the household's recent spend pace."""
		if turn.budget_remaining == float('inf'):
			progress = turn.day / self.days if self.days else 1.0
			return 0.75 + 0.75 * progress
		current_budget = turn.total_spent + turn.budget_remaining
		if self.estimated_budget is None or current_budget > self.estimated_budget:
			self.estimated_budget = current_budget
		reserve = self.estimated_budget * ENDGAME_RESERVE
		spendable = turn.budget_remaining - reserve
		if spendable < PACK_COST:
			return 0.0
		progress = turn.day / self.days if self.days else 1.0
		days_left = max(1, self.days - turn.day + 1)
		affordable_rate = spendable / days_left
		if self.recent_spend_rate <= 0.0:
			pace_factor = 1.5
		else:
			pace_factor = affordable_rate / self.recent_spend_rate
		pace_factor = min(1.5, max(0.25, pace_factor))
		# Time pressure rises smoothly, with an extra push during the final 20%.
		time_factor = 0.75 + 0.5 * progress
		if progress >= ENDGAME_START:
			time_factor += 0.25 * ((progress - ENDGAME_START) / (1.0 - ENDGAME_START))
		return min(2.0, pace_factor * time_factor)

	def _age_score(self, shade: int) -> float:
		# noramlized score of how worn sock is, 0 is fresh, 1 is terminal shade
		if self._is_black(shade):
			age = (shade - BLACK_AGE_START) / BLACK_AGE_RANGE
		else:
			age = (WHITE_AGE_START - shade) / WHITE_AGE_RANGE
		return min(1.0, max(0.0, age))

	def _outlier_score(self, shade: int) -> float:
		# return how unusual sock is in observed distribution
		# common low score, rare high score
		compatibility = self._compatibility(shade)
		return max(0.0, 1.0 - compatibility / 0.25)

	def _discard_score(self, shade: int, aggression: float) -> float:
		# basef on how worn it is, how unusual, how agressive we want to discard
		# calculate discard score, if over threshold discard
		age = self._age_score(shade)
		outlier = self._outlier_score(shade)
		# Population potential reduces the discard score.
		population = self._population_potential(shade)
		score = (
			AGE_SCORE_WEIGHT * age
			+ OUTLIER_WEIGHT * outlier
			- POPULATION_DISCARD_PENALTY * population
		)
		score *= 1.0 + AGGRESSION_SCORE_WEIGHT * (aggression - 1.0)
		score = max(0.0, score)
		if self._is_black(shade):
			if shade == 64:
				score += TERMINAL_DISCARD_SCORE
		else:
			if shade == 127:
				score += TERMINAL_DISCARD_SCORE
		return score

	def _leftover_value(self, shade: int) -> float:
		"""Estimate how valuable it is to leave this sock in the drawer.

		Higher population potential means the sock is likely to be useful for a
		future compatible pair. Older or unusual socks are less valuable.
		This is only a pair-selection tie-break; _discard_score still controls
		whether a leftover is actually thrown away.
		"""
		population = self._population_potential(shade)
		age = self._age_score(shade)
		outlier = self._outlier_score(shade)
		return population - 0.35 * age - 0.25 * outlier

	def select_socks(self, offered: tuple[int, ...], turn: TurnContext) -> Selection:
		"""Choose two socks to wear, and decide the fate of the rest.

		Called once per day, in an order that is reshuffled daily. Everything you
		are allowed to know is in the two arguments.

		``offered`` is a tuple of ``selection_unit`` shade values, 0-255.

		WHAT YOU CAN SEE

		offered[i]                  the shade of the i-th sock on offer
		turn.day                    today's day number, 1-based
		turn.total_spent            dollars spent by the household so far
		turn.embarrassment_history  your own daily scores, one per day
		turn.total_embarrassment    the sum of that history
		self.capacity / self.roommates / self.selection_unit / self.days

		WHAT YOU CANNOT SEE

		- Which sock is which. Indices are positions in THIS tuple only. The
		same index tomorrow is a different sock, so you cannot track an
		individual sock across turns or build up a map of the drawer.
		- Anyone else's socks, choices or embarrassment.
		- The shade distribution left in the drawer.
		- How many socks have been discarded, or how close the household is to
		the next six-pack. You see total_spent only, after the fact.

		With n == 1 you are alone with the drawer, so tracking its full state IS
		possible. That is intentional, not a leak - it is what makes the pooled
		versus separate comparison in goal 3 meaningful.

		WHAT THE SHADES MEAN

		White socks start at 255 and fade by 2 per wear, stopping at 127. Black
		socks start at 0 and rise by 1 per wear, stopping at 64. The two ranges
		never overlap, so a shade above 64 is a white sock and a shade at or below
		64 is a black one. Inferring colour from shade is fair game.

		Wearing a pair whose shades differ by MORE than 6 costs you that
		difference. A difference of exactly 6 is free.

		A sock already at 127 or 64 when you are handed it has a 25% chance of
		developing a hole when worn, and is thrown out immediately. Six discards
		of one colour buy a fresh six-pack for $10, and the surplus carries over.

		RETURNING A DECISION

		wear     exactly two distinct indices into ``offered``
		discard  any subset of the REMAINING indices, possibly empty

		Anything you neither wear nor discard goes back in the drawer unworn and
		keeps its shade. Only worn socks age.

		IF YOU GET IT WRONG

		An invalid selection, an exception, or taking longer than the --timeout
		budget forfeits your turn: the engine wears the first two socks and
		discards nothing. It is recorded as a fault and shown in the results, so a
		forfeit is visible rather than silent. Your failure never affects the
		other groups.
		"""
		self.days_seen += 1
		unit = len(offered)

		# Defensive fallback for malformed input.
		if unit < 2:
			return Selection(wear=(0, 1), discard=())

		self._update_histograms(offered)
		self._observe_spending(turn)
		self._update_population_histograms(offered)

		aggression, discard_limit = self._budget_discard_limit(turn, unit)

		best_pair = (0, 1)
		best_key = None
		for i, j in combinations(range(unit), 2):
			a = offered[i]
			b = offered[j]
			embarrassment = self._embarrassment(a, b)
			cross_colour = self._is_black(a) != self._is_black(b)
			closeness = abs(a - b)
			compatibility = self._compatibility(a) + self._compatibility(b)

			# Future population value is a tie-break signal.
			future_population = self._population_potential(
				self._shade_from_age(
					0 if self._is_black(a) else 1,
					min(64, self._age(a) + 1),
				)
			) + self._population_potential(
				self._shade_from_age(
					0 if self._is_black(b) else 1,
					min(64, self._age(b) + 1),
				)
			)

			# Prefer useful leftovers when higher-priority signals tie.
			leftover_value = sum(
				self._leftover_value(shade)
				for index, shade in enumerate(offered)
				if index not in (i, j)
			)

			key = (
				embarrassment,
				cross_colour,
				-future_population * POPULATION_WEIGHT,
				-compatibility,
				-leftover_value * LEFTOVER_VALUE_WEIGHT,
				closeness,
			)
			if best_key is None or key < best_key:
				best_key = key
				best_pair = (i, j)

		discard: list[int] = []
		if aggression > 0.0 and discard_limit > 0:
			candidates = []
			for i, shade in enumerate(offered):
				if i in best_pair:
					continue
				score = self._discard_score(shade, aggression)
				candidates.append((score, i))

			# Use a stronger threshold for a third discard.
			candidates.sort(reverse=True)
			selected = []
			for score, index in candidates:
				if len(selected) >= discard_limit:
					break
				threshold = DISCARD_THRESHOLD
				if len(selected) >= 2:
					threshold = EXTRA_DISCARD_THRESHOLD
				# Require stronger evidence in crowded 5-sock households.
				if self.roommates >= CROWDED_ROOMMATE_CUTOFF and unit >= CROWDED_UNIT:
					threshold = max(threshold, CROWDED_DISCARD_THRESHOLD)
				if score >= threshold:
					selected.append(index)

			discard = selected
			if discard:
				self.discard_credit = max(
					0.0,
					self.discard_credit - len(discard),
				)
				# Track local discard spending.
				self.my_virtual_discard_spend += len(discard) * SOCK_REPLACEMENT_COST

		return Selection(wear=best_pair, discard=tuple(discard))
