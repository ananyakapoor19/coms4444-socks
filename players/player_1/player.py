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

import math
import random
from collections import deque

import numpy as np

from models.player import GameContext, PlayerSnapshot, Selection, TurnContext
from models.player import Player as BasePlayer

# Wears a sock lasts: black stops rising at 64, white stops fading at 127 (64 wears of 2).
LIFETIME = 64
PACK_COST = 10.0
PACK_SIZE = 6
# Lowest useful threshold: a pair differing by 6 or less is free, so don't discard newer.
FLOOR = 6.0
# Above the 64-wear maximum, so nothing qualifies.
NEVER = 65.0
# Most extra embarrassment catch-up pairing pays to wear the freshest pair.
CATCHUP_COST = 10.0


def pair_cost(a: int, b: int) -> float:
	# Cost of a pair of socks
	diff = abs(a - b)
	return float(diff) if diff > 6 else 0.0


class Player1(BasePlayer):
	MIN_THRESHOLD = 6
	MAX_THRESHOLD = 65
	# use the average of last N calculated threshold as current "smoother" threshold
	# min is 1,
	THRESHOLD_AVG_N = 50
	SEED = 4444

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
		self.threshold_history = deque()
		self.count_discarded_over_threshold = 0
		self.sum_discarded_over_threshold = 0
		self.discard_probability = 0.4

		self.set_base_discard_probability_method = self.set_base_discard_probability_sigmoid
		self.dynamic_discard_probability_method = self.calculate_dynamic_discard_probability_linear

		# Safe random generator, and reseeds so 2 of our players don't share a random factor.
		self.rng = random.Random(self.SEED + self.index)

		self.black_avg = 0
		self.black_range = 0
		self.white_avg = 255
		self.white_range = 0

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
		if self.days_seen == 0:
			self.total_budget = turn.budget_remaining
			self.set_base_discard_probability_method()

			self.previous_budget = self.total_budget
		self.days_seen += 1

		# num_bought = (self.previous_budget - turn.budget_remaining) / 10
		# Can somehow use num_bought to widen range
		self.previous_budget = turn.budget_remaining

		black_socks = np.array(offered)[np.where(np.array(offered) <= 64)].astype(np.float64)
		self.black_avg, self.black_range = self.estimate_age(
			black_socks, self.black_avg, self.black_range, 1
		)
		white_socks = np.array(offered)[np.where(np.array(offered) >= 127)].astype(np.float64)
		self.white_avg, self.white_range = self.estimate_age(
			white_socks, self.white_avg, self.white_range, -2
		)

		# Today's hand is a random sample of the drawer, so its average wears left,
		# scaled up to the drawer, is a rough reading of the whole drawer's runway.
		wears_left_per_sock = sum(LIFETIME - self._wears(s) for s in offered) / len(offered)
		self.runway = wears_left_per_sock * self.capacity

		# if offered = [0, 1, 255, 253]
		# then wear_scores = [0, 1, 0, 1]
		# by_shade = [(0, 0), (1, 1), (253, 3), (255, 2)]
		# and selected_pair = (0, 1)
		# because the first two socks are closest in shade and have the lowest wear scores
		# although this also means black socks are preferred over white socks due to less color difference despite the same wear scores

		by_shade = sorted((sock, i) for i, sock in enumerate(offered))
		wear_scores = [self._wears(sock) for sock in offered]

		if self.is_well_clustered(turn):
			return self.well_clustered_selection(by_shade, wear_scores, turn)

		selected_pair = self.select_pair(by_shade, wear_scores)

		# modulized discard logic
		# discard is probablistic
		# but no discard happens after budget runs out
		discard_method = self.choose_discard_probablistic_dynamic
		discard = discard_method(offered, turn, selected_pair)
		return Selection(wear=selected_pair, discard=tuple(discard))

	@staticmethod
	def _wears(shade: int) -> float:
		"""Return the number of wears a sock has seen, as a float."""
		return (255 - shade) / 2 if shade > 64 else float(shade)

	def is_well_clustered(self, turn: TurnContext) -> bool:
		# Added is_rich check so we don't wait for another team to spend when budget is high.
		if turn.budget_remaining == float('inf') or self.is_rich():
			return False
		return self.total_budget == turn.budget_remaining

	def is_rich(self) -> bool:
		# checks if budget per roommate per day is > 0.35
		# Or if budget per roommate per day is > 0.12 and the drawer would last half the sim with no buys.
		budget = (self.total_budget // PACK_COST) * PACK_COST
		per_roommate_day = budget / (self.roommates * self.days)
		# How long drawer would last with no spending
		drawer_life = 32.0 * self.capacity / (self.roommates * self.days)
		return per_roommate_day >= 0.35 or (drawer_life >= 0.5 and per_roommate_day >= 0.12)

	def estimate_age(self, colored_socks, previous_age, previous_range, multiplier):
		# Estimate roommates (-1 because we did not pick color) * 2 (pick 2) / 2 (assume half pick each color) / half-capacity (population of each color)
		picked_by_roommates = (self.roommates - 1) * 2 / 2
		half_capacity = self.capacity / 2
		roommate_aging = multiplier * picked_by_roommates / half_capacity
		if colored_socks.size > 0:
			mean = colored_socks.mean()
			range = colored_socks.max() - colored_socks.min()
			# For range, if larger range observed, then set. If not, then average ranges to try to decay towards observations
			if range > previous_range:
				range_update = range
			else:
				range_update = (
					previous_range * (half_capacity - colored_socks.size) / half_capacity
					+ range * colored_socks.size / half_capacity
				)

			observed_age = mean * colored_socks.size / half_capacity
			previous_observed_age = (
				previous_age * (half_capacity - colored_socks.size) / half_capacity
			)
			# Take weighted average between observed and previous observed aged and add rommmates choice
			return previous_observed_age + observed_age + roommate_aging, range_update
		else:
			return previous_age + roommate_aging, previous_range

	def select_pair(
		self, by_shade: list[tuple[int, int]], wear_scores: list[float]
	) -> tuple[int, int]:
		pair = (by_shade[0][1], by_shade[1][1])
		best_diff = by_shade[1][0] - by_shade[0][0]
		pair_wear_score = wear_scores[pair[0]] + wear_scores[pair[1]]

		for (left, left_i), (right, right_i) in zip(by_shade, by_shade[1:], strict=False):
			diff = right - left
			current_pair_wear_score = wear_scores[left_i] + wear_scores[right_i]

			# wear score as tiebreaker
			# if there's no ties choose least embarassment
			# if there's a tie choose least wear score (newest socks)
			if best_diff <= 6 and diff <= 6 or best_diff == diff:
				if current_pair_wear_score < pair_wear_score:
					best_diff = diff
					pair = (left_i, right_i)
					pair_wear_score = current_pair_wear_score
			elif diff < best_diff:
				best_diff = diff
				pair = (left_i, right_i)
				pair_wear_score = current_pair_wear_score

		return pair

	def well_clustered_selection(
		self, by_shade: list[tuple[int, int]], wear_scores: list[float], turn: TurnContext
	) -> Selection:
		"""
		Returns the freshest free pair,
		then a fresh free pair within 10 of each other
		then a non fresh free pair
		last returns the option with least pair_cost.
		"""
		# create shade that holds sock shade at a given index.
		shade = [0] * len(by_shade)
		for s, i in by_shade:
			shade[i] = s
		idx = range(len(shade))

		# Loops through twice, checking for black socks, then white socks.
		fresh = []
		for white in (False, True):
			# creates an array of a single color of socks, sorted by wear scores
			color = sorted(
				(i for i in idx if (shade[i] > 64) == white), key=lambda i: (wear_scores[i], i)
			)
			#  If there is a pair, add the cost of the freshest pair to fresh
			if len(color) >= 2:
				pair = (color[0], color[1])
				fresh.append(
					(pair_cost(shade[pair[0]], shade[pair[1]]), wear_scores[pair[0]], pair)
				)
		# check if any of the pairs in fresh are free
		free = [f for f in fresh if f[0] == 0]
		if free:
			# return the freshest pair in free array.
			return Selection(wear=min(free, key=lambda f: f[1])[2])

		# create an array of pair indexes.
		pairs = [(i, j) for i in idx for j in idx if i < j]

		# Check all pairing of socks free options. Select the free pair with fewest wears.
		newest_free = min(
			(p for p in pairs if pair_cost(shade[p[0]], shade[p[1]]) == 0),
			key=lambda p: (wear_scores[p[0]] + wear_scores[p[1]], p),
			default=None,
		)

		# If nothing is free, return the lowest cost pair, wear score breaks ties.
		if newest_free is None:
			cheapest = min(
				pairs,
				key=lambda p: (
					pair_cost(shade[p[0]], shade[p[1]]),
					wear_scores[p[0]] + wear_scores[p[1]],
					p,
				),
			)
			return Selection(wear=cheapest)
		# catch is the lowest pair_cost of the fresh socks, and lower wear as a tiebreak.
		catch = min(fresh, key=lambda f: (f[0], f[1]))
		# Returns the freshest pair, if its cost is less than 10 (hard coded catchup_cost).
		# Else returns the free pair.
		return Selection(wear=catch[2] if catch[0] <= CATCHUP_COST else newest_free)

	# calculate a threshold for sock discard policy
	# return in number of days, the threshold over which socks need to be discarded
	# uses the last N days' threshold for smoothing
	def calculate_discard_threshold(self, turn: TurnContext) -> float:
		# raw current threshold using remaining day and budget
		days_left = float(self.days - turn.day)
		raw_threshold = (
			(10.0 * self.roommates * days_left + 32 * self.roommates) / (3 * turn.budget_remaining)
			if turn.budget_remaining > 0
			else self.MAX_THRESHOLD
		)
		runway_threshold = self.choose_discard_threshold_runway(turn)

		raw_threshold = min(
			raw_threshold if self.rng.random() < 0.5 else runway_threshold, self.MAX_THRESHOLD
		)

		# smoother threshold
		if len(self.threshold_history) >= self.THRESHOLD_AVG_N:
			self.threshold_history.popleft()
		self.threshold_history.append(raw_threshold)
		smooth_threshold = np.mean(np.array(self.threshold_history))

		return max(smooth_threshold, self.MIN_THRESHOLD)

	# discard sock over thresohld
	# with a probability
	@DeprecationWarning
	def _choose_discard_probablistic(
		self, offered: tuple[int, ...], turn: TurnContext, selected_pair: tuple[int, ...]
	) -> tuple[int, ...]:
		if self.is_well_clustered(turn):
			return tuple([])

		threshold = self.calculate_discard_threshold(turn)
		discard = []
		for c in range(len(offered)):
			if (
				c not in selected_pair
				and offered[c] >= threshold
				and offered[c] <= (255 - threshold * 2)
				and self.rng.random() < self.discard_probability
			):
				discard.append(c)
		return tuple(discard)

	# calculate a base discard probability based on budget / cost ratio
	# base budget calculated using 10 / 6 / 64 * 2
	# base probability found by experiments
	# linear version
	def set_base_discard_probability_linear(
		self,
		base_proability: float = 0.4,
		base_budget_per_day: float = 0.052,
		min_probability: float = 0.25,
		max_probability: float = 1.0,
		k: float = 7,  # sensitivity
	):
		daily_budget = self.total_budget / (self.roommates * self.days)
		delta_budget = daily_budget - base_budget_per_day
		raw_prob = base_proability + k * delta_budget

		# Clamp between min_probability and max_probability
		self.discard_probability = max(min_probability, min(max_probability, raw_prob))

	# calculate a base discard probability based on budget / cost ratio
	# sigmoid version
	def set_base_discard_probability_sigmoid(
		self,
		base_budget_per_day: float = 0.052,
		base_proability: float = 0.4,
		min_probability: float = 0.25,
		max_probability: float = 1.0,
		k: float = 5,
	):
		daily_budget = self.total_budget / (self.roommates * self.days)
		if not (min_probability < base_proability < max_probability):
			raise ValueError(
				'base_proability must strictly lie between min_probability and max_probability.'
			)

		# Calculate shift so that at daily_budget == base_budget_per_day, prob == base_proability
		shift = math.log((max_probability - base_proability) / (base_proability - min_probability))

		delta_budget = daily_budget - base_budget_per_day

		# Sigmoid formula bounded between min_probability and max_probability
		prob = min_probability + (max_probability - min_probability) / (
			1.0 + math.exp(-k * delta_budget + shift)
		)
		self.discard_probability = prob

	# discard policy based on how old the sock is
	# the older the sock is (over threshold), the larger the probability of getting discarded
	# using the base probability set on day 1
	# and then calculate probability based on how old the sock is
	def choose_discard_probablistic_dynamic(
		self, offered: tuple[int, ...], turn: TurnContext, selected_pair: tuple[int, ...]
	) -> tuple[int, ...]:
		if self.is_well_clustered(turn) or turn.budget_remaining < PACK_COST:
			return tuple([])

		discard = []
		threshold = self.calculate_discard_threshold(turn)

		for c in range(len(offered)):
			if c not in selected_pair:
				days_worn = self.get_days_worn(offered[c])

				discard_probability = self.dynamic_discard_probability_method(days_worn, threshold)
				if self.rng.random() < discard_probability:
					discard.append(c)
		return tuple(discard)

	def calculate_dynamic_discard_probability_linear(
		self, days_worn: float, threshold: float, N: int = 3
	) -> float:

		# Linear equation centered at threshold
		p = self.discard_probability + (days_worn - threshold) / (2 * N)

		# Clamp value between 0.0 and 1.0
		return max(0.0, min(1.0, p))

	def calculate_dynamic_discard_probability_sigmoid(
		self, days_worn: int, threshold: float, k: float = 0.5
	) -> float:

		offset = math.log(self.discard_probability / (1 - self.discard_probability))
		exponent = -k * (days_worn - threshold) - offset
		probability = 1 / (1 + math.exp(exponent))

		return probability

	@staticmethod
	def get_days_worn(shade: int) -> int:
		if shade < 65:
			return shade
		else:
			return int((256 - shade) / 2)

	# this method doesn't seem to provide definite benefit
	# @DeprecationWarning
	# def _choose_discard_probablistic_improved(
	# 		self,
	# 		offered: tuple[int, ...],
	# 		turn: TurnContext,
	# 		selected_pair: tuple[int, ...]) -> tuple[int, ...]:
	# 	self.DISCARD_PROBABILITY = 0.4

	# 	if self.is_well_clustered(turn):
	# 		return tuple([])

	# 	threshold = self.calculate_discard_threshold(turn)
	# 	discard = []
	# 	for c in range(len(offered)):
	# 		if c not in selected_pair:
	# 			offset_from_threshold = self.get_days_worn(offered[c]) - threshold
	# 			# old socks has a chance of being discarded
	# 			if offered[c] >= threshold \
	# 				and offered[c] <= (255 - threshold * 2):
	# 					if random.random() < self.DISCARD_PROBABILITY:
	# 						discard.append(c)
	# 						self.sum_discarded_over_threshold += offset_from_threshold # this is positive
	# 						self.count_discarded_over_threshold += 1
	# 			# if discarding socks not reaching threshold restores balance, also discard with a chance
	# 			else:
	# 				# this should be negative since this is an under-threshold sock
	# 				average_over_threshold = self.sum_discarded_over_threshold / self.count_discarded_over_threshold \
	# 					if self.count_discarded_over_threshold > 0 \
	# 					else 0
	# 				if abs(offset_from_threshold) <= average_over_threshold:
	# 					if random.random() < self.DISCARD_PROBABILITY:
	# 						discard.append(c)
	# 						self.sum_discarded_over_threshold += offset_from_threshold # this is negative
	# 						self.count_discarded_over_threshold -= 1
	# 	return tuple(discard)

	def choose_discard_threshold_runway(self, turn: TurnContext) -> float:
		if turn.budget_remaining == float('inf'):
			return FLOOR
		buyable = (turn.budget_remaining // PACK_COST) * PACK_SIZE
		need = 2 * self.roommates * (self.days - turn.day)
		need += 0.25 * need  # account for 25% chance of holes in worn socks
		surplus = self.runway + LIFETIME * buyable - need
		threshold = LIFETIME - surplus / (self.capacity + buyable)

		if buyable == 0:
			return NEVER
		return min(NEVER, max(FLOOR, threshold))
