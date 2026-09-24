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

# Short-horizon budget forecast. Spending happens in $10 jumps, so keep the
# individual daily observations as well as the smoother long-run rate.
FORECAST_MIN_DAYS = 3
FORECAST_MAX_DAYS = 14
SPEND_HISTORY_DAYS = 21
SPEND_MOMENTUM_WEIGHT = 0.35
HOLE_PROBABILITY = 0.25
FORECAST_RESERVE_MULTIPLIER = 1.5

# Lifecycle values are measured in expected embarrassment, so only one small
# scale is needed to turn a sock's long-run value into today's action cost.
IMMEDIATE_COST_WEIGHT = 4.0
PAIR_LIFECYCLE_COST_WEIGHT = 0.25
LIFECYCLE_COST_WEIGHT = 0.50
LIFECYCLE_DISCOUNT = 0.96
UNLIMITED_DAILY_SPEND_TARGET = 0.18
LEFTOVER_BURDEN_WEIGHT = 0.10
TERMINAL_KEEP_COST = 2.0
DISCARD_OPTION_COST_WEIGHT = 0.45
DISCARD_INVENTORY_COST = 0.55
MAX_MONEY_SHADOW_PRICE = 5.0

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
		self.spend_history: list[float] = []
		self.observed_terminal_rate = 0.0
		self.forecast_horizon = FORECAST_MIN_DAYS
		self.forecast_budget_pressure = 0.0
		self.projected_age_hist = [[0.0] * AGE_BINS for _ in range(2)]
		self.expected_replacement_rate = 0.0
		self.roommate_spend_rate = 0.0
		self.inventory_cost_scale = 1.0
		self.lifecycle_horizon = 0.0
		self.lifecycle_distributions: list[list[list[float]]] = [[], []]
		self.lifecycle_prefixes: list[list[tuple[list[float], list[float]]]] = [[], []]
		self._lifecycle_cache: dict[tuple[int, int], float] = {}

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

	def _prepare_lifecycle_model(self, turn: TurnContext) -> None:
		"""Build future drawer snapshots at each expected wear of one sock."""
		self._lifecycle_cache.clear()
		wears_per_day = 2.0 * self.roommates / max(1, self.capacity)
		self.lifecycle_horizon = min(64.0, max(0, self.days - turn.day) * wears_per_day)
		steps = max(1, int(self.lifecycle_horizon) + 1)
		wears_household_daily = max(1.0, 2.0 * self.roommates)
		turnover = min(0.45, self.expected_replacement_rate / wears_household_daily)
		all_snapshots: list[list[list[float]]] = []
		all_prefixes: list[list[tuple[list[float], list[float]]]] = []
		for colour in range(2):
			current = [
				FAST_HIST_WEIGHT * fast + SLOW_HIST_WEIGHT * slow
				for fast, slow in zip(
					self.fast_age_hist[colour], self.slow_age_hist[colour], strict=True
				)
			]
			total = sum(current)
			if total > 0.0:
				current = [mass / total for mass in current]
			snapshots = []
			prefixes = []
			for _ in range(steps):
				snapshots.append(current)
				mass_prefix = [0.0]
				age_prefix = [0.0]
				for age, mass in enumerate(current):
					mass_prefix.append(mass_prefix[-1] + mass)
					age_prefix.append(age_prefix[-1] + age * mass)
				prefixes.append((mass_prefix, age_prefix))
				aged = [0.0] * AGE_BINS
				for age, mass in enumerate(current):
					aged[min(64, age + 1)] += mass
				current = [mass * (1.0 - turnover) for mass in aged]
				current[0] += turnover
			all_snapshots.append(snapshots)
			all_prefixes.append(prefixes)
		self.lifecycle_distributions = all_snapshots
		self.lifecycle_prefixes = all_prefixes
		self.projected_age_hist = [snapshots[-1] for snapshots in all_snapshots]

	def _expected_mismatch(
		self,
		colour: int,
		candidate_age: int,
		prefixes: tuple[list[float], list[float]],
	) -> float:
		mass, moment = prefixes
		fade = 1 if colour == 0 else 2
		free_radius = THRESHOLD // fade
		left_end = max(0, candidate_age - free_radius)
		right_start = min(AGE_BINS, candidate_age + free_radius + 1)
		left = candidate_age * mass[left_end] - moment[left_end]
		right = (
			moment[AGE_BINS]
			- moment[right_start]
			- candidate_age * (mass[AGE_BINS] - mass[right_start])
		)
		return fade * (left + right)

	def _lifecycle_cost(self, shade: int) -> float:
		"""Average future embarrassment as this sock and the drawer age together."""
		colour = 0 if self._is_black(shade) else 1
		start_age = self._age(shade)
		key = (colour, start_age)
		if key in self._lifecycle_cache:
			return self._lifecycle_cache[key]
		if self.lifecycle_horizon <= 0.0:
			return 0.0
		total_cost = 0.0
		total_weight = 0.0
		for step, prefixes in enumerate(self.lifecycle_prefixes[colour]):
			remaining = self.lifecycle_horizon - step
			if remaining <= 0.0:
				break
			weight = min(1.0, remaining) * LIFECYCLE_DISCOUNT**step
			candidate_age = min(64, start_age + step)
			total_cost += weight * self._expected_mismatch(colour, candidate_age, prefixes)
			total_weight += weight
		value = total_cost / total_weight if total_weight else 0.0
		self._lifecycle_cache[key] = value
		return value

	def _observe_terminal_socks(self, offered: tuple[int, ...]) -> None:
		terminal_share = sum(shade in (64, 127) for shade in offered) / max(1, len(offered))
		alpha = 0.12
		self.observed_terminal_rate = (
			alpha * terminal_share + (1.0 - alpha) * self.observed_terminal_rate
		)

	def _budget_forecast(self, turn: TurnContext) -> tuple[int, float, float]:
		"""Predict near-term pack spending and return horizon, spend, pressure."""
		days_left = max(1, self.days - turn.day + 1)
		horizon = min(days_left, FORECAST_MAX_DAYS)
		if days_left > FORECAST_MIN_DAYS:
			horizon = max(FORECAST_MIN_DAYS, horizon)

		recent = self.spend_history[-7:]
		recent_rate = sum(recent) / len(recent) if recent else 0.0
		momentum_rate = (
			1.0 - SPEND_MOMENTUM_WEIGHT
		) * self.recent_spend_rate + SPEND_MOMENTUM_WEIGHT * recent_rate

		# Terminal socks can disappear without a voluntary discard. Convert their
		# observed frequency into expected pack spending across the household.
		expected_holes_per_day = (
			self.observed_terminal_rate * 2.0 * self.roommates * HOLE_PROBABILITY
		)
		hole_rate = expected_holes_per_day * SOCK_REPLACEMENT_COST
		forecast_rate = max(momentum_rate, hole_rate)
		forecast_spend = forecast_rate * horizon

		if turn.budget_remaining == float('inf'):
			self.inventory_cost_scale = 0.15
			return horizon, forecast_spend, 0.0

		reserve = max(PACK_COST, hole_rate * horizon * FORECAST_RESERVE_MULTIPLIER)
		usable = max(0.0, turn.budget_remaining - reserve)
		pressure = min(1.0, forecast_spend / max(PACK_COST, usable))
		return horizon, forecast_spend, pressure

	def _budget_discard_limit(
		self,
		turn: TurnContext,
		unit: int,
	) -> tuple[float, int]:
		"""Return discard aggression and today's discretionary discard limit."""
		max_leftovers = max(0, unit - 2)
		horizon, forecast_spend, pressure = self._budget_forecast(turn)
		self.forecast_horizon = horizon
		self.forecast_budget_pressure = pressure
		if turn.budget_remaining == float('inf'):
			daily_spend = max(UNLIMITED_DAILY_SPEND_TARGET, self.recent_spend_rate)
			self.expected_replacement_rate = daily_spend * 6.0 / PACK_COST
			self.discard_credit = min(
				CREDIT_CAP,
				self.discard_credit + UNLIMITED_DAILY_SPEND_TARGET / SOCK_REPLACEMENT_COST,
			)
			return 0.05, min(MAX_DISCARDS, max_leftovers, int(self.discard_credit))

		current_budget = turn.total_spent + turn.budget_remaining
		if self.estimated_budget is None or current_budget > self.estimated_budget:
			self.estimated_budget = current_budget
		base_reserve = max(
			RESERVE_PACKS * PACK_COST,
			self.estimated_budget * RESERVE_FRACTION,
		)
		forecast_reserve = forecast_spend * FORECAST_RESERVE_MULTIPLIER
		reserve = max(base_reserve, forecast_reserve)
		days_left = max(1, self.days - turn.day + 1)
		warmup = turn.day <= WARMUP_DAYS
		imputed_own_spend = min(turn.total_spent, self.my_virtual_discard_spend)
		self.roommate_spend_rate = max(
			0.0,
			(turn.total_spent - imputed_own_spend) / max(1, turn.day),
		)
		projected_roommate_spend = self.roommate_spend_rate * days_left
		personal_slack = max(
			0.0,
			turn.budget_remaining - reserve - projected_roommate_spend,
		)
		budget_dollars_per_day = self.estimated_budget / max(1, self.days)
		self.inventory_cost_scale = 1.0 / (1.0 + 3.0 * max(0.0, budget_dollars_per_day - 0.50))
		residual_share = min(
			1.0,
			1.0 / max(1, self.roommates) + 1.5 * max(0.0, budget_dollars_per_day - 0.20),
		)
		allowed_daily_dollars = personal_slack / days_left * residual_share
		self.expected_replacement_rate = (
			(
				max(self.roommate_spend_rate, self.recent_spend_rate)
				+ min(0.03, 0.20 * allowed_daily_dollars)
			)
			* 6.0
			/ PACK_COST
		)
		self.discard_credit = min(
			CREDIT_CAP,
			self.discard_credit + allowed_daily_dollars / SOCK_REPLACEMENT_COST,
		)
		if warmup:
			return 0.0, 0

		allowed = min(MAX_DISCARDS, max_leftovers, int(self.discard_credit))
		time_remaining = days_left / max(1, self.days)
		budget_remaining = turn.budget_remaining / max(1.0, self.estimated_budget)
		pace_shortfall = max(0.0, time_remaining - budget_remaining)
		money_shadow = min(
			MAX_MONEY_SHADOW_PRICE,
			0.35 + 3.0 * pressure + 4.0 * pace_shortfall,
		)
		if self.roommates >= CROWDED_ROOMMATE_CUTOFF and unit >= CROWDED_UNIT:
			if turn.day <= int(self.days * CROWDED_WARMUP_FRACTION):
				return money_shadow, 0
			allowed = min(allowed, CROWDED_MAX_DISCARDS)
		return money_shadow, allowed

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
			spend_delta = max(0.0, turn.total_spent - self.last_spend)
			daily_spend = spend_delta / elapsed
			self.spend_history.extend([0.0] * max(0, elapsed - 1))
			self.spend_history.append(spend_delta)
			if len(self.spend_history) > SPEND_HISTORY_DAYS:
				del self.spend_history[:-SPEND_HISTORY_DAYS]
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

	def _kept_sock_cost(self, shade: int) -> float:
		"""Future drawer burden of returning one leftover unchanged."""
		colour = 0 if self._is_black(shade) else 1
		projected_match = 1.0 - min(
			1.0,
			self._expected_mismatch(
				colour,
				self._age(shade),
				self.lifecycle_prefixes[colour][-1],
			)
			/ 255.0,
		)
		cost = LEFTOVER_BURDEN_WEIGHT * (
			0.85 * self._age_score(shade)
			+ 0.55 * self._outlier_score(shade)
			+ 0.65 * (1.0 - projected_match)
		)
		cost += 0.02 * LIFECYCLE_COST_WEIGHT * self._lifecycle_cost(shade)
		if shade in (64, 127):
			cost += TERMINAL_KEEP_COST
		return cost

	def _discarded_sock_cost(self, shade: int, money_shadow: float) -> float:
		"""Replacement pressure plus fresh-sock lifecycle value."""
		colour = 0 if self._is_black(shade) else 1
		fresh = self._shade_from_age(colour, 0)
		inventory_shadow = min(1.0, max(0.15, money_shadow / 0.35))
		return (
			DISCARD_INVENTORY_COST * inventory_shadow * self.inventory_cost_scale
			+ SOCK_REPLACEMENT_COST * money_shadow
			+ DISCARD_OPTION_COST_WEIGHT * self._population_potential(shade)
			+ 0.02 * LIFECYCLE_COST_WEIGHT * self._lifecycle_cost(fresh)
		)

	def _worn_sock_cost(self, shade: int, money_shadow: float) -> float:
		"""Near-term matching after washing, plus terminal replacement risk."""
		colour = 0 if self._is_black(shade) else 1
		post_wear = self._shade_from_age(colour, min(64, self._age(shade) + 1))
		future_cost = 1.0 - self._population_potential(post_wear)
		future_cost += PAIR_LIFECYCLE_COST_WEIGHT * self._lifecycle_cost(post_wear) / 255.0
		if shade in (64, 127):
			future_cost += HOLE_PROBABILITY * SOCK_REPLACEMENT_COST * money_shadow
		return future_cost

	def _action_cost(
		self,
		offered: tuple[int, ...],
		wear: tuple[int, int],
		discard: tuple[int, ...],
		money_shadow: float,
	) -> float:
		"""Price today's embarrassment, future drawer quality, and replacement."""
		a, b = offered[wear[0]], offered[wear[1]]
		cost = IMMEDIATE_COST_WEIGHT * self._embarrassment(a, b)
		cost += self._worn_sock_cost(a, money_shadow)
		cost += self._worn_sock_cost(b, money_shadow)
		discarded = set(discard)
		for index, shade in enumerate(offered):
			if index in wear:
				continue
			if index in discarded:
				cost += self._discarded_sock_cost(shade, money_shadow)
			else:
				cost += self._kept_sock_cost(shade)
		return cost

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
		self._observe_terminal_socks(offered)

		money_shadow, discard_limit = self._budget_discard_limit(turn, unit)
		self._prepare_lifecycle_model(turn)

		best_action = ((0, 1), ())
		best_key = None
		for wear in combinations(range(unit), 2):
			leftovers = tuple(index for index in range(unit) if index not in wear)
			for count in range(min(discard_limit, len(leftovers)) + 1):
				for discard in combinations(leftovers, count):
					cost = self._action_cost(offered, wear, discard, money_shadow)
					key = (
						cost,
						self._embarrassment(offered[wear[0]], offered[wear[1]]),
						len(discard),
						self._age(offered[wear[0]]) + self._age(offered[wear[1]]),
						wear,
						discard,
					)
					if best_key is None or key < best_key:
						best_key = key
						best_action = (wear, discard)

		best_pair, discard = best_action
		if discard:
			self.discard_credit = max(0.0, self.discard_credit - len(discard))
			self.my_virtual_discard_spend += len(discard) * SOCK_REPLACEMENT_COST

		return Selection(wear=best_pair, discard=discard)
