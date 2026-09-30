from dataclasses import dataclass
from itertools import combinations
from math import ceil

from core.engine import HOLE_PROBABILITY, PACK_COST, PACK_SIZE
from models.player import GameContext, PlayerSnapshot, Selection, TurnContext
from models.player import Player as BasePlayer

SOCK_PRICE = PACK_COST / PACK_SIZE
MAX_AGE = 64
FREE_SHADE_GAP = 6


def mismatch_curve(weights: list[float], shade_step: int) -> list[float]:
	"""Expected embarrassment at every age; weights may be unnormalized.

	Two cumulative totals let us sum only partners outside the free shade gap.
	A white sock's age gap is worth twice a black sock's age gap.
	"""
	prefix_mass = [0.0]
	prefix_age = [0.0]
	for age in range(MAX_AGE + 1):
		prefix_mass.append(prefix_mass[-1] + weights[age])
		prefix_age.append(prefix_age[-1] + weights[age] * age)
	free_radius = FREE_SHADE_GAP // shade_step
	curve = []
	for age in range(MAX_AGE + 1):
		lower = max(0, age - free_radius)
		upper = min(MAX_AGE + 1, age + free_radius + 1)
		younger = age * prefix_mass[lower] - prefix_age[lower]
		older = prefix_age[-1] - prefix_age[upper] - age * (prefix_mass[-1] - prefix_mass[upper])
		curve.append(max(0.0, shade_step * (younger + older)))
	return curve


def future_mismatch(
	age_weights: list[float],
	*,
	shade_step: int,
	replacement_share: float,
	survival: float,
	wear_horizon: float,
	survival_cap: float = 0.98,
) -> list[float]:
	"""Discount future mismatch for each current age over the remaining game.

	For each future wear, compare the candidate with equally aged observations
	and with the estimated continuing supply of replacements. Discount intervals
	by survival; retain the fraction of the last expected wear interval.
	"""
	forecast = [0.0] * (MAX_AGE + 1)
	if wear_horizon <= 0:
		return forecast
	mass = sum(age_weights)
	observed = [weight / mass for weight in age_weights] if mass else [0.0] * (MAX_AGE + 1)
	arrivals = [(1.0 - survival) * survival**age for age in range(MAX_AGE + 1)]
	arrivals[MAX_AGE] = survival**MAX_AGE
	arrival_cost = mismatch_curve(arrivals, shade_step)
	discount = min(survival, survival_cap)
	for wear in range(min(MAX_AGE, ceil(wear_horizon))):
		interval = discount**wear - discount ** min(wear + 1.0, wear_horizon)
		observed_cost = mismatch_curve(observed, shade_step)
		for age in range(MAX_AGE + 1):
			aged = min(MAX_AGE, age + wear)
			expected = (1.0 - replacement_share) * observed_cost[aged]
			expected += replacement_share * arrival_cost[aged]
			forecast[age] += interval * expected
		# All observed mass at or approaching the wear limit stays at the cap.
		observed = [0.0, *observed[: MAX_AGE - 1], observed[MAX_AGE - 1] + observed[MAX_AGE]]
	if wear_horizon > MAX_AGE:
		remaining = replacement_share * (discount**MAX_AGE - discount**wear_horizon)
		tail_cost = remaining * arrival_cost[MAX_AGE]
		forecast = [value + tail_cost for value in forecast]
	return forecast


@dataclass(frozen=True)
class SockObservation:
	"""The day and socks offered, in their original order."""

	day: int
	offered: tuple[int, ...]


class SockHistory:
	"""A separate history for each player."""

	def __init__(self, decay: float) -> None:
		self._records: list[SockObservation] = []
		self.decay = decay
		self._last_day = 0
		self._age_weights = {color: [0.0] * (MAX_AGE + 1) for color in ('black', 'white')}

	def record(self, *, day: int, offered: tuple[int, ...], socks: list) -> None:
		# Update the belief incrementally, so keeping the complete history does
		# not make each turn rescan every previous offer.
		decay = self.decay ** max(1, day - self._last_day)
		for weights in self._age_weights.values():
			for age, weight in enumerate(weights):
				aged_weight = weight * decay
				weights[age] = aged_weight if aged_weight >= 0.001 else 0.0
		for sock in socks:
			self._age_weights[sock['color']][sock['age']] += 1.0
		self._last_day = day
		observation = SockObservation(day=day, offered=tuple(offered))
		self._records.append(observation)

	def age_weights(self, color: str) -> list[float]:
		"""A copy of the recency-weighted observations for one color."""
		return self._age_weights[color].copy()

	@property
	def records(self) -> tuple[SockObservation, ...]:
		# Return a tuple so callers cannot change the stored list.
		return tuple(self._records)

	def recent_shades(self, window: int) -> tuple[int, ...]:
		"""Recent offers are samples of the drawer, not distinct sock identities."""
		if window < 1:
			raise ValueError('history window must be positive')
		return tuple(shade for record in self._records[-window:] for shade in record.offered)


class Player8(BasePlayer):
	"""Select for today; replace leftovers using recent offer history."""

	# The drawer changes as roommates wear and replace socks. A short effective
	# memory follows the current cohorts better than a long historical average.
	history_decay = 0.75
	cohort_spread_threshold = 2.0
	future_mismatch_weight = 0.5
	replacement_money_weight = 2.0
	discard_credit_cap = 10.0
	completion_packs = 4
	budget_warmup_days = 20
	replacement_horizon_days = 30
	hole_replacement_rate_per_roommate = 1 / 34
	unlimited_dollars_per_day = 7490 / 1080

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)
		self.maximum_discards_per_turn = max(0, self.selection_unit - 2)
		self.history = SockHistory(decay=self.history_decay)
		self.discard_credit = 0.0
		self.estimated_discard_spend = 0.0
		self.estimated_household_replacements = 0.0

	def select_socks(self, offered: tuple[int, ...], turn: TurnContext) -> Selection:
		"""Wear today's least embarrassing pair, then consider replacing leftovers."""
		socks = self._get_offered_sock_info(offered)
		self.history.record(day=turn.day, offered=offered, socks=socks)
		n = len(offered)
		if n == 0:
			return Selection(wear=(), discard=())
		if n == 1:
			return Selection(wear=(0,), discard=())

		# min keeps the first pair when every criterion ties.
		best_pair = min(
			self._calc_sock_pairs(offered, socks),
			key=lambda pair: (pair['embarrassment'], pair['terminal_count'], sum(pair['ages'])),
		)['indices']
		worn = set(best_pair)
		unworn = [i for i in range(n) if i not in worn]
		if unworn and self._budget_guarantees_all_discards(turn):
			# Even if both worn socks hole every day, this budget can pay for
			# every possible pack. Discard all leftovers to refresh the pool.
			self.estimated_household_replacements = 2.0 * self.roommates
			costs = self._discard_costs(socks, unworn, 0.0, turn.day)
			chosen = sorted(unworn, key=lambda index: (costs[index], -socks[index]['age'], index))[
				: self.maximum_discards_per_turn
			]
			discard = tuple(sorted(chosen))
			self.estimated_discard_spend += len(discard) * SOCK_PRICE
			return Selection(wear=best_pair, discard=discard)
		money_weight, allowance = self._discard_allowance(turn)
		if not unworn or allowance == 0:
			return Selection(wear=best_pair, discard=())

		costs = self._discard_costs(socks, unworn, money_weight, turn.day)
		beneficial = sorted((cost, index) for index, cost in costs.items() if cost < 0)
		discard = tuple(sorted(index for _, index in beneficial[:allowance]))
		self.discard_credit -= len(discard)
		self.estimated_discard_spend += len(discard) * SOCK_PRICE
		return Selection(wear=best_pair, discard=discard)

	def _budget_guarantees_all_discards(self, turn: TurnContext) -> bool:
		"""Can the starting budget buy every pack the entire game could need?

		Our turn can discard every unworn sock and lose two worn socks
		to holes. Each peer could discard every unworn sock and lose both worn
		socks. Six discarded socks trigger at most one $10 purchase, regardless
		of color. This worst-case bound covers all roommate strategies and days.
		"""
		if self.selection_unit <= 2:
			return False
		if turn.budget_remaining == float('inf'):
			return True
		own_per_day = min(self.maximum_discards_per_turn, self.selection_unit - 2) + 2
		peer_per_day = self.selection_unit * (self.roommates - 1)
		maximum_packs = self.days * (own_per_day + peer_per_day) // PACK_SIZE
		starting_budget = turn.total_spent + turn.budget_remaining
		return starting_budget >= PACK_COST * maximum_packs

	def _discard_allowance(self, turn: TurnContext) -> tuple[float, int]:
		"""Price replacements and bank the budget peers are expected to leave us."""
		allowed = self._earn_discard_credit(turn)
		money_weight = self.replacement_money_weight * max(
			0.0, 1.0 - self.discard_credit / self.discard_credit_cap
		)
		count = min(self.maximum_discards_per_turn, int(self.discard_credit)) if allowed else 0
		return money_weight, count

	def _completion_pack_allowance(self, turn: TurnContext) -> int:
		"""Limit the billing cushion when budget barely covers natural holes.

		A pack of six discarded socks can temporarily disappear when money is
		gone. When the budget per roommate-day is close to the expected cost of
		natural holes, keep the planning cushion within the drawer's spare socks.
		"""
		starting_budget = turn.total_spent + turn.budget_remaining
		dollars_per_roommate_day = starting_budget / (self.roommates * self.days)
		natural_hole_cost = 2 * SOCK_PRICE / (MAX_AGE + 1 / HOLE_PROBABILITY)
		if dollars_per_roommate_day > 1.25 * natural_hole_cost:
			return self.completion_packs
		spare_socks = max(0, self.capacity - self.selection_unit * self.roommates)
		return min(self.completion_packs, spare_socks // PACK_SIZE)

	def _earn_discard_credit(self, turn: TurnContext) -> bool:
		days_left = self.days - turn.day
		if days_left < 1 or turn.budget_remaining < PACK_COST:
			self.estimated_household_replacements = 0.0
			return False

		if turn.budget_remaining == float('inf'):
			budget = self.unlimited_dollars_per_day * self.days
			remaining = budget - turn.total_spent
		else:
			# This is a planning allowance, not real money. Tight budgets keep
			# it within the drawer's capacity margin.
			remaining = turn.budget_remaining + self._completion_pack_allowance(turn) * PACK_COST
			budget = turn.total_spent + remaining

		if turn.day < self.budget_warmup_days:
			# Six-pack billing is too lumpy to estimate peer spending yet.
			allowed = turn.total_spent + PACK_COST <= budget * turn.day / self.days
			self.discard_credit = min(self.discard_credit_cap, self.discard_credit + int(allowed))
			self.estimated_household_replacements = budget / self.days / SOCK_PRICE
			return allowed

		other_dollars_per_day = max(
			0.0, (turn.total_spent - self.estimated_discard_spend) / turn.day
		)
		available = remaining - other_dollars_per_day * days_left
		if available <= 0:
			self.discard_credit = 0.0
			self.estimated_household_replacements = other_dollars_per_day / SOCK_PRICE
			return False
		own_replacements_per_day = available / SOCK_PRICE / days_left
		self.estimated_household_replacements = (
			other_dollars_per_day / SOCK_PRICE + own_replacements_per_day
		)
		self.discard_credit = min(
			self.discard_credit_cap, self.discard_credit + own_replacements_per_day
		)
		return self.discard_credit >= 1.0

	def _discard_costs(
		self, socks: list, unworn: list[int], money_weight: float, day: int
	) -> dict[int, float]:
		"""Replacement price plus its estimated change in future mismatch."""
		days_left = max(0, self.days - day)
		replacements = (
			self.estimated_household_replacements
			+ self.hole_replacement_rate_per_roommate * self.roommates
		)
		replacement_share = min(
			1.0,
			replacements * min(self.replacement_horizon_days, max(1, days_left)) / self.capacity,
		)
		survival = 2 * self.roommates / (2 * self.roommates + replacements)
		wear_horizon = days_left * (2.0 * self.roommates / self.capacity)
		forecasts = {
			color: future_mismatch(
				self.history.age_weights(color),
				shade_step=1 if color == 'black' else 2,
				replacement_share=replacement_share,
				survival=survival,
				wear_horizon=wear_horizon,
			)
			for color in {socks[index]['color'] for index in unworn}
		}
		costs = {}
		for index in unworn:
			forecast = forecasts[socks[index]['color']]
			change = forecast[0] - forecast[socks[index]['age']]
			costs[index] = money_weight * SOCK_PRICE + self.future_mismatch_weight * change

		# When daily offers nearly fill the drawer, a tightly aged cohort is
		# already easy to match. A fresh replacement can split that cohort.
		spare_socks = self.capacity - self.selection_unit * self.roommates
		if spare_socks <= 2 * PACK_SIZE:
			for color in {socks[index]['color'] for index in unworn}:
				weights = self.history.age_weights(color)
				total = sum(weights)
				if not total:
					continue
				mean_age = sum(age * weight for age, weight in enumerate(weights)) / total
				variance = (
					sum(weight * (age - mean_age) ** 2 for age, weight in enumerate(weights))
					/ total
				)
				if variance < self.cohort_spread_threshold**2:
					for index in unworn:
						if socks[index]['color'] == color:
							costs[index] = float('inf')
		return costs

	@staticmethod
	def _get_offered_sock_info(offered: tuple[int, ...]) -> list:
		socks = []
		for i, shade in enumerate(offered):
			if shade <= MAX_AGE:
				socks.append({'index': i, 'color': 'black', 'age': shade})
			else:
				socks.append({'index': i, 'color': 'white', 'age': (255 - shade) // 2})
		return socks

	def _calc_sock_pairs(self, offered: tuple[int, ...], socks: list) -> list:
		sock_pairs = []
		for a, b in combinations(socks, 2):
			difference = abs(offered[a['index']] - offered[b['index']])
			embarrassment = difference if difference > FREE_SHADE_GAP else 0

			sock_pairs.append(
				{
					'indices': (a['index'], b['index']),
					'embarrassment': embarrassment,
					'ages': (a['age'], b['age']),
					'terminal_count': int(a['age'] == MAX_AGE) + int(b['age'] == MAX_AGE),
				}
			)
		return sock_pairs
