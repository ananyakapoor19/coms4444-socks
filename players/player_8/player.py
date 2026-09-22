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

from dataclasses import dataclass
from itertools import combinations
from math import isclose, pi, sin

from models.player import GameContext, PlayerSnapshot, Selection, TurnContext
from models.player import Player as BasePlayer


@dataclass(frozen=True)
class SockObservation:
	"""The day and socks offered, in their original order."""

	day: int
	offered: tuple[int, ...]

	@property
	def black_shades(self) -> tuple[int, ...]:
		shades = []
		for shade in self.offered:
			if shade < 65:
				shades.append(shade)
		return tuple(shades)

	@property
	def white_shades(self) -> tuple[int, ...]:
		shades = []
		for shade in self.offered:
			if shade >= 127:  # White socks stop fading at 127.
				shades.append(shade)
		return tuple(shades)


class SockHistory:
	"""A separate history for each player."""

	def __init__(self) -> None:
		self._records: list[SockObservation] = []

	def record(self, *, day: int, offered: tuple[int, ...]) -> None:
		observation = SockObservation(day=day, offered=tuple(offered))
		self._records.append(observation)

	@property
	def records(self) -> tuple[SockObservation, ...]:
		# Return a tuple so callers cannot change the stored list.
		return tuple(self._records)

	def recent_means(self, window: int, fallback: tuple[int, ...] = ()) -> tuple[float, float]:
		"""Average observed shades by colour over the last ``window`` rounds.

		Use the current offer only for colours missing from those rounds.
		"""
		if window < 1:
			raise ValueError('history window must be positive')
		recent = self._records[-window:]
		black = [shade for record in recent for shade in record.black_shades]
		white = [shade for record in recent for shade in record.white_shades]
		if not black:
			black = [shade for shade in fallback if shade <= 64]
		if not white:
			white = [shade for shade in fallback if shade >= 127]
		return (
			sum(black) / len(black) if black else 0.0,
			sum(white) / len(white) if white else 255.0,
		)


class Player8(BasePlayer):
	# 丢袜策略调参区
	# 预算比例均相对于实际总预算，0.05 表示 5 个百分点。
	# 参数依次为：历史回合数（不含当前回合）、预算下界、预算上界、中点偏移量、丢弃积极度。
	# 低于下界不丢，高于上界丢两只；中点正偏移右移、更保守，负偏移更积极。
	# 中心 = (下界 + 上界) / 2 + 偏移；必须严格位于上下界之间。
	# 中心处阈值等于历史平均，不保证丢一只。
	# 积极度必须为有限正数：1 为线性，大于 1 更积极，小于 1 更保守。
	# 积极度只改变区间内的曲线形状，不改变上下界和中点的阈值。

	# Tunable discard policy parameters
	# All budget ratios use the actual total budget; 0.05 means 5 percentage points.
	history_window = 20  # Previous rounds to average, excluding the current round.
	budget_lower_ratio = 0.00  # Discard none below this lower bound.
	budget_upper_ratio = 0.20  # Discard two above this upper bound.

	# A positive offset shifts the center right, making discards more conservative;
	# a negative offset makes them more aggressive.
	# Center = (lower + upper) / 2 + offset; it must stay strictly inside the bounds.
	# At the center, thresholds equal historical means, not a guaranteed discard.
	budget_center_offset = 0.00

	# Positive finite value: 1 keeps linear interpolation; >1 is more aggressive,
	# <1 is more conservative. Endpoints and the shifted center stay fixed.
	discard_aggressiveness = 1.0

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
		self.history = SockHistory()
		self.history_new = []

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
		center_ratio = (
			self.budget_lower_ratio + self.budget_upper_ratio
		) / 2 + self.budget_center_offset
		if not self.budget_lower_ratio < center_ratio < self.budget_upper_ratio:
			raise ValueError('budget center must be strictly between the lower and upper bounds')
		if not 0 < self.discard_aggressiveness < float('inf'):
			raise ValueError('discard aggressiveness must be positive and finite')

		self.days_seen += 1
		black_mean, white_mean = self.history.recent_means(self.history_window, offered)
		self.history.record(day=turn.day, offered=offered)

		# Calculate the expected budget (today's estimated remaining budget)
		total_budget: float = turn.budget_remaining + turn.total_spent
		exp_budget: float = self.get_expected_budget(total_budget)

		# Edge cases
		# Handle when a pair of socks cannot be made
		n = len(offered)
		if n == 0:
			return Selection(wear=(), discard=())
		if n == 1:
			return Selection(wear=(0,), discard=())

		# Get sock information (index, color, age)
		socks = self.get_offered_sock_info(offered)
		self.history_new.append({'day': turn.day, 'socks': socks})

		# Get embarrassment score of all sock pairs
		sock_pairs = self.calc_sock_pairs(offered, socks)

		# Find all pairs with minimum embarrassment
		minimum = min(pair['embarrassment'] for pair in sock_pairs)
		best_pairs = [pair for pair in sock_pairs if pair['embarrassment'] == minimum]

		# Select based on embarrassment > fewest terminal socks > combined sock age
		chosen_pair = min(best_pairs, key=lambda pair: (pair['terminal_count'], sum(pair['ages'])))
		best_pair = chosen_pair['indices']

		# Create an array of the remaining socks for discard method
		worn = set(best_pair)
		unworn = [i for i in range(n) if i not in worn]

		if turn.budget_remaining <= 0 or not unworn:
			return Selection(wear=best_pair, discard=())

		if turn.budget_remaining == float('inf'):
			# An unlimited budget always permits the high-budget discard count.
			discard_count = 2
		elif exp_budget >= total_budget:
			discard_count = 0
		else:
			# B = remaining budget, E = expected budget, T = actual total budget.
			# Delta = B - E; L = lower_ratio * T; U = upper_ratio * T.
			# C = ((lower_ratio + upper_ratio) / 2 + center_offset) * T.
			budget_gap = turn.budget_remaining - exp_budget
			lower_gap = self.budget_lower_ratio * total_budget
			upper_gap = self.budget_upper_ratio * total_budget
			center_gap = center_ratio * total_budget
			# Snap round-off at exact boundaries, e.g. 600.0000000000001.
			for boundary in (lower_gap, center_gap, upper_gap):
				if isclose(budget_gap, boundary, rel_tol=1e-12, abs_tol=1e-9):
					budget_gap = boundary
					break
			if budget_gap <= lower_gap:
				discard_count = 0
			elif budget_gap > upper_gap:
				discard_count = 2
			else:
				# Position -1/0/+1 means lower bound/shifted center/upper bound. Thresholds
				# move from worn-out shades, through historical means, to new shades.
				# mu_b, mu_w = mean black/white shades from the previous history_window rounds.
				# a = discard_aggressiveness > 0; p = raw position; q = curved position.
				# p = (Delta - C) / (C - L) if Delta <= C, otherwise (Delta - C) / (U - C).
				# q = -(-p)^a if p <= 0, otherwise p^(1/a). Here ^ means exponentiation.
				# a = 1 gives q = p; a > 1 increases q and makes discards more aggressive.
				# Both mappings keep p = -1, 0, +1 fixed and are continuous at p = 0.
				if budget_gap <= center_gap:
					position = (budget_gap - center_gap) / (center_gap - lower_gap)
					position = -((-position) ** self.discard_aggressiveness)
					# t_b = mu_b - q * (64 - mu_b); t_w = mu_w + q * (mu_w - 127).
					black_threshold = black_mean - position * (64 - black_mean)
					white_threshold = white_mean + position * (white_mean - 127)
				else:
					position = (budget_gap - center_gap) / (upper_gap - center_gap)
					position = position ** (1 / self.discard_aggressiveness)
					# t_b = mu_b * (1 - q); t_w = mu_w + q * (255 - mu_w).
					black_threshold = black_mean * (1 - position)
					white_threshold = white_mean + position * (255 - white_mean)
				# Eligible: black shade >= t_b, or white shade <= t_w.
				# Discard at most one eligible unworn sock; discard none if no sock qualifies.
				unworn = [
					i
					for i in unworn
					if (
						offered[i] >= black_threshold
						if offered[i] <= 64
						else offered[i] <= white_threshold
					)
				]
				if unworn:
					worst = max(
						unworn,
						key=lambda i: min(abs(offered[i] - offered[j]) for j in range(n) if j != i),
					)
					unworn = [worst]
				discard_count = 1

		# Prefer unworn socks closest to the requested black/white shade targets.
		discard = sorted(
			unworn,
			key=lambda i: abs(offered[i] - (64 if offered[i] <= 64 else 128)),
		)[:discard_count]

		return Selection(wear=best_pair, discard=tuple(discard))

	def get_expected_budget_simplified(self, total_budget: float) -> float:
		total_days: int = self.days
		current_day: int = self.days_seen

		# We consider three cases based on the `day_ratio`: [0, 0.333], (0.333, 0.667), [0.667, 1]
		day_ratio: float = current_day / total_days
		if day_ratio <= 0.333:
			# At the beginning, we don't want to use any budget
			return total_budget
		elif day_ratio >= 0.667:
			# At the final stage, we do not use any budget either
			return 0.0
		else:
			# We consider to spend the budget evenly
			return (2 - 3 * day_ratio) * total_budget

	def get_expected_budget(
		self, total_budget: float, k1: float = 0.666, k2: float = 0.95
	) -> float:
		assert 0 <= k1 < k2 <= 1

		total_days: int = self.days
		current_day: int = self.days_seen

		# We consider three cases based on the `day_ratio`: [0, k1], (k1, k2), [k2, 1]
		day_ratio: float = current_day / total_days
		if day_ratio <= k1:
			# At the beginning, we don't want to use any budget
			return total_budget
		elif day_ratio >= k2:
			# At the final stage, we do not use any budget either
			return 0.0
		else:
			# We consider to spend the budget evenly
			return (k2 - day_ratio) * total_budget / (k2 - k1)

	def get_expected_budget_smoothened(self, total_budget: float) -> float:
		total_days: int = self.days
		current_day: int = self.days_seen

		# We use `sin` to smoothen the expected budget
		return 0.5 * total_budget * (1 + sin(pi / total_days * current_day + 0.5 * pi))

	def get_offered_sock_info(self, offered: tuple[int, ...]) -> list:
		socks = []
		for i, shade in enumerate(offered):
			if shade <= 64:
				socks.append({'index': i, 'color': 'black', 'age': shade})
			else:
				socks.append({'index': i, 'color': 'white', 'age': (255 - shade) // 2})
		return socks

	def calc_sock_pairs(self, offered: tuple[int, ...], socks: list) -> list:
		sock_pairs = []
		for a, b in combinations(socks, 2):
			difference = abs(offered[a['index']] - offered[b['index']])
			embarrassment = difference if difference > 6 else 0

			sock_pairs.append(
				{
					'indices': (a['index'], b['index']),
					'embarrassment': embarrassment,
					'ages': (a['age'], b['age']),
					'terminal_count': int(a['age'] == 64) + int(b['age'] == 64),
				}
			)
		return sock_pairs
