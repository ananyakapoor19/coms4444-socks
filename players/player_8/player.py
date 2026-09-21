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
from math import pi, sin

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


class Player8(BasePlayer):
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
		self.history = SockHistory()

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
		self.history.record(day=turn.day, offered=offered)

		# Calculate the expected budget (today's estimated remaining budget)
		total_budget: float = turn.budget_remaining + turn.total_spent
		exp_budget_simplified: float = self.get_expected_budget_simplified(total_budget)
		exp_budget: float = self.get_expected_budget(total_budget)
		exp_budget_smoothened: float = self.get_expected_budget_smoothened(total_budget)
		print(
			f'budget: ({exp_budget_simplified:.3f}, {exp_budget:.3f}, {exp_budget_smoothened:.3f})'
		)

		# Edge cases
		# Handle when a pair of socks cannot be made
		n = len(offered)
		if n == 0:
			return Selection(wear=(), discard=())
		if n == 1:
			return Selection(wear=(0,), discard=())

		# Finds the index pair of socks with lowest embarrassment
		best_pair = min(
			combinations(range(n), 2),
			key=lambda pair: abs(offered[pair[0]] - offered[pair[1]]),
		)

		# Create an array of the remaining socks for discard method
		worn = set(best_pair)
		unworn = [i for i in range(n) if i not in worn]

		if turn.budget_remaining == 0:
			return Selection(wear=best_pair, discard=())

		discard = []
		for i in unworn:
			shade = offered[i]
			if shade <= 64:  # Black sock
				if shade > 58:
					discard.append(i)
			else:  # White sock
				if shade < 133:
					discard.append(i)

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
		self, total_budget: float, k1: float = 0.333, k2: float = 0.667
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
