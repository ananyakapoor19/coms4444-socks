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

from models.player import GameContext, PlayerSnapshot, Selection, TurnContext
from models.player import Player as BasePlayer


class Player1(BasePlayer):
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
		self.days_seen += 1

		if self.is_well_clustered(turn):
			return self.well_clustered_selection(offered, turn)

		free = [
			(a, b)
			for a, b in combinations(range(len(offered)), 2)
			if abs(offered[a] - offered[b]) <= 6
		]

		if free:
			pair = min(free, key=lambda p: self._wears(offered[p[0]]) + self._wears(offered[p[1]]))
		else:
			by_shade = sorted((sock, i) for i, sock in enumerate(offered))
			pair = (by_shade[0][1], by_shade[1][1])
			best_diff = by_shade[1][0] - by_shade[0][0]
			for (left, left_i), (right, right_i) in zip(by_shade, by_shade[1:], strict=False):
				diff = right - left
				if diff < best_diff:
					best_diff = diff
					pair = (left_i, right_i)

		threshold = self.choose_discard_threshold(turn)
		discard = []
		for c in range(len(offered)):
			if c not in pair and offered[c] >= threshold and offered[c] <= (255 - threshold * 2):
				discard.append(c)
		return Selection(wear=pair, discard=tuple(discard))

	@staticmethod
	def _wears(shade: int) -> float:
		"""Return the number of wears a sock has seen, as a float."""
		return (255 - shade) / 2 if shade > 64 else float(shade)

	def is_well_clustered(self, turn: TurnContext) -> bool:
		return self.total_budget == turn.budget_remaining

	def well_clustered_selection(self, offered: tuple[int, ...], turn: TurnContext) -> Selection:
		by_shade = sorted((sock, i) for i, sock in enumerate(offered))
		(darkest, darkest_i), (dark_next, dark_next_i) = by_shade[0], by_shade[1]
		(light_next, light_next_i), (lightest, lightest_i) = by_shade[-2], by_shade[-1]

		dark_pair = (darkest_i, dark_next_i)
		light_pair = (light_next_i, lightest_i)
		dark_diff = dark_next - darkest
		light_diff = lightest - light_next
		dark_free = dark_diff <= 6
		light_free = light_diff <= 6

		if dark_free and light_free:
			pair = dark_pair if self._wears(darkest) <= self._wears(lightest) else light_pair
		elif dark_free:
			pair = dark_pair
		elif light_free:
			pair = light_pair
		else:
			pair = dark_pair if dark_diff <= light_diff else light_pair

		return Selection(wear=pair)

	def choose_discard_threshold(self, turn: TurnContext) -> float:
		days_left = float(self.days - turn.day)

		threshold = (
			5.0 * self.roommates * days_left / turn.budget_remaining
			if turn.budget_remaining > 0
			else 65
		)
		return threshold if threshold > 6 else 6
