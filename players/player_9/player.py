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


class Player9(BasePlayer):
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
		self.total_budget = None
		self.lower_bound = 64
		self.upper_bound = 128

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

		# Replace everything below with your strategy. This baseline wears the
		# first two socks it is handed and never discards, which is the
		# do-nothing behaviour a real strategy should beat.

		# Pick the two closest socks
		left, right = min(
			combinations(range(len(offered)), 2), key=lambda p: abs(offered[p[0]] - offered[p[1]])
		)

		dis = []
		can_discard = False
		# Initialize total_budget
		if self.total_budget is None:
			self.total_budget = turn.total_spent + turn.budget_remaining

		# Monitor budget activity for the first 20 days, don't discard anything
		if turn.day > 20 and turn.budget_remaining > 0:
			can_discard = True
			remaining_days = self.days - turn.day + 1
			remaining_average = turn.budget_remaining / remaining_days
			total_average = self.total_budget / self.days

			# If we are underspending, loosen restrictions on discards
			if remaining_average > total_average:
				self.lower_bound = max(0, self.lower_bound - 5)
				self.upper_bound = min(255, self.upper_bound + 10)
			# If we are overspending, tighten restrictions on discards
			elif remaining_average < total_average:
				self.lower_bound = min(255, self.lower_bound + 5)
				self.upper_bound = max(0, self.upper_bound - 10)

		for i in range(len(offered)):
			if i in (left, right):
				pass
			else:
				if can_discard and offered[i] > self.lower_bound and offered[i] < self.upper_bound:
					dis.append(i)

		return Selection(wear=(left, right), discard=(dis))
