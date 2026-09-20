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

from core.engine import PACK_COST
from models.player import GameContext, PlayerSnapshot, Selection, TurnContext
from models.player import Player as BasePlayer

THRESHOLD = 6


class Player10(BasePlayer):
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

	def aging(self, shade: int) -> int:
		# check how much a sock has aged
		if shade >= 127:  # white sock
			return 255 - shade
		else:  # black sock
			return shade

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

		# check for all pairs within threshold of 6 since embarrassment is 0 for anything less than 6
		pairs_within_threshold = [
			p
			for p in combinations(range(len(offered)), 2)
			if abs(offered[p[0]] - offered[p[1]]) <= THRESHOLD
		]
		if pairs_within_threshold:  # if there are pairs that fall within 6
			# take the most extreme pair like closest to 255 since we want it to become more grey and uniform - white socks
			i, j = min(
				pairs_within_threshold,
				key=lambda p: self.aging(offered[p[0]]) + self.aging(offered[p[1]]),
			)
		else:
			# if there are no pairs within threshold, be greedy
			i, j = min(
				combinations(range(len(offered)), 2),
				key=lambda p: abs(offered[p[0]] - offered[p[1]]),
			)

		worn = (offered[i] + offered[j]) / 2

		discard: list[int] = []
		days_remaining = max(self.days - turn.day + 1, 1)

		# check if we have an inf budget, otherwise we add a variable to pace our spending based on days remaining and budget remaining
		if turn.budget_remaining == 'inf':
			buy_pack = True
		else:
			daily_rate = turn.budget_remaining / days_remaining
			buy_pack = daily_rate >= (PACK_COST / 6)

		if turn.budget_remaining >= PACK_COST and buy_pack:
			leftovers = [k for k in range(len(offered)) if k not in (i, j)]
			worst = max(leftovers, key=lambda k: abs(offered[k] - worn))
			if abs(offered[worst] - worn) > THRESHOLD:
				discard.append(worst)

		return Selection(wear=(i, j), discard=(tuple(discard)))
