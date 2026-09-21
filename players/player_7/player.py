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

from models.player import GameContext, PlayerSnapshot, Selection, TurnContext
from models.player import Player as BasePlayer


class Player7(BasePlayer):
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
		"""Choose the closest pair to wear and discard fully worn out leftovers."""
		self.days_seen += 1

		# Find the closest pair
		best_pair = (0, 1)
		min_diff = float('inf')

		for i in range(len(offered)):
			for j in range(i + 1, len(offered)):
				diff = abs(offered[i] - offered[j])
				if diff < min_diff:
					min_diff = diff
					best_pair = (i, j)

		# Identify worn-out socks
		discard_list = []
		for i in range(len(offered)):
			# Skip the ones player is wearing
			if i in best_pair:
				continue

			shade = offered[i]
			# White socks fade down to 127, black socks fade up to 64
			if shade == 127 or shade == 64:
				discard_list.append(i)

		# Return the decision
		return Selection(wear=best_pair, discard=tuple(discard_list))
