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

from core.engine import PACK_COST, PACK_SIZE
from models.player import GameContext, PlayerSnapshot, Selection, TurnContext
from models.player import Player as BasePlayer
from models.sock import BLACK_CEILING, WHITE_FADE, WHITE_START

from .budget import BudgetMethods
from .discarding import DiscardMethods
from .pairing import PairingMethods


class Player9(BudgetMethods, PairingMethods, DiscardMethods, BasePlayer):
	"""Rename me to Player<k>, where <k> is your group number."""

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)

		# super() has already set these from ctx and snapshot:
		#
		#   self.index           which roommate you are (0-based))
		#   self.id              your UUID, stable for the whole simulation
		#   self.capacity        C, the drawer size at the start
		#   self.roommates       n, how many of you share the drawer
		#   self.selection_unit  how many socks you are handed each day
		#   self.days            how long the simulation runs
		#
		# The engine constructs you once, before day 1, and it constructs you
		# itself - you cannot preload state into an already-built object. Anything
		# you want to carry between days lives on self, so initialise it here.
		self.days_remaining = self.days + 1
		self.budget_by_day = []
		self.black_bound = 64
		self.white_bound = 127
		self.budget_velocity = 0
		self.contains_bad_actor = False
		self.broke = False
		self.no_budget = False
		self.low_budget = False
		self.discard_high = 0
		self.mixed_mode = False

	def is_black(self, shade: int) -> bool:
		return shade <= BLACK_CEILING

	def wears(self, shade: int) -> float:
		"""How many times a sock has been worn. White fades 2 per wear, black rises 1."""
		return shade if self.is_black(shade) else (WHITE_START - shade) / WHITE_FADE

	def select_socks(self, offered: tuple[int, ...], turn: TurnContext) -> Selection:
		"""
		WHAT YOU CAN SEE

			offered[i]                  the shade of the i-th sock on offer
			turn.day                    today's day number, 1-based
			turn.total_spent            dollars spent by the household so far
			turn.embarrassment_history  your own daily scores, one per day
			turn.total_embarrassment    the sum of that history
			self.capacity / self.roommates / self.selection_unit / self.days
		"""

		def wears(sock):
			if not self.is_black(sock):
				return (255 - sock) // 2
			return sock

		SPPPD, projected_error = self._update_budget(turn)

		affordable_socks = PACK_SIZE / PACK_COST * max(0, turn.budget_remaining)
		required_age = 2 * self.roommates * self.days_remaining / (self.capacity + affordable_socks)
		proposed_high = 4 * round((self.capacity / 2 + required_age) / 4)
		proposed_high = max(40, min(64, proposed_high))

		# Raise the cutoff if the budget becomes less adequate, but never lower it.
		self.discard_high = max(self.discard_high, proposed_high)
		discard_low = self.discard_high // 2

		if turn.day == 60 and turn.total_spent >= 10:
			self.mixed_mode = True

		# picking socks to wear
		left, right = self._choose_wear(offered)

		minW, minB, maxW, maxB = 0, 0, 0, 0

		for sock in offered:
			if self.is_black(sock):
				minB = min(minB, wears(sock))
				maxB = max(maxB, wears(sock))
			else:
				minW = min(minW, wears(sock))
				maxB = max(maxW, wears(sock))

		if max(maxB - minB, maxW - minW) > self.discard_high:
			self.black_bound = (maxB + minB) // 2
			self.white_bound = 255 - (minW + maxW)

		if max(maxB - minB, maxW - minW) < discard_low:
			self.black_bound = 65
			self.white_bound = 100

		if max(maxW, maxB) > 63:
			self.black_bound = 32
			self.white_bound = 192

		if turn.budget_remaining <= max(self.budget_by_day[0] * (0.05), 20):
			self.black_bound = 65
			self.white_bound = 100

		dis = self._choose_discards(offered, left, right)

		return Selection(wear=(left, right), discard=(tuple(dis)))
