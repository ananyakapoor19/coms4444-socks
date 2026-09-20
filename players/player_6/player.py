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

THRESHOLD = 6
BUCKETS = 8
HIST_DECAY = 0.985
PACK_COST = 10.0
WHITE_CUTOFF = 200
ENDGAME_START = 0.8
ENDGAME_RESERVE = 0.2


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

	def _compatibility(self, shade: int) -> float:
		hist = self.black_hist if self._is_black(shade) else self.white_hist
		total = sum(hist)
		return hist[self._bucket(shade)] / total if total else 0.0

	def _embarrassment(self, a: int, b: int) -> int:
		diff = abs(a - b)
		return 0 if diff <= THRESHOLD else diff

	def _budget_allows_discard(self, turn: TurnContext) -> bool:
		if turn.budget_remaining == float('inf'):
			return True

		current_budget = turn.total_spent + turn.budget_remaining
		if self.estimated_budget is None or current_budget > self.estimated_budget:
			self.estimated_budget = current_budget

		if turn.budget_remaining < PACK_COST:
			return False

		progress = turn.day / self.days if self.days else 1.0
		if progress < ENDGAME_START or self.estimated_budget is None:
			return True

		reserve = self.estimated_budget * ENDGAME_RESERVE
		return turn.budget_remaining - PACK_COST >= reserve

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
		self._update_histograms(offered)

		best_pair = (0, 1)
		best_key = None
		for i in range(len(offered)):
			for j in range(i + 1, len(offered)):
				a = offered[i]
				b = offered[j]
				embarrassment = self._embarrassment(a, b)
				cross_colour = self._is_black(a) != self._is_black(b)
				closeness = abs(a - b)
				compatibility = self._compatibility(a) + self._compatibility(b)
				# Embarrassment is the first key, so the chosen pair always has
				# the minimum immediate score. Later keys only break ties.
				key = (embarrassment, cross_colour, -compatibility, closeness)
				if best_key is None or key < best_key:
					best_key = key
					best_pair = (i, j)

		discard: list[int] = []
		if self._budget_allows_discard(turn):
			for i, shade in enumerate(offered):
				if i in best_pair:
					continue

				compatibility = self._compatibility(shade)
				black = self._is_black(shade)
				worn_out = shade == 64 if black else shade == 127
				white_too_old = not black and shade < WHITE_CUTOFF

				if worn_out or white_too_old or compatibility < 0.08:
					discard.append(i)

		return Selection(wear=best_pair, discard=tuple(discard))
