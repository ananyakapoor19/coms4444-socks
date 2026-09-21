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

from collections import defaultdict, deque
from itertools import combinations

from models.player import GameContext, PlayerSnapshot, Selection, TurnContext
from models.player import Player as BasePlayer


class Player2(BasePlayer):
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
		self.budget_per_day = []
		self.sock_distributions_per_day = [defaultdict(int)]

		# window size for distribution statistics
		self.running_window_size = 20
		self.global_history = []
		self.local_black_history = deque(maxlen=self.running_window_size)
		self.local_white_history = deque(maxlen=self.running_window_size)
		self.black_history = []
		self.white_history = []

		self.embarassment_thresh = 6
		# TODO: maybe per-color thresholds?
		self.outlier_z = 1.5
		self.min_dist_samples = 10

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

		"""
		want to have:
		[x] want to track spending throughout the simulation
		[x] want to track what socks we've seen so far
		- prioritize white socks

		# black sock selection & discarding policy
		1. embarassment_thresh = X
		2. if minimum possible embarassment < embarassment_thresh
		3. check black sock STD & white sock STD, compare to previous day(s)
		4. discard black socks that fall outside of X STD from mean
		5. 

		# new_mean = (old_mean * window_size + new_value) / (window_size + 1)
		"""
		# update our distribution of sock colors seen so far
		for _idx, sock_value in enumerate(offered):
			if sock_value > 64:
				self.white_history.append(sock_value)
			else:
				self.black_history.append(sock_value)
			self.global_history.append(sock_value)

		# want to track budget / spending, per day
		self.budget_per_day.append(turn.budget_remaining)

		# cooperative policy - will only discard socks if we haven't been overspending as a household
		self.days_seen += 1
		initial_budget = turn.total_spent + turn.budget_remaining
		thresh = initial_budget / self.days  # avg
		bool_discard_socks = turn.total_spent / turn.day < thresh

		white_socks = []
		black_socks = []

		for i, sock_value in enumerate(offered[:4]):
			# if white sock
			if sock_value > 64:
				white_socks.append(i)
			# black sock
			else:
				black_socks.append(i)

		# TODO: need to pick 2 socks that fall below self.embarassment_thresh
		if len(black_socks) >= 2:
			socks_to_wear = [black_socks[0], black_socks[1]]
		else:
			socks_to_wear = [white_socks[0], white_socks[1]]

		if bool_discard_socks:
			return Selection(wear=socks_to_wear, discard=white_socks)
		else:
			return Selection(wear=socks_to_wear, discard=())


def pairwise_sock_embarassments(offered):
	results = []

	for i, j in combinations(range(len(offered)), 2):
		diff = abs(offered[i] - offered[j])
		embarrassment = diff if diff > 6 else 0

		results.append(
			{'pair': (i, j), 'shades': (offered[i], offered[j]), 'embarrassment': embarrassment}
		)
	return results
