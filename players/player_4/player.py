from itertools import combinations

from core.engine import PACK_COST
from models.player import GameContext, PlayerSnapshot, Selection, TurnContext
from models.player import Player as BasePlayer

THRESHOLD = 6

# Shades above this belong to a white sock: the drawer's black socks top out
# at BLACK_CEILING = 64 and its white ones bottom out at WHITE_FLOOR = 127, so
# nothing lands in between.
WHITE_CUTOFF = 64
# A white sock leaves the shop at 255 and loses WHITE_FADE per wash; a black
# one leaves at 0 and gains 1, so its shade already counts its washes.
WHITE_START = 255
WHITE_FADE = 2


def wears(shade: int) -> float:
	"""How many wash cycles a sock of this shade has been through."""
	if shade > WHITE_CUTOFF:
		return (WHITE_START - shade) / WHITE_FADE
	return float(shade)


class Player4(BasePlayer):
	"""Wears the closest-matching pair, like the greedy baseline.

	Ties on shade difference - and at the extremes they are common, since
	every pristine white sock reads 255 - go to the pair with the fewest
	washes behind it. Socks fade towards being worn out and a worn-out sock
	has a 25% chance of coming back as a hole, so leaving the older socks in
	the drawer keeps the household close to worn out without tipping over.

	Discard rule: a white sock (shade > 64) under WHITE_MIN is discarded, and
	one over WHITE_MIN is always kept. A black sock (shade <= 64) over
	BLACK_MAX is discarded, and one under BLACK_MAX is always kept. A leftover
	sitting exactly at a threshold falls back to the greedy baseline's discard
	rule.

	None of that applies while the money looks tight. Every discard eventually
	buys a six-pack, so throwing socks out at a rate the budget cannot sustain
	ends the run with an empty drawer and sockless days, which cost 256**2
	apiece. Spending so far is extrapolated over the days left, and discarding
	is allowed only while the projected total still leaves RESERVE in hand.

	WHITE_MIN, BLACK_MAX and RESERVE are class attributes (not module globals)
	so a sweep script can subclass with different values without touching the
	CLI.
	"""

	WHITE_MIN = 240
	BLACK_MAX = 7
	RESERVE = 30

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)

	def can_discard(self, turn: TurnContext) -> bool:
		"""True while the budget can absorb today's spending pace.

		``total_spent`` covers the days already replenished - the engine buys
		after the whole household has dressed - so the average is over
		``day - 1`` of them. On day one there is nothing to average and the
		projection is zero, which just asks whether the budget clears RESERVE
		at all. With no budget ``budget_remaining`` is ``inf`` and this is
		always true.
		"""
		elapsed = turn.day - 1
		average = turn.total_spent / elapsed if elapsed > 0 else 0.0
		days_left = self.days - turn.day + 1
		if days_left < 0:
			days_left = 0
		projected = average * days_left
		return turn.budget_remaining - projected >= min(self.capacity, self.RESERVE)

	def select_socks(self, offered: tuple[int, ...], turn: TurnContext) -> Selection:
		i, j = min(
			combinations(range(len(offered)), 2),
			key=lambda p: (
				abs(offered[p[0]] - offered[p[1]]),
				wears(offered[p[0]]) + wears(offered[p[1]]),
			),
		)
		worn = (offered[i] + offered[j]) / 2

		if not self.can_discard(turn):
			return Selection(wear=(i, j))

		leftovers = [k for k in range(len(offered)) if k not in (i, j)]
		discard: list[int] = []
		greedy_leftovers: list[int] = []

		for k in leftovers:
			shade = offered[k]
			if shade > WHITE_CUTOFF:
				if shade < self.WHITE_MIN:
					discard.append(k)
				elif shade > self.WHITE_MIN:
					pass
				else:
					greedy_leftovers.append(k)
			else:
				if shade > self.BLACK_MAX:
					discard.append(k)
				elif shade < self.BLACK_MAX:
					pass
				else:
					greedy_leftovers.append(k)

		if greedy_leftovers and turn.budget_remaining >= PACK_COST:
			worst = max(greedy_leftovers, key=lambda k: abs(offered[k] - worn))
			if abs(offered[worst] - worn) > THRESHOLD:
				discard.append(worst)

		return Selection(wear=(i, j), discard=tuple(discard))
