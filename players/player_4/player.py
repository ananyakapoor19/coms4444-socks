from itertools import combinations

from core.engine import PACK_COST
from models.player import GameContext, PlayerSnapshot, Selection, TurnContext
from models.player import Player as BasePlayer

THRESHOLD = 6


class Player4(BasePlayer):
	"""Wears the closest-matching pair, like the greedy baseline.

	Discard rule: a white sock (shade > 64) under WHITE_MIN is discarded, and
	one over WHITE_MIN is always kept. A black sock (shade <= 64) over
	BLACK_MAX is discarded, and one under BLACK_MAX is always kept. A leftover
	sitting exactly at a threshold falls back to the greedy baseline's discard
	rule.

	WHITE_MIN and BLACK_MAX are class attributes (not module globals) so a
	sweep script can subclass with different values without touching the CLI.
	"""

	WHITE_MIN = 240
	BLACK_MAX = 7

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)

	def select_socks(self, offered: tuple[int, ...], turn: TurnContext) -> Selection:
		i, j = min(
			combinations(range(len(offered)), 2), key=lambda p: abs(offered[p[0]] - offered[p[1]])
		)
		worn = (offered[i] + offered[j]) / 2

		leftovers = [k for k in range(len(offered)) if k not in (i, j)]
		discard: list[int] = []
		greedy_leftovers: list[int] = []

		for k in leftovers:
			shade = offered[k]
			if shade > 64:
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
