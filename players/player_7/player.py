from itertools import combinations

from models.player import GameContext, PlayerSnapshot, Selection, TurnContext
from models.player import Player as BasePlayer


class Player7(BasePlayer):
	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)
		self.days_seen = 0

	def select_socks(self, offered: tuple[int, ...], turn: TurnContext) -> Selection:
		n = len(offered)

		# white socks go from 255 down to 127 (2 per wash)
		# black socks go from 0 up to 64 (1 per wash)
		# so age tells us how many washes a sock has been through
		def age(shade: int) -> int:
			return (255 - shade) // 2 if shade > 64 else shade

		# first find all pairs that cost us nothing to wear (diff <= 6 is free)
		# if we have options, wear the freshest ones and save the old ones
		# to hole out on their own -- that gives us free restocks
		free_pairs = [
			(i, j) for i, j in combinations(range(n), 2) if abs(offered[i] - offered[j]) <= 6
		]

		if free_pairs:
			wear_idx = min(free_pairs, key=lambda p: age(offered[p[0]]) + age(offered[p[1]]))
		else:
			# no free pair today, just minimize the damage
			wear_idx = min(
				combinations(range(n), 2), key=lambda p: abs(offered[p[0]] - offered[p[1]])
			)

		leftovers = [i for i in range(n) if i not in wear_idx]
		discard_idx = []

		# figure out if we can afford to throw socks away
		# every 6 discards of the same color buys a fresh pack
		# rule of thumb: if budget / days left >= 10/6, we're keeping pace
		broke = turn.budget_remaining == 0
		days_remaining = max(self.days - turn.day + 1, 1)

		if turn.budget_remaining == float('inf'):
			can_spend = True
		else:
			daily_rate = turn.budget_remaining / days_remaining
			can_spend = daily_rate >= (10 / 6)

		worn_shade = (offered[wear_idx[0]] + offered[wear_idx[1]]) / 2

		for idx in leftovers:
			shade = offered[idx]

			if broke:
				# out of money -- keep everything, going sockless costs 65536
				break

			# terminal socks (fully faded) just clog the drawer, toss them
			if shade == 127 or shade == 64:
				discard_idx.append(idx)

			elif can_spend and abs(shade - worn_shade) > 6:
				# if we're on budget, also toss socks too far from what we wore
				# they're hard to match and not worth keeping around
				discard_idx.append(idx)

		return Selection(wear=wear_idx, discard=tuple(discard_idx))
