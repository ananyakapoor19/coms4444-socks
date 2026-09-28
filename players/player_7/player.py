from itertools import combinations

from models.player import GameContext, PlayerSnapshot, Selection, TurnContext
from models.player import Player as BasePlayer


class Player7(BasePlayer):
	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)
		self.days_seen = 0

	def select_socks(self, offered: tuple[int, ...], turn: TurnContext) -> Selection:
		n = len(offered)

		# rough wash count for a sock (white fades 255->127, black 0->64)
		def age(shade: int) -> int:
			return (255 - shade) // 2 if shade > 64 else shade

		# pairs within 6 shades cost nothing
		free_pairs = [
			(i, j) for i, j in combinations(range(n), 2) if abs(offered[i] - offered[j]) <= 6
		]

		if free_pairs:
			# go same colour when we can - white+black pairs split apart as they
			# fade but two whites stay matched. then wear the newest ones and
			# leave the old socks to wear out by themselves (free packs)
			same_color = [(i, j) for i, j in free_pairs if (offered[i] > 64) == (offered[j] > 64)]
			pool = same_color if same_color else free_pairs
			wear_idx = min(pool, key=lambda p: age(offered[p[0]]) + age(offered[p[1]]))
		else:
			# nothing free, just take the closest
			wear_idx = min(
				combinations(range(n), 2), key=lambda p: abs(offered[p[0]] - offered[p[1]])
			)

		leftovers = [i for i in range(n) if i not in wear_idx]
		discard_idx = []

		# only discard if we can still afford packs. 6 discards = one $10 pack,
		# so budget/days_left needs to stay above 10/6
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
				# broke - keep everything, going sockless is 65536 pts
				break

			# worn out socks (127/64) are useless, drop them
			if shade == 127 or shade == 64:
				discard_idx.append(idx)

			elif can_spend and shade != worn_shade and age(shade) >= 2:
				# drop the odd ones out so the drawer stays clustered and we
				# match more often later. keep the fresh ones though (age < 2)
				discard_idx.append(idx)

		return Selection(wear=wear_idx, discard=tuple(discard_idx))
