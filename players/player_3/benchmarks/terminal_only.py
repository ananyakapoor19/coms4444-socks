"""Terminal-only replacement benchmark.

Keep the original pair evaluation and controller, but discard terminal leftovers
whenever a pack is affordable, bypassing voluntary credit and its bookkeeping.
"""

from core.engine import PACK_COST
from models.player import Selection, TurnContext
from players.player_3.benchmarks.original import Player3, is_worn_out


class TerminalOnly3(Player3):
	def select_socks(self, offered: tuple[int, ...], turn: TurnContext) -> Selection:
		self._observe(offered, turn.day)
		lam, _ = self._controller(turn)
		wear, _ = self._decide(offered, lam, 0)
		discard = tuple(
			i
			for i, shade in enumerate(offered)
			if i not in wear and is_worn_out(shade) and turn.budget_remaining >= PACK_COST
		)
		return Selection(wear=wear, discard=discard)
