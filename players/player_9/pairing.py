"""Sock-pairing methods for Player99."""

from itertools import combinations

from core.engine import EMBARRASSMENT_THRESHOLD
from models.sock import BLACK_CEILING, WHITE_FLOOR

WORN_OUT = (WHITE_FLOOR, BLACK_CEILING)


class PairingMethods:
	"""Methods for choosing which offered socks to wear."""

	def _choose_wear(self, offered: tuple[int, ...]) -> tuple[int, int]:

		def wears_left(sock):
			if not self.is_black(sock):
				return (255 - sock) // 2
			return sock

		def preference(p: tuple[int, int]) -> tuple[int, float, int, float, int]:
			a, b = offered[p[0]], offered[p[1]]
			different_colors = self.is_black(a) != self.is_black(b)
			diff = abs(a - b)
			worn_left = min(wears_left(a), wears_left(b))
			worn_right = max(wears_left(a), wears_left(b))
			cost = diff if diff > EMBARRASSMENT_THRESHOLD else 0
			# Prefer black socks over white among pairs with the same embarrassment cost
			whites = (not self.is_black(a)) + (not self.is_black(b))
			# Prefer young socks over old among otherwise equivalent pairs
			age = self.wears(a) + self.wears(b)
			# A worn-out sock can get a hole, and with no money it is never replaced
			# hole_risk = (a in WORN_OUT) + (b in WORN_OUT) if self.broke else 0

			return (different_colors, cost, worn_left, worn_right, whites, age, diff)

		# Pick the least embarrassing pair, then the preferred one, then the two closest socks
		return min(combinations(range(len(offered)), 2), key=preference)

	def _choose_greedy_wear(self, offered: tuple[int, ...]) -> tuple[int, int]:
		left, right = min(
			combinations(range(len(offered)), 2), key=lambda p: abs(offered[p[0]] - offered[p[1]])
		)
		return left, right
