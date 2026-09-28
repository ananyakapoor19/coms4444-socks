"""Sock-discarding methods for Player99."""


class DiscardMethods:
	"""Methods for choosing which un-worn socks to discard."""

	def _choose_discards(self, offered: tuple[int, ...], left: int, right: int) -> list[int]:
		# choosing discarded socks
		dis = []
		for i in range(len(offered)):
			if i in (left, right):
				pass
			else:
				if offered[i] > self.black_bound and offered[i] < self.white_bound:
					dis.append(i)

		return dis
