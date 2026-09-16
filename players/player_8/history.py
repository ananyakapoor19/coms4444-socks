"""Socks seen by one player during a game. The same sock may appear more than once."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SockObservation:
	"""The day and socks offered, in their original order."""

	day: int
	offered: tuple[int, ...]

	@property
	def black_shades(self) -> tuple[int, ...]:
		shades = []
		for shade in self.offered:
			if shade < 65:
				shades.append(shade)
		return tuple(shades)

	@property
	def white_shades(self) -> tuple[int, ...]:
		shades = []
		for shade in self.offered:
			if shade >= 127:  # White socks stop fading at 127.
				shades.append(shade)
		return tuple(shades)


class SockHistory:
	"""A separate history for each player."""

	def __init__(self) -> None:
		self._records: list[SockObservation] = []

	def record(self, *, day: int, offered: tuple[int, ...]) -> None:
		observation = SockObservation(day=day, offered=tuple(offered))
		self._records.append(observation)

	@property
	def records(self) -> tuple[SockObservation, ...]:
		# Return a tuple so callers cannot change the stored list.
		return tuple(self._records)
