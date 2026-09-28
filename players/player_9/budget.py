"""Budget-management methods for Player99."""

import math

from core.engine import PACK_COST
from models.player import TurnContext


class BudgetMethods:
	"""Methods that update the shared Player99 budget state."""

	def _update_budget(self, turn: TurnContext) -> tuple[float, float]:

		# Spending per player per day if no socks are discarded, rounded up from an estimated 0.038
		SPPPD = 0.04
		WEIGHT_FACTOR = (turn.day - 1) / turn.day

		self.budget_by_day.append(turn.budget_remaining)

		# we want to increment early in case of timeout
		self.days_remaining -= 1

		# No budget means no money for a pack, or no budget set at all
		self.broke = turn.budget_remaining < PACK_COST
		self.no_budget = self.broke or math.isinf(turn.budget_remaining)
		min_budget = SPPPD * self.roommates * self.days_remaining
		self.low_budget = not self.no_budget and turn.budget_remaining <= min_budget

		base_velocity = SPPPD * self.roommates
		reserve = base_velocity * self.days_remaining
		active_days = max(self.days_remaining - (self.capacity / self.roommates), 1)
		spendable_budget = max(0, turn.budget_remaining - reserve)
		target_velocity = base_velocity + spendable_budget / active_days
		prev_spent = 0
		if len(self.budget_by_day) > 1:
			prev_spent = self.budget_by_day[-2] - self.budget_by_day[-1]

		# spending per day, weighted as an infinite series according to weight factor
		self.budget_velocity = WEIGHT_FACTOR * self.budget_velocity + (1 - WEIGHT_FACTOR) * (
			prev_spent
		)
		projected_error = (target_velocity - self.budget_velocity) * active_days

		return SPPPD, projected_error

	def spendMore(self):

		self.black_bound = max(0, self.black_bound - 1)
		self.white_bound = min(255, self.white_bound + 2)

	def spendLess(self):

		self.black_bound = min(64, self.black_bound + 1)
		self.white_bound = max(127, self.white_bound - 2)

	def _adjust_budget(self, turn: TurnContext, SPPPD: float, projected_error: float) -> None:
		# Discarding cases
		# Case 1, not enough budget left for discarding
		if self.budget_by_day[-1] <= SPPPD * self.roommates * (self.days_remaining):
			self.black_bound = 64
			self.white_bound = 127

		# Case 2, more budget than can spend
		elif self.budget_by_day[-1] >= self.roommates * (self.days_remaining) * 10:
			self.black_bound = -1
			self.white_bound = 257

		# Case 3, too early to discard
		elif (turn.day) * self.roommates < self.capacity:
			if self.budget_by_day[-1] < self.budget_by_day[0]:
				self.contains_bad_actor = True

		# Case 4, projected to leave at least one pack unspent
		elif projected_error > PACK_COST:
			self.spendMore()

		# Case 5, projected to overspend by at least one pack
		elif projected_error < -PACK_COST:
			self.spendLess()

		# Case 6, projected spending is within one pack of the target
		else:
			pass
