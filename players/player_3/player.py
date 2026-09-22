"""Group 3: Lagrangian scalarisation over a budget controller.

Every legal action for the day - a wear pair plus a discard subset of the
leftovers - is scored with one function and the argmin is returned:

    cost = embarrassment(i, j)
         + lam * dollars(i, j, S)
         + mu  * migration(i, j, S)
         + nu  * delta_potential(i, j, S)

``dollars`` is the household's expected replacement bill: $10/6 per voluntary
discard, plus a quarter of that for every worn-out sock we put on, because the
engine rolls a 25% hole on capped socks and a hole is a discard we did not
choose. ``migration`` charges the same expected number of new socks in a
second currency: every pristine replacement must absorb 64 wears before it
rejoins a saturated drawer, and the household only has 2n wears per day, so
each new sock consumes 64/(2n) days of the household's wear capacity.

``delta_potential`` is where the drawer's shape enters. The potential of a
sock is the expected (thresholded) mismatch, over the rest of its life,
against a same-colour sock drawn from a *target* belief: the decayed histogram
of every shade we have been offered, blended with the geometric age
distribution that the household's affordable replacement rate produces in
steady state. The blend is what lets the policy start churning from a
pristine drawer: scored against today's drawer alone, a pristine replacement
looks like an outlier, so a myopic potential never spends. Integrating over
the sock's remaining life is what makes it spend early enough: a white sock
three wears behind the mass costs nothing today and 8 points a day from its
next wear on. Wearing a sock moves it one wash step, so its contribution
changes by D(washed) - D(now); discarding replaces it with a pristine one, so
the drawer changes by D(pristine) - D(sock). The second is what decides
discards: a large gain for a sock stranded behind the mass, a loss for a
young one that would merge on its own, which is the asymmetry the greedy
baseline gets wrong half the time.

``lam`` is not fixed. The controller from the previous player survives: it
infers the roommates' spend rate from ``total_spent``, banks the shortfall
against the budget as discard credit, scales lam down to zero as the bank
fills (a slack constraint has a zero multiplier), and sets lam to a survival
value once the money is gone so that the cost of a hole dominates any
mismatch. With lam at infinity and no credit the policy degenerates to
never-discard wear-levelling; with lam near zero it is the churn regime. One
code path covers the whole range.

The roommates
-------------
Nothing in the cost function names them, which is the point: the engine
scores our pair against our pair, so the other n-1 roommates reach us only
through the two things the household shares. The drawer is one pool: they draw
from it uniformly at random, in a freshly shuffled order every day, and
whatever survives the day goes back in for anyone to draw tomorrow - socks
anyone wore, a wash older, and socks anyone declined, unaged. The budget is
one account: it is charged in $10 six-packs as soon as six discards of a
colour have piled up, from any mix of their discards, our discards and holes.
We never see a roommate's hand or their move, and the engine tells nobody who
discarded what, so our whole model of them is built from two signals.

The first is the handful we are dealt. It is a uniform sample of the pool as
the household left it, so the decayed histogram in ``_observe`` is a rolling
estimate of a drawer that everyone is reshaping, and every belief built on it
is household-wide rather than ours. That is why ``_potential`` scores a sock
against a partner drawn at random from the pool rather than against anything
we are holding, and why ``_life_potential`` ages it at the household's wear
rate: a sock we hand back is worn next by whoever draws it, and we see it
again only with probability unit/C. It is also why saving a good sock for
tomorrow is not an available move. Discarding is our only lever on what the
pool contains, and it is the lever that spends everybody's money.

The second signal is ``turn.total_spent``, and ``_may_discard`` is built
around it: subtract the spend we impute to ourselves, divide by the day, and
the remainder is the roommates' rate. Extrapolated to the end of the run and
netted against a reserve, it leaves the slack that decides whether we may
discard at all - so the two regimes fall out of one expression, with no
branch dedicated to either. Against roommates who churn freely their
extrapolated spend swallows the budget, the slack goes negative, the credit
bank empties, and lam sits at full price with zero discards allowed: we stop
paying and dress out of the fresh drawer they are buying. lam still earns its
keep there, because the 25% hole on a worn-out sock is the one way left to
spend the household's money without choosing to, and that term is what steers
us off capped socks. Against hoarders the slack is the whole budget less the
hole replacements their wears force, the bank fills, lam decays toward zero,
and we pay to churn the drawer ourselves. The free-rider asymmetry behind all
of this - a sock we replace improves every roommate's draws as much as our
own, and theirs improve ours - is not fought, only priced: a channel that
carries one number a day leaves nothing to signal, bargain or retaliate with,
so their rate is taken as exogenous.

That inference has two known weaknesses. Holes are charged to the household
but imputed to nobody, so they land in the roommates' rate - including the
ones our own worn-out wears cause - which grows with the length of the run and
errs toward underspending, the safe direction. And ``total_spent`` moves in
$10 lumps and lags the discards that earned it, so for the first few weeks the
rate derived from it is noise; before ``WARMUP_DAYS`` it is not consulted at
all and the warm-up paces against a straight line to the budget. That pacing
test
reads the household's ``total_spent`` rather than our own, which is also what
stops a roster of several copies of this player from each claiming the whole
slack - after the warm-up, each copy sees the others' spend in ``others_rate``
and asks only for what is left.

The endgame is the same arithmetic. Once the money is gone the pool only
shrinks, since everyone's holes still leave it and nothing replaces them, and
a draw of fewer than two socks is the 65,536-point sockless penalty.
``_collapse_expected`` divides the wears the drawer has left by the
household's 2n a day and asks whether the run outlasts it; if it does not,
lam goes to its survival value, we discard nothing, and a quarter-hole
outweighs any mismatch - the drawer being protected is the one everybody
dresses from.

Tuning (5 training seeds, 10 held-out, rosters of 1-4 of us with greedy and
random roommates, $7,490 and $3,000 over 1,080 days) settled on lam = 3,
mu = 0, nu = 0.5 and no wear weight. mu is zero because the migration burden
of a pristine replacement is already priced, in embarrassment units, by the
``D(pristine)`` term; charging it again only throttled spend. The wear weight
is zero because, once the mismatch of the pair actually worn is counted, the
one-step change in a sock's potential is noise next to it.
"""

from itertools import combinations

from core.engine import HOLE_PROBABILITY, PACK_COST, PACK_SIZE
from models.player import GameContext, PlayerSnapshot, Selection, TurnContext
from models.player import Player as BasePlayer
from models.sock import (
	BLACK_CEILING,
	BLACK_FADE,
	BLACK_START,
	WHITE_FADE,
	WHITE_FLOOR,
	WHITE_START,
	Color,
)

FREE_GAP = 6
SOCK_COST = PACK_COST / PACK_SIZE
WEARS_TO_CAP = 64
DEFAULT_DOLLARS_PER_DAY = 7490 / 1080


def color_of(shade: int) -> Color:
	return Color.WHITE if shade > BLACK_CEILING else Color.BLACK


def age_of(shade: int) -> int:
	if shade > BLACK_CEILING:
		return (WHITE_START - shade) // WHITE_FADE
	return shade


def is_worn_out(shade: int) -> bool:
	return shade in (WHITE_FLOOR, BLACK_CEILING)


def washed(shade: int) -> int:
	"""Shade after one wear, mirroring ``Sock.washed``."""
	if shade > BLACK_CEILING:
		return max(WHITE_FLOOR, shade - WHITE_FADE)
	return min(BLACK_CEILING, shade + BLACK_FADE)


def pristine_of(color: Color) -> int:
	return WHITE_START if color is Color.WHITE else BLACK_START


def mismatch(a: int, b: int) -> float:
	"""The engine's embarrassment for wearing shades ``a`` and ``b`` together."""
	gap = abs(a - b)
	return float(gap) if gap > FREE_GAP else 0.0


class Player3(BasePlayer):
	# --- cost multipliers -----------------------------------------------
	# Price of one expected new sock when the credit bank is empty, in
	# embarrassment points per dollar. Scaled down linearly as the bank
	# fills: a slack budget constraint has a zero multiplier.
	LAMBDA = 3.0
	# Price of one expected new sock's migration, in points per day of
	# household wear capacity it consumes.
	MU = 0.0
	# Weights on the drawer-shape potential. A discard reshapes the drawer
	# for the replacement's whole life; a wear moves one sock one step, so
	# its weight is a fraction of the other.
	NU = 0.5
	NU_WEAR = 0.0
	# Days ahead over which the affordable replacement rate reshapes the
	# drawer. The potential is scored against a blend of the observed
	# drawer and the age distribution that rate produces in steady state.
	HORIZON_DAYS = 30
	# Hole deaths per roommate per day once the drawer is saturated. With f
	# the fraction of wears landing on capped socks, deaths/day = 0.25*f*2n
	# and each death needs 64 wears to re-cap, (1-f)*2n = 64*deaths/day,
	# so deaths/day = n/34 whatever the policy. Floor on the replacement rate:
	# roommates who never discard a thing still turn the drawer over at this
	# rate by wearing it out, so the belief never freezes on a saturated
	# drawer even when nobody in the house is choosing to spend.
	HOLE_FLOOR_RATE = 1 / 34
	# Upper bound on per-wear survival inside the life-integrated
	# potential, so its horizon stays finite when nobody is replacing socks.
	LIFE_SURVIVAL_CAP = 0.98
	# lam once the money is gone and the drawer would collapse before the
	# run ends: a quarter-hole then costs more than any possible mismatch.
	LAMBDA_SURVIVAL = 1000.0

	# --- belief -----------------------------------------------------------
	HISTOGRAM_DECAY = 0.9

	# --- budget controller -------------------------------------------------
	MAX_DISCARDS = 2
	RESERVE_FRACTION = 0.03
	RESERVE_PACKS = 4
	CREDIT_CAP = 10.0
	WARMUP_DAYS = 20
	ENDGAME_DAYS = 15
	DRAWER_GUESS = 34

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)
		self.name = 'Group 3'
		self.histogram: dict[Color, dict[int, float]] = {Color.WHITE: {}, Color.BLACK: {}}
		self.last_day_seen = 0
		self.credit = 0.0
		self.my_spend = 0.0
		# Expected voluntary replacements per day across the household,
		# refreshed by the controller: the roommates' rate inferred from
		# total_spent, plus what our own allowance lets us add. Never observed
		# directly - nothing reports another roommate's discards.
		self.household_rate = 0.0
		# Days of household wear capacity one pristine sock consumes. All 2n
		# wears count, not just our two: a replacement is broken in by whoever
		# happens to draw it.
		self.migration_days = WEARS_TO_CAP / (2 * self.roommates)
		# Set by the controller each turn: weight of the steady-state
		# target in the potential, and that target's per-wear survival.
		self.alpha = 0.0
		self.geometric_survival = 1.0
		self._geo_cache: dict[tuple[float, Color, int], float] = {}
		# Observed-potential cache, valid for one turn (the histogram changes
		# once per turn, in _observe).
		self._obs_cache: dict[tuple[Color, int], float] = {}

	# ------------------------------------------------------------------
	# Turn
	# ------------------------------------------------------------------

	def select_socks(self, offered: tuple[int, ...], turn: TurnContext) -> Selection:
		self._observe(offered, turn.day)
		lam, max_discards = self._controller(turn)
		wear, discard = self._decide(offered, lam, max_discards)
		if discard:
			# The engine bills the household, not us: our discards queue with
			# the roommates' and with holes, and a $10 pack is bought once six
			# of a colour have piled up. So we impute our own share at $10/6 a
			# discard and keep it here, because the difference against
			# turn.total_spent is the only view we get of what they spend.
			self.credit -= len(discard)
			self.my_spend += len(discard) * SOCK_COST
		return Selection(wear=wear, discard=discard)

	# ------------------------------------------------------------------
	# Scalarised enumeration
	# ------------------------------------------------------------------

	def _decide(
		self, offered: tuple[int, ...], lam: float, max_discards: int
	) -> tuple[tuple[int, int], tuple[int, ...]]:
		n = len(offered)
		new_sock_price = lam * SOCK_COST + self.MU * self.migration_days

		# Per-sock terms. Everything in the score is additive over socks, so
		# the enumeration below only sums precomputed numbers.
		wear_cost = [0.0] * n
		discard_cost = [0.0] * n
		for k, shade in enumerate(offered):
			color = color_of(shade)
			now = self._life_potential(color, shade)
			after = self._life_potential(color, washed(shade))
			wear_cost[k] = self.NU_WEAR * (after - now)
			if is_worn_out(shade):
				# Putting on a capped sock is a 25% chance of spending the
				# household's money without choosing to, so it is priced at
				# the same lam as a discard. This is the only term lam touches
				# once the controller has forbidden discards, and it is what
				# keeps us off capped socks while the roommates are the ones
				# paying for the drawer.
				wear_cost[k] += HOLE_PROBABILITY * new_sock_price
			fresh = self._life_potential(color, pristine_of(color))
			discard_cost[k] = new_sock_price + self.NU * (fresh - now)

		best_cost = float('inf')
		best_key: tuple[float, ...] = ()
		best: tuple[tuple[int, int], tuple[int, ...]] = ((0, 1), ())
		for i, j in combinations(range(n), 2):
			base = mismatch(offered[i], offered[j]) + wear_cost[i] + wear_cost[j]
			leftovers = [k for k in range(n) if k != i and k != j]
			# Only discards with negative marginal cost can lower the total,
			# and they are independent of each other, so the best subset is
			# the cheapest few of those rather than all 2^len(leftovers).
			gains = sorted(
				((discard_cost[k], k) for k in leftovers if discard_cost[k] < 0),
			)[:max_discards]
			cost = base + sum(c for c, _ in gains)
			# Deterministic tie-break: fewer discards, then younger pair.
			key = (cost, len(gains), age_of(offered[i]) + age_of(offered[j]), i, j)
			if cost < best_cost or (cost == best_cost and key < best_key):
				best_cost = cost
				best_key = key
				best = ((i, j), tuple(sorted(k for _, k in gains)))
		return best

	def _life_potential(self, color: Color, shade: int) -> float:
		"""Expected mismatch over the rest of a sock's life, not just today.

		The sock keeps ageing: a white sock three wears behind a pristine mass
		costs nothing today and 8 points a day from its next wear on. So the
		potential is the survival-weighted mean of ``_potential`` along the
		sock's trajectory, with the same per-wear survival ``q`` that defines
		the target. Once it reaches the cap it stays there, so the tail of the
		geometric series lumps at the capped shade. ``q`` is bounded below 1
		so a never-replaced sock still has a finite (50-wear) horizon.

		The steps are household wears, not ours. A sock handed back goes to the
		common pool, where any of the n roommates may draw it, so it is mostly
		other people who will walk this trajectory - whether the decision here
		is to keep or to replace, the consequence is spread over everybody's
		future draws and we collect only our share of it.
		"""
		q = min(self.geometric_survival, self.LIFE_SURVIVAL_CAP)
		total = 0.0
		mass = 1.0 - q
		current = shade
		steps = 0
		while not is_worn_out(current) and steps < WEARS_TO_CAP:
			total += mass * self._potential(color, current)
			mass *= q
			current = washed(current)
			steps += 1
		# Remaining mass sits at the cap: mass / (1 - q) summed to infinity.
		total += (mass / (1.0 - q)) * self._potential(color, current)
		return total

	def _potential(self, color: Color, shade: int) -> float:
		"""Expected mismatch of ``shade`` against a same-colour sock drawn
		from the belief: the observed drawer blended, with weight ``alpha``,
		with the age distribution the affordable replacement rate produces."""
		observed = self._observed_potential(color, shade)
		if self.alpha <= 0.0:
			return observed
		return (1.0 - self.alpha) * observed + self.alpha * self._target_potential(color, shade)

	def _target_potential(self, color: Color, shade: int) -> float:
		"""Expected mismatch against a sock whose wear count is geometric:
		each day a sock survives replacement with probability ``q``, and
		wears accumulate at 2n/C per day, so P(age = a) is geometric in ``a``
		with the tail lumped at the cap."""
		key = (self.geometric_survival, color, shade)
		cached = self._geo_cache.get(key)
		if cached is not None:
			return cached
		fade = WHITE_FADE if color is Color.WHITE else BLACK_FADE
		start = pristine_of(color)
		q = self.geometric_survival
		p = 1.0 - q
		total = 0.0
		mass = 1.0
		for age in range(WEARS_TO_CAP):
			prob = p * mass
			other = start - fade * age if color is Color.WHITE else start + fade * age
			gap = abs(other - shade)
			if gap > FREE_GAP:
				total += gap * prob
			mass *= q
		capped = WHITE_FLOOR if color is Color.WHITE else BLACK_CEILING
		gap = abs(capped - shade)
		if gap > FREE_GAP:
			total += gap * mass
		if len(self._geo_cache) > 4096:
			self._geo_cache.clear()
		self._geo_cache[key] = total
		return total

	def _observed_potential(self, color: Color, shade: int) -> float:
		cached = self._obs_cache.get((color, shade))
		if cached is not None:
			return cached
		hist = self.histogram[color]
		total = 0.0
		weight = 0.0
		for seen, w in hist.items():
			gap = seen - shade
			if gap < 0:
				gap = -gap
			if gap > FREE_GAP:
				total += gap * w
			weight += w
		value = total / weight if weight else 0.0
		self._obs_cache[(color, shade)] = value
		return value

	# ------------------------------------------------------------------
	# Belief: decayed histogram of offered shades, per colour
	# ------------------------------------------------------------------

	def _observe(self, offered: tuple[int, ...], day: int) -> None:
		"""Fold today's handful into the belief.

		Our handful is a uniform draw from the pool the whole household shares,
		so this histogram is the only estimate we get of a drawer that everyone
		is reshaping - and the estimate is a slow one, since we see ``unit`` of
		C socks a day and nothing at all about the moves that produced them.
		The decay is what keeps it honest about the roommates: at 0.9 the
		effective window is about ten days, which at the default C=40 and unit
		4 is roughly one drawer's worth of sightings - long enough to average
		over our own luck, short enough that a roommate who starts churning
		shows up as a younger drawer within a couple of turnovers.
		"""
		self._obs_cache.clear()
		factor = self.HISTOGRAM_DECAY ** max(day - self.last_day_seen, 1)
		self.last_day_seen = day
		for hist in self.histogram.values():
			for shade in list(hist):
				weight = hist[shade] * factor
				if weight < 1e-3:
					del hist[shade]
				else:
					hist[shade] = weight
		for shade in offered:
			hist = self.histogram[color_of(shade)]
			hist[shade] = hist.get(shade, 0.0) + 1.0

	def _mean_wears_left(self) -> float:
		total = 0.0
		weight = 0.0
		for hist in self.histogram.values():
			for shade, w in hist.items():
				total += (WEARS_TO_CAP - age_of(shade)) * w
				weight += w
		return total / weight if weight else float(WEARS_TO_CAP)

	# ------------------------------------------------------------------
	# Budget controller -> (lam, discards allowed today)
	# ------------------------------------------------------------------

	def _collapse_expected(self, turn: TurnContext) -> bool:
		"""Would the shared drawer wear out before the run ends?

		Wears left in the pool, divided by the 2n a day the household gets
		through: the roommates' wears age the drawer just as fast as ours, and
		once the money is gone their holes drain it as well, down toward the
		draw of fewer than two socks that scores the sockless penalty. The size
		is guessed rather than read off ``capacity`` because by the time this
		matters the pool has been shrinking for a while and we cannot see by
		how much.
		"""
		life_days = self.DRAWER_GUESS * self._mean_wears_left() / (2 * self.roommates) + 10
		return self.days - turn.day > life_days

	def _controller(self, turn: TurnContext) -> tuple[float, int]:
		"""Set today's price of money and how many discards the credit bank
		covers. Returns ``(lam, max_discards)``.

		Both numbers come out of what the roommates are doing with the shared
		account: the bank is fed by whatever they leave unspent, so heavy
		spenders on their side drive lam to full price and the discard
		allowance to zero, and hoarders drive it the other way.
		"""
		broke = turn.budget_remaining < PACK_COST
		if broke and self._collapse_expected(turn):
			self._set_target(turn, socks_per_day=0.0)
			return self.LAMBDA_SURVIVAL, 0
		may = self._may_discard(turn)
		self._set_target(turn, socks_per_day=self.household_rate)
		lam = self.LAMBDA * max(0.0, 1.0 - self.credit / self.CREDIT_CAP)
		if not may:
			return lam, 0
		return lam, min(self.MAX_DISCARDS, int(self.credit))

	def _set_target(self, turn: TurnContext, socks_per_day: float) -> None:
		"""Turn the household's replacement rate into the steady-state age
		distribution the potential is scored against, and how much of the
		drawer that rate turns over within the horizon.

		``socks_per_day`` is the whole house's replacement rate, ours included,
		because the drawer the potential is scored against is the shared one: a
		roommate buying six socks freshens our draws exactly as much as we
		would ourselves. So a household that churns pushes ``alpha`` up and the
		target young, and this policy responds by discarding less and wearing
		what the others have paid for.
		"""
		rate = socks_per_day + self.HOLE_FLOOR_RATE * self.roommates
		horizon = min(self.HORIZON_DAYS, max(1, self.days - turn.day))
		self.alpha = min(1.0, rate * horizon / self.capacity)
		# A sock is replaced at rate/C per day and worn at 2n/C per day. The
		# chance the next event in its life is a wear rather than a
		# replacement is 2n/(2n + rate): strictly inside (0, 1), so the
		# life-integrated potential never degenerates to today's value.
		wears = 2 * self.roommates
		self.geometric_survival = wears / (wears + rate)

	def _may_discard(self, turn: TurnContext) -> bool:
		"""Decide whether today's allowance covers a discard, and bank it.

		This is where the roommates are actually inferred. ``total_spent`` is
		the household's, so the only estimate of their behaviour available is
		what is left of it after subtracting the spend we impute to ourselves,
		divided by the days elapsed. Two things bias that estimate, in opposite
		directions and on different timescales. Hole replacements are attributed
		to nobody, so the ones our own worn-out wears cause land on their side of
		the subtraction; that term grows with the run, and it credits them with
		more than they chose to spend, which shrinks our allowance - the safe
		direction to be wrong in on a shared account. Against it, the engine buys
		in packs, so ``total_spent`` trails the discards already committed by up
		to five socks a colour; that reads their rate low and our slack high, but
		it is bounded while the other term is not, so it only matters early -
		which is what the warm-up below is for.

		Whatever the estimate, it is treated as exogenous - extrapolated
		linearly to the end of the run and subtracted from what is left, and
		what survives that is ours to spend. There is no attempt to influence
		it: the roommates cannot be signalled, and a strategy that spent more in
		the hope of shaming them into spending less has nothing to send the
		message on.
		"""
		days_left = self.days - turn.day
		if days_left < self.ENDGAME_DAYS or turn.budget_remaining < PACK_COST:
			# Nothing we buy now pays for itself before the run ends, and a
			# household already out of pack money has no allowance to divide.
			self.household_rate = 0.0
			return False

		budget = turn.total_spent + turn.budget_remaining
		remaining = turn.budget_remaining
		if budget == float('inf'):
			# No budget was set, so the roommates cannot bankrupt us - but the
			# run still reports what the household spent, so we hold ourselves
			# to a figure and infer their rate against it exactly as we would
			# against a real one.
			budget = DEFAULT_DOLLARS_PER_DAY * self.days
			remaining = budget - turn.total_spent
		# Held back against the roommates being lumpier than a straight-line
		# extrapolation of their past: a late spending surge on their side must
		# not be the thing that leaves the house unable to buy a pack.
		reserve = max(self.RESERVE_PACKS * PACK_COST, budget * self.RESERVE_FRACTION)

		if turn.day < self.WARMUP_DAYS:
			# Too early to read anything off total_spent - it moves in $10 lumps
			# and lags the discards that earned it, so on day three the inferred
			# rate is noise. Pace against a straight line to the budget instead.
			# The test is on the household's total, not ours, so several copies
			# of this player in one house stop together at the pacing line
			# rather than each spending as if the other were idle.
			pace = (budget - reserve) * turn.day / self.days
			allowed = turn.total_spent + PACK_COST <= pace
			self.credit = min(self.credit + (1.0 if allowed else 0.0), self.CREDIT_CAP)
			self.household_rate = (budget - reserve) / self.days / SOCK_COST
			return allowed

		others_rate = max(0.0, (turn.total_spent - self.my_spend) / turn.day)
		slack = remaining - reserve - others_rate * days_left
		if slack <= 0:
			# The roommates' projected spend accounts for the rest of the
			# budget. Free-ride: bank nothing, discard nothing, and dress out of
			# the drawer they are keeping fresh. The belief still tracks their
			# churn through household_rate, so the potential knows the drawer is
			# young even though we are not the ones paying for it.
			self.credit = 0.0
			self.household_rate = others_rate / SOCK_COST
			return False
		my_rate = slack / SOCK_COST / days_left
		self.household_rate = others_rate / SOCK_COST + my_rate
		# The bank turns a fractional daily allowance into whole discards. It is
		# also what shares the slack between copies of this player: each one
		# sees the others' spend in others_rate, so the claim shrinks as they
		# spend it rather than every copy claiming all of it.
		self.credit = min(self.credit + my_rate, self.CREDIT_CAP)
		return self.credit >= 1.0
