# player 7 - pick how to spend from the budget on day 1, then stick with it.
#
# on the first morning we work out how many wears each sock would need to last
# if the house spent its money evenly (the house wears 2n socks a day, so if we
# can afford d new socks a day each one has to last about 2n/d wears). that
# number puts us in one of three modes for the rest of the run:
#   lavish - plenty of money, toss anything that doesn't match what we wore
#   paced  - some money, toss socks once they've had their share of wears so
#            the budget gets used evenly instead of all at once
#   priced - not much money, give every move a cost in embarrassment points
#            and pick the cheapest one
#
# ideas we took from other groups and wrote our own way: the 2n/d wear count
# (group 10), and scoring a whole move - mismatch, money and drawer fit - as one
# number, with a worn out sock's hole risk counted as money (group 3)

from dataclasses import dataclass, replace
from itertools import combinations

from models.player import GameContext, PlayerSnapshot, Selection, TurnContext
from models.player import Player as BasePlayer

PACK_COST = 10.0
SOCK_COST = PACK_COST / 6
FREE_GAP = 6
WHITE_NEW, WHITE_DONE = 255, 127
BLACK_NEW, BLACK_DONE = 0, 64


# ----------------------------------------------------------------------
# tuning
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Knobs:
	# all the numbers we can tweak, with what they do

	# lavish tossing needs at least this much money per roommate per day
	lavish_per_roommate: float = SOCK_COST / 2
	# socks per roommate per day lost to holes once the drawer is worn in
	hole_rate: float = 1 / 34
	# penalty for wearing a worn out sock when we expect to run dry
	dry_penalty: float = 1000.0
	# socks to keep above the bare minimum before we call it running dry
	dry_margin: int = 4
	# never toss a sock with fewer wears than this
	min_toss_wears: int = 2
	# daily fade of old sightings in the drawer memory (ruled modes)
	memory_fade: float = 0.97
	# how many hole replacements' worth of money to hold back
	hole_reserve: float = 1.0
	# a swap has to raise the free match chance by at least this much
	swap_min_gain: float = 0.1
	# swaps allowed per turn when money is tight
	swaps_per_turn: int = 2
	# only swap socks with this many wears (fresh ones blend in by
	# themselves, worn out ones still match the other old socks)
	swap_wears: tuple[int, int] = (3, 30)
	# with tight money keep worn out socks instead of paying to replace them
	keep_worn_when_tight: bool = True
	# stop lavish tossing if the house's pace so far would eat the hole money
	pace_guard: bool = True
	# a drawer is tight when C is below this many times the daily draw
	tight_drawer: float = 1.5
	# ...and a budget is thin below this many dollars per roommate per day
	thin_dollars: float = 0.05
	# paced mode only while each sock must last at most this many wears
	paced_max_wears: float = 25
	# priced mode when, on day 1, each sock must last more than this
	priced_min_wears: float = 13
	# priced mode: points per dollar when the credit bank is empty
	point_per_dollar: float = 3.0
	# priced mode: weight on how much better a fresh sock would fit
	fit_weight: float = 1.0
	# priced mode: daily fade of old sightings
	priced_memory_fade: float = 0.9
	# priced mode: most discard credit we can bank, and use in one turn
	bank_cap: float = 10.0
	priced_tosses_per_turn: int = 2
	# priced mode: packs of money we never touch, in case roommates spend in bursts
	reserve_packs: int = 1


TUNED = Knobs()
# with a roomy drawer the plainer rules did better in testing
ROOMY = replace(
	TUNED,
	pace_guard=False,
	keep_worn_when_tight=False,
	swaps_per_turn=1,
	swap_wears=(TUNED.min_toss_wears, 64),
)


# ----------------------------------------------------------------------
# shade arithmetic
# ----------------------------------------------------------------------


def is_white(shade: int) -> bool:
	return shade > BLACK_DONE


def wears(shade: int) -> int:
	# rough wash count - white fades 2 a wash, black goes up 1
	return (WHITE_NEW - shade) // 2 if is_white(shade) else shade


def is_done(shade: int) -> bool:
	# faded all the way, so every wear could put a hole in it
	return shade in (WHITE_DONE, BLACK_DONE)


def charge(a: int, b: int) -> int:
	# what the engine charges for wearing a and b together
	gap = abs(a - b)
	return gap if gap > FREE_GAP else 0


def brand_new(shade: int) -> int:
	return WHITE_NEW if is_white(shade) else BLACK_NEW


def after_wash(shade: int) -> int:
	return max(WHITE_DONE, shade - 2) if is_white(shade) else min(BLACK_DONE, shade + 1)


def colour(shade: int) -> str:
	return 'w' if is_white(shade) else 'b'


# ----------------------------------------------------------------------
# what we remember about the drawer
# ----------------------------------------------------------------------


class ShadeMemory:
	# faded counts of the shades we've been handed, per colour

	def __init__(self, fade: float, forget_below: float) -> None:
		self.fade = fade
		self.forget_below = forget_below
		self.counts: dict[str, dict[int, float]] = {'w': {}, 'b': {}}
		self.last_day = 0

	def observe(self, hand: tuple[int, ...], day: int) -> None:
		factor = self.fade ** max(day - self.last_day, 1)
		self.last_day = day
		for counts in self.counts.values():
			for shade in list(counts):
				counts[shade] *= factor
				if counts[shade] < self.forget_below:
					del counts[shade]
		for shade in hand:
			counts = self.counts[colour(shade)]
			counts[shade] = counts.get(shade, 0.0) + 1.0

	def same_colour(self, shade: int) -> dict[int, float]:
		return self.counts[colour(shade)]

	def total(self) -> float:
		return sum(self.counts['w'].values()) + sum(self.counts['b'].values())


class SoloLedger:
	# living alone we can keep track of the whole drawer: shade -> how many.
	# holes are the one thing we can't see, so a worn out sock we wear comes back
	# counted as 0.75 of a sock and the other 0.25 waits for a new pack

	def __init__(self, capacity: int) -> None:
		half = float(capacity // 2)
		self.drawer: dict[int, float] = {WHITE_NEW: half, BLACK_NEW: half}
		self.owed = {'w': 0.0, 'b': 0.0}
		self.spent_seen = 0.0
		self.returning: list[tuple[int, float]] | None = None
		self.lost_today: set[str] = set()

	def morning(self, hand: tuple[int, ...], turn: TurnContext) -> None:
		# last night's socks come back, any new packs go in, today's hand comes out
		for shade, weight in self.returning or ():
			self.drawer[shade] = self.drawer.get(shade, 0.0) + weight
		packs = round((turn.total_spent - self.spent_seen) / PACK_COST)
		self.spent_seen = turn.total_spent
		for _ in range(packs):
			# a pack can only be for a colour we lost yesterday; if both, go by
			# whose count is closer to 6
			lost = [c for c in self.owed if c in self.lost_today] or list(self.owed)
			c = max(lost, key=lambda k: self.owed[k])
			self.owed[c] = max(0.0, self.owed[c] - 6)
			fresh = WHITE_NEW if c == 'w' else BLACK_NEW
			self.drawer[fresh] = self.drawer.get(fresh, 0.0) + 6
		for shade in hand:
			rest = self.drawer.get(shade, 0.0) - 1
			if rest > 0.01:
				self.drawer[shade] = rest
			else:
				self.drawer.pop(shade, None)

	def evening(self, hand: tuple[int, ...], wear: tuple[int, int], toss: list[int]) -> None:
		self.returning = []
		for i, shade in enumerate(hand):
			if i in wear:
				share = 0.75 if is_done(shade) else 1.0
				self.returning.append((after_wash(shade), share))
				self.owed[colour(shade)] += 1.0 - share
			elif i in toss:
				self.owed[colour(shade)] += 1
			else:
				self.returning.append((shade, 1.0))
		self.lost_today = {
			colour(hand[i])
			for i in range(len(hand))
			if i in toss or (i in wear and is_done(hand[i]))
		}

	def same_colour(self, shade: int) -> dict[int, float]:
		white = is_white(shade)
		return {s: w for s, w in self.drawer.items() if is_white(s) == white}

	def total(self) -> float:
		return sum(self.drawer.values())


# ----------------------------------------------------------------------
# money
# ----------------------------------------------------------------------


def hole_money(k: Knobs, roommates: int, days: int) -> float:
	# money to keep back for replacing holes over this many days
	return k.hole_reserve * k.hole_rate * roommates * days * SOCK_COST


def socks_per_day(k: Knobs, money: float, roommates: int, days: int) -> float:
	# new socks a day the house can afford after setting aside the hole money
	return (money - hole_money(k, roommates, days)) / days / SOCK_COST


@dataclass
class Wallet:
	# where the house's money stands this morning

	have_money: bool = True
	lavish: bool = True
	running_dry: bool = False
	# wears each sock should last when spending evenly (None: not pacing)
	share: float | None = None


# ----------------------------------------------------------------------
# the player
# ----------------------------------------------------------------------


class Player7(BasePlayer):
	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)
		self.regime: str | None = None
		self.knobs = TUNED
		self.memory = ShadeMemory(TUNED.memory_fade, 0.01)
		self.solo = SoloLedger(self.capacity) if self.roommates == 1 else None
		# ruled modes
		self.broke_since: int | None = None
		self.swap_credit = 0.0
		# priced mode
		self.bank = 0.0
		self.my_spend = 0.0
		self.target_wears = 64.0
		self.trust_target = 0.0

	def select_socks(self, offered: tuple[int, ...], turn: TurnContext) -> Selection:
		if self.regime is None:
			self.settle_regime(turn)
		if self.regime == 'priced':
			return self.priced_turn(offered, turn)
		return self.ruled_turn(offered, turn)

	def settle_regime(self, turn: TurnContext) -> None:
		# decided once, from the whole budget over the whole run
		k = TUNED
		budget = turn.total_spent + turn.budget_remaining
		if budget != float('inf'):
			rate = socks_per_day(k, budget, self.roommates, self.days)
			if rate <= 0 or 2 * self.roommates / rate > k.priced_min_wears:
				self.regime = 'priced'
				self.memory = ShadeMemory(k.priced_memory_fade, 1e-3)
				return
		self.regime = 'ruled'
		tight = self.capacity / (self.selection_unit * self.roommates) < k.tight_drawer
		thin = budget / (self.roommates * self.days) < k.thin_dollars
		self.knobs = TUNED if tight or thin else ROOMY
		self.memory = ShadeMemory(k.memory_fade, 0.01)

	# ------------------------------------------------------------------
	# lavish / paced: fixed rules
	# ------------------------------------------------------------------

	def ruled_turn(self, hand: tuple[int, ...], turn: TurnContext) -> Selection:
		self.memory.observe(hand, turn.day)
		if self.solo:
			self.solo.morning(hand, turn)

		wallet, k = self.check_wallet(turn)
		wear = self.closest_pair(hand, avoid_done=wallet.running_dry, k=k)
		rest = [i for i in range(len(hand)) if i not in wear]

		share = wallet.share
		spending = wallet.have_money and not wallet.running_dry and not wallet.lavish
		if share is not None and spending:
			toss = [i for i in rest if wears(hand[i]) >= max(share, k.min_toss_wears)]
			return self.finish(hand, wear, toss)

		toss = self.rule_tosses(hand, wear, rest, wallet, k)
		if wallet.have_money and not wallet.lavish and not wallet.running_dry:
			toss += self.swaps(hand, rest, toss, k)
		return self.finish(hand, wear, toss)

	def check_wallet(self, turn: TurnContext) -> tuple[Wallet, Knobs]:
		k, n = self.knobs, self.roommates
		days_left = max(self.days - turn.day + 1, 1)
		left = turn.budget_remaining
		if left == float('inf'):
			return Wallet(), k

		wallet = Wallet()
		rate = socks_per_day(TUNED, left, n, days_left)
		if rate > 0 and 2 * n / rate <= TUNED.paced_max_wears:
			wallet.share = 2 * n / rate
			k = TUNED

		wallet.have_money = left >= PACK_COST
		if not wallet.have_money and self.broke_since is None:
			self.broke_since = turn.day
		wallet.lavish = wallet.have_money and left / days_left >= k.lavish_per_roommate * n
		if wallet.lavish and k.pace_guard and wallet.share is None and turn.day > 20:
			# the house's spending pace so far, kept up to the end, must still
			# leave the money we need for holes
			pace = turn.total_spent / turn.day
			wallet.lavish = pace * days_left <= left - hole_money(k, n, days_left)
		wallet.running_dry = self.running_dry(turn, left if wallet.have_money else 0.0, days_left)

		# whatever isn't needed for holes is spare: spread it over the days
		# left and split it between roommates so copies of us don't each
		# spend all of it
		spare = left - hole_money(k, n, days_left)
		if spare > 0 and not wallet.running_dry:
			self.swap_credit = min(self.swap_credit + spare / days_left / n / SOCK_COST, 3.0)
		return wallet, k

	def running_dry(self, turn: TurnContext, left: float, days_left: int) -> bool:
		# will holes from here on eat the spare socks plus what money can replace?
		k, n = TUNED, self.roommates
		lost = 0.0
		if self.broke_since is not None:
			lost = k.hole_rate * n * (turn.day - self.broke_since)
		spare = self.capacity - lost - self.selection_unit * n - k.dry_margin
		return k.hole_rate * n * days_left > spare + left / SOCK_COST

	@staticmethod
	def closest_pair(hand: tuple[int, ...], avoid_done: bool, k: Knobs) -> tuple[int, int]:
		# cheapest pair, fresher if tied; when running dry every hole is a
		# sock we never get back, so stay off worn out socks
		penalty = k.dry_penalty if avoid_done else 0.0

		def score(pair: tuple[int, int]) -> tuple[float, int]:
			a, b = hand[pair[0]], hand[pair[1]]
			return (charge(a, b) + penalty * (is_done(a) + is_done(b)), wears(a) + wears(b))

		return min(combinations(range(len(hand)), 2), key=score)

	@staticmethod
	def rule_tosses(
		hand: tuple[int, ...], wear: tuple[int, int], rest: list[int], wallet: Wallet, k: Knobs
	) -> list[int]:
		worn_shade = (hand[wear[0]] + hand[wear[1]]) / 2
		toss = []
		for i in rest:
			shade = hand[i]
			if is_done(shade):
				# replace worn out socks while we can, unless every sock counts
				# or money is tight (they still match the other old ones)
				keep = wallet.running_dry or (not wallet.lavish and k.keep_worn_when_tight)
				if wallet.have_money and not keep:
					toss.append(i)
			elif wallet.lavish and shade != worn_shade and wears(shade) >= k.min_toss_wears:
				# plenty of money: drop the odd ones out
				toss.append(i)
		return toss

	def swaps(self, hand: tuple[int, ...], rest: list[int], toss: list[int], k: Knobs) -> list[int]:
		# tight money: spend banked credit on the mid-aged socks that fit in
		# worst, worst first
		lo, hi = k.swap_wears
		picks = [i for i in rest if i not in toss and lo <= wears(hand[i]) <= hi]
		picks.sort(key=lambda i: -self.swap_gain(hand[i]))
		chosen = []
		for i in picks[: k.swaps_per_turn]:
			if self.swap_credit < 1.0 or self.swap_gain(hand[i]) < k.swap_min_gain:
				break
			chosen.append(i)
			self.swap_credit -= 1.0
		return chosen

	def swap_gain(self, shade: int) -> float:
		# how much more often a brand new sock would find a free match
		return self.match_chance(brand_new(shade)) - self.match_chance(shade)

	def match_chance(self, shade: int) -> float:
		# chance another sock in the hand is within the free gap of this one
		source = self.solo or self.memory
		total = source.total()
		if total == 0:
			return 1.0
		near = sum(w for s, w in source.same_colour(shade).items() if abs(s - shade) <= FREE_GAP)
		return 1.0 - (1.0 - near / total) ** (self.selection_unit - 1)

	def finish(self, hand: tuple[int, ...], wear: tuple[int, int], toss: list[int]) -> Selection:
		if self.solo:
			self.solo.evening(hand, wear, toss)
		return Selection(wear=wear, discard=tuple(toss))

	# ------------------------------------------------------------------
	# priced: cheapest move wins
	# ------------------------------------------------------------------

	def priced_turn(self, hand: tuple[int, ...], turn: TurnContext) -> Selection:
		k, n = TUNED, self.roommates
		self.memory.observe(hand, turn.day)
		days_left = max(self.days - turn.day + 1, 1)
		left = turn.budget_remaining
		broke = left < PACK_COST

		tosses_allowed = self.fill_bank(turn, left, days_left) if not broke else 0

		# the drawer our budget is steering towards: ages spread evenly up to
		# the wear count we can afford, trusted more the faster we replace
		rate = max(0.0, socks_per_day(k, left, n, days_left)) + k.hole_rate * n
		self.target_wears = 2 * n / rate
		self.trust_target = min(1.0, rate * min(30, days_left) / self.capacity)

		# money is cheap while the bank is full, priceless once it's gone
		per_dollar = k.point_per_dollar * max(0.0, 1.0 - self.bank / k.bank_cap)
		if broke:
			per_dollar = 1000.0
		new_sock = per_dollar * SOCK_COST

		# wearing a worn out sock is a 25% chance of paying for a new one;
		# tossing pays for one but trades this sock's fit for a fresh one's
		wear_price = [0.25 * new_sock * is_done(s) for s in hand]
		toss_price = [
			new_sock + k.fit_weight * (self.misfit(brand_new(s)) - self.misfit(s)) for s in hand
		]

		def move(pair: tuple[int, int]) -> tuple[tuple[float, int, int], tuple[int, ...]]:
			i, j = pair
			worth_it = sorted(
				(toss_price[x], x) for x in range(len(hand)) if x not in pair and toss_price[x] < 0
			)[:tosses_allowed]
			price = charge(hand[i], hand[j]) + wear_price[i] + wear_price[j]
			price += sum(p for p, _ in worth_it)
			return (price, len(worth_it), wears(hand[i]) + wears(hand[j])), tuple(
				sorted(x for _, x in worth_it)
			)

		moves = [(pair, *move(pair)) for pair in combinations(range(len(hand)), 2)]
		wear, _, toss = min(moves, key=lambda m: m[1])
		self.bank -= len(toss)
		self.my_spend += len(toss) * SOCK_COST
		return Selection(wear=wear, discard=toss)

	def fill_bank(self, turn: TurnContext, left: float, days_left: int) -> int:
		# put our share of today's spare money in the bank and say how many tosses
		# we get today. what the house spent minus what we spent is how fast the
		# roommates spend - assume they keep that up, keep a reserve, rest is ours
		k = TUNED
		if days_left <= 15:
			return 0
		others = max(0.0, turn.total_spent - self.my_spend) / max(turn.day, 1)
		reserve = max(k.reserve_packs * PACK_COST, 0.03 * (turn.total_spent + left))
		slack = left - reserve - others * days_left
		if slack <= 0:
			self.bank = 0.0
			return 0
		self.bank = min(self.bank + slack / SOCK_COST / days_left, k.bank_cap)
		return min(k.priced_tosses_per_turn, int(self.bank))

	def misfit(self, shade: int) -> float:
		# how badly this shade would match a random sock of the same colour -
		# partly the drawer we see now, partly the one our spending is heading for
		seen = self.memory.same_colour(shade)
		weight = sum(seen.values())
		now = sum(w * charge(shade, s) for s, w in seen.items()) / weight if weight else 0.0
		top = min(64, max(1, int(self.target_wears)))
		new, step = (WHITE_NEW, -2) if is_white(shade) else (BLACK_NEW, 1)
		later = sum(charge(shade, new + step * a) for a in range(top + 1)) / (top + 1)
		return (1 - self.trust_target) * now + self.trust_target * later
