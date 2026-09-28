# Group 2 policy

The player first minimizes today's embarrassment. A shade gap of at most 6
costs zero; otherwise the cost is the full gap.

## Pair selection

Normally, if several pairs have zero embarrassment, the player simulates the
four offered socks returning to the drawer: worn socks return aged and the
other socks return unchanged. It chooses the pair that produces the smallest
sum of the projected black- and white-shade standard deviations. The projected
history keeps the latest 5 returned shades of each colour.

For four-sock hands with an initial budget of at least $400, the tie-breaker is
different. Using the latest 5 observed shades of each colour, it computes that
colour's mean and chooses the zero-embarrassment pair whose aging produces the
smallest change in squared distance from those means. This mode is fixed on the
first turn. If every pair has positive embarrassment, both modes simply choose
the minimum-embarrassment pair.

## Voluntary discards

For each leftover sock, the player compares its average matching embarrassment
against recently observed same-colour shades with the corresponding average
for a pristine replacement (black 0, white 255). Normally this history contains
20 observed shades per colour; the high-budget mode uses 5. After at least 3
observations, a leftover qualifies only when the replacement improves the
average by more than 6 points.

Qualifying leftovers are ranked by improvement. Four-sock hands may discard at
most 2 socks; five-sock hands may discard at most 1.

## Budget safeguards

- Never discard voluntarily with less than $10 remaining.
- Approximate the socks per colour as `max(1, (capacity - 14) / 2)`.
- Estimate their remaining useful wears from recent shades. A fresh sock
  supplies 68 expected wears.
- Reserve the estimated cost of future wear-out replacements plus $60.
- Require the fraction of budget remaining to exceed the fraction of game time
  remaining.
- Recalculate the reserve after each proposed discard, including the remaining
  useful life thrown away, so two discards must be affordable cumulatively.

The drawer count and remaining-life calculations are estimates because the
player cannot inspect the hidden drawer or observe other players' actions.

## Evidence

The 20/5 histories, minimum 3 samples, gain threshold 6, discard caps 2/1,
capacity buffer 14, six safety packs, and pace margin 0 came from a bounded
parameter search. The high-budget selection rule was then tested separately.

With capacity 28, four players, four-sock hands, 730 days, and a $400 budget,
the high-budget rule changed mean embarrassment from 155.50 to 24.43 in 30
self-play seeds and from 904.79 to 782.59 in 1,200 games against all 120
unordered three-opponent combinations with repetition from Groups
1, 3, 4, 6, 7, 8, 9, and 10. Neither comparison had sockless games or player
faults. Average finishing rank did not improve, and low-budget survival remains
an open weakness.

`results/high_budget_summary.csv` contains the aggregate comparison.
`run_experiments.py` and `experiment_config.json` provide the reusable runner;
raw per-game outputs are intentionally not stored in the submission repository.
