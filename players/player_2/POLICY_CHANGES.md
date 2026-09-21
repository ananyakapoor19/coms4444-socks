# Player 2: distribution-aware selection policy

Implements the policy described in `plan.md` (2026-09-20). Everything lives in
`players/player_2/player.py`; the entry point is still `Player2.select_socks()`.

## What was broken before

The previous `select_socks()` forfeited **every single turn**. In a 60-day run
against two greedy players it scored 80-110 embarrassment per day versus about
0.5 for greedy, and the fault list was one `AttributeError` per turn.

1. **Wrong attribute names.** `__init__` created `self.local_white_history` and
   `self.local_black_history`, but `select_socks()` appended to
   `self.white_history` / `self.black_history`, which never existed. That was
   the `AttributeError` that forfeited the turn before any decision was made.
2. **`socks_to_wear` was never defined.** The block that assigned it was
   commented out, so even with the attribute names fixed the return statement
   would have raised `NameError`.
3. **Pairwise embarrassment was computed twice** (`pairwise_sock_pairs` and
   `pairs`) and the resulting `min_embarassment` was never used.
4. **Dead state.** `sock_distributions_per_day`, `embarassment_thresh`,
   `outlier_z` and `global_history` were initialised and never read.
5. **Wrong discard semantics.** The cooperative path discarded *every* white
   sock among the first four offered, including ones it might have been
   wearing, which would have been an invalid selection (wear and discard must
   be disjoint).
6. **Unused imports** (`defaultdict`, `mean`, `pstdev`) that `ruff check` in CI
   would have flagged.

## What is implemented now

### Window-size statistics (`WindowedStats`)

One tracker per colour, each over the last `running_window_size = 20` shades
of that colour.

- Until the window is full, `add()` is the plain Welford update of the running
  mean and M2 (sum of squared deviations).
- Once the history exceeds the window, `add()` first evicts the oldest value
  with the reverse Welford update and then folds in the new one, so mean and
  std always describe exactly the values in the window without re-summing.
- `std` is the population std, matching `statistics.pstdev`. Verified against
  `pstdev` over a sliding window for window sizes 1, 2, 5 and 20 on 300 random
  shades (max error under 1e-9).

Black and white are tracked separately because their shade ranges are disjoint
(0-64 vs 127-255). A single mixed distribution would measure the black/white
ratio of the drawer, not how well socks of the same colour match.

### Sock selection

1. `pairwise_sock_embarassments(offered)` scores every unordered pair. The min
   and mean over all pairs are logged per day (`offered_pair_min`,
   `offered_pair_mean`) as a proxy for how well-matched the drawer is.
2. If any pair has zero immediate embarrassment (shade gap <= 6), `choose_pair()`
   projects the next tracked distribution for each zero-cost pair: worn socks
   return aged and all leftovers return unchanged. It chooses the free pair
   with the smallest sum of black + white std, using raw gap/index only as
   deterministic tie-breaks.
3. Only when every pair has positive embarrassment does `choose_pair()` wear
   the minimum-embarrassment pair.
4. The remaining indices are the leftovers that the discard policy branches
   from.

### Sock discard policy

`choose_discards()` enumerates every subset of the leftovers (4 actions with a
selection unit of 4, 8 with a unit of 5). For each action it copies both
trackers, applies the day's outcome and measures the sum of the two stds:

- worn socks are added back at their **aged** shade (white -2, black +1,
  clamped at 127 / 64), because that is what returns to the drawer;
- returned leftovers are added at their current shade;
- discarded leftovers are not added.

The action with the lowest total std wins; ties go to the action with fewer
discards. The chosen action is then committed to the real trackers with the
same `apply_action()` helper, so what we evaluate is exactly what we record.

Two guards were kept from the original code's intent:

- **Sample floor.** A leftover is only eligible for discard once its colour has
  at least `min_dist_samples = 10` values in the window.
- **Cooperative budget guard** (`can_discard()`). No discards when the
  household cannot afford a six-pack or is spending above its average pace
  (`total_spent / day > initial_budget / days`). On an unlimited run both
  thresholds are `inf`, so this is always true.

## Assumptions made where the plan was ambiguous

- "Global distribution metrics" is tracked **per colour**, for the reason
  above.
- The std minimised is the **sum of the two per-colour stds**. Only the colours
  touched by an action change, so this is equivalent to comparing the affected
  colour alone.
- Worn socks enter the distribution at their post-wear shade. The 25% hole
  chance on worn-out socks is ignored because we cannot observe it.
- The budget guard from the previous implementation was kept even though the
  plan does not mention spending. Without it the policy discards freely on an
  unlimited run (see results).

## Results (360 days, capacity 40, unit 4, 4 identical roommates)

Unlimited budget:

| Player  | Seed | Household spend | Household embarrassment |
| ------- | ---- | --------------- | ----------------------- |
| Player2 | 1    | $2740           | 0                       |
| Player2 | 2    | $2840           | 0                       |
| Player2 | 3    | $2830           | 0                       |
| Greedy  | 1    | $2220           | 92                      |
| Greedy  | 2    | $2260           | 203                     |
| Greedy  | 3    | $2250           | 147                     |
| Random  | 1-3  | $40             | ~127,000                |

With a finite budget (seed 1):

| Player  | Budget | Exhausted on day | Sockless days | Household embarrassment |
| ------- | ------ | ---------------- | ------------- | ----------------------- |
| Player2 | $500   | never            | 0             | 2,229                   |
| Greedy  | $500   | 279              | 5             | 332,388                 |
| Player2 | $1000  | never            | 0             | 357                     |
| Greedy  | $1000  | never            | 0             | 1,814                   |

`uv run ruff format`, `uv run ruff check` and `uv run pytest` (236 tests) all
pass.

## Known trade-off and knobs

Minimising std keeps a returned sock only when it sits within roughly one std
of its colour's window mean, so on an unlimited budget the policy discards a
large share of leftovers and spends about 25% more than greedy. The budget
guard is what makes it behave well under a real budget. If spend on unlimited
runs matters, the natural knobs are:

- `running_window_size` (20) and `min_dist_samples` (10);
- requiring a minimum std *improvement* before a discard is worth a sixth of a
  $10 pack, rather than any improvement at all;
- an explicit z-score cutoff (the old `outlier_z = 1.5`) instead of raw
  min-std.
