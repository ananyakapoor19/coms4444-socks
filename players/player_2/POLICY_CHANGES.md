# Group 2: replacement-aware discards

The idea is to ask whether replacing a sock would improve matching, rather than
discarding it simply because its shade is unusual.

## Which socks to wear

Keep the existing pair rule. If every pair has a penalty, wear the pair with the
smallest penalty. Among zero-penalty pairs, prefer the pair whose aging leaves
the projected same-colour shade distributions tightest.

## Which leftover to discard

Record the last 40 observed shades of each colour. For each leftover, compare:

- its average mismatch against recent socks of the same colour;
- a pristine sock's average mismatch against those same observations.

Mismatch is zero for a shade gap of at most 6, otherwise the full gap. A pristine
white sock has shade 255; a pristine black sock has shade 0. Discard at most one
leftover: the one whose replacement improves the average by more than 6 points.
Require at least 10 same-colour observations before using this estimate.

For example, if recent white socks are near 250, replacing a leftover at 200
with one at 255 should help. If recent white socks are near 200, keep it.

## Budget protection

Keep a reserve for estimated future hole replacements plus two safety packs.
Permit voluntary discards only when the fraction of budget remaining is more
than 0.05 ahead of the fraction of time remaining. Disable voluntary discards
when five or more socks are offered.

The reserve estimates available wear from recent observed shades, assumes about
C-4 socks split evenly between colours, and uses 68 expected wears per new sock.
These are approximations because the actual drawer and discard counters are
hidden. Replacement happens only after six household discards of a colour;
the matching estimate treats the eventual replacement as pristine and does not
predict the exact delay. It also averages individual pair costs rather than
predicting the best pair in a future hand. Budget protection is a heuristic,
not a survival guarantee against arbitrary roommate spending.

## Fixed-setting results

This submission uses a 40-observation window, improvement cutoff >6, and at most
one voluntary discard. Runs use 28 socks, 4 players, 4-sock hands, 730 days,
and seeds 4001–4030. Budgets are shared by the household.

| Household | $120 | $400 |
|---|---:|---:|
| Four copies of our policy | 1167.0 | 582.1 |
| Our policy + 3 RandomPlayers | 1777.9 | 1469.8 |
| Our policy + 3 Group 7 players | 1018.0 | 502.3 |
| Our policy + 3 Group 10 players | 1829.2 | 837.5 |
| Our policy + Group 7, Group 10, RandomPlayer | 1622.2 | 818.2 |

Scores are mean cumulative embarrassment over 30 seeds. Clone results average all
four players; mixed results report our player only. All 300 games had zero sockless
days and faults. These are absolute scores, not a comparison against the previous policy.

## Reproduce or run other configurations

From the repository root with Python 3.12+:

```bash
python players/player_2/run_experiments.py \
  --rosters 2,2,2,2 2,r,r,r 2,7,7,7 2,10,10,10 2,7,10,r \
  --windows 40 --discard-limits 1 --gain-threshold 6 --budgets 120 400 \
  --capacity 28 --hand-size 4 --days 730 --seed-start 4001 --num-seeds 30 \
  --opponent-revision 8ddbf1d --workers 6 --output /tmp/initial_policy_results.csv
```

The output path must be new. The CSV records individual game settings, scores,
spending, failures, source hashes and opponent revision. Player codes are group
numbers, `r` for RandomPlayer, and `g` for GreedyPlayer. Both RNGs are seeded.
Use `--help` to choose other game settings, rosters or policy parameters.

Group 7/10 results use the September 21 snapshot at `8ddbf1d`. Omit
`--opponent-revision` to use the current checkout. This option pins the opponents'
`player.py` files; helper imports and simulator code still use the checkout.
