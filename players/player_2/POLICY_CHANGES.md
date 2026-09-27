# Group 2: replacement-aware discards

The wear rule is unchanged: choose a minimum-penalty pair; among zero-penalty
pairs, prefer the one whose aging leaves the projected same-colour shade
distributions tightest. Keep the last **5 projected returned shades per colour**.

For discards, keep the last **20 observed shades per colour**. After at least
**3 observations**, compare each leftover's average mismatch with the average
mismatch of a pristine replacement against those same observations. Mismatch
is zero for a shade gap at most 6, otherwise the full gap. Require improvement
**greater than 6**; rank qualifying leftovers by improvement. Allow up to **2
discards with four-sock hands**, or **1 with five-sock hands**.

## Budget checks

- Never voluntarily discard with less than $10 left.
- Estimate remaining wear from recent shades and `(capacity - 14) / 2` socks
  per colour, clamped at zero. Each new sock supplies 68 wears in expectation.
- Reserve money for estimated future wear-out replacements, plus **$60**.
- Require the fraction of budget remaining to exceed the fraction of time
  remaining (margin **0**, previously 0.05).
- Before each proposed discard, subtract its expected remaining wears from
  estimated supply and recalculate the reserve. Two discards are checked
  cumulatively. If one candidate is unaffordable, try the next candidate.

The inventory buffer is never allowed below 10: while replacements remain
funded, each colour can have at most five pending losses after replenishment.
The selected buffer of 14 is more conservative. This remains an approximation,
not a measurement of the drawer. Replacement waits for six losses of a colour;
our matching calculation evaluates the eventual pristine replacement.
These checks cannot protect shared money from other players' spending.

## Tuning and results

We searched 80 configurations of the original policy and 80 with the reserve
repairs. `experiment_config.json` records every range and the selected settings.
Each search screened on one seed, validated its top 10 on four different seeds,
and selected among the top three on ten more seeds. The objective was mean
official embarrassment, including 65,536 per sockless day, equally weighting
both budgets and the specified lineups. Both searches selected the same values:
observed/returned windows 20/5, minimum samples 3, gain 6, discard cap 2,
inventory buffer 14, six extra reserve packs, and pace margin 0. This is a
bounded search; window 5, buffer 14 and six safety packs are grid boundaries.

Final four-sock comparison: capacity 28, four players, 730 days, 30 **new** seeds
(9279000–9279029). Opponents are Groups 1,3,4,6,7,8,9,10 at class commit
`c9b1643cea07aa4b997af61c1c01788f02f5a8e0` (September 23), including their helpers.
Eight lineups repeat one group three times; eight mix groups at cyclic offsets
0,1,3 in that list. Our seat is `seed % 4`. Each cell averages our score over
480 games, not the household's combined score.

| Policy | $120 | $400 | Sockless games at $120 |
|---|---:|---:|---:|
| Initial class submission (windows 40/20) | 575,867.3 | 1,031.9 | 122/480 |
| Tuned original | 535,792.9 | 894.9 | 121/480 |
| Tuned with reserve repairs | 535,792.9 | 894.6 | 121/480 |

Tuning reduced the $400 mean by 13.3%; the repairs themselves were effectively
tied with the tuned original (paired difference -0.29, 95% seed-bootstrap
interval [-0.96, 0.29]). No $400 games went sockless. Most $120 failures involved
Group 9; they remain in every reported mean. There were no player faults.

Separate controlled discard-cap tests used seeds 9283000–9283009. With the
repaired policy at $400, caps 0/1/2 scored 1,127.2 / 896.7 / 904.4. Two did not
reliably beat one; the search-selected cap stays two without retuning on test
data. At $120 none of these variants voluntarily discarded.

Five-sock hands were tuned separately over caps 0–3, using capacity 32 and seeds
9281000–9281003. Both policy families selected cap 1. On new seeds 9282000–9282009,
the $400 mean fell from 188.1 with no discards to 130.8 with one allowed.
Small-budget tests ($0/$10/$30/$60/$90, seeds 9280000–9280004) confirmed zero
voluntary discards for both tuned policies; this does not guarantee survival.
Per-game evidence is in `results/{main,discards,five,small-budget}.csv`.

## Reproduce or run another experiment

Use Python 3.12+ and the simulator's dependencies, including NumPy. The saved
results used Python 3.13.13 and NumPy 2.5.2. Saved suites
require a complete checkout of the pinned class revision, so opponents' helper
imports and the engine match. Only our player and runner come from this branch.

```bash
git fetch upstream c9b1643cea07aa4b997af61c1c01788f02f5a8e0
git worktree add --detach /tmp/socks-class c9b1643cea07aa4b997af61c1c01788f02f5a8e0
uv run python players/player_2/run_experiments.py \
  --simulator-root /tmp/socks-class --suite main --workers 6 --output /tmp/socks-main
```

Other suites: `discards`, `five`, `small-budget`, `sweep` (both 80-configuration
searches), and `five-sweep`. Use a new output directory for each. The saved CSVs
retain the original study's per-game measurements; the consolidated runner was
checked against 82 complete saved games, 32,000 sequential actions, and all
160 generated configurations before publishing.

For arbitrary rosters, budgets and seeds, omit `--suite`:

```bash
uv run python players/player_2/run_experiments.py \
  --simulator-root /tmp/socks-class --profiles fixed_tuned \
  --rosters 2,7,9,10 2,r,r,r --budgets 120 400 --seeds 1 2 3 \
  --windows 10 20 40 --discard-limits 0 1 2 --workers 6 --output /tmp/comparison.csv
```

`--help` lists the other controls. Edit a copy of the config to change any of
the policy parameters. Each run also saves inputs and source hashes as JSON;
existing outputs are not overwritten. Focal seats default to Group 2 seats,
whose scores are averaged if there are several. The timeout stays at one second.

The older `comparison_results.csv` covers September 21 opponents and is historical.
Its original runner and documentation remain available at commit `7b4ba66`.
