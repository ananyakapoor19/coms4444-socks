# Project 1: Socks

Simulator for COMS W4444, Fall 2026.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```
uv run main.py
```

## CLI Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--player CODE COUNT` | `r 4` | Add COUNT roommates running player CODE. Repeatable. |
| `-C`, `--capacity` | `40` | Drawer capacity C. Must be a multiple of 4 and greater than `unit * n + 10`. |
| `--unit` | `4` | Socks drawn per roommate per day. Either 4 or 5. |
| `--days` | `360` | Days to simulate. One year is 360 days. |
| `--seed` | `4444` | Random seed, for repeatable runs. |
| `--timeout` | `1.0` | Per-call wall-clock budget for player code, in seconds. `0` disables. Relies on `SIGALRM`; a no-op on Windows. |
| `--log PATH` | `logs/socks-*.log` | Per-day debug log (offered, wear, discard, embarrass, faults, drawer). |
| `--no-log` | off | Do not write a debug log. |
| `--summary-only` | off | Drop each day's `DayRecord` once scored. Results are unchanged; long runs stop holding one record per day. |
| `--gui` | off | Launch the visualizer. Without it, results print as JSON. |

Player codes are discovered automatically: `r` for the random baseline, `g` for
the greedy baseline, and any `player_<k>` directory (code `k`, no cap on how
many groups).

```
uv run main.py --player g 2 --player 4 2 --days 720 --seed 7
```

## Visualizer

```
uv run main.py --gui --player g 2 --player r 2
```

The drawer is drawn as sock icons in each sock's true greyscale shade, so the
white-to-grey and black-to-grey drift is visible at a glance. Discarded socks
that have not yet been replaced sit in a trash can, and a player fault gets a
red banner across the top of the window.

| Key | Action |
| --- | --- |
| `SPACE` | Play / pause (plays whole days) |
| `RIGHT` or `S` | Next roommate. After the last one, next day. |
| `LEFT` or `A` | Previous roommate (same day only) |
| `UP` / `DOWN` | Faster / slower (1-64 days per second) |
| `Q` or `ESC` | Quit |

The **Skip to day** box under the header is how you jump: type a day number and press Enter. It cannot rewind.

Every `main.py` run writes a debug log under `logs/` (override with `--log PATH`, disable with `--no-log`). It lists each morning's handful, wear/discard, embarrassment, faults, and the drawer shades after replenishment, then a JSON snapshot of the totals.

## Writing a player

Every `4` below is just an example. Replace it with your own group number
everywhere: the directory name, the class name, and the `--player` code. If
you are group 7, you use `player_7`, `Player7`, and `--player 7`.

Copy the template directory, renaming it for your group number, and rename the
class inside it to match. Group 4 would run:

```
cp -r players/player_template players/player_4
```

and edit `players/player_4/player.py` so the class is named `Player4`. Your
group directory ends up as:

```
players/player_4/
  __init__.py     required - see below
  player.py       class Player4(BasePlayer)
```

Then run it with `--player 4 <count>`:

```
uv run main.py --player 4 2 --player g 2
```

`players/player_template/player.py` documents the full interface inline: what
you can see, what you cannot, what the shades mean, and what happens when your
code misbehaves. The minimum is:

```python
from models.player import Player as BasePlayer, Selection, TurnContext


class Player4(BasePlayer):
	def select_socks(self, offered: tuple[int, ...], turn: TurnContext) -> Selection:
		return Selection(wear=(0, 1), discard=())
```

Three things that will make your player fail to appear, none of which produce a
loud error:

- **A missing `__init__.py`.** Discovery uses `pkgutil.iter_modules`, which
  only reports directories that have one. Without it your group is not
  reported as broken, it is not seen at all. The template ships with one.
- **A directory name that is not `player_<digits>`.** `player_4` works,
  `player_four` and `group4` do not.
- **A class named anything but `Player<k>`.** `Player4` works;
  `MyGreatSockStrategy` gets reported as an `AttributeError` at startup.

Anything that fails to import is printed as a warning and skipped, so a broken
commit from your group never stops anyone else's run. Check for your code in
the startup warnings before assuming it ran.

## Submitting your player

**Fork this repository**, then open a pull request back into `main` — see
[CONTRIBUTING.md](CONTRIBUTING.md) for the exact steps and rules (you only
ever touch `players/player_4/`, never anyone else's folder or the simulator
core). As before, `4` is just an example — use your actual group number
throughout.

```
# on your fork
cp -r players/player_template players/player_4
# edit players/player_4/player.py  →  class Player4

git add players/player_4
git commit -m "Group 4: <what changed>"
git push
```

Open the PR against this repo's `main`, name it with your group number, and
one of us will review and merge it. Once merged, `players/player_4/` lives on
`main` alongside everyone else's — that's expected, not a leak: other groups
can see your merged code, the same way this course's simulators have always
worked. Logs under `logs/` are only on the machine that ran the sim; they are
not a submission.

`offered` holds `selection_unit` shade values in the range 0–255. Pick two
indices to wear, and list any of the remaining indices you want thrown out.
Anything you neither wear nor discard goes back in the drawer unworn.

`turn` gives you the day number, your own full embarrassment history, the total
spent so far, and the budget remaining. That is all you get. If you build a `TurnContext` for your
own tests, pass it keywords:
`TurnContext(day=1, total_spent=0.0, total_embarrassment=0.0)`. You cannot see which socks came from
where, you cannot track a sock between turns, and you do not know how many
socks have been discarded.

### What the simulator enforces

- **Time limit.** `select_socks` gets `--timeout` seconds. Exceed it and you
  forfeit the turn: the simulator wears the first two socks and discards
  nothing. The limit is enforced with `SIGALRM`, which is Unix-only and
  main-thread-only: on Windows there is no alarm to set, so `--timeout` is a
  no-op and a slow player runs to completion rather than being cut off. Your
  code still has to finish quickly, because the tournament is run on Unix
  where the limit is real. `--timeout 0` disables it everywhere.
- **Validity.** Malformed selections (repeated indices, out-of-range indices,
  discarding a sock you are also wearing) are rejected the same way. Every
  forfeit is recorded in the `faults` list in the results.
- **Isolation.** Exceptions in your code forfeit your turn only. That includes
  handing back something malformed: a `Selection` whose `wear` is not a pair of
  indices, or not a `Selection` at all, is reported as a fault and costs you
  the turn, not the run. A module that fails to import is reported at startup
  and skipped, so a broken commit from one group does not stop anyone else's.

## Rules implemented

White socks start at 255 and lose 2 shade per wear/wash, stopping at 127. Black
socks start at 0 and gain 1 per wear/wash, stopping at 64. A pair worn with a
shade difference of more than 6 adds that difference to the wearer's
embarrassment; a difference of exactly 6 is free.

Socks that have already reached 127 or 64 have a 25% chance of developing a
hole when worn, and are discarded immediately. Once six socks of a colour have
been discarded, a fresh six-pack of that colour is bought for $10; any surplus
carries over to the next pack. Multiple packs can be bought on the same day.

## Checks before you push

```
uv run ruff format
```
Auto-formats your code to match the project style.

```
uv run ruff check --fix
```
Lints your code and auto-fixes what it safely can.

```
uv run pytest
```
Runs the test suite.
