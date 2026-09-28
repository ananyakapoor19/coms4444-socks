# Group 3 benchmarks

`Player3` ages the candidate and observed drawer together when estimating future
mismatch. The geometric replacement target stays stationary. Scoring ends after
`max(0, days - day) * 2 * roommates / capacity` expected wears. The budget
controller and recency histogram are unchanged.

- `original.py`: frozen player from commit `8fd0f247f6de1023fbede573b1d511d402e694fc`.
- `terminal_only.py`: original pair selection, with terminal leftovers discarded
  whenever a $10 pack is affordable. This benchmark bypasses discard credit and
  spend bookkeeping.
- `run.py`: runs all three policies on identical seeds.

## Results

Four copies of each policy, $120 shared budget, 720 days, capacity 40, selection
unit 4, seeds 1000–1029, and a one-second call timeout. Run on 2026-09-22 with
Python 3.13.2. These seeds repeat the earlier experiment; they are not a new
validation set. Random paths can diverge once policies choose different actions.

Means per household run:

| Policy | Embarrassment | Reduction | Spending | Sockless player-days | Faults |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 15,298.23 | — | $119.33 | 0 | 0 |
| Future aging | 7,805.67 | 49.0% | $102.00 | 0 | 0 |
| Terminal-only | 4,310.40 | 71.8% | $105.33 | 0 | 0 |

Both variants beat the original on all 30 seeds. All 90 runs match the earlier
experiment's scores, spending, sockless counts, and faults.
Per-seed data: [results.json](results.json). The initial original-only runs on
seeds 0–9 are in [original_benchmark.json](../original_benchmark.json).

## Run

From the repository root:

```sh
uv run python -m players.player_3.benchmarks.run
uv run ruff format players/player_3
uv run ruff check players/player_3
uv run pytest
```

Ruff checks passed; 285 tests passed, including 49 Group 3 cases. The projection
also matched the NumPy reference across 243 scenarios, both colors, and all 65
ages, with a maximum absolute difference of `9.24e-14`.

The model assumes equal expected aging and does not simulate selective wear,
individual holes, or pack arrivals. Other budgets and mixed rosters have not
been evaluated here.
