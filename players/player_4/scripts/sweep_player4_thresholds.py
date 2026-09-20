"""2D grid sweep over Player4's WHITE_MIN / BLACK_MAX thresholds.

Runs the engine directly (not via sweep.py, since that harness only varies
CLI-level knobs like capacity/unit/seed, not a player's internal parameters).
For each (white, black) cell, a throwaway subclass of Player4 is built with
those class attributes overridden, then run in a household alongside
GreedyPlayer opponents across several seeds. Cells run in a process pool
(threads would not parallelize this - it's CPU-bound pure Python and the GIL
serializes threads to a single core; a process pool is what sweep.py itself
uses for the same reason).

    uv run --with matplotlib,numpy players/player_4/scripts/sweep_player4_thresholds.py
"""

import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from core.engine import Engine
from players.player_4.player import Player4

WHITE_VALUES = list(range(220, 256, 5))  # 220..255, fine pass around the promising corner
BLACK_VALUES = list(range(0, 22, 2))  # 0..20
SEEDS = (4444, 4545, 4646, 4747, 4848)
DAYS = 360
CAPACITY = 40
SELECTION_UNIT = 4


def _variant(white: int, black: int) -> type:
	return type(
		f'Player4_{white}_{black}',
		(Player4,),
		{'WHITE_MIN': white, 'BLACK_MAX': black},
	)


def _run_cell(args: tuple[int, int]) -> dict:
	white, black = args
	variant = _variant(white, black)
	roster = [variant, variant, variant, variant]

	embarrassments = []
	spends = []
	for seed in SEEDS:
		engine = Engine(
			players=roster,
			capacity=CAPACITY,
			selection_unit=SELECTION_UNIT,
			days=DAYS,
			seed=seed,
			keep_records=False,
		)
		results = engine.run()
		embarrassments.append(
			statistics.fmean(p['mean_daily_embarrassment'] for p in results['players'])
		)
		spends.append(results['spend_per_year'])

	return {
		'white': white,
		'black': black,
		'mean_daily_embarrassment': statistics.fmean(embarrassments),
		'spend_per_year': statistics.fmean(spends),
	}


def main() -> None:
	cells = [(w, b) for w in WHITE_VALUES for b in BLACK_VALUES]
	print(
		f'{len(cells)} cells x {len(SEEDS)} seeds = {len(cells) * len(SEEDS)} runs', file=sys.stderr
	)

	results: list[dict] = []
	with ProcessPoolExecutor() as pool:
		futures = {pool.submit(_run_cell, cell): cell for cell in cells}
		for n, future in enumerate(as_completed(futures), 1):
			results.append(future.result())
			print(f'\r  {n}/{len(cells)} cells done', end='', file=sys.stderr, flush=True)
	print(file=sys.stderr)

	results.sort(key=lambda r: (r['white'], r['black']))

	import numpy as np

	emb_grid = np.array(
		[[r['mean_daily_embarrassment'] for r in results if r['white'] == w] for w in WHITE_VALUES]
	)
	spend_grid = np.array(
		[[r['spend_per_year'] for r in results if r['white'] == w] for w in WHITE_VALUES]
	)

	best = min(results, key=lambda r: r['mean_daily_embarrassment'])
	print(
		f'best: white={best["white"]} black={best["black"]} '
		f'embarrassment/day={best["mean_daily_embarrassment"]:.3f} '
		f'$/year={best["spend_per_year"]:.2f}'
	)

	out_dir = Path(__file__).resolve().parent.parent / 'results'
	out_dir.mkdir(exist_ok=True)

	import csv

	with (out_dir / 'player4_threshold_sweep_fine.csv').open('w', newline='') as fh:
		writer = csv.DictWriter(
			fh, fieldnames=['white', 'black', 'mean_daily_embarrassment', 'spend_per_year']
		)
		writer.writeheader()
		writer.writerows(results)

	_plot(emb_grid, spend_grid, out_dir, best)


def _plot(emb_grid, spend_grid, out_dir: Path, best: dict) -> None:
	import matplotlib

	matplotlib.use('Agg')
	import matplotlib.pyplot as plt

	fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

	for ax, grid, title, cmap in (
		(axes[0], emb_grid, 'Mean daily embarrassment', 'viridis'),
		(axes[1], spend_grid, '$/year', 'magma'),
	):
		im = ax.imshow(grid, origin='lower', aspect='auto', cmap=cmap)
		ax.set_xticks(range(len(BLACK_VALUES)))
		ax.set_xticklabels(BLACK_VALUES)
		ax.set_yticks(range(len(WHITE_VALUES)))
		ax.set_yticklabels(WHITE_VALUES)
		ax.set_xlabel('BLACK_MAX')
		ax.set_ylabel('WHITE_MIN')
		ax.set_title(title)
		fig.colorbar(im, ax=ax, shrink=0.8)

	best_x = BLACK_VALUES.index(best['black'])
	best_y = WHITE_VALUES.index(best['white'])
	for ax in axes:
		ax.scatter([best_x], [best_y], marker='*', s=250, color='red', edgecolor='white', zorder=5)

	fig.suptitle(
		f'Player4 threshold sweep (white={WHITE_VALUES[0]}..{WHITE_VALUES[-1]} step 5, '
		f'black={BLACK_VALUES[0]}..{BLACK_VALUES[-1]} step 2)'
	)
	fig.tight_layout()
	out_path = out_dir / 'player4_threshold_sweep_fine.png'
	fig.savefig(out_path, dpi=150)
	print(f'wrote: {out_path}')
	print(f'wrote: {out_dir / "player4_threshold_sweep_fine.csv"}')


if __name__ == '__main__':
	main()
