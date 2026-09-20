"""Sparse randomized sweep over CLI-tunable constants (capacity, days, budget),
picking the embarrassment-minimizing (WHITE_MIN, BLACK_MAX) for each combo,
then fitting a linear regression: threshold ~ capacity + days + budget.

Household is 4x Player4 (all four seats run the same threshold variant).

Note: the engine hard-caps six-pack purchases at what's left of the budget
(core/engine.py, `affordable = min(packs, int(self.budget_remaining //
PACK_COST))`), so total_spent can never exceed budget - that "constraint" is
automatic. What a too-aggressive discard threshold actually risks is running
the budget dry and going sockless (a 65536-point embarrassment penalty), so
the per-combo selection is a plain argmin over total_embarrassment.

Per-combo optimization: a brute-force grid (625 cells) is wasteful when the
embarrassment landscape is smooth and roughly unimodal in (white, black), as
the earlier heatmaps showed. Each combo instead runs a compass/pattern search
(Hooke-Jeeves): start at the domain centre, probe +/-step on each axis, jump
to any improving neighbour, halve the step when no neighbour improves, stop
once both steps are below 1. That is ~20-30 evaluations per combo instead of
625 (a ~25x cut), which is spent instead on covering far more combos.

    uv run --with numpy,matplotlib python players/player_4/scripts/sweep_cli_regression.py
"""

import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from core.engine import Engine
from players.player_4.player import Player4

SEED = 4444
UNIT = 4
ROOMMATES = 4
FLOOR = UNIT * ROOMMATES + 10  # 26, from Engine's own validity check

WHITE_DOMAIN = (65, 255)  # white shades only ever live above 64
BLACK_DOMAIN = (0, 64)
MAX_EVALS = 60  # safety cap; compass search normally stops well before this

META_SEED = 20260914
N_COMBOS = 400

CAPACITY_CHOICES = list(range(28, 121, 4))  # multiples of 4, all > FLOOR
DAYS_RANGE = (60, 720)
BUDGET_PER_DAY_RANGE = (0.3, 2.0)  # dollars/day multiplier sampled per combo


def _variant(white: int, black: int) -> type:
	return type(
		f'Player4_{white}_{black}',
		(Player4,),
		{'WHITE_MIN': white, 'BLACK_MAX': black},
	)


def _evaluate(
	capacity: int, days: int, budget: float, white: int, black: int
) -> tuple[float, float]:
	variant = _variant(white, black)
	engine = Engine(
		players=[variant] * ROOMMATES,
		capacity=capacity,
		selection_unit=UNIT,
		days=days,
		seed=SEED,
		budget=budget,
		keep_records=False,
	)
	results = engine.run()
	return results['total_embarrassment'], results['total_spent']


def _clamp(v: int, lo: int, hi: int) -> int:
	return max(lo, min(hi, v))


def _pattern_search(combo: dict) -> dict:
	"""Hooke-Jeeves compass search over (white, black) for one combo."""
	cache: dict[tuple[int, int], tuple[float, float]] = {}

	def ev(w: float, b: float) -> tuple[tuple[int, int], tuple[float, float]]:
		w = _clamp(round(w), *WHITE_DOMAIN)
		b = _clamp(round(b), *BLACK_DOMAIN)
		key = (w, b)
		if key not in cache:
			cache[key] = _evaluate(combo['capacity'], combo['days'], combo['budget'], w, b)
		return key, cache[key]

	white = (WHITE_DOMAIN[0] + WHITE_DOMAIN[1]) // 2
	black = (BLACK_DOMAIN[0] + BLACK_DOMAIN[1]) // 2
	w_step = (WHITE_DOMAIN[1] - WHITE_DOMAIN[0]) // 4
	b_step = (BLACK_DOMAIN[1] - BLACK_DOMAIN[0]) // 4

	(white, black), (best_emb, best_spend) = ev(white, black)

	while (w_step >= 1 or b_step >= 1) and len(cache) < MAX_EVALS:
		candidates = []
		if w_step >= 1:
			candidates += [(white + w_step, black), (white - w_step, black)]
		if b_step >= 1:
			candidates += [(white, black + b_step), (white, black - b_step)]

		improved = False
		for cw, cb in candidates:
			(kw, kb), (emb, spend) = ev(cw, cb)
			if emb < best_emb:
				white, black, best_emb, best_spend = kw, kb, emb, spend
				improved = True
			if len(cache) >= MAX_EVALS:
				break

		if not improved:
			w_step //= 2
			b_step //= 2

	return {
		'capacity': combo['capacity'],
		'days': combo['days'],
		'budget': combo['budget'],
		'white': white,
		'black': black,
		'embarrassment': best_emb,
		'spend': best_spend,
		'evals': len(cache),
	}


def sample_combos(n: int) -> list[dict]:
	rng = random.Random(META_SEED)
	combos = []
	for _ in range(n):
		capacity = rng.choice(CAPACITY_CHOICES)
		days = rng.randint(*DAYS_RANGE)
		factor = rng.uniform(*BUDGET_PER_DAY_RANGE)
		combos.append({'capacity': capacity, 'days': days, 'budget': round(days * factor, 2)})
	return combos


def main() -> None:
	combos = sample_combos(N_COMBOS)
	print(f'{len(combos)} combos, compass search (<= {MAX_EVALS} evals each)', file=sys.stderr)

	dataset = []
	with ProcessPoolExecutor() as pool:
		futures = {pool.submit(_pattern_search, combo): combo for combo in combos}
		for n, future in enumerate(as_completed(futures), 1):
			dataset.append(future.result())
			if n % 20 == 0 or n == len(combos):
				print(f'\r  {n}/{len(combos)} combos done', end='', file=sys.stderr, flush=True)
	print(file=sys.stderr)

	import statistics as stats

	total_evals = sum(d['evals'] for d in dataset)
	print(
		f'total evals: {total_evals} ({stats.fmean(d["evals"] for d in dataset):.1f} avg/combo, '
		f'vs 625 for a full grid)',
		file=sys.stderr,
	)

	import csv

	out_dir = Path(__file__).resolve().parent.parent / 'results'
	out_dir.mkdir(exist_ok=True)
	csv_path = out_dir / 'cli_param_threshold_dataset.csv'
	with csv_path.open('w', newline='') as fh:
		writer = csv.DictWriter(
			fh,
			fieldnames=[
				'capacity',
				'days',
				'budget',
				'white',
				'black',
				'embarrassment',
				'spend',
				'evals',
			],
		)
		writer.writeheader()
		writer.writerows(dataset)
	print(f'wrote: {csv_path}')

	_regress_and_plot(dataset, out_dir)


def _fit(X, y):
	import numpy as np

	coef, *_ = np.linalg.lstsq(X, y, rcond=None)
	pred = X @ coef
	ss_res = float(np.sum((y - pred) ** 2))
	ss_tot = float(np.sum((y - y.mean()) ** 2))
	r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
	return coef, r2, pred


def _regress_and_plot(dataset: list[dict], out_dir: Path) -> None:
	import numpy as np

	X = np.array([[1.0, d['capacity'], d['days'], d['budget']] for d in dataset])
	y_white = np.array([d['white'] for d in dataset], dtype=float)
	y_black = np.array([d['black'] for d in dataset], dtype=float)

	white_coef, white_r2, white_pred = _fit(X, y_white)
	black_coef, black_r2, black_pred = _fit(X, y_black)

	names = ('intercept', 'capacity', 'days', 'budget')
	print('\nWHITE_MIN ~ intercept + capacity + days + budget')
	for name, c in zip(names, white_coef, strict=True):
		print(f'  {name:<10} {c:+.5f}')
	print(f'  R^2 = {white_r2:.3f}')

	print('\nBLACK_MAX ~ intercept + capacity + days + budget')
	for name, c in zip(names, black_coef, strict=True):
		print(f'  {name:<10} {c:+.5f}')
	print(f'  R^2 = {black_r2:.3f}')

	import matplotlib

	matplotlib.use('Agg')
	import matplotlib.pyplot as plt

	fig, axes = plt.subplots(2, 2, figsize=(11, 9))

	for ax, actual, pred, r2, title in (
		(axes[0, 0], y_white, white_pred, white_r2, 'WHITE_MIN'),
		(axes[0, 1], y_black, black_pred, black_r2, 'BLACK_MAX'),
	):
		ax.scatter(actual, pred)
		lims = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
		ax.plot(lims, lims, 'r--', linewidth=1)
		ax.set_xlabel(f'actual {title}')
		ax.set_ylabel(f'predicted {title}')
		ax.set_title(f'{title} fit (R2={r2:.2f})')

	budgets = np.array([d['budget'] for d in dataset])
	days_ = np.array([d['days'] for d in dataset])

	sc = axes[1, 0].scatter(budgets, y_white, c=days_, cmap='viridis')
	axes[1, 0].set_xlabel('budget')
	axes[1, 0].set_ylabel('chosen WHITE_MIN')
	axes[1, 0].set_title('WHITE_MIN vs budget (color = days)')
	fig.colorbar(sc, ax=axes[1, 0], shrink=0.8)

	sc2 = axes[1, 1].scatter(budgets, y_black, c=days_, cmap='viridis')
	axes[1, 1].set_xlabel('budget')
	axes[1, 1].set_ylabel('chosen BLACK_MAX')
	axes[1, 1].set_title('BLACK_MAX vs budget (color = days)')
	fig.colorbar(sc2, ax=axes[1, 1], shrink=0.8)

	fig.tight_layout()
	png_path = out_dir / 'cli_param_threshold_regression.png'
	fig.savefig(png_path, dpi=150)
	print(f'wrote: {png_path}')


if __name__ == '__main__':
	main()
