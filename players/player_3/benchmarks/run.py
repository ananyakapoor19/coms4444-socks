"""Paired comparison: python -m players.player_3.benchmarks.run.

Each seed uses four independent copies of one policy and the same engine setup.
Identical seeds do not imply identical random paths once policy actions differ.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean

from core.engine import Engine
from players.player_3.benchmarks.original import Player3 as Original3
from players.player_3.benchmarks.terminal_only import TerminalOnly3
from players.player_3.player import Player3

CONFIGURATION = dict(capacity=40, selection_unit=4, days=720, budget=120.0, timeout=1.0)
STRATEGIES = dict(original=Original3, future_aging=Player3, terminal_only=TerminalOnly3)
ORIGINAL_REVISION = '8fd0f247f6de1023fbede573b1d511d402e694fc'


def run(job: tuple[str, int]) -> dict:
	variant, seed = job
	engine = Engine(
		players=[STRATEGIES[variant]] * 4, seed=seed, keep_records=False, **CONFIGURATION
	)
	result = engine.run()
	return dict(
		variant=variant,
		seed=seed,
		embarrassment=result['total_embarrassment'],
		spent=result['total_spent'],
		sockless=result['total_sockless_days'],
		faults=len(result['faults']),
		fault_details=result['faults'],
	)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--seeds', type=int, nargs='+', default=list(range(1000, 1030)))
	parser.add_argument('--workers', type=int, default=4)
	parser.add_argument('--output', type=Path, default=Path(__file__).with_name('results.json'))
	args = parser.parse_args()
	jobs = [(variant, seed) for seed in args.seeds for variant in STRATEGIES]
	rows = []
	with ProcessPoolExecutor(max_workers=args.workers) as pool:
		for row in pool.map(run, jobs):
			rows.append(row)
			if len(rows) % 3 == 0:
				print(f'Completed {len(rows)}/{len(jobs)} runs', flush=True)
	by_seed = {(row['variant'], row['seed']): row for row in rows}
	summary = {}
	for variant in STRATEGIES:
		selected = [row for row in rows if row['variant'] == variant]
		summary[variant] = {
			key: mean(row[key] for row in selected)
			for key in ('embarrassment', 'spent', 'sockless', 'faults')
		}
		summary[variant]['wins_vs_original'] = sum(
			row['embarrassment'] < by_seed['original', row['seed']]['embarrassment']
			for row in selected
		)
	args.output.write_text(
		json.dumps(
			dict(
				original_revision=ORIGINAL_REVISION,
				configuration=dict(players=4, **CONFIGURATION),
				seeds=args.seeds,
				summary=summary,
				runs=rows,
			),
			indent=2,
		)
		+ '\n'
	)
	print(json.dumps(summary, indent=2))


if __name__ == '__main__':
	main()
