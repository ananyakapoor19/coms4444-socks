"""Run configurable sock-policy comparisons; use --help for options."""

import argparse
import csv
import hashlib
import importlib
import random
import statistics
import subprocess
import sys
import types
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.engine import Engine  # noqa: E402
from players.player_2.player import Player2  # noqa: E402


@cache
def load_player(code, revision):
	if code in ('r', 'g'):
		module = 'players.' + ('random_player' if code == 'r' else 'greedy_player')
		classname = 'RandomPlayer' if code == 'r' else 'GreedyPlayer'
	elif code.isdigit():
		module, classname = f'players.player_{code}.player', f'Player{code}'
	else:
		raise ValueError(f'Unknown player code: {code}')
	path = module.replace('.', '/') + '.py'
	if revision and code not in ('2', 'r', 'g'):
		source = subprocess.check_output(['git', 'show', f'{revision}:{path}'], cwd=ROOT)
		loaded = types.ModuleType(f'experiment_player_{code}')
		loaded.__file__ = str(ROOT / path)
		loaded.__package__ = module.rpartition('.')[0]
		sys.modules[loaded.__name__] = loaded
		exec(compile(source, loaded.__file__, 'exec'), loaded.__dict__)
	else:
		loaded = importlib.import_module(module)
		source = (ROOT / path).read_bytes()
	return getattr(loaded, classname), source


@cache
def source_digest(roster, revision):
	paths = [
		'core/engine.py',
		'core/sandbox.py',
		'models/player.py',
		'models/sock.py',
		'players/player_2/player.py',
		'players/player_2/run_experiments.py',
	]
	parts = [(ROOT / path).read_bytes() for path in paths]
	parts.extend(load_player(code, revision)[1] for code in roster.split(','))
	return hashlib.sha256(b'\0'.join(parts)).hexdigest()


def run_game(job):
	digest = source_digest(job['roster'], job['opponent_revision'])
	codes = job['roster'].split(',')
	classes = [load_player(code, job['opponent_revision'])[0] for code in codes]

	class ConfiguredPlayer2(Player2):
		def __init__(self, snapshot, ctx):
			super().__init__(snapshot, ctx)
			self.raw_window_size = job['window']
			self.raw_history = {c: deque(maxlen=self.raw_window_size) for c in self.raw_history}
			self.max_discards = job['discard_limit']
			self.replacement_gain_threshold = job['gain_threshold']

	classes = [
		ConfiguredPlayer2 if code == '2' else cls for code, cls in zip(codes, classes, strict=True)
	]
	# RandomPlayer uses global random; Engine has its own independently seeded RNG.
	random.seed(job['seed'])
	try:
		import numpy as np

		np.random.seed(job['seed'])
	except ImportError:
		pass
	engine = Engine(
		players=classes,
		capacity=job['capacity'],
		selection_unit=job['hand_size'],
		days=job['days'],
		seed=job['seed'],
		budget=job['budget'],
		timeout=job['timeout'],
		keep_records=False,
	)
	result = engine.run()
	focal = [int(i) for i in job['focal_seats'].split(',')]
	return {
		**job,
		'embarrassment': statistics.mean(
			result['players'][i]['total_embarrassment'] for i in focal
		),
		'total_spent': result['total_spent'],
		'budget_remaining': result['budget_remaining'],
		'sockless_days': result['total_sockless_days'],
		'faults': len(result['faults']),
		'fault_details': ' | '.join(result['faults']),
		'budget_exhausted_on_day': result['budget_exhausted_on_day'],
		'source_sha256': digest,
		'python_version': sys.version.split()[0],
	}


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		'--rosters',
		nargs='+',
		default=['2,2,2,2'],
		help='Seat order: group numbers, r=random, g=greedy',
	)
	parser.add_argument('--windows', type=int, nargs='+', default=[40])
	parser.add_argument('--discard-limits', type=int, nargs='+', default=[1])
	parser.add_argument('--gain-threshold', type=float, default=6)
	parser.add_argument('--budgets', type=float, nargs='+', default=[120, 400])
	parser.add_argument('--capacity', type=int, default=28)
	parser.add_argument('--hand-size', type=int, default=4)
	parser.add_argument('--days', type=int, default=730)
	parser.add_argument(
		'--seeds', type=int, nargs='+', help='Explicit seeds; overrides seed-start and num-seeds'
	)
	parser.add_argument('--seed-start', type=int, default=4001)
	parser.add_argument('--num-seeds', type=int, default=30)
	parser.add_argument(
		'--focal-seats',
		type=int,
		nargs='+',
		help='Zero-based seats to score; default: Group 2 seats, or all if absent',
	)
	parser.add_argument(
		'--opponent-revision',
		default='',
		help='Optional git revision for numbered opponents other than Group 2',
	)
	parser.add_argument('--workers', type=int, default=1)
	parser.add_argument('--timeout', type=float, default=1)
	parser.add_argument(
		'--output',
		type=Path,
		required=True,
		help='New CSV path; existing files are never overwritten',
	)
	args = parser.parse_args()
	seeds = (
		args.seeds
		if args.seeds is not None
		else list(range(args.seed_start, args.seed_start + args.num_seeds))
	)
	if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0 or max(seeds) >= 2**32:
		parser.error('Use distinct seeds between 0 and 2**32-1.')
	if min(args.windows) < 10 or min(args.discard_limits) < 0:
		parser.error(
			'Windows must be >=10 (minimum observation count); discard limits must be >=0.'
		)
	if (
		args.workers < 1
		or args.days < 1
		or args.hand_size < 2
		or args.timeout <= 0
		or min(args.budgets) < 0
	):
		parser.error('Invalid workers, days, hand size, timeout, or budget.')
	if args.opponent_revision:
		args.opponent_revision = subprocess.check_output(
			['git', 'rev-parse', '--verify', args.opponent_revision + '^{commit}'],
			cwd=ROOT,
			text=True,
		).strip()
	jobs = []
	for roster in args.rosters:
		codes = roster.split(',')
		if args.capacity % 4 or args.capacity <= args.hand_size * len(codes) + 10:
			parser.error(
				f'Invalid capacity for roster {roster}; follow the simulator capacity constraint.'
			)
		for code in codes:
			load_player(code, args.opponent_revision)
		focal = (
			args.focal_seats
			if args.focal_seats is not None
			else ([i for i, c in enumerate(codes) if c == '2'] or list(range(len(codes))))
		)
		if len(set(focal)) != len(focal) or any(i < 0 or i >= len(codes) for i in focal):
			parser.error(f'Invalid focal seats for roster {roster}.')
		for window, limit, budget, seed in product(
			args.windows, args.discard_limits, args.budgets, seeds
		):
			jobs.append(
				dict(
					roster=roster,
					capacity=args.capacity,
					hand_size=args.hand_size,
					days=args.days,
					window=window,
					discard_limit=limit,
					gain_threshold=args.gain_threshold,
					budget=budget,
					seed=seed,
					focal_seats=','.join(map(str, focal)),
					opponent_revision=args.opponent_revision,
					timeout=args.timeout,
				)
			)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	groups = {}
	with (
		args.output.open('x', newline='') as output,
		ProcessPoolExecutor(max_workers=args.workers) as pool,
	):
		writer = None
		for count, row in enumerate(pool.map(run_game, jobs), 1):
			if writer is None:
				writer = csv.DictWriter(output, fieldnames=list(row), lineterminator='\n')
				writer.writeheader()
			writer.writerow(row)
			output.flush()
			key = (row['roster'], row['budget'], row['window'], row['discard_limit'])
			groups.setdefault(key, []).append(row['embarrassment'])
			if count % 100 == 0 or count == len(jobs):
				print(f'{count}/{len(jobs)} games', flush=True)
	for (roster, budget, window, limit), scores in groups.items():
		print(
			f'{roster} budget={budget:g} window={window} discards<={limit}: {statistics.mean(scores):.1f}'
		)


if __name__ == '__main__':
	main()
