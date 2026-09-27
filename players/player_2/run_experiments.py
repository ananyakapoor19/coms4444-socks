"""Run policy comparisons or a parameter sweep against a complete simulator checkout."""

import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import random
import statistics
import subprocess
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import cache
from itertools import product
from math import ceil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / 'experiment_config.json'


@cache
def runtime(root):
	# Import the engine, opponents, and their helpers from ONE complete checkout.
	# Load our submitted policy separately, so that checkout's old Player2 is unused.
	sys.path.insert(0, root)
	engine = importlib.import_module('core.engine').Engine
	spec = importlib.util.spec_from_file_location('experiment_group2', HERE / 'player.py')
	module = importlib.util.module_from_spec(spec)
	sys.modules[spec.name] = module
	spec.loader.exec_module(module)
	return engine, module


@cache
def player_class(root, family, settings_json):
	_, policy = runtime(root)
	settings = json.loads(settings_json)

	class ConfiguredPlayer2(policy.Player2):
		def __init__(self, snapshot, ctx):
			super().__init__(snapshot, ctx)
			p = settings
			self.raw_window_size = p['raw_window']
			self.raw_history = {c: deque(maxlen=p['raw_window']) for c in self.raw_history}
			self.running_window_size = p['return_window']
			self.stats = {c: policy.WindowedStats(p['return_window']) for c in self.stats}
			self.min_dist_samples = p['min_samples']
			self.max_discards = p['discard_limit']
			self.replacement_gain_threshold = p['gain']
			self.reserve_capacity_buffer = p['inventory_buffer']
			self.reserve_safety_packs = p['safety_packs']
			self.budget_pace_margin = p['pace_margin']
			self.voluntary_discards = 0
			self.two_discard_turns = 0

		def expected_replacement_reserve(self, turn, lost_wears=0.0):
			if family == 'fixed':
				return super().expected_replacement_reserve(turn, lost_wears)
			count = max(1.0, (self.capacity - self.reserve_capacity_buffer) / 2)
			services = 0.0
			for colour in (policy.WHITE, policy.BLACK):
				history = self.raw_history[colour]
				life = (
					statistics.mean(policy.expected_remaining_wears(s) for s in history)
					if history
					else 68.0
				)
				services += count * life
			demand = 2 * self.roommates * (self.days - turn.day + 1)
			return 10 * (ceil(max(0.0, demand - services) / (68.0 * 6)) + self.reserve_safety_packs)

		def can_discard(self, turn):
			if self.selection_unit >= 5 and settings['disable_five']:
				return False
			if family == 'fixed' and turn.budget_remaining < 10:
				return False
			if turn.budget_remaining < self.expected_replacement_reserve(turn):
				return False
			initial = turn.total_spent + turn.budget_remaining
			if initial == float('inf'):
				return True
			if initial <= 0:
				return False
			remaining_time = max(0.0, (self.days - turn.day) / self.days)
			return turn.budget_remaining / initial - remaining_time > self.budget_pace_margin

		def choose_discards(self, offered, wear, leftovers, turn):
			if family == 'fixed':
				return super().choose_discards(offered, wear, leftovers, turn)
			if not leftovers or not self.can_discard(turn):
				return ()
			candidates = []
			for i in leftovers:
				shade = offered[i]
				history = self.raw_history[policy.colour_of(shade)]
				if len(history) < self.min_dist_samples:
					continue
				fresh = 255 if policy.colour_of(shade) == policy.WHITE else 0
				gain = statistics.mean(
					policy.pair_embarrassment(shade, s) - policy.pair_embarrassment(fresh, s)
					for s in history
				)
				if gain > self.replacement_gain_threshold:
					candidates.append((gain, i))
			ranked = sorted(candidates, key=lambda x: (-x[0], x[1]))
			return tuple(i for _, i in ranked[: self.max_discards])

		def select_socks(self, offered, turn):
			choice = super().select_socks(offered, turn)
			self.voluntary_discards += len(choice.discard)
			self.two_discard_turns += len(choice.discard) >= 2
			return choice

	return ConfiguredPlayer2


def play(job):
	engine_class, _ = runtime(job['root'])
	config = job['config']
	own = player_class(
		job['root'], config['family'], json.dumps(config['settings'], sort_keys=True)
	)
	codes = job['roster'].split(',')
	classes = []
	for code in codes:
		if code == '2':
			classes.append(own)
		else:
			module, name = (
				(f'players.player_{code}.player', f'Player{code}')
				if code.isdigit()
				else ('players.random_player', 'RandomPlayer')
				if code == 'r'
				else ('players.greedy_player', 'GreedyPlayer')
			)
			classes.append(getattr(importlib.import_module(module), name))
	random.seed(job['seed'])
	import numpy as np

	np.random.seed(job['seed'])
	engine = engine_class(
		players=classes,
		capacity=job['capacity'],
		selection_unit=job['hand_size'],
		days=job['days'],
		seed=job['seed'],
		budget=job['budget'],
		timeout=1.0,
		keep_records=False,
	)
	result = engine.run()
	focal = job['focal']
	return dict(
		config_id=config['id'],
		family=config['family'],
		roster=job['roster'],
		focal_seats=','.join(map(str, focal)),
		budget=job['budget'],
		seed=job['seed'],
		capacity=job['capacity'],
		hand_size=job['hand_size'],
		days=job['days'],
		embarrassment=statistics.mean(result['players'][i]['total_embarrassment'] for i in focal),
		focal_sockless=statistics.mean(result['players'][i]['sockless_days'] for i in focal),
		household_sockless=result['total_sockless_days'],
		total_spent=result['total_spent'],
		budget_remaining=result['budget_remaining'],
		focal_discards=statistics.mean(
			getattr(engine.players[i], 'voluntary_discards', 0) for i in focal
		),
		two_discard_turns=statistics.mean(
			getattr(engine.players[i], 'two_discard_turns', 0) for i in focal
		),
		faults=len(result['faults']),
		fault_details=' | '.join(result['faults']),
		drawer_size=result['drawer_size'],
	)


def run(path, jobs, workers):
	path.parent.mkdir(parents=True, exist_ok=True)
	rows = []
	with path.open('x', newline='') as output, ProcessPoolExecutor(max_workers=workers) as pool:
		writer = None
		for future in as_completed([pool.submit(play, job) for job in jobs]):
			row = future.result()
			if writer is None:
				writer = csv.DictWriter(output, fieldnames=list(row), lineterminator='\n')
				writer.writeheader()
			writer.writerow(row)
			output.flush()
			rows.append(row)
			if len(rows) % 100 == 0 or len(rows) == len(jobs):
				print(f'{path.name}: {len(rows)}/{len(jobs)}', flush=True)
	return rows


def panel_jobs(configs, panels, budgets, seeds, root, hand=4):
	jobs = []
	for config, opponents, budget, seed in product(configs, panels, budgets, seeds):
		seat = seed % 4
		codes = list(map(str, opponents))
		codes.insert(seat, '2')
		jobs.append(
			dict(
				config=config,
				roster=','.join(codes),
				focal=[seat],
				root=root,
				budget=budget,
				seed=seed,
				capacity=28 if hand == 4 else 32,
				hand_size=hand,
				days=730,
			)
		)
	return jobs


def ranked(rows, configs):
	scores = []
	for config in configs:
		selected = [r for r in rows if r['config_id'] == config['id']]
		score = (
			float('inf')
			if any(r['faults'] for r in selected)
			else statistics.mean(r['embarrassment'] for r in selected)
		)
		scores.append((score, config['id'], config))
	return [c for _, _, c in sorted(scores)]


def sweep_configs(spec, family, anchor=None):
	grid = dict(spec['grid'])
	base = dict(spec['profiles']['initial']['settings'])
	if family == 'fixed':
		grid['inventory_buffer'] = [10, 12, 14]
		base['inventory_buffer'] = 10
	found = {}

	def add(p):
		p = dict(p)
		p['min_samples'] = min(p['min_samples'], p['raw_window'])
		if family == 'fixed':
			p['inventory_buffer'] = max(10, p['inventory_buffer'])
		found[json.dumps(p, sort_keys=True)] = p

	add(base)
	add({**base, 'raw_window': 10})
	if anchor:
		add(anchor)
	for key, values in grid.items():
		for value in values:
			add({**base, key: value})
	rng = random.Random(20260927)
	while len(found) < 80:
		add({**base, **{key: rng.choice(values) for key, values in grid.items()}})
	return [
		dict(id=f'{family}_{i:03}', family=family, settings=p) for i, p in enumerate(found.values())
	]


def suites(args, spec):
	groups = spec['groups']
	mixed = [tuple(groups[(i + j) % len(groups)] for j in (0, 1, 3)) for i in range(len(groups))]
	repeated = [(g, g, g) for g in groups]
	panels = mixed + repeated
	profiles = [dict(id=k, **v) for k, v in spec['profiles'].items()]
	output = args.output
	output.mkdir(parents=True, exist_ok=True)
	if args.suite == 'sweep':
		anchor = None
		for family in ['original', 'fixed']:
			configs = sweep_configs(spec, family, anchor)
			(output / f'{family}_configs.json').write_text(json.dumps(configs, indent=2))
			for stage, roster, seeds, count in [
				('screen', mixed, [9271000], 10),
				('validation', panels, range(9272000, 9272004), 3),
				('selection', panels, range(9273000, 9273010), 1),
			]:
				rows = run(
					output / f'{family}_{stage}.csv',
					panel_jobs(configs, roster, [120, 400], seeds, args.simulator_root),
					args.workers,
				)
				configs = ranked(rows, configs)[:count]
				(output / f'{family}_{stage}_selected.json').write_text(
					json.dumps(configs, indent=2)
				)
			anchor = configs[0]['settings']
		return
	if args.suite in ('main', 'small-budget'):
		seeds = range(9279000, 9279030) if args.suite == 'main' else range(9280000, 9280005)
		budgets = [120, 400] if args.suite == 'main' else [0, 10, 30, 60, 90]
		jobs = panel_jobs(
			profiles,
			panels if args.suite == 'main' else repeated,
			budgets,
			seeds,
			args.simulator_root,
		)
	else:
		hand = 5 if args.suite in ('five', 'five-sweep') else 4
		configs = []
		for family in ('original', 'fixed'):
			for limit in (
				[0, 1, 2, 3] if args.suite == 'five-sweep' else [0, 1] if hand == 5 else [0, 1, 2]
			):
				p = dict(spec['profiles'][f'{family}_tuned']['settings'], discard_limit=limit)
				if hand == 5:
					p['disable_five'] = False
				configs.append(
					dict(
						id=f'{family}_{"five" if hand == 5 else "limit"}_{limit}',
						family=family,
						settings=p,
					)
				)
		start = 9281000 if args.suite == 'five-sweep' else 9282000 if hand == 5 else 9283000
		count = 4 if args.suite == 'five-sweep' else 10
		jobs = panel_jobs(
			configs,
			mixed if args.suite == 'five-sweep' else panels,
			[120, 400],
			range(start, start + count),
			args.simulator_root,
			hand,
		)
	run(output / f'{args.suite}.csv', jobs, args.workers)


def record_inputs(path, args, spec):
	root = Path(args.simulator_root)
	hashes = {
		str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
		for directory in ['core', 'models', 'players']
		for p in (root / directory).rglob('*.py')
		if 'local_study' not in p.parts and '__pycache__' not in p.parts
	}
	metadata = dict(
		command=sys.argv,
		python=sys.version,
		config=spec,
		simulator_sources=hashes,
		policy_sha256=hashlib.sha256((HERE / 'player.py').read_bytes()).hexdigest(),
		runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
	)
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open('x') as f:
		json.dump(metadata, f, indent=2)


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
	parser.add_argument(
		'--simulator-root',
		type=Path,
		default=HERE.parents[1],
		help='Complete checkout supplying engine, opponents and helper modules',
	)
	parser.add_argument(
		'--suite', choices=['main', 'discards', 'five', 'small-budget', 'sweep', 'five-sweep']
	)
	parser.add_argument('--profiles', nargs='+', default=['fixed_tuned'])
	parser.add_argument(
		'--rosters',
		nargs='+',
		default=['2,2,2,2'],
		help='Seat order: group numbers, r=random, g=greedy',
	)
	parser.add_argument('--windows', nargs='+', type=int)
	parser.add_argument('--discard-limits', nargs='+', type=int)
	parser.add_argument('--gain-threshold', type=float)
	parser.add_argument('--budgets', nargs='+', type=int, default=[120, 400])
	parser.add_argument('--seeds', nargs='+', type=int)
	parser.add_argument('--seed-start', type=int, default=4001)
	parser.add_argument('--num-seeds', type=int, default=30)
	parser.add_argument('--capacity', type=int, default=28)
	parser.add_argument('--hand-size', type=int, choices=[4, 5], default=4)
	parser.add_argument('--days', type=int, default=730)
	parser.add_argument('--focal-seats', nargs='+', type=int)
	parser.add_argument('--workers', type=int, default=1)
	parser.add_argument(
		'--output', type=Path, required=True, help='New CSV path, or output directory for a suite'
	)
	args = parser.parse_args()
	args.simulator_root = str(args.simulator_root.resolve())
	spec = json.loads(args.config.read_text())
	if args.workers < 1 or args.days < 1 or min(args.budgets) < 0:
		parser.error('Workers and days must be positive; budgets must be nonnegative.')
	if args.suite:
		revision = subprocess.check_output(
			['git', 'rev-parse', 'HEAD'], cwd=args.simulator_root, text=True
		).strip()
		if revision != spec['simulator_revision']:
			parser.error(
				'Saved suites require the complete checkout at simulator_revision in the config.'
			)
		record_inputs(args.output / f'{args.suite}_inputs.json', args, spec)
		suites(args, spec)
		return
	seeds = (
		args.seeds
		if args.seeds is not None
		else list(range(args.seed_start, args.seed_start + args.num_seeds))
	)
	if not seeds or len(seeds) != len(set(seeds)) or min(seeds) < 0 or max(seeds) >= 2**32:
		parser.error('Use distinct seeds from 0 through 2**32-1.')
	jobs = []
	for name, roster in product(args.profiles, args.rosters):
		if name not in spec['profiles']:
			parser.error(f'Unknown profile: {name}')
		profile = spec['profiles'][name]
		p = dict(profile['settings'])
		if args.hand_size == 5 and name != 'initial':
			p.update(discard_limit=1, disable_five=False)
		codes = roster.split(',')
		if any(not (c.isdigit() or c in ('r', 'g')) for c in codes):
			parser.error(f'Invalid roster: {roster}')
		if args.capacity % 4 or args.capacity <= args.hand_size * len(codes) + 10:
			parser.error('Capacity must be a multiple of four and exceed hand_size * players + 10.')
		focal = (
			args.focal_seats
			if args.focal_seats is not None
			else ([i for i, c in enumerate(codes) if c == '2'] or list(range(len(codes))))
		)
		if (
			not focal
			or len(set(focal)) != len(focal)
			or any(i < 0 or i >= len(codes) for i in focal)
		):
			parser.error('Invalid focal seats.')
		for window, limit, budget, seed in product(
			args.windows or [p['raw_window']],
			args.discard_limits or [p['discard_limit']],
			args.budgets,
			seeds,
		):
			if window < 1 or limit < 0:
				parser.error('Windows must be positive; discard limits must be nonnegative.')
			settings = dict(p, raw_window=window, discard_limit=limit)
			if args.gain_threshold is not None:
				settings['gain'] = args.gain_threshold
			config = dict(
				id=f'{name}_w{window}_d{limit}', family=profile['family'], settings=settings
			)
			jobs.append(
				dict(
					config=config,
					roster=roster,
					focal=focal,
					root=args.simulator_root,
					budget=budget,
					seed=seed,
					capacity=args.capacity,
					hand_size=args.hand_size,
					days=args.days,
				)
			)
	record_inputs(args.output.with_suffix('.inputs.json'), args, spec)
	run(args.output, jobs, args.workers)


if __name__ == '__main__':
	main()
