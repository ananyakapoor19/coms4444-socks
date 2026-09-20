"""Fit a small neural net to the 400-combo threshold dataset from
scripts/sweep_cli_regression.py: (capacity, days, budget) -> (white, black).

    uv run --with scikit-learn,numpy,matplotlib python players/player_4/scripts/fit_nn_thresholds.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

DATASET = Path(__file__).resolve().parent.parent / 'results' / 'cli_param_threshold_dataset.csv'
HIDDEN_LAYERS = (16, 8)
TEST_FRACTION = 0.2
RANDOM_STATE = 4444


def main() -> None:
	import numpy as np
	from sklearn.model_selection import train_test_split
	from sklearn.neural_network import MLPRegressor
	from sklearn.preprocessing import StandardScaler

	rows = list(csv.DictReader(DATASET.open()))
	X = np.array([[float(r['capacity']), float(r['days']), float(r['budget'])] for r in rows])
	y = np.array([[float(r['white']), float(r['black'])] for r in rows])

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=TEST_FRACTION, random_state=RANDOM_STATE
	)

	x_scaler = StandardScaler().fit(X_train)
	y_scaler = StandardScaler().fit(y_train)
	X_train_s = x_scaler.transform(X_train)
	X_test_s = x_scaler.transform(X_test)
	y_train_s = y_scaler.transform(y_train)

	model = MLPRegressor(
		hidden_layer_sizes=HIDDEN_LAYERS,
		activation='relu',
		max_iter=5000,
		random_state=RANDOM_STATE,
		early_stopping=True,
		n_iter_no_change=50,
	)
	model.fit(X_train_s, y_train_s)

	pred_train = y_scaler.inverse_transform(model.predict(X_train_s))
	pred_test = y_scaler.inverse_transform(model.predict(X_test_s))

	def r2(y_true, y_pred):
		ss_res = np.sum((y_true - y_pred) ** 2)
		ss_tot = np.sum((y_true - y_true.mean()) ** 2)
		return 1 - ss_res / ss_tot

	print(
		f'{len(rows)} rows, {len(X_train)} train / {len(X_test)} test, hidden layers {HIDDEN_LAYERS}'
	)
	print(f'training iterations: {model.n_iter_}')
	for i, name in enumerate(('WHITE_MIN', 'BLACK_MAX')):
		print(
			f'{name}: train R^2 = {r2(y_train[:, i], pred_train[:, i]):.3f}  '
			f'test R^2 = {r2(y_test[:, i], pred_test[:, i]):.3f}'
		)

	import matplotlib

	matplotlib.use('Agg')
	import matplotlib.pyplot as plt

	fig, axes = plt.subplots(1, 2, figsize=(11, 5))
	for i, (ax, name) in enumerate(zip(axes, ('WHITE_MIN', 'BLACK_MAX'), strict=True)):
		ax.scatter(y_train[:, i], pred_train[:, i], alpha=0.5, label='train')
		ax.scatter(y_test[:, i], pred_test[:, i], alpha=0.8, label='test', color='red')
		lo = min(y[:, i].min(), pred_train[:, i].min(), pred_test[:, i].min())
		hi = max(y[:, i].max(), pred_train[:, i].max(), pred_test[:, i].max())
		ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1)
		ax.set_xlabel(f'actual {name}')
		ax.set_ylabel(f'predicted {name}')
		ax.set_title(name)
		ax.legend()

	fig.suptitle(f'MLP{HIDDEN_LAYERS} fit: (capacity, days, budget) -> (white, black)')
	fig.tight_layout()
	out_path = DATASET.parent / 'nn_threshold_fit.png'
	fig.savefig(out_path, dpi=150)
	print(f'wrote: {out_path}')


if __name__ == '__main__':
	main()
