"""Future mismatch with equal expected aging for the candidate and observed drawer.

One future candidate wear advances the observed partner by one expected wash.
This is a uniform-wear approximation; the geometric replacement target stays
stationary. Individual holes, selective wearing and pack arrivals are not simulated.
"""

from math import ceil


def mismatch_values(masses: list[float], fade: int) -> list[float]:
	"""Thresholded mismatch for every candidate age against an age distribution.

	Prefix sums give the mass and first moment outside the free shade gap,
	so all 65 candidate ages can be evaluated in linear time.
	"""
	weights = [0.0]
	moments = [0.0]
	for age, mass in enumerate(masses):
		weights.append(weights[-1] + mass)
		moments.append(moments[-1] + age * mass)
	free_ages = 6 // fade
	values = []
	for age in range(65):
		lo = max(0, age - free_ages)
		hi = min(65, age + free_ages + 1)
		gap = age * weights[lo] - moments[lo]
		gap += moments[65] - moments[hi] - age * (weights[65] - weights[hi])
		values.append(max(0.0, fade * gap))
	return values


def project_life_values(
	observed_mass: list[float],
	fade: int,
	alpha: float,
	geometric_survival: float,
	life_survival_cap: float,
	horizon: float,
) -> list[float]:
	"""Return the discounted future mismatch for each starting age, 0..64.

	The finite horizon is days_left * 2n/C expected wears. A fractional last
	interval receives q**step - q**horizon weight.
	After 64 steps the observed pair shares a cap, leaving only target cost.
	"""
	values = [0.0] * 65
	if horizon <= 0:
		return values
	q = min(geometric_survival, life_survival_cap)
	total = sum(observed_mass)
	masses = [mass / total for mass in observed_mass] if total else [0.0] * 65
	target_mass = [(1.0 - geometric_survival) * geometric_survival**age for age in range(65)]
	target_mass[-1] = geometric_survival**64
	target_costs = mismatch_values(target_mass, fade)
	for step in range(min(64, ceil(horizon))):
		weight = q**step - q ** min(step + 1.0, horizon)
		observed_costs = mismatch_values(masses, fade)
		for age in range(65):
			future_age = min(64, age + step)
			values[age] += weight * (
				(1.0 - alpha) * observed_costs[future_age] + alpha * target_costs[future_age]
			)
		masses = [0.0, *masses[:63], masses[63] + masses[64]]
	if horizon > 64:
		tail = alpha * (q**64 - q**horizon) * target_costs[64]
		values = [value + tail for value in values]
	return values
