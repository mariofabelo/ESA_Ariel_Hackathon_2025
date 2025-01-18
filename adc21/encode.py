# coding: utf-8

# Luis F. Simoes <luis.simoes@mlanalytics.ai>, 2021-2025


# ==================================== ## ==================================== #
# ------------------------------------ # Imports


import numpy as np

import matplotlib.pylab as plt

from .constants import *
from .data import *



# ==================================== ## ==================================== #
# ------------------------------------ # Data encoding defaults


GRID_W        = (10, 0)
GRID_GAP      = 5

WINDOW_RADIUS = 10
WINDOW_AGGR_P = -1  # harmonic mean

MIRROR_STACKING = True
#
#print("Per channel we take a grid of %d points (sum(GRID_W) + 1), " % (sum(GRID_W) + 1), end='')
#print("spaced %d timesteps apart (GRID_GAP).\nEach grid point becomes an aggregation over " % GRID_GAP, end='')
#print("a sliding window of size %d (2 * WINDOW_RADIUS + 1)." % (2 * WINDOW_RADIUS + 1))
#print("That aggregation is made by a generalized mean (WINDOW_AGGR_P).")



# ==================================== ## ==================================== #
# ------------------------------------ # Data transformations


def generalized_mean_of_axes(x, p, axis):
	"""
	Calculate the generalized mean of values in `x` across a
	given sequence (tuple) of axes.
	
	https://en.wikipedia.org/wiki/Generalized_mean
	
	Examples
	--------
	>>> x = np.arange(100).reshape(2, 5, 10) + 1
	>>> generalized_mean_of_axes(x, 1, (0, 2))
	array([30.5, 40.5, 50.5, 60.5, 70.5])
	>>> np.mean(x, axis=(0, 2))
	array([30.5, 40.5, 50.5, 60.5, 70.5])
	>>> generalized_mean_of_axes(x, -1, (0, 2)) # harmonic mean
	array([ 6.43162646, 24.33866529, 37.743683  , 49.91922671, 61.44974433])
	"""
	assert isinstance(axis, tuple)
	
	if x.size == 0:
		return np.nan
	
	p = float(p)
	n = np.prod(np.array(x.shape)[list(axis)])
	
	if p == 0:
		# the geometric mean uses a different equation
		# (would otherwise lead to division by 0 below)
		m = x.prod(axis)**(1 / n)
	else:
		m = (np.sum(np.power(x, p), axis) / n)**(1 / p)
	
	return m



def aggregate_obs(
		obs, window_radius=WINDOW_RADIUS, agg=WINDOW_AGGR_P, grid=None,
		mirror_stacking=MIRROR_STACKING,
		agg_stellar=False, agg_photon=False):
	"""
	Aggregates a planet's observations over a sliding window, using
	the requested `agg` aggregator (ex: 'mean', 'median', 'min', ...).
	An int or float `agg`, indicates the `p` parameter of the generalized
	mean that will be used to aggregate values (example: p=-1 to use the
	harmonic mean).
	
	The aggregation window has a size of `2 * window_radius + 1`. So, the
	central timestep plus `window_radius` timesteps to either side.
	
	Optionally aggregates also across independent observations if
	`agg_photon=True` or `agg_stellar=True` (or both).
	"""
#	assert obs.shape == (10, 10, 55, 300)
#	assert obs.shape == (10, 10, 55, 300) or obs.shape == (1, 10, 55, 300)
	assert isinstance(agg, (str, int, float, tuple))
	
	# pick the aggregation function
	if isinstance(agg, str):
		agg_ = getattr(np, agg)
		agg_args = dict()
	elif isinstance(agg, (int, float)):
		agg_ = generalized_mean_of_axes
		agg_args = dict(p = float(agg))
#	elif isinstance(agg, tuple) and agg[0] == 'polyval':
#		agg_ = polyval_of_axes
#		agg_args = dict(deg = agg[1])
	else:
		raise Exception('Unknown `agg` specification: %s' % repr(agg))
	
	
	# stack observations with their mirror around t=149.5.
	# * Because there's an even number of timesteps, stacking with the mirrored
	#   time series means t=149 gets stacked with t=150, and t=150 with t=149.
	# * with mirror_stacking aggregation will now produce a symmetric time series
	if mirror_stacking:
#		obs = np.stack([obs, obs[:, :, :, ::-1]], axis=1).reshape((10,20,55,300))
#		obs = np.stack([obs, obs[:, :, :, ::-1]], axis=1).reshape((obs.shape[0],20,55,300))
		obs = np.stack([obs, obs[:, :, :, ::-1]], axis=1).reshape((obs.shape[0],2*obs.shape[1],55,300))
	
	
	# define the aggregation axes, according to the requested aggregation levels
	axis = (3,)
	if agg_photon:  axis = (1,) + axis
	if agg_stellar: axis = (0,) + axis
	agg_args['axis'] = axis
	
	
	# define the grid of timesteps for which we'll generate aggregations
	# defaults to all.
	if grid is None:
		grid = np.arange(300) + 1
	grid = np.array(grid) - 1   # convert 1-based timestep to 0-based indexing
	
	
	# determine output matrix shape, and initialize it
	s = tuple([
		s
		for (d, s) in enumerate(obs.shape[:-1])
		if d not in axis] + [len(grid)]
		)
	obs_agg = np.ones(shape=s) * np.nan
	# a slicer over `obs_agg` that adjusts itself to the chosen aggregation level.
	# without agg_stellar and agg_photon, obs_agg[obs_idx] == obs_agg[:, :, :]
	obs_idx = tuple([slice(None)] * (obs_agg.ndim - 1))
	
	
	for i, t in enumerate(grid):
		#obs_agg[:, :, :, i] = agg_(obs[:, :, :, t - window_radius : t + window_radius + 1], axis=-1)
		
		# window of observations we'll be aggregating
		w = obs[:, :, :, t - window_radius : t + window_radius + 1]
		
		# skip aggregation if window goes out of bounds
		if w.shape[-1] != 2 * window_radius + 1:
			continue
		
		obs_agg[obs_idx + (i,)] = agg_(w, **agg_args)
	
	return obs_agg



def obs_transform(
		obs, window_radius=WINDOW_RADIUS, agg=WINDOW_AGGR_P, grid=None,
		mirror_stacking=MIRROR_STACKING,
		agg_stellar=True, agg_photon=True,
		as_relative_radii=True):
	"""
	Wrapper to `aggregate_obs()` that handles the aggregation of the planet's observations (`obs`).
	May optionally convert the observations, post-aggregation, to relative radii.
	"""
	assert obs.shape == (10, 10, 55, 300) or obs.shape == (1, 10, 55, 300)
	
	if not isinstance(agg_photon, bool):
		assert 1 <= agg_photon <= 10
		
		# keep only the number of photon noise instances relevant to the aggregation
		obs = obs[:,:agg_photon,:,:]
		
		# there will be an aggregation across photon noise instances if a
		# request was made to aggregate across more than one set of observations
		if agg_photon > 1 or mirror_stacking:
			agg_photon = True
		else:
			agg_photon = False
	
	o = aggregate_obs(obs, window_radius, agg, grid, mirror_stacking, agg_stellar, agg_photon)
	
	#print(obs.shape, o.shape)
	
	if as_relative_radii:
		o = get_relative_radii(np.clip(o, 0, 1))
	
	return o



# ==================================== ## ==================================== #
# ------------------------------------ # Time grid
# Determining which time-steps to use in a regular grid to be taken over the time series.


def t_grid_ixs(w=GRID_W, gap=GRID_GAP):
	"Regular grid moving away from the center."
	mid = 150
	if not isinstance(w, tuple):
		w1 = w2 = w
	else:
		w1, w2 = w
	return mid + gap * (np.arange(w1 + w2 + 1) - w1)


def show_t_grid(w=GRID_W, gap=GRID_GAP, hvline=plt.axvline,
				color='grey', alpha=0.66, t_range=(0, 300)):
	"Place vertical lines on the plot showing a regular time grid."
	t_min, t_max = t_range
	for t in t_grid_ixs(w, gap):
		if t_min <= t <= t_max:
			hvline(t, color=color, alpha=alpha, zorder=0, lw=0.4)
	hvline(150, color='black', alpha=.33, zorder=0, lw=0.5)


