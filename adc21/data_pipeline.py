# coding: utf-8

# Luis F. Simoes <luis.simoes@mlanalytics.ai>, 2021-2025


# ==================================== ## ==================================== #
# ------------------------------------ # Imports


from functools import partial

import numpy as np
import pandas as pd

from fastprogress import progress_bar

from .constants import *
from .encode import get_observations, obs_transform, t_grid_ixs



# ==================================== ## ==================================== #
# ------------------------------------ #  Data container


class data_container:
	
	def __init__(self, data, planet_idxs=None):
		
		# by default, use all of the available planets
		if planet_idxs is None:
			planet_idxs = data.planet_idxs
		
		self.planet_idxs = np.array(planet_idxs, dtype=np.int64)
		
		# are we dealing with train or test set planets?
		self.data_source = 'train' if data.y != None else 'test'
		
		# get observations of the requested planets
		self.X_obs = [
			get_observations(data.X, planet_ix=pix)
			for pix in self.planet_idxs]
		# ^ list of (10, 10, 55, 300) matrices
		
		# get the star/planet parameters
		self.X_par = data.X_params.loc[data.obs_to_fname.loc[self.planet_idxs]]
		# ^ shape: (125600, 3+6) -- has the OBSERV_INDEX + X_PARAMS columns
		
		# get the targets, if we're dealing with train set information
		if self.data_source == 'train':
			self.y = np.array([
				data.y[fname]
				for fname in self.X_par.index])
			# ^ shape: (125600, 55)
		else:
			self.y = None
		
		# store .y under an alias, so raw encoding remains available in
		# case we decide to encode the target
		#self.y_raw = self.y
		
	

# ==================================== ## ==================================== #
# ------------------------------------ #  Data transformations


def standardizer(tr_X, te_X=None, skip_cols=None, avg_std=None):
	"""
	Standardize tr_X, and use its mean/std to standardize also te_X (if given).
	If `skip_cols` (int) is given, those many columns at the start are excluded
	from the standardization (but still included as copies in the outputs).
	"""
	#avg = tr_X.mean(axis=0)
	#std = tr_X.std(axis=0)
	#tr_X = (tr_X - avg) / std
	#te_X = (te_X - avg) / std
	
	# a view over X that skips the stellar/planet parameters columns.
	no_par = lambda X : X[:, skip_cols:]
	
	tr_X = tr_X.copy()
	
	if avg_std is None:
		avg = no_par(tr_X).mean(axis=0)
		std = no_par(tr_X).std(axis=0)
		# standardization that uses common settings across channels:
		#avg = no_par(tr_X).mean()
		#std = no_par(tr_X).std()
	else:
		avg, std = avg_std
	
	tr_X[:, skip_cols:] = (no_par(tr_X) - avg) / std
	
	if te_X is not None:
		te_X = te_X.copy()
		te_X[:, skip_cols:] = (no_par(te_X) - avg) / std
	
	return tr_X, te_X, (avg, std)



# ==================================== ## ==================================== #
# ------------------------------------ #  Dataset aggregations (spots/photon)


def Xy_aggregate_noise_instances(data, agg_stellar=False):
	"""
	Aggregate `data.X`, `data.X_par` and `data.y` across
	photon noise instances, with a .mean() in the case of .X
	and a .first() elsewhere.
	If `agg_stellar=True`, the aggregation is then across
	the planet's 10*10 observations.
	
	The original values are made available as `data.X_all`,
	`data.X_par_all` and `data.y_all`.
	
	*Update 2021-04-25: the aggregation of X is now handled elsewhere.*
	"""
	X_par = getattr(data, 'X_par' + '_all', getattr(data, 'X_par'))
	
	group_key = X_par['planet'].values
	if not agg_stellar:
		group_key = [
			group_key,
			X_par['stellar_spot'].values]
	
	aggr_vars = [
		('X_par', 'first'),
		#('X', 'mean')
		]
	if data.y is not None:
		aggr_vars.append(('y', 'first'))
	
	for var_name, agg_func in aggr_vars:
		
		# if a variant of of var_name exists with the '_all' suffix, 
		# reads values from it, as it will contain the original non-aggregated
		# values stored there in a previous execution of this function.
		var_agg = var_values = getattr(
			data, var_name + '_all',
			getattr(data, var_name)) # < default
		
		if isinstance(var_agg, pd.DataFrame):
			# to preserve DataFrames' indices (X_par in this case) we
			# reset the index here, and re-set it post-aggregation below
			var_agg = var_agg.reset_index()
		
		# aggregate across noise instances
		var_agg = pd.DataFrame(var_agg).groupby(group_key, sort=False).agg(agg_func)
		
		if isinstance(var_values, np.ndarray):
			var_agg = var_agg.values
		else:
			var_agg.set_index('index', inplace=True)
		
		data.__dict__[var_name + '_all'] = var_values
		data.__dict__[var_name         ] = var_agg



def y_aggregate(data, y_true, y_pred, quantile=None, obs_ix=None):
	"""
	Aggregate predictions at a planet level,
	across stellar spots/photon noise predictions.
	
	If a `quantile` is given we first take the mean across photon
	noise instances, and then take that quantile across stellar
	noise instances, to produce the planet's final prediction.
	"""
	if obs_ix is None:
		obs_ix = data.X_par
	planet_id = obs_ix['planet'].values
	stellr_id = obs_ix['stellar_spot'].values
	
	if y_true is None:
		y_true_agg = None
	else:
		y_true_agg = pd.DataFrame(y_true).groupby(planet_id, sort=False).first().values
	
	if quantile is None:
		y_pred_agg = pd.DataFrame(y_pred).groupby(planet_id, sort=False).mean().values
	else:
		y_pred_agg = pd.DataFrame(y_pred) \
			.groupby([planet_id, stellr_id], sort=False).mean() \
			.groupby(level=0, sort=False).quantile(quantile).values
	
	return y_true_agg, y_pred_agg



# ==================================== ## ==================================== #
# ------------------------------------ #  Data Pipeline


def X__set_to__obs_encodings(
		data, window_radius=10, agg='mean', grid=None, mirror_stacking=False,
		agg_stellar=False, agg_photon=False, verbose=True):
	"""
	Initialize `data.X` with an encoding where each observation
	gets represented by a vertical stacking of checkpoints taken
	over transforms of each channel's fluxes.
	"""
	# agg_photon can be defined as either bool or int in {1..10}
	# get boolean flag indicating whether aggregation will be performed across
	# photon noise instances
	agg_photon_ = agg_photon
	if not isinstance(agg_photon, bool):
#		agg_photon_ = agg_photon > 1
		agg_photon_ = True	# an int param means we will no longer keep all 10 instances
	
	# can only make use of mirror_stacking in setups where we aggregate
	# at least across photon noise instances
	if not agg_photon_:
		mirror_stacking = False
	
	grid = t_grid_ixs() if grid is None else grid
	
	# observation grid of generalized means over overlapping time windows
	data.X = np.vstack([
		obs_transform(
			o, window_radius, agg, grid, mirror_stacking, agg_stellar, agg_photon
			).reshape(-1, 55 * len(grid)).astype(o.dtype)
		for o in (progress_bar(data.X_obs) if verbose else data.X_obs)
		])
	
	# if X now contains aggregations across stellar and/or photon noise instances,
	# then we need to ensure .X_par and .y will match it aggregation-wise.
	assert not agg_stellar or (agg_stellar and agg_photon_)
	if agg_photon_:
		Xy_aggregate_noise_instances(data, agg_stellar=agg_stellar)
	assert len(data.X) == len(data.X_par)
	assert data.y is None or len(data.X) == len(data.y)



def X__stack_with__par(data, par_var='X_par', **kwargs):
	"""
	Update data.X, by merging into it the stellar/planet parameters (.X_par or .X_par_st).
	"""
	assert par_var == 'X_par' # ensure there are no left overs in the notebooks still calling this with 'X_par_st'
	
	data.X = np.hstack([
		getattr(data, par_var)[X_PARAMS].values.astype(data.X.dtype),
		data.X
		])
	
	# Precompute a version of .X that has all observations standardized.
	# On a case-by-case basis, models will make use of either .X or .X_st.
	# The decision to standardize or nor the stellar/planet parameters is
	# left to other steps of the pipeline (which ultimately determine `par_var`).
#	data.X_st, _, data.X_st_params = standardizer(data.X) # , skip_cols=len(X_PARAMS)



# ------------------------------------ #  Data Pipeline: Configure & RUN


def setup_summary(setup):
	"""
	Display a string containing a summary of the data representation obtained
	as a consequence of the parameter choices.
	"""
	MEAN_TYPE = {
		'min': 'p=-∞ (min)',
		-3: 'p=-3',
		-2: 'p=-2',
		-1: 'p=-1 (Harmonic mean)',
			0: 'p=0 (Geometric mean)',
			1: 'p=1 (Arithmetic mean)',
	'mean': 'p=1 (Arithmetic mean)',
			2: 'p=2 (Quadratic mean)',
			3: 'p=3 (Cubic mean)',
		'max': 'p=+∞ (max)'}.get(setup['MEAN_TYPE_P'], setup['MEAN_TYPE_P'])
	
	# dimensionality of the encoded representation (model inputs)
	dim = setup['GRID_PTS_NUM'] * (1 if setup['MIRROR_STACKING'] else 2) + 1
	dim *= 55   # wavelength channels
	dim += 6    # star/planet parameters
	
	# number of distinct observations encoded into each grid point
	num_agg  = 2 * setup['WINDOW_RADIUS'] + 1
	num_agg *= setup['AGGR_NUM_OBS']
	num_agg *= (2 if setup['MIRROR_STACKING'] else 1)
	
	s = (
		f"With these settings, a star has its observations transformed into a {dim}-dimensional representation: " +
		f"{'' if setup['MIRROR_STACKING'] else '2×'}{setup['GRID_PTS_NUM']}+1 grid points × 55 wavelength channels + 6 stellar/planet parameters.\n" +
		f"Each grid point aggregates with a {MEAN_TYPE} a total of {num_agg} values: " +
		f"a time window 2×{setup['WINDOW_RADIUS']}+1 minutes wide × {setup['AGGR_NUM_OBS']} planetary transits{'' if not setup['MIRROR_STACKING'] else ' × 2 time symmetrical observations'}.")
	
	print(s)


def configure_pipeline(
		GRID_PTS_NUM, GRID_PTS_SPACING,
		WINDOW_RADIUS, AGGR_NUM_OBS, MEAN_TYPE_P, MIRROR_STACKING):
	"""
	Define the data pre-processing pipeline.
	"""
	GRID_PTS_NUM = (GRID_PTS_NUM, 0) if MIRROR_STACKING else GRID_PTS_NUM
	GRID = tuple(t_grid_ixs(w=GRID_PTS_NUM, gap=GRID_PTS_SPACING))
	
	OBS_GRID_PARAMS = dict(
		agg=MEAN_TYPE_P, window_radius=WINDOW_RADIUS, grid=GRID,
		mirror_stacking=MIRROR_STACKING)
	
	X_pipeline = [
		partial(
			X__set_to__obs_encodings,
			agg_stellar=False, agg_photon=AGGR_NUM_OBS, **OBS_GRID_PARAMS),
		partial(X__stack_with__par, par_var='X_par'),
		]
	
	return X_pipeline



def prepare_data(data, X_pipeline, verbose=True):
	
	if not isinstance(data, data_container):
		data = data_container(data)
	
	for step in X_pipeline:
		if verbose:
			print(str(step))
		step(data, verbose=verbose)
	
	return data
