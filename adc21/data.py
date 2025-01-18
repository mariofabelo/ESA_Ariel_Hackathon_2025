# coding: utf-8

# Luis F. Simoes <luis.simoes@mlanalytics.ai>, 2021-2025


# ==================================== ## ==================================== #
# ------------------------------------ # Imports


import pickle

import numpy as np
import pandas as pd

from .constants import *



# ==================================== ## ==================================== #
# ------------------------------------ #  Dataset


def load(fname, mode='rb', open=open):
	with open(fname, mode) as f:
		d = pickle.load(f)
	return d


class dataset:
	def __init__(self, **kwargs):
		self.__dict__.update(**kwargs)



# exporting/importing of pandas data, without saving DataFrames/Series to disk
#
def df_export(df):
	return (
		df.to_dict(),
		tuple(df.index.names) if df.index.name is None else df.index.name)

def df_import(df_dict, dtype=pd.DataFrame):
	df_data, index_name = df_dict
	return dtype(df_data).rename_axis(index=index_name)



def load_dataset(fname):
	DATA = load(fname)
	DATA = dataset(**DATA)
	
	pandas_variables = [
		'obs_to_fname',
		'planet',
		'planet_mcs19',
		#'planet_mcs24',
		'X_params',
		'y_params',]
	
	for v in pandas_variables:
		if v not in DATA.__dict__:
			continue
		DATA.__dict__[v] = df_import(
			DATA.__dict__[v],
			dtype=(pd.Series if v == 'obs_to_fname' else pd.DataFrame))
	
	return DATA



# ==================================== ## ==================================== #
# ------------------------------------ # Data retrieval


def get_relative_fluxes(train_y, planet_ix):
	"""
	From the retrieved data (i.e. the targets) for `planet_ix` obtain
	its 1D array of relative radii (planet-to-star-radius ratios)
	and convert it to relative fluxes (relative brightness).
	"""
	# sufficient to retrieve from file '????_01_01.txt', as targets
	# don't change across a planet's many 'Parameters' files.
	f = '%04d_01_01.txt' % planet_ix
	
	# "Note the planet to host star relative radius $\frac{R_p}{R_∗}$
	# is directly connected to the transit depth of the light curve,
	# as the latter is equal to $(\frac{R_p}{R_∗})^2$." -- ADC19 paper, Sec. 3.2
	true_dip_min = 1.0 - train_y[f] **2
	
	return true_dip_min



def get_relative_radii(relative_fluxes):
	"""
	Convert from relative fluxes (relative brightness) to relative radii
	(planet-to-star-radius ratios), the space of the model's targets/predictions.
	"""
	return np.sqrt(1.0 - relative_fluxes)



def get_observations(dataset, planet_ix, spot_ix=None):
	"""
	Get all observations of a given planet x stellar spot combination.
	Retrieves 10 gaussian photon noise instances.
	If `spot_ix=None` will instead return all of a planet's observations.
	
	Parameters
	----------
	dataset : dict
		dictionary mapping filenames to their (55, 300)
		observation matrices
	planet_ix : int
		planet observed
	spot_ix : int
		stellar spot noise instance
	
	Returns
	-------
	obs : ndarray
		array of shape (1, 10, 55, 300) if a spot_ix is given,
		otherwise of shape (10, 10, 55, 300).
	"""
	spot_ixs = list(range(1, 10+1)) if spot_ix is None else [spot_ix]
	
	K = [
		['%04d_%02d_%02d.txt' % (planet_ix, spot_ix, noise_ix) for noise_ix in range(1, 10+1)]
		for spot_ix in spot_ixs]
	
	obs = [
		[dataset[kk] for kk in k if kk in dataset]
		for k in K]
	obs = np.array([o for o in obs if o != []])
	# ^ returns a matrix of shape (n, 10, 55, 300) <- n=10 by default, or as many as exist in `dataset`
	
	return obs


