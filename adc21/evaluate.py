# coding: utf-8

# Luis F. Simoes <luis.simoes@mlanalytics.ai>, 2021-2025


# ==================================== ## ==================================== #
# ------------------------------------ # Imports


import numpy as np
import pandas as pd

from fastprogress import progress_bar

from sklearn.model_selection import KFold
from sklearn.metrics import (r2_score, mean_absolute_error,
							 median_absolute_error, mean_squared_error)

from .constants import OBSERV_INDEX
from .data_pipeline import standardizer, y_aggregate



# ==================================== ## ==================================== #
# ------------------------------------ #  Cross-validation


def train_test_split(data, n_splits=10, shuffle=True, random_state=0):
	"""
	Train/test data slitter that ensures noise instances of the same
	planet all end on the same dataset.
	
	`data` is an instance of `data_container`.
	"""
	kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
	
	for (train_index, test_index) in kf.split(data.planet_idxs):
		
		tr_ix = data.planet_idxs[train_index]
		te_ix = data.planet_idxs[ test_index]
		tr_m  = data.X_par['planet'].isin(tr_ix).values
		te_m  = ~tr_m
		
		# dataframe mapping fname -> (planet, stellar, photon) indices,
		# of planets in the train and test sets
		tr_ix = data.X_par.loc[tr_m, OBSERV_INDEX]
		te_ix = data.X_par.loc[te_m, OBSERV_INDEX]
		
		tr_X = data.X[tr_m]
		te_X = data.X[te_m]
		
		tr_y = data.y[tr_m]
		te_y = data.y[te_m]
		
		yield (tr_X, tr_y), (te_X, te_y), (tr_ix, te_ix)



def cross_validate_setup(
		data, init_model, n_splits=10, split_seed=0, standardize=True,
		prefit_transform=None, y_encode=None, y_decode=None,
		fit_err_track=False, verbose=True, **fit_args):
	"""
	Cross-validate a modelling setup.
	
	Returns results as a dictionary, where each key maps to a list of results
	per train/validation fold:
	
	```python
	# Initialization of the CV results dictionary:
	cv_res = dict(
		test_ids   = [],  # ids of planets that were included in the test-fold used to evaluate this model.
		data_cfg   = [],  # parameters configuring processing of data sent to each model (standardization parameters)
		models     = [],  # the trained models
		y_true     = [],  # the prediction targets of samples on the test-fold
		y_pred     = [],  # the y_pred of each model on its test-fold
		y_pred_agg = [],  # test-fold predictions aggregated at planet-level
		evals      = [],  # score() over y_pred
		evals_agg  = [],  # score() over y_pred_agg
		)
	```
	"""
	if prefit_transform is None:
		prefit_transform = lambda X, y : (X, y)
	
	# y transforms: either neither are given, or both need to be given
	assert not (y_encode == None) ^ (y_decode == None)
	if y_encode == None:
		y_encode = lambda y : y
	if y_decode == None:
		y_decode = lambda y : y
	
	cv_res = dict(
		test_ids   = [],  # ids of planets that were included in the test-fold used to evaluate this model.
		data_cfg   = [],  # parameters configuring processing of data sent to each model (standardization parameters)
		models     = [],  # the trained models
		y_true     = [],  # the prediction targets of samples on the test-fold
		y_pred     = [],  # the y_pred of each model on its test-fold
		y_pred_agg = [],  # test-fold predictions aggregated at planet-level
		evals      = [],  # score() over y_pred
		evals_agg  = [],  # score() over y_pred_agg
		)
	
	splitter = train_test_split(data, n_splits, random_state=split_seed)
	if verbose:
		splitter = progress_bar(splitter, total=n_splits)
	
	for (tr_X, tr_y), (te_X, te_y), ix in splitter:
		
		if standardize:
			(tr_X, te_X, std_params) = standardizer(tr_X, te_X)  # , skip_cols=len(X_PARAMS)
		else:
			std_params = None
		
		m = init_model()
		
		# if model can track the test set error as it trains, send it the current test set
		track_err = dict(test_set=(te_X.copy(), te_y.copy())) if fit_err_track else {}
		
		m.fit(
			*prefit_transform(tr_X, y_encode(tr_y)),
			**track_err,
			**fit_args)
		
		yp  = m.predict(te_X)
		yp  = y_decode(yp)
		
		ya  = y_aggregate(data, te_y, yp, obs_ix=ix[1])
		
		s   = score(te_y, yp, verbose=False)
		sa  = score(*ya, verbose=False)
		
		cv_res['test_ids'  ].append(ix[1])
		cv_res['data_cfg'  ].append(std_params)
		cv_res['models'    ].append(m)
		cv_res['y_true'    ].append(te_y)
		cv_res['y_pred'    ].append(yp)
		cv_res['y_pred_agg'].append(ya)
		cv_res['evals'     ].append(s)
		cv_res['evals_agg' ].append(sa)
	
	return cv_res



# ==================================== ## ==================================== #
# ------------------------------------ #  Performance metrics


def score(y_true, y_pred, verbose=True):
	"""
	Regression metrics, plus a partially weighted version of the
	competition's scoring function.
	"""
	if verbose:
		print('[ Scoring with y_true.shape: %s ]' % str(y_true.shape))
	
	y_true = y_true.ravel()
	y_pred = y_pred.ravel()
	
	R_2 = r2_score(y_true, y_pred)
	MSE = mean_squared_error(y_true, y_pred)
	MAE = mean_absolute_error(y_true, y_pred)
	mAE = median_absolute_error(y_true, y_pred)
	
	# Root Mean Squared Percentage Error (RMSPE)
	# ^ It was the metric used in the ADC22 light track. See also:
	# https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
	RMSPE = np.sqrt(((y_pred / y_true - 1) ** 2).mean())
	
	# The competition's score function (rescaled MAE & partially weighted versions)
	# https://web.archive.org/web/20210701095034/https://www.ariel-datachallenge.space/ML/documentation/scoring
	S1 = 1e4 - MAE * 1e6
	S2 = 1e4 - 2 * (y_true * np.abs(y_pred - y_true)).mean() * 1e6
	#
	# ^ See also:
	# https://github.com/ucl-exoplanets/ML-challenge-baseline/blob/main/utils.py#L95
	# https://github.com/ucl-exoplanets/ML-challenge-baseline/blob/main/walkthrough.ipynb
	# "Let's define it roughly here, with unity weights as we don't have the actual weights
	# available. Note that this might likely lead to conservative (pessimistic) score
	# estimation as the real metric gives smaller weights to the hardest samples (in terms
	# of signal-to-noise), whereas here all the samples have equal weighting."
	
	evals = pd.Series(dict((
		('R^2'               , R_2),
		('MSE'               , MSE),
		('RMSE'              , np.sqrt(MSE)),
		('RMSPE'             , RMSPE),
		('MAE'               , MAE),
		('MedianAE'          , mAE),
		('Score, w: none'    , S1),
		('Score, w: 2*radii' , S2),
		)))
	
	return evals



def score_light(y_true, y_pred, axis=None):
	d = y_true - y_pred
	da = np.abs(d)
	
	MSE = np.mean(d*d, axis=axis)  # mean_squared_error(y_true, y_pred)
	MAE = np.mean(da, axis=axis)   # mean_absolute_error(y_true, y_pred)
	mAE = np.median(da, axis=axis) # median_absolute_error(y_true, y_pred)
	
	# Root Mean Squared Percentage Error (RMSPE)
	RMSPE = np.sqrt(((y_pred / y_true - 1) ** 2).mean(axis=axis))
	
	S1 = 1e4 - MAE * 1e6
	S2 = 1e4 - 2 * (y_true * da).mean(axis=axis) * 1e6
	
	evals = dict((
		('MSE'               , MSE),
		('RMSE'              , np.sqrt(MSE)),
		('RMSPE'             , RMSPE),
		('MAE'               , MAE),
		('MedianAE'          , mAE),
		('Score, w: none'    , S1),
		('Score, w: 2*radii' , S2)))
	
	return evals

