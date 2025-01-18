# coding: utf-8

# Luis F. Simoes <luis.simoes@mlanalytics.ai>, 2021-2025


# ==================================== ## ==================================== #
# ------------------------------------ # Imports


import os

import pandas as pd
import seaborn as sns



# ==================================== ## ==================================== #


# path in which the module is located
MODULE_PATH = os.path.dirname(os.path.abspath(__file__)) + os.sep



# ==================================== ## ==================================== #


# FIELDS PRESENT IN THE PARAMETERS DATA
#
# See the Data Documentation:
# https://web.archive.org/web/20210511051247/https://www.ariel-datachallenge.space/ML/documentation/data


# "The files are named following the convention: AAAA_BB_CC.txt
# The name is unique for each observation (i.e. datapoint) and
# AAAA (0001 to 2097) is an index for the planet observed,
# BB (01 to 10) is an index for the stellar spot noise instance observed and
# CC (01 to 10) is an index for the gaussian photon noise instance observed."
OBSERV_INDEX = ['planet', 'stellar_spot', 'photon']


# X (observed data, i.e. the features) -- available in the train and test set
# 6 stellar and planet parameters
X_PARAMS = [
	'star_temp', 'star_logg', 'star_rad', 'star_mass', 'star_k_mag', 'period']

X_PARAMS_long_name = {
	'star_temp'  : 'stellar temperature',
	'star_logg'  : 'stellar surface gravity',
	'star_rad'   : 'stellar radius',
	'star_mass'  : 'stellar mass',
	'star_k_mag' : 'stellar K magnitude',
	'period'     : 'orbital period'}


# y (retrieved data, i.e. the targets) -- only available in the train set
# 2 planet parameters: 'sma' (semimajor axis) and 'incl' (inclination)
# (can be used as intermediate targets or be ignored)
Y_PARAMS = ['sma', 'incl']

Y_PARAMS_long_name = {
	'sma'  : 'semimajor axis',
	'incl' : 'inclination'}		# https://en.wikipedia.org/wiki/Orbital_elements


# Alternate mapping between short and long names for the star/planet parameters
# present in the dataset.
param_name_map = pd.Series({
	'star_temp'  : 'Star Temperature [K]',
	'star_logg'  : 'Star Surface Gravity [log(g)]',
	'star_rad'   : 'Star Radius [Rs]',
	'star_mass'  : 'Star Mass [Ms]',
	'star_k_mag' : 'Star K Mag',
	'period'     : 'Planet Period [days]',
	'sma'        : 'Planet Semi-major Axis [m]',
	'incl'       : 'Inclination'})



# ==================================== ## ==================================== #


# **Wavelength grid** used to generate the ADC19/ADC21 datasets
wlgrid = pd.read_csv(
	MODULE_PATH + 'wlgrid_edwards2019updated_ComparisionTiers.png_tier2.csv',
	header=None).round(2)[0].sort_values().values

# picking a color palette for the 55 channels
# https://seaborn.pydata.org/tutorial/color_palettes.html
wlgrid_palette = sns.color_palette("flare", n_colors=55)


