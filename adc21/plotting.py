# coding: utf-8

# Luis F. Simoes <luis.simoes@mlanalytics.ai>, 2021-2025


# ==================================== ## ==================================== #
# ------------------------------------ # Imports


import numpy as np
import pandas as pd

import matplotlib.pylab as plt
import seaborn as sns

from .constants import *
from .encode import *



# ==================================== ## ==================================== #
# ------------------------------------ #  Light curve plotting code


def create_panel(ncols=2, nrows=1, figsize=5, tight=False, **kwargs):
	"""
	Initialize a panel of subplots with `ncols` columns and `nrows` rows that
	all share the same properties.

	For available `kwargs`, see:
	https://matplotlib.org/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure.add_subplot

	Examples:

	>>> fig, axs = create_panel(xlabel='x', ylabel='y', xlim=[-1, 10], yscale='log')
	>>> axs[0].scatter(*np.random.lognormal(size=(2, 1000)), s=1)

	>>> fig, axs = create_panel(2, 2, xticks=[], yticks=[], aspect='equal', frame_on=False)
	>>> axs[3].imshow(image)
	"""
	ax_cfg = dict(**kwargs)

	fig, axs = plt.subplots(
		nrows, ncols,
		figsize=(figsize * ncols, figsize * nrows),
		subplot_kw=ax_cfg)

	if tight:
		fig.tight_layout()

	return fig, axs.flat



def custom_int_formatter(x, pos):
	if x.is_integer(): return f'{int(x)}'
	else: return f'{x}'

def show_spectrum(spectrum, ax=None, label=None, c='C1',
				  wlgrid_min=0, wlgrid_max=54, **kwargs):
	"""
	"""
#	with sns.plotting_context('notebook'), sns.axes_style("whitegrid"):  # , font_scale=1.25
	if True:
		if ax is None:
			fig, ax = plt.subplots()
		
		wlrange = slice(wlgrid_min, wlgrid_max+1)
		
		for plot in [ax.plot if wlgrid_min!=wlgrid_max else ax.scatter]:
#		for plot in [ax.plot, ax.scatter]:
			plot(
				#np.arange(55) + 1,
				wlgrid[wlrange],
				spectrum[wlrange], label=label, color=c, **kwargs)
			label = None
		
		# plt.errorbar(
		#     x =    observed_spectrum[pid][:, ix_wlgrid],
		#     y =    observed_spectrum[pid][:, ix_spectrum],
		#     yerr = observed_spectrum[pid][:, ix_noise],
		#     xerr = observed_spectrum[pid][:, ix_wlwidth] / 2,
		#     ecolor='black'
		#     );
		
		#ax.set_xlabel('instrument_wlgrid')   # SpectralData_label[ix_wlgrid]
		#ax.set_ylabel('instrument_spectrum') # SpectralData_label[ix_spectrum]
#		ax.set_xlabel('Wavelength ($\mu$m)')
		ax.set_xlabel('Wavelength (µm)')
#		ax.set_ylabel('$R_p/R_s$ Relative radii (planet-to-star-radius ratios)')
		ax.set_ylabel('Rₚ/Rₛ Relative radii (planet-to-star-radius ratios)')
		# ^ https://en.wikipedia.org/wiki/Unicode_subscripts_and_superscripts

#		ax.legend()
		ax.set_xscale('log')
		ax.set_title('Transmission spectrum')
		#ax.set_title('Transmission spectrum ($\mathbf{Y}$)')
		ax.set_xlim(0.5, 10)
		
		
		ticks = [0.5, 1, 2, 5, 10]
		ax.set_xticks(ticks)
		ax.set_xticklabels(ticks)
		#ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
		ax.get_xaxis().set_major_formatter(plt.FuncFormatter(custom_int_formatter))
	
	return ax



# optional kwargs for `show_obs_transforms` that generates a view over the raw data
raw_data_setup = dict(
	agg_stellar = False,
	agg_photon = False,
	mirror_stacking = False,
	
	window_radius = 0,
	p = 'mean',
	
	grid_pts_num = None,
	grid_pts_spacing = None,
	show_grid = False,
	
	as_relative_radii = False,
	)


def show_obs_transforms(data, planet_ix, agg_stellar=True, agg_photon=True, mirror_stacking=True, ax=None, 
						window_radius=WINDOW_RADIUS, p=WINDOW_AGGR_P,
						grid_pts_num=GRID_W, grid_pts_spacing=GRID_GAP,
						as_relative_radii=True, show_targets=True, show_grid=True,
						zoom_in=False, wlgrid_min=0, wlgrid_max=54):
	
	o = obs_transform(get_observations(data.X, planet_ix=planet_ix),
					   window_radius=window_radius, agg=p,
					   agg_stellar=agg_stellar, agg_photon=agg_photon, mirror_stacking=mirror_stacking,
					   as_relative_radii=as_relative_radii)
#	if not agg_stellar:
#		o = o[0]
##		if not agg_photon:
#		if agg_photon == False or (isinstance(agg_photon, int) and agg_photon == 1):
#			o = o[0]
	while o.shape != (55, 300):
		o = o[0]
	
	rr = get_relative_fluxes(data.y, planet_ix)
	if as_relative_radii:
		rr = get_relative_radii(rr)
	
	colors = sns.color_palette("flare", n_colors=55)
	colors = colors[wlgrid_min : wlgrid_max+1]
	
#	with sns.color_palette("flare", n_colors=55):
	if True:
		if ax is None:
			ax = plt.figure().gca()
		x_range = np.arange(300) + 1
		t_range = slice(30-1, -30) if not zoom_in else slice(130-1, -130)
		
		for ch, colr in zip(o[wlgrid_min : wlgrid_max+1], colors):
			ax.plot(x_range[t_range], ch[t_range], lw=.5, c=colr)
		
		if show_targets:
#			for ch in range(55):
			for ch in range(wlgrid_min, wlgrid_max+1):
#				ax.axhline(rr[ch], color='red', lw=0.2, zorder=0) # ls=':', 
				# adaptive alpha: darken when fewer overlapping lines shown
				alpha = 0.66 - 0.33 * (wlgrid_max - wlgrid_min)/55
				ax.axhline(rr[ch], color='grey', lw=0.4, zorder=0, alpha=alpha)
		
		if show_grid:
			show_t_grid(
				grid_pts_num, grid_pts_spacing, ax.axvline,
				t_range=x_range[t_range][[0, -1]])
		
		#ax.set_title('planet: %d' % planet_ix)
#		ax.set_title('planet: %d; aggregation: p=%s,\nmirror_stacking=%s, agg_stellar=%s' % (
#			planet_ix, repr(p), repr(mirror_stacking), repr(agg_stellar)))

		#title = 'planet: %d; aggregation: p=%s' % (planet_ix, repr(p))
		#ax.text(.5, .96, title, horizontalalignment='center', transform=ax.transAxes)
#		title = '%s planet %d' % (data.dataset_name, planet_ix)
#		ax.text(.025, .96, title, horizontalalignment='left', transform=ax.transAxes,
#			   bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 2.5} if not agg_photon else None)
	
	set_lightcurve_xy_labels(**locals())
	
	return ax



def set_lightcurve_xy_labels(ax, set_ylabel=True, **kwargs):
	ax.set_xlabel('Time (minutes)')
	
	if kwargs['as_relative_radii']:
		ylabel = 'Relative radii (planet-to-star-radius ratios)'
	else:
		ylabel = 'Relative flux'
		#ylabel = 'Relative brightness'
		if not kwargs['zoom_in']:
			ax.axhline(y=1.0, ls=':', c='black')
	
	if set_ylabel:
		ax.set_ylabel(ylabel)
	
	ax.set_title('Light curve')
	#ax.set_title("Light curve ($\mathbf{X}'$)")



def showcase_obs_transforms_xy(data, planet_selection, ylims=None, figsize=5, **kwargs):
	
	# default to showing a single planet's light curve and its spectrum,
	# with the same dataset used as the source for the data shown in both plots.
	if not isinstance(data, list):
		data = [data]*2
	if not isinstance(planet_selection, list):
		planet_selection = [planet_selection, 'spectrum']
	
	obs_args = dict(
		agg_stellar = False,
		mirror_stacking = True,
		show_grid = True,
		as_relative_radii = True,
		zoom_in = False,
		wlgrid_min=0,
		wlgrid_max=54,
		)
	obs_args.update(kwargs)

#	with sns.plotting_context('notebook'): # , font_scale=1.1
	if True:
#		with sns.color_palette("flare", n_colors=55):
		if True:
			
			fig, axs = create_panel(2, 1, figsize)
			
			for i,(d, p, ax) in enumerate(zip(data, planet_selection, axs)):
				
				# option to have panel on the right be a spectrum plot
				if i == 1 and p == 'spectrum':
					for i_,(d_, p_) in enumerate(zip(data[:1], planet_selection[:1])):
						show_spectrum(
							get_relative_radii(get_relative_fluxes(d_.y, p_)),
							#label='%s planet %d: %s' % (d_.dataset_name, p_, d_.planet_mcs19.loc[p_, 'Planet Name']),
							#c=['C0', 'C1'][i_],
							c=sns.color_palette("Set2")[i_],
							ax=ax, lw=2,
							wlgrid_min = obs_args['wlgrid_min'],
							wlgrid_max = obs_args['wlgrid_max'],)
					continue
				
				show_obs_transforms(data=d, planet_ix=p, ax=ax, **obs_args)
				
				set_lightcurve_xy_labels(ax, set_ylabel=i == 0, **obs_args)
				
				if ylims == 'show':
					print(ax.get_ylim())
				elif ylims is not None:
					ax.set_ylim(*ylims[i])
			
			fig.tight_layout()
	
	return fig, axs



def visualize_setup(data, setup, planet_id):
	"""
	Generate a visualization of how a certain data pre-processing
	setup transforms a planet's observations.
	"""
	grid_pts_num = setup['GRID_PTS_NUM']
	if setup['MIRROR_STACKING']:
		grid_pts_num = (grid_pts_num, 0)

	showcase_obs_transforms_xy(
		data,
		planet_id,
		
		# configure aggregation
		window_radius = setup['WINDOW_RADIUS'],
		agg_photon = setup['AGGR_NUM_OBS'],
		mirror_stacking = setup['MIRROR_STACKING'],
		p = setup['MEAN_TYPE_P'],

		# configure grid
		grid_pts_num = grid_pts_num,
		grid_pts_spacing = setup['GRID_PTS_SPACING'],
		
		# display options:
		show_grid = True,
		show_targets = True,
		
		as_relative_radii = False)


