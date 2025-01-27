#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 12:30:15 2024

@author: Miguel Llamas Lanza
"""

import pandas as pd
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

try:
	from early_tde_classification.config import Config

except ModuleNotFoundError:
	from config import Config


from light_curve.light_curve_py import RainbowFit

logging.basicConfig(encoding='utf-8', level=logging.INFO)


def load_data_full_lightcurves(object_list = None, alert_list = None):
	"""
	Load data containing concatenated lightcurves

	Parameters
	----------
	object_list : list of strings, optional
		List of object IDs to include. The default is None (include all loaded objects).
	alert_list : list of strings, optional
		List of alert IDs to include. The default is None (include all loaded alerts).

	Returns
	-------
	df : pd. DataFrame
		Dataframe with input data. Each row contains the LC (and other info) for one alert.

	"""

	df = pd.read_parquet(os.path.join(Config.INPUT_DIR,
							   Config.INPUT_DATA_FNAME))

	df.columns = df.columns.str.lstrip('i:')  # strip prefix
	if object_list:
		df = df[df.objectId.isin(object_list)]
	if alert_list:
		df = df[df.candid.isin(alert_list)]

	return df


def is_sorted(a):
	"""
	Check whether array "a" is sorted. Returns True or False.

	"""
	return np.all(a[:-1] <= a[1:])


def get_rising_flags_per_filter(mjd, flux, fluxerr, flt, min_data_points=5, list_filters=['g', 'r'],
								low_bound=-10, sigma_rise = 1, sigma_decay=1):
	"""Filter only rising alerts for Rainbow fit.

	Parameters
	----------
	data_all: pd.DataFrame
		Pandas DataFrame with at least ['MJD', 'FLT', 'FLUXCAL', 'FLUXCALERR']
		as columns.
	min_data_points: int (optional)
		Minimum number of data points in all filters. Default is 5.
	list_filters: list (optional)
		List of filters to consider. Default is ['g', 'r'].
	low_bound: float (optional)
		Lower bound of FLUXCAL to consider. Default is -10.
	sigma_rise: float (optional)
		Sigma value to consider for the rising cut.

	Returns
	-------
	filter_flags: dict
		Dictionary if light curve survived selection cuts.
	"""

	if is_sorted(mjd):

		# flags if filter survived selection cuts
		rising_flags = dict([[item, False] for item in list_filters])
		decay_flags = dict([[item, False] for item in list_filters])

		if mjd.shape[0] >= min_data_points:

			for flt_str in list_filters:
				filter_flag = flt == flt_str

				# mask negative flux below low bound
				flux_flag = flux >= low_bound

				final_flag = np.logical_and(filter_flag, flux_flag)

				# select filter
				flux_filter = flux[final_flag]
				fluxerr_filter= fluxerr[final_flag]

				rising_flag = False
				decay_flag = False

				if len(flux_filter)>1:
					lc = pd.DataFrame()
					lc['FLUXCAL'] = flux_filter
					lc['FLUXCALERR'] = fluxerr_filter
					lc['MJD'] = mjd[final_flag]

					# check if it is rising (lower bound of last alert is larger than the smallest upper bound)
					# and not yet decreasing (upper bound of last alert is larger than the largest lower bound)
					sigma_rise_factor = sigma_rise / np.sqrt(2)
					sigma_decay_factor = sigma_decay / np.sqrt(2)
					if len(lc) > 1:
						rising_flag = (
							  (lc['FLUXCAL'].values[-1] - sigma_rise_factor * lc['FLUXCALERR'].values[-1]
						  >np.nanmin(lc['FLUXCAL'].values + sigma_rise_factor * lc['FLUXCALERR'].values)))
						decay_flag = ((lc['FLUXCAL'].values[-1] + sigma_decay_factor*lc['FLUXCALERR'].values[-1])
						  < np.nanmax(lc['FLUXCAL'].values - sigma_decay_factor*lc['FLUXCALERR'].values))

					rising_flags[flt_str] = rising_flag
					decay_flags[flt_str] = decay_flag
				else:
					rising_flags[flt_str] = False
					decay_flags[flt_str] = False
		else:
			for flt_str in list_filters:
				rising_flags[flt_str] = False
				decay_flags[flt_str] = False

	else:
		raise ValueError('MJD is not sorted!')

	return rising_flags, decay_flags


def plot_lightcurve_and_fit(lc_values, filt_list, values_fit, err_fit, feature,
							 post_fit_features, title = ''):
	"""
	Plot lightcurve in all bands, with the rainbow fit.

	Parameters
	----------
	lc_values : 2d array
		Values from the lc (mjd, flux, fluxerr, filter).
	filt_list : list
		list of filters like ['g', 'r', 'i'].
	values_fit : list
		Values from the fit (including chisq).
	err_fit : list
		Errors from the fit.
	title : str, optional
		To put as plot title (e.g. ZTF name + transient type)


	"""

	mjd, flux, fluxerr, _ = lc_values
	X = np.linspace(lc_values[0].min() - 10, lc_values[0].max() + 10, 500)

	ax = plt.figure(figsize=(12, 8)).gca()

	ax.set_xlabel('Time (MDJ)', fontsize = 14)
	ax.set_ylabel('Flux', fontsize = 14)

# 	colors = ['green', 'red', 'black']
	colors = ['#15284F', '#F5622E']
	for idx, flt_str in enumerate(['g', 'r']):

		flt_mask = filt_list == flt_str

		rainbow = feature.model(X, flt_str, *values_fit)

		plt.errorbar(mjd[flt_mask], flux[flt_mask], yerr=fluxerr[flt_mask], fmt='o', alpha=.7,
			   color=colors[idx], label = flt_str)
		plt.plot(X, rainbow, linewidth=2, color=colors[idx])
# 		# Error plots
		if not np.isnan(err_fit).all():
			generated_params = np.random.multivariate_normal(values_fit[:-1], np.diag(err_fit)**2, 1000)
			generated_lightcurves = np.array([feature.model(X, flt_str, *generated_values)
										for generated_values in generated_params])
			generated_envelope = np.nanpercentile(generated_lightcurves, [16, 84], axis=0)
			plt.fill_between(X, generated_envelope[0], generated_envelope[1], alpha=0.2, color=colors[idx])

	# Add text
	plt.text(0.01, 0.8,
				r'Amplitude ($A$)'': {0:.2f}$\pm$ {4:.2f}\n'
				r'Rise time $\tau$:'' {1:.2f}$\pm$ {5:.2f}  \n'
				'Temperature ($T$): {2:.2f}$\pm$ {6:.2f}\n'
				'r_chisq ($\chi^2_r$): ''{3:.2f}'.format(*values_fit[1:], *err_fit[1:]), transform=ax.transAxes)

	plt.text(0.01, 0.6, r'Sigmoid compression ($\psi$):'
					  ' %.2f\nRise Time SNR: %.2f\n'
					  'Amplitude SNR: %.2f\n'
					  'Temperature SNR : %.2f'
					   %(*post_fit_features, err_fit[-1]), transform=ax.transAxes)

	plt.title(title, fontsize = 16)
	plt.legend()


def get_std_and_snr(lc_values, filt_list, filters = ['g', 'r']):
	"""
	Compute std of flux, and of then SNR for the lightcurve, in each filter

	Parameters
	----------
	lc_values : 2d array
		Values from the lc (mjd, flux, fluxerr, filter).
	filt_list : list
		list of filters like ['g', 'r', 'i'].
	filters : list of str, optional
		List of filters for which the values are computed. The default is ['g', 'r'] (without 'i').

	Returns
	-------
	list
		Std and SNR_mean values in each of the filters. e.g. [std_g, std_r, snr_std_g, snr_std_r]

	"""

	# Add general statistics of lc as features
	std_dev = {}
	snr = {}
	for flt_str in filters:
		flt_mask = filt_list == flt_str
		if len(lc_values[1][flt_mask]) > 0:
			std_dev[flt_str] = np.std(lc_values[1][flt_mask])
			snr[flt_str] = np.std(lc_values[1][flt_mask] / lc_values[2][flt_mask])
		else:
			std_dev[flt_str] = snr[flt_str] = -999
	return list(std_dev.values()) + list(snr.values())


def compute_fractional_variability(fluxcal, fluxcalerr):
	"""
	Compute fractional variability from Flux and Fluxerror

	"""

	# Taken from https://www.mdpi.com/2075-4434/7/2/62

	if np.var(fluxcal, ddof=1) - np.mean(fluxcalerr**2) <= 0:
		return np.nan

	return np.sqrt(np.var(fluxcal, ddof=1) - np.mean(fluxcalerr**2)) / np.mean(fluxcal)


def crop_old_history_and_get_Fvar(lc_values):
	"""
	Crop old history from the lightcurve (keep only recent alerts, e.g. from the last 100 days)
	and compute fractional variability.

	Parameters
	----------
	lc_values : 2d array
		Values (mjd, flux, fluxerr, filter) from the lightcurve.

	Returns
	-------
	lc_values : 2d array
		Cropped lc_values.
	Fvar : float
		Fractional variability (https://www.mdpi.com/2075-4434/7/2/62).

	"""

	# Crop everything before a certain date (last alert - Config.days_history_lc)
	mask = lc_values[0] >= (lc_values[0][-1] - Config.days_history_lc)

	if lc_values.shape[1] - sum(mask) <= 1:
		Fvar = 0
	else:

		# Pre-history LC
		prehist_lc = lc_values[:, ~mask]
		Fvar = compute_fractional_variability(prehist_lc[1], prehist_lc[2])

	return lc_values[:, mask], Fvar


def is_lc_on_the_rise(lc_values):
	"""
	Check whether last alert of the lightcurve is on the rise (showing rise and not decay yet)

	Parameters
	----------
	lc_values : 2d array
		Cropped lc_values.

	Returns True or False

	"""

	# Check whether it is rising and not decaying
	rising_flag, decaying_flag = get_rising_flags_per_filter(*lc_values, list_filters = [1, 2],
														  sigma_rise = 2)
	final_flag = any(rising_flag.values()) and not any(decaying_flag.values())
	return final_flag


def get_post_fit_features(lc_values, values_fit, err_fit):
	"""
	Compute and get additional features derived from the Rainbow fit.

	Parameters
	----------
	lc_values : 2d array
		Cropped lc_values.
	values_fit : list
		Values from the fit (including chisq).
	err_fit : list
		Errors from the Rainbow fit.

	Returns list of additional features

	"""

	# Relative distance of the last datapoint to the center of the sigmoid, wrt the rise time
	sigmoid_center_ref = (lc_values[0, -1] - values_fit[0]) / values_fit[2]
	# Get values on the uncertainty of the fitted values (i.e. value divided by error)
	snr_rise_time = values_fit[2] / err_fit[2]
	snr_amplitude = values_fit[1] / err_fit[1]

	return [sigmoid_center_ref, snr_rise_time, snr_amplitude]


def flag_based_on_post_fit_criteria(post_fit_feat, values_fit):
	"""
	Compute a flag to drop values with criteria based on post-fit features.

	Parameters
	----------
	post_fit_feat : list
		list of features computed after the fit.
	values_fit : list
		Values from the Rainbow fit (including chisq).

	Returns True or False

	"""

	sigmoid_center_ref, snr_rise_time, snr_amplitude = post_fit_feat
	flag = (sigmoid_center_ref > Config.sigdist_lim[0] and sigmoid_center_ref < Config.sigdist_lim[1]
		and snr_rise_time > Config.min_snr_features and snr_amplitude > Config.min_snr_features
		#and values_fit[1] < Config.max_ampl
# 		and values_fit[2] < Config.max_risetime
		# and values_fit[3] > Config.min_temp
		and values_fit[4] < Config.max_rchisq)
		# rise time, temperature and rchisq, respectively < 100 days, > 10**4K, and < 10.

	return flag


def extract_features_for_lc(lc_values_orig, feature, min_nb_points_fit = 5,
							show_plots = False, title_plot = '', post_fit_cuts = True):
	"""
	Perform the feature extraction for a specific lightcurve

	Parameters
	----------
	lc_values_orig : 2d array
		Values (mjd, flux, fluxerr, filter) from the lightcurve.
	feature : TYPE
		DESCRIPTION.
	min_nb_points_fit : int, optional
		Mininum number of points required by the fit. The default is 5.
	show_plots : float, optional
		Whether to generate and show plots. The default is False.
	title_plot : string, optional
		Title of the plot, if show_plots is True. The default is ''.
	post_fit_cuts : float, optional
		Whether to apply cuts derived from the fit. The default is True.

	Returns list of features

	"""

	# Create dummy return
	nan_arrays = list(np.full((18), np.nan))
	# Create copy
	lc_values = lc_values_orig.copy()
	# Delete duplicate times
	lc_values = np.delete(lc_values, np.where(np.diff(lc_values[0]) == 0)[0], axis = 1)
	# Crop out old history
	lc_values, Fvar = crop_old_history_and_get_Fvar(lc_values)

	if not (lc_values.shape[1] >= min_nb_points_fit and is_lc_on_the_rise(lc_values)) or Fvar!=0:
		return nan_arrays

	# Convert filter list to g, r, i strings
	filt_list = np.vectorize(Config.filt_conv.get)(lc_values[3].astype(int))

	# Fit
	try:
		values_fit, err_fit = feature._eval_and_get_errors(t=lc_values[0], m=lc_values[1],
												  sigma = lc_values[2], band = filt_list)
		# Get general statistics of lc as features: std_g, std_r, snr_g, snr_r
		std_and_snr = get_std_and_snr(lc_values, filt_list)
		# Add postfit features
		post_fit_feat = get_post_fit_features(lc_values, values_fit, err_fit)

		if show_plots:
			plot_lightcurve_and_fit(lc_values, filt_list, values_fit, err_fit, feature,
						   post_fit_feat, title = title_plot)

		if post_fit_cuts and not flag_based_on_post_fit_criteria(post_fit_feat, values_fit):
			return nan_arrays

		# Plot
		if show_plots:
			plot_lightcurve_and_fit(lc_values, filt_list, values_fit, err_fit, feature,
						   post_fit_feat, title = title_plot)

		return list(values_fit) + list(err_fit) + std_and_snr + post_fit_feat + [Fvar, lc_values.shape[1]]

	except RuntimeError:
		return nan_arrays


def feature_extractor_for_row_df(row_obj, feature, min_nb_points_fit = 5, show_plots = False):
	"""
	Extract features with rainbow for one row of the DataFrame (row_obj).

	Parameters
	----------
	row_obj : pd.Series
		DESCRIPTION.
	feature : TYPE
		DESCRIPTION.
	min_nb_points_fit : int, optional
		Mininum number of points required by the fit. The default is 5.
	show_plots : float, optional
		Whether to generate and show plots. The default is False.

	Returns  (list)
	-------
		Output including metadata (name, alert ID, transient type) and features.

	"""

	name = row_obj.objectId
	trans_type = row_obj.type
	alertid = row_obj.candid

	lc_values = np.stack(row_obj[['jd', 'FLUXCAL', 'FLUXCALERR', 'fid']])

	# Delete duplicate times
	lc_values = np.delete(lc_values, np.where(np.diff(lc_values[0]) == 0)[0], axis = 1)

	# sort by MJD
	lc_values = lc_values[:, lc_values[0, :].argsort()]

	out_feat = extract_features_for_lc(lc_values, feature, min_nb_points_fit, show_plots,
						title_plot = 'objectId: %s. Transient type: %s. ' %(name, trans_type))

	return [name, alertid, trans_type] + out_feat


def load_data_and_extract_features(save = True, show_plots = False, object_list = None,
								alert_list = None):
	"""
	Main function to extract the features.

	Parameters
	----------
	save : float, optional
		Whether to save features to a csv. The default is True.
	show_plots : float, optional
		Whether to generate and show plots. The default is False.
	object_list : list of strings, optional
		List of object IDs to include. The default is None (include all loaded objects).
	alert_list : list of strings, optional
		List of alert IDs to include. The default is None (include all loaded alerts).


	Returns
	-------
	feature_matrix : pd.DataFrame
		output data, including the extracted features and metadata for each lightcurve.

	"""

	# Initialise Rainbow
	feature = RainbowFit.from_angstrom(Config.band_wave_aa, with_baseline = False,
									temperature='constant', bolometric='sigmoid')
	columns_feat = ['objectId', 'alertId', 'type',  # 'norm',
				  'ref_time', 'amplitude', 'rise_time', 'temperature', 'r_chisq',
					'err_ref_time', 'err_amplitude', 'err_rise_time', 'err_temperature',
					'std_flux_g', 'std_flux_r', 'std_snr_g', 'std_snr_r',
					'sigmoid_dist', 'snr_rise_time', 'snr_amplitude', 'Fvar', 'nb_points']

	feature_matrix = pd.DataFrame([], columns = columns_feat)

	# Load data
	all_data = load_data_full_lightcurves(object_list = object_list,
														 alert_list = alert_list)

	# Extract features
	feature_matrix[columns_feat] = all_data.apply(
		lambda x: feature_extractor_for_row_df(x, feature, show_plots = show_plots),
				result_type = 'expand', axis = 1)

	feature_matrix.dropna(inplace = True)

	# Save features into csv in general folder
	if save:
		feature_matrix.to_csv(os.path.join(Config.OUT_FEATURES_DIR, 'features.csv'),
							  index = False)
	return feature_matrix


if __name__ == '__main__':

	start = dt.datetime.now()

	feature_data = load_data_and_extract_features(save = True, show_plots = False)

	logging .info("Done in {} seconds.".format(dt.datetime.now() - start))
