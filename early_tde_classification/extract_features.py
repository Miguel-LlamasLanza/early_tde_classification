#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 12:30:15 2024

@author: Miguel Llamas Lanza
"""

import pandas as pd
import os
import glob
import logging
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt


from conversion_tools import mag2fluxcal_snana, add_alert_history_to_df
from light_curve.light_curve_py import RainbowFit
from config import Config

logging.basicConfig(encoding='utf-8', level=logging.INFO)


def load_zenodo_data(which_data = 'all_zenodo', keep_only_one_per_object = False, min_points_fit = 5,
					 nb_files = None):

	path_parquet_files = Config.ZENODO_DATA_DIR
	if which_data == 'all_zenodo':
		all_parquet_fnames = glob.glob(os.path.join(path_parquet_files, '*.parquet'))
	else:  # if TNS or SIMBAD
		all_parquet_fnames = glob.glob(os.path.join(path_parquet_files,
											  '*{}*.parquet'.format(which_data)))
	if nb_files:
		all_parquet_fnames = all_parquet_fnames[:nb_files]

	logging.info('Loading parquet files')
	all_obj_df = pd.concat([pd.read_parquet(parquet_fname) for parquet_fname in all_parquet_fnames])

	# Get lenngth of alerts history per alert
	all_obj_df['length'] = all_obj_df['cjd'].apply(lambda x: len(x))

	if keep_only_one_per_object:
		# Keep only row with all alerts
		all_obj_df.sort_values(['objectId', 'length'], inplace = True)
		names = np.array(all_obj_df['objectId'])
		mask = np.append((names[:-1] != names[1:]), True)
		all_obj_df = all_obj_df[mask]

	# Drop all rows with less points than needed for the fit
	all_obj_df = all_obj_df[all_obj_df.length >= min_points_fit]

	# Rename column
	all_obj_df.rename(columns = {'TNS': 'type',
								  'cdsxmatch': 'type'}, inplace = True)

	logging.info('All zenodo files loaded')

	return all_obj_df


def load_tdes_ztf(min_points_fit = 5):

	df = pd.read_csv(os.path.join(Config.ZTF_TDE_DATA_DIR, 'all_tde_in_ztf.csv'), dtype={'id': str})
	df['type'] = 'TDE'
	df.drop(columns = 'id', inplace=True)

	# Convert filter name (str) to filter ID (int)
	inverse_filt_conv = {v: k for k, v in Config.filt_conv.items()}
	df['FLT'] = np.vectorize(inverse_filt_conv.get)(df.FLT.astype(str))

	# Add alert history
	df = add_alert_history_to_df(df)

	# Get lenngth of alerts history per alert
	df['length'] = df['cMJD'].apply(lambda x: len(x))
	# Drop all rows with less points than needed for the fit
	df = df[df['length'] >= min_points_fit]
	# Create a dummy candid based on the length
	df['candid'] = df['length']

	return df


def average_intraday_data(df_intra):
	"""Average over intraday data points

	 Parameters
	 ----------
	 df_intra: pd.DataFrame
		containing the history of the flux
		with intraday data

	 Returns
	 -------
	 df_average: pd.DataFrame
		containing only daily data
	"""

	df_average = df_intra.copy()
	df_average['MJD'] = df_average['MJD'].apply(
		lambda x: np.around(x, decimals=0))
	df_average = df_average.groupby('MJD').mean()
	df_average['MJD'] = df_average.index.values

	return df_average


def is_sorted(a):
	return np.all(a[:-1] <= a[1:])


def get_rising_flags_per_filter(mjd, flux, fluxerr, flt,
						min_data_points=5, list_filters=['g', 'r'], low_bound=-10):
	"""Filter only rising alerts for Rainbow fit.

	Parameters
	----------
	data_all: pd.DataFrame
		Pandas DataFrame with at least ['MJD', 'FLT', 'FLUXCAL', 'FLUXCALERR']
		as columns.
	min_data_points: int (optional)
		Minimum number of data points in all filters. Default is 7.
	list_filters: list (optional)
		List of filters to consider. Default is ['g', 'r'].
	low_bound: float (optional)
		Lower bound of FLUXCAL to consider. Default is -10.

	Returns
	-------
	filter_flags: dict
		Dictionary if light curve survived selection cuts.
	"""

	if is_sorted(mjd):

		# flags if filter survived selection cuts
		filter_flags = dict([[item, False] for item in list_filters])

		if mjd.shape[0] >= min_data_points:

			for i in list_filters:
				filter_flag = flt == i

				# mask negative flux below low bound
				flux_flag = flux >= low_bound

				final_flag = np.logical_and(filter_flag, flux_flag)

				# select filter
				flux_filter = flux[final_flag]
				fluxerr_filter= fluxerr[final_flag]

				rising_flag = False
				if len(flux_filter)>1:
					lc = pd.DataFrame()
					lc['FLUXCAL'] = flux_filter
					lc['FLUXCALERR'] = fluxerr_filter
					lc['MJD'] = 0.1 * mjd[final_flag]

					# check if it is rising (lower bound of last alert is larger than the smallest upper bound)
					# and not yet decreasing (upper bound of last alert is larger than the largest lower bound)
					avg_data = average_intraday_data(lc)

					if len(avg_data) > 1:
						rising_flag = ((avg_data['FLUXCAL'].values[-1]+avg_data['FLUXCALERR'].values[-1])\
										> np.nanmax(avg_data['FLUXCAL'].values-avg_data['FLUXCALERR'].values))\
										& (avg_data['FLUXCAL'].values[-1]-avg_data['FLUXCALERR'].values[-1]\
										 >np.nanmin(avg_data['FLUXCAL'].values+avg_data['FLUXCALERR'].values))

					filter_flags[i] = rising_flag
				else:
					filter_flags[i] = False
		else:
			for i in list_filters:
				filter_flags[i] = False

	else:
		raise ValueError('MJD is not sorted!')

	return filter_flags


def plot_lightcurve_and_fit(lc_values, filt_list, values_fit, err_fit, feature, title = ''):
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
		DESCRIPTION.
	title : str, optional
		To put as plot title (e.g. ZTF name + transient type)


	"""

	mjd, flux, fluxerr, _ = lc_values
	X = np.linspace(lc_values[0].min() - 10, lc_values[0].max() + 10, 500)

	plt.figure(figsize=(12, 8))
	# ax = plt.gca()

	colors = ['green', 'red', 'black']
	for idx, i in enumerate(['g', 'r', 'i']):

		mask = filt_list == i
		f = flux[mask]
		ferr = fluxerr[mask]
		t = mjd[mask]

		rainbow = feature.model(X, i, *values_fit)

		plt.errorbar(t, f, yerr=ferr, fmt='o', alpha=.7, color=colors[idx])
		plt.plot(X, rainbow, linewidth=5, label=i, color=colors[idx])
# 		# Error plots
# 		generated_params = np.random.multivariate_normal(values_fit[:-1], np.diag(err_fit)**2, 1000)
# 		generated_lightcurves = np.array([feature.model(X, i, generated_values)
# 									for generated_values in generated_params])
# 		generated_envelope = np.nanpercentile(generated_lightcurves, [16, 84], axis=0)
# 		plt.fill_between(X, generated_envelope[0], generated_envelope[1], alpha=0.2, color=colors[idx])

		plt.title(title)


def feature_extractor_for_row_df(row_obj, feature, flux_conv_required = True,
								 min_nb_points_fit = 7, show_plots = False):
	"""
	Extract features with rainbow for one row of the DataFrame (row_obj).

	Parameters
	----------
	row_obj : pd.Series
		DESCRIPTION.
	feature : TYPE
		DESCRIPTION.
	min_nb_points_fit : TYPE, optional
		DESCRIPTION. The default is 5.
	show_plots : TYPE, optional
		DESCRIPTION. The default is False.

	Returns
	-------
	TYPE
		DESCRIPTION.

	"""

	name = row_obj.objectId
	trans_type = row_obj.type
	alertid = row_obj.candid

	if flux_conv_required:
		lc_values = np.stack(row_obj[['cjd', 'cmagpsf', 'csigmapsf', 'cfid']])
		# Remove alerts with nan values
		lc_values = lc_values[:, ~np.isnan(lc_values).any(axis=0)]
		# Flux conversion
		lc_values[1], lc_values[2] = mag2fluxcal_snana(lc_values[1], lc_values[2])
	else:
		lc_values = np.stack(row_obj[['cMJD', 'cFLUXCAL', 'cFLUXCALERR', 'cFLT']])

	# Delete duplicate times
	lc_values = np.delete(lc_values, np.where(np.diff(lc_values[0]) == 0)[0], axis = 1)
	# Check whether it is rising
	rising_flag = any(get_rising_flags_per_filter(*lc_values, list_filters = [1, 2]).values())

	if (rising_flag) & (lc_values.shape[1] >= min_nb_points_fit):

		# Normalization
		norm = np.max(lc_values[1])
		lc_values[1], lc_values[2] = lc_values[1] / norm, lc_values[2] / norm

		# Convert filter list to g, r, i strings
		filt_list = np.vectorize(Config.filt_conv.get)(lc_values[3].astype(int))

		# Fit
# 		values_fit, err_fit = feature(*lc_values[:-1], filt_list)
		try:
			values_fit = feature(*lc_values[:-1], filt_list)
		except RuntimeError:
			values_fit = list(np.full((5), np.nan))


# 		# Plot
		if show_plots:
# 			plot_lightcurve_and_fit(lc_values, filt_list, values_fit, err_fit, feature,
# 							title = 'objectId: %s. Transient type: %s. '%(name, trans_type))
			plot_lightcurve_and_fit(lc_values, filt_list, values_fit, None, feature,
							title = 'objectId: %s. Transient type: %s. '%(name, trans_type))

# 		return [name, alertid, trans_type] + list(values_fit) + list(err_fit)
		return [name, alertid, trans_type] + list(values_fit)

# 	return [name, alertid, trans_type] + list(np.full((9), np.nan))
	return [name, alertid, trans_type] + list(np.full((5), np.nan))



def extract_features_tdes(save = True):
	"""
	Extract features (with rainbow) for TDEs from ZTF, from lightcurves (one lightcurve per alert)

	Parameters
	----------
	save : bool, optional
		Whether to save the features into a csv. The default is True.

	Returns
	-------
	feature_matrix : pd.DataFrame
		Rainbow features extracted.

	"""
	# Initialise
	feature = RainbowFit.from_angstrom(Config.band_wave_aa, with_baseline = False,
									temperature='constant', bolometric='sigmoid')
# 	columns_feat = ['objId', 'alertId', 'type', 'ref_time', 'amplitude', 'rise_time', 'temperature',
# 					'r_chisq',
# 					'err_ref_time', 'err_amplitude', 'err_rise_time', 'err_temperature']
#
	columns_feat = ['objId', 'alertId', 'type', 'ref_time', 'amplitude', 'rise_time', 'temperature',
					'r_chisq']
	feature_matrix = pd.DataFrame([], columns = columns_feat)

	# Load
	ztf_tde_data = load_tdes_ztf()

	# Get features
	feature_matrix[columns_feat] = ztf_tde_data.apply(
		lambda x: feature_extractor_for_row_df(x, feature, flux_conv_required=False,
										  show_plots = True), result_type = 'expand', axis = 1)
	feature_matrix.dropna(inplace = True)
	feature_matrix['data_origin'] = 'ztf_tdes'

	# Save features into csv
	if save:
		feature_matrix.to_csv(os.path.join(Config.OUT_FEATURES_DIR, 'features_tdes_ztf_noerrors.csv'),
							  index = False)
	return feature_matrix


def extract_features_nontdes_zenodo(which_data, save = True, nb_files = None):
	"""
	Extract features (with rainbow) for non-TDE objects from Zenodo dataset (one LC per alert)


	Parameters
	----------
	which_data : str
		Whether to load 'simbad', 'tns' or 'all_zenodo' parquet files.
	save : bool, optional
		whether to save into csv file. The default is True.
	nb_files : int, optional
		Maximum number of parquet files to load. If None (by default), all files are taken.

	Returns
	-------
	feature_matrix : pd.DataFrame
		Rainbow features extracted.

	"""

	# Initialise
	feature = RainbowFit.from_angstrom(Config.band_wave_aa, with_baseline = False)
	columns_feat = ['objId', 'alertId', 'type', 'ref_time', 'amplitude', 'rise_time', 'temperature',
					'r_chisq',
					'err_ref_time', 'err_amplitude', 'err_rise_time', 'err_temperature']
	feature_matrix = pd.DataFrame([], columns = columns_feat)

	# Load
	zenodo_data = load_zenodo_data(which_data, nb_files = nb_files)

	# Get features
	feature_matrix[columns_feat] = zenodo_data.apply(
		lambda x: feature_extractor_for_row_df(x, feature, show_plots = False),
												result_type = 'expand', axis = 1)
	feature_matrix.dropna(inplace = True)
	feature_matrix['data_origin'] = which_data

	# Save features into csv
	if save:
		feature_matrix.to_csv(os.path.join(
			Config.OUT_FEATURES_DIR, 'features_{}.csv'.format(which_data)), index = False)
	return feature_matrix


def extract_features(data_origin, **kwargs):
	"""
	Main function to extract the features.

	Parameters
	----------
	data_origin : str
		Which data to process. 'tdes_ztf', 'tns', 'simbad', or 'all'.
	**kwargs: Other arguments, such as save, or nb_files.

	"""


	if data_origin == 'tdes_ztf':
		extract_features_tdes(**kwargs)
	elif data_origin in ['simbad', 'tns']:
		extract_features_nontdes_zenodo(data_origin, **kwargs)
	elif data_origin == 'all':
		extract_features_tdes(**kwargs)
		extract_features_nontdes_zenodo('simbad', **kwargs)
		extract_features_nontdes_zenodo('tns', **kwargs)

if __name__ == '__main__':

	start = dt.datetime.now()

	extract_features('tdes_ztf')

	logging .info("Done in {} seconds.".format(dt.datetime.now() - start))
