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


def load_zenodo_data(which_data = 'all_zenodo', keep_only_one_per_object = False,
					 min_points_fit = 5, nb_files = None):
	"""
	Load data from parquet files downloaded from zenodo (from SNIa paper)

	Parameters
	----------
	which_data : str, optional
		"Whether to load 'simbad', 'tns' or 'all_zenodo' parquet files.". The default is 'all_zenodo'.
	keep_only_one_per_object : bool, optional
		Keep only last alert of the object. The default is False. NOT USED CURRENTLY !!
	min_points_fit : int, optional
		Min number of datapoints in the LC required for the fit, in all filters. The default is 5.
	nb_files : int (or None), optional
		(maximum) number of parquet files to load. The default value "None" loads all files.

	Returns
	-------
	all_obj_df : pd.DataFrame
		Dta loaded from zenodo.

	"""

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
		# Keep only row with all alerts  (not used anymore here, but left to see if it is worth using after extracting feat)
		all_obj_df.sort_values(['objectId', 'length'], inplace = True)
		names = np.array(all_obj_df['objectId'])
		mask = np.append((names[:-1] != names[1:]), True)
		all_obj_df = all_obj_df[mask]

	# Drop all rows with less points than needed for the fit
	all_obj_df = all_obj_df[all_obj_df.length >= min_points_fit]

	# Drop duplicates
	all_obj_df.drop_duplicates(subset = 'candid', inplace = True)

	# Rename column
	all_obj_df.rename(columns = {'TNS': 'type',
								  'cdsxmatch': 'type'}, inplace = True)

	logging.info('All zenodo files loaded')

	return all_obj_df


def load_tdes_ztf(min_points_fit = 5):
	"""
	Load data from TDEs csv (and format it as required)

	Parameters
	----------
	min_points_fit : int, optional
		Min number of datapoints in the LC required for the fit, in all filters. The default is 5.

	Returns
	-------
	df : pd.DataFrame
		Dataset for TDEs.

	"""

	df = pd.read_csv(os.path.join(Config.ZTF_TDE_DATA_DIR, 'all_tde_in_ztf.csv'), dtype={'id': str})
	df['type'] = 'TDE'
	df.drop(columns = 'id', inplace=True)

	# Remove outliers
	df = df[~df.objectId.isin(['ZTF18acpdvos', 'ZTF18aabtxvd', 'ZTF18aahqkbt'])]
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
	df['candid'] = df['objectId'] + '-' + df['length'].astype(str)

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
						min_data_points=5, list_filters=['g', 'r'], low_bound=-10, sigma_rise = 1):
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
					sigma_factor = sigma_rise / np.sqrt(2)
					if len(avg_data) > 1:
						rising_flag = (
							((avg_data['FLUXCAL'].values[-1] + avg_data['FLUXCALERR'].values[-1])
							> np.nanmax(avg_data['FLUXCAL'].values - avg_data['FLUXCALERR'].values))
							& (avg_data['FLUXCAL'].values[-1] - sigma_factor * avg_data['FLUXCALERR'].values[-1]
							>np.nanmin(avg_data['FLUXCAL'].values + sigma_factor * avg_data['FLUXCALERR'].values)))

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
	for idx, flt_str in enumerate(['g', 'r', 'i']):

		flt_mask = filt_list == flt_str

		rainbow = feature.model(X, flt_str, *values_fit)

		plt.errorbar(mjd[flt_mask], flux[flt_mask], yerr=fluxerr[flt_mask], fmt='o', alpha=.7,
			   color=colors[idx])
		plt.plot(X, rainbow, linewidth=5, label=flt_str, color=colors[idx])
# 		# Error plots
		if not np.isnan(err_fit).all():
			generated_params = np.random.multivariate_normal(values_fit[:-1], np.diag(err_fit)**2, 1000)
			generated_lightcurves = np.array([feature.model(X, flt_str, *generated_values)
										for generated_values in generated_params])
			generated_envelope = np.nanpercentile(generated_lightcurves, [16, 84], axis=0)
			plt.fill_between(X, generated_envelope[0], generated_envelope[1], alpha=0.2, color=colors[idx])

		plt.title(title)


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
	# Crop to keep the last 30 days only, to simulate alerts
	last_good_index = np.where(lc_values[0] > lc_values[0][-1] - 30)[0][0]
	lc_values = lc_values[:, last_good_index:]

	# Check whether it is rising
	rising_flag = any(get_rising_flags_per_filter(*lc_values, list_filters = [1, 2]).values())

	if (rising_flag) & (lc_values.shape[1] >= min_nb_points_fit):

		# Normalization
		norm = np.max(lc_values[1])
		lc_values[1], lc_values[2] = lc_values[1] / norm, lc_values[2] / norm

		# Convert filter list to g, r, i strings
		filt_list = np.vectorize(Config.filt_conv.get)(lc_values[3].astype(int))

		# Fit
		try:
			values_fit, err_fit = feature._eval_and_get_errors(t=lc_values[0], m=lc_values[1],
													  sigma = lc_values[2], band = filt_list)
		except RuntimeError:
			values_fit = list(np.full((5), np.nan))
			err_fit = list(np.full((4), np.nan))

		# Add general statistics of lc as features: std_g, std_r, snr_g, snr_r
		std_and_snr = get_std_and_snr(lc_values, filt_list)

		# Plot
		if show_plots:
			plot_lightcurve_and_fit(lc_values, filt_list, values_fit, err_fit, feature,
							title = 'objectId: %s. Transient type: %s. ' %(name, trans_type))

		return [name, alertid, trans_type,
				   len(lc_values[1]), norm] + list(values_fit) + list(err_fit) + std_and_snr
	return [name, alertid, trans_type, len(lc_values[1])] + list(np.full((14), np.nan))


def keep_only_feat_last_alert_per_object(feature_matrix, input_data):
	"""
	Function to drop all alerts except the last one for each object, in the features dataframe.

	Parameters
	----------
	feature_matrix : pd.DataFrame
		Feature Dataframe obtained after applying cuts and fitting with Rainbow. It should have at
		least 'alertId' and 'objId' columns
	input_data : pd.DataFrame
		Input df corresponding to the features. Should have at least 'length' and 'candid' columns.

	Returns
	-------
	feature_matrix : pd.DataFrame
		Features df after with only the last alert per object.

	"""

	merged = feature_matrix[['alertId', 'objId']].merge(input_data[['length', 'candid']], how = 'left',
							   left_on = 'alertId', right_on='candid')
	feature_matrix = feature_matrix.iloc[merged.groupby('objId')['length'].idxmax()]
	return feature_matrix


def get_final_feature_dataframe_and_save(feature_matrix, input_df, save, keep_only_last_alert,
										 data_origin):
	"""
	Common 'postprocessing' function after feature extraction for TDEs and other objects.

	Saves (if 'save'==True) all features in csv inside 'all_alerts_per_object' folder and filters
	'features_matrix' to only one alert per object if 'keep_only_last_alert' is True. In this case,
	also saves the reduced dataframe into csv in 'one_alert_per_object' folder.

	Finally, it saves (if 'save' is True) the resulting feature_matrix df in the main output folder.

	Parameters
	----------
	feature_matrix : pd.DataFrame
		Extracted features matrix.
	input_df : pd.DataFrame
		Input df with alert data, which was used to obtain feature_matrix.
	save : bool
		Whether to save the features into a csv.
	keep_only_last_alert : bool
		Keep only last alert that passed the cuts per object
	which_data : str
		Whether data is load 'simbad', 'tns' or 'tdes_ztf'

	Returns
	-------
	feature_matrix : pd.DataFrame
		Extracted features matrix.

	"""

	if save:
		# Save all features
		os.mkdirs(os.path.join(Config.OUT_FEATURES_DIR, 'all_alerts_per_object'), exist_ok = True)
		feature_matrix.to_csv(os.path.join(Config.OUT_FEATURES_DIR, 'all_alerts_per_object',
									 'features_tdes_ztf.csv'), index = False)

	if keep_only_last_alert:
		# Keep only last alert that passed the cut per object
		feature_matrix = keep_only_feat_last_alert_per_object(feature_matrix, input_df)
		if save:
			os.mkdirs(os.path.join(Config.OUT_FEATURES_DIR, 'one_alert_per_object'), exist_ok = True)
			feature_matrix.to_csv(os.path.join(Config.OUT_FEATURES_DIR, 'one_alert_per_object',
									 'features_tdes_ztf.csv'), index = False)

	# Save features into csv in general folder
	if save:
		feature_matrix.to_csv(os.path.join(Config.OUT_FEATURES_DIR, 'features_tdes_ztf.csv'),
							  index = False)
	return feature_matrix


def extract_features_tdes(save = True, show_plots = False, keep_only_last_alert = False):
	"""
	Extract features (with rainbow) for TDEs from ZTF, from lightcurves (one lightcurve per alert)

	Parameters
	----------
	save : bool, optional
		Whether to save the features into a csv. The default is True.
	keep_only_last_alert : bool, optional
		Keep only last alert that passed the cuts per object. The default is False.

	Returns
	-------
	feature_matrix : pd.DataFrame
		Rainbow features extracted.

	"""
	# Initialise
	feature = RainbowFit.from_angstrom(Config.band_wave_aa, with_baseline = False,
									temperature='constant', bolometric='sigmoid')
	columns_feat = ['objId', 'alertId', 'type', 'nb_points', 'norm',
				  'ref_time', 'amplitude', 'rise_time', 'temperature', 'r_chisq',
					'err_ref_time', 'err_amplitude', 'err_rise_time', 'err_temperature',
					'std_flux_g', 'std_flux_r', 'std_snr_g', 'std_snr_r']

	feature_matrix = pd.DataFrame([], columns = columns_feat)

	# Load
	ztf_tde_data = load_tdes_ztf()

	# Get features
	feature_matrix[columns_feat] = ztf_tde_data.apply(
		lambda x: feature_extractor_for_row_df(x, feature, flux_conv_required=False,
										  show_plots = show_plots), result_type = 'expand', axis = 1)
	feature_matrix.dropna(inplace = True)
	feature_matrix['data_origin'] = 'tdes_ztf'

	feature_matrix = get_final_feature_dataframe_and_save(feature_matrix, ztf_tde_data, save,
													   keep_only_last_alert, 'tdes_ztf')
	return feature_matrix


def extract_features_nontdes_zenodo(which_data, save = True, nb_files = None, show_plots = False,
									keep_only_last_alert = False):
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
	keep_only_last_alert : bool, optional
		Keep only last alert that passed the cuts per object. The default is False.

	Returns
	-------
	feature_matrix : pd.DataFrame
		Rainbow features extracted.

	"""

	# Initialise
	feature = RainbowFit.from_angstrom(Config.band_wave_aa, with_baseline = False,
									temperature='constant', bolometric='sigmoid')
	columns_feat = ['objId', 'alertId', 'type', 'nb_points', 'norm',
				  'ref_time', 'amplitude', 'rise_time', 'temperature', 'r_chisq',
					'err_ref_time', 'err_amplitude', 'err_rise_time', 'err_temperature',
					'std_flux_g', 'std_flux_r', 'std_snr_g', 'std_snr_r']
	feature_matrix = pd.DataFrame([], columns = columns_feat)

	# Load
	zenodo_data = load_zenodo_data(which_data, nb_files = nb_files)

	# Get features
	feature_matrix[columns_feat] = zenodo_data.apply(
		lambda x: feature_extractor_for_row_df(x, feature, show_plots = show_plots),
												result_type = 'expand', axis = 1)
	feature_matrix.dropna(inplace = True)
	feature_matrix['data_origin'] = which_data
	feature_matrix = get_final_feature_dataframe_and_save(feature_matrix, zenodo_data, save,
													   keep_only_last_alert, 'tdes_ztf')
	return feature_matrix


def extract_features(data_origin, max_nb_files_simbad = None, **kwargs):
	"""
	Main function to extract the features.

	Parameters
	----------
	data_origin : str
		Which data to process. 'tdes_ztf', 'tns', 'simbad', or 'all'.
	**kwargs: Other arguments, such as save, nb_files or show_plots.

	"""

	if data_origin == 'tdes_ztf':
		return extract_features_tdes(**kwargs)
	elif data_origin in ['simbad', 'tns']:
		return extract_features_nontdes_zenodo(data_origin, nb_files = max_nb_files_simbad, **kwargs)
	elif data_origin == 'all':
		feat_tdes = extract_features_tdes(**kwargs)
		feat_simbad = extract_features_nontdes_zenodo('simbad', nb_files = max_nb_files_simbad, **kwargs)
		feat_tns = extract_features_nontdes_zenodo('tns', **kwargs)

		# Merge and save features
		all_features = pd.concat([feat_tdes, feat_simbad, feat_tns])
		if "save" in kwargs:
			all_features.to_csv(os.path.join(
				Config.OUT_FEATURES_DIR, 'features_all.csv'), index = False)
		return all_features
	else:
		print('Wrong string given as data origin. Must be "simbad", "tns", "tdes_ztf" or "all"')


if __name__ == '__main__':

	start = dt.datetime.now()

	extract_features('tdes_ztf', max_nb_files_simbad = 2, keep_only_last_alert=True, save = True,
				   show_plots = False)

	logging .info("Done in {} seconds.".format(dt.datetime.now() - start))
