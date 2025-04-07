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
	from early_tde_classification.conversion_tools import add_alert_history_to_df
	from early_tde_classification.config import Config

except ModuleNotFoundError:
	from conversion_tools import add_alert_history_to_df
	from config import Config


from light_curve.light_curve_py import RainbowFit

logging.basicConfig(encoding='utf-8', level=logging.INFO)


def load_tdes_ztf(min_points_fit = 5, object_list = None, alert_list = None):
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

	df = pd.read_csv(os.path.join(Config.INPUT_DIR, 'all_tde_in_ztf.csv'), dtype={'id': str})
	df['type'] = 'TDE'
	df.drop(columns = 'id', inplace=True)

	if object_list:
		df = df[df.objectId.isin(object_list)]

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
	df['candid'] = df['objectId'] + '-' + df['length'].astype(str)

	if alert_list:
		df = df[df.candid.isin(alert_list)]

	return df


def load_extragalatic_data_full_lightcurves(object_list = None, alert_list = None):

	df = pd.read_parquet(os.path.join(Config.INPUT_DIR,
							   Config.EXTRAGAL_FNAME))

	df.columns = df.columns.str.lstrip('i:')  # strip prefix
	if object_list:
		df = df[df.objectId.isin(object_list)]
	if alert_list:
		df = df[df.candid.isin(alert_list)]

	return df


def is_sorted(a):
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
		DESCRIPTION.
	title : str, optional
		To put as plot title (e.g. ZTF name + transient type)


	"""

	mjd, flux, fluxerr, _ = lc_values
	X = np.linspace(lc_values[0].min() - 10, lc_values[0].max() + 10, 500)

	fig = plt.figure(figsize=(12, 8))
	ax = plt.gca()

	ax.set_xlabel('Time (MDJ)', fontsize = 14)
# 	ax.set_ylabel('Normalised flux', fontsize = 14)
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
# 	if len(lc_values[0]) == 17:
# 		tuple_to_save = (lc_values, filt_list)
# 		with open('/home/lmiguel/Thesis/generate_plots/tdes/data_lc.pickle', 'wb') as handle:
# 			pickle.dump(tuple_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)




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

	# Taken from https://www.mdpi.com/2075-4434/7/2/62

	if np.var(fluxcal, ddof=1) - np.mean(fluxcalerr**2) <= 0:
		return np.nan

	return np.sqrt(np.var(fluxcal, ddof=1) - np.mean(fluxcalerr**2)) / np.mean(fluxcal)


def crop_old_history_and_get_Fvar(lc_values):

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

	# Check whether it is rising and not decaying
	rising_flag, decaying_flag = get_rising_flags_per_filter(*lc_values, list_filters = [1, 2],
														  sigma_rise = 2)
	final_flag = any(rising_flag.values()) and not any(decaying_flag.values())
	return final_flag


def get_post_fit_features(lc_values, values_fit, err_fit):

	# Relative distance of the last datapoint to the center of the sigmoid, wrt the rise time
	sigmoid_center_ref = (lc_values[0, -1] - values_fit[0]) / values_fit[2]
	# Get values on the uncertainty of the fitted values (i.e. value divided by error)
	snr_rise_time = values_fit[2] / err_fit[2]
	snr_amplitude = values_fit[1] / err_fit[1]

	return [sigmoid_center_ref, snr_rise_time, snr_amplitude]


def flag_based_on_post_fit_criteria(post_fit_feat, values_fit):

	sigmoid_center_ref, snr_rise_time, snr_amplitude = post_fit_feat
	flag = (sigmoid_center_ref > Config.sigdist_lim[0] and sigmoid_center_ref < Config.sigdist_lim[1]
		and snr_rise_time > Config.min_snr_features and snr_amplitude > Config.min_snr_features
		#and values_fit[1] < Config.max_ampl
# 		and values_fit[2] < Config.max_risetime
		# and values_fit[3] > Config.min_temp
		and values_fit[4] < Config.max_rchisq)
		# rise time, temperature and rchisq, respectively < 100 days, > 10**4K, and < 10.

	return flag


def extract_features_for_lc(lc_values_unnormalised, feature, min_nb_points_fit = 5,
							show_plots = False, title_plot = '', post_fit_cuts = True):
	# Create copy
	lc_values = lc_values_unnormalised.copy()
	# Delete duplicate times
	lc_values = np.delete(lc_values, np.where(np.diff(lc_values[0]) == 0)[0], axis = 1)
	# Crop out old history
	lc_values, Fvar = crop_old_history_and_get_Fvar(lc_values)

	if not (lc_values.shape[1] >= min_nb_points_fit and is_lc_on_the_rise(lc_values)) or Fvar!=0:
		return list(np.full((19), np.nan))

	# Convert filter list to g, r, i strings
	filt_list = np.vectorize(Config.filt_conv.get)(lc_values[3].astype(int))

# 	from early_tde_classification import extra_tools
# 	plt.figure()
# 	extra_tools.plot_lc_from_lc_values_array(lc_values)

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
			return list(np.full((18), np.nan))

		# Plot
		if show_plots:
			plot_lightcurve_and_fit(lc_values, filt_list, values_fit, err_fit, feature,
						   post_fit_feat, title = title_plot)


		return list(values_fit) + list(err_fit) + std_and_snr + post_fit_feat + [Fvar, lc_values.shape[1]]


	except RuntimeError:
		return list(np.full((18), np.nan))



def feature_extractor_full_LC_row_df(row_obj, feature, min_nb_points_fit = 5, show_plots = False,
									 keep_only_last_alert = True):

	name = row_obj.objectId
	trans_type = row_obj.type
	alertid = row_obj.candid

	full_lc_values = np.stack(row_obj[['jd', 'FLUXCAL', 'FLUXCALERR', 'fid']])

	# Delete elements with duplicate time (just in case)
	no_duplicate_mask = np.where(np.diff(full_lc_values[0]) == 0)[0]
	full_lc_values = np.delete(full_lc_values, no_duplicate_mask, axis=1)
	alertid = np.delete(alertid, no_duplicate_mask)  # Also delete corresponding alertid values

	# sort by MJD (apply to lc_values and also alertID)
	sorted_indices = full_lc_values[0, :].argsort()
	full_lc_values = full_lc_values[:, sorted_indices]
	alertid = alertid[sorted_indices]

# 	full_lc_values = full_lc_values[:, full_lc_values[0, :].argsort()]

	# Iterate over full lightcurve (from the end) until filter flags are passed.
	lc_values_cropped = full_lc_values

	if keep_only_last_alert:  # Default. Only last alert that passes the cuts

		for i in range(full_lc_values.shape[1] - min_nb_points_fit + 1):

			out_feat = extract_features_for_lc(lc_values_cropped, feature, min_nb_points_fit, show_plots,
								title_plot = 'objectId: %s. Transient type: %s. ' %(name, trans_type))

			if np.isnan(out_feat[1]):
				# Remove last alert for next iteration
				lc_values_cropped = lc_values_cropped[:, :-1]
			else:
				return [name, alertid[out_feat[-1] - 1], trans_type] + out_feat
		return [name, alertid[-1], trans_type] + list(np.full((18), np.nan))

	else:  # All alerts that pass the cuts are returned
		out_features_list = []
		for i in range(full_lc_values.shape[1] - min_nb_points_fit + 1):

			out_feat = extract_features_for_lc(lc_values_cropped, feature, min_nb_points_fit, show_plots,
								title_plot = 'objectId: %s. Transient type: %s. ' %(name, trans_type))
			if not np.isnan(out_feat[0]):
				out_features_list.append([name, alertid[out_feat[-1] - 1], trans_type] + out_feat)

			# Remove last alert for next iteration
			lc_values_cropped = lc_values_cropped[:, :-1]
		else:
			out_features_list.append([name, alertid[-1], trans_type] + list(np.full((18), np.nan)))

		return out_features_list


def feature_extractor_for_row_df(row_obj, feature, flux_conv_required = True,
								 min_nb_points_fit = 5, show_plots = False):
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

	lc_values = np.stack(row_obj[['cMJD', 'cFLUXCAL', 'cFLUXCALERR', 'cFLT']])

	out_feat = extract_features_for_lc(lc_values, feature, min_nb_points_fit, show_plots,
						title_plot = 'objectId: %s. Transient type: %s. ' %(name, trans_type))

	return [name, alertid, trans_type] + out_feat


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
		os.makedirs(os.path.join(Config.OUT_FEATURES_DIR, 'all_alerts_per_tde'), exist_ok = True)
		feature_matrix.to_csv(os.path.join(Config.OUT_FEATURES_DIR, 'all_alerts_per_tde',
									 'features_tdes_ztf.csv'), index = False)

	if keep_only_last_alert:
		# Keep only last alert that passed the cut per object
		feature_matrix = keep_only_feat_last_alert_per_object(feature_matrix, input_df)
		if save:
			os.makedirs(os.path.join(Config.OUT_FEATURES_DIR, 'one_alert_per_tde'), exist_ok = True)
			feature_matrix.to_csv(os.path.join(Config.OUT_FEATURES_DIR, 'one_alert_per_tde',
									 'features_tdes_ztf.csv'), index = False)

	# Save features into csv in general folder
	if save:
		feature_matrix.to_csv(os.path.join(Config.OUT_FEATURES_DIR, 'features_tdes_ztf.csv'),
							  index = False)
	return feature_matrix


def extract_features_non_tdes_extragal(save = True, show_plots = False, object_list = None,
											   alert_list = None, keep_only_last_alert = True):

	# Initialise
	feature = RainbowFit.from_angstrom(Config.band_wave_aa, with_baseline = False,
									temperature='constant', bolometric='sigmoid')
	columns_feat = ['objId', 'alertId', 'type',  # 'norm',
				  'ref_time', 'amplitude', 'rise_time', 'temperature', 'r_chisq',
					'err_ref_time', 'err_amplitude', 'err_rise_time', 'err_temperature',
					'std_flux_g', 'std_flux_r', 'std_snr_g', 'std_snr_r',
					'sigmoid_dist', 'snr_rise_time', 'snr_amplitude', 'Fvar', 'nb_points']

	feature_matrix = pd.DataFrame([], columns = columns_feat)

	# Load
	extragal_data = load_extragalatic_data_full_lightcurves(object_list = object_list,
														 alert_list = alert_list)
	if keep_only_last_alert:
		# Get features
		feature_matrix[columns_feat] = extragal_data.apply(
			lambda x: feature_extractor_full_LC_row_df(x, feature, show_plots = show_plots),
										 result_type = 'expand', axis = 1)
	else:
		feature_matrix = extract_features_all_alerts_per_object_full_LC_df(extragal_data,
																	 feature,
																	 columns_feat,
																	 show_plots = show_plots)
	feature_matrix.dropna(inplace = True)
	feature_matrix['data_origin'] = 'extragal'

	# Save features into csv in general folder
	if save:
		feature_matrix.to_csv(os.path.join(Config.OUT_FEATURES_DIR, 'features_extragal.csv'),
							  index = False)
	return feature_matrix


def extract_features_all_alerts_per_object_full_LC_df(extragal_data, feature,
													  columns_feat,
													  show_plots = False):

	result_dfs = []

	# Loop over each row in extragal_data
	for _, row in extragal_data.iterrows():
		# Call the feature extraction function
		results = feature_extractor_full_LC_row_df(row, feature, show_plots=show_plots,
												keep_only_last_alert = False)

		# Check if the result is a list of lists
		if isinstance(results, list) and all(isinstance(res, list) for res in results):
			# If so, convert each list into a DataFrame with columns_feat as columns
			temp_df = pd.DataFrame(results, columns=columns_feat)
		else:
			# Otherwise, assume it's a single list and convert it into a single-row DataFrame
			temp_df = pd.DataFrame([results], columns=columns_feat)

		# Append the resulting DataFrame to the list
		result_dfs.append(temp_df)
	# Concatenate all DataFrames into the final feature matrix
	feature_matrix = pd.concat(result_dfs, ignore_index=True)

	return feature_matrix




def extract_features_tdes(save = True, show_plots = False, keep_only_last_alert = True,
						  object_list = None, alert_list = None):
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
	columns_feat = ['objId', 'alertId', 'type',  # 'norm',
				  'ref_time', 'amplitude', 'rise_time', 'temperature', 'r_chisq',
					'err_ref_time', 'err_amplitude', 'err_rise_time', 'err_temperature',
					'std_flux_g', 'std_flux_r', 'std_snr_g', 'std_snr_r',
					'sigmoid_dist', 'snr_rise_time', 'snr_amplitude', 'Fvar', 'nb_points']

	feature_matrix = pd.DataFrame([], columns = columns_feat)

	# Load
	ztf_tde_data = load_tdes_ztf(object_list = object_list, alert_list = alert_list)

	# Get features
	feature_matrix[columns_feat] = ztf_tde_data.apply(
		lambda x: feature_extractor_for_row_df(x, feature, flux_conv_required=False,
										  show_plots = show_plots), result_type = 'expand', axis = 1)
	feature_matrix.dropna(inplace = True)
	feature_matrix['data_origin'] = 'tdes_ztf'

	feature_matrix = get_final_feature_dataframe_and_save(feature_matrix, ztf_tde_data, save,
													   keep_only_last_alert, 'tdes_ztf')
	return feature_matrix


def extract_features(data_origin, **kwargs):
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
	elif data_origin == 'extragal':
		return extract_features_non_tdes_extragal(**kwargs)
	elif data_origin == 'all':
		feat_tdes = extract_features_tdes(**kwargs)
		feat_extragal = extract_features_non_tdes_extragal(**kwargs)

		# Merge and save features
		all_features = pd.concat([feat_tdes, feat_extragal])
		if "save" in kwargs:
 			all_features.to_csv(os.path.join(
				Config.OUT_FEATURES_DIR, 'features_all.csv'), index = False)
		return all_features
	else:
		print('Wrong string given as data origin. Must be "extragal", "tdes_ztf" or "all"')


if __name__ == '__main__':

	start = dt.datetime.now()

	extract_features('extragal', save = True, show_plots = False, keep_only_last_alert=False)

	logging .info("Done in {} seconds.".format(dt.datetime.now() - start))
