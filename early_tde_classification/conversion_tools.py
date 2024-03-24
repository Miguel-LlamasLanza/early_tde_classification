#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 17:29:36 2024

This file contains tools dedicated for converting data.

@author: Miguel Llamas Lanza
"""

import numpy as np
import pandas as pd


def mag2fluxcal_snana(magpsf: float, sigmapsf: float):
	""" Conversion from magnitude to Fluxcal from SNANA manual. Taken from SN classifier.
	Parameters
	----------
	magpsf: float
		PSF-fit magnitude from ZTF.
	sigmapsf: float
		Error on PSF-fit magnitude from ZTF.

	Returns
	----------
	fluxcal: float
		Flux cal as used by SNANA
	fluxcal_err: float
		Absolute error on fluxcal (the derivative has a minus sign)
	"""
	if magpsf is None:
		return None, None
	fluxcal = 10 ** (-0.4 * magpsf) * 10 ** (11)
	fluxcal_err = 9.21034 * 10 ** 10 * np.exp(-0.921034 * magpsf) * sigmapsf

	return fluxcal, fluxcal_err


def convert_full_dataset(pdf: pd.DataFrame, obj_id_header='candid'):
	"""Convert an entire data set from mag to fluxcal. Adapted from SN classifier.

	Parameters
	----------
	pdf: pd.DataFrame
		DataFrame with same format as the parquet files.
	obj_id_header: str (optional)
		Object identifier. Options are ['objectId', 'candid'].
		Default is 'candid'.

	Returns
	-------
	pd.DataFrame
		Columns are ['objectId', 'type', 'MJD', 'FLT',
		'FLUXCAL', 'FLUXCALERR'].
	"""
	# Ia types in TNS
	Ia_group = ['SN Ia', 'SN Ia-91T-like', 'SN Ia-91bg-like', 'SN Ia-CSM',
				'SN Ia-pec', 'SN Iax[02cx-like]']

	# hard code ZTF filters
	filters = ['g', 'r']

	lc_flux_sig = []

	for index in range(pdf.shape[0]):
		print('\r Objects converted: {}/{}'.format(index + 1, pdf.shape[0]), end='\r')

		name = pdf[obj_id_header].values[index]

		try:
			sntype_orig = pdf['TNS'].values[index]
			if sntype_orig == -99:
				sntype_orig = pdf['cdsxmatch'].values[index]

			if sntype_orig in Ia_group:
				transient_type = 'Ia'
			else:
				transient_type = str(sntype_orig).replace(" ", "")
		except KeyError:
			transient_type = 'TDE'

		for f in range(1,3):

			if isinstance(pdf.iloc[index]['fid'], str):
				ffs = np.array([int(item) for item in pdf.iloc[index]['fid'][1:-1].split()])
				filter_flag = ffs == f
				mjd = np.array([float(item) for item in pdf.iloc[index]['jd'][1:-1].split()])[filter_flag]
				mag = np.array([float(item) for item in pdf.iloc[index]['magpsf'][1:-1].split()])[filter_flag]
				magerr = np.array([float(item) for item in pdf.iloc[index]['sigmapsf'][1:-1].split()])[filter_flag]
			else:
				filter_flag = pdf['fid'].values[index] == f
				mjd = pdf['jd'].values[index][filter_flag]
				mag = pdf['magpsf'].values[index][filter_flag]
				magerr = pdf['sigmapsf'].values[index][filter_flag]

			fluxcal = []
			fluxcal_err = []
			for k in range(len(mag)):
				f1, f1err = mag2fluxcal_snana(mag[k], magerr[k])
				fluxcal.append(f1)
				fluxcal_err.append(f1err)

			for i in range(len(fluxcal)):
				lc_flux_sig.append([name, transient_type, mjd[i], filters[f - 1],
									fluxcal[i], fluxcal_err[i]])

	lc_flux_sig = pd.DataFrame(lc_flux_sig, columns=['id', 'type', 'MJD',
													 'FLT', 'FLUXCAL',
													 'FLUXCALERR'])

	return lc_flux_sig



def add_alert_history_to_df(df, prefix = 'c'):
	"""
	Add the history of alerts as a list in each cell of FLUXCAL, FLUXCALERR, MJD and FLT columns.
	This way, we convert from the df obtained from Fink to the one from the zenodo parquet files.

	Parameters
	----------
	df : pd.DataFrame
		DataFrame with one value (of flux, error, time, filter) per row.

	Returns
	-------
	df_sorted : pd.DataFrame
		DataFrame with list of (flux, error, time, filter) values in each row, i.e. alert history.

	"""


	# Sort DataFrame by objectId and MJD
	df_sorted = df.sort_values(by=['objectId', 'MJD'])

	# Define columns for which history will be tracked
	history_columns = ['FLUXCAL', 'FLUXCALERR', 'FLT', 'MJD']

	# Create empty dictionaries to store history for each column
	history_dict = {col: [] for col in history_columns}

	# Iterate over each group
	for _, group in df_sorted.groupby('objectId'):
		# Initialize lists of history for each column for this group
		group_history = {col: [] for col in history_columns}
		# Iterate over each row in the group
		for index, row in group.iterrows():
			# Get history for each column for this row
			row_history = {col: group.loc[group.index <= index, col].tolist() for col in history_columns}
			# Add to the history for each column for this row
			for col in history_columns:
				group_history[col].append(row_history[col])
		# Extend the history for each column for this group
		for col in history_columns:
			history_dict[col].extend(group_history[col])

	# Add the history for each column as new columns to the DataFrame
	for col in history_columns:
		df_sorted[prefix+col] = history_dict[col]


	return df_sorted
