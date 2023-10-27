import requests
import numpy as np
import glob
import pandas as pd
from io import BytesIO
import sys
import tools_from_sn_classifier as sn_tools


def get_data_from_FINK(save = True):

	# Get object names
	TDEs_hammerstein = pd.read_fwf('ZTF_TDE_Data/Table1_Hammerstein', skiprows = 34, header = None)
	ztf_names = TDEs_hammerstein[1].to_list()

	what_to_retrieve = ['i:jd', 'i:fid', 'i:magpsf', 'i:sigmapsf', 'i:candid', 'i:objectId', 'i:ra', 'i:dec']

	r = requests.post('https://fink-portal.org/api/v1/objects',
	  json={
		  'objectId': ','.join(ztf_names),
		  'columns': ','.join(what_to_retrieve)
	  }
	)
	df = pd.read_json(BytesIO(r.content))

# 	df = df.rename(columns = {'i:jd': 'jd', 'i:fid': 'fid', 'i:magpsf': 'magpsf',
# 						   'i:sigmapsf': 'sigmapsf','i:candid': 'candid',
# 						   'i:objectId': 'objectId'})
	df.columns = df.columns.str.strip('i:')  # strip prefix
	if save:
		df.to_csv('ZTF_TDE_Data/from_Fink.csv', index = None)
	return df


def load_forced_photometry_data(fink_df):


	# Load forced-photometry data
	forced_phot_data = glob.glob('ZTF_TDE_Data/forced_photometry/batchfp_*.txt')
	list_of_dfs = []
	for forced_phot_fname in forced_phot_data:
		obj_id = find_objectId_for_forced_phot_data(forced_phot_fname, fink_df)
		df_forced_phot = pd.read_csv(forced_phot_fname, comment = '#', sep = ' ')
		df_forced_phot.columns = df_forced_phot.columns.str.strip(',')  # strip prefix
		df_forced_phot['objectId'] = obj_id
		list_of_dfs.append(df_forced_phot)

	all_objects_df = pd.concat(list_of_dfs)

	return all_objects_df


def merge_features_tdes_SN(csv_tdes, csv_other, out_csv):

	feat_tdes = pd.read_csv(csv_tdes)
	feat_other = pd.read_csv(csv_other)

	merged_df = pd.concat([feat_tdes, feat_other])

	merged_df.to_csv(out_csv, index = False)


def crop_lc_to_rsing_part(converted_df: pd.DataFrame, minimum_nb_obs: int = 3, save_csv = True):
	"""
	Crop the light-curve to retain only the rising part. Drop every observation after the max flux.
	Keep observations of an object only if the object presents at least "minimum_nb_obs" observations.
	# TODO: Maybe trim based on histogram (get e.g. 90 pr cent)

	Parameters
	----------
	converted_df : pd.DataFrame
		Dataset with columns ['objectId', 'type', 'MJD', 'FLT','FLUXCAL', 'FLUXCALERR'].
	minimum_nb_obs : int, optional
		Minimum number of observations (otherwise drop alerts of object). The default is 3.
	save_csv : bool, optional
		Wehther to save the output dataframe onto a csv. The default is True.

	Returns
	-------
	converted_df_early :  pd.DataFrame
		Data prepared for the fitting, after dropping the decaying part of the LC.

	"""

	df_list = []
	for indx in range(np.unique(converted_df['id'].values).shape[0]):

		name = np.unique(converted_df['id'].values)[indx]
		obj_flag = converted_df['id'].values == name

		for filt in ['g', 'r']:
			object_df = converted_df[obj_flag][converted_df['FLT'] == filt].copy()
			if len(object_df) > minimum_nb_obs:
				tmax = object_df['MJD'][object_df['FLUXCAL'].idxmax()]
				object_df = object_df[object_df.MJD <= tmax]
				df_list.append(object_df)

	converted_df_early = pd.concat(df_list)
	if save_csv:
		converted_df_early.to_csv('input_for_feature_extractor.csv', index = False)
	return converted_df_early


def is_unique(s):
	""" Check if all values of a (pandas) series are equal. """
	a = s.to_numpy()
	return (a[0] == a).all()


def find_objectId_for_forced_phot_data(forced_phot_fname, df_fink, deg_tolerance = 0.001):
	"""
	Correlate the forced photometry data with the Fink data, to find the object ID corresponding
	to the forced-photometry data file.

	Parameters
	----------
	forced_phot_fname : str
		filename containing the forced_phot data of one object.
	df_fink : pd.DataFrame
		Data from Fink with all the objects at interest.
	deg_tolerance : float, optional
		Margin in degrees, for the search in RA and DEC. The default is 0.001.

	Returns
	-------
	obj_id : str
		Object Identifier.

	"""


	with open(forced_phot_fname) as f:
	    for i, line in enumerate(f):
	        if i == 3:
	            req_ra = float(line.split(' ')[-2])
	        elif i ==4:
	            req_dec = float(line.split(' ')[-2])
	        elif i > 4:
	            break

	df_obj = df_fink[(df_fink.ra > req_ra - deg_tolerance) & (df_fink.ra < req_ra + deg_tolerance) &
			 (df_fink.dec > req_dec - deg_tolerance) & (df_fink.dec < req_dec + deg_tolerance)]
	if len(df_obj) == 0:
		print('Error while correlating the forced-phometry data with the objectId: '+
				'No object was found in this position, for ' + forced_phot_fname)

		obj_id = None

	elif is_unique(df_obj.objectId):
		obj_id = df_obj.objectId.iloc[0]

	else:
		print('Error while correlating the forced-phometry data with the objectId: '+
				'more than one object are within the given position given.')
		obj_id = None
	return obj_id

if __name__ == '__main__':

	# Get data and prepare for fitting
# 	fink_df = get_data_from_FINK(save = False)
	fink_df = pd.read_csv('ZTF_TDE_Data/from_Fink.csv')
	df = load_forced_photometry_data(fink_df)
# 	converted_df = sn_tools.convert_full_dataset(df, obj_id_header='objectId')
# 	converted_df_early = crop_lc_to_rsing_part(converted_df)

# 	# Obtain features and save
# 	feature_matrix = sn_tools.featurize_full_dataset(converted_df_early, screen = True)
# 	feature_matrix.to_csv('Features_check/features_tdes.csv', index = None)
# 	merge_features_tdes_SN('Features_check/features_tdes.csv', 'Features_check/features.csv',
# 						   'Features_check/merged_features.csv')


