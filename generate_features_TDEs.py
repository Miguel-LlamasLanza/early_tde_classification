import requests
import numpy as np
import glob
import pandas as pd
from io import BytesIO
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


if __name__ == '__main__':

	# Get data and prepare for fitting
# 	df = get_data_from_FINK(save = False)
	df = pd.read_csv('ZTF_TDE_Data/from_Fink.csv')
	converted_df = sn_tools.convert_full_dataset(df, obj_id_header='objectId')
	converted_df_early = crop_lc_to_rsing_part(converted_df)

	# Obtain features and save
	feature_matrix = sn_tools.featurize_full_dataset(converted_df_early, screen = True)
	feature_matrix.to_csv('Features_check/features_tdes.csv', index = None)
	merge_features_tdes_SN('Features_check/features_tdes.csv', 'Features_check/features.csv',
						   'Features_check/merged_features.csv')



