#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:09:32 2024

@author: lmiguel
"""
import numpy as np
import io
import os
import requests
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table

from early_tde_classification.config import Config



def plot_lc_from_lc_values_array(lc_values):

	colors = ['#15284F', '#F5622E']
	filt_list = np.vectorize(Config.filt_conv.get)(lc_values[3].astype(int))
	for idx, flt_str in enumerate(['g', 'r']):

		flt_mask = filt_list == flt_str

		plt.errorbar(lc_values[0][flt_mask], lc_values[1][flt_mask], yerr=lc_values[2][flt_mask], fmt='o', alpha=.7,
			   color=colors[idx], label = flt_str)
	plt.legend()
	plt.ylabel('Flux')
	plt.xlabel('Time (JD)')


def get_objectId_with_tns_labels(save = False):


	r = requests.post(
	    'https://fink-portal.org/api/v1/resolver',
	    json={
	        'resolver': 'tns',
	        'name': '',
	        'nmax': 1000000
	    }
	)
	tns = pd.read_json(io.BytesIO(r.content))
	tns = tns.sort_values('key:time')
	print(len(tns), 'TNS entries')

	# Only select ZTF ids
	tns_ztf = Table([{'tns':_['d:type'], 'objectId':_['d:internalname']} for __,_ in tns.iterrows() if _['d:internalname'].startswith('ZTF') ])
	print(len(tns_ztf), 'TNS entries with ZTF objectId')

	# Only keep TDEs (there are several sub-types in TNS)
	tns_ztf_tde = tns_ztf[['TDE' in _ for _ in tns_ztf['tns']]]
	print(len(tns_ztf_tde), 'TNS TDEs with ZTF')

	# Only keep non TDEs (there are several sub-types in TNS)
	tns_ztf_nontde = tns_ztf[['TDE' not in _ for _ in tns_ztf['tns']]]
	print(len(tns_ztf_nontde), 'TNS non-TDEs with ZTF')

	if save:
		tns_ztf.write(os.path.join(Config.INPUT_DIR, 'ztf_tns_crossmatch.csv'), format='csv',
							 overwrite=True)

	return tns_ztf, tns_ztf_tde

def correct_alertid_in_feat_data():


	import pandas as pd
	import numpy as np
	from early_tde_classification.extract_features import load_extragalatic_data_full_lightcurves as load_lcs

	# Load features data
	feat_data = pd.read_csv('features_with_tns_labels.csv')
	orig_data = load_lcs()

	# Create mapping of original alertid to corrected one
	mapping = {}
	for _,row_orig in orig_data.iterrows():

		original_alertid = np.array(row_orig['candid'])  # Unsorted alertid list
		original_jd = np.array(row_orig['jd'])  # Unsorted jd list

		# Sort JD and get sorting indices
		sorted_indices = np.argsort(original_jd)

		# Sort alertid using the same order as jd
		sorted_alertid = original_alertid[sorted_indices]

		mapping.update(dict(zip(original_alertid, sorted_alertid)))

	def correct_alertId(row_feat):

		return mapping[row_feat.alertId]
		

	# Apply correction
	feat_data['alertId'] = feat_data.apply(correct_alertId, axis=1)

	# Save corrected feature data
	feat_data.to_csv('features_with_tns_labels_correct_alertids.csv', index=False)




def get_labelled_dataset(selection = 'all', save = False):

	# Load dataset
	df = pd.read_parquet(os.path.join(Config.INPUT_DIR,
							   Config.EXTRAGAL_FNAME))
	# Crossmatch with TNS labels
	tns_ztf, tns_ztf_tdes = get_objectId_with_tns_labels()

	# Filter
	if selection == 'all':
		df = df[df.objectId.isin(tns_ztf['objectId'])]
	elif selection == 'tdes':
		df = df[df.objectId.isin(tns_ztf_tdes['objectId'])]
	elif selection == 'non-tdes':
		tns_ztf_non_tdes = tns_ztf[['TDE' not in _ for _ in tns_ztf['tns']]]
		df = df[df.objectId.isin(tns_ztf_non_tdes['objectId'])]

	if save:
		df.to_parquet(os.path.join(Config.INPUT_DIR,
								   Config.EXTRAGAL_FNAME.split('.')[0] + '_tns_{}.parquet'.format(selection)))

	return df
